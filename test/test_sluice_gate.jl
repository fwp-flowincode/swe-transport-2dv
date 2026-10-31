include("test_common.jl")

# Sluice gate (issue #48): flow under a gate in the wetting-drying step against the classical gate relations of
# src/reference_solutions.jl (Bernoulli up to the vena contracta, momentum balance behind it; Henderson 1966, ch. 6).
# See docs/TESTS_TECHNICAL_DOCUMENTATION.md, section 26.

const GATE_LENGTH = 20.0        # m
const GATE_X = 8.0              # m, gate position (a u-face)
const GATE_UPSTREAM_X = 3.2     # m, section of the upstream energy head
const GATE_GRID_DEPTH = 0.6     # m
const GATE_NX = 100             # -, dx = 0.2 m
const GATE_RUN_TIME = 150.0     # s, until steady (about 10 travel times of the upstream wave over the reach)
const GATE_COURANT = 0.3        # -, (u_jet + sqrt(g H)) dt / dx
const GATE_CC = 0.61            # -, classical contraction coefficient of a vertical gate (Henderson 1966, ch. 6)
# Measured: free flow 4e-4 (1D and 2DV), submerged 1D 1e-4 relative error of the upstream energy head
const GATE_HEAD_RTOL = 2e-3             # -
const GATE_STEADY_Q_RTOL = 5e-3         # -, spread of the discharge over the faces at the end of the run
# Non-hydrostatic contraction (contraction=1, the jet contracts by itself), 6 m channel, 12 layers: measured C_c = 0.82
# from the upstream head (0.76 and 0.64 with 2 and 4 times the resolution in x and z, docs section 26; the hydrostatic
# step on the same grid gives 1.00)
const GATE_NH_LENGTH = 6.0      # m
const GATE_NH_X = 2.4           # m
const GATE_NH_NX = 120          # -, dx = 0.05 m
const GATE_NH_NZ = 12           # -, dz = 0.05 m: the opening a = 0.1 m spans two layers
const GATE_NH_RUN_TIME = 40.0   # s
const GATE_NH_CC_MAX = 0.85     # -, contraction clearly resolved
const GATE_HYDRO_CC_MIN = 0.98  # -, the hydrostatic step does not contract

const GATE_FREE_CASES = (
    (q=0.15, a=0.1, contraction=GATE_CC),
    (q=0.08, a=0.05, contraction=1.0),
)

# Gate model: discharge inlet, gate at `x_gate`, free outflow (tailwater `nothing`) or outlet level (tailwater depth, m).
# Start: the free-flow energy head upstream of the gate, the tailwater (or a 1 cm film) downstream.
function gate_model(; q, a, contraction, tailwater=nothing, nx=GATE_NX, nz=1, nu=1e-4, length=GATE_LENGTH, x_gate=GATE_X,
    nonhydro=false, lower_edge=a)
    dx = length / nx
    xc = ((1:nx) .- 0.5) .* dx
    dt = GATE_COURANT * dx / (q / (contraction * a) + sqrt(9.81 * GATE_GRID_DEPTH))
    cfg = ModelConfig(nx=nx, nz=nz, length=length, depth=GATE_GRID_DEPTH, dt=dt, theta=1.0, implicit_vertical_terms=true,
        nu=nu, rayleigh_damping=0.0, apply_froude_limiter=false, momentum_advection_scheme="tvd2", wetting_drying=true,
        use_hydrostatic=!nonhydro, write_plots=false)
    model = Model(cfg)
    fill!(model.state.rho, 1000.0)
    refresh_density_metadata!(model)
    E = sluice_gate_free_head(q, a; contraction=contraction)
    initialize_water_level!(model, [x < x_gate ? E : (tailwater === nothing ? 0.01 : tailwater) for x in xc])
    lower_edge === nothing || set_sluice_gate!(model, x_gate; lower_edge=lower_edge, contraction=contraction)
    if tailwater === nothing
        set_x_inflow_outflow!(model, ones(nz); left_discharge=q, right_free=true)
    else
        set_x_inflow_outflow!(model, ones(nz); left_discharge=q, right_level=tailwater - GATE_GRID_DEPTH)
    end
    return model, xc
end

gate_face_discharges(model) = [sum(model.wetdry.hf[:, f] .* model.state.u[:, f]) for f in 1:(model.config.nx + 1)]

# Steady run: upstream energy head (m), relative spread of the discharge, largest volume-budget error (m^2).
function run_gate(; run_time=GATE_RUN_TIME, upstream_x=GATE_UPSTREAM_X, kwargs...)
    model, xc = gate_model(; kwargs...)
    dx, dt = model.dx, model.config.dt
    volume() = sum(water_depth(model)) * dx
    budget_error = 0.0
    for _ in 1:round(Int, run_time / dt)
        V0 = volume()
        step_forward!(model)
        net = dt * (sum(model.wetdry.q[:, 1]) - sum(model.wetdry.q[:, end]))
        budget_error = max(budget_error, abs(volume() - V0 - net))
    end
    q = kwargs[:q]
    h1 = water_depth(model)[argmin(abs.(xc .- upstream_x))]
    spread = maximum(abs.(gate_face_discharges(model) .- q)) / q
    return h1 + q^2 / (2 * 9.81 * h1^2), spread, budget_error
end

# Contraction coefficient of the jet from the upstream energy head (m): E = C_c a + q^2 / (2 g (C_c a)^2), supercritical root
gate_contraction_from_head(E, q, a) = bernoulli_depth(E, q; supercritical=true) / a

@testset "Sluice gate (issue #48)" begin
    @testset "test_verify_sluice_gate_reference_solutions" begin
        q, a = 0.15, 0.1
        E = sluice_gate_free_head(q, a; contraction=GATE_CC)
        @test q ≈ GATE_CC * a * sqrt(2 * 9.81 * (E - GATE_CC * a)) rtol = TOL_IDENTITY   # Henderson's discharge law
        yt = sluice_gate_modular_limit(q, a; contraction=GATE_CC)
        @test yt ≈ conjugate_depth(GATE_CC * a, q) rtol = TOL_IDENTITY
        # continuous at the modular limit, increasing above it, the free head below it
        @test sluice_gate_submerged_head(q, a, yt * (1 + 1e-12); contraction=GATE_CC) ≈ E rtol = 1e-9
        @test sluice_gate_submerged_head(q, a, 1.2yt; contraction=GATE_CC) > E
        @test sluice_gate_submerged_head(q, a, 0.8yt; contraction=GATE_CC) == E
        @test gate_contraction_from_head(E, q, a) ≈ GATE_CC rtol = 1e-9
    end

    @testset "test_verify_sluice_gate_inactive_is_bitwise_identity" begin
        # a gate above the water surface changes nothing; clear_sluice_gates! removes a gate
        runs = map((nothing, 10.0, :cleared)) do edge
            model, _ = gate_model(; q=0.15, a=0.1, contraction=1.0, nz=4, lower_edge=edge === :cleared ? 0.1 : edge)
            edge === :cleared && clear_sluice_gates!(model)
            for _ in 1:200
                step_forward!(model)
            end
            @test !any(model.wetdry.gate_active)
            (copy(model.state.u), copy(model.state.eta), copy(model.state.w))
        end
        @test runs[2] == runs[1]
        @test runs[3] == runs[1]
    end

    @testset "test_verify_sluice_gate_closed_holds_the_water" begin
        # gate down to the bed: no flow through the face; the inflow fills the reach upstream, the tailwater stays at rest
        for nz in (1, 4)
            tailwater = 0.2
            model, xc = gate_model(; q=0.05, a=0.1, contraction=1.0, nz=nz, tailwater=tailwater, lower_edge=0.0)
            fg = round(Int, GATE_X / model.dx) + 1
            up = xc .< GATE_X
            V0 = sum(water_depth(model)[up]) * model.dx
            nsteps = round(Int, 20.0 / model.config.dt)
            closed = true
            for _ in 1:nsteps
                step_forward!(model)
                closed &= all(model.wetdry.hf[:, fg] .== 0.0)
            end
            @test closed
            @test sum(water_depth(model)[up]) * model.dx - V0 ≈ 0.05 * nsteps * model.config.dt rtol = TOL_MODEL_IDENTITY
            @test maximum(abs.(water_depth(model)[.!up] .- tailwater)) < TOL_MODEL_IDENTITY
        end
    end

    @testset "test_validate_sluice_gate_free_flow" begin
        # Bernoulli up to the vena contracta: E_1 = C_c a + q^2 / (2 g (C_c a)^2); 1D and 2DV
        for c in GATE_FREE_CASES, nz in (1, 8)
            E, spread, budget = run_gate(; q=c.q, a=c.a, contraction=c.contraction, nz=nz)
            Eref = sluice_gate_free_head(c.q, c.a; contraction=c.contraction)
            @test abs(E - Eref) < GATE_HEAD_RTOL * Eref
            @test spread < GATE_STEADY_Q_RTOL
            @test budget < TOL_MODEL_IDENTITY * GATE_GRID_DEPTH * GATE_LENGTH
        end
    end

    @testset "test_validate_sluice_gate_submerged_flow" begin
        # tailwater above the modular limit: Bernoulli to the vena contracta, momentum balance from there to the
        # tailwater (Henderson 1966, ch. 6); 1D (2DV: docs section 26)
        q, a = 0.15, 0.1
        @test sluice_gate_modular_limit(q, a; contraction=GATE_CC) < 0.28
        for tailwater in (0.28, 0.34)
            E, spread, _ = run_gate(; q=q, a=a, contraction=GATE_CC, tailwater=tailwater)
            Eref = sluice_gate_submerged_head(q, a, tailwater; contraction=GATE_CC)
            @test Eref > sluice_gate_free_head(q, a; contraction=GATE_CC)
            @test abs(E - Eref) < GATE_HEAD_RTOL * Eref
            @test spread < GATE_STEADY_Q_RTOL
        end
    end

    @testset "test_validate_sluice_gate_nonhydro_contraction" begin
        # the non-hydrostatic step contracts the jet by itself (contraction=1); the hydrostatic step on the same grid does not
        q, a = 0.15, 0.1
        Cc = map((true, false)) do nonhydro
            E, spread, budget = run_gate(; q=q, a=a, contraction=1.0, nonhydro=nonhydro, nx=GATE_NH_NX, nz=GATE_NH_NZ,
                length=GATE_NH_LENGTH, x_gate=GATE_NH_X, run_time=GATE_NH_RUN_TIME, upstream_x=0.4 * GATE_NH_X)
            @test spread < GATE_STEADY_Q_RTOL
            @test budget < TOL_MODEL_IDENTITY * GATE_GRID_DEPTH * GATE_NH_LENGTH
            gate_contraction_from_head(E, q, a)
        end
        @test GATE_CC < Cc[1] < GATE_NH_CC_MAX
        @test Cc[2] > GATE_HYDRO_CC_MIN
    end

    @testset "test_verify_sluice_gate_config_checks" begin
        model, _ = gate_model(; q=0.15, a=0.1, contraction=1.0, lower_edge=nothing)
        @test_throws ArgumentError set_sluice_gate!(model, GATE_X + 0.05; lower_edge=0.1)       # not a u-face
        @test_throws ArgumentError set_sluice_gate!(model, 0.0; lower_edge=0.1)                 # boundary face
        @test_throws ArgumentError set_sluice_gate!(model, GATE_LENGTH; lower_edge=0.1)
        @test_throws ArgumentError set_sluice_gate!(model, GATE_X; lower_edge=-0.01)            # below the bed
        @test_throws ArgumentError set_sluice_gate!(model, GATE_X; lower_edge=0.1, contraction=0.0)
        @test_throws ArgumentError set_sluice_gate!(model, GATE_X; lower_edge=0.1, contraction=1.2)
        nh, _ = gate_model(; q=0.15, a=0.1, contraction=1.0, nz=4, nonhydro=true, lower_edge=nothing)
        @test_throws ArgumentError set_sluice_gate!(nh, GATE_X; lower_edge=0.1, contraction=GATE_CC)
        set_sluice_gate!(nh, GATE_X; lower_edge=0.1)
        @test nh.wetdry.gate_edge[round(Int, GATE_X / nh.dx) + 1] == 0.1
        fixed = Model(ModelConfig(nx=20, nz=4, write_plots=false))
        @test_throws ArgumentError set_sluice_gate!(fixed, fixed.dx * 5; lower_edge=0.1)
    end
end
