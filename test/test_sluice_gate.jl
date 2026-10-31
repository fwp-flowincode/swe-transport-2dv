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

# Hydraulic jump behind the gate with Chezy friction (issue #57), C_c = GATE_CC, q = 0.15 m^2/s, a = 0.1 m: the theory
# puts the jump at 14.6, 12.9 and 11.1 m (gate at 8 m, outlet at 20 m). Measured (1D, dx = 0.2 m): the jump 0.07-0.14 m
# downstream of the theory (0.03-0.07 m with dx = 0.1 m), jet and subcritical depths within 1.3e-3, upstream depth
# 1.7e-3; 2DV (8 layers): jump 0.05-0.43 m (0.24 m at 0.18 m, not monotone in the outlet depth; docs section 26),
# upstream and subcritical depths 7e-3
const GATE_JUMP_CHEZY = 50.0                    # m^(1/2)/s
const GATE_JUMP_TAILWATERS = (0.16, 0.18, 0.20) # m, outlet depths
const GATE_JUMP_RUN_TIME = 400.0                # s, the jump settles slowly (its position drifts by cm at 200 s)
const GATE_JUMP_POSITION_CELLS = 1.0            # -, jump position tolerance in cells (1D)
const GATE_JUMP_2DV_POSITION_CELLS = 3.0        # -, 2DV: up to 2.2 cells over the outlet depths 0.155-0.21 m
const GATE_JUMP_DEPTH_RTOL = 5e-3               # -
const GATE_JUMP_2DV_DEPTH_RTOL = 1e-2           # -, subcritical depths; sheared velocity profile (coefficients > 1)
const GATE_JUMP_REFINED_RATIO = 0.6             # -, position error with dx/2 against dx: first order gives 0.5
# Non-hydrostatic jump (contraction=1): 8 m channel, gate at 2.4 m, C = 40, outlet 0.15 m, 160 x 12 cells. The theory
# starts at the vena contracta of the model (its smallest depth behind the gate). Measured: the jump (half its height)
# 0.26 m = 1.4 h_2 downstream of the theory (0.19 m with 320 x 24), the jet 1 % and the subcritical depths up to 1.4 %
# from the gradually varied flow (docs section 26)
const GATE_NH_JUMP_LENGTH = 8.0         # m
const GATE_NH_JUMP_CHEZY = 40.0         # m^(1/2)/s
const GATE_NH_JUMP_TAILWATER = 0.15     # m
const GATE_NH_JUMP_NX = 160             # -, dx = 0.05 m
const GATE_NH_JUMP_RUN_TIME = 120.0     # s, steady after about 100 s
const GATE_NH_JUMP_POSITION_H2 = 2.5    # -, jump position tolerance in conjugate depths h_2
const GATE_NH_JUMP_DEPTH_RTOL = 3e-2    # -, jet and subcritical depths (sheared, non-hydrostatic flow)

# Gate model: discharge inlet, gate at `x_gate`, free outflow (tailwater `nothing`) or outlet level (tailwater depth, m).
# Start: the free-flow energy head upstream of the gate, the tailwater (or a 1 cm film) downstream.
function gate_model(; q, a, contraction, tailwater=nothing, nx=GATE_NX, nz=1, nu=1e-4, length=GATE_LENGTH, x_gate=GATE_X,
    nonhydro=false, lower_edge=a, chezy=0.0)
    dx = length / nx
    xc = ((1:nx) .- 0.5) .* dx
    dt = GATE_COURANT * dx / (q / (contraction * a) + sqrt(9.81 * GATE_GRID_DEPTH))
    cfg = ModelConfig(nx=nx, nz=nz, length=length, depth=GATE_GRID_DEPTH, dt=dt, theta=1.0, implicit_vertical_terms=true,
        nu=nu, rayleigh_damping=0.0, apply_froude_limiter=false, momentum_advection_scheme="tvd2", wetting_drying=true,
        use_hydrostatic=!nonhydro, chezy_coefficient=chezy, write_plots=false)
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

# Jump position (m) in the depth profile: where the depth first rises through the mean of the theoretical jet depth and
# its conjugate depth downstream of the gate (linear between cell centres)
function jump_position(h, xc, xj, q, a, C)
    hj = chezy_supercritical_depth(xj, q, C, 0.0, GATE_X, GATE_CC * a)
    mid = 0.5 * (hj + conjugate_depth(hj, q))
    i = findfirst(j -> xc[j] > GATE_X && h[j] > mid, eachindex(xc))
    return xc[i - 1] + (mid - h[i - 1]) / (h[i] - h[i - 1]) * (xc[i] - xc[i - 1])
end

# Steady jump run: depth profile (m), cell centres (m), relative spread of the discharge
function run_gate_jump(tailwater; q=0.15, a=0.1, nx=GATE_NX, nz=1, nu=1e-4)
    model, xc = gate_model(; q=q, a=a, contraction=GATE_CC, tailwater=tailwater, nx=nx, nz=nz, nu=nu, chezy=GATE_JUMP_CHEZY)
    for _ in 1:round(Int, GATE_JUMP_RUN_TIME / model.config.dt)
        step_forward!(model)
    end
    spread = maximum(abs.(gate_face_discharges(model) .- q)) / q
    return water_depth(model), xc, spread
end

@testset "Sluice gate (issues #48, #57)" begin
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

    @testset "test_verify_sluice_gate_jump_reference_solutions" begin
        q, a, C = 0.15, 0.1, GATE_JUMP_CHEZY
        h0, hc = GATE_CC * a, critical_depth(q)
        # H3 curve (horizontal bed), closed form: x = C^2 / q^2 (h_c^3 (h - h0) - (h^4 - h0^4) / 4)
        for h in (0.07, 0.1, hc)
            @test chezy_supercritical_distance(h, q, C, 0.0, h0) ≈ C^2 / q^2 * (hc^3 * (h - h0) - (h^4 - h0^4) / 4) rtol = TOL_IDENTITY
        end
        # M3 curve (mild slope): the depth satisfies dh/dx = (i_b - i_f) / (1 - Fr^2) (central difference)
        slope, x, dxs = 1e-3, 2.0, 1e-3
        h = chezy_supercritical_depth(x, q, C, slope, 0.0, h0)
        dhdx = (chezy_supercritical_depth(x + dxs, q, C, slope, 0.0, h0) - chezy_supercritical_depth(x - dxs, q, C, slope, 0.0, h0)) / 2dxs
        @test dhdx ≈ (slope - q^2 / (C^2 * h^3)) / (1 - q^2 / (9.81 * h^3)) rtol = 1e-6
        @test chezy_supercritical_distance(h, q, C, slope, h0) ≈ x rtol = 1e-9
        # the jump: conjugate of the jet depth = subcritical depth; a higher outlet pushes it towards the gate
        xjs = map(GATE_JUMP_TAILWATERS) do t
            xj = sluice_gate_jump_position(q, a, C, t; contraction=GATE_CC, x_gate=GATE_X, x_end=GATE_LENGTH)
            hj = chezy_supercritical_depth(xj, q, C, 0.0, GATE_X, h0)
            @test conjugate_depth(hj, q) ≈ chezy_backwater_depth(xj, q, C, 0.0, GATE_LENGTH, t) rtol = 1e-9
            xj
        end
        @test issorted(collect(xjs); rev=true)
        @test all(GATE_X + 10 * GATE_LENGTH / GATE_NX .< collect(xjs) .< GATE_LENGTH - 10 * GATE_LENGTH / GATE_NX)
        # frictionless limit: on the horizontal bed both profiles scale with C^2, so the jump moves away from the gate
        # as C^2 (into an infinitely long reach without friction)
        t = GATE_JUMP_TAILWATERS[2]
        x1 = sluice_gate_jump_position(q, a, C, t; contraction=GATE_CC, x_end=GATE_LENGTH)
        x4 = sluice_gate_jump_position(q, a, 2C, t; contraction=GATE_CC, x_end=4GATE_LENGTH)
        @test x4 ≈ 4x1 rtol = 1e-9
        # submerged (outlet above the modular limit) and washed out (outlet too low) throw
        @test_throws DomainError sluice_gate_jump_position(q, a, C, 0.25; contraction=GATE_CC, x_gate=GATE_X, x_end=GATE_LENGTH)
        @test_throws DomainError sluice_gate_jump_position(q, a, C, 0.16; contraction=GATE_CC, x_gate=GATE_X, x_end=GATE_X + 2)
    end

    @testset "test_validate_sluice_gate_hydraulic_jump" begin
        # Chezy friction: the jet deepens from C_c a (H3 curve), the subcritical flow from the outlet depth (H2 curve);
        # the jump stands where the conjugate depth of the jet equals the subcritical depth (Chow 1959, ch. 9-10, 15)
        q, a, C = 0.15, 0.1, GATE_JUMP_CHEZY
        dx = GATE_LENGTH / GATE_NX
        E = sluice_gate_free_head(q, a; contraction=GATE_CC)
        h1 = bernoulli_depth(E, q)   # m, upstream depth at the gate
        for (nz, cells, sub_rtol) in ((1, GATE_JUMP_POSITION_CELLS, GATE_JUMP_DEPTH_RTOL),
            (8, GATE_JUMP_2DV_POSITION_CELLS, GATE_JUMP_2DV_DEPTH_RTOL))
            positions = map(GATE_JUMP_TAILWATERS) do t
                h, xc, spread = run_gate_jump(t; nz=nz)
                xj = sluice_gate_jump_position(q, a, C, t; contraction=GATE_CC, x_gate=GATE_X, x_end=GATE_LENGTH)
                xm = jump_position(h, xc, xj, q, a, C)
                @test abs(xm - xj) < cells * dx
                @test spread < GATE_STEADY_Q_RTOL
                # upstream (H2 curve from the gate depth), in the jet, and between the jump and the outlet
                iu = argmin(abs.(xc .- GATE_UPSTREAM_X))
                @test h[iu] ≈ chezy_backwater_depth(xc[iu], q, C, 0.0, GATE_X, h1) rtol = sub_rtol
                ij = argmin(abs.(xc .- 0.5 * (GATE_X + xj)))
                @test h[ij] ≈ chezy_supercritical_depth(xc[ij], q, C, 0.0, GATE_X, GATE_CC * a) rtol = GATE_JUMP_DEPTH_RTOL
                is = argmin(abs.(xc .- 0.5 * (xj + GATE_LENGTH)))
                @test h[is] ≈ chezy_backwater_depth(xc[is], q, C, 0.0, GATE_LENGTH, t) rtol = sub_rtol
                xm
            end
            @test issorted(collect(positions); rev=true)
        end
        # first order in dx: the position error halves with the grid
        t = GATE_JUMP_TAILWATERS[1]
        xj = sluice_gate_jump_position(q, a, C, t; contraction=GATE_CC, x_gate=GATE_X, x_end=GATE_LENGTH)
        err = map((GATE_NX, 2GATE_NX)) do nx
            h, xc, _ = run_gate_jump(t; nx=nx)
            abs(jump_position(h, xc, xj, q, a, C) - xj)
        end
        @test err[2] < GATE_JUMP_REFINED_RATIO * err[1]
    end

    @testset "test_validate_sluice_gate_nonhydro_jump" begin
        # non-hydrostatic, the jet contracts by itself (contraction=1); theory (H3, conjugate depth, H2) from the vena
        # contracta of the model, so the test checks friction and momentum, not the contraction coefficient
        q, a, C, t = 0.15, 0.1, GATE_NH_JUMP_CHEZY, GATE_NH_JUMP_TAILWATER
        model, xc = gate_model(; q=q, a=a, contraction=1.0, tailwater=t, nx=GATE_NH_JUMP_NX, nz=GATE_NH_NZ, length=GATE_NH_JUMP_LENGTH,
            x_gate=GATE_NH_X, nonhydro=true, chezy=C)
        for _ in 1:round(Int, GATE_NH_JUMP_RUN_TIME / model.config.dt)
            step_forward!(model)
        end
        @test maximum(abs.(gate_face_discharges(model) .- q)) / q < GATE_STEADY_Q_RTOL
        h = water_depth(model)
        behind = findall(x -> GATE_NH_X < x < GATE_NH_X + 1.0, xc)
        iv = behind[argmin(h[behind])]   # vena contracta
        xj = sluice_gate_jump_position(q, h[iv], C, t; contraction=1.0, x_gate=xc[iv], x_end=GATE_NH_JUMP_LENGTH)
        hj = chezy_supercritical_depth(xj, q, C, 0.0, xc[iv], h[iv])
        h2 = conjugate_depth(hj, q)
        mid = 0.5 * (hj + h2)
        k = findfirst(j -> j > iv && h[j] > mid, eachindex(xc))
        xm = xc[k - 1] + (mid - h[k - 1]) / (h[k] - h[k - 1]) * (xc[k] - xc[k - 1])
        @test abs(xm - xj) < GATE_NH_JUMP_POSITION_H2 * h2
        ij = argmin(abs.(xc .- 0.5 * (xc[iv] + xj)))
        @test h[ij] ≈ chezy_supercritical_depth(xc[ij], q, C, 0.0, xc[iv], h[iv]) rtol = GATE_NH_JUMP_DEPTH_RTOL
        for xs in (xj + 1.5, 0.5 * (xj + GATE_NH_JUMP_LENGTH))
            is = argmin(abs.(xc .- xs))
            @test h[is] ≈ chezy_backwater_depth(xc[is], q, C, 0.0, GATE_NH_JUMP_LENGTH, t) rtol = GATE_NH_JUMP_DEPTH_RTOL
        end
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
