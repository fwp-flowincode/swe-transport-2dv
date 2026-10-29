include("test_common.jl")

# Flow over obstacles (issue #47): the SWASHES bump (Delestre et al. 2013, section 3.1.1), frictionless, against the
# Bernoulli / hydraulic-jump solutions of src/reference_solutions.jl; 1D (wetting-drying step, nz = 1) and 2DV.
# See docs/TESTS_TECHNICAL_DOCUMENTATION.md, section 23.

const BUMP_LENGTH = 25.0     # m
const BUMP_CREST = 10.0      # m
const BUMP_RUN_TIME = 200.0  # s, until steady (about 10 travel times of the slowest wave over the channel)
const BUMP_COURANT = 0.4     # -, (|U| + sqrt(g h)) dt / dx of the reference solution
const BUMP_NX = 125          # -, dx = 0.2 m
# Measured errors (nx = 125): subcritical 9.0e-3 m, transcritical 2.2e-2 m, shock 1.0e-2 m away from the jump (1D) and
# 1.8e-2 m (2DV), jump position 5e-2 m (1D); first-order convergence (errors halve with nx = 250, 500).
const BUMP_SUB_TOL = 1.5e-2          # m
const BUMP_TRANS_TOL = 3e-2          # m
const BUMP_SHOCK_TOL_1D = 2e-2       # m, at least 0.5 m from the jump
const BUMP_SHOCK_TOL_2DV = 3e-2      # m, the same in 2DV (the jump sits 0.15 m further upstream, see section 23)
const BUMP_JUMP_POSITION_TOL = 0.1   # m, half a cell at nx = 125
const BUMP_ORDER_MIN = 0.8           # -, observed order in space from nx = 125 and 250
const BUMP_STEADY_Q_RTOL = 1e-3      # -, spread of the discharge over the faces at the end of the run
const BUMP_JUMP_MARGIN = 0.5         # m, distance from the jump of the points compared in the shock case
# -, depth at the energy minimum (critical flow): E(h) is flat there, E(h) - E(h_c) ~ (h - h_c)^2, so the round-off of E
# (~eps) determines h only to ~sqrt(eps) = 1.5e-8
const BERNOULLI_CRITICAL_RTOL = 1e-7

const BUMP_CASES = (
    subcritical=(q=4.42, outlet_depth=2.0, regime=:subcritical, theta=0.75, grid_depth=2.5),
    transcritical=(q=1.53, outlet_depth=nothing, regime=:transcritical, theta=1.0, grid_depth=1.2),
    shock=(q=0.18, outlet_depth=0.33, regime=:shock, theta=1.0, grid_depth=0.5),
)

bump_reference(c, x) = bump_flow_depth(x, c.q; bed=swashes_bump_bed, x_crest=BUMP_CREST, regime=c.regime,
    outlet_depth=c.outlet_depth)

# Steady bump flow: discharge inlet, outlet level (subcritical outlet) or free outflow. Start: a lake at the outlet level,
# or (transcritical) the reference level upstream of the crest and a 1 cm film downstream, so that the crest takes
# control (without friction and outlet level, a lake carrying q is a steady state as well).
function bump_model(c; nx=BUMP_NX, nz=1, nu=1e-4, left_rho=nothing)
    dx = BUMP_LENGTH / nx
    xc = ((1:nx) .- 0.5) .* dx
    bed = swashes_bump_bed.(xc)
    href = [bump_reference(c, x) for x in xc]
    dt = BUMP_COURANT * dx / (maximum(c.q ./ href) + sqrt(9.81 * maximum(href)))
    cfg = ModelConfig(nx=nx, nz=nz, length=BUMP_LENGTH, depth=c.grid_depth, dt=dt, theta=c.theta,
        implicit_vertical_terms=true, nu=nu, rayleigh_damping=0.0, apply_froude_limiter=false,
        momentum_advection_scheme="tvd2", wetting_drying=true, bed_level=bed, write_plots=false)
    model = Model(cfg)
    fill!(model.state.rho, 1000.0)
    refresh_density_metadata!(model)
    if c.outlet_depth === nothing
        initialize_water_level!(model, [x <= BUMP_CREST ? href[i] + bed[i] : bed[i] + 0.01 for (i, x) in enumerate(xc)])
        set_x_inflow_outflow!(model, ones(nz); left_discharge=c.q, left_rho=left_rho, right_free=true)
    else
        initialize_water_level!(model, max.(c.outlet_depth, bed .+ 1e-3))
        set_x_inflow_outflow!(model, ones(nz); left_discharge=c.q, left_rho=left_rho, right_level=c.outlet_depth - c.grid_depth)
    end
    return model, xc, href
end

function run_bump(c; kwargs...)
    model, xc, href = bump_model(c; kwargs...)
    for _ in 1:round(Int, BUMP_RUN_TIME / model.config.dt)
        step_forward!(model)
    end
    return model, xc, water_depth(model), href
end

# Broad-crested (trapezoidal) weir in a 40 m channel, see test_validate_broad_crested_weir_free_and_submerged.
const WEIR_HEIGHT = 0.5          # m
const WEIR_Q = 1.0               # m^2/s
const WEIR_LENGTH = 40.0         # m
const WEIR_GRID_DEPTH = 1.6      # m
const WEIR_NX = 100              # -, dx = 0.4 m
const WEIR_RUN_TIME = 300.0      # s
const WEIR_Q_RTOL = 2e-2         # -, discharge from the weir law at the model head (measured 1.25e-2 at nx = 100 and
                                 #    4.5e-3 at nx = 200: first order)
const WEIR_MODULAR_TOL = 1e-3    # m, upstream head for two free-flow tailwaters (measured < 1e-4)
const WEIR_SUBMERGED_TOL = 3e-3  # m, submerged upstream energy against the tailwater energy (measured 1.3e-3)
weir_bed(x) = x < 15 ? 0.0 : x < 18 ? WEIR_HEIGHT * (x - 15) / 3 : x <= 22 ? WEIR_HEIGHT : x < 25 ? WEIR_HEIGHT * (25 - x) / 3 : 0.0

# Steady weir flow with the tailwater depth `ht` (m); returns the energy head 5 m downstream of the inlet and that of the
# tailwater (m above the channel bed).
function run_weir(ht; nz=1)
    g = 9.81
    dx = WEIR_LENGTH / WEIR_NX
    xc = ((1:WEIR_NX) .- 0.5) .* dx
    bed = weir_bed.(xc)
    dt = BUMP_COURANT * dx / (WEIR_Q / critical_depth(WEIR_Q) + sqrt(g * WEIR_GRID_DEPTH))
    cfg = ModelConfig(nx=WEIR_NX, nz=nz, length=WEIR_LENGTH, depth=WEIR_GRID_DEPTH, dt=dt, theta=1.0,
        implicit_vertical_terms=true, nu=nz == 1 ? 1e-4 : 1e-3, rayleigh_damping=0.0, apply_froude_limiter=false,
        momentum_advection_scheme="tvd2", wetting_drying=true, bed_level=bed, write_plots=false)
    model = Model(cfg)
    fill!(model.state.rho, 1000.0)
    refresh_density_metadata!(model)
    initialize_water_level!(model, max.(ht, bed .+ 1e-3))
    set_x_inflow_outflow!(model, ones(nz); left_discharge=WEIR_Q, right_level=ht - WEIR_GRID_DEPTH)
    for _ in 1:round(Int, WEIR_RUN_TIME / dt)
        step_forward!(model)
    end
    h_up = water_depth(model)[argmin(abs.(xc .- 5.0))]
    return h_up + WEIR_Q^2 / (2g * h_up^2), ht + WEIR_Q^2 / (2g * ht^2)
end

face_discharges(model) =[sum(model.wetdry.hf[:, f] .* model.state.u[:, f]) for f in 1:(model.config.nx + 1)]

@testset "Flow over obstacles (issue #47)" begin
    @testset "test_verify_bump_reference_solutions" begin
        q = 1.53
        hc = critical_depth(q)
        @test q^2 / (9.81 * hc^3) ≈ 1 rtol = TOL_IDENTITY                     # Fr = 1
        h_sub = bernoulli_depth(1.0, q)
        h_sup = bernoulli_depth(1.0, q; supercritical=true)
        @test h_sub > hc > h_sup
        for h in (h_sub, h_sup)
            @test q^2 / (2 * 9.81 * h^2) + h ≈ 1.0 rtol = TOL_IDENTITY        # same specific energy
        end
        @test conjugate_depth(conjugate_depth(h_sup, q), q) ≈ h_sup rtol = TOL_IDENTITY
        @test_throws DomainError bernoulli_depth(1.4 * hc, q)                 # below 3/2 h_c
        @test bump_reference(BUMP_CASES.transcritical, BUMP_CREST) ≈ hc rtol = BERNOULLI_CRITICAL_RTOL
        # SWASHES: jump at x = 11.6655 m for q = 0.18 m^2/s, outlet depth 0.33 m
        xj = hydraulic_jump_position(0.18; bed=swashes_bump_bed, x_crest=BUMP_CREST, outlet_depth=0.33, x_max=BUMP_LENGTH)
        @test xj ≈ 11.6656 atol = 1e-3
        h1 = bump_reference(BUMP_CASES.shock, xj - 1e-9)
        h2 = bump_reference(BUMP_CASES.shock, xj + 1e-9)
        @test conjugate_depth(h1, 0.18) ≈ h2 rtol = 1e-6
    end

    @testset "test_validate_swashes_bump_subcritical" begin
        c = BUMP_CASES.subcritical
        errors = Float64[]
        for (nx, nz) in ((BUMP_NX, 1), (2 * BUMP_NX, 1), (BUMP_NX, 8))
            model, xc, h, href = run_bump(c; nx=nx, nz=nz)
            err = maximum(abs.(h .- href))
            nz == 1 && push!(errors, err)
            nx == BUMP_NX && @test err < BUMP_SUB_TOL
            q = face_discharges(model)
            @test maximum(abs.(q .- c.q)) < BUMP_STEADY_Q_RTOL * c.q
        end
        @test log2(errors[1] / errors[2]) > BUMP_ORDER_MIN
    end

    @testset "test_validate_swashes_bump_transcritical" begin
        # critical depth at the crest, supercritical downstream of it
        c = BUMP_CASES.transcritical
        for nz in (1, 8)
            model, xc, h, href = run_bump(c; nz=nz)
            @test maximum(abs.(h .- href)) < BUMP_TRANS_TOL
            i = argmin(abs.(xc .- 15.0))
            @test c.q / h[i] / sqrt(9.81 * h[i]) > 1   # supercritical after the bump
            q = face_discharges(model)
            @test maximum(abs.(q .- c.q)) < BUMP_STEADY_Q_RTOL * c.q
        end
    end

    @testset "test_validate_swashes_bump_shock" begin
        # hydraulic jump on the downstream slope: position and conjugate depths. 2DV with nu = 1e-3 m^2/s: with less
        # viscosity the hydrostatic jump does not become steady (its roller is resolved by one or two layers).
        c = BUMP_CASES.shock
        xj = hydraulic_jump_position(c.q; bed=swashes_bump_bed, x_crest=BUMP_CREST, outlet_depth=c.outlet_depth,
            x_max=BUMP_LENGTH)
        for (nz, nu, tol) in ((1, 1e-4, BUMP_SHOCK_TOL_1D), (8, 1e-3, BUMP_SHOCK_TOL_2DV))
            model, xc, h, href = run_bump(c; nz=nz, nu=nu)
            away = abs.(xc .- xj) .> BUMP_JUMP_MARGIN
            @test maximum(abs.(h[away] .- href[away])) < tol
            q = face_discharges(model)
            @test maximum(abs.(q .- c.q)) < BUMP_STEADY_Q_RTOL * c.q
            if nz == 1
                # jump position: where the depth crosses the mean of the conjugate depths
                hmid = 0.5 * (bump_reference(c, xj - 1e-9) + bump_reference(c, xj + 1e-9))
                i = findfirst(j -> xc[j] > BUMP_CREST && h[j] > hmid, eachindex(xc))
                x_model = xc[i - 1] + (hmid - h[i - 1]) / (h[i] - h[i - 1]) * (xc[i] - xc[i - 1])
                @test abs(x_model - xj) < BUMP_JUMP_POSITION_TOL
            end
        end
    end

    @testset "test_verify_tracer_through_bump_monotone_and_conserved" begin
        # A tracer (density 1000.001 kg/m^3 at the inlet, 1000 in the channel; buoyancy negligible) is carried over the
        # bump: every step its content changes exactly by the boundary fluxes (inflow q rho_in, outflow upwind), the
        # values stay within [1000, 1000.001] (tvd2 transport), and the front has passed the crest.
        rho_in = 1000.001
        for (name, c) in pairs(BUMP_CASES), nz in (1, 8)
            model, xc, href = bump_model(c; nz=nz, left_rho=rho_in)
            model.config.transport_advection_scheme = "tvd2"
            wd = model.wetdry
            dx, dt = model.dx, model.config.dt
            content() = sum(wd.h .* model.state.rho) * dx   # kg/m (per unit width)
            budget_error = 0.0
            for _ in 1:round(Int, BUMP_RUN_TIME / dt / 4)
                rho_out = copy(model.state.rho[:, end])
                C0 = content()
                step_forward!(model)
                influx = dt * sum(max.(wd.q[:, 1], 0.0)) * rho_in
                outflux = dt * sum(wd.q[k, end] * rho_out[k] for k in 1:nz if wd.q[k, end] > 0.0; init=0.0)
                budget_error = max(budget_error, abs(content() - C0 - influx + outflux))
            end
            @test budget_error < TOL_MODEL_IDENTITY * 1000.0 * c.grid_depth * BUMP_LENGTH
            @test minimum(model.state.rho) >= 1000.0
            @test maximum(model.state.rho) <= rho_in
            @test maximum(model.state.rho[:, argmin(abs.(xc .- 15.0))]) > 1000.0 + 0.5 * (rho_in - 1000.0)
        end
    end

    @testset "test_validate_broad_crested_weir_free_and_submerged" begin
        # Trapezoidal weir (height p = 0.5 m, crest 18-22 m, 3 m ramps), q = 1 m^2/s, frictionless. Free flow: critical
        # depth on the crest, upstream energy head H = 3/2 h_c above the crest, q = (2/3)^(3/2) sqrt(g) H^(3/2), the same
        # for every tailwater as long as the jump stays on the downstream ramp (tailwater 0.9 and 1.0 m: modular flow).
        # Submerged (tailwater 1.3 m, its energy above the free-flow head): subcritical everywhere, upstream energy =
        # tailwater energy (Bernoulli). Measured (nx = 100): discharge from the head within 1.25 % (0.45 % at nx = 200),
        # the same upstream energy for both tailwaters within 1e-4 m, submerged energy within 1.3e-3 m.
        for nz in (1, 8)
            heads = Float64[]
            for ht in (0.9, 1.0, 1.3)
                E_up, E_t = run_weir(ht; nz=nz)
                if ht < 1.2
                    push!(heads, E_up - WEIR_HEIGHT)
                    q_law = (2 / 3)^1.5 * sqrt(9.81) * heads[end]^1.5
                    @test abs(q_law - WEIR_Q) < WEIR_Q_RTOL * WEIR_Q
                else
                    @test abs(E_up - E_t) < WEIR_SUBMERGED_TOL
                end
            end
            @test abs(heads[1] - heads[2]) < WEIR_MODULAR_TOL
        end
    end

    @testset "test_verify_inflow_density_config_checks" begin
        model = Model(ModelConfig(nx=8, nz=4, implicit_vertical_terms=true))
        @test_throws ArgumentError set_x_inflow_outflow!(model, ones(4); left_discharge=1.0, left_rho=1001.0)
        model, _, _ = bump_model(BUMP_CASES.subcritical)
        @test_throws ArgumentError set_x_inflow_outflow!(model, ones(1); left_discharge=4.42, left_rho=NaN)
        set_x_inflow_outflow!(model, ones(1); left_discharge=4.42, left_rho=999.0, right_level=-0.5)
        @test model.rho_min_init == 999.0   # the inflow density widens the transport bounds
    end
end
