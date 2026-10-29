include("test_common.jl")

# Implicit vertical terms (issue #34, ModelConfig(implicit_vertical_terms=true)): vertical viscosity, upwind vertical
# momentum advection with w^n and linearised bed friction in one tridiagonal matrix A per u-face column,
#   A u = dZ .* uhat,   A r = dZ   (column response r; the free surface sees sum(dZ .* r) instead of the depth),
# see docs/plan_issue_34_theta_scheme.md, section 2.

const IMPLICIT_STIFF_COURANT = 100.0   # -, W dt / dZ of the stiff kernel test (explicit upwind is stable only up to 1)
const IMPLICIT_STEADY_MATCH = 1e-6     # -, relative difference of steady profiles computed with two time steps
const IMPLICIT_SPINUP_TIME = 2e4       # s, about 20 friction time scales H / (c_b u) of the steady channel

# One column with m layers, random positive thicknesses dZ (m), interface velocities W (m/s), viscosities nu (m^2/s).
function random_column(rng, m)
    dZ = 0.1 .+ rand(rng, m)
    W = 0.2 .* randn(rng, m + 1)
    nu = 0.01 .* rand(rng, m + 1)
    uhat = randn(rng, m)
    return dZ, W, nu, uhat
end

function solve_column(uhat, dZ, W, nu, gamma, dt; advective=false, h_adv=dZ, top_advection_rate=0.0)
    m = length(dZ)
    u, r = zeros(m), zeros(m)
    work = zeros(m, 3)
    SweTransport2DV.implicit_vertical_column!(u, r, uhat, dZ, W, nu, gamma, dt, m, view(work, :, 1), view(work, :, 2),
        view(work, :, 3); advective=advective, h_adv=h_adv, top_advection_rate=top_advection_rate)
    return u, r
end

@testset "Implicit vertical terms" begin
    @testset "test_verify_implicit_vertical_column_momentum_and_response" begin
        # Flux form: the vertical terms only move momentum within the column (no flux through bed and surface), so
        # sum(dZ u) = sum(dZ uhat) - dt gamma u_1 and sum(dZ r) = H - dt gamma r_1; r = A^{-1} dZ >= 0 (M-matrix).
        rng = MersenneTwister(34)
        for m in (1, 2, 7), gamma in (0.0, 0.03), dt in (0.1, 10.0)
            dZ, W, nu, uhat = random_column(rng, m)
            u, r = solve_column(uhat, dZ, W, nu, gamma, dt)
            @test isapprox(sum(dZ .* u), sum(dZ .* uhat) - dt * gamma * u[1]; atol=TOL_IDENTITY)
            @test isapprox(sum(dZ .* r), sum(dZ) - dt * gamma * r[1]; atol=TOL_IDENTITY)
            @test all(>=(0.0), r)
            gamma == 0.0 && @test isapprox(sum(dZ .* r), sum(dZ); atol=TOL_IDENTITY)
        end
    end

    @testset "test_verify_implicit_vertical_advective_form_keeps_uniform_flow" begin
        # Advective form (wetting and drying): rows sum to dZ, so a uniform uhat stays uniform and r = 1 without
        # friction; with friction max(u) <= max(uhat) (discrete maximum principle).
        rng = MersenneTwister(35)
        for m in (1, 3, 8)
            dZ, W, nu, _ = random_column(rng, m)
            h_adv = dZ .* (0.5 .+ rand(rng, m))
            u, r = solve_column(fill(0.7, m), dZ, W, nu, 0.0, 5.0; advective=true, h_adv=h_adv)
            @test maximum(abs.(u .- 0.7)) < TOL_IDENTITY
            @test maximum(abs.(r .- 1.0)) < TOL_IDENTITY
            uhat = randn(rng, m)
            u, _ = solve_column(uhat, dZ, W, nu, 0.05, 5.0; advective=true, h_adv=h_adv)
            @test maximum(u) <= max(maximum(uhat), 0.0) + TOL_IDENTITY
            @test minimum(u) >= min(minimum(uhat), 0.0) - TOL_IDENTITY
        end
    end

    @testset "test_verify_implicit_vertical_stiff_advection_is_contractive" begin
        # W dt / dZ = 100: explicit upwind would amplify; the implicit flux form is an L1 contraction,
        # sum(dZ |u|) <= sum(dZ |uhat|) (A^{-1} >= 0 and column sums dZ).
        rng = MersenneTwister(36)
        m = 10
        dZ = fill(0.1, m)
        for _ in 1:20
            W = IMPLICIT_STIFF_COURANT * 0.1 .* randn(rng, m + 1)
            uhat = randn(rng, m)
            u, _ = solve_column(uhat, dZ, W, zeros(m + 1), 0.0, 1.0)
            @test sum(dZ .* abs.(u)) <= sum(dZ .* abs.(uhat)) + TOL_IDENTITY
        end
    end

    @testset "test_verify_implicit_bed_friction_single_layer" begin
        # One layer, friction only: dZ (u - uhat) = -dt gamma u, so u = uhat / (1 + dt gamma / dZ) exactly, for any dt
        # (no reversal of the flow, unlike explicit friction for dt gamma / dZ > 1).
        for dt_gamma_over_dz in (0.1, 1.0, 50.0)
            u, r = solve_column([2.0], [0.5], zeros(2), zeros(2), 0.02, dt_gamma_over_dz * 0.5 / 0.02)
            @test u[1] ≈ 2.0 / (1 + dt_gamma_over_dz) rtol = TOL_IDENTITY
            @test r[1] ≈ 1.0 / (1 + dt_gamma_over_dz) rtol = TOL_IDENTITY
            @test u[1] > 0.0
        end
    end

    @testset "test_verify_implicit_top_row_vertical_advection" begin
        # The variable top-layer correction is an implicit top-cell source.
        rate, dt = 2.0, 0.1
        u, r = solve_column([2.0], [0.5], zeros(2), zeros(2), 0.0, dt; top_advection_rate=rate)
        @test u[1] ≈ 2.0 / (1 + dt * rate) rtol = TOL_IDENTITY
        @test r[1] ≈ 1.0 / (1 + dt * rate) rtol = TOL_IDENTITY
    end

    # Periodic channel of depth H with uniform flow, no viscosity: only the bed layer feels the friction.
    function channel_model(; dt, nz=4, H=1.0, z0=1e-3, nu=0.0, body_force=0.0, theta=1.0)
        cfg = ModelConfig(nx=8, nz=nz, length=8.0, depth=H, dt=dt, theta=theta, periodic_x=true,
            pressure_solver_method="sparse", variable_top_layer=false, implicit_vertical_terms=true,
            bed_roughness_length=z0, nu=nu, body_force_x=body_force, rayleigh_damping=0.0, apply_froude_limiter=false,
            momentum_advection_scheme="upwind1")
        model = Model(cfg)
        fill!(model.state.rho, 1000.0)
        refresh_density_metadata!(model)
        return model
    end

    @testset "test_verify_implicit_bed_friction_uniform_channel" begin
        # u^{n+1}_1 = u^n_1 / (1 + dt c_b |u^n_1| / dz) exactly (linearised implicit friction), the upper layers keep their
        # velocity, and the surface stays flat; dt c_b |u| / dz = 20, where the explicit friction stopped the run.
        H, nz, z0, u0 = 1.0, 4, 1e-3, 1.0
        dz = H / nz
        cb = bed_friction_coefficient(z0, 0.5 * dz)
        dt = 20.0 * dz / (cb * u0)
        model = channel_model(; dt=dt, nz=nz, H=H, z0=z0)
        fill!(model.state.u, u0)
        expected = u0
        for _ in 1:10
            step_forward!(model)
            expected = expected / (1 + dt * cb * abs(expected) / dz)
            @test maximum(abs.(model.state.u[1, :] .- expected)) < TOL_IDENTITY
            @test maximum(abs.(model.state.u[2:end, :] .- u0)) < TOL_IDENTITY
            @test maximum(abs, model.state.eta) < TOL_IDENTITY
        end
    end

    @testset "test_verify_implicit_steady_channel_independent_of_dt" begin
        # Body force against bed friction and viscosity: the steady state of the implicit terms solves B u = dZ f
        # (A = diag(dZ) + dt B), which does not contain dt; two step sizes give the same profile.
        H, z0, nu, force = 1.0, 1e-3, 1e-2, 1e-4
        profiles = map((0.5, 4.0)) do dt
            model = channel_model(; dt=dt, nz=8, H=H, z0=z0, nu=nu, body_force=force)
            for _ in 1:round(Int, IMPLICIT_SPINUP_TIME / dt)
                step_forward!(model)
            end
            model.state.u[:, 1]
        end
        @test maximum(abs.(profiles[1] .- profiles[2])) < IMPLICIT_STEADY_MATCH * maximum(abs, profiles[1])
    end

    # Wetting and drying with implicit vertical terms and bed friction: beach rising out of the water (0 ... 2.5 m over
    # 20 m, grid depth 3 m).
    function wd_friction_model(; theta=0.55, dt=0.1, use_hydrostatic=true, z0=1e-3, nx=40, nz=8)
        cfg = ModelConfig(nx=nx, nz=nz, length=20.0, depth=3.0, dt=dt, theta=theta, wetting_drying=true,
            bed_level=[2.5 * (i - 0.5) / nx for i in 1:nx], implicit_vertical_terms=true, bed_roughness_length=z0,
            nu=1e-3, use_hydrostatic=use_hydrostatic, rayleigh_damping=0.0, apply_froude_limiter=false,
            momentum_advection_scheme="upwind1")
        return Model(cfg)
    end

    @testset "test_verify_implicit_wd_lake_at_rest_with_friction" begin
        for hydro in (true, false)
            model = wd_friction_model(; use_hydrostatic=hydro)
            initialize_water_level!(model, 1.5; rho=1000.0)
            eta0 = copy(model.state.eta)
            for _ in 1:300
                step_forward!(model)
            end
            @test maximum(abs, model.state.u) < TOL_IDENTITY
            @test maximum(abs.(model.state.eta .- eta0)) < TOL_IDENTITY
        end
    end

    @testset "test_verify_implicit_wd_dam_break_positivity_and_conservation" begin
        # Dam break onto the dry beach at a large step (gravity-wave Courant number about 6) with bed friction: depth >= 0
        # in every step, water volume and constituent mass conserved, density within its bounds.
        for theta in (0.5, 0.55), hydro in (true, false)
            model = wd_friction_model(; theta=theta, dt=0.4, use_hydrostatic=hydro)
            xc = [(i - 0.5) * 0.5 for i in 1:40]
            rho = [i <= 10 ? 1001.0 : 1000.0 for _ in 1:8, i in 1:40]
            initialize_water_level!(model, [x < 8.0 ? 1.8 : 0.0 for x in xc]; rho=rho)
            volume0 = sum(water_depth(model))
            mass0 = SweTransport2DV._constituent_mass(model)
            for _ in 1:200
                step_forward!(model)
                @test minimum(water_depth(model)) >= 0.0
            end
            @test abs(sum(water_depth(model)) - volume0) / volume0 < TOL_MODEL_IDENTITY
            @test abs(SweTransport2DV._constituent_mass(model) - mass0) / mass0 < TOL_MODEL_IDENTITY
            @test 1000.0 - TOL_MODEL_IDENTITY <= minimum(model.state.rho) && maximum(model.state.rho) <= 1001.0 + TOL_MODEL_IDENTITY
        end
    end
end
