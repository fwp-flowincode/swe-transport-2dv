# Barotropic free-surface solve, kinematic surface condition and continuity closure (docs sections 8 and 9).
# Run with the full suite (runtests.jl) or on its own: julia --project=. test/test_free_surface.jl

include(joinpath(@__DIR__, "test_common.jl"))

@testset "Free surface" begin
    @testset "test_verify_barotropic_solver_zero_flow_preserves_eta" begin
        cfg = ModelConfig(nx=10, nz=4, dt=0.01)
        solver = BarotropicSolver(cfg, cfg.length / cfg.nx, cfg.depth / cfg.nz)
        eta0 = fill(0.0123, cfg.nx)
        u_hat = zeros(cfg.nz, cfg.nx + 1)
        eta = copy(eta0)
        solve!(solver, eta, u_hat, eta0)

        @test all(isapprox.(eta, eta0, atol=1e-12))
    end

    @testset "test_verify_free_surface_kinematic_bc_consistency" begin
        cfg = ModelConfig(nx=60, nz=20, dt=5e-4, nt=1)
        model = Model(cfg)
        for _ in 1:150
            eta_old = copy(model.state.eta)
            step_forward!(model)
            eta_t = (model.state.eta .- eta_old) ./ cfg.dt
            @test all(isapprox.(model.state.w[end, :], eta_t, atol=1e-10))
        end
    end

    @testset "test_verify_hydrostatic_continuity_and_surface_closure_residuals" begin
        # WHY: variable_top_layer=false pinned explicitly. This test verifies the LEGACY fixed-grid closure
        # identity (div=0 on every row including the top one, w_top=eta_t with a zero flux at a FIXED top face).
        # With the (now-default) variable top layer the top row instead closes eta_t = w_nz - div(h*u) (a different
        # identity, covered by test_verify_variable_top_layer_closes_kinematic_and_volume_budgets), so max_top_div
        # is not expected to vanish there.
        cfg = ModelConfig(nx=60, nz=20, dt=5e-4, nt=1, use_hydrostatic=true, variable_top_layer=false, momentum_advection_scheme="upwind1", rayleigh_damping=0.0, apply_froude_limiter=false)
        model = Model(cfg)

        max_div = 0.0
        max_top_div = 0.0
        max_kinematic = 0.0

        for _ in 1:200
            step_forward!(model)
            div = div_x_u(model.state.u, model.dx) .+ div_z_w(model.state.w, model.dz)
            max_div = max(max_div, maximum(abs, div))
            max_top_div = max(max_top_div, maximum(abs, @view div[end, :]))
            max_kinematic = max(max_kinematic, model.last_kinematic_mismatch)
        end

        @test max_div < 1e-10
        @test max_top_div < 1e-10
        @test max_kinematic < 1e-10
    end

    @testset "test_verify_wind_forcing_closes_volume_and_kinematic_budgets" begin
        # Closed basin under wind stress with an eddy viscosity, in the three free-surface treatments. Two former
        # defects (issue #4): the u predictor's wall faces entered the fixed-grid barotropic solve (water volume
        # drifted, about 1e-8 m^2 after 200 steps here), and the wind increment was added after the barotropic solve
        # and the projection (kinematic mismatch about 2e-5 m/s per step in the wall columns).
        NSTEPS = 200                 # -
        WIND_STRESS_X = 0.2          # Pa
        NU_EDDY = 1e-3               # m^2/s, gives non-zero diffusion at the wall faces of the predictor
        VOLUME_TOL = 1e-12           # m^2, |sum(eta) dx| after NSTEPS; round-off level is ~1e-17
        KINEMATIC_TOL = 1e-10        # m/s, model.last_kinematic_mismatch (same bound as the closure test above)
        for (use_hydrostatic, variable_top_layer) in ((true, true), (true, false), (false, false))
            cfg = ModelConfig(length=10.0, depth=4.0, nx=40, nz=20, dt=0.01, nt=NSTEPS, use_hydrostatic=use_hydrostatic,
                variable_top_layer=variable_top_layer, nu=NU_EDDY, wind_stress_x=WIND_STRESS_X, rayleigh_damping=0.0,
                apply_froude_limiter=false, write_plots=false, write_animation=false,
                write_pressure_buoyancy_diagnostics=false)
            model = Model(cfg)
            initialize_internal_seiche!(model; rho_top=999.0, rho_bottom=1001.0, interface_height_fraction=0.5,
                interface_tilt_amplitude_fraction=0.0)
            max_kinematic = 0.0
            for _ in 1:NSTEPS
                step_forward!(model)
                max_kinematic = max(max_kinematic, model.last_kinematic_mismatch)
            end
            @test abs(sum(model.state.eta) * model.dx) < VOLUME_TOL
            @test max_kinematic < KINEMATIC_TOL
            @test maximum(abs, model.state.eta) > 0.0   # the wind does set up the surface
        end
    end
end
