# Barotropic free-surface solve, kinematic surface condition and continuity closure (docs sections 8 and 9).
# Run with the full suite (runtests.jl) or on its own: julia --project=. test/test_free_surface.jl

include(joinpath(@__DIR__, "test_common.jl"))

@testset "Free surface" begin
    @testset "test_barotropic_solver_zero_flow_preserves_eta" begin
        cfg = ModelConfig(nx=10, nz=4, dt=0.01)
        solver = BarotropicSolver(cfg, cfg.length / cfg.nx, cfg.depth / cfg.nz)
        eta0 = fill(0.0123, cfg.nx)
        u_hat = zeros(cfg.nz, cfg.nx + 1)
        eta = copy(eta0)
        solve!(solver, eta, u_hat, eta0)

        @test all(isapprox.(eta, eta0, atol=1e-12))
    end

    @testset "test_free_surface_kinematic_bc_consistency" begin
        cfg = ModelConfig(nx=60, nz=20, dt=5e-4, nt=1)
        model = Model(cfg)
        for _ in 1:150
            eta_old = copy(model.state.eta)
            step_forward!(model)
            eta_t = (model.state.eta .- eta_old) ./ cfg.dt
            @test all(isapprox.(model.state.w[end, :], eta_t, atol=1e-10))
        end
    end

    @testset "test_hydrostatic_continuity_and_surface_closure_residuals" begin
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
end
