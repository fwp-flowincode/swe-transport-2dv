using Test
using Random
using LinearAlgebra
using SparseArrays
using Statistics
using SweTransport2DV

@testset "2DV SWE Transport Tests" begin
    @testset "test_state_shapes" begin
        st = from_shape(6, 8)
        @test size(st.u) == (6, 9)
        @test size(st.w) == (7, 8)
        @test size(st.p) == (6, 8)
        @test size(st.rho) == (6, 8)
        @test size(st.eta) == (8,)
    end

    @testset "test_grad_and_div_shapes_and_consistency" begin
        nz, nx = 4, 5
        dx = 0.2
        dz = 0.3
        p = repeat(collect(0.0:(nx - 1.0))', nz, 1)

        gx = grad_x_c(p, dx)
        gz = grad_z_c(p, dz)

        @test size(gx) == (nz, nx + 1)
        @test size(gz) == (nz + 1, nx)
        @test all(isapprox.(gx[:, 2:end-1], 1.0 / dx, atol=1e-12))
        @test all(isapprox.(gz[2:end-1, :], 0.0, atol=1e-12))

        u = zeros(nz, nx + 1)
        w = zeros(nz + 1, nx)
        u[:, 3] .= 1.0
        w[3, :] .= -0.5

        div = div_x_u(u, dx) .+ div_z_w(w, dz)
        @test size(div) == (nz, nx)
    end

    @testset "test_density_advection_zero_for_uniform_density" begin
        nz, nx = 8, 10
        rho = fill(1000.0, nz, nx)
        dx, dz = 0.1, 0.1

        zc = reshape(collect(0:nz) ./ nz, nz + 1, 1)
        xc = reshape(collect(0:nx) ./ nx, 1, nx + 1)
        psi = 0.1 .* sin.(pi .* zc) .* sin.(pi .* xc)

        u = (psi[2:end, :] .- psi[1:end-1, :]) ./ dz
        w = -(psi[:, 2:end] .- psi[:, 1:end-1]) ./ dx

        adv = advection_term_rho(u, w, rho, dx, dz)
        @test all(isapprox.(adv, 0.0, atol=1e-12))
    end

    @testset "test_density_advection_conservative_closed_box" begin
        nz, nx = 12, 14
        dx, dz, dt = 0.1, 0.1, 0.02

        Random.seed!(42)
        rho = 998.0 .+ 4.0 .* rand(nz, nx)

        u = randn(nz, nx + 1) .* 0.05
        w = randn(nz + 1, nx) .* 0.05
        u[:, 1] .= 0.0
        u[:, end] .= 0.0
        w[1, :] .= 0.0
        w[end, :] .= 0.0

        m0 = constituent_mass(rho, dx, dz)
        rho1 = advect_density_upwind(rho, u, w, dt, dx, dz)
        m1 = constituent_mass(rho1, dx, dz)

        @test isapprox(m0, m1, atol=1e-10, rtol=1e-10)
    end

    @testset "test_hydrostatic_pressure_monotone_with_depth" begin
        nz, nx = 6, 7
        rho = fill(1000.0, nz, nx)
        p = solve_hydrostatic_pressure(zeros(nz, nx), rho, 9.81, 0.5)

        @test all(isapprox.(p[end, :], 0.0, atol=1e-12))
        @test all(p[1:end-1, :] .>= p[2:end, :])
    end

    @testset "test_barotropic_solver_zero_flow_preserves_eta" begin
        cfg = ModelConfig(nx=10, nz=4, dt=0.01)
        solver = BarotropicSolver(cfg, cfg.length / cfg.nx, cfg.depth / cfg.nz)
        eta0 = fill(0.0123, cfg.nx)
        u_hat = zeros(cfg.nz, cfg.nx + 1)
        eta = copy(eta0)
        solve!(solver, eta, u_hat, eta0)

        @test all(isapprox.(eta, eta0, atol=1e-12))
    end

    @testset "test_model_step_conservation_and_bounds_short_run" begin
        cfg = ModelConfig(nx=30, nz=15, dt=2e-4, nt=1)
        model = Model(cfg)

        d0 = compute_diagnostics(model)
        for _ in 1:30
            step_forward!(model)
        end
        d1 = compute_diagnostics(model)

        @test abs(d1["boundary_flux"]) < 1e-12
        @test abs(d1["water_volume"] - d0["water_volume"]) < 1e-8
        @test abs(d1["constituent_mass"] - d0["constituent_mass"]) < 1e-8
        @test d1["rho_min"] >= model.rho_min_init - 1e-12
        @test d1["rho_max"] <= model.rho_max_init + 1e-12
    end

    @testset "test_froude_diagnostic_behaviour" begin
        rho = [1002.0 1000.0; 1002.0 1000.0]
        u = [0.05 -0.05 0.05; 0.05 -0.05 0.05]
        fr = froude_number(u, rho, 1000.0, 9.81, 1.0)
        @test fr > 0.0
        @test isfinite(fr)
    end

    @testset "test_diagnostics_boundary_flux_closed_streamfunction_field" begin
        nz, nx = 10, 12
        dx, dz = 0.1, 0.1
        zc = reshape(collect(0:nz) ./ nz, nz + 1, 1)
        xc = reshape(collect(0:nx) ./ nx, 1, nx + 1)
        psi = 0.2 .* sin.(pi .* zc) .* sin.(pi .* xc)
        u = (psi[2:end, :] .- psi[1:end-1, :]) ./ dz
        w = -(psi[:, 2:end] .- psi[:, 1:end-1]) ./ dx

        flux = boundary_normal_flux(u, w, dx, dz)
        @test abs(flux) < 1e-12
    end

    @testset "test_poisson_matrix_and_nonhydrostatic_zero_rhs" begin
        nz, nx = 5, 7
        dx = dz = 0.2
        A = build_poisson_matrix((nz, nx), dx, dz)
        @test size(A) == (nz * nx, nz * nx)

        @test isapprox(A[1, 1], 1.0)
        @test nnz(A[1, :]) == 1

        top_edge_idx = 2
        expected_top_edge_diag = -(2.0 / dx^2 + 1.0 / dz^2)
        @test isapprox(A[top_edge_idx, top_edge_idx], expected_top_edge_diag)

        top_right_idx = nx
        expected_corner_diag = -(1.0 / dx^2 + 1.0 / dz^2)
        @test isapprox(A[top_right_idx, top_right_idx], expected_corner_diag)

        pc_solver = PressureCorrectionSolver((nz, nx), dx, dz; method="spectral")

        u = zeros(nz, nx + 1)
        w = zeros(nz + 1, nx)
        p_corr = zeros(nz, nx)
        sol = solve_nonhydrostatic_pressure(u, w, p_corr, 1000.0, 0.1, dx, dz, A)
        @test all(isapprox.(sol, 0.0, atol=1e-12))
        @test abs(mean(sol)) < 1e-14

        sol_fast = solve!(pc_solver, u, w, 1000.0, 0.1, dx, dz)
        @test all(isapprox.(sol_fast, 0.0, atol=1e-10))
        @test abs(mean(sol_fast)) < 1e-12
    end

    @testset "test_verify_poisson_matrix_top_dirichlet_matches_manufactured_solution" begin
        # Analytical solution: p'' = C on z in (0,H), p'(0)=0 (Neumann bottom), p(H)=0 (Dirichlet top at the domain
        # edge, matching the ghost-cell closure in build_poisson_matrix): p(z) = C/2*(z^2 - H^2).
        C = 3.0
        H = 2.0
        errors = Float64[]
        for nz in (10, 40)
            dx, dz = 1.0, H / nz
            A = build_poisson_matrix((nz, 1), dx, dz; top_dirichlet=true)
            @test !isapprox(det(Matrix(A)), 0.0)   # Dirichlet removes the pure-Neumann null space

            p = A \ fill(C, nz)
            z = [(k - 0.5) * dz for k in 1:nz]
            p_exact = 0.5 .* C .* (z .^ 2 .- H^2)
            push!(errors, maximum(abs, p .- p_exact))
        end
        @test errors[2] < errors[1] / 8   # expect ~4x reduction (2nd order) for a 4x grid refinement; 8x margin

        # Same check through the full PressureCorrectionSolver/solve! API (RHS built from w = C*z, so dw/dz = C).
        nz = 40
        dx, dz = 1.0, H / nz
        solver = PressureCorrectionSolver((nz, 1), dx, dz; method="sparse", top_dirichlet=true)
        u = zeros(nz, 2)
        w = reshape([(k - 1) * dz * C for k in 1:(nz + 1)], nz + 1, 1)
        p = solve!(solver, u, w, 1.0, 1.0, dx, dz)
        z = [(k - 0.5) * dz for k in 1:nz]
        p_exact = 0.5 .* C .* (z .^ 2 .- H^2)
        @test maximum(abs, p[:, 1] .- p_exact) < 1e-3

        # method="spectral" cannot represent a Dirichlet top condition.
        @test_throws ArgumentError PressureCorrectionSolver((5, 5), 1.0, 1.0; method="spectral", top_dirichlet=true)

        # Legacy default (top_dirichlet=false) is unaffected.
        legacy = build_poisson_matrix((5, 7), 0.2, 0.2)
        @test isapprox(legacy[1, 1], 1.0)
        @test nnz(legacy[1, :]) == 1
    end

    @testset "test_pressure_solver_spectral_matches_sparse_reference" begin
        nz, nx = 6, 8
        dx, dz = 0.25, 0.3
        dt = 0.05
        rho0 = 1000.0

        Random.seed!(12)
        u = randn(nz, nx + 1) .* 0.03
        w = randn(nz + 1, nx) .* 0.03
        u[:, 1] .= 0.0
        u[:, end] .= 0.0
        w[1, :] .= 0.0
        w[end, :] .= 0.0

        sol_spec = solve!(PressureCorrectionSolver((nz, nx), dx, dz; method="spectral"), u, w, rho0, dt, dx, dz)
        sol_sparse = solve!(PressureCorrectionSolver((nz, nx), dx, dz; method="sparse"), u, w, rho0, dt, dx, dz)

        gx_spec = grad_x_c(sol_spec, dx)
        gz_spec = grad_z_c(sol_spec, dz)
        gx_sparse = grad_x_c(sol_sparse, dx)
        gz_sparse = grad_z_c(sol_sparse, dz)

        @test all(isapprox.(gx_spec, gx_sparse, atol=1e-6, rtol=1e-6))
        @test all(isapprox.(gz_spec, gz_sparse, atol=1e-6, rtol=1e-6))
    end

    @testset "test_hydrostatic_pressure_vectorized_matches_reference_loop" begin
        Random.seed!(7)
        nz, nx = 9, 11
        rho = 998.0 .+ 4.0 .* rand(nz, nx)
        g = 9.81
        dz = 0.2

        p_vec = solve_hydrostatic_pressure(zeros(nz, nx), rho, g, dz)

        p_ref = zeros(nz, nx)
        p_ref[end, :] .= 0.0
        for k in (nz - 1):-1:1
            p_ref[k, :] .= p_ref[k + 1, :] .+ 0.5 .* (rho[k + 1, :] .+ rho[k, :]) .* g .* dz
        end

        @test all(isapprox.(p_vec, p_ref, atol=1e-12, rtol=1e-12))
    end

    @testset "test_buoyancy_sign_heavy_down_light_up" begin
        rho = [1001.0 999.0]
        rho_ref = [1000.0 1000.0]
        b = buoyancy(rho, rho_ref)
        @test b[1] < 0.0
        @test b[2] > 0.0
    end

    @testset "test_pressure_component_decomposition_matches_total_pressure" begin
        cfg = ModelConfig(nx=40, nz=12, dt=5e-4, nt=1, use_hydrostatic=false, momentum_advection_scheme="tvd2")
        model = Model(cfg)
        for _ in 1:30
            step_forward!(model)
        end

        p_hydro, p_corr, buoy_w = pressure_buoyancy_components(model)
        @test size(p_hydro) == size(model.state.p)
        @test size(p_corr) == size(model.state.p)
        @test size(buoy_w) == (cfg.nz - 1, cfg.nx)
        @test all(isapprox.(p_hydro .+ p_corr, model.state.p, atol=1e-8, rtol=1e-8))
    end

    @testset "test_model_froude_limiter_short_run" begin
        cfg = ModelConfig(nx=40, nz=20, dt=5e-4, nt=1, max_froude=0.9, apply_froude_limiter=true)
        model = Model(cfg)
        max_fr = 0.0
        for _ in 1:200
            step_forward!(model)
            max_fr = max(max_fr, compute_diagnostics(model)["froude"])
        end

        @test max_fr <= cfg.max_froude + 1e-6
    end

    @testset "test_requested_domain_grid_dimensions" begin
        cfg = ModelConfig(length=30.0, depth=4.0, nx=100, nz=20)
        model = Model(cfg)
        @test isapprox(model.dx, 0.3)
        @test isapprox(model.dz, 0.2)
    end

    @testset "test_requested_salinity_based_initial_densities" begin
        cfg = ModelConfig(nx=20, nz=6, rho_left=1000.722, rho_right=999.972)
        model = Model(cfg)
        @test all(isapprox.(model.state.rho[:, 1:div(cfg.nx, 2)], cfg.rho_left, atol=1e-12))
        @test all(isapprox.(model.state.rho[:, (div(cfg.nx, 2)+1):end], cfg.rho_right, atol=1e-12))
    end

    @testset "test_initialize_rayleigh_taylor_layering_and_reset" begin
        cfg = ModelConfig(nx=40, nz=20, length=8.0, depth=4.0)
        model = Model(cfg)

        initialize_rayleigh_taylor!(
            model;
            rho_top=1001.0,
            rho_bottom=999.0,
            interface_height_fraction=0.85,
            perturbation_amplitude_fraction=0.0,
            perturbation_wavenumber=1,
        )

        @test all(isapprox.(model.state.rho[1:17, :], 999.0, atol=1e-12))
        @test all(isapprox.(model.state.rho[18:20, :], 1001.0, atol=1e-12))
        @test all(isapprox.(model.state.u, 0.0, atol=1e-12))
        @test all(isapprox.(model.state.w, 0.0, atol=1e-12))
        @test all(isapprox.(model.state.eta, 0.0, atol=1e-12))
        @test model.rho_min_init == 999.0
        @test model.rho_max_init == 1001.0

        initialize_lock_exchange!(model; rho_left=1000.5, rho_right=999.5)
        @test all(isapprox.(model.state.rho[:, 1:20], 1000.5, atol=1e-12))
        @test all(isapprox.(model.state.rho[:, 21:40], 999.5, atol=1e-12))
    end

    @testset "test_initialize_rayleigh_taylor_argument_validation" begin
        model = Model(ModelConfig(nx=20, nz=10, length=10.0, depth=5.0))
        @test_throws ArgumentError initialize_rayleigh_taylor!(model; rho_top=1001.0, rho_bottom=999.0, interface_height_fraction=1.1)
        @test_throws ArgumentError initialize_rayleigh_taylor!(model; rho_top=1001.0, rho_bottom=999.0, perturbation_amplitude_fraction=-1e-3)
        @test_throws ArgumentError initialize_rayleigh_taylor!(model; rho_top=1001.0, rho_bottom=999.0, perturbation_wavenumber=0)
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

    @testset "test_long_run_conservation_and_froude_control" begin
        cfg = ModelConfig(length=30.0, depth=4.0, nx=100, nz=20, dt=5e-4, nt=1, max_froude=0.9, apply_froude_limiter=true)
        model = Model(cfg)
        d0 = compute_diagnostics(model)
        max_fr = 0.0

        for _ in 1:1200
            step_forward!(model)
            d = compute_diagnostics(model)
            max_fr = max(max_fr, d["froude"])
        end

        d1 = compute_diagnostics(model)
        @test abs(d1["water_volume"] - d0["water_volume"]) < 1e-8
        @test abs(d1["constituent_mass"] - d0["constituent_mass"]) < 1e-8
        @test max_fr <= cfg.max_froude + 1e-6
        @test isfinite(d1["div_max"])
    end

    @testset "test_momentum_advection_zero_for_uniform_velocity_fields" begin
        nz, nx = 12, 14
        u = fill(0.1, nz, nx + 1)
        w = fill(-0.03, nz + 1, nx)

        adv_u_1 = advection_term_u(u, w, 0.2, 0.2; scheme="upwind1")
        adv_w_1 = advection_term_w(u, w, 0.2, 0.2; scheme="upwind1")
        adv_u_2 = advection_term_u(u, w, 0.2, 0.2; scheme="tvd2")
        adv_w_2 = advection_term_w(u, w, 0.2, 0.2; scheme="tvd2")

        @test all(isapprox.(adv_u_1[2:end-1, 2:end-1], 0.0, atol=1e-12))
        @test all(isapprox.(adv_w_1[2:end-1, 2:end-1], 0.0, atol=1e-12))
        @test all(isapprox.(adv_u_2[2:end-1, 2:end-1], 0.0, atol=1e-12))
        @test all(isapprox.(adv_w_2[2:end-1, 2:end-1], 0.0, atol=1e-12))
    end

    @testset "test_momentum_advection_conservative_in_closed_box" begin
        nz, nx = 10, 12
        dx, dz = 0.2, 0.25
        Random.seed!(123)

        u = randn(nz, nx + 1) .* 0.2
        w = randn(nz + 1, nx) .* 0.2

        u[:, 1] .= 0.0
        u[:, end] .= 0.0
        w[1, :] .= 0.0
        w[end, :] .= 0.0

        for scheme in ("upwind1", "tvd2")
            adv_u = advection_term_u(u, w, dx, dz; scheme=scheme)
            adv_w = advection_term_w(u, w, dx, dz; scheme=scheme)

            net_u_tendency = sum(adv_u) * dx * dz
            net_w_tendency = sum(adv_w) * dx * dz

            @test abs(net_u_tendency) < 1e-10
            @test abs(net_w_tendency) < 1e-10
        end
    end

    @testset "test_model_runs_with_tvd2_momentum_advection" begin
        cfg = ModelConfig(nx=30, nz=12, dt=5e-4, nt=1, momentum_advection_scheme="tvd2", max_froude=0.9, apply_froude_limiter=true)
        model = Model(cfg)
        for _ in 1:50
            step_forward!(model)
        end
        d = compute_diagnostics(model)
        @test isfinite(d["froude"])
        @test isfinite(d["div_max"])
    end

    @testset "test_baroclinic_pressure_gradient_accelerates_dense_to_right" begin
        cfg = ModelConfig(nx=40, nz=12, dt=5e-4, nt=1)
        model = Model(cfg)
        p_internal = solve_hydrostatic_pressure(model.state.p, model.state.rho, cfg.g, model.dz)
        grad_p_x = grad_x_c(p_internal, model.dx)

        j = div(cfg.nx, 2) + 1
        @test grad_p_x[1, j] < 0.0
        @test (-(1.0 / cfg.rho0) * grad_p_x[1, j]) > 0.0
    end

    @testset "test_short_time_dense_front_moves_right_with_low_damping" begin
        cfg = ModelConfig(nx=100, nz=20, dt=5e-4, nt=1, rayleigh_damping=0.0, apply_froude_limiter=false, momentum_advection_scheme="upwind1")
        model = Model(cfg)

        x = collect(range(model.dx / 2.0, cfg.length - model.dx / 2.0, length=cfg.nx))
        threshold = cfg.rho_right + 0.8 * (cfg.rho_left - cfg.rho_right)

        function rightmost_front(rho_row)
            idx = findall(>(threshold), rho_row)
            if isempty(idx)
                return NaN
            end
            j = maximum(idx)
            if j >= length(rho_row)
                return x[end]
            end
            rj = rho_row[j]
            rjp = rho_row[j + 1]
            if rjp >= threshold || abs(rj - rjp) < 1e-14
                return x[j]
            end
            return x[j] + (rj - threshold) / (rj - rjp) * (x[j + 1] - x[j])
        end

        x0 = rightmost_front(@view model.state.rho[1, :])
        for _ in 1:2000
            step_forward!(model)
        end
        x1 = rightmost_front(@view model.state.rho[1, :])

        @test isfinite(x0)
        @test isfinite(x1)
        @test x1 > x0 + 2e-3
    end

    @testset "test_wind_shear_drives_transport_and_surface_mixing" begin
        # Named tolerances/setup (no hardcoded values inside the test body below).
        NSTEPS = 3000               # -, simulated time = NSTEPS*dt = 6 s
        WIND_STRESS_X = 1.0         # Pa (kg/(m s^2)), applied wind stress
        U_TOP_MARGIN_MIN = 3e-4     # m/s, required u_top_wind - u_top_calm
        DRHO_MARGIN_MIN = 1e-3      # kg/m^3, required drho_interface_calm - drho_interface_wind
        VARZ_MARGIN_MIN = 1e-3      # (kg/m^3)^2, required varz_calm - varz_wind
        MASS_TOL_ATOL = 1e-8        # kg/m
        MASS_TOL_RTOL = 1e-8        # -

        # WHY (NSTEPS, WIND_STRESS_X): under the (now-default) variable top layer the interior mixing this test
        # measures is driven only by the wind-induced overturning circulation (advection), not by a numerical
        # artifact. It is genuine but small over a short run: with the legacy fixed top face, a moving surface with a
        # zero top-face flux acted as an additional, non-physical density source (see docs section 13) that used to
        # dominate this test's signal by roughly an order of magnitude. NSTEPS and WIND_STRESS_X were picked (see
        # test/TESTS_TECHNICAL_DOCUMENTATION.md) so the genuine effect still clears the margins below with headroom.
        cfg_calm = ModelConfig(
            length=10.0,
            depth=4.0,
            nx=80,
            nz=40,
            dt=0.002,
            nt=1,
            use_hydrostatic=true,
            momentum_advection_scheme="upwind1",
            rayleigh_damping=0.0,
            apply_froude_limiter=false,
            write_plots=false,
            write_animation=false,
            write_pressure_buoyancy_diagnostics=false,
        )
        cfg_wind = copy_config(cfg_calm)
        cfg_wind.wind_stress_x = WIND_STRESS_X
        cfg_wind.wind_mixing_diffusivity = 0.02

        model_calm = Model(cfg_calm)
        model_wind = Model(cfg_wind)

        for m in (model_calm, model_wind)
            initialize_internal_seiche!(
                m;
                rho_top=999.0,
                rho_bottom=1001.0,
                interface_height_fraction=0.5,
                interface_tilt_amplitude_fraction=0.0,
            )
        end

        mass0_calm = constituent_mass(model_calm.state.rho, model_calm.dx, model_calm.dz)
        mass0_wind = constituent_mass(model_wind.state.rho, model_wind.dx, model_wind.dz)

        for _ in 1:NSTEPS
            step_forward!(model_calm)
            step_forward!(model_wind)
        end

        u_top_calm = mean(abs.(@view model_calm.state.u[cfg_calm.nz, 2:cfg_calm.nx]))
        u_top_wind = mean(abs.(@view model_wind.state.u[cfg_wind.nz, 2:cfg_wind.nx]))

        k_interface = Int(cld(cfg_calm.nz, 2))
        drho_interface_calm = mean(abs.(@view(model_calm.state.rho[k_interface, :]) .- @view(model_calm.state.rho[k_interface + 1, :])))
        drho_interface_wind = mean(abs.(@view(model_wind.state.rho[k_interface, :]) .- @view(model_wind.state.rho[k_interface + 1, :])))

        varz_calm = mean([var(@view model_calm.state.rho[:, i]) for i in 1:cfg_calm.nx])
        varz_wind = mean([var(@view model_wind.state.rho[:, i]) for i in 1:cfg_wind.nx])

        # WHY (no `eta=`): mass conservation is checked with the plain (non-eta) constituent_mass on purpose. The
        # wind-driven set-up is close to antisymmetric about the domain centre (classic closed-basin tilt), so
        # sum(eta)*dx stays near zero and the plain and eta-aware diagnostics agree; using the plain one here also
        # exercises that the top-layer density kernel does not silently break the legacy mass diagnostic.
        mass1_calm = constituent_mass(model_calm.state.rho, model_calm.dx, model_calm.dz)
        mass1_wind = constituent_mass(model_wind.state.rho, model_wind.dx, model_wind.dz)

        @test u_top_wind > u_top_calm + U_TOP_MARGIN_MIN
        @test drho_interface_wind < drho_interface_calm - DRHO_MARGIN_MIN
        @test varz_wind < varz_calm - VARZ_MARGIN_MIN
        @test isapprox(mass1_calm, mass0_calm, atol=MASS_TOL_ATOL, rtol=MASS_TOL_RTOL)
        @test isapprox(mass1_wind, mass0_wind, atol=MASS_TOL_ATOL, rtol=MASS_TOL_RTOL)
    end

    @testset "test_single_vegetation_stem_under_prescribed_flow" begin
        cfg = ModelConfig(length=20.0, depth=4.0, nx=80, nz=20, dt=0.01, nt=1, use_hydrostatic=true, rayleigh_damping=0.0, apply_froude_limiter=false, write_plots=false, write_animation=false, write_pressure_buoyancy_diagnostics=false)
        model = Model(cfg)

        stem = VegetationStem(base_x=cfg.length / 2.0, height=0.8 * cfg.depth, diameter=0.03, drag_coefficient=1.5, stiffness=1.0, damping=0.2, mass_per_length=0.8, sample_points=12, source_strength=1.0)
        attach_vegetation!(model, VegetationModel(stem))

        z_centers = collect(range(model.dz / 2.0, cfg.depth - model.dz / 2.0, length=cfg.nz))
        z0 = 0.05 * cfg.depth
        log_profile = log.(1 .+ 20.0 .* (z_centers .- z0) ./ (cfg.depth - z0))
        log_profile ./= maximum(log_profile)

        @views for i in 1:(cfg.nx + 1)
            model.state.u[:, i] .= 0.12 .* log_profile
        end
        fill!(model.state.w, 0.0)
        fill!(model.state.eta, 0.0)
        fill!(model.state.p, 0.0)

        source = zeros(size(model.state.u))
        vegetation_source!(source, model.vegetation, model.state.u, cfg.dt, model.dx, model.dz, cfg.rho0)

        @test model.vegetation.stems[1].angle > 0.0
        @test sum(source) < 0.0
        @test maximum(abs.(source)) > 0.0
    end

    @testset "test_single_vegetation_stem_tracks_time_oscillating_flow" begin
        cfg = ModelConfig(length=10.0, depth=2.0, nx=50, nz=12, dt=0.01, nt=1, use_hydrostatic=true, rayleigh_damping=0.0, apply_froude_limiter=false, write_plots=false, write_animation=false, write_pressure_buoyancy_diagnostics=false)
        model = Model(cfg)

        stem = VegetationStem(base_x=0.5 * cfg.length, height=0.75 * cfg.depth, diameter=0.025, drag_coefficient=1.6, stiffness=0.25, damping=0.05, mass_per_length=0.7, sample_points=10, source_strength=1.0)
        veg = VegetationModel(stem)

        angles = Float64[]
        net_sources = Float64[]
        source = zeros(size(model.state.u))
        z_centers = collect(range(model.dz / 2.0, cfg.depth - model.dz / 2.0, length=cfg.nz))

        for n in 1:120
            t = (n - 1) * cfg.dt
            amplitude = 0.45 * sin(2.0 * pi * 0.8 * t)
            for i in 1:(cfg.nx + 1)
                for k in 1:cfg.nz
                    model.state.u[k, i] = amplitude * (0.35 + 0.65 * z_centers[k] / cfg.depth)
                end
            end
            vegetation_source!(source, veg, model.state.u, cfg.dt, model.dx, model.dz, cfg.rho0)
            push!(angles, veg.stems[1].angle)
            push!(net_sources, sum(source))
        end

        @test maximum(angles) > 0.0
        @test maximum(abs.(angles)) > 1e-4
        @test maximum(net_sources) > 0.0
        @test minimum(net_sources) < 0.0
    end

    @testset "test_vegetation_damps_free_surface_wave_motion" begin
        cfg = ModelConfig(length=20.0, depth=4.0, nx=80, nz=20, dt=5e-4, nt=1, use_hydrostatic=true, momentum_advection_scheme="upwind1", rayleigh_damping=0.0, apply_froude_limiter=false, write_plots=false, write_animation=false, write_pressure_buoyancy_diagnostics=false)
        model_ref = Model(cfg)
        model_veg = Model(cfg)

        stem = VegetationStem(base_x=0.5 * cfg.length, height=0.85 * cfg.depth, diameter=0.03, drag_coefficient=1.8, stiffness=1.0, damping=0.25, mass_per_length=0.8, sample_points=12, source_strength=1.0)
        attach_vegetation!(model_veg, VegetationModel(stem))

        x_centers = collect(range(model_ref.dx / 2.0, cfg.length - model_ref.dx / 2.0, length=cfg.nx))
        wave_amp = 0.08
        for i in 1:cfg.nx
            model_ref.state.eta[i] = wave_amp * sin(2.0 * pi * x_centers[i] / cfg.length)
            model_veg.state.eta[i] = wave_amp * sin(2.0 * pi * x_centers[i] / cfg.length)
        end
        fill!(model_ref.state.u, 0.0)
        fill!(model_ref.state.w, 0.0)
        fill!(model_ref.state.p, 0.0)
        fill!(model_veg.state.u, 0.0)
        fill!(model_veg.state.w, 0.0)
        fill!(model_veg.state.p, 0.0)

        for _ in 1:250
            step_forward!(model_ref)
            step_forward!(model_veg)
        end

        amp_ref = maximum(abs, model_ref.state.eta)
        amp_veg = maximum(abs, model_veg.state.eta)

        @test model_veg.vegetation.stems[1].angle != 0.0
        @test amp_veg < amp_ref
    end

    @testset "test_vegetation_slows_lock_exchange_front" begin
        cfg_ref = ModelConfig(length=30.0, depth=4.0, nx=100, nz=20, dt=5e-4, nt=1, use_hydrostatic=false, momentum_advection_scheme="tvd2", rayleigh_damping=0.01, apply_froude_limiter=true, max_froude=0.9, write_plots=false, write_animation=false, write_pressure_buoyancy_diagnostics=false)
        cfg_veg = copy_config(cfg_ref)

        model_ref = Model(cfg_ref)
        model_veg = Model(cfg_veg)

        stem = VegetationStem(base_x=0.5 * cfg_ref.length, height=0.9 * cfg_ref.depth, diameter=0.035, drag_coefficient=2.0, stiffness=1.2, damping=0.35, mass_per_length=0.9, sample_points=14, source_strength=1.1)
        attach_vegetation!(model_veg, VegetationModel(stem))

        initialize_lock_exchange!(model_ref)
        initialize_lock_exchange!(model_veg)

        x_centers = collect(range(model_ref.dx / 2.0, cfg_ref.length - model_ref.dx / 2.0, length=cfg_ref.nx))
        threshold = cfg_ref.rho_right + 0.8 * (cfg_ref.rho_left - cfg_ref.rho_right)

        function rightmost_front(rho_row)
            idx = findall(>(threshold), rho_row)
            if isempty(idx)
                return NaN
            end
            j = maximum(idx)
            if j >= length(rho_row)
                return x_centers[end]
            end
            rj = rho_row[j]
            rjp = rho_row[j + 1]
            if rjp >= threshold || abs(rj - rjp) < 1e-14
                return x_centers[j]
            end
            return x_centers[j] + (rj - threshold) / (rj - rjp) * (x_centers[j + 1] - x_centers[j])
        end

        for _ in 1:600
            step_forward!(model_ref)
            step_forward!(model_veg)
        end

        x_ref = rightmost_front(@view model_ref.state.rho[1, :])
        x_veg = rightmost_front(@view model_veg.state.rho[1, :])

        @test isfinite(x_ref)
        @test isfinite(x_veg)
        @test x_veg < x_ref
        @test model_veg.vegetation.stems[1].angle != 0.0
    end

    @testset "test_nonhydrostatic_branch_produces_pressure_correction" begin
        cfg = ModelConfig(nx=40, nz=12, dt=5e-4, nt=1, use_hydrostatic=false, momentum_advection_scheme="tvd2")
        model = Model(cfg)
        for _ in 1:30
            step_forward!(model)
        end

        d = compute_diagnostics(model)
        @test d["nonhydro_pcorr_max"] > 0.0
        @test isfinite(d["nonhydro_pcorr_max"])
    end

    @testset "test_nonhydrostatic_solution_differs_from_hydrostatic" begin
        base = ModelConfig(nx=40, nz=12, dt=5e-4, nt=1, momentum_advection_scheme="tvd2", rayleigh_damping=0.01)
        cfg_h = copy_config(base)
        cfg_h.use_hydrostatic = true
        cfg_nh = copy_config(base)
        cfg_nh.use_hydrostatic = false

        model_h = Model(cfg_h)
        model_nh = Model(cfg_nh)

        for _ in 1:50
            step_forward!(model_h)
            step_forward!(model_nh)
        end

        diff_u = maximum(abs, model_h.state.u .- model_nh.state.u)
        diff_eta = maximum(abs, model_h.state.eta .- model_nh.state.eta)
        @test diff_u > 1e-12
        @test diff_eta > 1e-12
    end

    # ---- Non-hydrostatic free surface (docs section 14) --------------------------------------------------------
    NH_ETA_RESPONSE_MIN = 1e-5     # m: max|eta| a genuinely internally-forced non-hydrostatic run must clear
    NH_KINEMATIC_TOL = 1e-9        # round-off-level kinematic/continuity residual after the reordering fix
    NH_PERIOD_REL_TOL = 5e-3       # relative tolerance for the (still shallow-water) surface-wave period match

    @testset "test_verify_nonhydrostatic_free_surface_responds_to_internal_forcing" begin
        # WHY: before the section-14 reordering fix, the barotropic solve used the non-hydrostatic projection's
        # (exactly divergence-free) u_hat; summed over depth that forced the free-surface transport divergence to
        # equal minus the previous step's stale top-face w_hat, giving eta^{n+1} ~= eta^n + dt*eta_t^n -- a
        # homogeneous recursion with eta=0 a self-consistent fixed point. That made the non-hydrostatic mode a rigid
        # lid (eta pinned at round-off, ~1e-13 m here) regardless of real internal (baroclinic) forcing. This lock
        # exchange has strong internal forcing (Delta_rho/rho0 ~ 7.5e-4) and no free-surface forcing of its own
        # (eta starts at 0); a genuinely coupled free surface must still respond.
        cfg = ModelConfig(nx=60, nz=20, length=30.0, depth=4.0, dt=5e-4, nt=1, use_hydrostatic=false,
            momentum_advection_scheme="tvd2", rayleigh_damping=0.01, apply_froude_limiter=true, max_froude=0.9)
        model = Model(cfg)
        for _ in 1:400
            step_forward!(model)
        end

        div = div_x_u(model.state.u, model.dx) .+ div_z_w(model.state.w, model.dz)
        @test maximum(abs, model.state.eta) > NH_ETA_RESPONSE_MIN
        @test model.last_kinematic_mismatch < NH_KINEMATIC_TOL
        @test maximum(abs, div) < NH_KINEMATIC_TOL
    end

    @testset "test_nonhydrostatic_surface_wave_period_matches_shallow_water_not_full_dispersion" begin
        # WHY: this is a documented CURRENT LIMITATION, not a physics validation (contrast the test_verify_* /
        # test_validate_* naming convention). The reordering fix above lets eta respond to internal (baroclinic)
        # forcing, but a pure free-surface gravity wave (uniform density, so no baroclinic pressure gradient at all)
        # is still governed only by the depth-uniform barotropic surface-slope force -g*d(eta)/dx, applied outside
        # the elliptic (non-hydrostatic) solve; with uniform density the predictor's w has no buoyancy/internal-
        # pressure forcing either, so the field entering the projection is already very close to the
        # continuity-consistent (depth-uniform-u, linear-w) state and the projection has essentially nothing left to
        # correct. The result: this mode's surface-wave period still matches the shallow-water (hydrostatic)
        # relation T=2L/sqrt(gH), not the full linear dispersion relation omega^2=g*k*tanh(k*H) (they differ by
        # about 31% for the k*H~1.57 case here). Achieving genuine surface-wave dispersion needs the free-surface
        # dynamic condition (p=0 at the surface) coupled INTO the elliptic solve as a Dirichlet condition, replacing
        # the current depth-uniform force applied outside it -- a substantially larger change (see docs section 14)
        # left for a possible further stage. This test pins the CURRENT behaviour so a future fix updates or removes
        # it deliberately, rather than an unnoticed silent change.
        L, H, g_ref, dt = 8.0, 4.0, 9.81, 0.005
        k_wave = pi / L
        period_shallow = 2.0 * pi / (k_wave * sqrt(g_ref * H))
        period_exact = 2.0 * pi / sqrt(g_ref * k_wave * tanh(k_wave * H))
        @test (period_exact - period_shallow) / period_shallow > 0.25   # confirms this domain resolves the two relations

        function surface_wave_period(nx::Int, nz::Int)
            cfg = ModelConfig(nx=nx, nz=nz, length=L, depth=H, dt=dt, use_hydrostatic=false,
                momentum_advection_scheme="upwind1", rayleigh_damping=0.0, apply_froude_limiter=false)
            model = Model(cfg)
            x = [(i - 0.5) * model.dx for i in 1:nx]
            model.state.eta .= 4e-5 .* cos.(pi .* x ./ L)   # tiny amplitude: stays in the linear regime
            fill!(model.state.rho, 1000.0)
            refresh_density_metadata!(model)

            t_prev, e_prev = 0.0, model.state.eta[1]
            for n in 1:round(Int, 0.6 * period_shallow / dt)
                step_forward!(model)
                e = model.state.eta[1]
                if e <= 0.0
                    t = n * dt
                    return 4.0 * (t_prev + e_prev / (e_prev - e) * (t - t_prev))
                end
                t_prev, e_prev = n * dt, e
            end
            return NaN
        end

        period_num = surface_wave_period(40, 20)
        @test isfinite(period_num)
        @test isapprox(period_num, period_shallow; rtol=NH_PERIOD_REL_TOL)
    end

    # ---- Non-hydrostatic Dirichlet free surface (docs section 15) ----------------------------------------------
    DIRICHLET_TOL_IDENTITY = 1e-11    # discrete identities that hold to round-off (same magnitude as TOL_IDENTITY
                                       # below; defined again here since this section runs before that one)
    DIRICHLET_PERIOD_REL_TOL = 0.02   # accepted relative period error on the finer grid (observed: ~0.4-0.8%)
    DIRICHLET_SHALLOW_GAP_MIN = 0.25  # same domain/gap check as the Neumann (legacy) test above
    DIRICHLET_PCORR_MAX_SANE = 5.0    # Pa: sanity bound on |p_corr| for a ~1e-4 m amplitude wave (rho0*g*eta ~ 1 Pa)

    @testset "test_verify_nonhydro_dirichlet_reference_pressure_removes_uniform_density_offset" begin
        # WHY: solve_hydrostatic_pressure! integrates the FULL density, not the anomaly relative to the reference
        # profile; grad_p_z_reference (Model field) removes that reference-profile contribution before it reaches the
        # Dirichlet-top projection (docs section 15, Model struct docstring). This test guards the mechanism directly:
        # a regression here would silently reintroduce the ~1e4x-oversized, physically-meaningless p_corr magnitude
        # that motivated it (found empirically during development; see docs section 15).
        cfg_off = ModelConfig(nx=10, nz=8, use_hydrostatic=false)
        @test all(iszero, Model(cfg_off).grad_p_z_reference)   # not computed unless nonhydro_surface_dirichlet

        L, H, rho0, g = 8.0, 4.0, 1000.0, 9.81
        cfg_on = ModelConfig(nx=10, nz=8, length=L, depth=H, use_hydrostatic=false, pressure_solver_method="sparse", nonhydro_surface_dirichlet=true)
        model = Model(cfg_on)
        fill!(model.state.rho, rho0)
        refresh_density_metadata!(model)
        # Uniform density equal to rho0: the reference IS the actual state, so its gradient is the full (analytically
        # exact) hydrostatic one, -rho0*g, at every interior face.
        @test all(isapprox.(model.grad_p_z_reference[2:end-1, :], -rho0 * g; rtol=DIRICHLET_TOL_IDENTITY))
        @test all(iszero, model.grad_p_z_reference[[1, end], :])   # grad_z_c! leaves the two boundary faces at 0

        # A non-uniform (two-layer) reference profile: x-independent (built from column 1, repeated across x) but
        # z-dependent (not the same value in every layer).
        cfg_layered = ModelConfig(nx=10, nz=8, length=L, depth=H, use_hydrostatic=false, pressure_solver_method="sparse", nonhydro_surface_dirichlet=true)
        model2 = Model(cfg_layered)
        initialize_internal_seiche!(model2; rho_top=999.0, rho_bottom=1001.0, interface_height_fraction=0.5, interface_tilt_amplitude_fraction=0.0)
        gref = model2.grad_p_z_reference
        @test all(isapprox.(gref, gref[:, 1]; rtol=DIRICHLET_TOL_IDENTITY))   # x-independent (matches column 1)
        @test maximum(gref[2:end-1, 1]) - minimum(gref[2:end-1, 1]) > DIRICHLET_TOL_IDENTITY   # genuinely z-dependent

        # Full-model regression guard: for a tiny-amplitude (~1e-4 m) surface wave, the corrected p_corr stays at a
        # physically sane magnitude (rho0*g*eta ~ 1 Pa), not the ~3.7e4 Pa the uncorrected reference profile gave.
        cfg_wave = ModelConfig(nx=20, nz=15, length=L, depth=H, dt=0.005, use_hydrostatic=false,
            pressure_solver_method="sparse", nonhydro_surface_dirichlet=true,
            momentum_advection_scheme="upwind1", rayleigh_damping=0.0, apply_froude_limiter=false)
        wave_model = Model(cfg_wave)
        x = [(i - 0.5) * wave_model.dx for i in 1:cfg_wave.nx]
        wave_model.state.eta .= 1e-4 .* cos.(pi .* x ./ L)
        fill!(wave_model.state.rho, rho0)
        refresh_density_metadata!(wave_model)
        for _ in 1:20
            step_forward!(wave_model)
        end
        @test wave_model.last_nonhydro_pcorr_max < DIRICHLET_PCORR_MAX_SANE
    end

    @testset "test_validate_nonhydrostatic_dirichlet_surface_wave_matches_full_dispersion" begin
        # Companion to test_nonhydrostatic_surface_wave_period_matches_shallow_water_not_full_dispersion above (same
        # domain and setup, k*H~1.57 so the two relations differ by >25%): with nonhydro_surface_dirichlet=true (docs
        # section 15) the SAME kind of pure free-surface gravity wave becomes genuinely dispersive, converging to the
        # full relation omega^2=g*k*tanh(k*H) instead of the shallow-water one.
        #
        # WHY (pressure_solver_method="sparse"): nonhydro_surface_dirichlet requires it; the spectral (DCT) solver's
        # fast transform represents only a Neumann-Neumann eigenbasis in z and cannot express the Dirichlet top
        # condition this needs (docs section 15, PressureCorrectionSolver docstring).
        L, H, g_ref = 8.0, 4.0, 9.81
        k_wave = pi / L
        period_shallow = 2.0 * pi / (k_wave * sqrt(g_ref * H))
        period_exact = 2.0 * pi / sqrt(g_ref * k_wave * tanh(k_wave * H))
        @test (period_exact - period_shallow) / period_shallow > DIRICHLET_SHALLOW_GAP_MIN

        function dirichlet_surface_wave_period(nx::Int, nz::Int, dt::Real)
            cfg = ModelConfig(nx=nx, nz=nz, length=L, depth=H, dt=dt, use_hydrostatic=false,
                pressure_solver_method="sparse", nonhydro_surface_dirichlet=true,
                momentum_advection_scheme="upwind1", rayleigh_damping=0.0, apply_froude_limiter=false)
            model = Model(cfg)
            x = [(i - 0.5) * model.dx for i in 1:nx]
            model.state.eta .= 4e-5 .* cos.(pi .* x ./ L)   # tiny amplitude: stays in the linear regime
            fill!(model.state.rho, 1000.0)
            refresh_density_metadata!(model)

            t_prev, e_prev = 0.0, model.state.eta[1]
            for n in 1:round(Int, 0.6 * period_exact / dt)
                step_forward!(model)
                e = model.state.eta[1]
                if e <= 0.0
                    t = n * dt
                    return 4.0 * (t_prev + e_prev / (e_prev - e) * (t - t_prev))
                end
                t_prev, e_prev = n * dt, e
            end
            return NaN
        end

        period_coarse = dirichlet_surface_wave_period(80, 80, 0.002)
        period_fine = dirichlet_surface_wave_period(80, 160, 0.001)
        @test isfinite(period_coarse) && isfinite(period_fine)

        err_coarse = abs(period_coarse - period_exact) / period_exact
        err_fine = abs(period_fine - period_exact) / period_exact
        @test err_fine < err_coarse            # converges toward the exact relation under grid refinement
        @test err_fine < DIRICHLET_PERIOD_REL_TOL
        # decisively dispersive, not merely "a bit off shallow water": closer to the exact relation than to shallow water
        @test abs(period_fine - period_exact) < abs(period_fine - period_shallow)
    end

    # ---- Variable-thickness top layer (hydrostatic mode) ------------------------------------------------------
    # Named tolerances (no hardcoded tolerances inside the tests).
    TOL_IDENTITY = 1e-11          # discrete identities that hold to round-off (kernel level)
    TOL_MODEL_IDENTITY = 1e-9     # same, after many steps of the full model (round-off accumulates, mass ~ 1e5 kg/m)
    TOL_LEGACY_VIOLATION = 1e-6   # kg/m^3: the legacy top-face flux must violate constancy by at least this much
    CONTRAST_MIN = 1e-4           # m/s^2: the legacy flux-form top-row tendency of a uniform u must exceed this
    SEICHE_ORDER_MIN = 1.6        # accepted range of the observed convergence order of a second-order scheme
    SEICHE_ORDER_MAX = 2.4
    SEICHE_REL_ERROR_MAX = 2e-3   # accepted relative period error on the finest grid

    @testset "test_verify_hydrostatic_pressure_top_half_layer" begin
        nz, nx = 6, 5
        g, dz, rho0 = 9.81, 0.5, 1000.0
        rho = fill(rho0, nz, nx)

        p = solve_hydrostatic_pressure(zeros(nz, nx), rho, g, dz; top_half_layer=true)
        # analytical: the centre of layer k lies (nz - k + 1/2)*dz below the surface, p = rho0*g*depth
        p_exact = [rho0 * g * dz * (nz - k + 0.5) for k in 1:nz, i in 1:nx]
        @test all(isapprox.(p, p_exact; rtol=TOL_IDENTITY))

        p_legacy = solve_hydrostatic_pressure(zeros(nz, nx), rho, g, dz)
        @test all(isapprox.(p .- p_legacy, 0.5 * rho0 * g * dz; rtol=TOL_IDENTITY))
    end

    @testset "test_verify_constituent_mass_with_variable_top_layer" begin
        nz, nx = 5, 7
        dx, dz = 0.4, 0.3
        rho = fill(1000.0, nz, nx)
        eta = collect(range(-0.05, 0.05, length=nx))
        # analytical: rho0 * dx * sum_i (H + eta_i) with H = nz*dz
        exact = 1000.0 * dx * sum(nz * dz .+ eta)
        @test isapprox(constituent_mass(rho, dx, dz; eta=eta), exact; rtol=TOL_IDENTITY)
        @test isapprox(constituent_mass(rho, dx, dz), 1000.0 * dx * nx * nz * dz; rtol=TOL_IDENTITY)
    end

    @testset "test_verify_variable_thickness_barotropic_reduces_to_constant_depth" begin
        cfg = ModelConfig(nx=24, nz=6, dt=0.02)
        dx, dz = cfg.length / cfg.nx, cfg.depth / cfg.nz
        solver = BarotropicSolver(cfg, dx, dz)

        Random.seed!(3)
        u_hat = 0.05 .* randn(cfg.nz, cfg.nx + 1)
        u_hat[:, 1] .= 0.0
        u_hat[:, end] .= 0.0
        eta_old = 0.01 .* randn(cfg.nx)

        eta_const = zeros(cfg.nx)
        eta_var = zeros(cfg.nx)
        solve!(solver, eta_const, u_hat, eta_old)
        # WHY: thickness dz at every face (i.e. the thickness of a flat surface) must reproduce the constant-depth solve
        solve_variable_thickness!(solver, eta_var, u_hat, eta_old, fill(dz, cfg.nx + 1))

        @test maximum(abs, eta_var .- eta_const) < TOL_IDENTITY
    end

    @testset "test_verify_variable_thickness_barotropic_closes_volume_budget" begin
        cfg = ModelConfig(nx=24, nz=6, dt=0.02)
        nx, nz, dt, g = cfg.nx, cfg.nz, cfg.dt, cfg.g
        dx, dz = cfg.length / nx, cfg.depth / nz
        solver = BarotropicSolver(cfg, dx, dz)

        Random.seed!(4)
        u_hat = 0.05 .* randn(nz, nx + 1)
        u_hat[:, 1] .= 0.0
        u_hat[:, end] .= 0.0
        eta_old = 0.3 * dz .* (2.0 .* rand(nx) .- 1.0)   # surface within +-30 % of the top layer

        h_face = zeros(nx + 1)
        top_layer_face_thickness!(h_face, eta_old, dz)
        eta_new = zeros(nx)
        solve_variable_thickness!(solver, eta_new, u_hat, eta_old, h_face)

        # final velocities as in step_forward!: free-surface slope in every layer, walls stay closed
        u = copy(u_hat)
        for i in 2:nx
            u[:, i] .= u_hat[:, i] .- g * dt * (eta_new[i] - eta_new[i - 1]) / dx
        end
        # layer-integrated transport with the OLD top-layer thickness
        transport = [dz * sum(@view u[1:(nz - 1), i]) + h_face[i] * u[nz, i] for i in 1:(nx + 1)]
        residual = eta_new .- eta_old .+ dt .* (transport[2:end] .- transport[1:end-1]) ./ dx

        @test maximum(abs, residual) < TOL_IDENTITY
    end

    @testset "test_verify_top_layer_density_transport_conserves_mass_and_constancy" begin
        cfg = ModelConfig(nx=30, nz=8, length=15.0, depth=4.0, dt=0.005, use_hydrostatic=true, variable_top_layer=true, rayleigh_damping=0.0, apply_froude_limiter=false, momentum_advection_scheme="upwind1")
        model = Model(cfg)
        x = [(i - 0.5) * model.dx for i in 1:cfg.nx]
        model.state.eta .= 0.2 * model.dz .* cos.(pi .* x ./ cfg.length)
        for _ in 1:20
            step_forward!(model)
        end

        # After a step: ws.eta_old = eta^n, state.eta = eta^{n+1}, ws.h_top_face built from eta^n, (u, w) consistent with both.
        st, ws = model.state, model.ws
        nz, nx, dx, dz = cfg.nz, cfg.nx, model.dx, model.dz

        Random.seed!(5)
        rho = 998.0 .+ 4.0 .* rand(nz, nx)
        rho_new = similar(rho)
        advect_density_upwind_top_layer!(rho_new, rho, st.u, st.w, cfg.dt, dx, dz, ws.density_ws, ws.h_top_face, ws.eta_old, st.eta)
        mass_old = constituent_mass(rho, dx, dz; eta=ws.eta_old)
        mass_new = constituent_mass(rho_new, dx, dz; eta=st.eta)
        @test abs(mass_new - mass_old) < TOL_IDENTITY * abs(mass_old)

        rho_uniform = fill(1000.0, nz, nx)
        advect_density_upwind_top_layer!(rho_new, rho_uniform, st.u, st.w, cfg.dt, dx, dz, ws.density_ws, ws.h_top_face, ws.eta_old, st.eta)
        @test maximum(abs, rho_new .- 1000.0) < TOL_IDENTITY
    end

    @testset "test_verify_variable_top_layer_uniform_density_stays_uniform" begin
        function max_density_deviation(variable_top::Bool)
            cfg = ModelConfig(nx=40, nz=8, length=20.0, depth=4.0, dt=0.005, use_hydrostatic=true, variable_top_layer=variable_top, rayleigh_damping=0.0, apply_froude_limiter=false, momentum_advection_scheme="upwind1")
            model = Model(cfg)
            x = [(i - 0.5) * model.dx for i in 1:cfg.nx]
            model.state.eta .= 0.05 .* cos.(pi .* x ./ cfg.length)
            fill!(model.state.rho, 1000.0)
            refresh_density_metadata!(model)
            # WHY: with a uniform initial density the clamp bounds collapse to [1000, 1000] and would hide any deviation
            model.rho_min_init = 900.0
            model.rho_max_init = 1100.0
            for _ in 1:100
                step_forward!(model)
            end
            return maximum(abs, model.state.rho .- 1000.0)
        end

        @test max_density_deviation(true) < TOL_MODEL_IDENTITY
        # legacy fixed-grid treatment: zero flux through a top face that moves with d(eta)/dt is a spurious source
        @test max_density_deviation(false) > TOL_LEGACY_VIOLATION
    end

    @testset "test_verify_variable_top_layer_closes_kinematic_and_volume_budgets" begin
        cfg = ModelConfig(nx=60, nz=10, dt=0.005, use_hydrostatic=true, variable_top_layer=true, momentum_advection_scheme="tvd2")
        model = Model(cfg)
        x = [(i - 0.5) * model.dx for i in 1:cfg.nx]
        model.state.eta .= 0.02 .* cos.(pi .* x ./ cfg.length)
        refresh_density_metadata!(model)

        d0 = compute_diagnostics(model)
        max_mismatch = 0.0
        for _ in 1:200
            step_forward!(model)
            max_mismatch = max(max_mismatch, model.last_kinematic_mismatch)
        end
        d1 = compute_diagnostics(model)

        @test max_mismatch < TOL_IDENTITY
        @test abs(d1["water_volume"] - d0["water_volume"]) < TOL_MODEL_IDENTITY * d0["water_volume"]
        @test abs(d1["constituent_mass"] - d0["constituent_mass"]) < TOL_MODEL_IDENTITY * d0["constituent_mass"]
        @test all(isapprox.(model.state.w[end, :], (model.state.eta .- model.ws.eta_old) ./ cfg.dt; atol=TOL_IDENTITY))
    end

    @testset "test_verify_top_row_advective_form_zero_for_uniform_u" begin
        nz, nx = 8, 12
        dx, dz = 0.3, 0.25
        u = fill(0.3, nz, nx + 1)
        Random.seed!(11)
        w = 0.05 .* randn(nz + 1, nx)   # arbitrary vertical velocity: not divergence free

        for scheme in ("upwind1", "tvd2")
            ws_adv = SweTransport2DV.AdvectionWorkspace(nz, nx + 1)
            ws_interp = SweTransport2DV.InterpolationWorkspace(nz, nx)
            out = zeros(nz, nx + 1)

            SweTransport2DV.advection_term_u!(out, u, w, dx, dz, scheme, ws_adv, ws_interp; top_row_advective_form=true)
            @test maximum(abs, out[nz, :]) < TOL_IDENTITY

            # contrast: the flux form is not constancy-preserving when the transporting velocity has a divergence
            SweTransport2DV.advection_term_u!(out, u, w, dx, dz, scheme, ws_adv, ws_interp)
            @test maximum(abs, out[nz, 2:end-1]) > CONTRAST_MIN
        end
    end

    @testset "test_verify_variable_top_layer_config_and_thickness_guards" begin
        @test_throws ArgumentError Model(ModelConfig(nx=20, nz=6, use_hydrostatic=false, variable_top_layer=true))
        @test_throws ArgumentError Model(ModelConfig(nx=20, nz=6, variable_top_layer=true, top_layer_min_fraction=1.5))

        cfg = ModelConfig(nx=20, nz=6, dt=0.005, use_hydrostatic=true, variable_top_layer=true)
        model = Model(cfg)
        model.state.eta .= -0.95 * model.dz   # surface leaves the top layer
        @test_throws ErrorException step_forward!(model)

        # copy_config carries the new options
        cfg2 = copy_config(cfg)
        @test cfg2.variable_top_layer == true
        @test cfg2.top_layer_min_fraction == cfg.top_layer_min_fraction
    end

    @testset "test_verify_variable_top_layer_seiche_period_and_convergence" begin
        # Linear standing wave in a closed basin of uniform density: eta = A*cos(pi*x/L)*cos(omega*t),
        # omega = pi*sqrt(g*H)/L, period T = 2L/sqrt(g*H). Cell 1 samples cos(pi*x_1/L)*cos(omega*t), so its zero
        # crossing lies at omega*t = pi/2 independent of x_1: T_num = 4*t_crossing.
        # WHY: A is tiny on purpose. The zero-crossing time of a finite-amplitude standing wave carries an
        # amplitude-dependent shift of about 0.15*A/H (measured, present in the legacy scheme too), which would mask the
        # second-order spatial error; A/H = 2.5e-5 keeps it below 1e-5 (spatial error on the finest grid: 2.6e-4).
        L, H, A, dt = 20.0, 4.0, 1e-4, 0.005
        g = 9.81
        T_exact = 2.0 * L / sqrt(g * H)

        function seiche_period(nx::Int)
            cfg = ModelConfig(nx=nx, nz=8, length=L, depth=H, dt=dt, use_hydrostatic=true, variable_top_layer=true, rayleigh_damping=0.0, apply_froude_limiter=false, momentum_advection_scheme="upwind1")
            model = Model(cfg)
            x = [(i - 0.5) * model.dx for i in 1:nx]
            model.state.eta .= A .* cos.(pi .* x ./ L)
            fill!(model.state.rho, 1000.0)
            refresh_density_metadata!(model)

            t_prev, e_prev = 0.0, model.state.eta[1]
            for n in 1:round(Int, 0.5 * T_exact / dt)
                step_forward!(model)
                e = model.state.eta[1]
                if e <= 0.0
                    t = n * dt
                    t_cross = t_prev + e_prev / (e_prev - e) * (t - t_prev)   # linear interpolation of the sign change
                    return 4.0 * t_cross
                end
                t_prev, e_prev = n * dt, e
            end
            return NaN
        end

        errors = [abs(seiche_period(nx) - T_exact) / T_exact for nx in (10, 20, 40)]
        @test all(isfinite, errors)
        @test errors[3] < SEICHE_REL_ERROR_MAX
        order_coarse = log2(errors[1] / errors[2])
        order_fine = log2(errors[2] / errors[3])
        @test SEICHE_ORDER_MIN < order_coarse < SEICHE_ORDER_MAX
        @test SEICHE_ORDER_MIN < order_fine < SEICHE_ORDER_MAX
    end
end
