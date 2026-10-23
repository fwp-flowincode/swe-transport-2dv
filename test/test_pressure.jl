# Hydrostatic integration, Poisson matrix and non-hydrostatic pressure solvers (docs sections 6 and 7).
# Run with the full suite (runtests.jl) or on its own: julia --project=. test/test_pressure.jl

include(joinpath(@__DIR__, "test_common.jl"))

@testset "Pressure" begin
    @testset "test_hydrostatic_pressure_monotone_with_depth" begin
        nz, nx = 6, 7
        rho = fill(1000.0, nz, nx)
        p = solve_hydrostatic_pressure(zeros(nz, nx), rho, 9.81, 0.5)

        @test all(isapprox.(p[end, :], 0.0, atol=1e-12))
        @test all(p[1:end-1, :] .>= p[2:end, :])
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
end
