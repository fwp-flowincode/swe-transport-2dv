# State layout, discrete grad/div operators, buoyancy and diagnostic quantities.
# Run with the full suite (runtests.jl) or on its own: julia --project=. test/test_operators_and_diagnostics.jl

include(joinpath(@__DIR__, "test_common.jl"))

@testset "Operators and diagnostics" begin
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

    @testset "test_buoyancy_sign_heavy_down_light_up" begin
        rho = [1001.0 999.0]
        rho_ref = [1000.0 1000.0]
        b = buoyancy(rho, rho_ref)
        @test b[1] < 0.0
        @test b[2] > 0.0
    end
end
