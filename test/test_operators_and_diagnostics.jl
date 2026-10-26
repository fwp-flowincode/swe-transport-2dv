# State layout, discrete grad/div operators, buoyancy and diagnostic quantities.
# Run with the full suite (runtests.jl) or on its own: julia --project=. test/test_operators_and_diagnostics.jl

include(joinpath(@__DIR__, "test_common.jl"))

@testset "Operators and diagnostics" begin
    @testset "test_verify_state_shapes" begin
        st = from_shape(6, 8)
        @test size(st.u) == (6, 9)
        @test size(st.w) == (7, 8)
        @test size(st.p) == (6, 8)
        @test size(st.rho) == (6, 8)
        @test size(st.eta) == (8,)
    end

    @testset "test_verify_grad_and_div_shapes_and_consistency" begin
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

    @testset "test_verify_froude_diagnostic_behaviour" begin
        rho = [1002.0 1000.0; 1002.0 1000.0]
        u = [0.05 -0.05 0.05; 0.05 -0.05 0.05]
        fr = froude_number(u, rho, 1000.0, 9.81, 1.0)
        @test fr > 0.0
        @test isfinite(fr)
    end

    @testset "test_verify_diagnostics_boundary_flux_closed_streamfunction_field" begin
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

    @testset "test_verify_buoyancy_sign_heavy_down_light_up" begin
        rho = [1001.0 999.0]
        rho_ref = [1000.0 1000.0]
        b = buoyancy(rho, rho_ref)
        @test b[1] < 0.0
        @test b[2] > 0.0
    end

    @testset "test_verify_diffusion_flux_form_second_order_convergence" begin
        # Manufactured field on the u-grid (x faces x_i = (i-1) dx, walls at x = 0 and L where phi = 0; cell-centred z):
        #   phi = sin(pi x / L) cos(pi z / H),  nu(z) = NU0 (1 + NU_SLOPE z / H),
        #   d/dx(nu phi_x) + d/dz(nu phi_z) = nu phi_xx + nu' phi_z + nu phi_zz.
        # phi_z = 0 at z = 0 and H, consistent with the zero-flux bed and surface of diffusion_term!.
        L = 2.0                      # m
        H = 1.0                      # m
        NU0 = 1e-3                   # m^2/s
        NU_SLOPE = 0.5               # -, relative change of nu over the depth
        GRIDS = (16, 32, 64)         # nz; nx = 2 nz
        ORDER_MIN = 1.9              # -, expected order 2 in the max norm (top and bottom rows included)
        errors = Float64[]
        for nz in GRIDS
            nx = 2nz
            dx, dz = L / nx, H / nz
            x = [(i - 1) * dx for i in 1:(nx + 1)]
            z = [(k - 0.5) * dz for k in 1:nz]
            phi = [sin(pi * xi / L) * cos(pi * zk / H) for zk in z, xi in x]
            nu = [NU0 * (1 + NU_SLOPE * zk / H) for zk in z, xi in x]
            exact = [NU0 * (1 + NU_SLOPE * zk / H) * (-(pi / L)^2 - (pi / H)^2) * sin(pi * xi / L) * cos(pi * zk / H) -
                     NU0 * NU_SLOPE / H * (pi / H) * sin(pi * xi / L) * sin(pi * zk / H) for zk in z, xi in x]
            res = similar(phi)
            diffusion_term!(res, phi, nu, dx, dz)
            push!(errors, maximum(abs, res[:, 2:nx] .- exact[:, 2:nx]))
            @test all(iszero, res[:, 1]) && all(iszero, res[:, end])   # wall faces
        end
        orders = log2.(errors[1:end-1] ./ errors[2:end])
        @test all(orders .> ORDER_MIN)
    end

    @testset "test_verify_diffusion_flux_form_conserves_column_momentum" begin
        # Periodic in x and uniform in x: only the vertical fluxes act, and with zero flux through the bed and the
        # surface sum_k res[k, i] dz vanishes for any profile and any positive viscosity profile.
        NZ = 24                      # -
        NX = 8                       # -, physical columns (nx + 1 faces with the periodic seam)
        DZ = 0.1                     # m
        DX = 0.2                     # m
        rng = MersenneTwister(4)
        profile = randn(rng, NZ)                     # m/s
        nu_profile = 1e-3 .* (1 .+ rand(rng, NZ))    # m^2/s
        phi = repeat(profile, 1, NX + 1)
        nu = repeat(nu_profile, 1, NX + 1)
        res = similar(phi)
        diffusion_term!(res, phi, nu, DX, DZ; periodic_x=true)
        @test maximum(abs, vec(sum(res; dims=1)) .* DZ) < TOL_IDENTITY
        @test maximum(abs, res) > 0.0
        # constant nu: the scalar method gives the same result as a constant matrix
        res_scalar = similar(phi)
        diffusion_term!(res_scalar, phi, nu_profile[1], DX, DZ; periodic_x=true)
        diffusion_term!(res, phi, fill(nu_profile[1], NZ, NX + 1), DX, DZ; periodic_x=true)
        @test res_scalar == res
    end
end
