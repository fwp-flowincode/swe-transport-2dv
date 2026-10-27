# Density (transport) and momentum advection schemes: upwind1 and tvd2 (docs sections 3 and 4).
# Run with the full suite (runtests.jl) or on its own: julia --project=. test/test_advection.jl

include(joinpath(@__DIR__, "test_common.jl"))

@testset "Advection" begin
    @testset "test_verify_tvd2_momentum_flux_equals_transport_scheme" begin
        # With a constant velocity the tvd2 momentum flux (point velocities averaged to the faces) and the tvd2 density
        # flux (face velocities) are the same flux-limited Lax-Wendroff scheme: identical tendencies for the same dt.
        # Contrast: the semi-discrete momentum flux (dt = 0, no time correction) differs.
        nz, nx = 12, 16
        dx, dz, dt = 0.1, 0.05, 0.02          # m, m, s
        xc = [(i - 0.5) * dx for i in 1:nx]
        zc = [(k - 0.5) * dz for k in 1:nz]
        phi = [sin(2pi * x / (nx * dx)) * cos(pi * z / (nz * dz)) + 0.3 * (x > 0.8) for z in zc, x in xc]
        cu, cw = 0.7, -0.4                    # m/s, Courant numbers 0.14 and 0.16
        ws = SweTransport2DV.AdvectionWorkspace(nz, nx)
        momentum = zeros(nz, nx)
        SweTransport2DV._advective_flux_divergence!(momentum, phi, fill(cu, nz, nx), fill(cw, nz, nx), dx, dz, "tvd2", ws; dt=dt)
        transport = advection_term_rho(fill(cu, nz, nx + 1), fill(cw, nz + 1, nx), phi, dx, dz; scheme="tvd2", dt=dt)
        @test maximum(abs, momentum .- transport) < TOL_IDENTITY * maximum(abs, transport)

        semi_discrete = zeros(nz, nx)
        SweTransport2DV._advective_flux_divergence!(semi_discrete, phi, fill(cu, nz, nx), fill(cw, nz, nx), dx, dz, "tvd2", ws; dt=0.0)
        @test maximum(abs, semi_discrete .- transport) > 1e3 * TOL_IDENTITY * maximum(abs, transport)
    end

    @testset "test_verify_density_advection_zero_for_uniform_density" begin
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

    @testset "test_verify_density_advection_conservative_closed_box" begin
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
        rho1 = advect_density(rho, u, w, dt, dx, dz; scheme="upwind1")
        m1 = constituent_mass(rho1, dx, dz)

        @test isapprox(m0, m1, atol=1e-10, rtol=1e-10)
    end

    @testset "test_verify_periodic_density_advection_conservative" begin
        nz, nx = 6, 12
        dx, dz, dt = 0.1, 0.2, 0.01
        xc = [(i - 0.5) * dx for i in 1:nx]
        rho = repeat(reshape(1000.0 .+ sin.(2.0 * pi .* xc ./ (nx * dx)), 1, nx), nz, 1)
        u = fill(0.15, nz, nx + 1)
        w = zeros(nz + 1, nx)

        for scheme in ("upwind1", "tvd2")
            tendency = advection_term_rho(u, w, rho, dx, dz; scheme=scheme, dt=dt, periodic_x=true)
            @test abs(sum(tendency) * dx * dz) < 1e-10
            rho_new = advect_density(rho, u, w, dt, dx, dz; scheme=scheme, periodic_x=true)
            @test isapprox(constituent_mass(rho_new, dx, dz), constituent_mass(rho, dx, dz); atol=1e-10, rtol=1e-12)
        end
    end

    @testset "test_verify_momentum_advection_zero_for_uniform_velocity_fields" begin
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

    @testset "test_verify_momentum_advection_conservative_in_closed_box" begin
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

    @testset "test_verify_model_runs_with_tvd2_momentum_advection" begin
        cfg = ModelConfig(nx=30, nz=12, dt=5e-4, nt=1, momentum_advection_scheme="tvd2", max_froude=0.9, apply_froude_limiter=true)
        model = Model(cfg)
        for _ in 1:50
            step_forward!(model)
        end
        d = compute_diagnostics(model)
        @test isfinite(d["froude"])
        @test isfinite(d["div_max"])
    end

    TRANSPORT_TVD2_ORDER_MIN = 1.8          # - , expected order 2 (measured 2.4 / 2.2 on the grids below)
    TRANSPORT_TVD2_ORDER_MAX = 2.6          # -
    TRANSPORT_UPWIND1_ORDER_MAX = 1.1       # - , expected order 1 (measured 0.64 / 0.77, pre-asymptotic)
    TRANSPORT_ERROR_RATIO_MIN = 20.0        # - , upwind1 / tvd2 L1 error on the finest grid (measured 75)

    @testset "test_verify_transport_tvd2_convergence_order_smooth_advection" begin
        # Analytical solution: uniform velocity v translates the profile, rho(x, T) = rho0(x - v*T). Gaussian
        # (sigma 0.05 m) in a 1 m closed box, kept far from the walls so that the zero wall fluxes do not matter.
        # Fixed Courant number 0.4, so the L1 error measures the full space-time order. Run along x and along z.
        profile(x) = exp(-((x - 0.3) / 0.05)^2)   # kg/m^3 (perturbation density)
        vel, T_end, courant = 1.0, 0.4, 0.4       # m/s, s, -
        n_transverse = 4

        function l1_error(n::Int, scheme::String, direction::Symbol)
            h = 1.0 / n
            nt = round(Int, T_end / (courant * h / vel))
            dt = T_end / nt
            xc = [(i - 0.5) * h for i in 1:n]
            if direction == :x
                rho = repeat(profile.(xc)', n_transverse, 1)
                u = fill(vel, n_transverse, n + 1); u[:, 1] .= 0.0; u[:, end] .= 0.0
                w = zeros(n_transverse + 1, n)
                dx, dz = h, 1.0
            else
                rho = repeat(profile.(xc), 1, n_transverse)
                u = zeros(n, n_transverse + 1)
                w = fill(vel, n + 1, n_transverse); w[1, :] .= 0.0; w[end, :] .= 0.0
                dx, dz = 1.0, h
            end
            for _ in 1:nt
                rho = advect_density(rho, u, w, dt, dx, dz; scheme=scheme)
            end
            numerical = direction == :x ? rho[2, :] : rho[:, 2]
            return sum(abs, numerical .- profile.(xc .- vel * T_end)) * h
        end

        for direction in (:x, :z)
            ns = (100, 200, 400)
            e_tvd = [l1_error(n, "tvd2", direction) for n in ns]
            e_up = [l1_error(n, "upwind1", direction) for n in ns]
            orders_tvd = log2.(e_tvd[1:end-1] ./ e_tvd[2:end])
            orders_up = log2.(e_up[1:end-1] ./ e_up[2:end])
            @test all(o -> TRANSPORT_TVD2_ORDER_MIN < o < TRANSPORT_TVD2_ORDER_MAX, orders_tvd)
            @test all(o -> 0.0 < o < TRANSPORT_UPWIND1_ORDER_MAX, orders_up)
            @test e_up[end] / e_tvd[end] > TRANSPORT_ERROR_RATIO_MIN
        end
    end

    @testset "test_verify_transport_tvd2_conservative_bounded_and_constancy" begin
        # Divergence-free cellular flow from a streamfunction (zero normal velocity at the walls) at
        # CFL_x + CFL_z <= 1/2: the tvd2 update must conserve mass, keep a uniform field uniform and create no new
        # extrema WITHOUT clamping (local maximum principle of the scheme, see _density_face_fluxes!).
        nz, nx = 24, 32
        dx, dz = 0.1, 0.1                         # m
        zf = reshape(collect(0:nz) ./ nz, nz + 1, 1)
        xf = reshape(collect(0:nx) ./ nx, 1, nx + 1)
        psi = 0.05 .* sin.(pi .* zf) .* sin.(pi .* xf)   # m^2/s at cell corners
        u = (psi[2:end, :] .- psi[1:end-1, :]) ./ dz
        w = -(psi[:, 2:end] .- psi[:, 1:end-1]) ./ dx
        _, _, cfl_total_per_dt = cfl_numbers(u, w, 1.0, dx, dz)
        dt = 0.45 / cfl_total_per_dt              # s, CFL_x + CFL_z = 0.45

        rho_lo, rho_hi = 999.0, 1001.0            # kg/m^3, lock-exchange-like step
        rho = [i <= nx ÷ 2 ? rho_hi : rho_lo for k in 1:nz, i in 1:nx]
        m0 = constituent_mass(rho, dx, dz)
        for _ in 1:200
            rho = advect_density(rho, u, w, dt, dx, dz; scheme="tvd2")
        end
        @test abs(constituent_mass(rho, dx, dz) - m0) < TOL_IDENTITY * m0
        @test minimum(rho) >= rho_lo - TOL_IDENTITY * rho_lo
        @test maximum(rho) <= rho_hi + TOL_IDENTITY * rho_hi
        @test maximum(rho) - minimum(rho) > 0.5 * (rho_hi - rho_lo)   # contrast: the front was not smeared out

        rho_uniform = advect_density(fill(1000.0, nz, nx), u, w, dt, dx, dz; scheme="tvd2")
        @test maximum(abs, rho_uniform .- 1000.0) < TOL_IDENTITY * 1000.0
        @test maximum(abs, advection_term_rho(u, w, fill(1000.0, nz, nx), dx, dz; scheme="tvd2")) < TOL_IDENTITY * 1000.0
    end

    @testset "test_verify_model_runs_with_tvd2_transport_and_config_guards" begin
        for variable_top in (true, false)
            cfg = ModelConfig(nx=40, nz=10, dt=0.005, use_hydrostatic=true, variable_top_layer=variable_top, transport_advection_scheme="tvd2", write_plots=false, write_animation=false, write_pressure_buoyancy_diagnostics=false)
            model = Model(cfg)
            d0 = compute_diagnostics(model)
            for _ in 1:100
                step_forward!(model)
            end
            d1 = compute_diagnostics(model)
            @test abs(d1["constituent_mass"] - d0["constituent_mass"]) < TOL_MODEL_IDENTITY * abs(d0["constituent_mass"])
            @test d1["rho_min"] >= model.rho_min_init - TOL_IDENTITY
            @test d1["rho_max"] <= model.rho_max_init + TOL_IDENTITY

            # contrast: the scheme choice actually changes the density field
            cfg_up = copy_config(cfg)
            cfg_up.transport_advection_scheme = "upwind1"
            model_up = Model(cfg_up)
            for _ in 1:100
                step_forward!(model_up)
            end
            @test maximum(abs, model.state.rho .- model_up.state.rho) > TOL_IDENTITY
        end

        @test_throws ArgumentError Model(ModelConfig(nx=10, nz=4, transport_advection_scheme="tvd3"))
        @test copy_config(ModelConfig(transport_advection_scheme="tvd2")).transport_advection_scheme == "tvd2"
        @test ModelConfig().transport_advection_scheme == "upwind1"

        # output file tag names both advection schemes
        tag_cfg = ModelConfig(nx=100, nz=20, length=30.0, depth=4.0, dt=0.0025, use_hydrostatic=true, momentum_advection_scheme="tvd2", transport_advection_scheme="upwind1")
        @test output_case_tag(tag_cfg) == "hydro_mom-tvd2_trans-upwind1_dt0p0025_L30p0_H4p0_dx0p300_dz0p200"
        tag_cfg.transport_advection_scheme = "tvd2"
        @test occursin("_trans-tvd2_", output_case_tag(tag_cfg))
    end
end
