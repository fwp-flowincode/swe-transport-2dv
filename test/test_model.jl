# Full-model runs: conservation, stabilisation, configuration and initial conditions, dense-front and wind-shear behaviour.
# Run with the full suite (runtests.jl) or on its own: julia --project=. test/test_model.jl

include(joinpath(@__DIR__, "test_common.jl"))

@testset "Model" begin
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
        # Named setup/tolerances (no hardcoded values inside the test body below).
        T_END = 120.0                        # s, long enough for the closed-basin circulation to develop
        DT = 0.01                            # s
        WIND_STRESS_X_MAX = 0.2              # Pa (kg/(m s^2))
        WIND_RAMP_DURATION = 20.0             # s, stress increases linearly from zero
        U_TOP_MIN = 1e-2                     # m/s, surface flow in the wind direction
        U_BOTTOM_MAX = -1e-4                 # m/s, lower-layer return flow
        W_MIN = 1e-4                         # m/s, vertical motion closing the circulation at the walls
        DRHO_MARGIN_MIN = 1e-3               # kg/m^3, reduction in interface density jump
        VARZ_MARGIN_MIN = 1e-3               # (kg/m^3)^2, reduction in vertical density variance
        MASS_TOL_ATOL = 1e-8        # kg/m
        MASS_TOL_RTOL = 1e-8        # -

        for use_hydrostatic in (true, false)
            cfg = ModelConfig(
                length=10.0,
                depth=4.0,
                nx=80,
                nz=40,
                dt=DT,
                nt=round(Int, T_END / DT),
                use_hydrostatic=use_hydrostatic,
                periodic_x=false,
                momentum_advection_scheme="tvd2",
                transport_advection_scheme="tvd2",
                rayleigh_damping=0.0005,
                wind_stress_x=0.0,
                wind_mixing_diffusivity=0.02,
                write_plots=false,
                write_animation=false,
                write_pressure_buoyancy_diagnostics=false,
            )
            model = Model(cfg)
            initialize_internal_seiche!(
                model;
                rho_top=999.0,
                rho_bottom=1001.0,
                interface_height_fraction=0.5,
                interface_tilt_amplitude_fraction=0.0,
            )
            @test all(iszero, model.state.u)
            @test all(iszero, model.state.w)
            @test all(iszero, model.state.eta)

            eta0 = use_variable_top_layer(cfg) ? model.state.eta : nothing
            mass0 = constituent_mass(model.state.rho, model.dx, model.dz; eta=eta0)
            varz0 = mean([var(@view model.state.rho[:, i]) for i in 1:cfg.nx])
            k_interface = Int(cld(cfg.nz, 2))
            drho_interface0 = mean(abs.(@view(model.state.rho[k_interface, :]) .- @view(model.state.rho[k_interface + 1, :])))

            ramp_steps = round(Int, WIND_RAMP_DURATION / cfg.dt)
            for n in 1:cfg.nt
                cfg.wind_stress_x = WIND_STRESS_X_MAX * min(n / ramp_steps, 1.0)
                step_forward!(model)
            end

            u_top = mean(@view model.state.u[cfg.nz, 2:cfg.nx])
            u_bottom = mean(@view model.state.u[1:cld(cfg.nz, 4), 2:cfg.nx])
            drho_interface1 = mean(abs.(@view(model.state.rho[k_interface, :]) .- @view(model.state.rho[k_interface + 1, :])))
            varz1 = mean([var(@view model.state.rho[:, i]) for i in 1:cfg.nx])
            eta1 = use_variable_top_layer(cfg) ? model.state.eta : nothing
            mass1 = constituent_mass(model.state.rho, model.dx, model.dz; eta=eta1)

            # The closed walls create the opposing lower-layer return flow of the double gyre.
            @test u_top > U_TOP_MIN
            @test u_bottom < U_BOTTOM_MAX
            @test maximum(abs, model.state.w) > W_MIN
            @test drho_interface1 < drho_interface0 - DRHO_MARGIN_MIN
            @test varz1 < varz0 - VARZ_MARGIN_MIN
            @test isapprox(mass1, mass0, atol=MASS_TOL_ATOL, rtol=MASS_TOL_RTOL)
            @test compute_diagnostics(model)["cfl_total"] < 0.5
        end
    end

    @testset "test_periodic_kelvin_helmholtz_boundaries_and_conservation" begin
        cfg = ModelConfig(
            length=8.0,
            depth=4.0,
            nx=32,
            nz=16,
            dt=1e-4,
            nt=1,
            use_hydrostatic=false,
            periodic_x=true,
            pressure_solver_method="sparse",
            rayleigh_damping=0.0,
            apply_froude_limiter=false,
            write_plots=false,
            write_animation=false,
            write_pressure_buoyancy_diagnostics=false,
        )
        model = Model(cfg)
        initialize_kelvin_helmholtz!(
            model;
            rho_top=999.95,
            rho_bottom=1000.05,
            interface_perturbation_amplitude_fraction=0.02,
            interface_perturbation_wavenumber=2,
            u_top=0.2,
            u_bottom=-0.2,
        )

        @test model.state.u[:, 1] ≈ model.state.u[:, end]
        @test model.state.rho[:, 1] ≈ model.state.rho[:, end]
        # the seam face carries the same shear profile as the interior faces (not a wall with u = 0)
        @test model.state.u[:, 1] ≈ model.state.u[:, cfg.nx ÷ 2 + 1]
        @test maximum(model.state.u[:, 1]) > 0.5 * 0.2
        mass0 = constituent_mass(model.state.rho, model.dx, model.dz)
        for _ in 1:3
            step_forward!(model)
            @test model.state.u[:, 1] ≈ model.state.u[:, end]
        end
        @test isapprox(constituent_mass(model.state.rho, model.dx, model.dz), mass0; atol=1e-8, rtol=1e-10)

        A = model.pressure_correction_solver.poisson_matrix
        idx(k, i) = (k - 1) * cfg.nx + i
        @test A[idx(2, 1), idx(2, cfg.nx)] ≈ 1.0 / model.dx^2
    end
end
