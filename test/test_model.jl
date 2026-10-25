# Full-model runs: conservation, stabilisation, configuration and initial conditions, dense-front and wind-shear behaviour.
# Run with the full suite (runtests.jl) or on its own: julia --project=. test/test_model.jl

include(joinpath(@__DIR__, "test_common.jl"))
# Exact steady wind solutions, wind ramp and diagnostics of the wind-shear experiment.
include(joinpath(@__DIR__, "..", "experiments", "wind_shear", "cases.jl"))

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

    @testset "test_validate_wind_driven_double_gyre" begin
        # Issue #4: a wind stress that builds up gradually over a closed basin with an initially stagnant two-layer
        # stratification drives a double gyre: a clockwise cell in the light upper layer (downwind at the surface,
        # upwind just above the interface) and, through the interfacial shear stress, a counter-clockwise cell in the
        # heavy lower layer (upwind just below the interface, downwind along the bed). Diagnosed with the streamfunction
        # psi(z) = int_0^z u dz' (m^2/s) of the mean u over the middle half of the basin: psi < 0 in the upper and
        # psi > 0 in the lower layer.
        # Named setup/tolerances (no hardcoded values inside the test body below).
        T_END = 600.0                # s, gyres spun up after about 300 s; an internal seiche remains on top of them
        DT = 0.01                    # s
        WIND_STRESS_X_MAX = 0.2      # Pa (kg/(m s^2))
        WIND_RAMP_DURATION = 20.0    # s, stress increases linearly from zero
        AVERAGE_START = 300.0        # s, start of the averaging window (gyres spun up)
        SAMPLE_INTERVAL = 1.0        # s, sampling interval of the averaged profile
        MIXING_LENGTH = 0.2          # m, turbulence_mixing_length
        MAX_EDDY_VISCOSITY = 5e-3    # m^2/s, max_turbulent_viscosity
        PSI_UPPER_MAX = -5e-3        # m^2/s, upper-layer cell (simulated -0.0184 in both modes)
        PSI_LOWER_MIN = 1.5e-3       # m^2/s, lower-layer cell (simulated +0.0054 hydrostatic, +0.0057 non-hydrostatic)
        U_TOP_MIN = 2e-2             # m/s, mean surface flow in the wind direction (simulated 0.045 in both modes)
        U_BED_MIN = 1.5e-3           # m/s, mean downwind flow along the bed (simulated 0.0057 and 0.0060)
        DRHO_MARGIN_MIN = 0.3        # kg/m^3, reduction of the 2 kg/m^3 interface jump (simulated 1.05 and 1.26)
        VARZ_MARGIN_MIN = 0.015      # (kg/m^3)^2, reduction of the vertical density variance 1.0 (simulated 0.051 and 0.060)
        CFL_LIMIT = 0.5              # -, tvd2 monotonicity limit (CFL_x + CFL_z)
        VOLUME_TOL = 1e-10           # m^2, |sum(eta) dx| at the end
        MASS_TOL_ATOL = 1e-8         # kg/m
        MASS_TOL_RTOL = 1e-8         # -

        # WHY the mixing-length eddy viscosity: with the molecular nu = 1e-6 m^2/s alone the wind momentum stays near the
        # surface (vertical diffusion time dz^2/nu ~ 1e4 s) and there is no interfacial stress, so the lower layer only
        # sloshes with the internal seiche and psi_lower changes sign with its phase; no lower cell forms.
        # WHY no Rayleigh damping: it would damp the steady gyres as well.
        # WHY the mixing checks: the density has no diffusivity, so the interface thickens only through the advection
        # scheme's numerical diffusion in the sheared, seiching interface; the checks guard that this stays bounded
        # (mass conservation) and present, not a physical entrainment rate.
        for use_hydrostatic in (true, false)
            cfg = ModelConfig(
                length=10.0,
                depth=4.0,
                nx=80,
                nz=40,
                dt=DT,
                nt=round(Int, T_END / DT),
                use_hydrostatic=use_hydrostatic,
                momentum_advection_scheme="tvd2",
                transport_advection_scheme="tvd2",
                use_turbulence_model=true,
                turbulence_mixing_length=MIXING_LENGTH,
                max_turbulent_viscosity=MAX_EDDY_VISCOSITY,
                rayleigh_damping=0.0,
                apply_froude_limiter=false,
                wind_stress_x=0.0,
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
            @test all(iszero, model.state.eta)

            mass0 = constituent_mass(model.state.rho, model.dx, model.dz; eta=use_variable_top_layer(cfg) ? model.state.eta : nothing)
            varz0 = mean([var(@view model.state.rho[:, i]) for i in 1:cfg.nx])
            k_interface = Int(cld(cfg.nz, 2))
            drho_interface0 = mean(abs.(@view(model.state.rho[k_interface, :]) .- @view(model.state.rho[k_interface + 1, :])))

            mid = (cfg.nx ÷ 4 + 1):(3 * cfg.nx ÷ 4 + 1)   # u-faces of the middle half of the basin
            u_mean = zeros(cfg.nz)                        # m/s, time-averaged mid-basin profile
            n_samples = 0
            max_cfl = 0.0
            ramp_steps = round(Int, WIND_RAMP_DURATION / cfg.dt)
            average_start_step = round(Int, AVERAGE_START / cfg.dt)
            sample_stride = round(Int, SAMPLE_INTERVAL / cfg.dt)
            for n in 1:cfg.nt
                cfg.wind_stress_x = WIND_STRESS_X_MAX * min(n / ramp_steps, 1.0)
                step_forward!(model)
                max_cfl = max(max_cfl, cfl_numbers(model.state.u, model.state.w, cfg.dt, model.dx, model.dz)[3])
                if n > average_start_step && n % sample_stride == 0
                    for k in 1:cfg.nz
                        u_mean[k] += mean(@view model.state.u[k, mid])
                    end
                    n_samples += 1
                end
            end
            u_mean ./= n_samples
            psi = cumsum(u_mean) .* model.dz                 # m^2/s, psi at the upper face of each row
            psi_lower = maximum(@view psi[1:k_interface])
            psi_upper = minimum(@view psi[(k_interface + 1):(cfg.nz - 1)])

            drho_interface1 = mean(abs.(@view(model.state.rho[k_interface, :]) .- @view(model.state.rho[k_interface + 1, :])))
            varz1 = mean([var(@view model.state.rho[:, i]) for i in 1:cfg.nx])
            mass1 = constituent_mass(model.state.rho, model.dx, model.dz; eta=use_variable_top_layer(cfg) ? model.state.eta : nothing)

            @test psi_upper < PSI_UPPER_MAX      # clockwise wind cell in the upper layer
            @test psi_lower > PSI_LOWER_MIN      # counter-rotating cell in the lower layer
            @test u_mean[end] > U_TOP_MIN
            @test u_mean[1] > U_BED_MIN
            @test drho_interface1 < drho_interface0 - DRHO_MARGIN_MIN
            @test varz1 < varz0 - VARZ_MARGIN_MIN
            @test max_cfl < CFL_LIMIT
            @test abs(sum(model.state.eta) * model.dx) < VOLUME_TOL
            @test isapprox(mass1, mass0, atol=MASS_TOL_ATOL, rtol=MASS_TOL_RTOL)
        end
    end

    # Exact steady solutions for a constant viscosity (experiments/wind_shear/cases.jl, which documents the derivation).
    # Both tests start from the exact steady state (u, eta and, for two layers, the tilted interface) in a 40 m long
    # basin, run on and compare the time-averaged mean u over x = 15-25 m (more than 3.5 depths from the end walls,
    # where the end circulation has decayed) with the exact profile at the cell centres.
    function run_steady_wind_case(use_hydrostatic, nz, solution, two_layer; t_end, t_average, dt, nx, basin_length)
        cfg = wind_analytic_config(; use_hydrostatic=use_hydrostatic, t_end=t_end, nx=nx, nz=nz, dt=dt, length=basin_length,
            wind_stress_x=WS_WIND_STRESS_X, write_plots=false, write_animation=false, write_pressure_buoyancy_diagnostics=false)
        model = Model(cfg)
        initialize_wind_steady_state!(model, solution; two_layer=two_layer)
        mid = (3 * nx ÷ 8 + 1):(5 * nx ÷ 8 + 1)
        u_mean = zeros(nz)                      # m/s
        n_mean = 0
        for n in 1:cfg.nt
            step_forward!(model)
            if n * dt > t_end - t_average
                for k in 1:nz
                    u_mean[k] += mean(@view model.state.u[k, mid])
                end
                n_mean += 1
            end
        end
        u_mean ./= n_mean
        z = [(k - 0.5) * model.dz for k in 1:nz]
        x_centres = [(i - 0.5) * model.dx for i in 1:nx]
        half = (nx ÷ 4 + 1):(3 * nx ÷ 4)        # middle half: slopes
        return (error=maximum(abs, u_mean .- solution.u.(z)), u=u_mean, model=model,
            surface_slope=linear_slope(x_centres[half], model.state.eta[half]),
            interface_slope=two_layer ? linear_slope(x_centres[half], interface_heights(model.state.rho, model.dz)[half]) : NaN)
    end

    @testset "test_verify_steady_wind_profile_uniform_density_second_order" begin
        # u(z) = s/(2H) (z^2 - H^2/3), s = tau/(rho0 nu). The discrete steady state differs from it by the midpoint-rule
        # error of the zero-flux condition, a uniform offset s dz^2/(24 H): second order. Observed errors (hydrostatic)
        # 1.3e-4, 3.4e-5 and 9.2e-6 m/s for nz = 5, 10, 20; below about 3e-6 m/s (1e-4 of the surface velocity) the
        # error stops decreasing, independent of the run length (not investigated further).
        GRIDS = (5, 10, 20)          # nz
        ORDER_MIN = 1.8              # -
        SLOPE_RTOL = 0.01            # -, surface slope (observed within 0.07 %)
        T_END = 600.0                # s, several viscous decay times (H/pi)^2/nu = 160 s of the profile error
        T_AVERAGE = 200.0            # s, averaging window (about 15 barotropic seiche periods 2L/sqrt(gH))
        DT = 0.1                     # s, 2 nu dt (1/dx^2 + 1/dz^2) <= 0.13
        NX = 40                      # -
        BASIN_LENGTH = 40.0          # m
        solution = wind_steady_uniform(; tau=WS_WIND_STRESS_X, nu=WS_ANALYTIC_NU, depth=WS_DEPTH)
        for use_hydrostatic in (true, false)
            results = [run_steady_wind_case(use_hydrostatic, nz, solution, false; t_end=T_END, t_average=T_AVERAGE, dt=DT,
                nx=NX, basin_length=BASIN_LENGTH) for nz in GRIDS]
            errors = [r.error for r in results]
            @test all(log2.(errors[1:end-1] ./ errors[2:end]) .> ORDER_MIN)
            @test all(isapprox(r.surface_slope, solution.surface_slope; rtol=SLOPE_RTOL) for r in results)
        end
    end

    @testset "test_validate_steady_two_layer_wind_double_gyre" begin
        # Exact steady double gyre for one constant viscosity (wind_steady_two_layer): lower layer upwind at the interface
        # and downwind along the bed; surface and interface slopes from the hydrostatic pressure. Observed (nz = 40, both
        # modes): profile error 0.16-0.17 % of the surface velocity, surface slope within 0.09 %, interface slope within 0.5 %.
        NZ = 40                      # -
        NX = 40                      # -
        BASIN_LENGTH = 40.0          # m
        T_END = 2400.0               # s; at 600 s the internal seiche excited by the staircase interface is not yet damped
        T_AVERAGE = 600.0            # s, about one internal seiche period 2L/c = 570 s
        DT = 0.1                     # s
        PROFILE_RTOL = 0.02          # -, max |u - u_exact| relative to the exact surface velocity
        SURFACE_SLOPE_RTOL = 0.02    # -
        INTERFACE_SLOPE_RTOL = 0.03  # -, interface is a staircase of whole cells (tilt 0.3 m over the basin); observed 0.5 %
        solution = wind_steady_two_layer(; tau=WS_WIND_STRESS_X, nu=WS_ANALYTIC_NU, h_lower=WS_INTERFACE_HEIGHT_FRACTION * WS_DEPTH,
            depth=WS_DEPTH, drho=WS_RHO_BOTTOM - WS_RHO_TOP)
        u_surface = solution.u(WS_DEPTH)   # m/s
        for use_hydrostatic in (true, false)
            r = run_steady_wind_case(use_hydrostatic, NZ, solution, true; t_end=T_END, t_average=T_AVERAGE, dt=DT, nx=NX,
                basin_length=BASIN_LENGTH)
            @test r.error < PROFILE_RTOL * u_surface
            @test r.u[1] > 0.0                       # downwind along the bed: counter-rotating lower cell
            @test r.u[NZ ÷ 2] < 0.0                  # upwind just below the interface
            @test isapprox(r.surface_slope, solution.surface_slope; rtol=SURFACE_SLOPE_RTOL)
            @test isapprox(r.interface_slope, solution.interface_slope; rtol=INTERFACE_SLOPE_RTOL)
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
