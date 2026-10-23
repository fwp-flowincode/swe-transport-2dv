# Non-hydrostatic branch and free surface (docs sections 14 and 15).
# Run with the full suite (runtests.jl) or on its own: julia --project=. test/test_nonhydrostatic.jl

include(joinpath(@__DIR__, "test_common.jl"))

@testset "Non-hydrostatic" begin
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
        @test all(isapprox.(model.grad_p_z_reference[2:end-1, :], -rho0 * g; rtol=TOL_IDENTITY))
        @test all(iszero, model.grad_p_z_reference[[1, end], :])   # grad_z_c! leaves the two boundary faces at 0

        # A non-uniform (two-layer) reference profile: x-independent (built from column 1, repeated across x) but
        # z-dependent (not the same value in every layer).
        cfg_layered = ModelConfig(nx=10, nz=8, length=L, depth=H, use_hydrostatic=false, pressure_solver_method="sparse", nonhydro_surface_dirichlet=true)
        model2 = Model(cfg_layered)
        initialize_internal_seiche!(model2; rho_top=999.0, rho_bottom=1001.0, interface_height_fraction=0.5, interface_tilt_amplitude_fraction=0.0)
        gref = model2.grad_p_z_reference
        @test all(isapprox.(gref, gref[:, 1]; rtol=TOL_IDENTITY))   # x-independent (matches column 1)
        @test maximum(gref[2:end-1, 1]) - minimum(gref[2:end-1, 1]) > TOL_IDENTITY   # genuinely z-dependent

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
end
