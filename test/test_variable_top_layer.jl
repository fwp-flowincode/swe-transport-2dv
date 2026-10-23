# Variable-thickness top layer, hydrostatic mode (docs section 13).
# Run with the full suite (runtests.jl) or on its own: julia --project=. test/test_variable_top_layer.jl

include(joinpath(@__DIR__, "test_common.jl"))

@testset "Variable top layer" begin
    # ---- Variable-thickness top layer (hydrostatic mode) ------------------------------------------------------
    # Named tolerances (no hardcoded tolerances inside the tests).
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
        rho_uniform = fill(1000.0, nz, nx)
        for scheme in ("upwind1", "tvd2")
            advect_density_top_layer!(rho_new, rho, st.u, st.w, cfg.dt, dx, dz, ws.density_ws, ws.h_top_face, ws.eta_old, st.eta; scheme=scheme)
            mass_old = constituent_mass(rho, dx, dz; eta=ws.eta_old)
            mass_new = constituent_mass(rho_new, dx, dz; eta=st.eta)
            @test abs(mass_new - mass_old) < TOL_IDENTITY * abs(mass_old)

            advect_density_top_layer!(rho_new, rho_uniform, st.u, st.w, cfg.dt, dx, dz, ws.density_ws, ws.h_top_face, ws.eta_old, st.eta; scheme=scheme)
            @test maximum(abs, rho_new .- 1000.0) < TOL_IDENTITY
        end
    end

    @testset "test_verify_variable_top_layer_uniform_density_stays_uniform" begin
        function run_uniform_density(variable_top::Bool)
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
            return model
        end

        # Both treatments keep a uniform density uniform: the variable top layer through its thickness-integrated budget,
        # the fixed grid through the top-row advective form (_density_top_row_advective_form!, docs section 4).
        model_variable = run_uniform_density(true)
        model_fixed = run_uniform_density(false)
        @test maximum(abs, model_variable.state.rho .- 1000.0) < TOL_MODEL_IDENTITY
        @test maximum(abs, model_fixed.state.rho .- 1000.0) < TOL_MODEL_IDENTITY

        # contrast: the plain flux form with zero flux through a top face that moves with d(eta)/dt is a spurious source
        st = model_fixed.state
        rho_uniform = fill(1000.0, size(st.rho))
        flux_form = advect_density(rho_uniform, st.u, st.w, model_fixed.config.dt, model_fixed.dx, model_fixed.dz; top_row_advective_form=false)
        @test maximum(abs, flux_form .- 1000.0) > TOL_LEGACY_VIOLATION
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
