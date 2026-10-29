include("test_common.jl")

# Theta scheme of the time integration (issue #34, docs/plan_issue_34_theta_scheme.md).
#
# Linear standing wave of one Fourier mode: on the C-grid the semi-discrete frequency is omega_h = (2c/dx) sin(k dx/2),
# c = sqrt(g H), and the theta scheme advances the mode by the factor
#   G = (1 + i mu (1 - theta)) / (1 - i mu theta),   mu = omega_h dt   (plan, section 3),
# so that a standing wave eta = A cos(k x) with u = 0 at t = 0 is eta^n = A Re(G^n) cos(k x) exactly (the two travelling
# modes have the factors G and conj(G)). The comparison with this discrete solution separates the time integration
# from the spatial error.

const THETA_WAVE_AMPLITUDE_RATIO = 2.5e-5   # -, A/H: nonlinear effects (advection, top-layer thickness) stay negligible
const THETA_WAVE_MATCH_MAX = 5e-4           # -, max |eta - A Re(G^n) cos(kx)| / A over the run (observed <= 1.1e-4)
const THETA_ORDER_TOL = 0.25                # -, allowed deviation of the observed temporal order from 1 or 2
const FROUDE_RECORD_RTOL = 1e-6             # -, uniform channel after one step: U and eta change only by round-off

theta_amplification(mu::Real, theta::Real) = (1 + im * mu * (1 - theta)) / (1 - im * mu * theta)

"""
Standing wave in a basin of length `L` (m) and depth `H` (m) with mode number `m` (closed: k = m pi / L; periodic:
k = 2 m pi / L). `kwargs` go to ModelConfig. Returns the model, k (1/m) and omega_h (1/s).
"""
function theta_wave_model(; theta::Real, dt::Real, nx::Int=20, nz::Int=4, L::Real=20.0, H::Real=4.0, periodic::Bool=false,
    bed::Real=0.0, kwargs...)
    extra = Dict{Symbol,Any}(:use_hydrostatic => true, :rayleigh_damping => 0.0, :apply_froude_limiter => false,
        :momentum_advection_scheme => "upwind1", :nu => 0.0)
    merge!(extra, Dict{Symbol,Any}(kwargs))
    periodic && (extra[:pressure_solver_method] = "sparse"; extra[:periodic_x] = true; extra[:variable_top_layer] = false)
    bed > 0.0 && (extra[:bed_level] = fill(float(bed), nx))
    cfg = ModelConfig(; nx=nx, nz=nz, length=L, depth=H, dt=dt, theta=theta, extra...)
    model = Model(cfg)
    k = periodic ? 2pi / L : pi / L
    x = [(i - 0.5) * model.dx for i in 1:nx]
    A = THETA_WAVE_AMPLITUDE_RATIO * H
    model.state.eta .= A .* cos.(k .* x)
    fill!(model.state.rho, 1000.0)
    refresh_density_metadata!(model)
    model.rho_min_init = 900.0
    model.rho_max_init = 1100.0
    c = sqrt(cfg.g * (H - bed))
    omega_h = 2c / model.dx * sin(k * model.dx / 2)
    return model, k, omega_h, A, x
end

# Largest deviation (relative to A) of eta from the discrete theta solution over `nsteps` steps.
function theta_wave_mismatch(model, k, omega_h, A, x, nsteps)
    G = theta_amplification(omega_h * model.config.dt, model.config.theta)
    worst = 0.0
    for n in 1:nsteps
        step_forward!(model)
        expected = A * real(G^n)
        worst = max(worst, maximum(abs.(model.state.eta .- expected .* cos.(k .* x))) / A)
    end
    return worst
end

# Largest deviation (relative to A) from the semi-discrete solution A cos(omega_h t) cos(k x) over one period.
function theta_wave_time_error(theta, steps_per_period; nx::Int=20, L::Real=20.0, H::Real=4.0)
    dx = L / nx
    period = 2pi / (2sqrt(9.81 * H) / dx * sin(pi / L * dx / 2))   # s, of the semi-discrete mode 1
    model, k, omega_h, A, x = theta_wave_model(; theta=theta, dt=period / steps_per_period, nx=nx, L=L, H=H)
    worst = 0.0
    for n in 1:steps_per_period
        step_forward!(model)
        worst = max(worst, abs(model.state.eta[1] - A * cos(omega_h * n * model.config.dt) * cos(k * x[1])) / A)
    end
    return worst
end

@testset "Theta scheme" begin
    @testset "test_verify_theta_range_is_checked" begin
        for bad in (0.4, 1.1)
            @test_throws ArgumentError Model(ModelConfig(nx=10, nz=4, theta=bad))
        end
        model = Model(ModelConfig(nx=10, nz=4, theta=0.6))
        model.config.theta = 0.7
        @test_throws ArgumentError step_forward!(model)   # the free-surface matrix was built for theta = 0.6

        model32 = Model(ModelConfig(nx=10, nz=4); T=Float32)
        @test_nowarn step_forward!(model32)
    end

    @testset "test_verify_theta_froude_bound_matches_linear_stability" begin
        # Mean flow U with explicit Lax-Wendroff advection of u and eta (Courant number C) and the theta free surface
        # (gravity-wave Courant number Cg = C / Fr): per Fourier mode phi the eigenvalues solve
        # (alpha - l)^2 + mu^2 (1 - theta + theta l)^2 = 0 (plan, section 3.3), i.e. l = (alpha -+ i mu (1 - theta)) /
        # (1 +- i mu theta). theta_froude_max(theta) must separate stable from unstable Froude numbers.
        STABLE_TOL = 1e-12    # -, round-off allowance of the spectral radius
        FR_MARGIN = 0.1       # -, relative distance from the bound of the probed Froude numbers
        function radius(theta, C, Fr)
            Cg = C / Fr
            worst = 0.0
            for phi in range(pi / 400, pi; length=400)
                alpha = 1 - im * C * sin(phi) - C^2 * (1 - cos(phi))
                mu = 2Cg * sin(phi / 2)
                for s in (1, -1)
                    worst = max(worst, abs((alpha - s * im * mu * (1 - theta)) / (1 + s * im * mu * theta)))
                end
            end
            return worst
        end
        @test theta_froude_max(0.5) == 0.0
        @test theta_froude_max(0.7) ≈ 2 / 3
        @test theta_froude_max(0.75) ≈ 1.0
        @test theta_froude_max(1.0) == Inf
        for theta in (0.55, 0.6, 0.7, 0.8), C in (0.1, 0.5, 0.9)
            fr_max = theta_froude_max(theta)
            @test radius(theta, C, (1 - FR_MARGIN) * fr_max) <= 1 + STABLE_TOL
            @test radius(theta, C, (1 + FR_MARGIN) * fr_max) > 1 + STABLE_TOL
        end
    end

    @testset "test_verify_theta_nonhydro_projection_keeps_column_transport" begin
        # The projection must not change the column transports (they are fixed by the free-surface solve), so the depth
        # sum of q stays uniform in x. Its top velocity uses the column divergence of u^n, not the stored w: here the
        # initial w is zero while u^0 is not divergence-free per column. With the stored w, the first step put a
        # depth-uniform q with a slope into the run, and the incremental form kept it (steady wind set-up: surface
        # slope -3 times the exact one).
        QSUM_REL_TOL = 1e-9   # -, variation of the depth sum of q relative to max |q| (round-off of the Poisson solve)
        for theta in (0.7, 1.0)
            cfg = ModelConfig(nx=20, nz=8, length=20.0, depth=4.0, dt=0.05, theta=theta, use_hydrostatic=false,
                variable_top_layer=false, pressure_solver_method="sparse", rayleigh_damping=0.0,
                apply_froude_limiter=false, momentum_advection_scheme="upwind1", nu=0.0)
            model = Model(cfg)
            fill!(model.state.rho, 1000.0)
            refresh_density_metadata!(model)
            for i in 2:cfg.nx, k in 1:cfg.nz
                model.state.u[k, i] = 0.05 * sin(pi * (i - 1) / cfg.nx) * (1 + 0.5 * cos(pi * (k - 0.5) / cfg.nz))
            end
            fill!(model.state.w, 0.0)
            for _ in 1:3
                step_forward!(model)
                q = model.ws.p_corr
                qsum = vec(sum(q; dims=1))
                @test maximum(qsum) - minimum(qsum) < QSUM_REL_TOL * cfg.nz * maximum(abs, q)
            end
        end
    end

    # dt = 0.3 s gives mu ~ 0.3 (about 20 steps per period): the damping of theta = 1 (|G| ~ 0.96 per step) and the
    # phase errors are large, so the comparison distinguishes the theta values clearly.
    # WHY the non-hydrostatic Neumann mode here: with the legacy Neumann surface condition a barotropic standing wave
    # has no non-hydrostatic correction (docs section 15), so it follows the same discrete solution.
    for (name, opts) in (("fixed_grid", (variable_top_layer=false,)), ("variable_top_layer", (variable_top_layer=true,)),
        ("masked_bed", (variable_top_layer=false, bed=1.0)), ("periodic", (periodic=true,)),
        ("nonhydro_neumann", (use_hydrostatic=false,)), ("wetting_drying_1d", (wetting_drying=true, nz=1)),
        ("wetting_drying_2dv", (wetting_drying=true,)),
        # implicit vertical terms (issue #34): without friction the column response is 1 for a depth-uniform wave
        ("fixed_grid_implicit_vertical", (variable_top_layer=false, implicit_vertical_terms=true)),
        ("variable_top_layer_implicit_vertical", (variable_top_layer=true, implicit_vertical_terms=true)),
        ("masked_bed_implicit_vertical", (variable_top_layer=false, bed=1.0, implicit_vertical_terms=true)),
        ("periodic_implicit_vertical", (periodic=true, implicit_vertical_terms=true)),
        ("wetting_drying_2dv_implicit_vertical", (wetting_drying=true, implicit_vertical_terms=true)))
        @testset "test_verify_theta_standing_wave_amplification_$(name)" begin
            for theta in (0.5, 0.55, 0.75, 1.0)
                model, k, omega_h, A, x = theta_wave_model(; theta=theta, dt=0.3, opts...)
                nsteps = round(Int, 2 * 2pi / (omega_h * model.config.dt))   # two periods
                @test theta_wave_mismatch(model, k, omega_h, A, x, nsteps) < THETA_WAVE_MATCH_MAX
            end
        end
    end

    @testset "test_verify_theta_temporal_order" begin
        # Error against the semi-discrete solution: second order at theta = 1/2, first order otherwise.
        # WHY finer steps for theta = 1: its error is 60 % of the amplitude at 20 steps per period and only reaches the
        # asymptotic order beyond about 80 steps per period (observed 0.67, 0.83, 0.91 from 20 to 160 steps).
        for (theta, order, steps) in ((0.5, 2.0, (20, 40, 80)), (0.55, 1.0, (20, 40, 80)), (1.0, 1.0, (80, 160, 320)))
            errs = [theta_wave_time_error(theta, n) for n in steps]
            observed = log2.(errs[1:end-1] ./ errs[2:end])
            @test all(abs.(observed .- order) .< THETA_ORDER_TOL)
        end
    end

    # Dispersive standing wave (k H = pi/2) of the fixed grid (Dirichlet surface condition) and of wetting and drying
    # (q = 0 in the surface cell): no closed discrete solution, so the error is measured against a run with a 16 times
    # smaller step on the same grid (theta = 1/2, error 256 times smaller for the second-order case). Second order at
    # theta = 1/2, first order at theta = 1.
    for (name, opts) in (("fixed_grid_dirichlet", (nonhydro_surface_dirichlet=true, pressure_solver_method="sparse")),
        ("wetting_drying", (wetting_drying=true,)))
    @testset "test_verify_theta_nonhydro_dispersive_temporal_order_$(name)" begin
        L, H, nx, nz = 8.0, 4.0, 16, 8
        T_shallow = 2L / sqrt(9.81 * H)   # s, the dispersive period is longer
        dispersive(theta, dt) = theta_wave_model(; theta=theta, dt=dt, nx=nx, nz=nz, L=L, H=H, use_hydrostatic=false, opts...)
        function eta_after(theta, dt, t_end)
            model, = dispersive(theta, dt)
            for _ in 1:round(Int, t_end / dt)
                step_forward!(model)
            end
            return copy(model.state.eta)
        end
        t_end = T_shallow
        for (theta, order, steps) in ((0.5, 2.0, (20, 40, 80)), (1.0, 1.0, (80, 160, 320)))
            reference = eta_after(0.5, t_end / (16 * steps[end]), t_end)
            A = THETA_WAVE_AMPLITUDE_RATIO * H
            errs = [maximum(abs.(eta_after(theta, t_end / n, t_end) .- reference)) / A for n in steps]
            observed = log2.(errs[1:end-1] ./ errs[2:end])
            @test all(abs.(observed .- order) .< THETA_ORDER_TOL)
        end
    end
    end

    # Wetting and drying with theta < 1: beach rising out of the water (0 ... 2.5 m over 20 m, still level 1.5 m).
    wd_beach(nx) = [2.5 * (i - 0.5) / nx for i in 1:nx]
    function wd_theta_model(theta; nx=40, nz=8, dt=0.1, use_hydrostatic=true, bed=wd_beach(nx))
        cfg = ModelConfig(nx=nx, nz=nz, length=20.0, depth=3.0, dt=dt, theta=theta, wetting_drying=true, bed_level=bed,
            use_hydrostatic=use_hydrostatic, rayleigh_damping=0.0, apply_froude_limiter=false,
            momentum_advection_scheme="upwind1", nu=0.0)
        return Model(cfg)
    end

    @testset "test_verify_theta_wd_lake_at_rest" begin
        for theta in (0.5, 0.55), hydro in (true, false)
            model = wd_theta_model(theta; use_hydrostatic=hydro)
            initialize_water_level!(model, 1.5; rho=1000.0)
            eta0 = copy(model.state.eta)
            for _ in 1:300
                step_forward!(model)
            end
            @test maximum(abs, model.state.u) < TOL_IDENTITY
            @test maximum(abs.(model.state.eta .- eta0)) < TOL_IDENTITY
        end
    end

    @testset "test_verify_theta_wd_conservation_and_positivity" begin
        # Sloshing over the beach with two water masses (theta = 0.55), and a dam break onto the dry beach at a large
        # step (theta = 1/2, gravity-wave Courant number about 6): depth >= 0 in every step, water volume and constituent
        # mass conserved, density within its initial bounds.
        for (theta, dt, level) in ((0.55, 0.05, x -> 1.5 + 0.3 * cos(pi * x / 20.0)), (0.5, 0.4, x -> x < 8.0 ? 1.8 : 0.0)),
            hydro in (true, false)
            model = wd_theta_model(theta; dt=dt, use_hydrostatic=hydro)
            xc = [(i - 0.5) * 0.5 for i in 1:40]
            rho = [i <= 20 ? 1001.0 : 1000.0 for _ in 1:8, i in 1:40]
            initialize_water_level!(model, level.(xc); rho=rho)
            volume0 = sum(water_depth(model))
            mass0 = SweTransport2DV._constituent_mass(model)
            for _ in 1:200
                step_forward!(model)
                @test minimum(water_depth(model)) >= 0.0
            end
            @test abs(sum(water_depth(model)) - volume0) / volume0 < TOL_MODEL_IDENTITY
            @test abs(SweTransport2DV._constituent_mass(model) - mass0) / mass0 < TOL_MODEL_IDENTITY
            @test 1000.0 - TOL_MODEL_IDENTITY <= minimum(model.state.rho) && maximum(model.state.rho) <= 1001.0 + TOL_MODEL_IDENTITY
        end
    end

    @testset "test_verify_theta_nonhydro_volume_and_kinematic_closure" begin
        # Nonlinear wave (A/H = 0.05) with theta = 0.55 in both surface conditions of the non-hydrostatic mode: volume
        # conserved; with the Neumann condition the surface moves with the theta velocities (kinematic condition exact).
        # WHY not for the Dirichlet condition: its projection leaves the surface face of w uncorrected, so the column
        # transport of u^{n+1} differs from the solved surface by the (unapplied) top-face correction; this holds for the
        # former theta = 1 scheme as well (observed 3e-3 m/s there, 1e-3 m/s at theta = 0.55).
        for dirichlet in (false, true)
            model, = theta_wave_model(; theta=0.55, dt=0.05, nx=32, nz=8, L=16.0, use_hydrostatic=false,
                nonhydro_surface_dirichlet=dirichlet, pressure_solver_method="sparse")
            model.state.eta .*= 2000.0
            volume0 = sum(model.state.eta)
            for _ in 1:200
                step_forward!(model)
                dirichlet || @test model.last_kinematic_mismatch < TOL_IDENTITY
            end
            @test abs(sum(model.state.eta) - volume0) < TOL_MODEL_IDENTITY
            @test maximum(abs.(model.state.rho .- 1000.0)) < TOL_MODEL_IDENTITY
        end
    end

    @testset "test_verify_theta_volume_mass_and_kinematic_closure" begin
        # Nonlinear sloshing of a two-layer stratification (A/H = 0.05) with theta = 0.55: the water volume and the
        # constituent mass are conserved and the surface moves with the theta velocity (kinematic condition exact),
        # on the variable top layer and on the fixed grid.
        for variable_top in (true, false)
            cfg = ModelConfig(nx=40, nz=10, length=20.0, depth=4.0, dt=0.05, theta=0.55, use_hydrostatic=true,
                variable_top_layer=variable_top, rayleigh_damping=0.0, apply_froude_limiter=false)
            model = Model(cfg)
            x = [(i - 0.5) * model.dx for i in 1:cfg.nx]
            model.state.eta .= 0.2 .* cos.(pi .* x ./ cfg.length)
            for i in 1:cfg.nx, kk in 1:cfg.nz
                model.state.rho[kk, i] = kk <= cfg.nz ÷ 2 ? 1001.0 : 1000.0
            end
            refresh_density_metadata!(model)
            volume0 = sum(model.state.eta)
            mass0 = SweTransport2DV._constituent_mass(model)
            for _ in 1:200
                step_forward!(model)
                @test model.last_kinematic_mismatch < TOL_IDENTITY
            end
            @test abs(sum(model.state.eta) - volume0) < TOL_MODEL_IDENTITY
            @test abs(SweTransport2DV._constituent_mass(model) - mass0) / mass0 < TOL_MODEL_IDENTITY
            @test minimum(model.state.rho) >= 1000.0 - TOL_MODEL_IDENTITY
            @test maximum(model.state.rho) <= 1001.0 + TOL_MODEL_IDENTITY
        end
    end

    @testset "test_verify_theta_uniform_density_stays_uniform" begin
        # A uniform constituent stays uniform with the theta transport velocities (consistency of transport and volume).
        for variable_top in (true, false)
            model, k, omega_h, A, x = theta_wave_model(; theta=0.55, dt=0.05, variable_top_layer=variable_top)
            model.state.eta .*= 2000.0   # A/H = 0.05: nonlinear
            for _ in 1:200
                step_forward!(model)
            end
            @test maximum(abs.(model.state.rho .- 1000.0)) < TOL_MODEL_IDENTITY
        end
    end

    # Surface Froude number and the warning above the theta bound (issue #39).
    function froude_channel(; U, theta, H=4.0, kwargs...)
        cfg = ModelConfig(; nx=8, nz=4, length=80.0, depth=H, dt=1.0, theta=theta, periodic_x=true,
            pressure_solver_method="sparse", variable_top_layer=false, rayleigh_damping=0.0, apply_froude_limiter=false,
            nu=0.0, write_plots=false, kwargs...)
        model = Model(cfg)
        fill!(model.state.rho, 1000.0)
        refresh_density_metadata!(model)
        fill!(model.state.u, U)
        return model
    end

    @testset "test_verify_theta_for_froude_inverts_bound" begin
        for theta in (0.5, 0.55, 0.7, 0.75, 0.9)
            @test theta_for_froude(theta_froude_max(theta)) ≈ theta rtol = TOL_IDENTITY
        end
        @test theta_for_froude(0.0) == 0.5
        @test theta_for_froude(1.0) ≈ 0.75
        @test theta_for_froude(Inf) == 1.0
        @test_throws ArgumentError theta_for_froude(-0.1)
    end

    @testset "test_verify_surface_froude_number_of_uniform_flow" begin
        # periodic channel: Fr = U / sqrt(g H) on every face
        U, H, g = 2.0, 4.0, 9.81
        fr, face = surface_froude_number(froude_channel(; U=U, theta=1.0, H=H))
        @test fr ≈ U / sqrt(g * H) rtol = TOL_IDENTITY
        @test 1 <= face <= 9
        # variable top layer: the depth of the face includes eta (and the top layer dz + eta)
        model = Model(ModelConfig(nx=4, nz=4, length=40.0, depth=H, dt=1.0, rayleigh_damping=0.0, write_plots=false))
        model.state.eta .= 0.5
        fill!(model.state.u, U)
        @test surface_froude_number(model)[1] ≈ U / sqrt(g * (H + 0.5)) rtol = TOL_IDENTITY
        # staircase bed: the shallowest faces are the fastest at the same U
        model = Model(ModelConfig(nx=4, nz=4, length=40.0, depth=H, dt=1.0, bed_level=[0.0, 2.0, 2.0, 0.0], write_plots=false))
        fill!(model.state.u, U)
        fr, face = surface_froude_number(model)
        @test fr ≈ U / sqrt(g * (H - 2.0)) rtol = TOL_IDENTITY
        @test face in 2:4   # the faces next to and over the raised bed (bed level of a face: the higher neighbour)
        # wetting and drying: faces shallower than SURFACE_FROUDE_MIN_DEPTH (a shoreline film) are left out
        model = Model(ModelConfig(nx=4, nz=2, length=40.0, depth=2.0, dt=1.0, wetting_drying=true, bed_level=zeros(4),
            write_plots=false))
        wd = model.wetdry
        fill!(wd.hf, 0.0)
        wd.hf[:, 2] .= 1.0                                              # 2 m deep
        wd.hf[1, 4] = 0.5 * SweTransport2DV.SURFACE_FROUDE_MIN_DEPTH   # film
        model.state.u[:, 2] .= 1.0
        model.state.u[:, 4] .= 50.0
        fr, face = surface_froude_number(model)
        @test fr ≈ 1.0 / sqrt(g * 2.0) rtol = TOL_IDENTITY
        @test face == 2
    end

    @testset "test_verify_froude_bound_warning" begin
        # theta = 0.7 allows Fr <= 2/3: Fr = 0.8 warns once (with the theta it needs, 0.72), Fr = 0.5 does not; theta = 0.75
        # allows Fr = 0.8, theta = 1 any Fr. The largest Froude number of the run is recorded in the diagnostics.
        g, H = 9.81, 4.0
        model = froude_channel(; U=0.8 * sqrt(g * H), theta=0.7)
        @test_logs (:warn, r"exceeds theta_froude_max.*theta_for_froude\(Fr\) = 0\.72") step_forward!(model)
        @test_logs step_forward!(model)   # once per model
        @test model.froude_warning_issued
        @test compute_diagnostics(model)["surface_froude_max"] ≈ 0.8 rtol = FROUDE_RECORD_RTOL
        # below the bound, or above it by less than SURFACE_FROUDE_WARNING_MARGIN (theta = 1/2 has the bound 0: small
        # wave velocities must not warn)
        margin = SweTransport2DV.SURFACE_FROUDE_WARNING_MARGIN
        for (Fr, theta) in ((0.5, 0.7), (0.8, 0.75), (2.0, 1.0), (2 / 3 + 0.5 * margin, 0.7), (1e-5, 0.5))
            model = froude_channel(; U=Fr * sqrt(g * H), theta=theta)
            @test_logs step_forward!(model)
            @test !model.froude_warning_issued
            @test model.max_surface_froude ≈ Fr rtol = FROUDE_RECORD_RTOL
        end
    end
end
