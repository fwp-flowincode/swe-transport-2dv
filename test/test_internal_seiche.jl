# Internal seiche in a closed basin against the two-layer relation, hydrostatic and non-hydrostatic, on three grids
# (issue #23), and the density extrapolation of the baroclinic pressure that keeps hydrostatic internal waves stable for
# theta < 1 (baroclinic_extrapolation_weight).
# Run with the full suite (runtests.jl) or on its own: julia --project=. test/test_internal_seiche.jl

include(joinpath(@__DIR__, "test_common.jl"))

# ---- Case (issue #23) -------------------------------------------------------------------------------------------------
const IS_LENGTH = 8.0              # m, basin length = one wavelength (mode 2 of the closed basin)
const IS_DEPTH = 4.0               # m
const IS_RHO_TOP = 998.5           # kg/m^3, upper layer
const IS_RHO_BOTTOM = 1001.5       # kg/m^3, lower layer
const IS_RHO0 = 1000.0             # kg/m^3, reference density of the model (ModelConfig default)
const IS_G = 9.81                  # m/s^2, ModelConfig default
const IS_INTERFACE_FRACTION = 0.5  # -, interface at mid-depth (h_upper = h_lower = 2 m)
const IS_AMPLITUDE = 0.12          # m, interface displacement (k a = 0.09: linear)
const IS_DT = 0.05                 # s, the same on every grid (omega dt <= 0.007: the time error is small)
const IS_PERIODS = 3               # -, simulated periods
const IS_GRIDS = ((40, 40), (80, 80), (160, 160))   # (nx, nz): dx = dz = 0.2, 0.1, 0.05 m

# ---- Bounds (measured values in the comments; docs/TESTS_TECHNICAL_DOCUMENTATION.md, section 34) ----------------------
const IS_PERIOD_HYDRO_MAX = 0.015      # -, relative period error on the finest grid (observed +1.00 %; 3.06, 1.74 % coarser)
const IS_PERIOD_NONHYDRO_MAX = 0.025   # -, same, non-hydrostatic (observed +1.86 %; 5.09, 3.06 % coarser)
const IS_CONVERGENCE_RATIO = 0.7       # -, the period error must drop at least by this factor per refinement (0.57-0.61)
const IS_DAMPING_MIN = 0.98            # -, interface amplitude kept per period (observed 0.981-0.991)
const IS_SURFACE_RATIO_HYDRO_RTOL = 0.01      # -, surface / interface displacement against the relation (observed 0.1 %)
const IS_SURFACE_RATIO_DIRICHLET_RTOL = 0.15  # -, same, non-hydrostatic with the Dirichlet top condition, 80 x 80 (8 %)
const IS_SURFACE_RATIO_NEUMANN_RTOL = 0.01    # -, the Neumann top condition gives the hydrostatic ratio (docs section 15)
const IS_VELOCITY_MAX = 0.02           # m/s, largest |u| (observed 0.011; before issue #23 the hydrostatic run on
                                       # 160 x 160 grew to 0.06 m/s and stopped at t = 58 s)
const IS_VOLUME_TOL = 1e-12            # m^2, water volume change (round-off)
const IS_MASS_RTOL = 1e-12             # -, relative salt mass change (round-off)
const IS_EXTRAPOLATION_S = (0.05, 0.1, 0.3, 0.5, 0.8)   # -, omega dt of the oscillator check (all below s_max)
const IS_REFERENCE_PERIODS = (61.0, 46.6)               # s, the periods of the issue (rigid lid, non-/hydrostatic)
const IS_REFERENCE_PERIOD_TOL = 0.05                    # s, the rigid-lid formulas give them to this
const IS_RESIDUAL_TOL = 1e-10          # -, relative residual of the linear two-layer system

is_k() = 2pi / IS_LENGTH
is_h() = (IS_DEPTH * (1 - IS_INTERFACE_FRACTION), IS_DEPTH * IS_INTERFACE_FRACTION)   # m, (upper, lower)

function is_reference(; hydrostatic)
    hu, hl = is_h()
    return two_layer_internal_wave(; k=is_k(), h_upper=hu, h_lower=hl, rho_upper=IS_RHO_TOP, rho_lower=IS_RHO_BOTTOM,
        rho0=IS_RHO0, g=IS_G, hydrostatic=hydrostatic)
end

"""
    is_run(nx, nz; hydrostatic, kwargs...)

IS_PERIODS periods of the seiche (tvd2, no damping or limiter, cell-averaged initial interface). The interface height of
a column is the depth of the heavy fluid, sum dz (rho - rho_top) / (rho_bottom - rho_top) (m; exact under conservation,
insensitive to the smearing of the interface), projected on cos(k x): the mode amplitude a(t) (m). Returns the period
from the zero crossings of a (s), the amplitude kept per period (-), the surface / interface ratio (least squares of the
cos(k x) projections of eta and of the interface), the largest |u| (m/s), the volume change (m^2), the relative salt
mass change (-) and the model.
"""
function is_run(nx, nz; hydrostatic, kwargs...)
    T_ref = 2pi / is_reference(; hydrostatic=hydrostatic)[1]   # s
    cfg = ModelConfig(; length=IS_LENGTH, depth=IS_DEPTH, nx=nx, nz=nz, dt=IS_DT, use_hydrostatic=hydrostatic,
        momentum_advection_scheme="tvd2", transport_advection_scheme="tvd2", rayleigh_damping=0.0,
        apply_froude_limiter=false, nu=0.0, write_plots=false, write_animation=false,
        write_pressure_buoyancy_diagnostics=false, kwargs...)
    model = Model(cfg)
    initialize_internal_seiche!(model; rho_top=IS_RHO_TOP, rho_bottom=IS_RHO_BOTTOM,
        interface_height_fraction=IS_INTERFACE_FRACTION, interface_tilt_amplitude_fraction=IS_AMPLITUDE / IS_DEPTH,
        cell_average=true)
    dx, dz = model.dx, model.dz
    c = [cos(is_k() * (i - 0.5) * dx) for i in 1:nx]
    project(v) = 2 / IS_LENGTH * dx * sum(v .* c)
    interface() = project(vec(sum((model.state.rho .- IS_RHO_TOP) ./ (IS_RHO_BOTTOM - IS_RHO_TOP); dims=1)) .* dz .-
        IS_DEPTH * IS_INTERFACE_FRACTION)
    volume0 = water_volume(cfg.depth, model.state.eta, dx)
    # WHY the model's own measure: the top layer is dz + eta only with the variable top layer (hydrostatic default)
    mass0 = SweTransport2DV._constituent_mass(model)
    a, b = [interface()], [project(model.state.eta)]
    umax = 0.0
    for _ in 1:round(Int, IS_PERIODS * T_ref / IS_DT)
        step_forward!(model)
        push!(a, interface())
        push!(b, project(model.state.eta))
        umax = max(umax, maximum(abs, model.state.u))
    end
    crossings = Float64[]
    for j in 2:length(a)
        a[j - 1] * a[j] < 0 && push!(crossings, (j - 2 + a[j - 1] / (a[j - 1] - a[j])) * IS_DT)
    end
    period = 2 * (crossings[end] - crossings[1]) / (length(crossings) - 1)
    peaks = [maximum(abs, a[(floor(Int, crossings[j] / IS_DT) + 1):(ceil(Int, crossings[j + 1] / IS_DT) + 1)])
             for j in 1:(length(crossings) - 1)]
    kept = (peaks[end] / peaks[1])^(2 / (length(peaks) - 1))
    return (; period, kept, surface_ratio=sum(a .* b) / sum(a .* a), umax,
        dvolume=abs(water_volume(cfg.depth, model.state.eta, dx) - volume0),
        dmass=abs(SweTransport2DV._constituent_mass(model) - mass0) / mass0, model)
end

# Largest |lambda| of the oscillator model of the baroclinic coupling (baroclinic_extrapolation_weight):
# lambda (lambda - 1)^2 + s^2 ((1 + beta) lambda - beta) (theta lambda + 1 - theta) = 0.
function is_oscillator_amplification(theta, beta, s)
    b = -2 + s^2 * (1 + beta) * theta
    c = 1 + s^2 * ((1 + beta) * (1 - theta) - beta * theta)
    d = -s^2 * beta * (1 - theta)
    return maximum(abs, eigvals([-b -c -d; 1.0 0.0 0.0; 0.0 1.0 0.0]))
end

@testset "Internal seiche (issue #23)" begin
    @testset "test_verify_two_layer_internal_wave_relation" begin
        hu, hl = is_h()
        gp = IS_G * (IS_RHO_BOTTOM - IS_RHO_TOP) / IS_RHO0
        for (k, h_u, h_l) in ((is_k(), hu, hl), (0.3, 1.0, 3.0), (2.0, 3.0, 0.5))
            om, ratio = two_layer_internal_wave(; k=k, h_upper=h_u, h_lower=h_l, rho_upper=IS_RHO_TOP,
                rho_lower=IS_RHO_BOTTOM)
            # Residuals of the linear system for zeta = 1 (upper layer phi = A cosh kz + B sinh kz, z = 0 at the surface;
            # lower layer phi = D cosh k(z + H)): kinematic and dynamic surface conditions give B = -i om eta / k,
            # A = -i g eta / om; the interface conditions (kinematic in both layers, pressure continuity) must hold.
            eta = ratio
            A, B = -im * IS_G * eta / om, -im * om * eta / k
            D = -im * om / (k * sinh(k * h_l))
            upper_w = k * (-A * sinh(k * h_u) + B * cosh(k * h_u))   # phi_z of the upper layer at the interface
            pressure = -im * om * (D * cosh(k * h_l) - (A * cosh(k * h_u) - B * sinh(k * h_u))) + gp   # (5): = 0
            @test abs(upper_w + im * om) <= IS_RESIDUAL_TOL * om
            @test abs(pressure) <= IS_RESIDUAL_TOL * gp
            # rigid-lid limit (Gill 1982) to O(g'/g)
            @test om^2 ≈ gp * k / (coth(k * h_u) + coth(k * h_l)) rtol = 2gp / IS_G
            @test ratio < 0
        end
        # hydrostatic: the quartic of two-layer shallow water, its long-wave limit, and the non-hydrostatic relation
        # for k H -> 0
        omh, ratioh = is_reference(; hydrostatic=true)
        c2 = (omh / is_k())^2
        @test c2^2 - (IS_G * IS_DEPTH + gp * hl) * c2 + IS_G * gp * hu * hl ≈ 0 atol = 1e-12 * (IS_G * IS_DEPTH)^2
        # the hydrostatic system itself: lower-layer momentum and mass give the ratio, upper-layer mass must hold
        @test ratioh ≈ c2 / (c2 - IS_G * hu) rtol = 1e-12
        @test c2 ≈ gp * hu * hl / IS_DEPTH rtol = 2gp / IS_G
        @test ratioh ≈ -gp * hl / (IS_G * IS_DEPTH) rtol = 2gp / IS_G
        k_long = 1e-3 / IS_DEPTH
        om_nh = two_layer_internal_wave(; k=k_long, h_upper=hu, h_lower=hl, rho_upper=IS_RHO_TOP, rho_lower=IS_RHO_BOTTOM)[1]
        om_h = two_layer_internal_wave(; k=k_long, h_upper=hu, h_lower=hl, rho_upper=IS_RHO_TOP, rho_lower=IS_RHO_BOTTOM,
            hydrostatic=true)[1]
        @test om_nh ≈ om_h rtol = 1e-6
        # the periods quoted in issue #23 are the rigid-lid ones (61.0 s, 46.6 s); the free surface lengthens them by
        # O(g'/g) (to 61.04 s and 46.65 s)
        T_rigid = (2pi / sqrt(gp * is_k() / (coth(is_k() * hu) + coth(is_k() * hl))), IS_LENGTH / sqrt(gp * hu * hl / IS_DEPTH))
        @test T_rigid[1] ≈ IS_REFERENCE_PERIODS[1] atol = IS_REFERENCE_PERIOD_TOL
        @test T_rigid[2] ≈ IS_REFERENCE_PERIODS[2] atol = IS_REFERENCE_PERIOD_TOL
        @test 2pi / is_reference(; hydrostatic=false)[1] ≈ T_rigid[1] rtol = gp / IS_G
        @test 2pi / omh ≈ T_rigid[2] rtol = gp / IS_G
        @test_throws ArgumentError two_layer_internal_wave(; k=1.0, h_upper=1.0, h_lower=1.0, rho_upper=1001.0, rho_lower=1000.0)
        @test_throws ArgumentError two_layer_internal_wave(; k=0.0, h_upper=1.0, h_lower=1.0, rho_upper=1000.0, rho_lower=1001.0)
    end

    @testset "test_verify_baroclinic_extrapolation_oscillator" begin
        # without the extrapolation the coupling grows for every theta < 1; with beta = 2 (1 - theta) it does not up to
        # the s_max of the docstring (0.82 at theta = 1/2); theta = 1 needs none
        for theta in (0.5, 0.55, 0.7, 0.9)
            beta = baroclinic_extrapolation_weight(theta)
            @test beta ≈ 2 * (1 - theta)
            for s in IS_EXTRAPOLATION_S
                @test is_oscillator_amplification(theta, 0.0, s) > 1
                @test is_oscillator_amplification(theta, beta, s) <= 1 + 1e-12
            end
        end
        @test baroclinic_extrapolation_weight(1.0) == 0
        @test all(is_oscillator_amplification(1.0, 0.0, s) <= 1 + 1e-12 for s in IS_EXTRAPOLATION_S)
    end

    @testset "test_verify_baroclinic_extrapolation_history" begin
        cfg(theta) = ModelConfig(length=IS_LENGTH, depth=IS_DEPTH, nx=20, nz=10, dt=IS_DT, theta=theta,
            rayleigh_damping=0.0, apply_froude_limiter=false, write_plots=false, write_animation=false,
            write_pressure_buoyancy_diagnostics=false)
        init!(m) = initialize_internal_seiche!(m; rho_top=IS_RHO_TOP, rho_bottom=IS_RHO_BOTTOM,
            interface_tilt_amplitude_fraction=IS_AMPLITUDE / IS_DEPTH, cell_average=true)
        # theta = 1: the predictor uses rho^n itself (the former scheme, bit for bit)
        m1 = Model(cfg(1.0))
        init!(m1)
        @test SweTransport2DV._baroclinic_density(m1, m1.state.rho) === m1.state.rho
        # a re-initialised model takes the same next step as a fresh one (no stale density history)
        used = Model(cfg(0.7))
        init!(used)
        for _ in 1:5
            step_forward!(used)
        end
        init!(used)
        fresh = Model(cfg(0.7))
        init!(fresh)
        step_forward!(used)
        step_forward!(fresh)
        @test maximum(abs, used.state.u .- fresh.state.u) <= TOL_IDENTITY
        @test maximum(abs, used.state.rho .- fresh.state.rho) <= TOL_IDENTITY * IS_RHO0
    end

    # Periods converge at about first order: the interface spreads over about one cell (cell-averaged start, tvd2
    # transport), which lowers the frequency. Measured (T_ref 46.65 s / 61.03 s): hydrostatic +3.06, +1.74, +1.00 %
    # (orders 0.82, 0.80), non-hydrostatic +5.09, +3.06, +1.86 % (0.73, 0.72); amplitude kept per period 0.981-0.991
    # (about 0.006 of the loss is the damping of the density extrapolation, pi (1 - theta) omega dt).
    for (hydrostatic, name, bound) in ((true, "hydrostatic", IS_PERIOD_HYDRO_MAX), (false, "nonhydrostatic", IS_PERIOD_NONHYDRO_MAX))
        @testset "test_validate_internal_seiche_period_$(name)" begin
            om, ratio_ref = is_reference(; hydrostatic=hydrostatic)
            T_ref = 2pi / om
            errors = Float64[]
            for (nx, nz) in IS_GRIDS
                r = is_run(nx, nz; hydrostatic=hydrostatic)
                push!(errors, abs(r.period - T_ref) / T_ref)
                @test r.kept >= IS_DAMPING_MIN
                @test r.umax <= IS_VELOCITY_MAX
                @test r.dvolume <= IS_VOLUME_TOL
                @test r.dmass <= IS_MASS_RTOL
                if (nx, nz) == IS_GRIDS[end]
                    # Neumann top condition (default): the hydrostatic surface response in both modes
                    ratio_h = is_reference(; hydrostatic=true)[2]
                    @test r.surface_ratio ≈ ratio_h rtol = hydrostatic ? IS_SURFACE_RATIO_HYDRO_RTOL : IS_SURFACE_RATIO_NEUMANN_RTOL
                end
            end
            @test errors[end] <= bound
            @test all(errors[j + 1] <= IS_CONVERGENCE_RATIO * errors[j] for j in 1:(length(errors) - 1))
            # the non-hydrostatic period is the full relation, not the long-wave one (they differ by 31 %)
            if !hydrostatic
                T_long = 2pi / is_reference(; hydrostatic=true)[1]
                @test abs(T_long - T_ref) / T_ref > 10 * errors[end]
            end
        end
    end

    @testset "test_validate_internal_seiche_surface_ratio_nonhydrostatic_dirichlet" begin
        # with the Dirichlet top condition (docs section 15) the surface follows the non-hydrostatic relation
        # (-0.00060; Neumann: -0.00150): observed -0.00070 on 40 x 40, -0.00065 on 80 x 80
        _, ratio_ref = is_reference(; hydrostatic=false)
        r = is_run(80, 80; hydrostatic=false, nonhydro_surface_dirichlet=true, pressure_solver_method="sparse")
        @test r.surface_ratio ≈ ratio_ref rtol = IS_SURFACE_RATIO_DIRICHLET_RTOL
    end
end
