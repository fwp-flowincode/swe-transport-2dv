include("open_boundary_helpers.jl")

# Wave generation through open faces, the calibrated radiation speed, the relaxation of the incoming characteristic
# variable and the characteristic open face (issue #86, docs/numerics_open_boundaries.md sections 7.4 and 8),
# wetting-drying step. Section 40 of docs/TESTS_TECHNICAL_DOCUMENTATION.md.

# Measured values in brackets; the tolerances leave room for platform round-off and small changes of the scheme.
const OG_KH = 1.2                       # -, design wave (Klopman's flume: T = 1.44 s, h = 0.5 m)
const OG_CURRENT = 0.16                 # m/s
const OG_AMPLITUDE_RTOL = 0.01          # -, |generated amplitude / target - 1| (still 0.0004, following 0.0009, opposing 0.0017)
const OG_R_MAX_GENERATION = 0.005       # -, reflection at the absorbing face while generating (0.0018, 0.0027, 0.0038)
const OG_R_MAX_CALIBRATED = 0.004       # -, k h = 1.2 with the calibrated speed factor (0.0018; factor 1: 0.0149)
const OG_FACTOR_RANGE = (1.025, 1.040)  # -, numerical_phase_speed_factor at k h = 1.2, 10 layers (1.0328; progressive fit 1.0320)
const OG_R_MAX_INFLOW = 0.005           # -, hydrostatic k h = 0.3, current entering the face (radiation 0.0022, box 0.0015)
const OG_R_MIN_INFLOW_FORMER = 0.02     # -, the same with backward-Euler radiation (U + c, no intrinsic speed; 0.041)
const OG_R_MAX_BOX = 0.0005             # -, hydrostatic k h = 0.3, box form beta = -1/2 (0.00013)
const OG_BETA0_ANALYSIS_RTOL = 0.25     # -, beta = 0 against (k dx)^2/16 = 0.00154 (0.00154)
const OG_RELAX_TAU_PERIODS = 2          # -, short relaxation time (in periods) of the relaxation comparison
const OG_R_MAX_INCOMING_RELAX = 0.006   # -, the incoming variable relaxed with tau = 2 T (0.0022)
const OG_R_MIN_LEVEL_RELAX = 0.02       # -, the level relaxed with tau = 2 T: 1/sqrt(1 + 4 omega^2 tau^2) = 0.040 (0.051)

# Regular wave of k h (still-water value) generated at the free left face as the incoming level eta_in(t) (ramped over
# 3 periods), absorbed at the free right face, on a uniform current U; both faces relaxed with tau when given (target
# discharge U h). Returns (generated amplitude / target at x = 0, reflection at x = L), from the complex amplitude over
# the inner channel fitted by an incident and a reflected wave with a complex numerical wave number.
function og_generate(; kh=OG_KH, U=0.0, options=(;), calibrate=true, tau_periods=nothing, nlam=8)
    h, a, nz, cpw = RD_DEPTH, RD_AMPLITUDE, RD_NZ, RD_CELLS_PER_WAVELENGTH
    omega = sqrt(RD_G * (kh / h) * tanh(kh))
    T = 2pi / omega
    k = rd_doppler_k(omega, h, U)
    k2 = rd_doppler_k(omega, h, -U)
    lam = 2pi / k
    L = nlam * lam
    nx = nlam * cpw
    cfg = ModelConfig(; nx=nx, nz=nz, length=L, depth=h, dt=T / RD_STEPS_PER_PERIOD, theta=0.7,
        implicit_vertical_terms=true, nu=1e-6, rayleigh_damping=0.0, apply_froude_limiter=false,
        momentum_advection_scheme="tvd2", use_hydrostatic=false, wetting_drying=true, bed_level=zeros(nx),
        write_plots=false, radiation_theta=true, nonhydro_open_face="radiation", radiation_period=T, options...)
    calibrate && (cfg.radiation_speed_factor = numerical_phase_speed_factor(cfg; current=U))
    model = Model(cfg)
    fill!(model.state.rho, 1000.0)
    refresh_density_metadata!(model)
    initialize_water_level!(model, fill(h, nx))
    model.state.u .= U
    eta_in(t) = a * min(t / (3T), 1.0) * sin(omega * t)
    if tau_periods === nothing
        set_x_inflow_outflow!(model, zeros(nz); left_free=true, right_free=true, left_incoming_level=eta_in)
    else
        tau = tau_periods * T
        q_target = cfg.open_face_relaxation == "incoming" ? (; left_target_discharge=U * h, right_target_discharge=U * h) : (;)
        set_x_inflow_outflow!(model, zeros(nz); left_free=true, left_level=0.0, left_relaxation_time=tau, right_free=true,
            right_level=0.0, right_relaxation_time=tau, left_incoming_level=eta_in, q_target...)
    end
    cg = 0.5 * (omega - k * U) / k * (1 + 2k * h / sinh(2k * h)) + U   # m/s, absolute group velocity
    nsteps = round(Int, (2.5L / cg + 8T) / cfg.dt)
    navg = round(Int, RD_AVERAGE_PERIODS * T / cfg.dt)
    acc = zeros(ComplexF64, nx)
    for n in 1:nsteps
        step_forward!(model)
        n > nsteps - navg && (acc .+= (water_depth(model) .- h) .* exp(im * omega * n * cfg.dt))
    end
    A = 2 .* acc ./ navg
    x = ((1:nx) .- 0.5) .* (L / nx)
    sel = findall(xi -> lam <= xi <= L - lam, x)
    function fit(kr, ki)
        M = hcat(exp.(im * (kr + im * ki) .* x[sel]), exp.(-im * (k2 * kr / k + im * ki) .* (x[sel] .- L)))
        c = M \ A[sel]
        return norm(M * c - A[sel]), abs(c[1]), abs(c[2]) / (abs(c[1]) * exp(-ki * L))
    end
    best = (Inf, 0.0, 0.0, 0.0, 0.0)
    for kr in range(0.9k, 1.1k; length=201), ki in range(0.0, 0.03k; length=61)
        res, amp, R = fit(kr, ki)
        res < best[1] && (best = (res, amp, R, kr, ki))
    end
    for _ in 1:3
        kr0 = best[4]
        for kr in range(kr0 - 0.002k, kr0 + 0.002k; length=41), ki in range(0.0, 0.03k; length=301)
            res, amp, R = fit(kr, ki)
            res < best[1] && (best = (res, amp, R, kr, ki))
        end
    end
    return best[2] / a, best[3]
end

# Uniform current U through two free faces, both relaxed (tau 20 s, targets 0 and U h); max |u - U| and max |eta| after
# 1000 steps.
function og_uniform_current(; U=OG_CURRENT, options=(;))
    h, nx, nz = RD_DEPTH, 80, RD_NZ
    cfg = ModelConfig(; nx=nx, nz=nz, length=20.0, depth=h, dt=0.0144, theta=0.7, implicit_vertical_terms=true, nu=1e-6,
        rayleigh_damping=0.0, apply_froude_limiter=false, momentum_advection_scheme="tvd2", use_hydrostatic=false,
        wetting_drying=true, bed_level=zeros(nx), write_plots=false, radiation_theta=true, options...)
    model = Model(cfg)
    fill!(model.state.rho, 1000.0)
    refresh_density_metadata!(model)
    initialize_water_level!(model, fill(h, nx))
    model.state.u .= U
    q = (cfg.open_face_scheme == "characteristic" || cfg.open_face_relaxation == "incoming") ?
        (; left_target_discharge=U * h, right_target_discharge=U * h) : (;)
    set_x_inflow_outflow!(model, zeros(nz); left_free=true, left_level=0.0, left_relaxation_time=20.0, right_free=true,
        right_level=0.0, right_relaxation_time=20.0, q...)
    for _ in 1:1000
        step_forward!(model)
    end
    return maximum(abs, model.state.u .- U), maximum(abs, water_depth(model) .- h)
end

@testset "Open boundary: generation, calibrated speed, characteristic face (issue #86)" begin
    @testset "test_verify_open_face_options_config_checks" begin
        h, nx = RD_DEPTH, 20
        base = (; nx=nx, nz=4, length=2.0, depth=h, dt=0.01, implicit_vertical_terms=true, wetting_drying=true,
            bed_level=zeros(nx), write_plots=false)
        @test_throws ArgumentError Model(ModelConfig(; base..., open_face_scheme="box"))
        @test_throws ArgumentError Model(ModelConfig(; base..., open_face_relaxation="target"))
        @test_throws ArgumentError Model(ModelConfig(; base..., open_face_beta=0.5))
        @test_throws ArgumentError Model(ModelConfig(; base..., open_face_wet_depth=-1.0))
        @test_throws ArgumentError Model(ModelConfig(; base..., radiation_speed_factor=3.0))
        @test_throws ArgumentError Model(ModelConfig(nx=nx, nz=4, implicit_vertical_terms=true, write_plots=false,
            open_face_scheme="characteristic"))   # fixed grid
        @test_throws ArgumentError Model(ModelConfig(nx=nx, nz=4, implicit_vertical_terms=true, write_plots=false,
            open_face_relaxation="incoming"))
        m = Model(ModelConfig(; base...))
        eta_in = t -> 0.001 * sin(t)
        @test_throws ArgumentError set_x_inflow_outflow!(m, zeros(4); right_free=true, right_incoming_level=eta_in)  # no radiation_theta
        mt = Model(ModelConfig(; base..., radiation_theta=true))
        @test_throws ArgumentError set_x_inflow_outflow!(mt, zeros(4); left_incoming_level=eta_in)   # not a free face
        @test_throws ArgumentError set_x_inflow_outflow!(mt, zeros(4); right_free=true, right_level=0.0,
            right_relaxation_time=10.0, right_target_discharge=0.1)   # level relaxation has no target discharge
        mi = Model(ModelConfig(; base..., radiation_theta=true, open_face_relaxation="incoming"))
        @test_throws ArgumentError set_x_inflow_outflow!(mi, zeros(4); right_free=true, right_target_discharge=0.1)   # no tau
        set_x_inflow_outflow!(mi, zeros(4); left_free=true, left_level=0.0, left_relaxation_time=10.0,
            left_target_discharge=0.05, right_free=true, right_incoming_level=eta_in)
        @test mi.wetdry.char_left.target_discharge == 0.05 && mi.wetdry.char_right.incoming === eta_in
        @test_throws ArgumentError numerical_phase_speed_factor(ModelConfig(; base...))   # no design period
    end

    @testset "test_verify_radiation_speed_factor" begin
        cfg = ModelConfig(nx=20, nz=4, depth=RD_DEPTH, wetting_drying=true, bed_level=zeros(20), implicit_vertical_terms=true,
            use_hydrostatic=false, nonhydro_open_face="radiation", radiation_period=rd_period(OG_KH), write_plots=false)
        U, f = 0.16, 1.03
        c1 = SweTransport2DV._radiation_speed(cfg, U, RD_DEPTH)
        cfg.radiation_speed_factor = f
        @test SweTransport2DV._radiation_speed(cfg, U, RD_DEPTH) ≈ U + f * (c1 - U) rtol = TOL_MODEL_IDENTITY
        cfg.radiation_period = 0.0
        @test SweTransport2DV._radiation_speed(cfg, U, RD_DEPTH) ≈ U + f * sqrt(RD_G * RD_DEPTH) rtol = TOL_MODEL_IDENTITY
    end

    @testset "test_verify_open_face_schemes_steady_current" begin
        # a uniform current through two relaxed free faces is a steady state, for every scheme and relaxation
        for opts in ((;), (; open_face_relaxation="incoming"), (; open_face_scheme="characteristic"))
            du, deta = og_uniform_current(; options=opts)
            @test du < TOL_MODEL_IDENTITY
            @test deta < TOL_MODEL_IDENTITY
        end
    end

    @testset "test_verify_characteristic_face_mirror_symmetry" begin
        # the characteristic left face is the exact mirror of the right one
        @test rd_mirror_asymmetry(; options=(; open_face_scheme="characteristic")) < RD_MIRROR_TOL
        @test rd_mirror_asymmetry(; options=(; open_face_scheme="characteristic", radiation_theta=true,
            open_face_relaxation="incoming")) < RD_MIRROR_TOL
    end

    @testset "test_validate_characteristic_box_form_long_waves" begin
        # hydrostatic long waves at 40 cells per wavelength: beta = 0 reflects (k dx)^2/16 (normal-mode analysis), the box
        # form beta = -1/2 an order less (docs section 8.3)
        hydro = (; use_hydrostatic=true, open_face_scheme="characteristic")
        R0, _ = rd_reflection(; kh=0.3, nz=4, options=merge(hydro, (; open_face_beta=0.0)))
        Rb, _ = rd_reflection(; kh=0.3, nz=4, options=hydro)
        kdx = 2pi / RD_CELLS_PER_WAVELENGTH
        @test abs(R0 / (kdx^2 / 16) - 1) < OG_BETA0_ANALYSIS_RTOL
        @test Rb < OG_R_MAX_BOX
    end

    @testset "test_validate_current_entering_open_face" begin
        # a current of 0.16 m/s entering through the absorbing face (hydrostatic, k h = 0.3): the intrinsic radiation
        # speed (radiation_theta) and the characteristic face absorb; the former backward-Euler radiation with U + c not
        hydro = (; use_hydrostatic=true)
        R_rad, _ = rd_reflection(; kh=0.3, U=-OG_CURRENT, nz=4, options=merge(hydro, (; radiation_theta=true)))
        R_char, _ = rd_reflection(; kh=0.3, U=-OG_CURRENT, nz=4, options=merge(hydro, (; open_face_scheme="characteristic")))
        R_former, _ = rd_reflection(; kh=0.3, U=-OG_CURRENT, nz=4, options=hydro)
        @test R_rad < OG_R_MAX_INFLOW
        @test R_char < OG_R_MAX_INFLOW
        @test R_former > OG_R_MIN_INFLOW_FORMER
    end

    @testset "test_validate_calibrated_radiation_speed" begin
        # numerical_phase_speed_factor matches the phase speed of the model's progressive wave; with it the reflection
        # of k h = 1.2 drops from the interior's dispersion error (1.5 %) to 0.18 %
        h = RD_DEPTH
        T = rd_period(OG_KH)
        lam = 2pi / (OG_KH / h)
        nx = RD_WAVELENGTHS * RD_CELLS_PER_WAVELENGTH
        cfg = ModelConfig(; nx=nx, nz=RD_NZ, length=RD_WAVELENGTHS * lam, depth=h, dt=T / RD_STEPS_PER_PERIOD, theta=0.7,
            implicit_vertical_terms=true, nu=1e-6, rayleigh_damping=0.0, apply_froude_limiter=false,
            momentum_advection_scheme="tvd2", use_hydrostatic=false, wetting_drying=true, bed_level=zeros(nx),
            write_plots=false, rd_new(OG_KH)...)
        f = numerical_phase_speed_factor(cfg)
        @test OG_FACTOR_RANGE[1] < f < OG_FACTOR_RANGE[2]
        R, _ = rd_reflection(; kh=OG_KH, options=merge(rd_new(OG_KH), (; radiation_speed_factor=f)))
        @test R < OG_R_MAX_CALIBRATED
    end

    @testset "test_validate_wave_generation_still_water" begin
        amp, R = og_generate()
        @test abs(amp - 1) < OG_AMPLITUDE_RTOL
        @test R < OG_R_MAX_GENERATION
    end

    @testset "test_validate_wave_generation_on_current" begin
        # Klopman's cases: the waves enter with the current (following) or against it (opposing; the current enters
        # through the absorbing face); relaxation of the incoming variable with tau = 20 T
        for U in (OG_CURRENT, -OG_CURRENT)
            amp, R = og_generate(; U=U, options=(; open_face_relaxation="incoming"), tau_periods=20)
            @test abs(amp - 1) < OG_AMPLITUDE_RTOL
            @test R < OG_R_MAX_GENERATION
        end
    end

    @testset "test_validate_incoming_relaxation_does_not_reflect" begin
        # a short relaxation time (2 periods): the level relaxation reflects 1/sqrt(1 + 4 omega^2 tau^2), the relaxation
        # of the incoming characteristic variable does not see the outgoing wave
        _, R_level = og_generate(; tau_periods=OG_RELAX_TAU_PERIODS)
        _, R_incoming = og_generate(; tau_periods=OG_RELAX_TAU_PERIODS, options=(; open_face_relaxation="incoming"))
        @test R_level > OG_R_MIN_LEVEL_RELAX
        @test R_incoming < OG_R_MAX_INCOMING_RELAX
    end
end
