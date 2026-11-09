include("test_common.jl")
using LinearAlgebra

# Radiation of dispersive waves at the open faces, right and left (issue #86, docs/numerics_open_boundaries.md section 7):
# theta-weighted radiation (radiation_theta), radiation speed of a design period on the current
# (radiation_period) and radiation of the non-hydrostatic pressure of the face (nonhydro_open_face = "radiation"),
# wetting-drying step. Section 39 of docs/TESTS_TECHNICAL_DOCUMENTATION.md.

const RD_G = 9.81                    # m/s^2, ModelConfig default
const RD_DEPTH = 0.5                 # m, still-water depth (Klopman's flume, issue #85)
const RD_AMPLITUDE = 0.002           # m, linear waves (a k <= 0.01)
const RD_NZ = 10
const RD_CELLS_PER_WAVELENGTH = 40
const RD_STEPS_PER_PERIOD = 100      # gravity-wave Courant number c dt/dx = 0.4
const RD_WAVELENGTHS = 8             # domain length in wavelengths
const RD_FIT_WINDOW = (4.0, 0.5)     # wavelengths: the fit uses L - 4 lambda <= x <= L - lambda/2
const RD_AVERAGE_PERIODS = 2         # periods of the complex amplitude at the end of the run
# Reflection bounds (measured values in brackets; |R| at the open face, fitted from the standing-wave pattern)
const RD_R_MAX_LONG = 0.002          # k h = 0.3, theta-weighted with the pressure radiation (0.0003)
const RD_R_MIN_LONG_BASELINE = 0.005 # k h = 0.3, the former condition: (1 - theta)/2 Cr k dx (0.0093)
const RD_R_MAX_DISPERSIVE = 0.025    # k h = 1.2 (0.0149)
const RD_R_MIN_DISPERSIVE_BASELINE = 0.05   # k h = 1.2, the former condition: long-wave speed, dq/dx = 0 (0.0687)
const RD_R_MAX_CURRENT = 0.015       # k h = 1.2 on a following current of 0.16 m/s (0.0051)
const RD_CURRENT = 0.16              # m/s, Klopman's mean current
# |R| against half the relative phase-speed error of the interior, |c_num/c - 1|/2 (k h = 2: 0.0366 against 0.0387, ratio 0.95)
const RD_DISPERSION_RATIO_RANGE = (0.7, 1.3)
const RD_SOLVER_TOL = 1e-6           # -, max |eta_direct - eta_multigrid| / a after 300 steps (CG tolerance 1e-8; measured 2.9e-10)
const RD_RADIATION_SPEED_RTOL = 1e-10   # -, residual of the Doppler dispersion relation / (g k tanh k h)
# Free left face (left_free): mirror symmetry and wave packets
const RD_MIRROR_TOL = 1e-12          # m, max |eta(x) - eta(L - x)| of a symmetric hump of 1 cm (measured 7.8e-16)
const RD_MIRROR_STEPS = 600
const RD_PACKET_WAVELENGTHS = 12     # channel; the reference channel is three times as long
const RD_PACKET_WIDTH = 1.5          # wavelengths, Gaussian envelope of the packet
const RD_PACKET_MAX_NEW = 0.012      # -, max |eta - eta_ref| / a after the packet left, new options (0.0068 both ends)
const RD_PACKET_MIN_FORMER = 0.015   # -, the same with the former condition (0.0217 both ends)
const RD_PACKET_MIRROR_RTOL = 0.01   # -, relative difference of the left and the right face

function rd_doppler_k(omega, h, U)
    k = omega^2 / RD_G
    for _ in 1:100
        f = (omega - k * U)^2 - RD_G * k * tanh(k * h)
        df = -2U * (omega - k * U) - RD_G * tanh(k * h) - RD_G * k * h / cosh(k * h)^2
        k -= f / df
    end
    return k
end

# Regular waves of k h (still water) on the current U from a velocity wave maker at x = 0 (linear profile, ramped over
# two periods), right free outflow. Runs until the reflection has travelled 4 wavelengths back; returns (|R|, kr/k)
# with kr the fitted numerical wave number. options: radiation options of ModelConfig; nz, budget (volume check).
function rd_reflection(; kh, U=0.0, options=(;), nz=RD_NZ, nonhydro_solver="auto", budget=false, steps=nothing)
    h = RD_DEPTH
    omega = sqrt(RD_G * (kh / h) * tanh(kh))
    T = 2pi / omega
    k = rd_doppler_k(omega, h, U)
    sigma = omega - k * U
    lam = 2pi / k
    L = RD_WAVELENGTHS * lam
    nx = RD_WAVELENGTHS * RD_CELLS_PER_WAVELENGTH
    dt = T / RD_STEPS_PER_PERIOD
    cg = 0.5 * sigma / k * (1 + 2k * h / sinh(2k * h))   # m/s, relative group velocity
    cfg = ModelConfig(; nx=nx, nz=nz, length=L, depth=h, dt=dt, theta=0.7, implicit_vertical_terms=true, nu=1e-6,
        rayleigh_damping=0.0, apply_froude_limiter=false, momentum_advection_scheme="tvd2", use_hydrostatic=false,
        wetting_drying=true, bed_level=zeros(nx), write_plots=false, nonhydro_solver=nonhydro_solver, options...)
    model = Model(cfg)
    fill!(model.state.rho, 1000.0)
    refresh_density_metadata!(model)
    initialize_water_level!(model, fill(h, nx))
    model.state.u .= U
    z = [(j - 0.5) * h / nz for j in 1:nz]
    profile = [RD_AMPLITUDE * sigma * cosh(k * zj) / sinh(k * h) for zj in z]
    set_x_inflow_outflow!(model, t -> U .+ profile .* (min(t / (2T), 1.0) * sin(omega * t)); right_free=true)
    nsteps = steps === nothing ? round(Int, (L / (U + cg) + RD_FIT_WINDOW[1] * lam / (cg - U) + 6T) / dt) : steps
    navg = round(Int, RD_AVERAGE_PERIODS * T / dt)
    acc = zeros(ComplexF64, nx)
    budget_error = 0.0
    for n in 1:nsteps
        V0 = sum(water_depth(model)) * model.dx
        step_forward!(model)
        if budget
            wd = model.wetdry
            budget_error = max(budget_error, abs(sum(water_depth(model)) * model.dx - V0 - dt * (sum(wd.q[:, 1]) - sum(wd.q[:, end]))))
        end
        n > nsteps - navg && (acc .+= (water_depth(model) .- h) .* exp(im * omega * n * dt))
    end
    steps === nothing || return model, budget_error
    A = 2 .* acc ./ navg
    x = ((1:nx) .- 0.5) .* (L / nx)
    sel = findall(xi -> L - RD_FIT_WINDOW[1] * lam <= xi <= L - RD_FIT_WINDOW[2] * lam, x)
    k2 = rd_doppler_k(omega, h, -U)   # reflected wave against the current
    function fit(kr, ki)
        M = hcat(exp.(im * (kr + im * ki) .* (x[sel] .- L)), exp.(-im * (k2 * kr / k + im * ki) .* (x[sel] .- L)))
        coef = M \ A[sel]
        return norm(M * coef - A[sel]), abs(coef[2] / coef[1])
    end
    best = (Inf, 0.0, 0.0, 0.0)
    for kr in range(0.8k, 1.2k; length=161), ki in range(-0.02k, 0.02k; length=41)
        res, R = fit(kr, ki)
        res < best[1] && (best = (res, R, kr, ki))
    end
    for _ in 1:3
        kr0, ki0 = best[3], best[4]
        for kr in range(kr0 - 0.0025k, kr0 + 0.0025k; length=41), ki in range(ki0 - 0.001k, ki0 + 0.001k; length=41)
            res, R = fit(kr, ki)
            res < best[1] && (best = (res, R, kr, ki))
        end
    end
    return best[2], best[3] / k
end

# Options of the new boundary for a wave of k h on still water: its own period
rd_period(kh) = 2pi / sqrt(RD_G * (kh / RD_DEPTH) * tanh(kh))
rd_new(kh) = (; radiation_theta=true, nonhydro_open_face="radiation", radiation_period=rd_period(kh))

# Free faces at both ends; a symmetric Gaussian hump (1 cm, width 0.5 m) in the middle of a 10 m channel: the solution
# must stay mirror-symmetric. Returns max over the run of |eta(x) - eta(L - x)| (m).
function rd_mirror_asymmetry(; options=(;))
    h, L, nx = RD_DEPTH, 10.0, 200
    cfg = ModelConfig(; nx=nx, nz=RD_NZ, length=L, depth=h, dt=0.01, theta=0.7, implicit_vertical_terms=true, nu=1e-6,
        rayleigh_damping=0.0, apply_froude_limiter=false, momentum_advection_scheme="tvd2", use_hydrostatic=false,
        wetting_drying=true, bed_level=zeros(nx), write_plots=false, options...)
    model = Model(cfg)
    fill!(model.state.rho, 1000.0)
    refresh_density_metadata!(model)
    xc = ((1:nx) .- 0.5) .* (L / nx)
    initialize_water_level!(model, h .+ 0.01 .* exp.(-((xc .- L / 2) ./ 0.5) .^ 2))
    set_x_inflow_outflow!(model, zeros(RD_NZ); left_free=true, right_free=true)
    asymmetry = 0.0
    for _ in 1:RD_MIRROR_STEPS
        step_forward!(model)
        d = water_depth(model)
        asymmetry = max(asymmetry, maximum(abs.(d .- reverse(d))))
    end
    return asymmetry
end

# Wave packet of k h (Gaussian envelope, linear-theory velocity profile) travelling in direction dir (-1: to the left
# face, +1: to the right face) from the middle of a channel with free faces at both ends; returns max |eta - eta_ref| / a
# over the half of the channel next to that face, once the reflection has travelled back a quarter of the channel. The
# reference is the same packet in a three times longer channel.
function rd_packet_deviation(; kh=1.2, dir, options=(;))
    h, a, nz = RD_DEPTH, RD_AMPLITUDE, RD_NZ
    k = kh / h
    omega = sqrt(RD_G * k * tanh(kh))
    T = 2pi / omega
    lam = 2pi / k
    cg = 0.5 * omega / k * (1 + 2kh / sinh(2kh))
    function run(nlam, x0)
        L = nlam * lam
        nx = nlam * RD_CELLS_PER_WAVELENGTH
        dx = L / nx
        cfg = ModelConfig(; nx=nx, nz=nz, length=L, depth=h, dt=T / RD_STEPS_PER_PERIOD, theta=0.7,
            implicit_vertical_terms=true, nu=1e-6, rayleigh_damping=0.0, apply_froude_limiter=false,
            momentum_advection_scheme="tvd2", use_hydrostatic=false, wetting_drying=true, bed_level=zeros(nx),
            write_plots=false, options...)
        model = Model(cfg)
        fill!(model.state.rho, 1000.0)
        refresh_density_metadata!(model)
        env(x) = exp(-((x - x0) / (RD_PACKET_WIDTH * lam))^2)
        xc = ((1:nx) .- 0.5) .* dx
        initialize_water_level!(model, h .+ a .* env.(xc) .* cos.(k .* (xc .- x0)))
        z = [(j - 0.5) * h / nz for j in 1:nz]
        for f in 1:(nx + 1), j in 1:nz
            x = (f - 1) * dx
            model.state.u[j, f] = dir * a * omega * cosh(k * z[j]) / sinh(kh) * env(x) * cos(k * (x - x0))
        end
        set_x_inflow_outflow!(model, zeros(nz); left_free=true, right_free=true)
        for _ in 1:round(Int, 0.75 * RD_PACKET_WAVELENGTHS * lam / cg / cfg.dt)
            step_forward!(model)
        end
        return water_depth(model) .- h
    end
    n = RD_PACKET_WAVELENGTHS * RD_CELLS_PER_WAVELENGTH
    eta = run(RD_PACKET_WAVELENGTHS, 0.5 * RD_PACKET_WAVELENGTHS * lam)
    ref = run(3 * RD_PACKET_WAVELENGTHS, 1.5 * RD_PACKET_WAVELENGTHS * lam)
    region = dir < 0 ? (1:(n ÷ 2)) : ((n ÷ 2 + 1):n)
    return maximum(abs.(eta[region] .- ref[region .+ n])) / a
end

@testset "Open boundary: radiation of dispersive waves (issue #86)" begin
    @testset "test_verify_radiation_speed_dispersion_relation" begin
        h = RD_DEPTH
        for U in (-0.16, 0.0, 0.16), T in (0.8, 1.44, 3.0)
            cfg = ModelConfig(nx=10, nz=4, wetting_drying=true, use_hydrostatic=false, nonhydro_open_face="radiation",
                radiation_period=T, write_plots=false)
            c = SweTransport2DV._radiation_speed(cfg, U, h)
            omega = 2pi / T
            k = omega / c
            @test abs((omega - k * U)^2 - RD_G * k * tanh(k * h)) <= RD_RADIATION_SPEED_RTOL * RD_G * k * tanh(k * h)
            @test c < U + sqrt(RD_G * h)   # dispersive: slower than the long wave
        end
        cfg0 = ModelConfig(nx=10, nz=4, write_plots=false)
        @test SweTransport2DV._radiation_speed(cfg0, 0.1, h) == 0.1 + sqrt(RD_G * h)   # period 0: long-wave speed
    end

    @testset "test_verify_radiation_options_config_checks" begin
        @test_throws ArgumentError ModelConfig(nx=10, nz=4, nonhydro_open_face="dirichlet", write_plots=false) |> Model
        @test_throws ArgumentError Model(ModelConfig(nx=10, nz=4, use_hydrostatic=false, nonhydro_open_face="radiation",
            pressure_solver_method="sparse", write_plots=false))   # fixed grid
        @test_throws ArgumentError Model(ModelConfig(nx=10, nz=4, wetting_drying=true, use_hydrostatic=false,
            radiation_period=-1.0, nonhydro_open_face="radiation", write_plots=false))
        @test_throws ArgumentError Model(ModelConfig(nx=10, nz=4, wetting_drying=true, use_hydrostatic=false,
            radiation_period=1.0, write_plots=false))   # period without the pressure radiation
        @test_throws ArgumentError Model(ModelConfig(nx=10, nz=4, wetting_drying=true, use_hydrostatic=true,
            radiation_period=1.0, nonhydro_open_face="radiation", write_plots=false))   # hydrostatic
    end

    @testset "test_verify_radiation_face_lake_at_rest_and_volume_budget" begin
        # still water: no wave maker motion (amplitude 0 through steps of a run with U = 0 and no forcing)
        h = RD_DEPTH
        nx = 40
        cfg = ModelConfig(; nx=nx, nz=RD_NZ, length=10.0, depth=h, dt=0.01, theta=0.7, implicit_vertical_terms=true,
            nu=1e-6, use_hydrostatic=false, wetting_drying=true, bed_level=zeros(nx), write_plots=false, rd_new(1.2)...)
        model = Model(cfg)
        fill!(model.state.rho, 1000.0)
        refresh_density_metadata!(model)
        initialize_water_level!(model, fill(h, nx))
        set_x_inflow_outflow!(model, zeros(RD_NZ); right_free=true)
        for _ in 1:200
            step_forward!(model)
        end
        @test maximum(abs, model.state.u) == 0.0
        @test maximum(abs, water_depth(model) .- h) == 0.0
        @test maximum(abs, model.wetdry.q_right) == 0.0
        # waves: the water volume changes only by the boundary fluxes of the step (measured 8.8e-15 m^2)
        _, budget_error = rd_reflection(; kh=1.2, options=rd_new(1.2), budget=true, steps=600)
        @test budget_error < TOL_MODEL_IDENTITY * RD_DEPTH * RD_WAVELENGTHS * 2pi / (1.2 / RD_DEPTH)
    end

    @testset "test_verify_radiation_face_direct_and_multigrid_agree" begin
        # the pressure radiation only adds to the diagonal: the system stays SPD and both solvers apply
        m1, _ = rd_reflection(; kh=1.2, options=rd_new(1.2), nonhydro_solver="direct", steps=300)
        m2, _ = rd_reflection(; kh=1.2, options=rd_new(1.2), nonhydro_solver="multigrid", steps=300)
        @test m2.wetdry.nh_multigrid_solves > 0 && m2.wetdry.nh_fallbacks == 0
        @test maximum(abs, water_depth(m1) .- water_depth(m2)) < RD_SOLVER_TOL * RD_AMPLITUDE
    end

    @testset "test_validate_theta_weighted_radiation_long_wave" begin
        R_new, _ = rd_reflection(; kh=0.3, options=rd_new(0.3))
        R_old, _ = rd_reflection(; kh=0.3)
        @test R_new < RD_R_MAX_LONG
        @test R_old > RD_R_MIN_LONG_BASELINE
    end

    @testset "test_validate_pressure_radiation_dispersive_wave" begin
        R_new, _ = rd_reflection(; kh=1.2, options=rd_new(1.2))
        R_old, _ = rd_reflection(; kh=1.2)
        @test R_new < RD_R_MAX_DISPERSIVE
        @test R_old > RD_R_MIN_DISPERSIVE_BASELINE
    end

    @testset "test_validate_pressure_radiation_following_current" begin
        R, _ = rd_reflection(; kh=1.2, U=RD_CURRENT, options=rd_new(1.2))
        @test R < RD_R_MAX_CURRENT
    end

    @testset "test_verify_left_free_face_config_checks" begin
        h, nx = RD_DEPTH, 20
        wd_cfg = ModelConfig(nx=nx, nz=4, length=2.0, depth=h, dt=0.01, implicit_vertical_terms=true, wetting_drying=true,
            bed_level=zeros(nx), write_plots=false)
        m = Model(wd_cfg)
        @test_throws ArgumentError set_x_inflow_outflow!(m, zeros(4); left_free=true, left_discharge=0.1)
        @test_throws ArgumentError set_x_inflow_outflow!(m, zeros(4); left_free=true, left_level=0.0)   # no tau
        @test_throws ArgumentError set_x_inflow_outflow!(m, zeros(4); left_free=true, left_relaxation_time=10.0)   # no target
        @test_throws ArgumentError set_x_inflow_outflow!(m, zeros(4); left_relaxation_time=10.0)   # not free
        @test_throws ArgumentError set_x_inflow_outflow!(m, zeros(4); left_free=true, left_level=0.0, left_relaxation_time=0.0)
        fixed = Model(ModelConfig(nx=nx, nz=4, implicit_vertical_terms=true, write_plots=false))
        @test_throws ArgumentError set_x_inflow_outflow!(fixed, zeros(4); left_free=true)   # fixed grid
        set_x_inflow_outflow!(m, zeros(4); left_free=true, left_level=0.0, left_relaxation_time=10.0)
        @test m.wetdry.left_open !== nothing && m.wetdry.left_open.relaxation_time == 10.0
        clear_x_inflow_outflow!(m)
        @test m.wetdry.left_open === nothing
    end

    @testset "test_verify_left_free_face_mirror_symmetry" begin
        # the left face is the mirror of the right one: a symmetric hump stays symmetric (former and new options)
        @test rd_mirror_asymmetry() < RD_MIRROR_TOL
        @test rd_mirror_asymmetry(; options=rd_new(1.2)) < RD_MIRROR_TOL
    end

    @testset "test_verify_left_free_face_uniform_outflow_steady" begin
        # a current of 0.16 m/s to the left: velocity inflow at the right face, relaxed free left face (target 0); the
        # uniform flow is a steady state to round-off
        h, nx, U = RD_DEPTH, 40, -RD_CURRENT
        cfg = ModelConfig(; nx=nx, nz=RD_NZ, length=10.0, depth=h, dt=0.01, theta=0.7, implicit_vertical_terms=true,
            nu=1e-6, rayleigh_damping=0.0, apply_froude_limiter=false, use_hydrostatic=false, wetting_drying=true,
            bed_level=zeros(nx), write_plots=false, rd_new(1.2)...)
        model = Model(cfg)
        fill!(model.state.rho, 1000.0)
        refresh_density_metadata!(model)
        initialize_water_level!(model, fill(h, nx))
        model.state.u .= U
        set_x_inflow_outflow!(model, zeros(RD_NZ); left_free=true, left_level=0.0, left_relaxation_time=10.0,
            open_right=false, right_profile=fill(U, RD_NZ))
        for _ in 1:300
            step_forward!(model)
        end
        @test maximum(abs, model.state.u .- U) < TOL_MODEL_IDENTITY
        @test maximum(abs, water_depth(model) .- h) < TOL_MODEL_IDENTITY
    end

    @testset "test_validate_left_free_face_reflection_matches_right" begin
        # k h = 1.2 packet: the left face reflects as the right one, and the new options reduce the reflection there too
        new_left = rd_packet_deviation(; dir=-1, options=rd_new(1.2))
        new_right = rd_packet_deviation(; dir=1, options=rd_new(1.2))
        former_left = rd_packet_deviation(; dir=-1)
        @test new_left < RD_PACKET_MAX_NEW
        @test former_left > RD_PACKET_MIN_FORMER
        @test abs(new_left - new_right) <= RD_PACKET_MIRROR_RTOL * new_right
    end

    @testset "test_validate_residual_reflection_is_interior_dispersion_error" begin
        # the boundary radiates the linear-theory phase speed; the interior's phase speed differs by c_num/c - 1 =
        # k/k_num - 1, and the mismatch reflects about half of it
        R, kr = rd_reflection(; kh=2.0, options=rd_new(2.0))
        ratio = R / (abs(1 / kr - 1) / 2)
        @test RD_DISPERSION_RATIO_RANGE[1] < ratio < RD_DISPERSION_RATIO_RANGE[2]
    end
end
