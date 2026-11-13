# Helpers and constants of the open-boundary tests of issue #86 (test_open_boundary_radiation.jl,
# test_open_boundary_generation.jl): regular-wave reflection, mirror symmetry and wave packets in the wetting-drying
# step. WHY a file of its own: both test files use them, and each test file can also run on its own.
include("test_common.jl")
using LinearAlgebra

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
# Merged top layer (wd_top_layer = "merged", issue #88): the interior's phase error at k h = 2 is 0.4 % instead of 7 %, so
# the ratio above is no longer a measure (1.83); the reflection itself is bounded instead (0.0039; "cut" 0.0366)
const RD_R_MAX_KH2_MERGED = 0.006
# the expected values of the dispersion-dependent tests follow the default top layer (they switch with the default)
const OB_MERGED_DEFAULT = ModelConfig().wd_top_layer == "merged"
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

