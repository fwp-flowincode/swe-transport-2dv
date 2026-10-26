# Vegetation cases of issue #8 (docs/plan_issue_8_vegetation.md): settings and references shared by the plot scripts in
# this folder and by test/test_vegetation.jl.
#
# References:
# - Nepf (2012), Annu. Rev. Fluid Mech. 44, 123-142 (flow through emergent and submerged canopies)
# - Luhar & Nepf (2011), Limnol. Oceanogr. 56(6), 2003-2017 (reconfiguration of flexible vegetation)
# - Gosselin, de Langre & Machado-Almeida (2010), J. Fluid Mech. 650, 319-341 (reconfiguration regimes)

using SweTransport2DV
using LinearAlgebra

const VEG_RHO0 = 1000.0     # kg/m^3, water density (uniform)

"""
    vegetation_case_config(; kwargs...) -> ModelConfig

Uniform-density configuration for the vegetation cases: rho0 = VEG_RHO0 everywhere, `tvd2` momentum advection, no
Rayleigh damping, no Froude limiter, no output files. Keyword arguments override or add `ModelConfig` fields.
"""
function vegetation_case_config(; kwargs...)
    return ModelConfig(; rho0=VEG_RHO0, rho_left=VEG_RHO0, rho_right=VEG_RHO0, rayleigh_damping=0.0, apply_froude_limiter=false,
        momentum_advection_scheme="tvd2", write_plots=false, write_animation=false, write_pressure_buoyancy_diagnostics=false,
        kwargs...)
end

"""
    uniform_density_model(cfg) -> Model

Model of `cfg` with the density VEG_RHO0 everywhere.
"""
function uniform_density_model(cfg::ModelConfig)
    model = Model(cfg)
    fill!(model.state.rho, VEG_RHO0)
    refresh_density_metadata!(model)
    return model
end

"""
    canopy_reference_profile(; H, hv, a, cd, G, nu, M=4000) -> (z, u)

Steady 1D reference of a canopy in a periodic channel driven by the body force `G` (m/s²), with constant viscosity
`nu` (m²/s), free-slip bed and stress-free surface:

    nu u'' = 1/2 C_D a phi(z) u|u| - G,    u'(0) = u'(H) = 0,

phi = 1 below the stem height `hv` (m) and 0 above (fraction of the cell below hv), frontal area per unit volume `a`
(1/m), drag coefficient `cd`, depth `H` (m). Emergent canopy (hv >= H): u = sqrt(2G/(C_D a)) exactly. Solved
independently of the model on `M` cell-centred finite volumes with Newton's method (tolerance 1e-14 m/s); returns the
cell centres z (m) and u (m/s).
"""
function canopy_reference_profile(; H, hv, a, cd, G, nu, M=4000)
    dz = H / M
    z = [(k - 0.5) * dz for k in 1:M]
    phi = [clamp((hv - (k - 1) * dz) / dz, 0.0, 1.0) for k in 1:M]
    u = fill(sqrt(2G * H / (cd * a * min(hv, H))), M)
    for _ in 1:100
        diag = zeros(M); lower = zeros(M - 1); upper = zeros(M - 1); r = zeros(M)
        for k in 1:M
            if k > 1
                r[k] += nu * (u[k - 1] - u[k]) / dz^2; lower[k - 1] = nu / dz^2; diag[k] -= nu / dz^2
            end
            if k < M
                r[k] += nu * (u[k + 1] - u[k]) / dz^2; upper[k] = nu / dz^2; diag[k] -= nu / dz^2
            end
            r[k] += G - 0.5 * cd * a * phi[k] * u[k] * abs(u[k])
            diag[k] -= cd * a * phi[k] * abs(u[k])
        end
        du = -(Tridiagonal(lower, diag, upper) \ r)
        u .+= du
        maximum(abs, du) < 1e-14 && break
    end
    return z, u
end

"""
    linear_interpolation(zq, z, u) -> Float64

Linear interpolation of the samples `u` at the increasing positions `z` to `zq` (constant extrapolation of the slope
at the ends).
"""
function linear_interpolation(zq::Real, z::AbstractVector{<:Real}, u::AbstractVector{<:Real})
    i = clamp(searchsortedlast(z, zq), 1, length(z) - 1)
    t = (zq - z[i]) / (z[i + 1] - z[i])
    return (1 - t) * u[i] + t * u[i + 1]
end

"""
    reconfiguration_number(cauchy; n_segments=80, buoyancy_number=0.0) -> (R, theta)

Reconfiguration number R = F / (1/2 rho C_D b l U²) (= effective length l_e/l of Luhar & Nepf 2011) of a single
flexible stem in a uniform current, at the Cauchy number Ca = rho C_D b U² l³ / (2 EI) and the buoyancy number
B = B_l l³ / EI (B_l net buoyancy per length, N/m), and the segment angles of the posture (rad). Dimensionless: R and
the shape depend only on Ca and B.
"""
function reconfiguration_number(cauchy::Real; n_segments::Int=80, buoyancy_number::Real=0.0, theta=zeros(n_segments))
    b, cd, l, ei = 0.01, 1.0, 1.0, 1.0      # m, -, m, N m^2 (any values; R depends on Ca and B only)
    u = sqrt(2cauchy * ei / (VEG_RHO0 * cd * b * l^3))
    _, force, _ = update_stem_posture!(theta, (k, x) -> u; length=l, diameter=b, drag_coefficient=cd, bending_stiffness=ei,
        net_buoyancy=buoyancy_number * ei / l^3, rho0=VEG_RHO0, dz=l / 1000, nz=100_000, water_level=100l)
    return force / (0.5 * VEG_RHO0 * cd * b * l * u^2), theta
end

# ---------------------------------------------------------------------------------------------------------------------
# Regular waves through a rigid canopy (step 2, reference R8)
# ---------------------------------------------------------------------------------------------------------------------

"""
    wave_number(omega, h; g=9.81, dispersive=true) -> Float64

Wave number k (1/m) of a linear wave of angular frequency `omega` (1/s) in depth `h` (m): omega² = g k tanh(k h)
(Newton), or omega = k sqrt(g h) for `dispersive=false` (shallow water; the hydrostatic model and the default
non-hydrostatic free surface, section 14 of the test documentation).
"""
function wave_number(omega::Real, h::Real; g::Real=9.81, dispersive::Bool=true)
    k = omega / sqrt(g * h)
    dispersive || return k
    for _ in 1:50
        f = g * k * tanh(k * h) - omega^2
        k -= f / (g * tanh(k * h) + g * k * h / cosh(k * h)^2)
    end
    return k
end

"""
    dalrymple_decay_coefficient(; cd, a, H0, k, h, alpha) -> Float64

Decay coefficient beta (1/m) of the wave height H(x) = H0 / (1 + beta x) of a linear regular wave over rigid vertical
cylinders (Dalrymple, Kirby & Hwang 1984; Mendez & Losada 2004):

    beta = 4/(9 pi) C_D a H0 k (sinh³(k alpha h) + 3 sinh(k alpha h)) / ((sinh(2kh) + 2kh) sinh(kh)),

frontal area per unit volume `a` (1/m), wave height `H0` (m) at x = 0, wave number `k` (1/m), depth `h` (m), relative
stem height `alpha` (1 = emergent). It follows from the wave energy balance d(E c_g)/dx = -epsilon with the time-averaged
dissipation epsilon = int <1/2 rho C_D a |u| u²> dz of the linear-wave velocity (<|cos|³> = 4/(3 pi)); shallow-water
limit: beta = C_D a alpha H0 / (3 pi h).
"""
function dalrymple_decay_coefficient(; cd::Real, a::Real, H0::Real, k::Real, h::Real, alpha::Real)
    s = sinh(k * alpha * h)
    return 4 / (9pi) * cd * a * H0 * k * (s^3 + 3s) / ((sinh(2k * h) + 2k * h) * sinh(k * h))
end

"""
    run_wave_flume(; depth, kh, wave_height, frontal_area, alpha=1.0, use_hydrostatic, dispersive=false, canopy=(10.0, 70.0),
                   cells_per_wavelength=60, steps_per_period=4000, periods=20, average_periods=4, nz=10) -> NamedTuple

Regular waves generated at the left boundary (linear-theory velocity profile, ramped over two periods) run through a rigid
canopy over x in `canopy` (m) with frontal area `frontal_area` (1/m; 0 = no canopy) and stem height alpha*depth, and are
absorbed in a sponge of two wavelengths at the right end. Returns x (m, cell centres), the first-harmonic wave height
H(x) (m) over the last `average_periods`, k (1/m), the wavelength (m) and the period (s). `dispersive=true` (non-hydrostatic
only) uses the Dirichlet surface condition of section 15 and the full linear dispersion relation.
WHY steps_per_period = 4000: the free surface is integrated fully implicitly, which damps a wave by about 2 pi²/N per
period with N steps per period (0.5 % at N = 4000).
"""
function run_wave_flume(; depth, kh, wave_height, frontal_area, alpha=1.0, use_hydrostatic, dispersive=false, canopy=(10.0, 70.0),
    cells_per_wavelength=60, steps_per_period=4000, periods=20, average_periods=4, nz=10)
    k = kh / depth
    omega = dispersive ? sqrt(9.81 * k * tanh(kh)) : k * sqrt(9.81 * depth)
    period = 2pi / omega
    wavelength = 2pi / k
    domain = canopy[2] + 2wavelength
    nx = round(Int, domain / wavelength * cells_per_wavelength)
    dt = period / steps_per_period
    mode = use_hydrostatic ? (; use_hydrostatic=true) :
        (; use_hydrostatic=false, pressure_solver_method="sparse", nonhydro_surface_dirichlet=dispersive)
    cfg = vegetation_case_config(; length=domain, depth=depth, nx=nx, nz=nz, dt=dt, nt=round(Int, periods * period / dt),
        variable_top_layer=false, nu=1e-6, sponge_length=2wavelength, sponge_rate=4omega, mode...)
    model = uniform_density_model(cfg)
    if frontal_area > 0
        attach_vegetation!(model, VegetationModel(; patches=[VegetationPatch(x_start=canopy[1], x_end=canopy[2], stem_height=alpha * depth,
            stem_diameter=0.01, stem_density=frontal_area / 0.01, drag_coefficient=1.0)]))
    end
    z = [(j - 0.5) * model.dz for j in 1:nz]
    amplitude = dispersive ? [wave_height / 2 * omega * cosh(k * zj) / sinh(kh) for zj in z] : fill(wave_height / 2 * sqrt(9.81 / depth), nz)
    n_avg = round(Int, average_periods * period / dt)
    acc_c = zeros(nx); acc_s = zeros(nx)
    for n in 1:cfg.nt
        t = n * dt
        set_x_inflow_outflow!(model, amplitude .* (min(t / (2period), 1.0) * sin(omega * t)); open_right=false, right_profile=zeros(nz))
        step_forward!(model)
        if n > cfg.nt - n_avg
            acc_c .+= model.state.eta .* cos(omega * t)
            acc_s .+= model.state.eta .* sin(omega * t)
        end
    end
    H = 2 .* hypot.(2 .* acc_c ./ n_avg, 2 .* acc_s ./ n_avg)
    return (x=[(i - 0.5) * model.dx for i in 1:nx], H=H, k=k, wavelength=wavelength, period=period)
end

# ---------------------------------------------------------------------------------------------------------------------
# Lock exchange through an emergent rigid canopy (step 2, reference R9)
# ---------------------------------------------------------------------------------------------------------------------

"""
    run_canopy_lock_exchange(; frontal_area, use_hydrostatic, length=8.0, depth=0.2, reduced_gravity=0.02, nx=400, nz=10,
                             dt=0.02, t_end=400.0, sample_interval=1.0, snapshot_times=()) -> NamedTuple

Lock exchange (heavy water in the left half, light in the right half, density difference from `reduced_gravity` g'
(m/s²)) through an emergent rigid canopy over the whole tank (frontal area `frontal_area` (1/m), stem diameter 6.4 mm,
C_D = 1; 0 = no canopy). Returns the times (s) and positions (m, from the gate) of the dense front along the bed
(heavy-water fraction 0.5), and density snapshots at `snapshot_times` (s).

Expected behaviour: without canopy the front moves at a constant speed (about 0.5 sqrt(g' H); Benjamin 1968). In a
dense canopy the current becomes drag-dominated and its profile triangular (Tanino, Nepf & Kulis 2005): the dense layer
thins linearly from the gate to the toe, so its pressure gradient is about g' H / x_f and balances the quadratic drag
1/2 C_D a u², giving u_toe proportional to (g' H / (C_D a x_f))^(1/2) and x_f proportional to t^(2/3) (linear drag:
t^(1/2)). Derived here; the comparison with the toe velocity of Tanino et al. (2005) is still open.
"""
function run_canopy_lock_exchange(; frontal_area, use_hydrostatic, length=8.0, depth=0.2, reduced_gravity=0.02, nx=400, nz=10,
    dt=0.02, t_end=400.0, sample_interval=1.0, snapshot_times=())
    drho = reduced_gravity / 9.81 * VEG_RHO0
    cfg = ModelConfig(length=length, depth=depth, nx=nx, nz=nz, dt=dt, nt=round(Int, t_end / dt), use_hydrostatic=use_hydrostatic,
        rho0=VEG_RHO0, rho_left=VEG_RHO0 + drho, rho_right=VEG_RHO0, rayleigh_damping=0.0, apply_froude_limiter=false,
        momentum_advection_scheme="tvd2", transport_advection_scheme="tvd2", nu=1e-6, write_plots=false, write_animation=false,
        write_pressure_buoyancy_diagnostics=false)
    model = Model(cfg)
    initialize_lock_exchange!(model)
    if frontal_area > 0
        attach_vegetation!(model, VegetationModel(; patches=[VegetationPatch(x_start=0.0, x_end=length, stem_height=2depth,
            stem_diameter=0.0064, stem_density=frontal_area / 0.0064, drag_coefficient=1.0)]))
    end
    threshold = VEG_RHO0 + 0.5 * drho
    stride = round(Int, sample_interval / dt)
    snapshot_steps = Dict(round(Int, t / dt) => t for t in snapshot_times)
    times = Float64[]; fronts = Float64[]
    snapshots = Dict{Float64,Matrix{Float64}}()
    for n in 1:cfg.nt
        step_forward!(model)
        if n % stride == 0
            heavy = findall(>(threshold), @view model.state.rho[1, :])
            push!(times, n * dt)
            push!(fronts, (maximum(heavy) - 0.5) * model.dx - length / 2)
        end
        haskey(snapshot_steps, n) && (snapshots[snapshot_steps[n]] = copy(model.state.rho))
    end
    return (t=times, front=fronts, snapshots=snapshots, model=model)
end

"""
    power_law_exponent(t, x, t1, t2) -> Float64

Least-squares slope of log(x) against log(t) over t1 <= t <= t2 (positive x only).
"""
function power_law_exponent(t::AbstractVector{<:Real}, x::AbstractVector{<:Real}, t1::Real, t2::Real)
    s = (t .>= t1) .& (t .<= t2) .& (x .> 0)
    lt, lx = log.(t[s]), log.(x[s])
    return sum((lt .- sum(lt) / length(lt)) .* (lx .- sum(lx) / length(lx))) / sum((lt .- sum(lt) / length(lt)) .^ 2)
end

# ---------------------------------------------------------------------------------------------------------------------
# Independent references for flexible stems (shooting on the continuous elastica equations)
# ---------------------------------------------------------------------------------------------------------------------
#
# Continuous elastica of an inextensible stem (arc length s from the base, angle θ from the vertical, positive
# downstream), clamped at the base and free at the tip, under the horizontal drag per unit length
# f_x = 1/2 rho C_D b u|u| cos³θ and the vertical load per unit length f_z = -f_x tan θ + B_l (normal drag and net
# buoyancy):
#     θ' = T / EI,   T' = -cos θ F_x + sin θ F_z,   F_x' = -f_x,   F_z' = -f_z,   x' = sin θ,   z' = cos θ,
# with T the moment and (F_x, F_z) the resultant of the loads beyond s, and θ(0) = 0, T(l) = F_x(l) = F_z(l) = 0. The
# equations are integrated from the tip to the base with classical Runge-Kutta (4th order). This is a different method
# from the model (src/vegetation.jl: discrete segments, cell-split loads, Newton on the segment angles); it serves as an
# independent reference.

# One RK4 integration from the tip (s = l) to the base: state (θ, T, F_x, F_z, z); returns θ and z at the base, and the
# path (s, θ) on the integration grid. `u_of_z(z)` gives the horizontal velocity (m/s) at the height z above the base.
function _elastica_from_tip(theta_tip, z_tip; length, b, cd, ei, buoyancy, rho, u_of_z, steps)
    h = -length / steps
    function rhs(y)
        th, tq, fx, fz, z = y
        u = u_of_z(z)
        c, s = cos(th), sin(th)
        load_x = 0.5 * rho * cd * b * u * abs(u) * c^3
        load_z = -load_x * (c > 0 ? s / c : 0.0) + buoyancy
        return (tq / ei, -c * fx + s * fz, -load_x, -load_z, c)
    end
    y = (theta_tip, 0.0, 0.0, 0.0, z_tip)
    thetas = Vector{Float64}(undef, steps + 1)
    thetas[1] = theta_tip
    for i in 1:steps
        k1 = rhs(y)
        k2 = rhs(y .+ (h / 2) .* k1)
        k3 = rhs(y .+ (h / 2) .* k2)
        k4 = rhs(y .+ h .* k3)
        y = y .+ (h / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
        thetas[i + 1] = y[1]
    end
    return y, reverse(thetas)          # base state; θ from base to tip
end

"""
    elastica_reference_uniform(cauchy; buoyancy_number=0.0, steps=20000) -> NamedTuple (R, tip_x, tip_z, theta)

Independent reference for a stem of unit length in a uniform current at the Cauchy number Ca = rho C_D b U² l³/(2 EI)
and the buoyancy number B = B_l l³ / EI (both dimensionless): reconfiguration number R = F / (1/2 rho C_D b l U²), tip
position (tip_x, tip_z) relative to the base in units of l, and θ(s) on `steps` + 1 points from the base to the tip. In a
uniform current the loads depend on θ only, so the equations are integrated from the tip with the tip angle as the only
unknown, found by a downward scan from pi/2 and bisection on θ(0) = 0.
"""
function elastica_reference_uniform(cauchy::Real; buoyancy_number::Real=0.0, steps::Int=20000)
    b, cd, l, ei, rho = 0.01, 1.0, 1.0, 1.0, VEG_RHO0
    u = sqrt(2cauchy * ei / (rho * cd * b * l^3))
    base(theta_tip) = first(first(_elastica_from_tip(theta_tip, 0.0; length=l, b=b, cd=cd, ei=ei, buoyancy=buoyancy_number * ei / l^3,
        rho=rho, u_of_z=z -> u, steps=steps)))
    # WHY the downward scan: θ(0) as a function of θ_tip has further, unphysical roots at smaller tip angles, where the
    # integrated angle winds past -pi/2. The physical posture is the first root below θ_tip = pi/2, where θ(0) is
    # positive. Range: checked against the model (second-order convergence) for Ca <= 10^4 at B <= 10 and Ca <= 3162 at
    # B = 100. At large Ca with strong buoyancy the shooting from the tip becomes badly conditioned (θ(0) jumps by about
    # 8 rad within 0.02 rad of the tip angle at Ca = 10^4, B = 100) and the scan finds an unphysical root.
    hi = 0.5pi - 1e-9
    lo = hi
    f_lo = base(lo)
    for j in 1:2000
        lo = hi * (1 - j / 2000)
        f_lo = base(lo)
        f_lo < 0 && break
    end
    for _ in 1:100
        mid = (lo + hi) / 2
        f_mid = base(mid)
        if sign(f_mid) == sign(f_lo)
            lo, f_lo = mid, f_mid
        else
            hi = mid
        end
        hi - lo < 1e-14 && break
    end
    y, theta = _elastica_from_tip((lo + hi) / 2, 0.0; length=l, b=b, cd=cd, ei=ei, buoyancy=buoyancy_number * ei / l^3, rho=rho,
        u_of_z=z -> u, steps=steps)
    ds = l / steps
    # tip position relative to the base (trapezoidal rule on the RK4 grid, error O(ds^2))
    tip_x = ds * (sum(sin, theta) - 0.5 * (sin(theta[1]) + sin(theta[end])))
    tip_z = ds * (sum(cos, theta) - 0.5 * (cos(theta[1]) + cos(theta[end])))
    return (R=y[3] / (0.5 * rho * cd * b * l * u^2), tip_x=tip_x, tip_z=tip_z, theta=theta)
end

"""
    flexible_canopy_reference(; H, stem_length, a, b, cd, ei, buoyancy=0.0, G, nu, M=2000, steps=4000, tol=1e-12)
        -> NamedTuple (z, u, theta, s_z, tip_x, tip_z)

Independent steady 1D reference of a flexible canopy (stem length `stem_length`, frontal area a = n b, stem width b,
C_D, bending stiffness EI, net buoyancy per length) in a periodic channel of depth H driven by the body force G, with
constant viscosity nu, free-slip bed and stress-free surface:

    nu u'' = 1/2 C_D a cos²θ(z) phi(z) u|u| - G,

with θ(z) the stem angle at the height z and phi the fraction of the cell below the tip height. The stem posture is the
continuous elastica above (RK4 from the tip, `steps` steps) in the velocity along the stem u(z(s)); the outer iteration
alternates posture (with z(s) from the previous iterate, so that the shooting keeps one unknown, the tip angle) and the
velocity profile (Newton on M cells) until both change by less than `tol`. Returns the cell centres z (m), u (m/s), θ(s)
from the base to the tip and the heights z(s) (m), and the tip position (m).
"""
function flexible_canopy_reference(; H, stem_length, a, b, cd, ei, buoyancy=0.0, G, nu, M=2000, steps=4000, tol=1e-12)
    dz = H / M
    zc = [(k - 0.5) * dz for k in 1:M]
    ds = stem_length / steps
    s_grid = [(j - 1) * ds for j in 1:(steps + 1)]
    z_of_s = copy(s_grid)                          # start: upright stem
    theta = zeros(steps + 1)
    _, u = canopy_reference_profile(; H=H, hv=stem_length, a=a, cd=cd, G=G, nu=nu, M=M)   # rigid start
    theta_tip_prev = 0.0
    for outer in 1:200
        # posture in the velocity u(z(s)) of the previous iterate (θ_tip by downward scan + bisection, as above)
        u_along(s) = (z = linear_interpolation(s, s_grid, z_of_s); linear_interpolation(clamp(z, zc[1], zc[end]), zc, u))
        function base_theta(theta_tip)
            h = -ds
            y = (theta_tip, 0.0, 0.0, 0.0)
            path = Vector{Float64}(undef, steps + 1); path[steps + 1] = theta_tip
            rhs(y, s) = (c = cos(y[1]); sn = sin(y[1]); v = u_along(s); lx = 0.5 * VEG_RHO0 * cd * b * v * abs(v) * c^3;
                lz = -lx * (c > 0 ? sn / c : 0.0) + buoyancy; (y[2] / ei, -c * y[3] + sn * y[4], -lx, -lz))
            for i in steps:-1:1
                s = s_grid[i + 1]
                k1 = rhs(y, s); k2 = rhs(y .+ (h / 2) .* k1, s + h / 2); k3 = rhs(y .+ (h / 2) .* k2, s + h / 2); k4 = rhs(y .+ h .* k3, s + h)
                y = y .+ (h / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
                path[i] = y[1]
            end
            return y[1], path
        end
        # WHY the start just above the previous tip angle: the posture changes little between outer iterations, and
        # scanning from pi/2 in every iteration dominated the cost; the full scan is the fallback.
        hi = 0.5pi - 1e-9
        if outer > 1 && first(base_theta(min(theta_tip_prev + 0.02, hi))) > 0
            hi = min(theta_tip_prev + 0.02, hi)
        end
        lo = hi; f_lo = first(base_theta(lo))
        for j in 1:4000
            lo = max(hi - j * 5e-4, 0.0); f_lo = first(base_theta(lo)); (f_lo < 0 || lo == 0.0) && break
        end
        for _ in 1:100
            mid = (lo + hi) / 2; f_mid = first(base_theta(mid))
            if sign(f_mid) == sign(f_lo); lo, f_lo = mid, f_mid; else hi = mid; end
            hi - lo < 1e-15 && break
        end
        theta_tip_prev = (lo + hi) / 2
        theta_new = last(base_theta(theta_tip_prev))
        z_new = [0.0; cumsum(ds .* (cos.(theta_new[1:end-1]) .+ cos.(theta_new[2:end])) ./ 2)]
        # canopy drag coefficient per cell: 1/2 C_D a <cos²θ> over the part of the cell below the tip
        coef = zeros(M)
        for k in 1:M
            lo_z, hi_z = (k - 1) * dz, min(k * dz, z_new[end])
            hi_z > lo_z || continue
            zm = (lo_z + hi_z) / 2
            th = linear_interpolation(linear_interpolation(zm, z_new, s_grid), s_grid, theta_new)
            coef[k] = 0.5 * cd * a * cos(th)^2 * (hi_z - lo_z) / dz
        end
        # velocity profile for this drag (Newton, tridiagonal)
        u_new = copy(u)
        for _ in 1:100
            diag = zeros(M); lower = zeros(M - 1); upper = zeros(M - 1); r = zeros(M)
            for k in 1:M
                if k > 1; r[k] += nu * (u_new[k - 1] - u_new[k]) / dz^2; lower[k - 1] = nu / dz^2; diag[k] -= nu / dz^2; end
                if k < M; r[k] += nu * (u_new[k + 1] - u_new[k]) / dz^2; upper[k] = nu / dz^2; diag[k] -= nu / dz^2; end
                r[k] += G - coef[k] * u_new[k] * abs(u_new[k])
                diag[k] -= 2 * coef[k] * abs(u_new[k])
            end
            du = -(Tridiagonal(lower, diag, upper) \ r)
            u_new .+= du
            maximum(abs, du) < 1e-14 && break
        end
        change = max(maximum(abs, u_new .- u), maximum(abs, theta_new .- theta))
        u .= 0.5 .* (u .+ u_new)                 # WHY under-relaxation: the posture-velocity iteration can overshoot
        theta .= theta_new; z_of_s .= z_new
        change < tol && break
    end
    tip_x = ds * (sum(sin, theta) - 0.5 * (sin(theta[1]) + sin(theta[end])))
    return (z=zc, u=u, theta=theta, s_z=z_of_s, tip_x=tip_x, tip_z=z_of_s[end])
end
