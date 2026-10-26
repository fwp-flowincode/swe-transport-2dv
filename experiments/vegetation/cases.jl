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

# ---------------------------------------------------------------------------------------------------------------------
# Laboratory data: Luhar & Nepf (2011), model blades in uniform flow (Limnol. Oceanogr. 56(6), 2003-2017)
# ---------------------------------------------------------------------------------------------------------------------
#
# Flat blades of width b = 1 cm, lengths 5-25 cm, clamped at the base in a uniform current U = 3.6-32 cm/s (their
# Table 1); drag coefficient of a flat plate normal to the flow C_D = 1.95 (Vogel 1994), as in their model. Two
# materials (their Table 1): HDPE (E = 0.93 GPa, blade density 50 kg/m^3 below water, t = 0.4 mm; stiffness-dominated)
# and silicone foam (E = 500 kPa, 330 kg/m^3 below water, t = 1.9 mm; buoyancy-dominated). Bending stiffness
# EI = E b t³ / 12, net buoyancy per length B_l = Δρ g b t.
# Measured horizontal forces of the 5 cm and 25 cm blades: read from their Fig. 4 (rendered at 400 dpi; reading
# accuracy about 0.3 mN; load-cell resolution 1 mN and accuracy 10 % according to the paper). NaN: symbol hidden.

const LN2011_RHO = 1000.0                      # kg/m^3, flume water
const LN2011_CD = 1.95                         # -, flat plate normal to the flow (Vogel 1994), as in Luhar & Nepf
const LN2011_WIDTH = 0.01                      # m, blade width b
const LN2011_MATERIALS = Dict(
    "HDPE" => (E=0.93e9, drho=50.0, t=0.4e-3),     # Pa, kg/m^3, m
    "foam" => (E=500e3, drho=330.0, t=1.9e-3),
)
const LN2011_LENGTHS = (0.05, 0.10, 0.15, 0.20, 0.25)                        # m
const LN2011_VELOCITIES = (0.036, 0.071, 0.11, 0.14, 0.16, 0.22, 0.27, 0.32)  # m/s
const LN2011_FORCES = Dict(                                                  # N, from Fig. 4
    ("HDPE", 0.05) => [0.0005, 0.0020, 0.0043, 0.0075, 0.0093, 0.0163, 0.0256, 0.0360],
    ("HDPE", 0.25) => [0.0017, 0.0039, 0.0061, 0.0089, 0.0112, 0.0173, 0.0242, 0.0307],
    ("foam", 0.05) => [0.00049, 0.00185, 0.00377, 0.00664, 0.0077, 0.0105, NaN, 0.01566],
    ("foam", 0.25) => [0.00385, 0.00747, 0.00894, 0.01023, 0.01158, 0.01208, 0.01336, 0.01517],
)

"""
    luhar_nepf_blade(material, length, velocity; n_segments=80) -> NamedTuple (force, effective_length, cauchy, buoyancy, theta)

Our stem model (update_stem_posture!) for a Luhar & Nepf (2011) model blade of `material` ("HDPE" or "foam") and
`length` (m) in the uniform current `velocity` (m/s): horizontal drag (N), effective length l_e/l = F / (1/2 rho C_D b l U²),
Cauchy number Ca = 1/2 rho C_D b U² l³ / EI and buoyancy parameter B = Δρ g b t l³ / EI (their Eqs. 7-9), and the posture.
"""
function luhar_nepf_blade(material::AbstractString, length::Real, velocity::Real; n_segments::Int=80)
    m = LN2011_MATERIALS[material]
    ei = m.E * LN2011_WIDTH * m.t^3 / 12
    buoyancy = m.drho * 9.81 * LN2011_WIDTH * m.t
    theta = zeros(n_segments)
    _, force, _ = update_stem_posture!(theta, (k, x) -> velocity; length=length, diameter=LN2011_WIDTH, drag_coefficient=LN2011_CD,
        bending_stiffness=ei, net_buoyancy=buoyancy, rho0=LN2011_RHO, dz=1e-3, nz=1000, water_level=0.3)
    rigid = 0.5 * LN2011_RHO * LN2011_CD * LN2011_WIDTH * length * velocity^2
    return (force=force, effective_length=force / rigid, cauchy=rigid * length^2 / ei, buoyancy=buoyancy * length^3 / ei, theta=theta)
end

# ---------------------------------------------------------------------------------------------------------------------
# Laboratory data: Dunn, Lopez & Garcia (1996), Hydraulic Engineering Series 51, University of Illinois
# ---------------------------------------------------------------------------------------------------------------------
#
# Uniform flow through submerged cylinder canopies in a 19.5 m long, 0.91 m wide tilting flume (section 4.3.1):
# rigid wooden dowels (D = 6.35 mm, mean height in the flow h_p = 11.8 +- 1.67 cm) and flexible drinking straws
# (D = 6.35 mm, undeformed height 16.9 +- 1.61 cm, bent at a corrugated hinge near the bed; flexural rigidity not
# measured; the straws became more flexible over the runs and swayed strongly, section 5.4), staggered patterns. Columns
# from Table 4.1 (Q, S, spacing), Tables 5.1/5.2 (bulk drag coefficient C_DB), Tables 5.3/5.4 (D/H, a H, h_p/H) and
# Table 5.5 (flexible: canopy velocity U_c, deflection angle, deflected height). Derived: depth H = D/(D/H), frontal area
# a = (aH)/H, h_p = (h_p/H) H, mean velocity U = Q / (0.91 m H). Horizontally averaged profiles from Appendix C
# (DUNN1996_PROFILES).

const DUNN1996_D = 0.00635                     # m, cylinder diameter
const DUNN1996_WIDTH = 0.91                    # m, flume width
# run => (Q (L/s), S (-), spacing (cm), C_DB (-), D/H (-), aH (-), h_p/H (-))
const DUNN1996_RIGID = Dict(
    1 => (179, 0.0036, 7.62, 1.01, 0.0190, 0.365, 0.351), 2 => (88, 0.0036, 7.62, 0.95, 0.0277, 0.250, 0.518),
    3 => (46, 0.0036, 7.62, 0.86, 0.0387, 0.179, 0.714), 4 => (178, 0.0076, 7.62, 1.29, 0.0230, 0.301, 0.426),
    5 => (98, 0.0076, 7.62, 1.18, 0.0313, 0.221, 0.578), 6 => (178, 0.0036, 15.24, 1.46, 0.0238, 0.073, 0.441),
    7 => (95, 0.0036, 15.24, 1.39, 0.0347, 0.050, 0.641), 8 => (180, 0.0036, 5.08, 0.94, 0.0162, 0.962, 0.300),
    9 => (58, 0.0036, 5.08, 1.13, 0.0297, 0.526, 0.549), 10 => (180, 0.0161, 5.08, 1.19, 0.0240, 0.652, 0.442),
    11 => (177, 0.0036, 10.16, 1.06, 0.0204, 0.191, 0.377), 12 => (181, 0.0110, 10.16, 1.14, 0.0273, 0.143, 0.505),
)
# run => (Q (L/s), S (-), spacing (cm), C_DB (-), D/H (-), aH (-), h_p/H (-), U_c (cm/s), deflection angle (deg), h_p (cm))
const DUNN1996_FLEXIBLE = Dict(
    13 => (179, 0.0036, 7.62, 1.13, 0.0173, 0.401, 0.413, 32.8, 35, 15.2), 14 => (180, 0.0101, 7.62, 0.33, 0.0274, 0.253, 0.495, 72.2, 51, 11.5),
    15 => (93, 0.0036, 7.62, 1.45, 0.0247, 0.280, 0.513, 25.6, 34, 13.2), 16 => (179, 0.0036, 15.24, 0.55, 0.0276, 0.063, 0.422, 73.4, 65, 9.7),
    17 => (78, 0.0036, 5.08, 1.19, 0.0228, 0.686, 0.578, 18.0, 12, 16.1), 18 => (179, 0.0101, 5.08, 0.59, 0.0224, 0.699, 0.426, 44.4, 45, 12.1),
)
# Appendix C (pp. 143-148): horizontally averaged profiles (four verticals at the flume centre, 0.16 m wide) of the
# streamwise velocity u_h (m/s), the Reynolds stress -<u'w'>_h (m²/s²) and the local drag coefficient C_D' (-, NaN
# where the report gives none, above the canopy) at the heights z (m above the bed), top to bottom as in the report.
# Run 16, third row: the report prints z = 1.1000 m, read as 0.1000 m (the depth is 0.230 m).
const DUNN1996_PROFILES = Dict(
    1 => (z=[0.2390, 0.2003, 0.1704, 0.1504, 0.1207, 0.1001, 0.0806, 0.0552, 0.0306, 0.0107],
          u=[0.7394, 0.6805, 0.6459, 0.6144, 0.5353, 0.4632, 0.4032, 0.3560, 0.3159, 0.2932],
          uw=[2.12e-3, 3.81e-3, 4.40e-3, 5.49e-3, 6.29e-3, 5.24e-3, 3.91e-3, 2.38e-3, 1.35e-3, 7.50e-4],
          cd=[NaN, NaN, NaN, NaN, 0.55, 0.82, 1.10, 1.31, 1.32, 1.03]),
    2 => (z=[0.1562, 0.1397, 0.1203, 0.1102, 0.1000, 0.0803, 0.0651, 0.0500, 0.0302, 0.0102],
          u=[0.5296, 0.5098, 0.4649, 0.4317, 0.3914, 0.3476, 0.3259, 0.3008, 0.2673, 0.2577],
          uw=[1.18e-3, 1.82e-3, 2.55e-3, 2.74e-3, 2.24e-3, 1.53e-3, 1.33e-3, 8.60e-4, 3.65e-4, 2.08e-4],
          cd=[NaN, NaN, NaN, 0.75, 0.87, 1.00, 1.06, 1.15, 1.33, 1.32]),
    3 => (z=[0.0928, 0.0876, 0.0796, 0.0696, 0.0597, 0.0499, 0.0422, 0.0328, 0.0174, 0.0071],
          u=[0.3015, 0.2881, 0.2742, 0.2626, 0.2453, 0.2320, 0.2268, 0.2139, 0.2163, 0.2101],
          uw=[3.14e-4, 2.94e-4, 2.72e-4, 2.95e-4, 2.16e-4, 1.13e-4, 7.32e-5, 8.43e-5, 3.18e-5, 5.39e-5],
          cd=[0.68, 0.79, 0.93, 1.07, 1.26, 1.41, 1.45, 1.57, 1.35, 1.25]),
    4 => (z=[0.1916, 0.1801, 0.1651, 0.1400, 0.1200, 0.0997, 0.0801, 0.0597, 0.0376, 0.0100],
          u=[0.7401, 0.7366, 0.7144, 0.6588, 0.5891, 0.5199, 0.4471, 0.4000, 0.3599, 0.3310],
          uw=[4.14e-3, 4.44e-3, 5.61e-3, 6.90e-3, 7.36e-3, 6.59e-3, 4.27e-3, 2.38e-3, 1.27e-3, 4.77e-4],
          cd=[NaN, NaN, NaN, NaN, 0.61, 1.04, 1.55, 1.86, 1.81, 0.78]),
    5 => (z=[0.1266, 0.1200, 0.1099, 0.0998, 0.0800, 0.0654, 0.0502, 0.0375, 0.0173, 0.0075],
          u=[0.5979, 0.5782, 0.5270, 0.4793, 0.4220, 0.3986, 0.3710, 0.3490, 0.3265, 0.3189],
          uw=[3.73e-3, 4.03e-3, 3.74e-3, 2.96e-3, 1.85e-3, 1.27e-3, 9.23e-4, 5.83e-4, 2.86e-4, 2.10e-4],
          cd=[NaN, 0.77, 0.89, 1.04, 1.23, 1.29, 1.37, 1.43, 1.40, 1.35]),
    6 => (z=[0.1785, 0.1596, 0.1400, 0.1201, 0.1047, 0.0901, 0.0702, 0.0500, 0.0096],
          u=[0.7249, 0.6982, 0.6535, 0.6063, 0.5779, 0.5480, 0.5192, 0.4925, 0.4472],
          uw=[2.22e-3, 2.73e-3, 2.90e-3, 2.78e-3, 2.59e-3, 2.01e-3, 1.53e-3, 9.29e-4, 5.09e-4],
          cd=[NaN, NaN, NaN, 1.00, 1.31, 1.58, 1.79, 1.81, 0.99]),
    7 => (z=[0.1147, 0.1049, 0.0948, 0.0849, 0.0702, 0.0602, 0.0499, 0.0375, 0.0176, 0.0074],
          u=[0.5869, 0.5563, 0.5423, 0.5214, 0.4972, 0.4837, 0.4745, 0.4633, 0.4385, 0.4248],
          uw=[1.75e-3, 1.61e-3, 1.37e-3, 9.50e-4, 6.68e-4, 4.42e-4, 5.43e-4, 5.41e-4, 4.61e-4, 3.98e-4],
          cd=[1.53, 1.58, 1.53, 1.53, 1.49, 1.45, 1.39, 1.32, 1.28, 1.29]),
    8 => (z=[0.3188, 0.2798, 0.2397, 0.2002, 0.1597, 0.1200, 0.0898, 0.0600, 0.0355, 0.0102],
          u=[0.7627, 0.7436, 0.7099, 0.6640, 0.5782, 0.4409, 0.3249, 0.2499, 0.2104, 0.1781],
          uw=[6.29e-4, 1.57e-3, 2.34e-3, 3.70e-3, 5.33e-3, 7.22e-3, 4.79e-3, 2.15e-3, 9.81e-4, 3.85e-4],
          cd=[NaN, NaN, NaN, NaN, NaN, 0.47, 0.94, 1.39, 1.40, 0.72]),
    9 => (z=[0.1435, 0.1299, 0.1200, 0.1053, 0.0898, 0.0697, 0.0495, 0.0358, 0.0174, 0.0074],
          u=[0.3402, 0.3130, 0.2915, 0.2462, 0.2151, 0.1882, 0.1644, 0.1525, 0.1449, 0.1425],
          uw=[1.64e-3, 1.87e-3, 1.96e-3, 1.59e-3, 1.20e-3, 6.41e-4, 3.23e-4, 1.39e-4, 8.60e-5, 5.30e-5],
          cd=[NaN, NaN, 0.59, 0.83, 1.07, 1.31, 1.52, 1.57, 1.37, 1.17]),
    10 => (z=[0.1678, 0.1499, 0.1350, 0.1200, 0.1049, 0.0901, 0.0702, 0.0550, 0.0375, 0.0098],
           u=[0.8280, 0.8022, 0.7255, 0.6422, 0.5601, 0.4755, 0.4057, 0.3636, 0.3296, 0.3037],
           uw=[1.11e-2, 1.17e-2, 1.42e-2, 1.30e-2, 1.14e-2, 7.97e-3, 5.23e-3, 2.90e-3, 1.61e-3, 4.29e-4],
           cd=[NaN, NaN, NaN, 0.60, 0.82, 1.15, 1.50, 1.71, 1.75, 1.14]),
    11 => (z=[0.2285, 0.2000, 0.1699, 0.1404, 0.1200, 0.1002, 0.0801, 0.0550, 0.0379, 0.0102],
           u=[0.7395, 0.7064, 0.6744, 0.6185, 0.5849, 0.5263, 0.4865, 0.4283, 0.4039, 0.3733],
           uw=[1.83e-3, 2.64e-3, 3.52e-3, 4.33e-3, 4.63e-3, 4.20e-3, 3.57e-3, 2.07e-3, 1.33e-3, 5.81e-4],
           cd=[NaN, NaN, NaN, NaN, 0.33, 0.78, 1.16, 1.56, 1.55, 0.96]),
    12 => (z=[0.1203, 0.1051, 0.0950, 0.0804, 0.0703, 0.0603, 0.0476, 0.0379, 0.0179, 0.0075],
           u=[0.8541, 0.7812, 0.7444, 0.6891, 0.6534, 0.6324, 0.5959, 0.5834, 0.5565, 0.5634],
           uw=[6.23e-3, 6.34e-3, 5.64e-3, 4.54e-3, 3.76e-3, 2.99e-3, 2.33e-3, 2.05e-3, 1.25e-3, 1.00e-3],
           cd=[NaN, NaN, 1.02, 1.10, 1.23, 1.33, 1.37, 1.46, 1.43, 1.30]),
    13 => (z=[0.2874, 0.2500, 0.2002, 0.1596, 0.1399, 0.1200, 0.0900, 0.0649, 0.0377, 0.0099],
           u=[0.7174, 0.6714, 0.5986, 0.5013, 0.4366, 0.3760, 0.3284, 0.3052, 0.2761, 0.2545],
           uw=[1.82e-3, 2.98e-3, 4.42e-3, 5.20e-3, 5.04e-3, 3.67e-3, 1.70e-3, 9.49e-4, 4.09e-4, 1.52e-4],
           cd=[NaN, NaN, NaN, NaN, 1.18, 1.36, 1.37, 1.25, 1.17, 1.05]),
    14 => (z=[0.1187, 0.1100, 0.0948, 0.0798, 0.0649, 0.0497, 0.0376, 0.0176, 0.0076],
           u=[1.0164, 0.9978, 0.9336, 0.8725, 0.8297, 0.7638, 0.6825, 0.5669, 0.5312],
           uw=[2.71e-3, 2.85e-3, 3.45e-3, 3.36e-3, 2.53e-3, 2.44e-3, 2.51e-3, 1.51e-3, 9.61e-4],
           cd=[NaN, 0.14, 0.18, 0.26, 0.32, 0.40, 0.53, 0.79, 0.90]),
    15 => (z=[0.1845, 0.1648, 0.1500, 0.1296, 0.1100, 0.0901, 0.0723, 0.0553, 0.0359, 0.0102],
           u=[0.4661, 0.4266, 0.3905, 0.3181, 0.2669, 0.2383, 0.2507, 0.2535, 0.2469, 0.2422],
           uw=[1.79e-3, 2.21e-3, 2.38e-3, 2.30e-3, 1.42e-3, 4.38e-4, 2.09e-4, 1.36e-4, 1.38e-4, 7.25e-5],
           cd=[NaN, NaN, NaN, 1.90, 2.06, 1.95, 1.38, 1.12, 1.05, 1.17]),
    16 => (z=[0.1186, 0.1098, 0.1000, 0.0852, 0.0699, 0.0601, 0.0501, 0.0377, 0.0175, 0.0074],
           u=[0.8905, 0.8830, 0.8614, 0.8295, 0.7990, 0.7705, 0.7430, 0.7016, 0.6622, 0.6438],
           uw=[1.39e-3, 1.42e-3, 1.48e-3, 1.40e-3, 1.54e-3, 1.33e-3, 1.29e-3, 1.20e-3, 1.01e-3, 8.17e-4],
           cd=[NaN, NaN, 0.34, 0.39, 0.45, 0.50, 0.57, 0.67, 0.81, 0.89]),
    17 => (z=[0.2045, 0.1801, 0.1649, 0.1550, 0.1402, 0.1149, 0.0901, 0.0648, 0.0401, 0.0101],
           u=[0.4099, 0.3494, 0.2984, 0.2769, 0.2441, 0.1711, 0.1599, 0.1555, 0.1550, 0.1563],
           uw=[2.01e-3, 2.81e-3, 2.59e-3, 2.08e-3, 1.73e-3, 6.22e-4, 2.32e-4, 5.33e-5, 1.63e-5, 4.32e-5],
           cd=[NaN, NaN, 0.80, 0.86, 0.99, 1.66, 1.57, 1.37, 1.18, 1.00]),
    18 => (z=[0.1752, 0.1603, 0.1450, 0.1350, 0.1201, 0.0998, 0.0799, 0.0573, 0.0374, 0.0099],
           u=[0.9152, 0.8791, 0.8301, 0.7990, 0.6937, 0.5687, 0.4795, 0.4151, 0.3654, 0.2968],
           uw=[4.11e-3, 4.69e-3, 6.01e-3, 6.56e-3, 7.98e-3, 7.59e-3, 3.95e-3, 1.66e-3, 1.23e-3, 3.79e-4],
           cd=[NaN, NaN, NaN, NaN, 0.26, 0.49, 0.76, 0.92, 0.92, 0.38]),
)

"""
    dunn1996_run(run) -> NamedTuple (H, a, hp, S, U, cdb)

Derived quantities of run `run` of Dunn et al. (1996), rigid (1-12) or flexible (13-18): depth H (m), frontal area
a (1/m), canopy height h_p (m; flexible runs: the deflected height of Table 5.5), bed slope S (-), mean velocity
U = Q / (0.91 m H) (m/s) and the measured bulk drag coefficient C_DB (-).
"""
function dunn1996_run(run::Int)
    if haskey(DUNN1996_FLEXIBLE, run)
        # WHY: Lopez & Garcia (1997) treat the bent straws as a rigid canopy of the deflected height (Table 5.5).
        q, s, _, cdb, d_over_h, ah, _, _, _, hp_cm = DUNN1996_FLEXIBLE[run]
        h = DUNN1996_D / d_over_h
        return (H=h, a=ah / h, hp=hp_cm / 100, S=s, U=q / 1000 / (DUNN1996_WIDTH * h), cdb=cdb)
    end
    q, s, _, cdb, d_over_h, ah, hp_over_h = DUNN1996_RIGID[run]
    h = DUNN1996_D / d_over_h
    return (H=h, a=ah / h, hp=hp_over_h * h, S=s, U=q / 1000 / (DUNN1996_WIDTH * h), cdb=cdb)
end

const CANOPY_KE_NU_T_SCALE = 1.5 * 0.41 / 4    # -, nu_t <= this * u* H (1.5 times the parabolic maximum) for the dt limit

"""
    canopy_channel_k_epsilon(; depth, slope, canopy_height, frontal_area, stem_diameter, drag_coefficient,
                             bed_roughness, nz=40, t_end=400.0) -> NamedTuple

Uniform open-channel flow of depth `depth` (m) and bed slope `slope` (-) through a homogeneous rigid canopy (height
`canopy_height` m, frontal area per volume `frontal_area` 1/m, stems of `stem_diameter` m, `drag_coefficient`) with the
k-epsilon closure (vegetation terms of Lopez & Garcia 1997) over a bed of roughness length `bed_roughness` (m), run in a
periodic channel to `t_end` (s; steady within 0.6 % after 400 s for the Dunn runs). Returns the cell-centre heights `z`
(m), velocity `u` (m/s), the w-face heights `zf` (m), the total shear stress `tau` = (nu + nu_t) du/dz at the faces
(m²/s²), the bed shear velocity `ustar_bed` (m/s), the discharge per width `q` (m²/s) and the model.
"""
function canopy_channel_k_epsilon(; depth::Real, slope::Real, canopy_height::Real, frontal_area::Real, stem_diameter::Real,
    drag_coefficient::Real, bed_roughness::Real, nz::Int=40, t_end::Real=400.0, nu::Real=1e-6)
    dz = depth / nz
    # WHY: momentum diffusion is explicit; dt <= 0.2 dz²/nu_t keeps it stable with margin
    nu_t_max = CANOPY_KE_NU_T_SCALE * sqrt(9.81 * slope * depth) * depth
    dt = min(0.01, 0.2 * dz^2 / nu_t_max)
    cfg = vegetation_case_config(length=1.0, depth=depth, nx=4, nz=nz, dt=dt, nt=round(Int, t_end / dt), use_hydrostatic=true,
        periodic_x=true, variable_top_layer=false, pressure_solver_method="sparse", nu=nu, body_force_x=9.81 * slope,
        turbulence_closure="k_epsilon", bed_roughness_length=bed_roughness)
    model = uniform_density_model(cfg)
    attach_vegetation!(model, VegetationModel(; patches=[VegetationPatch(x_start=0.0, x_end=1.0, stem_height=canopy_height,
        stem_diameter=stem_diameter, stem_density=frontal_area / stem_diameter, drag_coefficient=drag_coefficient)]))
    for _ in 1:cfg.nt
        step_forward!(model)
    end
    u = vec(sum(model.state.u[:, 1:end-1]; dims=2)) ./ cfg.nx
    nu_t = vec(sum(model.turbulence.nu_t; dims=2)) ./ cfg.nx
    z = [(k - 0.5) * dz for k in 1:nz]
    zf = [k * dz for k in 1:nz-1]
    tau = [(nu + 0.5 * (nu_t[k] + nu_t[k + 1])) * (u[k + 1] - u[k]) / dz for k in 1:nz-1]
    ustar_bed = sqrt(bed_friction_coefficient(bed_roughness, dz / 2)) * abs(u[1])
    return (z=z, u=u, zf=zf, tau=tau, ustar_bed=ustar_bed, q=sum(u) * dz, model=model)
end

const DUNN1996_BED_ROUGHNESS = 5e-6             # m, smooth flume bed (Strickler k_s = (21.1 n)^6, n = 0.011, z0 = k_s/30)

"""
    dunn1996_k_epsilon(run; drag_coefficient=:measured, nz=40, t_end=400.0) -> NamedTuple

`canopy_channel_k_epsilon` for run `run` of Dunn et al. (1996) (`dunn1996_run`), with the measured bulk drag
coefficient of the run (`:measured`) or a given value; the run data are added as field `run`.
"""
function dunn1996_k_epsilon(run::Int; drag_coefficient=:measured, nz::Int=40, t_end::Real=400.0)
    d = dunn1996_run(run)
    cd = drag_coefficient === :measured ? d.cdb : drag_coefficient
    r = canopy_channel_k_epsilon(; depth=d.H, slope=d.S, canopy_height=d.hp, frontal_area=d.a, stem_diameter=DUNN1996_D,
        drag_coefficient=cd, bed_roughness=DUNN1996_BED_ROUGHNESS, nz=nz, t_end=t_end)
    return merge(r, (run=d,))
end

# ---------------------------------------------------------------------------------------------------------------------
# Lopez & Garcia (1997), WES Technical Report WRP-CP-10: k-epsilon results for Dunn et al. (1996)
# ---------------------------------------------------------------------------------------------------------------------
#
# Their computed profiles (thick dotted k-epsilon curves) of the mean velocity u (m/s; Figs. 6.1 rigid, 6.2 flexible) and
# the Reynolds stress -<u'w'> (m²/s²; Figs. 6.4, 6.5) for runs 1, 9 (rigid) and 13, 17 (flexible, computed as rigid
# canopies of the deflected height), at the heights uz and tz (m above the bed). Digitized from the report scanned at
# 250 dpi: plot frames and ticks located in the image (axis calibration within 2 px, i.e. 0.4 % of the axis range),
# the k-epsilon dots separated from the hollow data symbols and thin data lines by size and fill, filled data symbols
# next to the curve removed by monotonicity; about every third dot kept. Reading accuracy about 1 % of the axis range;
# the stress peaks at the canopy top are cut off by the dashed canopy line (peak values about 5 % low).
const LG1997_PROFILES = Dict(
    1 => (uz=[0.022, 0.087, 0.125, 0.141, 0.159, 0.179, 0.200, 0.221, 0.243, 0.265, 0.288, 0.312, 0.320],
          u=[0.334, 0.451, 0.630, 0.704, 0.771, 0.832, 0.886, 0.930, 0.972, 1.005, 1.034, 1.055, 1.061],
          tz=[0.022, 0.051, 0.069, 0.079, 0.092, 0.104, 0.113, 0.131, 0.145, 0.161, 0.176, 0.191, 0.206, 0.222, 0.241, 0.256,
              0.272, 0.287, 0.302, 0.317, 0.322],
          tau=[8.41e-4, 1.60e-3, 2.35e-3, 2.99e-3, 4.32e-3, 6.12e-3, 7.22e-3, 7.21e-3, 6.68e-3, 6.15e-3, 5.62e-3, 5.09e-3,
               4.56e-3, 4.02e-3, 3.32e-3, 2.78e-3, 2.25e-3, 1.72e-3, 1.18e-3, 6.41e-4, 4.76e-4]),
    9 => (uz=[0.024, 0.043, 0.077, 0.108, 0.115, 0.127, 0.137, 0.148, 0.164, 0.176, 0.190, 0.204],
          u=[0.170, 0.182, 0.217, 0.305, 0.352, 0.411, 0.452, 0.490, 0.534, 0.562, 0.586, 0.602],
          tz=[0.014, 0.033, 0.093, 0.097, 0.100, 0.102, 0.104, 0.107, 0.110, 0.114, 0.123, 0.129, 0.136, 0.143, 0.150, 0.157,
              0.164, 0.171, 0.178, 0.185, 0.192, 0.199, 0.206],
          tau=[1.71e-4, 2.34e-4, 1.48e-3, 1.75e-3, 2.02e-3, 2.30e-3, 2.57e-3, 2.84e-3, 3.11e-3, 3.38e-3, 3.23e-3, 2.98e-3,
               2.73e-3, 2.50e-3, 2.25e-3, 2.01e-3, 1.76e-3, 1.52e-3, 1.27e-3, 1.02e-3, 7.76e-4, 5.32e-4, 2.92e-4]),
    13 => (uz=[0.017, 0.040, 0.071, 0.093, 0.114, 0.132, 0.147, 0.172, 0.190, 0.208, 0.227, 0.247, 0.268, 0.289, 0.312, 0.334,
               0.349],
           u=[0.288, 0.308, 0.340, 0.371, 0.412, 0.465, 0.530, 0.639, 0.696, 0.751, 0.799, 0.844, 0.884, 0.918, 0.947, 0.970,
              0.982],
           tz=[0.017, 0.045, 0.076, 0.126, 0.131, 0.136, 0.150, 0.165, 0.180, 0.195, 0.210, 0.225, 0.240, 0.256, 0.271, 0.286,
               0.301, 0.317, 0.332, 0.347, 0.352],
           tau=[6.70e-4, 9.44e-4, 1.67e-3, 5.24e-3, 5.93e-3, 6.61e-3, 7.66e-3, 7.16e-3, 6.64e-3, 6.11e-3, 5.57e-3, 5.03e-3,
                4.49e-3, 3.96e-3, 3.42e-3, 2.89e-3, 2.36e-3, 1.83e-3, 1.29e-3, 7.56e-4, 5.70e-4]),
    17 => (uz=[0.053, 0.076, 0.100, 0.116, 0.146, 0.167, 0.177, 0.189, 0.201, 0.215, 0.230, 0.246, 0.263],
           u=[0.170, 0.180, 0.196, 0.213, 0.272, 0.363, 0.407, 0.450, 0.489, 0.524, 0.556, 0.582, 0.601],
           tz=[0.018, 0.054, 0.071, 0.088, 0.104, 0.147, 0.151, 0.154, 0.158, 0.168, 0.178, 0.188, 0.198, 0.210, 0.221, 0.230,
               0.240, 0.250, 0.260, 0.266],
           tau=[1.27e-4, 1.96e-4, 2.87e-4, 4.36e-4, 6.39e-4, 2.39e-3, 2.79e-3, 3.19e-3, 3.61e-3, 3.93e-3, 3.59e-3, 3.23e-3,
                2.89e-3, 2.42e-3, 2.08e-3, 1.72e-3, 1.39e-3, 1.03e-3, 6.98e-4, 4.61e-4]),
)

# Fig. 6.18: their computed normal depth H (m) and Manning coefficient n (s/m^(1/3)) against the stem density lambda_d
# (stems/m²) for S = 0.0036, h_p = 0.10 m, D = 6.4 mm (a = lambda_d D), q = 1.12 m²/s (smooth bed, C_D not stated;
# 1.13 from their p. 20). Digitized as above (log axes calibrated on the frame and the major ticks). The figures are
# labelled q = 1.21 m²/s, the text (p. 76) says 1.12 m²/s; n = H^(5/3) S^(1/2) / q from the digitized H and n gives
# q = 1.13 +- 0.03 m²/s for all points, so 1.12 m²/s is the value used.
const LG1997_MANNING_S = 0.0036               # -
const LG1997_MANNING_HP = 0.10                # m
const LG1997_MANNING_D = 0.0064               # m
const LG1997_MANNING_Q = 1.12                 # m²/s
const LG1997_MANNING_LAMBDA = [0.162, 0.793, 1.57, 7.90, 15.8, 39.3, 78.7, 118.0, 158.0, 238.0, 789.0, 1581.0]   # 1/m²
const LG1997_MANNING_H = [0.379, 0.386, 0.396, 0.446, 0.480, 0.544, 0.611, 0.625, 0.672, 0.721, 0.764, 0.858]  # m
const LG1997_MANNING_N = [0.0110, 0.0110, 0.0110, 0.0140, 0.0160, 0.0189, 0.0238, 0.0248, 0.0268, 0.0307, 0.0337, 0.0406]
# Fig. 6.16: momentum transfer to the bed, u*_bed / sqrt(g H S), against Ha = H a (computed, the same model). The plotted
# polyline read at Ha of the first nine Fig. 6.18 cases (above Ha = 1 the curve runs into the frame ticks).
const LG1997_BED_SHEAR_HA = [0.000393, 0.00196, 0.00399, 0.0226, 0.0487, 0.137, 0.308, 0.472, 0.680]
const LG1997_BED_SHEAR = [0.981, 0.919, 0.855, 0.598, 0.481, 0.339, 0.245, 0.207, 0.176]
const LG1997_WALL_E = 9.0                     # -, smooth-wall constant of their wall functions (Eq. 28; Appendix A)

"""
    lg1997_manning_case(i; dz=0.01, t_end=300.0) -> NamedTuple

Case `i` (1-12) of the plant-density series of Lopez & Garcia (1997), Figs. 6.16-6.18, run with
`canopy_channel_k_epsilon` at their computed depth H (C_D = 1.13). Smooth bed as in their wall functions: roughness
length z0 = nu / (E u*_bed), E = 9, iterated once from u*_bed = sqrt(g H S) (the second pass changes q by at most
1.3 %). Returns H (m), Ha (-), the discharge per width q (m²/s) at that depth, the Manning coefficient
n = H^(5/3) S^(1/2) / q (s/m^(1/3)) and the bed shear ratio u*_bed / sqrt(g H S) (-).
"""
function lg1997_manning_case(i::Int; dz::Real=0.01, t_end::Real=300.0, nu::Real=1e-6)
    h = LG1997_MANNING_H[i]
    a = LG1997_MANNING_LAMBDA[i] * LG1997_MANNING_D
    nz = max(round(Int, h / dz), 10)
    ustar = sqrt(9.81 * h * LG1997_MANNING_S)
    r = nothing
    for _ in 1:2
        r = canopy_channel_k_epsilon(; depth=h, slope=LG1997_MANNING_S, canopy_height=LG1997_MANNING_HP, frontal_area=a,
            stem_diameter=LG1997_MANNING_D, drag_coefficient=1.13, bed_roughness=nu / (LG1997_WALL_E * ustar), nz=nz, t_end=t_end, nu=nu)
        ustar = r.ustar_bed
    end
    return (H=h, Ha=h * a, q=r.q, n=h^(5 / 3) * sqrt(LG1997_MANNING_S) / r.q, bed_shear=r.ustar_bed / sqrt(9.81 * h * LG1997_MANNING_S))
end
