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
