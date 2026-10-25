# Wind-driven double gyre in a closed, stratified basin (issue #4): settings shared by the scripts in this folder.
#
# A closed 20 m x 4 m basin of uniform depth holds two layers at rest (light water above heavy water, interface at
# mid-depth). A wind stress that builds up linearly over WS_RAMP_S drives the surface water downwind. The closed walls
# force a return flow: a clockwise cell forms in the upper layer, and the interfacial shear stress drives a
# counter-rotating (counter-clockwise) cell in the lower layer, the double gyre.
#
# References:
# - Monismith (1986), J. Fluid Mech. 171, 407-439 (wind-forced motions in stratified lakes)
# - Spigel & Imberger (1980), J. Phys. Oceanogr. 10, 1104-1121 (wind-induced mixing in a two-layer lake)
# - Casulli & Cheng (1992), Int. J. Numer. Methods Fluids 15, 629-648 (surface stress in the semi-implicit scheme)

using SweTransport2DV

const WS_LENGTH = 20.0                # m, basin length
const WS_DEPTH = 4.0                  # m, still-water depth
const WS_RHO_TOP = 999.0              # kg/m^3, upper (light) layer
const WS_RHO_BOTTOM = 1001.0          # kg/m^3, lower (heavy) layer
const WS_INTERFACE_HEIGHT_FRACTION = 0.5   # -, interface at mid-depth
const WS_WIND_STRESS_X = 0.2          # Pa, wind stress after the ramp (about a 12 m/s wind, Large & Pond 1981)
const WS_RAMP_S = 60.0                # s, linear build-up of the wind stress from zero
const WS_MIXING_LENGTH = 0.2          # m, mixing length of the eddy viscosity nu_t = l^2 |du/dz|
const WS_MAX_EDDY_VISCOSITY = 5e-3    # m^2/s, cap of nu_t

"""
    wind_stress_at(t) -> Float64

Wind stress (Pa) at time `t` (s): linear ramp from zero to WS_WIND_STRESS_X over WS_RAMP_S, constant afterwards.
"""
wind_stress_at(t::Real) = WS_WIND_STRESS_X * clamp(t / WS_RAMP_S, 0.0, 1.0)

"""
    wind_shear_config(; use_hydrostatic, t_end, nx=400, nz=160, dt=0.02, kwargs...) -> ModelConfig

Model configuration of the double-gyre case for the simulated time `t_end` (s): `nx` x `nz` cells in the closed
WS_LENGTH x WS_DEPTH basin, `tvd2` momentum and density advection, mixing-length eddy viscosity, spectral pressure
solver (non-hydrostatic mode), no Rayleigh damping, no Froude limiter. `wind_stress_x` starts at zero; the run loop
sets it with `wind_stress_at`. Keyword arguments override or add `ModelConfig` fields.

WHY the eddy viscosity: with the molecular nu = 1e-6 m^2/s alone the wind momentum stays in the top cell (diffusion
time dz^2/nu ~ 1e3 s per cell) and there is no stress across the interface, so the lower layer only sloshes with the
internal seiche and no lower cell forms.
WHY no Rayleigh damping and no Froude limiter: both would damp or scale down the steady wind-driven circulation.
WHY dt = 0.02 s: |u| dt/dx + |w| dt/dz stays near 0.1, well below the monotonicity limit 0.5 of `tvd2` transport.
"""
function wind_shear_config(; use_hydrostatic::Bool, t_end::Real, nx::Int=400, nz::Int=160, dt::Real=0.02, kwargs...)
    return ModelConfig(;
        length=WS_LENGTH,
        depth=WS_DEPTH,
        nx=nx,
        nz=nz,
        dt=dt,
        nt=round(Int, t_end / dt),
        use_hydrostatic=use_hydrostatic,
        momentum_advection_scheme="tvd2",
        transport_advection_scheme="tvd2",
        use_turbulence_model=true,
        turbulence_mixing_length=WS_MIXING_LENGTH,
        max_turbulent_viscosity=WS_MAX_EDDY_VISCOSITY,
        rayleigh_damping=0.0,
        apply_froude_limiter=false,
        pressure_solver_method="spectral",
        rho0=1000.0,
        wind_stress_x=0.0,
        kwargs...,
    )
end

"""
    initialize_wind_shear!(model)

Two layers at rest (WS_RHO_TOP above WS_RHO_BOTTOM, flat interface at WS_INTERFACE_HEIGHT_FRACTION of the depth),
flat free surface.
"""
function initialize_wind_shear!(model::Model)
    initialize_internal_seiche!(model; rho_top=WS_RHO_TOP, rho_bottom=WS_RHO_BOTTOM,
        interface_height_fraction=WS_INTERFACE_HEIGHT_FRACTION, interface_tilt_amplitude_fraction=0.0)
    return model
end

"""
    streamfunction(u, dz) -> Matrix{Float64}

Streamfunction psi (m^2/s, size (nz+1) x (nx+1)) of the face velocities `u` (m/s, nz x (nx+1)) on layers of
thickness `dz` (m): psi[k, i] = sum_{k' < k} u[k', i] dz, i.e. psi at the lower face of row k, zero at the bed.
With x to the right and z up, psi < 0 marks a clockwise cell and psi > 0 a counter-clockwise cell. The top-layer
thickness change eta is neglected (|eta| < 1 mm here).
"""
function streamfunction(u::AbstractMatrix{<:Real}, dz::Real)
    nz, nxp1 = size(u)
    psi = zeros(nz + 1, nxp1)
    @inbounds for i in 1:nxp1, k in 1:nz
        psi[k + 1, i] = psi[k, i] + u[k, i] * dz
    end
    return psi
end

"""
    mid_basin_profile(u) -> Vector{Float64}

Mean of the face velocities `u` (m/s, nz x (nx+1)) over the middle half of the basin, per row (m/s, length nz).
"""
function mid_basin_profile(u::AbstractMatrix{<:Real})
    nz, nxp1 = size(u)
    mid = (nxp1 ÷ 4 + 1):(3 * nxp1 ÷ 4)
    return [sum(@view u[k, mid]) / length(mid) for k in 1:nz]
end

"""
    gyre_strengths(u, dz, nz_interface) -> (psi_upper, psi_lower)

Strength (m^2/s) of the two cells from the mean u over the middle half of the basin: the most negative psi above the
interface face `nz_interface + 1` (upper, clockwise cell) and the most positive psi at or below it (lower,
counter-clockwise cell).
"""
function gyre_strengths(u::AbstractMatrix{<:Real}, dz::Real, nz_interface::Int)
    nz = size(u, 1)
    psi = cumsum(mid_basin_profile(u)) .* dz   # psi at the upper face of each row
    return minimum(@view psi[(nz_interface + 1):(nz - 1)]), maximum(@view psi[1:nz_interface])
end

# ---------------------------------------------------------------------------------------------------------------------
# Exact steady solutions for a constant viscosity (reference for plot_wind_shear_analytic.jl and the tests)
#
# Far from the end walls of a closed basin the steady, hydrostatic flow is horizontally uniform and obeys, per layer of
# uniform density,
#     nu d2u/dz2 = G,      G = (1/rho0) dp/dx (m/s^2, constant within the layer),
# with zero stress at the bed (free slip, as in the model: nu du/dz = 0 at z = 0), the wind stress at the surface
# (nu du/dz = tau/rho0 at z = H) and zero net flux through every vertical section of each layer (closed basin, no
# exchange across the interface in the steady state). The derivation is given in the docstrings below; the uniform
# case is the classical wind-drift profile of a closed basin with a free-slip bed.
# ---------------------------------------------------------------------------------------------------------------------

const WS_ANALYTIC_NU = 1e-2           # m^2/s, constant viscosity of the comparison with the exact solutions

"""
    wind_steady_uniform(; tau, nu, depth, rho0=1000.0, g=9.81) -> NamedTuple (u, surface_slope)

Exact steady wind-driven flow in a closed basin of uniform density, far from the end walls: `u(z)` (m/s, z in m above
the bed) and the surface slope d(eta)/dx (-). Inputs: wind stress `tau` (Pa), viscosity `nu` (m^2/s), `depth` H (m),
`rho0` (kg/m^3), `g` (m/s^2).

With s = tau/(rho0 nu) (1/s), nu u'' = G, u'(0) = 0, u'(H) = s and int_0^H u dz = 0:

    u(z) = s/(2H) (z^2 - H^2/3),        G = s nu / H = tau/(rho0 H),        d(eta)/dx = G/g = tau/(rho0 g H).

Surface velocity s H/3 downwind, bed velocity s H/6 upwind.
"""
function wind_steady_uniform(; tau::Real, nu::Real, depth::Real, rho0::Real=1000.0, g::Real=9.81)
    s = tau / (rho0 * nu)
    u(z) = s / (2 * depth) * (z^2 - depth^2 / 3)
    return (u=u, surface_slope=tau / (rho0 * g * depth))
end

"""
    wind_steady_two_layer(; tau, nu, h_lower, depth, drho, rho0=1000.0, g=9.81) -> NamedTuple (u, surface_slope, interface_slope)

Exact steady wind-driven flow in a closed two-layer basin with one viscosity `nu` (m^2/s), far from the end walls:
`u(z)` (m/s), the surface slope d(eta)/dx (-) and the interface slope d(zeta)/dx (-). Lower layer 0 < z < a = `h_lower`
(m), upper layer a < z < H = `depth` (m), b = H - a; `drho` (kg/m^3) is the density difference, `tau` the wind stress
(Pa).

Each layer obeys nu u'' = G_i with u'(0) = 0, u'(H) = s = tau/(rho0 nu), u and nu u' continuous at z = a, and zero
flux in each layer. With g_i = G_i/nu (1/(m s)) and zeta = z - a:

    lower:  u = g1 (z^2 - a^2/3) / 2,
    upper:  u = g2 zeta^2/2 + g1 a zeta + g1 a^2/3,
    g1 = -s b / (2 a H),    g2 = s (2H + b) / (2 b H).

(The lower-layer flux condition gives u(a) = g1 a^2/3, stress continuity u'(a) = g1 a, the surface condition
g2 b = s - g1 a, and the upper-layer flux condition then fixes g1.) Since g1 < 0, the lower layer flows upwind at the
interface and downwind along the bed: the counter-rotating lower cell. The velocity does not depend on drho.
Hydrostatic pressure gives G2 = g d(eta)/dx (upper layer) and G1 = G2 + g' d(zeta)/dx (lower layer), g' = g drho/rho0:

    d(eta)/dx = nu g2 / g,        d(zeta)/dx = nu (g1 - g2) / g'.
"""
function wind_steady_two_layer(; tau::Real, nu::Real, h_lower::Real, depth::Real, drho::Real, rho0::Real=1000.0, g::Real=9.81)
    s = tau / (rho0 * nu)
    a = h_lower
    b = depth - a
    g1 = -s * b / (2 * a * depth)
    g2 = s * (2 * depth + b) / (2 * b * depth)
    u(z) = z <= a ? g1 * (z^2 - a^2 / 3) / 2 : g2 * (z - a)^2 / 2 + g1 * a * (z - a) + g1 * a^2 / 3
    return (u=u, surface_slope=nu * g2 / g, interface_slope=nu * (g1 - g2) / (g * drho / rho0))
end

"""
    wind_analytic_config(; use_hydrostatic, t_end, nx=200, nz=80, dt=0.05, length=WS_LENGTH, kwargs...) -> ModelConfig

Configuration of the comparison with the exact steady solutions: as `wind_shear_config`, but with the constant
viscosity WS_ANALYTIC_NU instead of the mixing-length model. Keyword arguments override or add `ModelConfig` fields.
WHY dt = 0.05 s at 200 x 80 cells: the explicit viscous term needs 2 nu dt (1/dx^2 + 1/dz^2) <= 1 (here 0.42).
"""
function wind_analytic_config(; use_hydrostatic::Bool, t_end::Real, nx::Int=200, nz::Int=80, dt::Real=0.05,
    length::Real=WS_LENGTH, kwargs...)
    return wind_shear_config(; use_hydrostatic=use_hydrostatic, t_end=t_end, nx=nx, nz=nz, dt=dt, length=length,
        use_turbulence_model=false, nu=WS_ANALYTIC_NU, kwargs...)
end

"""
    initialize_wind_steady_state!(model, solution; two_layer)

Sets `model` to an exact steady solution (from `wind_steady_uniform` or `wind_steady_two_layer`, with the model's
WS_WIND_STRESS_X, viscosity and depth): u at the interior u-faces, eta with the surface slope (zero mean), and the
density: uniform WS_RHO_TOP, or (`two_layer`) WS_RHO_BOTTOM below the interface tilted with the interface slope about
mid-basin (as a staircase of whole cells).
"""
function initialize_wind_steady_state!(model::Model, solution; two_layer::Bool)
    cfg = model.config
    st = model.state
    x_mid = cfg.length / 2
    h_lower = WS_INTERFACE_HEIGHT_FRACTION * cfg.depth
    @inbounds for i in 1:cfg.nx, k in 1:cfg.nz
        z = (k - 0.5) * model.dz
        z_interface = two_layer ? h_lower + solution.interface_slope * ((i - 0.5) * model.dx - x_mid) : -Inf
        st.rho[k, i] = z < z_interface ? WS_RHO_BOTTOM : WS_RHO_TOP
    end
    fill!(st.u, 0.0)
    @inbounds for i in 2:cfg.nx, k in 1:cfg.nz
        st.u[k, i] = solution.u((k - 0.5) * model.dz)
    end
    fill!(st.w, 0.0)
    @inbounds for i in 1:cfg.nx
        st.eta[i] = solution.surface_slope * ((i - 0.5) * model.dx - x_mid)
    end
    refresh_density_metadata!(model)
    return model
end

"""
    interface_heights(rho, dz) -> Vector{Float64}

Height (m) of the interface in each column: the lowest z where the cell-centred density `rho` (kg/m^3, nz x nx)
crosses the mid density (WS_RHO_TOP + WS_RHO_BOTTOM)/2, linearly interpolated between cell centres; NaN without a
crossing.
"""
function interface_heights(rho::AbstractMatrix{<:Real}, dz::Real)
    rho_mid = (WS_RHO_TOP + WS_RHO_BOTTOM) / 2
    nz, nx = size(rho)
    heights = fill(NaN, nx)
    # WHY two loops: `break` in a combined `for i in ..., k in ...` loop would leave both loops.
    for i in 1:nx
        for k in 1:(nz - 1)
            if (rho[k, i] - rho_mid) * (rho[k + 1, i] - rho_mid) <= 0 && rho[k, i] != rho[k + 1, i]
                heights[i] = (k - 0.5 + (rho[k, i] - rho_mid) / (rho[k, i] - rho[k + 1, i])) * dz
                break
            end
        end
    end
    return heights
end

"""
    linear_slope(x, y) -> Float64

Least-squares slope of y against x (units of y per unit of x).
"""
function linear_slope(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    xm, ym = sum(x) / length(x), sum(y) / length(y)
    return sum((x .- xm) .* (y .- ym)) / sum((x .- xm) .^ 2)
end
