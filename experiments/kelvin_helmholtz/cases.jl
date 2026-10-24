# Kelvin-Helmholtz shear-layer instability: settings shared by the scripts in this folder
# (plot_kelvin_helmholtz.jl and run_kelvin_helmholtz_dt00025_120s_anim.jl).
#
# Two layers (light on top, heavy below) in an 8 m x 4 m domain, periodic in x, move in opposite directions
# (u_top = +0.2 m/s, u_bottom = -0.2 m/s) with a tanh shear profile across the interface. The interface is displaced
# by a cosine with two wavelengths in the domain.
#
# Kelvin-Helmholtz references:
# - Kelvin (1871), Philosophical Magazine
# - Helmholtz (1868), Monatsberichte der Koniglichen Akademie der Wissenschaften
# - Chandrasekhar (1981), Hydrodynamic and Hydromagnetic Stability

using SweTransport2DV

const KH_RHO_TOP = 999.95                     # kg/m^3, upper (light) layer
const KH_RHO_BOTTOM = 1000.05                 # kg/m^3, lower (heavy) layer
const KH_U_TOP = 0.20                         # m/s, velocity of the upper layer
const KH_U_BOTTOM = -0.20                     # m/s, velocity of the lower layer
const KH_INTERFACE_HEIGHT_FRACTION = 0.5      # -, interface at mid-depth
const KH_PERTURBATION_AMPLITUDE_FRACTION = 0.02   # -, interface displacement 0.08 m
const KH_PERTURBATION_WAVENUMBER = 2          # -, wavelengths in the domain (lambda = 4 m)

"""
    kelvin_helmholtz_config(; t_end, kwargs...) -> ModelConfig

Model configuration of the Kelvin-Helmholtz case for the simulated time `t_end` (s): 800 x 400 cells (dx = dz = 0.01 m)
in an 8 m x 4 m domain, dt = 0.0025 s, periodic in x, non-hydrostatic, `tvd2` momentum and density advection,
Rayleigh damping 0.001 1/s, spectral pressure solver (Fourier modes in x, cosine transform in z). Keyword arguments
override or add `ModelConfig` fields.

WHY no Froude limiter: the imposed shear starts at Fr = max|u| / sqrt(g' H) = 0.2 / 0.063 = 3.2, far above the limiter
threshold 0.9; the limiter would scale down the velocity jump that drives the instability.
WHY dt = 0.0025 s: keeps |u| dt/dx + |w| dt/dz well below the monotonicity limit 0.5 of `tvd2` transport while the
billows develop.
"""
function kelvin_helmholtz_config(; t_end::Real, kwargs...)
    dt = 0.0025   # s
    return ModelConfig(;
        length=8.0,
        depth=4.0,
        nx=800,
        nz=400,
        dt=dt,
        nt=round(Int, t_end / dt),
        use_hydrostatic=false,
        periodic_x=true,
        momentum_advection_scheme="tvd2",
        transport_advection_scheme="tvd2",
        rayleigh_damping=0.001,
        apply_froude_limiter=false,
        pressure_solver_method="spectral",
        kwargs...,
    )
end

"""
    initialize_kelvin_helmholtz_case!(model)

Initial state of the case: two layers with the densities and velocities above, tanh shear across the perturbed
interface, fluid otherwise at rest relative to the layer velocities, flat free surface.
"""
function initialize_kelvin_helmholtz_case!(model::Model)
    initialize_kelvin_helmholtz!(
        model;
        rho_top=KH_RHO_TOP,
        rho_bottom=KH_RHO_BOTTOM,
        interface_height_fraction=KH_INTERFACE_HEIGHT_FRACTION,
        interface_perturbation_amplitude_fraction=KH_PERTURBATION_AMPLITUDE_FRACTION,
        interface_perturbation_wavenumber=KH_PERTURBATION_WAVENUMBER,
        u_top=KH_U_TOP,
        u_bottom=KH_U_BOTTOM,
    )
    return model
end
