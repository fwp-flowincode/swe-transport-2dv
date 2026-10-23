# Rayleigh-Taylor instability: case definitions shared by the scripts in this folder
# (run_reference_dt0005_25s_anim.jl, run_fast_dt0004_25s_anim.jl, plot_rayleigh_taylor.jl).
#
# A thin heavy layer (rho_top) lies on top of a light layer (rho_bottom) in an 8 m x 4 m closed basin with a free
# surface. The interface is displaced by a cosine of wavenumber k = 2*pi*m/L; the fluid starts at rest.
#
# Canonical RTI references:
# - Rayleigh (1883), Proc. London Math. Soc., doi:10.1112/plms/s1-14.1.170
# - Taylor (1950), Proc. Royal Soc. A, doi:10.1098/rspa.1950.0052
# - Sharp (1984), Physica D, doi:10.1016/0167-2789(84)90510-4
# - Chandrasekhar (1961), Hydrodynamic and Hydromagnetic Stability, ch. X (linear theory incl. finite layer depths)

using SweTransport2DV

const RT_LENGTH_M = 8.0     # m, basin length
const RT_DEPTH_M = 4.0      # m, still-water depth
const RT_NX = 400           # -, cells in x (dx = 0.02 m)
const RT_NZ = 160           # -, cells in z (dz = 0.025 m)

# Physical case parameters. Densities in kg/m^3, interface height and perturbation amplitude as fractions of the depth,
# perturbation_wavenumber m = number of wavelengths in the basin, dt in s.
const RT_CASES = Dict(
    # Atwood number 3.5e-4, k = 1.57 1/m: slow growth (e-folding time ~15 s)
    "reference" => (
        rho_top=1000.60,
        rho_bottom=999.90,
        interface_height_fraction=0.85,
        perturbation_amplitude_fraction=0.01,
        perturbation_wavenumber=2,
        dt=0.005,
    ),
    # Atwood number 2.0e-3, k = 2.36 1/m: fast growth (e-folding time ~5 s)
    "fast" => (
        rho_top=1002.50,
        rho_bottom=998.50,
        interface_height_fraction=0.82,
        perturbation_amplitude_fraction=0.03,
        perturbation_wavenumber=3,
        dt=0.004,
    ),
)

"""
    rayleigh_taylor_config(case_name; kwargs...) -> ModelConfig

Model configuration of the Rayleigh-Taylor case `case_name` ("reference" or "fast"). Keyword arguments override or
add `ModelConfig` fields (e.g. `nt`, output switches).

WHY these settings:
- non-hydrostatic: the instability is a non-hydrostatic process (the hydrostatic growth rate is unbounded in k);
- `transport_advection_scheme="tvd2"`: keeps the interface sharp; the linear growth is nearly the same with `upwind1`;
- no Rayleigh damping and no Froude limiter: physical run without numerical velocity control. The limiter would only
  act above Fr = 0.9; the runs stay at or below 0.70 (measured: reference 60 s, fast 25 s), so switching it off does
  not change the result.
Time step: the advective CFL number |u| dt/dx + |w| dt/dz stays at or below 0.12 in both cases (measured; tvd2 transport
is monotone up to 0.5).
"""
function rayleigh_taylor_config(case_name::AbstractString; kwargs...)
    case = RT_CASES[case_name]
    return ModelConfig(;
        length=RT_LENGTH_M,
        depth=RT_DEPTH_M,
        nx=RT_NX,
        nz=RT_NZ,
        dt=case.dt,
        use_hydrostatic=false,
        momentum_advection_scheme="tvd2",
        transport_advection_scheme="tvd2",
        rayleigh_damping=0.0,
        apply_froude_limiter=false,
        pressure_solver_method="spectral",
        kwargs...,
    )
end

"""
    initialize_rayleigh_taylor_case!(model, case_name)

Set the initial density of the case `case_name` (fluid at rest, flat free surface).
"""
function initialize_rayleigh_taylor_case!(model::Model, case_name::AbstractString)
    case = RT_CASES[case_name]
    initialize_rayleigh_taylor!(
        model;
        rho_top=case.rho_top,
        rho_bottom=case.rho_bottom,
        interface_height_fraction=case.interface_height_fraction,
        perturbation_amplitude_fraction=case.perturbation_amplitude_fraction,
        perturbation_wavenumber=case.perturbation_wavenumber,
    )
    return model
end

"""
    linear_growth_rate(case_name; g=9.81) -> Float64

Inviscid linear growth rate sigma (1/s) of the interface mode of the case, for two layers between rigid horizontal
boundaries (Chandrasekhar 1961, ch. X; equation numbers not checked):

    sigma^2 = g k (rho_top - rho_bottom) / (rho_top coth(k h_top) + rho_bottom coth(k h_bottom)),

with k = 2 pi m / L (1/m), h_top and h_bottom the layer thicknesses (m) and g in m/s^2. A displacement released from
rest grows as a(t) = a0 cosh(sigma t).
WHY rigid lid: the free-surface displacement of an internal mode is O((rho_top - rho_bottom)/rho) times the interface
displacement, so the free surface acts as a rigid lid for this mode.
"""
function linear_growth_rate(case_name::AbstractString; g::Real=9.81)
    case = RT_CASES[case_name]
    k = 2.0 * pi * case.perturbation_wavenumber / RT_LENGTH_M          # 1/m
    h_top = (1.0 - case.interface_height_fraction) * RT_DEPTH_M        # m
    h_bottom = case.interface_height_fraction * RT_DEPTH_M             # m
    drho = case.rho_top - case.rho_bottom                              # kg/m^3
    return sqrt(g * k * drho / (case.rho_top * coth(k * h_top) + case.rho_bottom * coth(k * h_bottom)))
end

"""
    interface_mode_amplitude(rho, case_name, dx, dz) -> Float64

Amplitude (m) of the perturbation mode m of the interface height. The interface height of each column is
z_i = H - h_heavy with the heavy-fluid thickness h_heavy = sum_k (rho - rho_bottom)/(rho_top - rho_bottom) dz (m), which
is exact for a sharp single-valued interface and stays defined (as an equivalent interface) after overturning and
mixing. The amplitude is the cosine coefficient a = (2/nx) sum_i z_i cos(2 pi m x_i / L).
`rho` in kg/m^3 (nz x nx), `dx`, `dz` in m.
"""
function interface_mode_amplitude(rho::AbstractMatrix{<:Real}, case_name::AbstractString, dx::Real, dz::Real)
    case = RT_CASES[case_name]
    nz, nx = size(rho)
    a = 0.0
    @inbounds for i in 1:nx
        h_heavy = 0.0   # m
        for k in 1:nz
            h_heavy += (rho[k, i] - case.rho_bottom) / (case.rho_top - case.rho_bottom) * dz
        end
        x = (i - 0.5) * dx
        a += (nz * dz - h_heavy) * cos(2.0 * pi * case.perturbation_wavenumber * x / (nx * dx))
    end
    return 2.0 * a / nx
end
