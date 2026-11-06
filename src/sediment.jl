# Suspended cohesive sediment in the wetting-drying step (issue #64): one fraction, concentration c (kg/m^3) per cell.
#
#   d(h c)/dt + div(q c) = d/dz(K dc/dz) + d(w_s c)/dz,   K = nu_t / sigma_c (k-epsilon) or 0,
# with the bed exchange in the lowest wet cell
#   erosion     E = M (tau_b / tau_ce - 1) for tau_b > tau_ce, else 0         (Partheniades 1965),
#   deposition  D = w_s c_b                                                    (continuous deposition; Winterwerp &
#                                                                               van Kesteren 2004, sec. 7.4),
# M (kg/m^2/s) the erosion parameter, tau_ce (Pa) the critical stress for erosion, w_s (m/s) the settling velocity,
# c_b the concentration of the lowest wet cell. Equation numbers of the references not checked.
# - Advection: the conservative scalar transport of the density (`_wd_transport_scalar!`, upwind1 or tvd2 by
#   transport_advection_scheme), bounds [0, max(c)]; the inflow at the left boundary carries `inflow_concentration`,
#   water entering through the open right boundary `outside_concentration` (default 0: clean water from outside; WHY
#   not zero gradient: a flow reversal at the outflow would carry the plume of the last cell back in as new sediment).
# - Settling, diffusion and the bed exchange: per column backward Euler in finite volumes on the wet cells of h^{n+1},
#     h_k (c_k - c*_k) = dt [K (c_{k+1} - c_k)/d - K (c_k - c_{k-1})/d + w_s c_{k+1} - w_s c_k] + dt E (lowest cell),
#   the settling flux w_s c upwinded from above (out of cell k through its lower face, into cell k-1; out of the lowest
#   cell into the bed), zero flux at the surface. The matrix is an M-matrix (positive diagonal, non-positive off-diagonal,
#   column sums h_k + dt w_s for the lowest cell and h_k otherwise), so c stays >= 0 for any dt, and the suspended mass
#   changes exactly by dt (E - D) (WHY implicit: w_s dt / h is large in thin cut cells).
# - Bed shear stress: tau_b = rho0 c_f U_w^2 with the log-law coefficient c_f = (kappa / ln(z1/z0))^2 of the momentum
#   bed friction (`_wd_implicit_vertical!`): the wall cell (with the equilibrium wall function the lowest cell of at
#   least wall_cell_min_fraction dz), its velocity U_w and the height z1 of its centre.
# - Density coupling (optional): the mixture density rho_m = rho + c (1 - rho / rho_s) replaces rho in the baroclinic
#   pressure and in the buoyancy term of k-epsilon (`_wd_buoyant_density`); the transported rho is the water density.

const SED_SCHMIDT = 0.7              # -, turbulent Schmidt number of the suspended sediment (as the density, Rodi 1993)
const SED_DENSITY = 2650.0           # kg/m^3, density of the sediment particles (quartz and clay minerals)

"""
    SedimentState

Suspended cohesive sediment of the wetting-drying step (issue #64): concentration `c` (kg/m³, nz x nx), settling
velocity `settling_velocity` (m/s, > 0), critical bed shear stress for erosion `critical_erosion_stress` (Pa, > 0),
erosion parameter `erosion_parameter` (kg/m²/s), particle density `sediment_density` (kg/m³), `density_coupling`
(the sediment adds to the density of the baroclinic pressure and of the k-epsilon buoyancy), `inflow_concentration`
(kg/m³ at the left boundary face; one value, one per layer, or `nothing` for the first cell), `outside_concentration`
(kg/m³ of water entering through the right boundary face; `nothing` for the last cell), `bed_mass` (kg/m² per
column, the erodible mass; Inf: unlimited). Diagnostics of the last step per column: bed shear stress `tau_b` (Pa),
`erosion` and `deposition` (kg/m²/s); cumulative per column `bed_change` (kg/m², deposited minus eroded) and totals
per unit width (kg/m): `eroded`, `deposited`, `inflow`, `outflow`.
"""
mutable struct SedimentState{T<:Real}
    c::Matrix{T}
    settling_velocity::Float64
    critical_erosion_stress::Float64
    erosion_parameter::Float64
    sediment_density::Float64
    density_coupling::Bool
    inflow_concentration::Any
    outside_concentration::Any
    bed_mass::Vector{T}
    rho_mix::Matrix{T}       # kg/m^3, mixture density (density coupling)
    lower::Vector{T}         # work vectors of the column solve (length nz)
    diag::Vector{T}
    upper::Vector{T}
    rhs::Vector{T}
    sol::Vector{T}
    tau_b::Vector{T}
    erosion::Vector{T}
    deposition::Vector{T}
    bed_change::Vector{T}
    eroded::Float64
    deposited::Float64
    inflow::Float64
    outflow::Float64
end

"""
    attach_sediment!(model; settling_velocity, critical_erosion_stress, erosion_parameter,
                     sediment_density=2650, density_coupling=true, concentration=0, inflow_concentration=0,
                     outside_concentration=0, bed_mass=Inf) -> SedimentState

Adds one fraction of suspended cohesive sediment to a wetting-drying model (issue #64; see src/sediment.jl for the
equations). `concentration` (kg/m³): initial value, scalar or nz x nx. `bed_mass` (kg/m²): erodible mass per column,
scalar or one per column. Needs `wetting_drying=true` and, with erosion, a bed roughness length (the bed shear stress is
that of the log-law bed friction). With `settling_velocity=0`, `erosion_parameter=0` and `density_coupling=false` the
fraction is a passive tracer.
"""
function attach_sediment!(model::Model; settling_velocity::Real, critical_erosion_stress::Real, erosion_parameter::Real,
    sediment_density::Real=SED_DENSITY, density_coupling::Bool=true, concentration=0.0, inflow_concentration=0.0,
    outside_concentration=0.0, bed_mass=Inf)
    wd = model.wetdry
    wd === nothing && throw(ArgumentError("attach_sediment! needs wetting_drying=true"))
    cfg = model.config
    # WHY not without erosion: the bed shear stress only drives the erosion; with erosion_parameter = 0 a frictionless bed
    # (z0 = 0: tau_b = 0) is allowed, e.g. a passive tracer (settling_velocity = 0, density_coupling = false; issue #11)
    (cfg.bed_roughness_length > 0 || erosion_parameter == 0) ||
        throw(ArgumentError("attach_sediment! with erosion needs bed_roughness_length > 0 (log-law bed shear stress)"))
    (isfinite(settling_velocity) && settling_velocity >= 0) || throw(ArgumentError("settling_velocity must be >= 0"))
    (isfinite(critical_erosion_stress) && critical_erosion_stress > 0) || throw(ArgumentError("critical_erosion_stress must be > 0"))
    (isfinite(erosion_parameter) && erosion_parameter >= 0) || throw(ArgumentError("erosion_parameter must be >= 0"))
    sediment_density > cfg.rho0 || throw(ArgumentError("sediment_density must exceed rho0"))
    nz, nx = cfg.nz, cfg.nx
    T = eltype(model.state.rho)
    c = concentration isa Real ? fill(T(concentration), nz, nx) : T.(concentration)
    size(c) == (nz, nx) || throw(ArgumentError("concentration must be a scalar or nz x nx"))
    all(>=(0), c) || throw(ArgumentError("concentration must be >= 0"))
    if inflow_concentration !== nothing
        all(v -> isfinite(v) && v >= 0, inflow_concentration) || throw(ArgumentError("inflow_concentration must be >= 0"))
        inflow_concentration isa AbstractVector && length(inflow_concentration) != nz &&
            throw(ArgumentError("inflow_concentration needs one value per layer (length nz)"))
    end
    if outside_concentration !== nothing
        all(v -> isfinite(v) && v >= 0, outside_concentration) || throw(ArgumentError("outside_concentration must be >= 0"))
        outside_concentration isa AbstractVector && length(outside_concentration) != nz &&
            throw(ArgumentError("outside_concentration needs one value per layer (length nz)"))
    end
    bed = bed_mass isa Real ? fill(T(bed_mass), nx) : T.(collect(bed_mass))
    length(bed) == nx || throw(ArgumentError("bed_mass must be a scalar or one value per column"))
    all(>=(0), bed) || throw(ArgumentError("bed_mass must be >= 0"))
    z() = zeros(T, nx)
    sed = SedimentState{T}(c, Float64(settling_velocity), Float64(critical_erosion_stress), Float64(erosion_parameter),
        Float64(sediment_density), density_coupling,
        inflow_concentration isa AbstractVector ? Float64.(collect(inflow_concentration)) : inflow_concentration,
        outside_concentration isa AbstractVector ? Float64.(collect(outside_concentration)) : outside_concentration,
        bed, similar(c), zeros(T, nz), zeros(T, nz), zeros(T, nz), zeros(T, nz), zeros(T, nz), z(), z(), z(), z(),
        0.0, 0.0, 0.0, 0.0)
    wd.sediment = sed
    _wd_mixture_density!(model, sed)
    _wd_baroclinic_pressure!(model, wd)
    return sed
end

"""
    suspended_sediment_mass(model) -> Float64

Suspended sediment per unit width (kg/m): sum of c h dx over the wet cells.
"""
function suspended_sediment_mass(model::Model)
    sed = model.wetdry === nothing ? nothing : model.wetdry.sediment
    sed === nothing && return 0.0
    wd = model.wetdry
    _wd_cell_thicknesses!(wd.h, model, wd)
    return sum(sed.c .* wd.h) * model.dx
end

# Mixture density rho + c (1 - rho / rho_s) (kg/m^3) into sed.rho_mix.
function _wd_mixture_density!(model::Model, sed::SedimentState)
    rho = model.state.rho
    rs = sed.sediment_density
    @inbounds for i in eachindex(rho)
        sed.rho_mix[i] = rho[i] + sed.c[i] * (1.0 - rho[i] / rs)
    end
    return nothing
end

# Density (kg/m^3) for the baroclinic pressure and the k-epsilon buoyancy: the mixture density with coupled sediment,
# else the transported density.
# WHY the type assertion: wd.sediment is untyped (Any); without it every rho[k, i] of the callers is dynamically typed
# and boxed (2.5 MiB per call of the baroclinic pressure on 200 x 56 cells)
function _wd_buoyant_density(model::Model{T}, wd::WetDryState) where {T}
    sed = wd.sediment
    (sed === nothing || !sed.density_coupling) && return model.state.rho
    return sed.rho_mix::Matrix{T}
end

# Wall cell kw of column i in the thicknesses h, the thickness below it (m) and the height of its centre z1 (m): the cell
# of the log-law bed friction (as _wd_implicit_vertical! and the k-epsilon wall function).
function _wd_wall_cell(model::Model, h::AbstractMatrix{<:Real}, i::Int, klo::Int, khi::Int)
    ke = model.turbulence
    kw = klo
    below = 0.0
    if ke !== nothing && ke.wall_function === :equilibrium
        @inbounds while kw < khi && h[kw, i] < ke.wall_cell_min_fraction * model.dz
            below += h[kw, i]
            kw += 1
        end
    end
    z1 = max(below + 0.5 * h[kw, i], WD_FRICTION_Z1_MIN_FACTOR * model.config.bed_roughness_length)
    return kw, z1
end

# Turbulent diffusivity K = nu_t/sigma_c (m^2/s) at the interface between the cells ka and kb of column i (0 without
# k-epsilon). WHY a function, not a closure in the column loop: the closure was allocated for every cell (issue #10).
@inline _sed_diffusivity(::Nothing, ka::Int, kb::Int, i::Int) = 0.0
@inline _sed_diffusivity(ke::KEpsilonState, ka::Int, kb::Int, i::Int) =
    @inbounds 0.5 * (ke.nu_t[ka, i] + ke.nu_t[kb, i]) / SED_SCHMIDT

# Advection of c with the step fluxes, then settling, turbulent diffusion and the bed exchange per column.
function _wd_sediment_step!(model::Model, wd::WetDryState, sed::SedimentState)
    cfg = model.config
    dt = cfg.dt
    c = sed.c
    # WHY these bounds: the monotone transport creates no new maximum; the inflow value may be one
    hi = max(maximum(c), sed.inflow_concentration === nothing ? 0.0 : maximum(sed.inflow_concentration),
        sed.outside_concentration === nothing ? 0.0 : maximum(sed.outside_concentration))
    inflow, outflow = _wd_transport_scalar!(model, wd, c, sed.inflow_concentration, 0.0, hi;
        rho_right=sed.outside_concentration)
    sed.inflow += inflow
    sed.outflow += outflow
    ke = model.turbulence
    h = wd.h_new
    ws = sed.settling_velocity
    lower, diag, upper, rhs, sol = sed.lower, sed.diag, sed.upper, sed.rhs, sed.sol
    @inbounds for i in 1:cfg.nx
        sed.erosion[i] = sed.deposition[i] = sed.tau_b[i] = 0.0
        klo, khi = _wd_wet_range(h, i)
        klo == 0 && continue
        kw, z1 = _wd_wall_cell(model, h, i, klo, khi)
        uw = _wd_cell_velocity(model, wd, kw, i)
        tau = cfg.rho0 * bed_friction_coefficient(cfg.bed_roughness_length, z1) * uw^2   # Pa
        sed.tau_b[i] = tau
        excess = tau / sed.critical_erosion_stress - 1.0
        erosion = excess > 0 ? sed.erosion_parameter * excess : 0.0   # kg/m^2/s
        # WHY the limit: the erodible bed (bed_mass + net deposition) cannot go negative within the step
        available = sed.bed_mass[i] + sed.bed_change[i]
        erosion = min(erosion, max(available, 0.0) / dt)
        n = khi - klo + 1
        for j in 1:n
            k = klo + j - 1
            kd = j > 1 ? dt * _sed_diffusivity(ke, k, k - 1, i) / (0.5 * (h[k, i] + h[k - 1, i])) : 0.0
            ku = j < n ? dt * _sed_diffusivity(ke, k, k + 1, i) / (0.5 * (h[k, i] + h[k + 1, i])) : 0.0
            # settling: out through the lower face (into the cell below or the bed), in from the cell above
            lower[j] = -kd
            upper[j] = -ku - (j < n ? dt * ws : 0.0)
            diag[j] = h[k, i] + kd + ku + dt * ws
            rhs[j] = h[k, i] * c[k, i] + (j == 1 ? dt * erosion : 0.0)
        end
        _solve_tridiagonal_column!(sol, lower, diag, upper, rhs, n)
        # WHY no clipping: the M-matrix keeps c >= 0, and a clip would break the exact mass balance
        for j in 1:n
            c[klo + j - 1, i] = sol[j]
        end
        deposition = ws * c[klo, i]   # kg/m^2/s, of the new state (backward Euler)
        sed.erosion[i] = erosion
        sed.deposition[i] = deposition
        sed.bed_change[i] += dt * (deposition - erosion)
        sed.eroded += dt * erosion * model.dx
        sed.deposited += dt * deposition * model.dx
    end
    _wd_mixture_density!(model, sed)
    return nothing
end
