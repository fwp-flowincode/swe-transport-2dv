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
