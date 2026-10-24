# Lock exchange with obstacle (issue #3): density-stratified version of the dam break with obstacle, shared by the
# scripts in this folder. Plan and decisions: docs/plan_issue_3_lock_exchange_obstacle.md.
#
# Geometry after Robert et al. (Time-Splitting Procedure for the Resolution of the Volume Fraction for Unsteady Flows,
# section 3.1 and Figure 2; after Ubbink 1997): square tank 4a x 4a, column of heavy water a wide and 2a high at the
# left wall, obstacle d wide and 2d high on the bed, a gap of a after the column; a = 0.146 m, d = 0.024 m.
# In the original the column is water and the rest air; here the rest is light water up to a free surface.
#
# References:
# - Martin & Moyce (1952), Phil. Trans. R. Soc. Lond. A 244, 312-324
# - Koshizuka & Oka (1996), Nucl. Sci. Eng. 123, 421-434
# - Ubbink (1997), PhD thesis, Imperial College London
# - Benjamin (1968), J. Fluid Mech. 31, 209-248 (gravity currents)

using SweTransport2DV

const LEO_A = 0.146                  # m, column width (column height 2a, tank 4a x 4a)
const LEO_D = 0.024                  # m, obstacle width (height 2d)
const LEO_RHO_LIGHT = 1000.0         # kg/m^3, light water (also the Boussinesq reference density rho0)
const LEO_DT_REF = 0.0015            # s, time step for drho = 10 kg/m^3 on the 2 mm grid (max CFL ~0.3, see below)
const LEO_DRHO_REF = 10.0            # kg/m^3, density difference of LEO_DT_REF
const LEO_DX_REF = 0.002             # m, grid spacing of LEO_DT_REF

"""
    leo_reduced_gravity(drho; g=9.81) -> Float64

Reduced gravity g' = g drho / rho0 (m/s^2) for the density difference `drho` (kg/m^3), rho0 = LEO_RHO_LIGHT.
"""
leo_reduced_gravity(drho::Real; g::Real=9.81) = g * drho / LEO_RHO_LIGHT

"""
    leo_time_scale(drho) -> Float64

Time scale sqrt(a / g') (s) of the collapse; all runs are compared in t* = t / leo_time_scale(drho).
"""
leo_time_scale(drho::Real) = sqrt(LEO_A / leo_reduced_gravity(drho))

"""
    leo_time_step(n, drho) -> Float64

Time step (s) for n x n cells and the density difference `drho` (kg/m^3): dt scales with the cell size and with the
velocity scale sqrt(g' a), so the CFL number is the same in all runs. Reference: dt = 0.0015 s for 10 kg/m^3 on the
2 mm grid, where the largest |u| dt/dx + |w| dt/dz is about 0.3 (measured), below the tvd2 monotonicity limit 0.5.
"""
leo_time_step(n::Int, drho::Real) = LEO_DT_REF * ((4LEO_A / n) / LEO_DX_REF) * sqrt(LEO_DRHO_REF / drho)

"""
    leo_bed_level(n) -> Vector{Float64}

Bed level (m) of the n columns: the obstacle (height 2d) in the columns whose centre lies in (2a, 2a + d), zero
elsewhere. The edges lie on cell faces for n = 292 (2 mm) and n = 584 (1 mm).
"""
function leo_bed_level(n::Int)
    dx = 4LEO_A / n
    return [((i - 0.5) * dx > 2LEO_A && (i - 0.5) * dx < 2LEO_A + LEO_D) ? 2LEO_D : 0.0 for i in 1:n]
end

"""
    lock_exchange_obstacle_config(; n, drho, t_end_star, kwargs...) -> ModelConfig

Model configuration for n x n cells, density difference `drho` (kg/m^3) and simulated time t* = t_end_star (in units
of leo_time_scale). Non-hydrostatic, sparse pressure solver (the bed has solid cells), tvd2 momentum and transport,
free surface on top, no Rayleigh damping, no Froude limiter, nu = 1e-6 m^2/s (water). Walls: free-slip, as in the
reference case. Keyword arguments override or add ModelConfig fields.
"""
function lock_exchange_obstacle_config(; n::Int, drho::Real, t_end_star::Real, kwargs...)
    dt = leo_time_step(n, drho)
    return ModelConfig(;
        nx=n,
        nz=n,
        length=4LEO_A,
        depth=4LEO_A,
        dt=dt,
        nt=round(Int, t_end_star * leo_time_scale(drho) / dt),
        use_hydrostatic=false,
        pressure_solver_method="sparse",
        momentum_advection_scheme="tvd2",
        transport_advection_scheme="tvd2",
        rayleigh_damping=0.0,
        apply_froude_limiter=false,
        nu=1e-6,
        rho0=LEO_RHO_LIGHT,
        rho_left=LEO_RHO_LIGHT + drho,
        rho_right=LEO_RHO_LIGHT,
        bed_level=leo_bed_level(n),
        kwargs...,
    )
end

"""
    initialize_lock_exchange_obstacle!(model, drho)

Heavy water (rho0 + drho) in the column x < a, z < 2a; light water (rho0) elsewhere, at rest, flat free surface.
"""
function initialize_lock_exchange_obstacle!(model::Model, drho::Real)
    cfg = model.config
    st = model.state
    @inbounds for i in 1:cfg.nx, k in 1:cfg.nz
        heavy = (i - 0.5) * model.dx < LEO_A && (k - 0.5) * model.dz < 2LEO_A
        st.rho[k, i] = heavy ? LEO_RHO_LIGHT + drho : LEO_RHO_LIGHT
    end
    fill!(st.u, 0.0)
    fill!(st.w, 0.0)
    fill!(st.eta, 0.0)
    refresh_density_metadata!(model)
    return model
end

"""
    leo_front_position(model, drho) -> Float64

Position (m) of the leading edge of the heavy water: the largest x of a column with a wet cell whose heavy-water fraction
(rho - rho0)/drho exceeds 0.5.
"""
function leo_front_position(model::Model, drho::Real)
    rho, wet = model.state.rho, model.geometry.wet_cell
    for i in model.config.nx:-1:1, k in 1:model.config.nz
        if wet[k, i] && (rho[k, i] - LEO_RHO_LIGHT) / drho > 0.5
            return (i - 0.5) * model.dx
        end
    end
    return 0.0
end

"""
    leo_obstacle_force(model) -> Float64

Horizontal pressure force on the obstacle per unit width (N/m): sum over the obstacle rows of
(p_left - p_right) dz, with the total pressure p = hydrostatic + non-hydrostatic part (Pa) in the wet cells next to
the two obstacle walls. Positive when it pushes the obstacle towards +x.
"""
function leo_obstacle_force(model::Model)
    geo = model.geometry
    cols = findall(>(1), geo.k_bed)
    i_left, i_right = first(cols) - 1, last(cols) + 1
    p = model.state.p
    force = 0.0
    for k in 1:(geo.k_bed[first(cols)] - 1)
        force += (p[k, i_left] - p[k, i_right]) * model.dz
    end
    return force
end
