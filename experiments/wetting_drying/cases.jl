# Wetting and drying (issue #31): cases shared by the scripts in this folder and mirrored by test/test_wetting_drying.jl.
# Plan, method and status: docs/plan_issue_31_wetting_drying.md.
#
# References:
# - Ritter (1892), Z. Ver. Deutsch. Ing. 36, 947-954 (dam break on a dry bed)
# - Carrier & Greenspan (1958), J. Fluid Mech. 4, 97-109 (standing wave on a planar beach)
# - Delestre et al. (2013), Int. J. Numer. Methods Fluids 72, 269-300 (SWASHES: analytical solutions with wet/dry fronts)

using SweTransport2DV

const WD_RHO0 = 1000.0   # kg/m^3, reference density and density of the homogeneous cases

"""
    wd_experiment_config(; kwargs...) -> ModelConfig

Wetting-drying configuration with the numerical stabilisers (Rayleigh damping, Froude limiter) and the viscosity off;
`kwargs` override any field.
"""
function wd_experiment_config(; kwargs...)
    defaults = (wetting_drying=true, use_hydrostatic=true, momentum_advection_scheme="upwind1", nu=0.0,
        rayleigh_damping=0.0, apply_froude_limiter=false, rho_left=WD_RHO0, rho_right=WD_RHO0,
        write_plots=false, write_animation=false, write_pressure_buoyancy_diagnostics=false)
    return ModelConfig(; merge(defaults, kwargs)...)
end

"Cell-centre coordinates x (m) of a configuration."
wd_x_centres(cfg::ModelConfig) = [(i - 0.5) * cfg.length / cfg.nx for i in 1:cfg.nx]

"Water level zeta = depth + eta (m above the grid bottom) per column."
wd_water_level(model::Model) = model.config.depth .+ model.state.eta

# ---- Lake at rest -------------------------------------------------------------------------------------------------
const LAKE_LENGTH = 50.0              # m
const LAKE_GRID_DEPTH = 2.0           # m
const LAKE_LEVEL = 1.6                # m, still water level above the grid bottom
const LAKE_RHO_LOWER = 1010.0         # kg/m^3, lower layer of the stratified variant (upper layer WD_RHO0)

"""
    lake_bed(cfg) -> Vector{Float64}

Bed (m above the grid bottom): a beach rising above the still water level on the right, an island and a submerged bump;
no level is a whole number of cells.
"""
function lake_bed(cfg::ModelConfig)
    x = wd_x_centres(cfg)
    L = cfg.length
    beach = @. max(0.0, (x - 0.7L) / (0.3L)) * 1.4 * cfg.depth
    island = @. 1.15 * cfg.depth * exp(-((x - 0.3L) / (0.04L))^2)
    bump = @. 0.45 * cfg.depth * exp(-((x - 0.5L) / (0.05L))^2)
    return @. 0.037 + beach + island + bump
end

"""
    lake_model(nz; stratified=false, nx=200, dt=0.05) -> Model

Lake at rest over `lake_bed`; with `stratified` a two-layer density in z (interface at half the still water depth).
"""
function lake_model(nz::Int; stratified::Bool=false, nx::Int=200, dt::Real=0.05)
    cfg = wd_experiment_config(length=LAKE_LENGTH, depth=LAKE_GRID_DEPTH, nx=nx, nz=nz, dt=dt)
    cfg.bed_level = lake_bed(cfg)
    model = Model(cfg)
    dz = cfg.depth / cfg.nz
    rho = stratified ? [((k - 0.5) * dz < 0.5 * LAKE_LEVEL ? LAKE_RHO_LOWER : WD_RHO0) for k in 1:nz, i in 1:nx] : WD_RHO0
    initialize_water_level!(model, LAKE_LEVEL; rho=rho)
    return model
end

# ---- Ritter dam break on a dry bed ---------------------------------------------------------------------------------
const RITTER_LENGTH = 100.0      # m
const RITTER_H0 = 1.0            # m, depth behind the dam
const RITTER_X0 = 40.0           # m, dam position
const RITTER_GRID_DEPTH = 1.2    # m (2DV: layers of 0.1 m with nz = 12)
const RITTER_DT_DX = 0.01        # s/m, time step per grid spacing (CFL u dt/dx = 2 c0 dt/dx ~ 0.06)
const RITTER_FRONT_DEPTH = 0.01  # m, depth that marks the front

"""
    ritter_model(nx, nz) -> Model

Dam at RITTER_X0 with depth RITTER_H0 behind it and a dry, flat, frictionless bed in front of it.
"""
function ritter_model(nx::Int, nz::Int)
    dx = RITTER_LENGTH / nx
    cfg = wd_experiment_config(length=RITTER_LENGTH, depth=RITTER_GRID_DEPTH, nx=nx, nz=nz, dt=RITTER_DT_DX * dx)
    model = Model(cfg)
    initialize_water_level!(model, [x < RITTER_X0 ? RITTER_H0 : 0.0 for x in wd_x_centres(cfg)]; rho=WD_RHO0)
    return model
end

"Ritter depth (m) at the cell centres of `model` at time t (s)."
ritter_depth(model::Model, t::Real) = [ritter_dam_break(x, t; h0=RITTER_H0, x0=RITTER_X0, g=model.config.g)[1] for x in wd_x_centres(model.config)]

"Front position (m): centre of the last cell deeper than RITTER_FRONT_DEPTH."
ritter_front(x::AbstractVector, h::AbstractVector) = x[findlast(>(RITTER_FRONT_DEPTH), h)]

"Analytical position (m) of the depth RITTER_FRONT_DEPTH at time t (s): x0 + (2 c0 - 3 sqrt(g h_f)) t."
ritter_front_exact(t::Real; g::Real=9.81) = RITTER_X0 + (2sqrt(g * RITTER_H0) - 3sqrt(g * RITTER_FRONT_DEPTH)) * t

# ---- Carrier-Greenspan standing wave on a planar beach -------------------------------------------------------------
const CG_AMPLITUDE = 0.4         # -, dimensionless amplitude A (non-breaking for A < 1)
const CG_SLOPE = 0.05            # -, beach slope
const CG_LENGTH = 20.0           # m, length scale = offshore distance of the forced boundary from the still shoreline
const CG_STILL_LEVEL = 1.2       # m, still water level above the grid bottom
const CG_ONSHORE = 4.0           # m, dry beach landward of the still shoreline
const CG_GRID_DEPTH = 1.5        # m (2DV: layers of 0.1 m with nz = 15)
const CG_SHORE_DEPTH = 1e-3      # m, depth that marks the numerical shoreline

const CG_WAVE = CarrierGreenspanWave(amplitude=CG_AMPLITUDE, slope=CG_SLOPE, length_scale=CG_LENGTH)

"Offshore distance (m) from the still shoreline of the model coordinate x (m)."
cg_offshore(x::Real) = CG_LENGTH - x

"Runup amplitude (m) of the linear limit, A/4 * slope * L."
cg_runup_amplitude() = CG_AMPLITUDE / 4 * CG_SLOPE * CG_LENGTH

"""
    cg_model(nx, nz, dt) -> (model, t0)

Planar beach, started at the phase of maximum runup (u = 0 everywhere, analytical time t0 in s) and forced by the
analytical velocity at the offshore (left) boundary. Model time t corresponds to analytical time t0 + t.
"""
function cg_model(nx::Int, nz::Int, dt::Real)
    cfg = wd_experiment_config(length=CG_LENGTH + CG_ONSHORE, depth=CG_GRID_DEPTH, nx=nx, nz=nz, dt=dt)
    x = wd_x_centres(cfg)
    cfg.bed_level = @. CG_STILL_LEVEL - CG_SLOPE * cg_offshore(x)
    model = Model(cfg)
    t0 = carrier_greenspan_rest_time(CG_WAVE)
    level = map(x) do xi
        eta, _, wet = carrier_greenspan_solution(CG_WAVE, cg_offshore(xi), t0)
        wet ? CG_STILL_LEVEL + eta : 0.0
    end
    initialize_water_level!(model, level; rho=WD_RHO0)
    # model u points onshore (+x), the analytical u offshore
    set_x_inflow_outflow!(model, t -> -carrier_greenspan_solution(CG_WAVE, CG_LENGTH, t0 + t)[2]; open_right=false)
    return model, t0
end

"Numerical shoreline: water level (m above the still level) and x (m) of the most landward cell deeper than CG_SHORE_DEPTH."
function cg_shoreline(model::Model)
    i = findlast(>(CG_SHORE_DEPTH), water_depth(model))
    return wd_water_level(model)[i] - CG_STILL_LEVEL, wd_x_centres(model.config)[i]
end

"Analytical surface elevation (m above the still level; NaN where dry) at the model coordinates x (m), analytical time t (s)."
function cg_surface_exact(x::AbstractVector, t::Real)
    return map(x) do xi
        eta, _, wet = carrier_greenspan_solution(CG_WAVE, cg_offshore(xi), t)
        wet ? eta : NaN
    end
end

# ---- Sloshing over two beaches, two water masses ------------------------------------------------------------------
const SLOSH_LENGTH = 40.0        # m
const SLOSH_GRID_DEPTH = 1.5     # m
const SLOSH_RHO_DENSE = 1004.0   # kg/m^3, water mass on the left (the right one has WD_RHO0)
const SLOSH_RHO_BOTTOM = 2.0     # kg/m^3, added below z = 0.5 m (weak stratification)

"""
    sloshing_model(nz; nx=160, dt=0.02) -> Model

Closed basin with a parabolic bed rising above the water on both sides; the surface is tilted up to the left, so the
water sloshes, floods the right beach and drains again. Two water masses of different density (left / right) with a
weak stratification.
"""
function sloshing_model(nz::Int; nx::Int=160, dt::Real=0.02)
    cfg = wd_experiment_config(length=SLOSH_LENGTH, depth=SLOSH_GRID_DEPTH, nx=nx, nz=nz, dt=dt)
    x = wd_x_centres(cfg)
    L = cfg.length
    cfg.bed_level = @. 0.013 + 1.6 * ((x - 0.5L) / (0.5L))^2
    model = Model(cfg)
    level = @. 0.9 + 0.25 * (0.5L - x) / (0.5L)
    dz = cfg.depth / nz
    rho = [(x[i] < 0.4L ? SLOSH_RHO_DENSE : WD_RHO0) + ((k - 0.5) * dz < 0.5 ? SLOSH_RHO_BOTTOM : 0.0) for k in 1:nz, i in 1:nx]
    initialize_water_level!(model, level; rho=rho)
    return model
end
