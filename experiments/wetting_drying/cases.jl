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
function lake_model(nz::Int; stratified::Bool=false, nx::Int=200, dt::Real=0.05, kwargs...)
    cfg = wd_experiment_config(; length=LAKE_LENGTH, depth=LAKE_GRID_DEPTH, nx=nx, nz=nz, dt=dt, kwargs...)
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
    ritter_model(nx, nz; scheme="upwind1", kwargs...) -> Model

Dam at RITTER_X0 with depth RITTER_H0 behind it and a dry, flat, frictionless bed in front of it. `scheme` sets the
momentum and the transport advection; `kwargs` override other configuration fields (e.g. use_hydrostatic=false).
"""
function ritter_model(nx::Int, nz::Int; scheme::AbstractString="upwind1", kwargs...)
    dx = RITTER_LENGTH / nx
    cfg = wd_experiment_config(; length=RITTER_LENGTH, depth=RITTER_GRID_DEPTH, nx=nx, nz=nz, dt=RITTER_DT_DX * dx,
        momentum_advection_scheme=scheme, transport_advection_scheme=scheme, kwargs...)
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
    cg_model(nx, nz, dt; scheme="upwind1", kwargs...) -> (model, t0)

Planar beach, started at the phase of maximum runup (u = 0 everywhere, analytical time t0 in s) and forced by the
analytical velocity at the offshore (left) boundary. Model time t corresponds to analytical time t0 + t.
"""
function cg_model(nx::Int, nz::Int, dt::Real; scheme::AbstractString="upwind1", kwargs...)
    cfg = wd_experiment_config(; length=CG_LENGTH + CG_ONSHORE, depth=CG_GRID_DEPTH, nx=nx, nz=nz, dt=dt,
        momentum_advection_scheme=scheme, transport_advection_scheme=scheme, kwargs...)
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
function sloshing_model(nz::Int; nx::Int=160, dt::Real=0.02, scheme::AbstractString="upwind1", kwargs...)
    cfg = wd_experiment_config(; length=SLOSH_LENGTH, depth=SLOSH_GRID_DEPTH, nx=nx, nz=nz, dt=dt,
        momentum_advection_scheme=scheme, transport_advection_scheme=scheme, kwargs...)
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

# ---- Manufactured solution for the momentum advection (depth-uniform velocity) ---------------------------------------
# A depth-uniform u(x) under a surface zeta(x) that crosses several layer interfaces, with the interface fluxes W from the
# discrete layer continuity; hydrostatic theory gives the same tendency -u du/dx in every layer. Mirrored by
# test_verify_wd_advection_depth_uniform_mms_* (docs/TESTS_TECHNICAL_DOCUMENTATION.md).
const MMS_LENGTH = 100.0        # m
const MMS_GRID_DEPTH = 1.2      # m
const MMS_LAYER_MIN = 1e-3      # m, face layers thinner than this are left out
mms_zeta(x) = 0.6 + 0.3 * (1 - tanh((x - 50) / 15))    # m, from 1.2 m down to 0.6 m
mms_u(x) = 1.0 + 2.0 * tanh((x - 50) / 15)             # m/s, changes sign
mms_dudx(x) = 2.0 / 15 * (1 - tanh((x - 50) / 15)^2)   # 1/s

"""
    mms_advection_state(nx, nz) -> (model, x_faces)

Model in the manufactured state: geometry, discharges and interface fluxes set as in the time step.
"""
function mms_advection_state(nx::Int, nz::Int)
    cfg = wd_experiment_config(length=MMS_LENGTH, depth=MMS_GRID_DEPTH, nx=nx, nz=nz, dt=0.001)
    model = Model(cfg)
    wd = model.wetdry
    dx = model.dx
    xf = [(f - 1) * dx for f in 1:(nx + 1)]
    model.state.eta .= mms_zeta.(wd_x_centres(cfg)) .- cfg.depth
    for f in 2:nx, k in 1:nz
        model.state.u[k, f] = mms_u(xf[f])
    end
    SweTransport2DV._wd_cell_thicknesses!(wd.h, model, wd)
    SweTransport2DV._wd_face_thicknesses!(model, wd)
    model.state.u .*= wd.hf .> 0.0
    wd.q .= wd.hf .* model.state.u
    fill!(model.state.w, 0.0)
    for i in 1:nx
        zeta_t = -(sum(@view wd.q[:, i + 1]) - sum(@view wd.q[:, i])) / dx
        s = findlast(>(0.0), @view wd.h[:, i])
        for k in 1:nz
            model.state.w[k + 1, i] = model.state.w[k, i] - ((k == s ? zeta_t : 0.0) + (wd.q[k, i + 1] - wd.q[k, i]) / dx)
        end
    end
    return model, xf
end

# First-order tendency with the layer-wise weights qbar_k / hbar_k of the first version (horizontal part; the vertical part
# vanishes for a depth-uniform u), kept for the comparison figure.
function mms_layerwise_tendency(model::Model, k::Int, f::Int)
    wd = model.wetdry
    u = model.state.u
    hbar = 0.5 * (wd.h[k, f - 1] + wd.h[k, f])
    hbar > 1e-12 || return 0.0
    tend = 0.0
    ql = 0.5 * (wd.q[k, f - 1] + wd.q[k, f])
    ql > 0.0 && (tend += ql / (model.dx * hbar) * (u[k, f - 1] - u[k, f]))
    qr = 0.5 * (wd.q[k, f] + wd.q[k, f + 1])
    qr < 0.0 && (tend += -qr / (model.dx * hbar) * (u[k, f + 1] - u[k, f]))
    return tend
end

"""
    mms_advection_tendencies(nx, nz; method) -> (x_faces, tendency nz x (nx+1) (NaN where left out), exact)

method: "upwind1", "tvd2" (the model kernel) or "layerwise" (the first-version weights, upwind1).
"""
function mms_advection_tendencies(nx::Int, nz::Int; method::AbstractString="upwind1")
    model, xf = mms_advection_state(nx, nz)
    wd = model.wetdry
    tend = fill(NaN, nz, nx + 1)
    for f in 3:(nx - 1), k in 1:nz
        wd.hf[k, f] > MMS_LAYER_MIN || continue
        tend[k, f] = method == "layerwise" ? mms_layerwise_tendency(model, k, f) :
            SweTransport2DV._wd_advection_tendency(model, wd, k, f, method == "tvd2")
    end
    return xf, tend, [-mms_u(x) * mms_dudx(x) for x in xf]
end

# ---- Non-hydrostatic: standing wave in a closed basin ---------------------------------------------------------------
const NH_WAVE_DEPTH = 1.0        # m, still water depth of the dispersion cases
const NH_WAVE_AMPLITUDE = 1e-4   # m, linear regime

"Period (s) of a standing wave with wavenumber k (1/m) in water of depth H (m): omega^2 = g k tanh(k H) (Lamb 1932)."
standing_wave_period(k::Real, H::Real; g::Real=9.81) = 2pi / sqrt(g * k * tanh(k * H))

"""
    standing_wave_model(kH; nx_per_half_wave=40, nz=40, steps_per_period=400, kwargs...) -> (model, period_exact)

Closed basin of length L = pi H / (kH) (one half wavelength, k = pi / L) with the surface cos(pi x / L) of amplitude
NH_WAVE_AMPLITUDE; `kwargs` override configuration fields (use_hydrostatic, schemes).
"""
function standing_wave_model(kH::Real; nx_per_half_wave::Int=40, nz::Int=40, steps_per_period::Int=400, kwargs...)
    H = NH_WAVE_DEPTH
    L = pi * H / kH
    period = standing_wave_period(pi / L, H)
    cfg = wd_experiment_config(; length=L, depth=H, nx=nx_per_half_wave, nz=nz, dt=period / steps_per_period, kwargs...)
    model = Model(cfg)
    initialize_water_level!(model, H .+ NH_WAVE_AMPLITUDE .* cos.(pi .* wd_x_centres(cfg) ./ L); rho=WD_RHO0)
    return model, period
end

"Period (s) from the first zero crossing of eta at the left wall (a quarter period); NaN if none within 0.6 periods."
function standing_wave_measured_period(model::Model, period_guess::Real)
    dt = model.config.dt
    t_prev, e_prev = 0.0, model.state.eta[1]
    for n in 1:round(Int, 0.6 * period_guess / dt)
        step_forward!(model)
        e = model.state.eta[1]
        if e <= 0.0
            t = n * dt
            return 4.0 * (t_prev + e_prev / (e_prev - e) * (t - t_prev))
        end
        t_prev, e_prev = n * dt, e
    end
    return NaN
end

# ---- Solitary wave runup on a plane beach (Synolakis 1987) ------------------------------------------------------------
const SOL_DEPTH = 1.0            # m, offshore still water depth d
const SOL_COT_BETA = 19.85       # -, beach slope 1:19.85 (Synolakis 1987)
const SOL_HEIGHT = 0.019         # -, wave height H/d (non-breaking)
const SOL_STILL_LEVEL = 1.0      # m, still water level above the grid bottom (offshore bed at the grid bottom)
const SOL_GRID_DEPTH = 1.2       # m
const SOL_OFFSHORE = 60.0        # m, offshore distance of the left wall from the still shoreline
const SOL_ONSHORE = 5.0          # m, dry beach landward of the still shoreline
const SOL_SHORE_DEPTH = 1e-3     # m, depth that marks the numerical shoreline

"Maximum runup (m above the still level) of a non-breaking solitary wave on a plane beach, Synolakis (1987): R/d = 2.831 sqrt(cot beta) (H/d)^(5/4)."
solitary_runup_law() = 2.831 * sqrt(SOL_COT_BETA) * SOL_HEIGHT^(5 / 4) * SOL_DEPTH

"""
    solitary_runup_model(dx, nz; kwargs...) -> model

Plane beach 1:19.85 on the right (still shoreline at x = SOL_OFFSHORE), flat bed of depth d offshore, closed left wall.
A solitary wave eta = H sech^2(gamma (x - x1)), gamma = sqrt(3H/(4d^3)), centred half its length (to 5 % of H) seaward
of the toe of the beach, with the depth-uniform velocity u = c eta/(d + eta), c = sqrt(g (d + H)), towards the beach
(Synolakis 1987). `kwargs` override configuration fields.
"""
function solitary_runup_model(dx::Real, nz::Int; dt::Real=0.01, kwargs...)
    length = SOL_OFFSHORE + SOL_ONSHORE
    nx = round(Int, length / dx)
    cfg = wd_experiment_config(; length=length, depth=SOL_GRID_DEPTH, nx=nx, nz=nz, dt=dt, kwargs...)
    x = wd_x_centres(cfg)
    x_shore = SOL_OFFSHORE
    x_toe = x_shore - SOL_DEPTH * SOL_COT_BETA
    cfg.bed_level = [max(0.0, SOL_STILL_LEVEL - SOL_DEPTH + (xi - x_toe) / SOL_COT_BETA) for xi in x]
    model = Model(cfg)
    H = SOL_HEIGHT * SOL_DEPTH
    gamma = sqrt(3H / (4SOL_DEPTH^3))
    x1 = x_toe - acosh(sqrt(20.0)) / gamma
    eta(xx) = H * sech(gamma * (xx - x1))^2
    initialize_water_level!(model, [max(SOL_STILL_LEVEL + eta(xi), 0.0) for xi in x]; rho=WD_RHO0)
    c = sqrt(cfg.g * (SOL_DEPTH + H))
    for f in 2:nx, k in 1:nz
        xf = (f - 1) * model.dx
        model.state.u[k, f] = c * eta(xf) / (SOL_DEPTH + eta(xf))
    end
    return model
end
