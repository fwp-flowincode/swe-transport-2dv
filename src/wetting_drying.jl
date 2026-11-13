# Wetting and drying (issue #31, docs/plan_issue_31_wetting_drying.md): hydrostatic time step on a z-level grid whose
# cells are cut by the bed and by the free surface, with dry columns. Selected by ModelConfig(wetting_drying=true);
# the other time steps in src/model.jl are not used then.
#
# Geometry (all levels in m above the grid bottom): interfaces z_k = (k-1) dz, k = 1..nz+1; the top layer is open above
# (it holds all water above (nz-1) dz). Water level zeta = depth + eta; a dry column has zeta = z_b. Layer k holds
#   h_k = max(0, min(z_{k+1}, zeta) - max(z_k, z_b)).
# State fields in this mode: u (m/s) on the wet face layers (0 elsewhere), w (m/s) = vertical volume flux per unit area
# through the fixed interfaces (0 above the surface and below the bed), eta (m), rho (kg/m^3; placeholder in empty cells).
# WHY @inbounds in this file: every loop runs over 1:nz, 1:nx or 1:nx+1 of arrays allocated with exactly these sizes in
# WetDryState / Model, and neighbour indices (f-1, f+1, k-1, k+1) are guarded by the loop ranges or explicit checks.

const WD_EMPTY_CELL_THICKNESS = 1e-12   # m, a cell thinner than this is empty for the constituent transport
const WD_NEWTON_RTOL = 1e-13          # -, minimum relative Newton stopping tolerance
const WD_NEWTON_EXTRA_ITERATIONS = 10 # -, iterations allowed beyond the nx + 1 of the finite-termination bound
# -, safety factor on the round-off bound gamma_m S, m = 2 nz + 8, of a computed depth (docs/depth_positivity.md, section 5);
# a depth below -WD_POSITIVITY_ROUNDOFF_FACTOR gamma_m S stops the run
const WD_POSITIVITY_ROUNDOFF_FACTOR = 2
# -, smallest log-law height of the bed friction as a multiple of the roughness length z0 (implicit_vertical_terms):
# WHY: the cut layer at the bed can be arbitrarily thin, and z1 = hf/2 < z0 would make ln(z1/z0) <= 0; z1 >= 10 z0 caps
# the coefficient at c_b = (kappa/ln 10)^2 = 0.032
const WD_FRICTION_Z1_MIN_FACTOR = 10.0

# The WetDryState struct is defined in wetting_drying_state.jl (before Model, which holds it as a typed field).

function WetDryState(cfg::ModelConfig; T::Type{<:Real}=Float64)
    nz, nx = cfg.nz, cfg.nx
    bed = cfg.bed_level === nothing ? zeros(T, nx) : T.(cfg.bed_level)
    bed_face = zeros(T, nx + 1)
    bed_face[1], bed_face[nx + 1] = bed[1], bed[nx]
    for f in 2:nx
        bed_face[f] = max(bed[f - 1], bed[f])
    end
    return WetDryState(bed, bed_face, zeros(T, nz, nx), zeros(T, nz, nx), zeros(T, nz, nx + 1), zeros(T, nz, nx + 1),
        zeros(T, nz, nx), zeros(T, nz, nx), zeros(T, nx + 1), zeros(T, nx + 1), zeros(T, nx), zeros(T, nx), zeros(T, nx),
        zeros(T, nx), zeros(T, nx), zeros(T, nx), zeros(T, nz), zeros(T, nz), zeros(T, nz), zeros(T, nz),
        zeros(T, nz, nx + 1), zeros(T, nz + 1, nx), zeros(T, nz, nx), zeros(T, nz, nx), zeros(T, nz + 1, nx), zeros(Int, nz, nx),
        Int[], Int[], T[], T[], fill(T(-1), nz, nx + 1), 0.0, 0.0, 0, 0, nothing, fill(T(Inf), nx + 1), ones(T, nx + 1), fill(false, nx + 1),
        nothing, nothing, nothing, nothing, spzeros(T, 0, 0), Int[], Int[], Int[], 0, 0, nothing, 0, 0, 0, 0,
        RemapWork(nz, nx; T=T), zeros(T, nz), zeros(T, nz), NaN, nothing, zeros(T, nz), zeros(T, nz),
        CharacteristicFaceState(), CharacteristicFaceState())
end

"""
    layer_thicknesses!(h, zeta, zb, dz) -> h

Thicknesses (m) of the layers k = 1..nz (nz = length(h)) of a column with water level `zeta` (m) and bed level `zb` (m),
both above the grid bottom, on interfaces z_k = (k-1) dz (m):

    h_k = max(0, min(z_{k+1}, zeta) - max(z_k, zb)),   z_{nz+1} = +infinity (the top layer is open above).

sum(h) = max(zeta - zb, 0). For a flat bed and zeta in the top layer this is the variable top layer (dz + eta on top).

`merge` (-, 0 <= merge < 1; `ModelConfig.wd_top_layer = "merged"`, issue #88): a top layer s (the highest with h_s > 0)
thinner than merge * dz is joined with the layer below, if that holds water: h_{s-1} = zeta - max(z_{s-1}, zb),
h_s = 0. The top layer is then at least merge * dz thick (unless the whole column is thinner). merge = 0: no merging.
WHY: a top layer of arbitrarily small thickness makes the vertical coefficients 1/h of the pressure problem unbounded
and its interface appears and disappears whenever the surface crosses a level; joining it with the layer below is
common practice in z-layer models (e.g. the thin-top-layer treatment of Delft3D-FLOW's Z-model; reference not checked).
"""
function layer_thicknesses!(h::AbstractVector{<:Real}, zeta::Real, zb::Real, dz::Real; merge::Real=0.0)
    nz = length(h)
    s = 0
    @inbounds for k in 1:nz
        lo = max((k - 1) * dz, zb)
        hi = k == nz ? zeta : min(k * dz, zeta)
        h[k] = max(zero(eltype(h)), hi - lo)
        h[k] > 0 && (s = k)
    end
    @inbounds if s > 1 && h[s] < merge * dz && h[s - 1] > 0
        h[s - 1] = zeta - max((s - 2) * dz, zb)
        h[s] = zero(eltype(h))
    end
    return h
end

# Merge threshold (-) of the top layer for `layer_thicknesses!`: wd_merge_fraction with wd_top_layer = "merged", else 0
_wd_merge(cfg::ModelConfig) = cfg.wd_top_layer == "merged" ? cfg.wd_merge_fraction : 0.0

_wd_level(model::Model, i::Int) = model.config.depth + model.state.eta[i]   # m, water level zeta of column i

# Surface elevation eta (m) of a dry column: the smallest state-type float with fl(depth + eta) >= zb, so that the
# computed depth zeta - zb is exactly >= 0 (eta = zb - depth alone can round to one ulp below the bed).
function _wd_dry_eta(depth::Real, zb::Real, ::Type{T}) where {T<:AbstractFloat}
    eta = T(zb - depth)
    while depth + eta < zb
        eta = nextfloat(eta)
    end
    return eta
end

"""
    water_depth(model) -> Vector

Total water depth (m) per column: max(zeta - z_b, 0) with wetting and drying, depth + eta - bed_level otherwise.
"""
function water_depth(model::Model)
    cfg = model.config
    if model.wetdry !== nothing
        return [max(_wd_level(model, i) - model.wetdry.bed[i], 0.0) for i in 1:cfg.nx]
    end
    return cfg.depth .+ model.state.eta .- model.geometry.bed_level
end

"""
    initialize_water_level!(model, level; rho=nothing)

Wetting-drying mode: water at rest with the level `level` (m above the grid bottom; scalar or one value per column).
Columns whose bed lies above the level are dry (zeta = z_b). `rho` (kg/m^3): scalar or nz x nx matrix, or `nothing` to
keep the current density.
"""
function initialize_water_level!(model::Model, level::Union{Real,AbstractVector{<:Real}}; rho::Union{Nothing,Real,AbstractMatrix{<:Real}}=nothing)
    wd = model.wetdry
    wd === nothing && throw(ArgumentError("initialize_water_level! requires ModelConfig(wetting_drying=true)"))
    cfg = model.config
    st = model.state
    levels = level isa Real ? fill(float(level), cfg.nx) : collect(float.(level))
    length(levels) == cfg.nx || throw(ArgumentError("level must have nx = $(cfg.nx) entries"))
    @inbounds for i in 1:cfg.nx
        st.eta[i] = levels[i] > wd.bed[i] ? levels[i] - cfg.depth : _wd_dry_eta(cfg.depth, wd.bed[i], eltype(st.eta))
    end
    if rho isa Real
        fill!(st.rho, rho)
    elseif rho !== nothing
        size(rho) == size(st.rho) || throw(ArgumentError("rho must be nz x nx"))
        copyto!(st.rho, rho)
    end
    fill!(st.u, 0.0)
    fill!(st.w, 0.0)
    fill!(st.p, 0.0)
    _reset_nonhydrostatic_history!(model)
    wd.time = 0.0
    _refresh_density_metadata!(model)
    return nothing
end

# Clamps the water level of every column to its bed (a dry column has zeta = z_b), e.g. after a generic initializer.
function _wd_clamp_to_bed!(model::Model)
    wd = model.wetdry
    @inbounds for i in eachindex(model.state.eta)
        if model.config.depth + model.state.eta[i] < wd.bed[i]
            model.state.eta[i] = _wd_dry_eta(model.config.depth, wd.bed[i], eltype(model.state.eta))
        end
    end
    return nothing
end

# Constituent mass (kg/m): sum of rho h dx over the wet cells.
function _wd_constituent_mass(model::Model)
    wd = model.wetdry
    _wd_cell_thicknesses!(wd.h, model, wd)
    return sum(model.state.rho .* wd.h) * model.dx
end

function _wd_cell_thicknesses!(h::AbstractMatrix{<:Real}, model::Model, wd::WetDryState)
    @inbounds for i in axes(h, 2)
        layer_thicknesses!(view(h, :, i), _wd_level(model, i), wd.bed[i], model.dz; merge=_wd_merge(model.config))
    end
    return h
end

# Bed level (m) of u-face f for its layer thicknesses: wd.bed_face (the higher neighbour; the adjacent column at the
# boundaries), except at a boundary face with a known level (water-level boundary or free outflow at the right,
# left_level at the left): there the bed extrapolated linearly to the end of the domain (x = 0 or L), at least the grid
# bottom. WHY: the level of such a face is that at the end of the domain (_right_level_slope, _extrapolated_right_level),
# so its depth must be measured from the bed there. With the bed of the last column (i_b dx/2 higher on a slope), a free
# outflow over the issue #14 slope raised the level of the whole channel by 1.7 cm (the outlet face too shallow, nothing
# pinning the level); the prescribed level of #14 hid the 1 mm by fixing the outlet level.
function _wd_face_bed(model::Model, wd::WetDryState, f::Int)
    nx = model.config.nx
    nx >= 2 || return wd.bed_face[f]
    if f == nx + 1 && _right_face_computed(model)
        return max(1.5 * wd.bed[nx] - 0.5 * wd.bed[nx - 1], 0.0)
    elseif f == 1 && (model.x_boundary_eta_left !== nothing || wd.left_open !== nothing)
        return max(1.5 * wd.bed[1] - 0.5 * wd.bed[2], 0.0)
    end
    return wd.bed_face[f]
end

# Upper end (m above the grid bottom) of the flow area of face f under a sluice gate: the face bed plus the contracted
# opening, z_b,f + C_c (z_g - z_b,f) (Inf without a gate). See `set_sluice_gate!`.
function _wd_gate_level(wd::WetDryState, f::Int, zb_f::Real)
    edge = wd.gate_edge[f]
    isfinite(edge) || return edge
    return zb_f + wd.gate_contraction[f] * (edge - zb_f)
end

# Layer thicknesses at the u-faces: face bed (_wd_face_bed) and upwind water level (sign of the depth-summed u of time n;
# the higher level for u = 0), Stelling & Duinmeijer (2003), Sec. 2 (equation numbers not checked). Faces with an upwind
# depth <= dry_depth_threshold are closed (hf = 0). Boundary faces use the adjacent column. A sluice gate (issue #48)
# closes the face above its (contracted) lower edge z_g: hf_k = max(0, min(z_{k+1}, zeta_f, z_g) - max(z_k, z_b,f)), so
# the layer cut by the edge is partly open, like a layer cut by the surface; the face is closed if z_g - z_b,f is at
# most dry_depth_threshold.
"""
    _wd_characteristic_outflow_level(model, wd, f) -> zeta (m above the grid bottom) or nothing

Level that sets the layer thicknesses of a characteristic open face f (1 or nx + 1; `open_face_scheme="characteristic"`,
issue #86) while water leaves through it: the level of the adjacent column continued parallel to the bed to the face,
zeta_N + (z_b,N - z_b,N-1)/2 (right; mirror at the left), i.e. the face has the depth of that column over its
extrapolated bed (`_wd_face_bed`). nothing for inflow, a still face, other boundaries and the radiation scheme (the face
level eta_b, as before).
WHY the upwind level: the interior faces take the level of the upwind cell; with the face level eta_b instead, an
outgoing wave meets a face whose thickness lags by half a cell, which reflects to first order, about 0.27 Fr k dx
(normal-mode analysis, docs/numerics_open_boundaries.md section 8.3). WHY bed-parallel: uniform flow over a sloping bed
then keeps the same depth at the outlet face.
"""
function _wd_characteristic_outflow_level(model::Model, wd::WetDryState, f::Int)
    model.config.open_face_scheme == "characteristic" || return nothing
    nx = model.config.nx
    u = model.state.u
    if f == nx + 1 && model.x_boundary_right_free
        s = sum(view(u, :, f))
        s > 0.0 || return nothing
        return _wd_level(model, nx) + 0.5 * (wd.bed[nx] - wd.bed[nx - 1])
    elseif f == 1 && wd.left_open !== nothing
        s = sum(view(u, :, 1))
        s < 0.0 || return nothing
        return _wd_level(model, 1) + 0.5 * (wd.bed[1] - wd.bed[2])
    end
    return nothing
end

# Second-order face level of an interior face f (wd_face_level = "limited", issue #88): the upwind level plus half the
# monotonized-central slope of the upwind column (van Leer 1977),
#     zeta_f = zeta_up + MC(zeta_up - zeta_upup, zeta_down - zeta_up) / 2,
# which lies between zeta_up and zeta_down (MC <= 2 |zeta_down - zeta_up|, same sign) and equals the upwind level at an
# extremum. The upwind level `zeta_up` is kept where the column behind the upwind one is missing (f = 2, nx) or one of
# the three columns is not wet (depth <= dry_depth_threshold): the positivity of the free-surface solve needs only
# hf >= 0 and water upwind of an open face (docs/depth_positivity.md, condition C), both kept.
# WHY: the upwind level makes the continuity flux sum(hf u) a first-order upwind flux of eta, a diffusion of about
# |u_s| dx/2 on the level; a standing wave of ka = 0.13 lost about 0.012 1/s of energy rate to it at dx = 45 mm
# (issue #88). WHY no Lax-Wendroff time factor (1 - c): the flux is theta-implicit in u, not an explicit transport.
@inline function _wd_limited_face_level(model::Model, wd::WetDryState, f::Int, positive::Bool, zeta_up::Real)
    nx = model.config.nx
    iu, iuu, id = positive ? (f - 1, f - 2, f) : (f, f + 1, f - 1)
    1 <= iuu <= nx || return zeta_up
    thr = model.config.dry_depth_threshold
    zuu, zd = _wd_level(model, iuu), _wd_level(model, id)
    (zeta_up - wd.bed[iu] > thr && zuu - wd.bed[iuu] > thr && zd - wd.bed[id] > thr) || return zeta_up
    return zeta_up + 0.5 * _monotonized_central_scalar(zeta_up - zuu, zd - zeta_up)
end

function _wd_face_thicknesses!(model::Model, wd::WetDryState)
    cfg = model.config
    u = model.state.u
    nx = cfg.nx
    limited = cfg.wd_face_level == "limited"
    @inbounds for f in 1:(nx + 1)
        if (f == nx + 1 || f == 1) && (zc = _wd_characteristic_outflow_level(model, wd, f)) !== nothing
            zeta_f = zc   # characteristic face with outflow: the upwind level (issue #86)
        elseif f == nx + 1 && _right_boundary_level(model) !== nothing
            zeta_f = cfg.depth + _right_boundary_level(model)   # water level or free outflow: the face level is known
        elseif f == 1 && model.x_boundary_eta_left !== nothing
            zeta_f = cfg.depth + model.x_boundary_eta_left    # inlet level prescribed (supercritical inflow, #46)
        elseif f == 1 && wd.left_open !== nothing
            zeta_f = cfg.depth + wd.left_open.eta_b           # free left face (issue #86): the face level is known
        elseif f == 1 || f == nx + 1
            zeta_f = _wd_level(model, f == 1 ? 1 : nx)
        else
            s = zero(eltype(u))
            for k in axes(u, 1)
                s += u[k, f]
            end
            zl, zr = _wd_level(model, f - 1), _wd_level(model, f)
            zeta_f = s > 0 ? zl : (s < 0 ? zr : max(zl, zr))
            limited && s != 0 && (zeta_f = _wd_limited_face_level(model, wd, f, s > 0, zeta_f))
        end
        hf = view(wd.hf, :, f)
        zb_f = _wd_face_bed(model, wd, f)
        zg = _wd_gate_level(wd, f, zb_f)
        wd.gate_active[f] = zeta_f > zg
        if zeta_f - zb_f > cfg.dry_depth_threshold && zg - zb_f > cfg.dry_depth_threshold
            layer_thicknesses!(hf, min(zeta_f, zg), zb_f, model.dz; merge=_wd_merge(cfg))
        else
            fill!(hf, 0.0)
        end
    end
    return nothing
end

const GATE_FACE_POSITION_TOL = 1e-6   # -, allowed distance of a gate position from a u-face, in cells

"""
    set_sluice_gate!(model, x; lower_edge, contraction=1.0)

Sluice gate (issue #48) at the interior u-face at the position `x` (m, a multiple of dx): a thin vertical plate that closes
the water column above its lower edge `lower_edge` (m above the grid bottom, at or above the bed of the face; `Inf`
removes the gate). Wetting-drying step only (`ModelConfig(wetting_drying=true)`); the gate may be moved between steps.

The face carries water only below z_g = z_b,f + C_c (lower_edge - z_b,f) (`_wd_face_thicknesses!`): layers above it are
closed, the layer cut by it is partly open. While the upstream level lies below the edge the gate has no effect.

`contraction` (C_c, 0 < C_c <= 1) is the contraction coefficient of the jet: a hydrostatic model has no vena contracta
(the streamlines cannot curve), so the jet leaves the gate with the full opening a = lower_edge - z_b,f unless the
opening is narrowed to C_c a. With it, the step reproduces the classical gate relations (docs section 26):
- free flow: the upstream energy head is E_1 = C_c a + q^2 / (2 g (C_c a)^2) (Bernoulli up to the vena contracta; the
  momentum-conservative advection of Stelling & Duinmeijer (2003) conserves the energy head in a contraction);
- submerged flow: Bernoulli up to the vena contracta and the momentum balance from there to the tailwater (Henderson
  1966, Open Channel Flow, ch. 6; the advection conserves momentum in an expansion).
C_c = 0.61 is the classical value for a vertical gate with a small opening (Henderson 1966, sec. 6.4; von Mises 1917).
The non-hydrostatic mode resolves the contraction itself, so it requires `contraction=1` (the gate edge is then a solid
boundary of the pressure problem, through the face thicknesses).
"""
function set_sluice_gate!(model::Model, x::Real; lower_edge::Real, contraction::Real=1.0)
    wd = model.wetdry
    wd === nothing && throw(ArgumentError("set_sluice_gate! requires ModelConfig(wetting_drying=true)"))
    cfg = model.config
    cells = x / model.dx
    f = round(Int, cells) + 1   # u-face index
    abs(cells - (f - 1)) <= GATE_FACE_POSITION_TOL || throw(ArgumentError(@sprintf("gate position x = %.6g m is not a u-face (dx = %.6g m)", x, model.dx)))
    2 <= f <= cfg.nx || throw(ArgumentError(@sprintf("gate position x = %.6g m must be an interior u-face (0 < x < %.6g m)", x, cfg.length)))
    lower_edge >= wd.bed_face[f] || throw(ArgumentError(@sprintf("lower_edge = %.6g m lies below the bed of the face (%.6g m)", lower_edge, wd.bed_face[f])))
    0.0 < contraction <= 1.0 || throw(ArgumentError("contraction must be in (0, 1]"))
    if contraction != 1.0 && !cfg.use_hydrostatic && cfg.nz > 1
        throw(ArgumentError("the non-hydrostatic mode resolves the contraction of the gate jet; use contraction=1"))
    end
    wd.gate_edge[f] = lower_edge
    wd.gate_contraction[f] = contraction
    return nothing
end

"""
    clear_sluice_gates!(model)

Removes all sluice gates (`set_sluice_gate!`).
"""
function clear_sluice_gates!(model::Model)
    wd = model.wetdry
    wd === nothing && return nothing
    fill!(wd.gate_edge, Inf)
    fill!(wd.gate_contraction, 1.0)
    return nothing
end

# Transport (m^2/s per unit of the shape) of the profile `shape` (length nz) at the left boundary face with the face
# layers that the next step uses (`_wd_face_thicknesses!`: level of column 1 above the bed of face 1; a dry face carries
# none). PERF: allocates one vector of length nz; called by set_x_inflow_outflow!, not inside the time step.
function _wd_left_face_transport(model::Model, shape::AbstractVector{<:Real})
    wd = model.wetdry
    eta_left = model.x_boundary_eta_left
    zeta = eta_left === nothing ? _wd_level(model, 1) : model.config.depth + eta_left   # m, level of the inlet face
    zb = _wd_face_bed(model, wd, 1)
    zeta - zb > model.config.dry_depth_threshold || return 0.0
    h = zeros(eltype(wd.hf), model.config.nz)
    layer_thicknesses!(h, zeta, zb, model.dz; merge=_wd_merge(model.config))
    return sum(h .* shape)
end

# Baroclinic pressure p' = g * integral_z^zeta (rho - rho0) dz' (Pa) at the nominal layer centres z_c = z_k + dz/2. Layers
# above the surface extrapolate with the density of the surface layer.
# WHY nominal centres (not the centroids of cut cells): a horizontally uniform stratification at rest then gives
# bit-identical p' in neighbouring columns whatever their bed, i.e. no spurious baroclinic force (lake at rest).
# `rho` (kg/m^3): the buoyant density of time n (the mixture density with coupled sediment, issue #64), or its
# extrapolation for the momentum predictor (_baroclinic_density, issue #23).
function _wd_baroclinic_pressure!(model::Model, wd::WetDryState; rho::AbstractMatrix=_wd_buoyant_density(model, wd))
    cfg = model.config
    dz = model.dz
    nz = cfg.nz
    @inbounds for i in 1:cfg.nx
        zeta = _wd_level(model, i)
        s = 0
        for k in nz:-1:1
            if wd.h[k, i] > 0.0
                s = k
                break
            end
        end
        if s == 0
            for k in 1:nz
                wd.p_anom[k, i] = 0.0
            end
            continue
        end
        drho_s = rho[s, i] - cfg.rho0
        for k in nz:-1:s
            wd.p_anom[k, i] = cfg.g * drho_s * (zeta - ((k - 0.5) * dz))
        end
        acc = drho_s * wd.h[s, i]
        for k in (s - 1):-1:1
            wd.p_anom[k, i] = cfg.g * ((rho[k, i] - cfg.rho0) * 0.5 * dz + acc)
            acc += (rho[k, i] - cfg.rho0) * wd.h[k, i]
        end
    end
    return nothing
end

# Upwind value at a cell centre or interface for the momentum advection: first order u_up, or (second) the flux-limited
# Lax-Wendroff value u_up + (1 - c) s/2 with the MC slope s = MC(u_up - u_upup, u_down - u_up) of the upwind node (van
# Leer 1977) and the Courant number c (-) of the flux, first order where the node behind it is not available (`has_upup`
# false: boundary or dry face). The MC slope satisfies |s/2| <= |u_down - u_up| with the same sign, so the value lies
# between u_up and u_down.
@inline function _wd_upwind_value(u_up::Real, u_upup::Real, u_down::Real, has_upup::Bool, second::Bool, courant::Real)
    (second && has_upup) || return u_up
    return u_up + 0.5 * (1.0 - min(courant, 1.0)) * _monotonized_central_scalar(u_up - u_upup, u_down - u_up)
end

# Nearest layer at or below k that is open at face f (0 if none).
function _wd_open_layer_at_or_below(wd::WetDryState, k::Int, f::Int)
    (1 <= f <= size(wd.hf, 2)) || return 0
    @inbounds for kk in k:-1:1
        wd.hf[kk, f] > 0.0 && return kk
    end
    return 0
end

# Column totals (m) of the advective weights at face f: Hbar_f (cell thicknesses averaged to the face) and Htilde_L,
# Htilde_R (face thicknesses averaged to cells L and R); see `_wd_advection_tendency`.
# PERF: independent of the layer, so the predictor computes them once per face (O(nx nz) instead of O(nx nz^2)).
function _wd_advection_depth_totals(wd::WetDryState, f::Int)
    nx = size(wd.h, 2)
    R = min(f, nx)                  # right boundary face (water-level boundary): ghost cell = last cell
    L = max(f - 1, 1)               # free left face (issue #86): ghost cell = first cell
    fR = min(f + 1, size(wd.hf, 2))
    fL = max(f - 1, 1)
    Hbar = 0.0
    HtL = 0.0
    HtR = 0.0
    @inbounds for kk in axes(wd.h, 1)
        Hbar += 0.5 * (wd.h[kk, L] + wd.h[kk, R])
        HtL += 0.5 * (wd.hf[kk, fL] + wd.hf[kk, f])
        HtR += 0.5 * (wd.hf[kk, f] + wd.hf[kk, fR])
    end
    return Hbar, HtL, HtR
end

# Advection tendency du/dt of face f, layer k: upwind, momentum conservative in the form of Stelling & Duinmeijer (2003,
# Sec. 3; equation numbers not checked), horizontally with the cell discharges and vertically with the interface fluxes
# W of time n:
#   du/dt = [ a_L (u*_L - u) - a_R (u*_R - u) ] / dx + [ W_bot (u*_bot - u) - W_top (u*_top - u) ] / hbar,
# with u* the value upwinded by the sign of each flux: the upwind velocity (upwind1) or its limited extrapolation with the
# Lax-Wendroff time correction (tvd2, `_wd_upwind_value`): u* = u_up + (1 - c)/2 s_up with the Courant number c of the
# flux (|a| dt/dx, |W| dt/hbar), as the momentum tvd2 of the other modes (`_advective_flux_divergence!`) and the density
# transport. On an outflow side the upwind node is face f itself, so u* - u = 0 for upwind1 (a sum over the inflow sides,
# sum_j c_j (u_j - u), c_j >= 0) and u* - u = (1 - c) s_f/2 for tvd2.
# WHY the time correction: with forward Euler the plain MUSCL extrapolation (c = 0) is linearly unstable at every Courant
# number; in smooth regions (Fromm's slope) the amplification factor of a 4.6 dx wave is 1.15 per step at c = 0.6 and
# 1.005 at c = 0.2 (von Neumann analysis, test_verify_wd_tvd2_momentum_von_neumann). The limiter only stops the growth at
# extrema: in pure advection it stays bounded, but in a weakly stratified layer the internal-wave coupling let a 4 dx
# wave grow until the run failed (splitter plate, issue #42; fixed in #66). With the correction the scheme is the flux-limited
# Lax-Wendroff scheme, |G| <= 1 for c <= 1. Sweby (1984), SIAM J. Numer. Anal. 21, 995-1011; LeVeque (2002), Finite
# Volume Methods for Hyperbolic Problems, ch. 6 (equation numbers not checked).
#
# Advective velocities a_L, a_R of the cells left and right of the face (docs/TESTS_TECHNICAL_DOCUMENTATION.md,
# section "Wetting and drying: depth-uniform velocity"):
#   a_C = (qbar_{k,C} / htilde_{k,C}) * (Htilde_C / Hbar_f),
#   qbar_{k,C} = mean of the layer discharges of the two faces of cell C, htilde_{k,C} = mean of their layer thicknesses
#   (so qbar/htilde is the layer velocity of the cell), Htilde_C = sum_k htilde_{k,C}, Hbar_f = sum_k of the cell
#   thicknesses averaged to face f.
# For one layer this is the 1D weight qbar/hbar of Stelling & Duinmeijer. WHY not qbar_k / hbar_k per layer (the direct
# layer-wise form): its error relative to u is about dx |d zeta/dx| / h_k, which is O(1) wherever the surface cuts a layer
# thinly (every interface crossing) and does not vanish under refinement; it made the layers of a depth-uniform flow
# drift apart (2DV Ritter: 0.9 m/s spread). With the column factor, every layer of a depth-uniform flow has exactly the
# advective velocity of the 1D scheme, so 2DV reproduces 1D (verified by an operator-level manufactured solution).
# WHY the closed-face rule: if the far face of the inflow cell is closed for this layer (its upwind level lies below the
# layer), the water entering the layer there comes from the nearest open layer below at that face and rises into it; its
# velocity is the upwind value (no inflow if the face is closed in all layers below, e.g. dry or blocked by the bed).
# Counting the closed face as inflow with u = 0 would drag the face towards rest.
# WHY consistent stencils: tvd2 uses second order in a direction only if the whole stencil (faces f-2..f+2, layers
# k-2..k+2) is open; a one-sided mix of a second-order inflow and a first-order outflow value is inconsistent (it halves
# the tendency next to a closed face).
# WHY the cap: with C = dt * (sum of the inflow coefficients), the tendency is scaled by 1/C where C > 1; for upwind1 this
# keeps the update a convex combination of the neighbours (thin layers stay stable).
# `depth_totals` = (Hbar_f, Htilde_L, Htilde_R) of face f from `_wd_advection_depth_totals`.
function _wd_advection_tendency(model::Model, wd::WetDryState, k::Int, f::Int, second::Bool, depth_totals;
    vertical::Bool=true)
    u = model.state.u
    W = model.state.w
    nz = model.config.nz
    nxp1 = model.config.nx + 1
    dx = model.dx
    # right boundary face (f = nx + 1, water-level boundary): the ghost cell beyond it has the thicknesses and w of the
    # last cell and the velocity of the boundary face (zero gradient), so the cell R terms vanish (aR = 0 below)
    # free left face (f = 1, issue #86): the mirror, ghost cell = first cell, aL = 0
    L, R = max(f - 1, 1), min(f, nxp1 - 1)
    hbar = 0.5 * (wd.h[k, L] + wd.h[k, R])
    hbar > WD_EMPTY_CELL_THICKNESS || return 0.0
    uf = u[k, f]
    open_h(ff) = 1 <= ff <= nxp1 && wd.hf[k, ff] > 0.0
    open_v(kk) = 1 <= kk <= nz && wd.hf[kk, f] > 0.0
    dt = model.config.dt
    csum = 0.0   # 1/s, sum of the inflow coefficients
    tend = 0.0   # m/s^2
    @inbounds begin
        # advective velocities of cells L and R (m/s)
        Hbar, HtL, HtR = depth_totals
        aL = 0.0
        if f > 1
            htL = 0.5 * (wd.hf[k, f - 1] + wd.hf[k, f])
            aL = htL > 0.0 ? 0.5 * (wd.q[k, f - 1] + wd.q[k, f]) / htL * HtL / Hbar : 0.0
        end
        aR = 0.0
        if f < nxp1
            htR = 0.5 * (wd.hf[k, f] + wd.hf[k, f + 1])
            aR = htR > 0.0 ? 0.5 * (wd.q[k, f] + wd.q[k, f + 1]) / htR * HtR / Hbar : 0.0
        end

        # sluice gate cutting the face (issue #48): energy-head form in the contraction into the gate face, first order
        gate = wd.gate_active[f]
        if gate
            aL > 0.0 && (aL = max(0.5 * (u[k, f - 1] + uf), 0.0))
            f < nxp1 && aR < 0.0 && (aR = min(0.5 * (u[k, f + 1] + uf), 0.0))
        end

        # horizontal
        h2 = second && !gate && open_h(f - 2) && open_h(f - 1) && open_h(f + 1) && open_h(f + 2)
        kin = aL > 0.0 ? _wd_open_layer_at_or_below(wd, k, f - 1) : 0
        if kin > 0                         # inflow from cell L: upwind node f-1, behind it f-2
            c = aL / dx
            csum += c
            uin = kin == k ? _wd_upwind_value(u[k, f - 1], h2 ? u[k, f - 2] : 0.0, uf, h2, true, c * dt) : u[kin, f - 1]
            tend += c * (uin - uf)
        elseif aL < 0.0 && h2              # outflow into cell L: upwind node f, behind it f+1
            tend += aL / dx * (_wd_upwind_value(uf, u[k, f + 1], u[k, f - 1], true, true, -aL * dt / dx) - uf)
        end
        kin = aR < 0.0 ? _wd_open_layer_at_or_below(wd, k, f + 1) : 0
        if kin > 0                         # inflow from cell R: upwind node f+1, behind it f+2
            c = -aR / dx
            csum += c
            uin = kin == k ? _wd_upwind_value(u[k, f + 1], h2 ? u[k, f + 2] : 0.0, uf, h2, true, c * dt) : u[kin, f + 1]
            tend += c * (uin - uf)
        elseif aR > 0.0 && h2              # outflow into cell R: upwind node f, behind it f-1
            tend -= aR / dx * (_wd_upwind_value(uf, u[k, f - 1], u[k, f + 1], true, true, aR * dt / dx) - uf)
        end

        # vertical: interface k (below layer k) and k+1 (above); implicit instead with implicit_vertical_terms
        v2 = second && open_v(k - 2) && open_v(k - 1) && open_v(k + 1) && open_v(k + 2)
        if vertical && open_v(k - 1)
            w_bot = 0.5 * (W[k, L] + W[k, R])
            if w_bot > 0.0                 # inflow from below: upwind node k-1, behind it k-2
                c = w_bot / hbar
                csum += c
                tend += c * (_wd_upwind_value(u[k - 1, f], v2 ? u[k - 2, f] : 0.0, uf, v2, true, c * dt) - uf)
            elseif w_bot < 0.0 && v2       # outflow downwards: upwind node k, behind it k+1
                tend += w_bot / hbar * (_wd_upwind_value(uf, u[k + 1, f], u[k - 1, f], true, true, -w_bot * dt / hbar) - uf)
            end
        end
        if vertical && open_v(k + 1)
            w_top = 0.5 * (W[k + 1, L] + W[k + 1, R])
            if w_top < 0.0                 # inflow from above: upwind node k+1, behind it k+2
                c = -w_top / hbar
                csum += c
                tend += c * (_wd_upwind_value(u[k + 1, f], v2 ? u[k + 2, f] : 0.0, uf, v2, true, c * dt) - uf)
            elseif w_top > 0.0 && v2       # outflow upwards: upwind node k, behind it k-1
                tend -= w_top / hbar * (_wd_upwind_value(uf, u[k - 1, f], u[k + 1, f], true, true, w_top * dt / hbar) - uf)
            end
        end
    end
    return dt * csum > 1.0 ? tend / (dt * csum) : tend
end

_wd_advection_tendency(model::Model, wd::WetDryState, k::Int, f::Int, second::Bool) =
    _wd_advection_tendency(model, wd, k, f, second, _wd_advection_depth_totals(wd, f))

# ---- Remapped momentum advection (issue #80) ----------------------------------------------------------------------------
# Kleptsova, Stelling & Pietrzak (2010), Ocean Dyn. 60, 1447-1461, sec. 6.1, eqs. 29-33: a flow without vertical structure
# must be advected as in 1D in every layer, which needs h_{f,k} / h_{c,k} = h_f / h_c for every face f and neighbouring
# cell c (their eq. 29). The z-layers do not have it where the depth changes (a bore, a slope of the surface across an
# interface), and the column-scaled advective velocity of `_wd_advection_tendency` repairs it only for the advective
# velocity, not for the momentum fluxes: it lost 0.9 % of the momentum of a Stoker bore in 10 layers. Here the momentum
# equation of face f, layer k, is discretised on sub-layers with the layer shares of face f, phi_k = hf_k / H_f (the
# sigma interval [s_lo, s_hi] of the face depth): in the cells L and R of the face and at the neighbouring faces g the
# sub-layer k is the same interval of their own depth, its discharge q'_{g,k} the integral of u over that interval
# (piecewise linear in the layers of g, their eq. 33), the advection thickness phi_k Hbar_f (Hbar_f the cell depths
# averaged to the face), and the vertical fluxes between the sub-layers follow from their continuity,
#   omega_C(s) = -[(Qc_b(s) - Qc_a(s)) - s (Q_b - Q_a)] / dx      (Qc_g(s) = integral of u over [0, s H_g] of face g),
# zero for a flow without vertical structure. The tendency is the upwind form of `_wd_advection_tendency` (Stelling &
# Duinmeijer 2003) on these sub-layers, with the same tvd2 extension and cap. For a single layer it is that 1D scheme.

# Faces f-2..f+2 over a flat bed (the face beds and the beds of the cells between them equal) and without an active sluice
# gate: there the remapped form applies. WHY: the sub-layers follow the depth of each column (sigma-like); where the depth
# changes with the bed (a step, an obstacle, a gate) that transports momentum along tilted surfaces instead of
# horizontally, and behind a backward step the recirculation vanished (reattachment length 0 instead of 6 step heights).
# Over a flat bed the depth changes only with the surface, where the remapping is right (a bore, a surge).
# Tolerances of the flat-bed test, for beds of type T: absolute eps(T) (m) and relative sqrt(eps(T)) (-). WHY: bed levels
# that differ only by the round-off of their computation (a bed given by a formula) count as flat, a real step does not
# (1e-6 m at a bed level of 0 m is a step in Float64).
_wd_remap_flat_bed_atol(::Type{T}) where {T<:AbstractFloat} = eps(T)
_wd_remap_flat_bed_rtol(::Type{T}) where {T<:AbstractFloat} = sqrt(eps(T))

function _wd_remap_flat_neighbourhood(wd::WetDryState, f::Int)
    nf = length(wd.bed_face)
    lo, hi = max(f - 2, 1), min(f + 2, nf)
    b = wd.bed_face[f]
    T = eltype(wd.bed_face)
    atol, rtol = _wd_remap_flat_bed_atol(T), _wd_remap_flat_bed_rtol(T)
    @inbounds for g in lo:hi
        isapprox(wd.bed_face[g], b; atol=atol, rtol=rtol) || return false
        wd.gate_active[g] && return false
    end
    @inbounds for c in max(lo, 1):min(hi - 1, length(wd.bed))
        isapprox(wd.bed[c], b; atol=atol, rtol=rtol) || return false
    end
    return true
end

# Cumulative thickness and discharge of the open layers of face g (index j of the five faces f-2..f+2; a face outside
# the domain is closed).
function _wd_remap_face!(R::RemapWork, j::Int, wd::WetDryState, u::AbstractMatrix, g::Int)
    R.zc[1, j] = 0.0
    R.qc[1, j] = 0.0
    n = 0
    if 1 <= g <= size(wd.hf, 2)
        @inbounds for k in axes(wd.hf, 1)
            h = wd.hf[k, g]
            h > 0.0 || continue
            n += 1
            R.zc[n + 1, j] = R.zc[n, j] + h
            R.qc[n + 1, j] = R.qc[n, j] + h * u[k, g]
        end
    end
    R.count[j] = n
    R.depth[j] = n > 0 ? R.zc[n + 1, j] : 0.0
    return nothing
end

# Integral of u over [0, s H_g] of face j (m^2/s); piecewise linear in the layers
@inline function _wd_remap_cumulative(R::RemapWork{T}, j::Int, s::Real) where T
    n = R.count[j]
    n == 0 && return zero(T)
    z = T(s) * R.depth[j]
    i = 1
    @inbounds while i < n && R.zc[i + 1, j] < z
        i += 1
    end
    @inbounds begin
        dzl = R.zc[i + 1, j] - R.zc[i, j]
        return R.qc[i, j] + (dzl > zero(T) ? (z - R.zc[i, j]) / dzl * (R.qc[i + 1, j] - R.qc[i, j]) : zero(T))
    end
end

# Mean velocity of face j over the sigma interval [s0, s1] (m/s); NaN if the face is closed
@inline function _wd_remap_velocity(R::RemapWork{T}, j::Int, s0::Real, s1::Real) where T
    R.count[j] == 0 && return T(NaN)
    lo, hi = T(s0), T(s1)
    return (_wd_remap_cumulative(R, j, hi) - _wd_remap_cumulative(R, j, lo)) / ((hi - lo) * R.depth[j])
end

@inline function _wd_remap_omega(R::RemapWork{T}, s::Real, Q2::T, Q3::T, Q4::T, dx::Real) where T
    sT = T(s)
    q2 = _wd_remap_cumulative(R, 2, sT)
    q3 = _wd_remap_cumulative(R, 3, sT)
    q4 = _wd_remap_cumulative(R, 4, sT)
    return -T(0.5) * ((q3 - q2 - sT * (Q3 - Q2)) + (q4 - q3 - sT * (Q4 - Q3))) / T(dx)
end

# Advection tendencies (m/s^2) of the open layers of face f (2 <= f <= nx) into R.tend, and the sigma-layer fluxes and
# advection thicknesses of the face into R.omega[:, f], R.hadv[:, f] (for the implicit vertical solve). `second`: tvd2,
# `vertical`: explicit vertical advection (otherwise the implicit vertical solve takes R.omega, R.hadv).
function _wd_remapped_tendencies!(model::Model, wd::WetDryState, f::Int, second::Bool, vertical::Bool)
    R = wd.remap
    u = model.state.u
    nz = model.config.nz
    dx, dt = model.dx, model.config.dt
    for j in 1:5
        _wd_remap_face!(R, j, wd, u, f - 3 + j)
    end
    # faces: 1 = f-2, 2 = f-1, 3 = f, 4 = f+1, 5 = f+2
    Hf = R.depth[3]
    Hbar = 0.0
    @inbounds for k in 1:nz
        Hbar += 0.5 * (wd.h[k, f - 1] + wd.h[k, f])
        R.tend[k] = 0.0
        R.omega[k, f] = 0.0
        R.hadv[k, f] = floatmax(eltype(R.hadv))
    end
    Hf > 0.0 || return R
    R.active[f] = true
    Q2, Q3, Q4 = R.qc[R.count[2] + 1, 2], R.qc[R.count[3] + 1, 3], R.qc[R.count[4] + 1, 4]
    # sigma-layer flux through the level s of the face control volume: mean of cells L (faces f-1, f) and R (f, f+1)
    h2 = second && all(j -> R.count[j] > 0, 1:5)   # second order horizontally: all five faces open
    s_lo = 0.0
    @inbounds for k in 1:nz
        wd.hf[k, f] > 0.0 || continue
        s_hi = min(s_lo + wd.hf[k, f] / Hf, 1.0)
        hk = (s_hi - s_lo) * Hbar   # m, advection thickness of the sub-layer
        R.omega[k, f] = s_lo > 0.0 ? _wd_remap_omega(R, s_lo, Q2, Q3, Q4, dx) : 0.0
        if hk <= WD_EMPTY_CELL_THICKNESS
            s_lo = s_hi
            continue
        end
        R.hadv[k, f] = hk
        uf = u[k, f]
        um1, up1 = _wd_remap_velocity(R, 2, s_lo, s_hi), _wd_remap_velocity(R, 4, s_lo, s_hi)
        um2 = h2 ? _wd_remap_velocity(R, 1, s_lo, s_hi) : 0.0
        up2 = h2 ? _wd_remap_velocity(R, 5, s_lo, s_hi) : 0.0
        q_m1 = isnan(um1) ? 0.0 : um1 * (s_hi - s_lo) * R.depth[2]
        q_p1 = isnan(up1) ? 0.0 : up1 * (s_hi - s_lo) * R.depth[4]
        q_f = uf * wd.hf[k, f]
        QL, QR = 0.5 * (q_m1 + q_f), 0.5 * (q_f + q_p1)   # m^2/s, sub-layer discharges of cells L and R
        aL, aR = QL / hk, QR / hk                          # m/s, advective velocities
        csum, tend = 0.0, 0.0
        if aL > 0.0 && !isnan(um1)                         # inflow from cell L
            c = aL / dx
            csum += c
            tend += c * (_wd_upwind_value(um1, um2, uf, h2, true, c * dt) - uf)
        elseif aL < 0.0 && h2                              # outflow into cell L (tvd2 correction)
            tend += aL / dx * (_wd_upwind_value(uf, up1, um1, true, true, -aL * dt / dx) - uf)
        end
        if aR < 0.0 && !isnan(up1)                         # inflow from cell R
            c = -aR / dx
            csum += c
            tend += c * (_wd_upwind_value(up1, up2, uf, h2, true, c * dt) - uf)
        elseif aR > 0.0 && h2                              # outflow into cell R (tvd2 correction)
            tend -= aR / dx * (_wd_upwind_value(uf, um1, up1, true, true, aR * dt / dx) - uf)
        end
        if vertical
            open_v(kk) = 1 <= kk <= nz && wd.hf[kk, f] > 0.0
            v2 = second && open_v(k - 2) && open_v(k - 1) && open_v(k + 1) && open_v(k + 2)
            if open_v(k - 1)
                wb = R.omega[k, f]
                if wb > 0.0
                    c = wb / hk
                    csum += c
                    tend += c * (_wd_upwind_value(u[k - 1, f], v2 ? u[k - 2, f] : 0.0, uf, v2, true, c * dt) - uf)
                elseif wb < 0.0 && v2
                    tend += wb / hk * (_wd_upwind_value(uf, u[k + 1, f], u[k - 1, f], true, true, -wb * dt / hk) - uf)
                end
            end
            if open_v(k + 1)
                wt = _wd_remap_omega(R, s_hi, Q2, Q3, Q4, dx)
                if wt < 0.0
                    c = -wt / hk
                    csum += c
                    tend += c * (_wd_upwind_value(u[k + 1, f], v2 ? u[k + 2, f] : 0.0, uf, v2, true, c * dt) - uf)
                elseif wt > 0.0 && v2
                    tend -= wt / hk * (_wd_upwind_value(uf, u[k - 1, f], u[k + 1, f], true, true, wt * dt / hk) - uf)
                end
            end
        end
        R.tend[k] = dt * csum > 1.0 ? tend / (dt * csum) : tend
        s_lo = s_hi
    end
    return R
end

# Largest explicit horizontal diffusion number dt * (sum of the coefficients of a node or cell), -: 1/2 keeps the update a
# convex combination of the neighbours in the arbitrarily thin cut layers
const WD_HORIZONTAL_DIFFUSION_MAX = 0.5

# Horizontal eddy diffusion of u at face f (m/s^2), explicit, with nu_t (m^2/s) of time n in the cells either side:
#   hf_f d(u_f)/dt = [h_R nu_t,R (u_{f+1} - u_f) - h_L nu_t,L (u_f - u_{f-1})] / dx^2.
# Second order on a uniform grid. Zero flux where the neighbouring face is closed (the face of a step: free slip).
# Stability: dt (c_L + c_R) <= WD_HORIZONTAL_DIFFUSION_MAX, enforced by scaling the term in thin layers.
@inline function _wd_horizontal_eddy_diffusion(model::Model, wd::WetDryState, ke::KEpsilonState, k::Int, f::Int)
    u, dx, nxp1 = model.state.u, model.dx, model.config.nx + 1
    hf = wd.hf[k, f]
    @inbounds begin
        cL = f > 1 && wd.hf[k, f - 1] > 0.0 ? ke.nu_t[k, f - 1] * wd.h[k, f - 1] / (hf * dx^2) : 0.0   # 1/s
        cR = f < nxp1 && wd.hf[k, f + 1] > 0.0 ? ke.nu_t[k, f] * wd.h[k, f] / (hf * dx^2) : 0.0
        csum = cL + cR
        csum > 0.0 || return 0.0
        tend = (cL > 0.0 ? cL * (u[k, f - 1] - u[k, f]) : 0.0) + (cR > 0.0 ? cR * (u[k, f + 1] - u[k, f]) : 0.0)
    end
    return tend * min(1.0, WD_HORIZONTAL_DIFFUSION_MAX / (model.config.dt * csum))
end

# Implicit vertical viscosity nu (m^2/s) on the wet layers of face f (Thomas algorithm), zero stress at bed and surface:
#   hf_k (u_k - uhat_k)/dt = nu (u_{k+1} - u_k)/dz_{k+1/2} - nu (u_k - u_{k-1})/dz_{k-1/2},  dz_{k+1/2} = (hf_k + hf_{k+1})/2.
# WHY implicit: the cut layers at the bed and the surface can be arbitrarily thin.
function _wd_vertical_viscosity!(model::Model, wd::WetDryState, u_hat::AbstractMatrix{<:Real}, f::Int)
    nu = model.config.nu
    nu > 0.0 || return nothing
    dt = model.config.dt
    nz = model.config.nz
    a, b, c, d = wd.tri_a, wd.tri_b, wd.tri_c, wd.tri_d
    @inbounds for k in 1:nz
        hk = wd.hf[k, f]
        if hk <= 0.0
            a[k], b[k], c[k], d[k] = 0.0, 1.0, 0.0, u_hat[k, f]
            continue
        end
        lo = (k > 1 && wd.hf[k - 1, f] > 0.0) ? dt * nu / (0.5 * (hk + wd.hf[k - 1, f])) : 0.0
        up = (k < nz && wd.hf[k + 1, f] > 0.0) ? dt * nu / (0.5 * (hk + wd.hf[k + 1, f])) : 0.0
        a[k], b[k], c[k], d[k] = -lo, hk + lo + up, -up, hk * u_hat[k, f]
    end
    @inbounds for k in 2:nz
        m = a[k] / b[k - 1]
        b[k] -= m * c[k - 1]
        d[k] -= m * d[k - 1]
    end
    @inbounds u_hat[nz, f] = d[nz] / b[nz]
    @inbounds for k in (nz - 1):-1:1
        u_hat[k, f] = (d[k] - c[k] * u_hat[k + 1, f]) / b[k]
    end
    return nothing
end

# Implicit vertical terms of face f (implicit_vertical_terms=true; src/implicit_vertical.jl): viscosity nu, upwind
# vertical advection in the advective form of `_wd_advection_tendency` (interface fluxes of time n averaged to the face,
# advection thickness hbar of the two cells) and log-law bed friction c_b |u^n| on the lowest open layer
# (z1 = max(hf/2, WD_FRICTION_Z1_MIN_FACTOR z0)), on the open layers of the face. Replaces u_hat there by
# A^{-1}(hf .* u_hat) and writes the column response ws.r_vert and the effective depth ws.H_face[f] = sum_k hf_k r_k.
function _wd_implicit_vertical!(model::Model, wd::WetDryState, f::Int)
    cfg = model.config
    ws = model.ws
    W = model.state.w
    nz = cfg.nz
    col = ws.col
    klo, khi = 0, 0
    @inbounds for k in 1:nz
        if wd.hf[k, f] > 0.0
            klo == 0 && (klo = k)
            khi = k
        end
    end
    @inbounds for k in 1:nz
        ws.r_vert[k, f] = 0.0
    end
    if klo == 0
        ws.H_face[f] = 0.0
        return nothing
    end
    m = khi - klo + 1
    # open boundary faces (water level or free outflow at the right, free left face): w and h of the adjacent cell
    L, R = max(f - 1, 1), min(f, cfg.nx)
    ke = model.turbulence
    @inbounds for j in 1:m
        k = klo + j - 1
        col[j, 3] = ws.u_hat[k, f]
        col[j, 4] = wd.hf[k, f]
        # interface below layer k; remapped faces (issue #80): the sigma-layer flux of `_wd_remapped_tendencies!`
        col[j, 5] = wd.remap.active[f] ? (j == 1 ? 0.0 : wd.remap.omega[k, f]) : 0.5 * (W[k, L] + W[k, R])
        # interface below layer k: molecular plus eddy viscosity (k-epsilon, issue #30), face values of the cells L, R
        col[j, 6] = cfg.nu + (ke === nothing || j == 1 ? 0.0 : 0.5 * (_wd_face_eddy_viscosity(ke, wd, k - 1, L, R) +
                                                                       _wd_face_eddy_viscosity(ke, wd, k, L, R)))
        hbar = 0.5 * (wd.h[k, L] + wd.h[k, R])
        # WHY floatmax: an empty cell pair has no advection in the explicit scheme either (dZ/h_adv = 0)
        col[j, 10] = wd.remap.active[f] ? wd.remap.hadv[k, f] :
            (hbar > WD_EMPTY_CELL_THICKNESS ? hbar : floatmax(Float64))   # m, advection thickness
    end
    # WHY: no bed friction on an active gate face (issue #57). Its control volume straddles the vena contracta, and
    # friction there acts with the jet velocity on the face opening: a head loss i_f dx (first order in dx, 2.5 % of the
    # upstream head at dx = 0.2 m with C = 50) that the frictionless gate relation (Bernoulli to the vena contracta)
    # does not have. The jet friction acts from the next face on.
    gate = wd.gate_active[f]
    gamma = 0.0   # m/s
    friction_row = 1
    if cfg.bed_roughness_length > 0.0 && !gate
        kw = klo
        below = 0.0   # m
        if ke !== nothing && ke.wall_function === :equilibrium
            # the wall layer of the k-epsilon wall function (_wd_k_epsilon_step!): friction there, at its height above
            # the bed; the shaved layers below follow it through the eddy viscosity
            while kw < khi && wd.hf[kw, f] < ke.wall_cell_min_fraction * model.dz
                below += wd.hf[kw, f]
                kw += 1
            end
            friction_row = kw - klo + 1
        end
        z1 = max(below + 0.5 * wd.hf[kw, f], WD_FRICTION_Z1_MIN_FACTOR * cfg.bed_roughness_length)
        gamma = bed_friction_coefficient(cfg.bed_roughness_length, z1) * abs(ws.u_old[kw, f])
    end
    gamma_mean = 0.0   # m/s, Chezy: g |U^n| / C^2 with the depth-averaged velocity of time n over the open layers
    if cfg.chezy_coefficient > 0.0 && !gate
        transport = 0.0
        depth = 0.0
        @inbounds for j in 1:m
            transport += col[j, 4] * ws.u_old[klo + j - 1, f]
            depth += col[j, 4]
        end
        gamma_mean = cfg.g / cfg.chezy_coefficient^2 * abs(transport / depth)
    end
    implicit_vertical_column!(view(col, :, 1), view(col, :, 2), view(col, :, 3), view(col, :, 4), view(col, :, 5),
        view(col, :, 6), gamma, cfg.dt, m, view(col, :, 7), view(col, :, 8), view(col, :, 9); advective=true,
        h_adv=view(col, :, 10), gamma_mean=gamma_mean, zprime=view(col, :, 11), friction_row=friction_row)
    H = 0.0
    @inbounds for j in 1:m
        k = klo + j - 1
        ws.u_hat[k, f] = col[j, 1]
        ws.r_vert[k, f] = col[j, 2]
        H += col[j, 4] * col[j, 2]
    end
    ws.H_face[f] = H
    return nothing
end

# Boundary velocities of the predictor (wet layers only): left prescribed (vector or function of time) or wall; right
# prescribed, zero-gradient (open) or wall.
function _wd_boundary_velocities!(model::Model, wd::WetDryState, u::AbstractMatrix{<:Real}, t::Real)
    nz, nxp1 = size(u)
    left = wd.left_velocity !== nothing ? wd.left_velocity(t) : model.x_boundary_u_left_profile
    if left !== nothing && !(left isa Real)
        left isa AbstractVector || throw(ArgumentError("left boundary velocity must be a scalar or vector"))
        length(left) == nz || throw(ArgumentError("left boundary velocity length must match nz"))
    end
    right_profile = model.x_boundary_u_right_profile
    right_profile === nothing || length(right_profile) == nz || throw(ArgumentError("right outflow profile length must match nz"))
    # free left face (issue #86): u of the boundary face comes from its own momentum equation
    left_free = wd.left_open !== nothing
    @inbounds for k in 1:nz
        left_free && break
        wet_left = wd.hf[k, 1] > 0.0
        u[k, 1] = (left === nothing || !wet_left) ? 0.0 : (left isa Real ? left : left[k])
    end
    if model.x_boundary_q_left !== nothing && !left_free
        # discharge boundary: the shape scaled with the face layer thicknesses of time n (dry layers carry no flow)
        transport = 0.0
        @inbounds for k in 1:nz
            transport += wd.hf[k, 1] * u[k, 1]
        end
        transport > 0.0 || throw(DomainError(transport, "left_discharge: the inflow face is dry"))
        @views u[:, 1] .*= _left_discharge(model, t) / transport
    end
    @inbounds for k in 1:nz
        wet_right = wd.hf[k, nxp1] > 0.0
        if _right_face_computed(model)
            # water-level boundary or free outflow: u of the boundary face comes from its own momentum equation
        elseif right_profile !== nothing && wet_right
            u[k, nxp1] = right_profile[k]
        elseif model.x_boundary_u_right_open && wet_right
            u[k, nxp1] = u[k, nxp1 - 1]
        else
            u[k, nxp1] = 0.0
        end
    end
    return nothing
end

# Tolerance (m) of the depth of column i (docs/depth_positivity.md, section 5): the final residual |F_i| of the
# free-surface solve (the flux-derived depth equals V_i(eta) - F_i) plus WD_POSITIVITY_ROUNDOFF_FACTOR * gamma_m * S with
# gamma_m = m eps(T) / (1 - m eps(T)), T the state type, and m = 2 nz + 8 (rounding steps of the continuity update).
# S = depth + |eta^n| + |eta^{n+1}| + |z_b| + O_i are the magnitudes that enter it (Higham 2002, Lemma 3.1 and Sec. 4.2).
function _wd_depth_roundoff(model::Model, wd::WetDryState, i::Int)
    m = 2 * model.config.nz + 8
    machine_eps = eps(eltype(wd.h))
    gamma = m * machine_eps / (1 - m * machine_eps)
    S = model.config.depth + abs(model.ws.eta_old[i]) + abs(model.state.eta[i]) + abs(wd.bed[i]) + wd.outflow[i]
    return abs(wd.rhs[i]) + WD_POSITIVITY_ROUNDOFF_FACTOR * gamma * S
end

# A face layer that opens in this step (the upwind level has risen past its lower interface, or the face has become wet)
# carries u = 0 from the time it was closed. It takes the velocity of the nearest open layer below at the same face
# (or above, at a face that opens from the bed upwards), i.e. the fluid that rises into it.
# WHY: hydrostatic theory has no vertical shear without stress; a new layer at rest beside a moving column is a spurious
# shear that the advection only removes at its own rate (about u/dx), so the layer lags for many steps. Common practice in
# z-layer models (e.g. Delft3D-FLOW, Z-model: the velocity of a new top layer is copied from the layer below).
function _wd_initialize_new_face_layers!(model::Model, wd::WetDryState)
    u = model.state.u
    nz = model.config.nz
    @inbounds for f in 2:model.config.nx
        for k in 1:nz
            (wd.hf[k, f] > 0.0 && wd.hf_prev[k, f] == 0.0) || continue
            src = 0
            for kk in (k - 1):-1:1
                if wd.hf[kk, f] > 0.0 && wd.hf_prev[kk, f] > 0.0
                    src = kk
                    break
                end
            end
            if src == 0
                for kk in (k + 1):nz
                    if wd.hf[kk, f] > 0.0 && wd.hf_prev[kk, f] > 0.0
                        src = kk
                        break
                    end
                end
            end
            src > 0 && (u[k, f] = u[src, f])
        end
    end
    copyto!(wd.hf_prev, wd.hf)
    return nothing
end

"""
    _wd_remap_face_layers!(model, wd)

Conservative remap of the face velocities u (m/s) from the face layers of the last step (wd.hf_prev) to those of this
step (wd.hf), used with `wd_top_layer = "merged"` (issue #88) in place of `_wd_initialize_new_face_layers!`. At each
interior face whose set of open layers changed (the surface crossed a level, or the top layer was merged or split):

    u_k^new = 1/hf_k^new * integral over layer k^new of u^old(z) dz,

with u^old piecewise constant on the old layers, the old top layer extended upwards without limit (the fluid that rises
above it moves with it) and the old bottom layer downwards. A split keeps u in both parts (momentum and kinetic energy
conserved), a merge takes the thickness-weighted mean (momentum conserved, the kinetic energy of the shear lost); a
surface that crosses several levels in one step is mapped the same way. Faces whose layering is unchanged keep u bit
for bit; closed layers get u = 0. Remapping by overlap integrals is the standard conservative remap of arbitrary
Lagrangian-Eulerian vertical coordinates (e.g. White & Adcroft 2008, J. Comput. Phys. 227; equation numbers not
checked), here first order (piecewise constant).
PERF: no allocation; wd.tri_a is free at this point of the step (it is the work column of the implicit vertical terms).
"""
function _wd_remap_face_layers!(model::Model, wd::WetDryState)
    u = model.state.u
    nz = model.config.nz
    dz = model.dz
    buf = wd.tri_a
    @inbounds for f in 2:model.config.nx
        wd.hf_prev[1, f] < 0.0 && continue   # first step: no previous layering
        kb, kt = 0, 0   # lowest and highest open layer of the last step
        same = true
        for k in 1:nz
            if wd.hf_prev[k, f] > 0.0
                kb == 0 && (kb = k)
                kt = k
            end
            (wd.hf_prev[k, f] > 0.0) == (wd.hf[k, f] > 0.0) || (same = false)
        end
        (same || kb == 0) && continue
        zb_f = _wd_face_bed(model, wd, f)
        for k in 1:nz
            buf[k] = u[k, f]
        end
        for k in 1:nz
            if wd.hf[k, f] <= 0.0
                u[k, f] = 0.0
                continue
            end
            lo = max((k - 1) * dz, zb_f)
            hi = lo + wd.hf[k, f]
            acc = 0.0
            single = 0
            for kk in kb:kt
                wd.hf_prev[kk, f] > 0.0 || continue
                olo = kk == kb ? -Inf : max((kk - 1) * dz, zb_f)
                ohi = kk == kt ? Inf : max((kk - 1) * dz, zb_f) + wd.hf_prev[kk, f]
                if olo <= lo && hi <= ohi
                    single = kk   # the new layer lies in one old layer: u unchanged
                    break
                end
                overlap = min(hi, ohi) - max(lo, olo)
                overlap > 0.0 && (acc += overlap * buf[kk])
            end
            u[k, f] = single > 0 ? buf[single] : acc / (hi - lo)
        end
    end
    copyto!(wd.hf_prev, wd.hf)
    return nothing
end

"""
    _wd_rebuild_interface_velocities!(model, wd)

Vertical velocity w^{n+1} (m/s) of the state on the layers of time n+1 (wd.h_new), used with `wd_top_layer = "merged"`
(issue #88) after the constituent transport:

    w_1 = 0,   w_{k+1} = w_k - (q_{k,i+1/2} - q_{k,i-1/2}) / dx   for the layers k below the top layer s of the column,
    w_k = 0   for k > s (the top of the top layer and the interfaces above it, inside a merged layer included),

with the theta discharges q (m^2/s) of the step. The cells below the top layer have the same thickness at n and n+1
unless the surface crossed their upper interface in this step, so this equals the flux-consistent w of the continuity
step there; it differs only where the layering changed.
WHY: the continuity step of the transport also carries the volume that moves between cells when the layering changes
(a merged top layer that splits hands over its upper part, h = merge dz, to the new cell in one step: a flux of
merge dz / dt, about 0.35 m/s for dz = 25 mm and dt = 7.2 ms). That flux is a remap of the grid, not a velocity of
the fluid; kept as the state w it entered the vertical momentum advection of the next step and dissipated about 0.3 1/s
of the energy of a standing wave of ka = 0.13 (issue #88, energy budget).
"""
function _wd_rebuild_interface_velocities!(model::Model, wd::WetDryState)
    w = model.state.w
    nz = model.config.nz
    dx = model.dx
    @inbounds for i in axes(w, 2)
        s = 0
        for k in 1:nz
            wd.h_new[k, i] > 0.0 && (s = k)
        end
        w[1, i] = 0.0
        for k in 1:nz
            w[k + 1, i] = k < s ? w[k, i] - (wd.q[k, i + 1] - wd.q[k, i]) / dx : 0.0
        end
    end
    return nothing
end

"""
    _wd_check_depth_positivity(model, wd)

Stops the run if a column holds a negative depth zeta - z_b beyond round-off at the start of a step (only possible if the
state was set from outside, e.g. eta written directly below the bed; the step itself cannot produce it, see
docs/depth_positivity.md).
"""
function _wd_check_depth_positivity(model::Model, wd::WetDryState)
    @inbounds for i in eachindex(wd.bed)
        depth = _wd_level(model, i) - wd.bed[i]
        if depth < -_wd_depth_roundoff(model, wd, i)
            throw(DomainError(depth, @sprintf("negative water depth %.3e m in column %d (bed %.4f m, level %.4f m); wetting and drying requires depth >= 0",
                depth, i, wd.bed[i], _wd_level(model, i))))
        end
    end
    return nothing
end

# The characteristic open faces of the step: (state, face index f, adjacent column N, outward normal n, face level
# eta_b^n in m). Inactive faces (radiation scheme, not open, supercritical) are skipped by the callers.
function _wd_characteristic_faces(model::Model, wd::WetDryState)
    nx = model.config.nx
    right = (wd.char_right, nx + 1, nx, 1.0, model.x_boundary_right_free ? model.x_boundary_eta_free : NaN)
    left = (wd.char_left, 1, 1, -1.0, wd.left_open === nothing ? NaN : wd.left_open.eta_b)
    return (right, left)
end

"""
    _wd_characteristic_predictor!(model, wd)

Characteristic open faces (`open_face_scheme="characteristic"`, issue #86; Borsboom 2023, Mooiman 2024, adapted to the
staggered grid in docs/numerics_open_boundaries.md section 8). The natural condition of the face is its momentum equation
plus the box term c_beta beta d(eta_b - eta_N)/dt (outward discharge Q_n, D = eta_b - eta_N, c_beta = min(c_in, c_out),
which is c - |U_n| for long waves). Its explicit part c_beta beta D^n is added here to the predictor of the face, once
per step, distributed like the surface slope (Delta u_k = n c_beta beta D^n r_k / H_f, so that sum_k hf_k Delta u_k =
n c_beta beta D^n with H_f = sum_k hf_k r_k). It also fixes beta and phi = theta g dt H_f/(dx/2) (m/s) for the solves of
the step: beta = `open_face_beta`, limited to -phi/c_beta (the flux weight w >= 0) where the boundary column is shallower
than `open_face_wet_depth` at time n, so that the face cannot make the free-surface matrix lose the M-matrix property
if the column falls dry (section 8.4). beta = -1/2 reflects (k dx)^3/64, beta = 0 is the radiation condition
weighted with theta (normal-mode analysis, section 8.3).
"""
function _wd_characteristic_predictor!(model::Model, wd::WetDryState)
    cfg = model.config
    cfg.open_face_scheme == "characteristic" || return nothing
    ws = model.ws
    for (char, f, N, n, eta_b) in _wd_characteristic_faces(model, wd)
        char.active || continue
        Hf = ws.H_face[f]   # m, effective depth of the implicit column of the face
        if !(Hf > 0.0)
            char.active = false   # dry face: no flux, the radiation form (weight 1) is harmless
            continue
        end
        c_beta = min(char.c_in, char.c_out)   # m/s
        phi = cfg.theta * cfg.g * cfg.dt * Hf / (0.5 * model.dx)
        beta = cfg.open_face_beta
        depth_N = _wd_level(model, N) - wd.bed[N]
        depth_N < cfg.open_face_wet_depth && (beta = max(beta, -phi / c_beta))
        char.beta, char.phi = beta, phi
        char.d_old = eta_b - model.ws.eta_old[N]
        incr = n * c_beta * beta * char.d_old / Hf   # m/s per unit column response
        @inbounds for k in 1:cfg.nz
            wd.hf[k, f] > 0.0 && (ws.u_hat[k, f] += incr * ws.r_vert[k, f])
        end
    end
    return nothing
end

"""
    _wd_characteristic_dirichlet!(model, wd)

Dirichlet form (eta_eff, w) of the characteristic open faces for the free-surface solve, from the predictor u_hat of this
solve (it is called again after the non-hydrostatic correction). Essential condition on the incoming characteristic
variable w_in = c_out eta - Q_n, as a time increment with the drift correction of a relaxed face (Borsboom 2023, eq. 3-4):

    c_out (eta_b^{n+1} - eta_b^n) - (Q_n^{n+1} - Q_n^n) = G,   G = -eps (c_out eta_b^n - Q_n^n - (c_out eta_T - Q_T,n)),

eps = dt/tau (0 without relaxation). With Q_n^{n+1} = Qhat_n - (phi + c_beta beta) D^{n+1} (Qhat_n from the predictor,
which holds the explicit box term) the solved face reads

    Q_n^{n+1} = Qhat_n + phi w (eta_N^{n+1} - eta_eff),   eta_b^{n+1} = eta_N^{n+1} + w_l (eta_eff - eta_N^{n+1}),
    w = (phi + c_beta beta) c_out / (phi S),   w_l = c_out / S,   S = c_out + phi + c_beta beta,
    eta_eff = eta_b^n + (Qhat_n - Q_n^n + G) / c_out,

the Dirichlet form of the radiation scheme (`_right_boundary_dirichlet`, `_left_boundary_slope`) with the flux weight w;
the level weight w_l goes to `_finish_free_outflow_level!` / `_finish_left_free_level!`. For beta = 0 both weights are
1/(1 + phi/c_out), the theta-weighted radiation condition (docs/numerics_open_boundaries.md section 8.2). The drift
correction acts on w_in, which an outgoing linear wave does not change: unlike the level relaxation of the radiation
scheme it reflects nothing at the design frequency (section 8.1).
"""
function _wd_characteristic_dirichlet!(model::Model, wd::WetDryState)
    cfg = model.config
    cfg.open_face_scheme == "characteristic" || return nothing
    ws = model.ws
    for (char, f, _, n, eta_b) in _wd_characteristic_faces(model, wd)
        char.active || continue
        Qhat = 0.0   # m^2/s, outward discharges of the predictor and of time n
        Qold = 0.0
        @inbounds for k in 1:cfg.nz
            Qhat += wd.hf[k, f] * ws.u_hat[k, f]
            Qold += wd.hf[k, f] * ws.u_old[k, f]
        end
        Qhat *= n
        Qold *= n
        tau, eta_T = f == 1 ? (wd.left_open.relaxation_time, wd.left_open.eta_target) :
                              (model.x_boundary_relaxation_time, model.x_boundary_eta_target)
        c_out = char.c_out
        # m^2/s: the incoming wave, Delta F = (c_out + c_in) Delta eta_in (Mooiman 2024, eq. 17), and the drift correction
        # of the incoming characteristic variable (its target carries the incoming wave)
        G = (c_out + char.c_in) * (char.e_b_new - char.e_b_old)
        if isfinite(tau)
            w_target = c_out * eta_T - n * char.target_discharge + (c_out + char.c_in) * char.e_b_old
            G -= (cfg.dt / tau) * (c_out * eta_b - Qold - w_target)
        end
        cb_beta = min(char.c_in, char.c_out) * char.beta   # m/s
        # WHY the intrinsic speed c_out - U_n for the increments: the face thicknesses are frozen within the step, so the
        # discharge increment of the face is sum_k hf_k Delta u_k = H Delta u_n without the part U_n Delta eta of
        # Delta Q_n; the essential condition then reads (c_out - U_n) Delta eta_b - Delta Q_n = G (its velocity form,
        # docs/numerics_open_boundaries.md section 8.1). With c_out the outgoing wave of a current of 0.16 m/s reflected
        # 2.9 % instead of 0.3 % (k h = 0.3, hydrostatic). The drift correction acts on states and keeps c_out.
        c_io = c_out - char.u_out   # m/s, intrinsic speed of the outgoing wave
        S = c_io + char.phi + cb_beta
        weight = (char.phi + cb_beta) * c_io / (char.phi * S)
        char.level_weight = c_io / S
        eta_eff = eta_b + (Qhat - Qold + G) / c_io
        if f == 1
            wd.left_open.eta_eff, wd.left_open.weight = eta_eff, weight
        else
            model.x_boundary_free_dirichlet = (eta_eff, weight)
        end
    end
    return nothing
end

"""
    _wd_free_surface_solve!(model, wd)

Semi-implicit free surface with exact volume positivity (Casulli 2009; nonlinear solver of Brugnano & Casulli 2008;
equation numbers not checked; proof and conditions in docs/depth_positivity.md). The water volume per unit area of column
i is the non-negative, convex, piecewise-linear function of its level

    V_i(eta) = max(0, H + eta - z_b,i)      (m; H the grid depth),

and eta^{n+1} solves the mildly nonlinear system

    F(eta) = V(eta) + T eta - b = 0,
    (T eta)_i = a_{i+1/2} (eta_i - eta_{i+1}) + a_{i-1/2} (eta_i - eta_{i-1}),   a_f = g dt^2 H_f / dx^2 (closed and
                boundary faces: a = 0),
    b_i = V_i(eta^n) - dt/dx (Uhat_{i+1/2} - Uhat_{i-1/2}),   Uhat_f = sum_k hf_{k,f} uhat_{k,f},

i.e. the depth-integrated continuity with u^{n+1} = uhat - g dt d(eta^{n+1})/dx in every wet face layer and the face
thicknesses of time n. Newton's method with the right derivative p_i = 1 if H + eta_i >= z_b,i, else 0:

    (P(eta^m) + T) (eta^{m+1} - eta^m) = -F(eta^m),   started from eta^0 = eta^n,

where P(eta^0) = I (all depths >= 0), so the first matrix is non-singular. Because V is convex, the iterates m >= 1 are
upper bounds of the solution and decrease monotonically, and the iteration ends with F = 0 after at most nx + 1 steps
(exact arithmetic); the stopping tolerance is max(WD_NEWTON_RTOL, eps(T)) for state type T. The velocities are then
corrected with the solved eta (step 4), and their flux divergence reproduces V(eta^{n+1}) >= 0: depth positivity and
volume conservation hold for any dt without changing the solution afterwards.
For a lake at rest F(eta^n) = 0 exactly: no iteration, eta unchanged. The solved eta of a column that falls dry lies at
or below its bed (only V = 0 matters); `_wd_update_level_from_fluxes!` stores it as an exactly dry level.
"""
function _wd_free_surface_solve!(model::Model, wd::WetDryState)
    cfg = model.config
    st = model.state
    ws = model.ws
    nz, nx = cfg.nz, cfg.nx
    dt, dx, g, H = cfg.dt, model.dx, cfg.g, cfg.depth
    theta = cfg.theta
    _wd_characteristic_dirichlet!(model, wd)   # characteristic open faces: (eta_eff, w) from this predictor (issue #86)
    # WHY theta^2 and the theta transport: u^{n+1} = u_hat - theta g dt d(eta^{n+1})/dx enters the continuity flux
    # sum_k hf (theta u^{n+1} + (1 - theta) u^n) (issue #34); theta = 1 is the former scheme exactly
    coeff = theta * theta * dt * dt * g / (dx * dx)   # 1/m
    a, eta, F, b = wd.a_face, wd.eta_iter, wd.rhs, wd.dp
    @inbounds for f in 1:(nx + 1)
        Hf = 0.0
        U = 0.0
        for k in 1:nz
            Hf += wd.hf[k, f]
            U += wd.hf[k, f] * (theta * ws.u_hat[k, f] + (1.0 - theta) * ws.u_old[k, f])
        end
        # implicit vertical terms: effective depth sum_k hf_k r_k of the column response (src/implicit_vertical.jl)
        a[f] = (f == 1 || f == nx + 1) ? 0.0 : coeff * (cfg.implicit_vertical_terms ? ws.H_face[f] : Hf)
        wd.U_hat[f] = U
    end
    # water-level boundary (prescribed, or the radiated level of a free outflow, _right_boundary_level): face nx + 1
    # couples column nx to the level half a cell away (_right_level_slope); (P + T) stays an M-matrix (T only gains
    # diagonal weight) and the flux divergence still reproduces V(eta) >= 0
    open_bc = _right_face_computed(model)
    eta_beyond, weight = open_bc ? _right_boundary_dirichlet(model) : (0.0, 1.0)   # m, level beyond column nx; -
    open_bc && (a[nx + 1] = weight * 2.0 * coeff * ws.H_face[nx + 1])
    # free left face (issue #86): the mirror, column 1 coupled to the level eta_eff half a cell to its left
    left_open = wd.left_open !== nothing
    eta_before = left_open ? wd.left_open.eta_eff : 0.0   # m, level before column 1
    left_open && (a[1] = wd.left_open.weight * 2.0 * coeff * ws.H_face[1])
    @inbounds for i in 1:nx
        b[i] = max(H + ws.eta_old[i] - wd.bed[i], 0.0) - dt * (wd.U_hat[i + 1] - wd.U_hat[i]) / dx
        eta[i] = ws.eta_old[i]
    end
    max_iterations = nx + WD_NEWTON_EXTRA_ITERATIONS
    iterations = 0
    while true
        converged = true
        @inbounds for i in 1:nx
            V = max(H + eta[i] - wd.bed[i], 0.0)
            Tl = a[i] * (eta[i] - (i > 1 ? eta[i - 1] : eta_before))
            Tr = a[i + 1] * (eta[i] - (i < nx ? eta[i + 1] : eta_beyond))
            F[i] = V + Tl + Tr - b[i]
            scale = H + V + abs(b[i]) + abs(Tl) + abs(Tr)   # m, magnitudes that enter F_i (round-off scale)
            abs(F[i]) <= max(WD_NEWTON_RTOL, eps(eltype(wd.h))) * scale || (converged = false)
        end
        converged && break
        iterations += 1
        iterations <= max_iterations || throw(ErrorException(@sprintf(
            "free-surface Newton iteration did not converge in %d iterations (max |F| = %.3e m)", max_iterations, maximum(abs, F))))
        # Thomas algorithm for (P + T) d = -F: lower(i) = upper(i-1) = -a_i, main(i) = p_i + a_i + a_{i+1}
        @inbounds begin
            for i in 1:nx
                p = H + eta[i] >= wd.bed[i] ? 1.0 : 0.0
                # characteristic face with a negative flux weight (box term, open_face_beta < 0 at a small Courant
                # number): (P + T) is an M-matrix only while its column is wet (docs/numerics_open_boundaries.md 8.4)
                if p == 0.0 && ((i == nx && a[nx + 1] < 0.0) || (i == 1 && a[1] < 0.0))
                    throw(DomainError(i, @sprintf("boundary column %d falls dry with a negative open-face weight; increase open_face_wet_depth (now %.3g m)",
                        i, cfg.open_face_wet_depth)))
                end
                denom = p + a[i] + a[i + 1] + (i > 1 ? a[i] * wd.cp[i - 1] : 0.0)
                if !(denom > 0.0)
                    # a connected group of columns that is dry in this iterate and receives no water: only possible if a
                    # prescribed boundary outflow exceeds the available volume (docs/depth_positivity.md, section 4)
                    throw(DomainError(denom, @sprintf("free-surface Newton matrix singular at column %d: the prescribed boundary outflow exceeds the available water volume", i)))
                end
                wd.cp[i] = -a[i + 1] / denom
                wd.delta[i] = (-F[i] + (i > 1 ? a[i] * wd.delta[i - 1] : 0.0)) / denom
            end
            for i in (nx - 1):-1:1
                wd.delta[i] -= wd.cp[i] * wd.delta[i + 1]
            end
            for i in 1:nx
                eta[i] += wd.delta[i]
            end
        end
    end
    wd.newton_iterations = iterations
    wd.newton_iterations_max = max(wd.newton_iterations_max, iterations)
    @inbounds for i in 1:nx
        wd.delta[i] = eta[i] - ws.eta_old[i]   # increment of the solve, eta^{n+1} = eta_old + delta
    end

    # 4. velocity correction with the solved level (also below the bed of a column that falls dry)
    @inbounds for f in 2:nx
        grad = (eta[f] - eta[f - 1]) / dx
        for k in 1:nz
            r = cfg.implicit_vertical_terms ? ws.r_vert[k, f] : 1.0   # column response of the implicit vertical terms
            st.u[k, f] = wd.hf[k, f] > 0.0 ? ws.u_hat[k, f] - g * dt * theta * grad * r : 0.0
        end
    end
    if left_open
        grad = _left_boundary_slope(eta, model, dx)
        @inbounds for k in 1:nz
            st.u[k, 1] = wd.hf[k, 1] > 0.0 ? ws.u_hat[k, 1] - g * dt * theta * grad * ws.r_vert[k, 1] : 0.0
        end
    else
        @views st.u[:, 1] .= ws.u_hat[:, 1]
    end
    if !_right_face_computed(model)
        @views st.u[:, nx + 1] .= ws.u_hat[:, nx + 1]
    else
        grad = _right_boundary_slope(eta, model, dx)
        @inbounds for k in 1:nz
            st.u[k, nx + 1] = wd.hf[k, nx + 1] > 0.0 ? ws.u_hat[k, nx + 1] - g * dt * theta * grad * ws.r_vert[k, nx + 1] : 0.0
        end
    end
    return nothing
end

"""
    _wd_update_level_from_fluxes!(model, wd)

Continuity step: eta_i^{n+1} = eta_i^n - dt/dx sum_k (q_{k,i+1/2} - q_{k,i-1/2}) with q = hf u of the velocities of the
free-surface solve, which conserves the water volume to round-off (the sum over columns telescopes). By construction of
the solve (`_wd_free_surface_solve!`) the new depth equals V(eta^{n+1}) >= 0 up to round-off (docs/depth_positivity.md):
a depth a few ulps below zero is stored as an exactly dry level (the volume added is accumulated in wd.clamped_volume),
a larger negative depth stops the run (it would mean that the solve did not satisfy its equations).
"""
function _wd_update_level_from_fluxes!(model::Model, wd::WetDryState)
    cfg = model.config
    st = model.state
    ws = model.ws
    nz, nx = cfg.nz, cfg.nx
    dt, dx = cfg.dt, model.dx
    theta = cfg.theta
    # theta fluxes (m^2/s): they move the level, give the vertical fluxes and transport the constituent
    @. wd.q = wd.hf * (theta * st.u + (1.0 - theta) * ws.u_old)
    @inbounds for i in 1:nx
        div = 0.0
        out = 0.0
        for k in 1:nz
            div += wd.q[k, i + 1] - wd.q[k, i]
            out += max(wd.q[k, i + 1], 0.0) + max(-wd.q[k, i], 0.0)
        end
        wd.outflow[i] = out * dt / dx
        st.eta[i] = ws.eta_old[i] - dt * div / dx
        depth = _wd_level(model, i) - wd.bed[i]
        if depth < 0.0
            if depth < -_wd_depth_roundoff(model, wd, i)
                throw(DomainError(depth, @sprintf("negative water depth %.3e m in column %d after the free-surface solve (outflow %.3e m); this violates the positivity of the solve (docs/depth_positivity.md)",
                    depth, i, wd.outflow[i])))
            end
            st.eta[i] = _wd_dry_eta(cfg.depth, wd.bed[i], eltype(st.eta))
            wd.clamped_volume += (_wd_level(model, i) - wd.bed[i] - depth) * dx
        end
        ws.eta_t[i] = (st.eta[i] - ws.eta_old[i]) / dt
    end
    return nothing
end

"""
    _incoming_pressure_profile(k_in, z, zs) -> P (-)

Non-hydrostatic part of the dynamic pressure of the incoming wave of wave number k_in (rad/m), at the height z (m) above
the bed, per unit of rho0 g eta: P = cosh(k_in z)/cosh(k_in zs) - 1, zero at the height zs (m) of the centre of the
surface cell and above (the hydrostatic part rho0 g eta is carried by the level). Linear wave theory (e.g. Dean &
Dalrymple 1991, Sec. 4.4; equation numbers not checked) has zs = h, the depth.
WHY zs at the surface-cell centre: the model's q is zero in the whole surface cell (`_wd_nonhydrostatic_correction!`),
so its wave carries this profile, not the one that vanishes at the surface. Generated amplitude / target with zs = h:
0.988, 0.963, 0.942 at k h = 0.6, 1.2, 2.0 (10 layers); with the surface-cell centre 1.002, 1.000, 0.999.
"""
function _incoming_pressure_profile(k_in::Real, z::Real, zs::Real)
    (zs > 0.0 && 0.0 <= z <= zs) || return 0.0
    return cosh(k_in * z) / cosh(k_in * zs) - 1.0
end

"""
    _incoming_inflow_advection(model, wd, char, k, f, n) -> m/s^2

Momentum advection of the incoming wave at a free face (f = 1 or nx + 1, outward normal n) through which the current
enters (issue #86): the face's own advection has no inflow term there (the outer face carries the zero-gradient flux
of `_wd_advection_tendency`), which drops U du/dx of the incoming wave and generated 3.5 % too little on a following
current of 0.16 m/s (k h = 1.2). The outer node gets the velocity of zero gradient of the deviation from the incoming
wave, u_ghost = u_b + (u_in(ghost) - u_in(b)), so the inflow term is a/dx (u_in(ghost) - u_in(b)) with a = -n u_k (m/s,
> 0 for inflow), u_in = -n eta_in sigma cosh(k_in z)/sinh(k_in h) (linear theory, intrinsic frequency sigma =
omega - k_in U_in) and the ghost a cell outside, where the incoming wave passes dx/c_in earlier (`char.e_g`, time n).
0 without an incoming wave, for outflow, without a design period, or for a dry face layer.
"""
function _incoming_inflow_advection(model::Model, wd::WetDryState, char::CharacteristicFaceState, k::Int, f::Int,
        n::Real)
    (char.incoming === nothing || char.k_in == 0.0 || wd.hf[k, f] <= 0.0) && return 0.0
    cfg = model.config
    a = -n * model.state.u[k, f]   # m/s, inflow velocity of layer k
    a > 0.0 || return 0.0
    zb = _wd_face_bed(model, wd, f)
    h = cfg.depth - zb
    z = (k - 0.5) * model.dz - zb
    (h > 0.0 && 0.0 <= z <= h) || return 0.0
    sigma = 2pi / cfg.radiation_period + char.k_in * char.u_out   # rad/s, intrinsic: U_in = -U_out
    profile = sigma * cosh(char.k_in * z) / sinh(char.k_in * h)
    return a / model.dx * (-n) * profile * (char.e_g - char.e_b_old)
end

"""
    _add_incoming_pressure!(model, wd, char, E, f, th, mu)

Incoming wave at an open face with pressure radiation (issue #86): the face pressure radiates the deviation from the
incoming wave's non-hydrostatic pressure q_in = rho0 g eta_in P_k (`_incoming_pressure_profile` at the layer centres
above the face bed, zero from the centre of the still-water surface cell up, wave number `char.k_in`), with eta_in at the face (e_b) and at the adjacent cell centre (e_n):

    (q_b - q_in,b)^{n+1} (1 + th mu) = (q_b - q_in,b)^n - (1 - th) mu [(q_b - q_in,b) - (q_N - q_in,N)]^n
                                        + th mu (q_N - q_in,N)^{n+1},

so the explicit part E of q_b^{n+1} (1 + th mu) gains (q_in,b^{n+1} - q_in,b^n) + (1 - th) mu (q_in,b - q_in,N)^n +
th mu (q_in,b - q_in,N)^{n+1}. Nothing without an incoming wave or design period.
"""
function _add_incoming_pressure!(model::Model, wd::WetDryState, char::CharacteristicFaceState, E::AbstractVector,
        f::Int, th::Real, mu::Real)
    (char.incoming === nothing || char.k_in == 0.0) && return nothing
    cfg = model.config
    zb = _wd_face_bed(model, wd, f)
    # m: height above the face bed where the model's q is zero: "cut" the centre of the still-water surface cell (wet part
    # of the layer that holds the still level cfg.depth), "merged" the still level itself (q = 0 at zeta, issue #88; the
    # profile is then evaluated at the centres of the still-water wet parts of the layers, the top cell's below zeta)
    merged = cfg.wd_top_layer == "merged"
    ks = clamp(ceil(Int, cfg.depth / model.dz), 1, cfg.nz)
    zs = merged ? cfg.depth - zb : 0.5 * ((ks - 1) * model.dz + cfg.depth) - zb
    scale = cfg.rho0 * cfg.g   # Pa/m
    @inbounds for k in 1:cfg.nz
        zc = merged ? 0.5 * (max((k - 1) * model.dz, zb) + (k == cfg.nz ? cfg.depth : min(k * model.dz, cfg.depth))) :
            (k - 0.5) * model.dz
        P = _incoming_pressure_profile(char.k_in, zc - zb, zs)
        P == 0.0 && continue
        E[k] += scale * P * ((char.e_b_new - char.e_b_old) + (1.0 - th) * mu * (char.e_b_old - char.e_n_old) +
                             th * mu * (char.e_b_new - char.e_n_new))
    end
    return nothing
end

"""
    _wd_nonhydrostatic_correction!(model, wd)

Non-hydrostatic pressure correction of the wetting-drying step (pressure split p = p_hydrostatic + q; Casulli 1999,
Casulli & Zanolli 2002; equation numbers not checked), applied to the velocities of the free-surface step before the
repeated free-surface solve:

1. provisional geometry h_nh from the level of the free-surface solve, eta_old + delta (clamped to the bed);
2. vertical velocity predictor at the interfaces between two wet cells, w* = w^n - dt * adv(w): first-order upwind in the
   advective form with the velocities of time n, capped like the u advection;
3. pressure problem for q on the wet cells below the surface cell of each column (q = 0 in the surface cell, the
   dynamic free-surface condition, and in empty cells): with the corrected velocities
       u = u_fs - dt/rho0 (q_R - q_L)/dx  (wet interior face layers),   w = w* - dt/rho0 (q_k - q_{k-1})/dz_k,
       dz_k = (h_{k-1} + h_k)/2,  w = 0 at the bed,
   every such cell satisfies its continuity (h_nh - h^n)/dt + d(hf u)/dx + w_top - w_bot = 0. This gives the symmetric
   positive definite system
       sum_faces hf/dx^2 (q_P - q_N) + sum_interfaces (q_P - q_N)/dz = -rho0/dt (div* + (h_nh - h^n)/dt)
   (neighbours with q = 0 enter the diagonal only), solved with a sparse Cholesky factorization;
4. the gradient -dt/rho0 dq/dx is added to the predictor u_hat on the wet interior face layers, and the caller repeats
   the free-surface solve (`_wd_free_surface_solve!`). The final velocities therefore come from the positivity-preserving
   solve; they differ from the projected ones by the depth-uniform change of the surface gradient between the two solves,
   so the continuity of the cells below the surface holds approximately, and w follows from the layer continuity (step 6).
   WHY not correct u directly (as Casulli 1999 does): that changes the column transports after the solve, and the new
   level computed from them is no longer guaranteed to be at or above the bed.
With `wd_top_layer = "merged"` (issue #88) the geometry is the grid of time n (h_nh = h^n), w^n comes from the
continuity of u^n at every interface, the w advection is second order (limited), and the top cell is an unknown with
q = 0 at the surface zeta (Dirichlet at h_s/2 above its centre, predicted surface velocity delta/dt).
WHY q = 0 in the surface cell (not at the surface itself) with "cut": the surface cell is cut by the moving surface and
can be arbitrarily thin; the simple condition keeps the system SPD for every geometry. It is first order in the surface-cell
thickness (period error of a standing wave, test_validate_wd_nonhydro_standing_wave_full_dispersion_*).
For a lake at rest the right-hand side is exactly zero, so q = 0 and nothing changes.
Open right boundary (water level or free outflow, issue #49): the boundary face is not corrected (dq/dx = 0 there).
WHY not q = 0 on the face: q of this scheme is zero in the surface cell, not at the surface, so it has a part that a
Dirichlet condition at the outlet does not match even for long waves (the fixed grid has dq/dx = 0 as well, docs
section 24). Measured with a Gaussian wave of width 1 to 5 m in 1 m depth leaving through a free outflow (reflected amplitude
relative to a closed boundary): q = 0 on the face 10-29 %, q = 0 one cell beyond 8-21 %, dq/dx = 0 2-5 %.
"""
function _wd_nonhydrostatic_correction!(model::Model, wd::WetDryState)
    cfg = model.config
    st = model.state
    ws = model.ws
    nz, nx = cfg.nz, cfg.nx
    dt, dx, rho0 = cfg.dt, model.dx, cfg.rho0
    theta = cfg.theta
    h_nh, w_star, idx = wd.h_nh, wd.w_star, wd.nh_index
    wet(k, i) = h_nh[k, i] > WD_EMPTY_CELL_THICKNESS

    # 1. geometry of the pressure problem: "cut" the provisional level zeta^n + delta of the free-surface solve; "merged"
    # (issue #88) the grid of time n, on which the whole step is computed (the layering change to n+1 is a remap)
    # WHY the grid of time n with "merged": with the provisional geometry, a merged top layer of time n that splits in
    # h_nh gave the cell below a divergence source (h_nh - h^n)/dt of about -merge dz/dt, which the projection turned
    # into a spurious w and pressure gradient (energy budget, issue #88). On the grid of time n every interface between
    # two wet cells lies below the top layer, so w^n there is the exact continuity integral of u^n (no fallback).
    if cfg.wd_top_layer == "merged"
        copyto!(h_nh, wd.h)
    else
        @inbounds for i in 1:nx
            zeta = max(cfg.depth + ws.eta_old[i] + wd.delta[i], wd.bed[i])
            layer_thicknesses!(view(h_nh, :, i), zeta, wd.bed[i], model.dz)
        end
    end

    # 2. w predictor at the interfaces k = 2..nz between two wet cells (k-1 below, k above), from w^n
    # WHY st.w at theta = 1: the interface fluxes of the last step are w^n there (the former scheme, bit for bit)
    # theta < 1: w^n from the layer continuity of the discharges of time n (wd.q = hf u^n, set at the start of the step),
    # as on the fixed grid (st.w from u^{n+1}), at the interfaces between two cells that are wet at time n: all cells
    # below such an interface are full at n and n+1 (or cut by the fixed bed), so their thickness does not change and the
    # integral is exact. Elsewhere (the surface rises through the interface in this step) the interface flux of the last
    # step, st.w, as at theta = 1.
    # WHY not w^{n+1} = (w_theta - (1 - theta) w^n) / theta from the stored w^n (the former scheme, issue #38): an error
    # of that recursion is multiplied by -(1 - theta)/theta every step, -1 at theta = 1/2, and the undamped sawtooth in
    # time grew offshore on the Carrier-Greenspan beach (surface error 0.37 of the runup amplitude at theta = 0.55, blow-up
    # at 0.5). Computed from u^n, w^n has no history and needs no initialisation.
    # WHY not the continuity of u^n everywhere: where a bore lifts the surface through several layers in one step, the
    # cells below the interface fill up and the integral without their thickness change gave w of metres per second (the
    # non-hydrostatic Stoker bore blew up).
    if theta != 1.0
        @inbounds for i in 1:nx
            wd.w_nh[1, i] = 0.0
            w_cont = 0.0   # m/s, continuity integral of the discharges of time n up to interface k + 1
            for k in 1:nz
                w_cont -= (wd.q[k, i + 1] - wd.q[k, i]) / dx
                full_below = k < nz && wd.h[k, i] > WD_EMPTY_CELL_THICKNESS && wd.h[k + 1, i] > WD_EMPTY_CELL_THICKNESS
                wd.w_nh[k + 1, i] = full_below ? w_cont : st.w[k + 1, i]
            end
        end
    end
    W = theta == 1.0 ? st.w : wd.w_nh
    u0 = wd.u_old
    fill!(w_star, 0.0)
    # Explicit upwind advection of w. With wd_top_layer = "merged" (issue #88) second order where the stencil exists: the
    # upwind difference c (w_up - w_k) plus the anti-diffusive correction -c (phi_k - phi_up) of the flux-limited
    # Lax-Wendroff face values, phi_j = (1 - C)/2 MC(w_j - w_{j,up}, w_{j,down} - w_j), C = c dt (van Leer 1977,
    # Sweby 1984; the same limiter as the tvd2 transport). phi = 0 where a neighbour interface is missing (first order),
    # and the correction is left out where the first-order step is capped (dt sum c > 1).
    # WHY: the first-order upwind form diffuses w by |u| dx/2 and |w| h/2, an energy loss proportional to the amplitude
    # (0.007 to 0.014 1/s of a standing wave of ka = 0.13 at nz = 10..40, energy budget of issue #88).
    second_w = cfg.wd_top_layer == "merged"
    wv(kk, j) = 2 <= kk <= nz && 1 <= j <= nx && wet(kk - 1, j) && wet(kk, j)   # interface kk of column j holds a w
    phi_x(kk, j, d, C) = (wv(kk, j + d) && wv(kk, j - d)) ?
        0.5 * (1.0 - C) * _monotonized_central_scalar(W[kk, j] - W[kk, j + d], W[kk, j - d] - W[kk, j]) : 0.0
    # WHY no surface value as the upper stencil point (d(zeta)/dt of the last step at the top of the top cell): tried,
    # it raised the energy loss at both amplitudes (issue #88)
    phi_z(kk, j, d, C) = (wv(kk + d, j) && wv(kk - d, j)) ?
        0.5 * (1.0 - C) * _monotonized_central_scalar(W[kk, j] - W[kk + d, j], W[kk - d, j] - W[kk, j]) : 0.0
    @inbounds for i in 1:nx, k in 2:nz
        (wet(k - 1, i) && wet(k, i)) || continue
        wk = W[k, i]
        ubar = 0.25 * (u0[k - 1, i] + u0[k - 1, i + 1] + u0[k, i] + u0[k, i + 1])   # m/s, u at the interface
        csum = 0.0
        tend = 0.0
        anti = 0.0   # 1/s * m/s, second-order correction of the tendency
        if ubar > 0.0 && i > 1 && wet(k - 1, i - 1) && wet(k, i - 1)
            c = ubar / dx
            csum += c
            tend += c * (W[k, i - 1] - wk)
            second_w && (anti -= c * (phi_x(k, i, -1, c * dt) - phi_x(k, i - 1, -1, c * dt)))
        elseif ubar < 0.0 && i < nx && wet(k - 1, i + 1) && wet(k, i + 1)
            c = -ubar / dx
            csum += c
            tend += c * (W[k, i + 1] - wk)
            second_w && (anti -= c * (phi_x(k, i, 1, c * dt) - phi_x(k, i + 1, 1, c * dt)))
        end
        w_below = 0.5 * (W[k - 1, i] + wk)   # m/s, at the centre of cell k-1
        if w_below > 0.0 && k > 2 && wet(k - 2, i)
            c = w_below / h_nh[k - 1, i]
            csum += c
            tend += c * (W[k - 1, i] - wk)
            second_w && (anti -= c * (phi_z(k, i, -1, c * dt) - phi_z(k - 1, i, -1, c * dt)))
        end
        w_above = 0.5 * (wk + W[k + 1, i])   # m/s, at the centre of cell k
        if w_above < 0.0 && k < nz && wet(k + 1, i)
            c = -w_above / h_nh[k, i]
            csum += c
            tend += c * (W[k + 1, i] - wk)
            second_w && (anti -= c * (phi_z(k, i, 1, c * dt) - phi_z(k + 1, i, 1, c * dt)))
        end
        w_star[k, i] = wk + (dt * csum > 1.0 ? tend / csum : dt * (tend + anti))
    end
    if theta != 1.0
        # The continuity below holds for the theta velocities; with w^{n+1} = w_star - dt/rho0 dz(theta q^{n+1} +
        # (1 - theta) q^n) the predicted theta interface velocity is theta w_star + (1 - theta) w^n - theta (1 - theta)
        # dt/rho0 dz q^n (wd.q_nh still holds q^n), and the system adds -dt/rho0 dz(theta^2 q^{n+1}).
        @inbounds for i in 1:nx, k in 2:nz
            (wet(k - 1, i) && wet(k, i)) || continue
            dzq = (wd.q_nh[k, i] - wd.q_nh[k - 1, i]) / (0.5 * (h_nh[k - 1, i] + h_nh[k, i]))
            w_star[k, i] = theta * w_star[k, i] + (1.0 - theta) * W[k, i] - theta * (1.0 - theta) * dt / rho0 * dzq
        end
    end

    # 3. pressure problem: unknowns are the wet cells below the surface cell of each column ("cut"), or all wet cells
    # with q = 0 at the surface zeta ("merged", issue #88): the top cell s couples to q = 0 at the distance h_s/2 above
    # its centre, through the surface velocity w_s = w*_s - theta dt/rho0 (0 - q_s)/(h_s/2), predicted from the vertical
    # momentum like the interior interfaces (w^n from the continuity of u^n over the column, theta split of q).
    # WHY: q = 0 in the whole surface cell puts the pressure node of the surface half a cell too low; with the merged top
    # layer (up to (1 + merge) dz thick) that is a large part of the wave's pressure profile. Phase speed / linear theory
    # at k h = 0.6, 1.2, 2.0 (10 layers, 40 cells per wavelength, theta = 0.7): "cut" 1.0072, 1.0328, 1.0845; "merged"
    # 0.9981, 0.9971, 0.9950 (20 layers: 1.0028, 1.0157, 1.0417 against 0.9983, 0.9981, 0.9984; issue #88).
    # WHY not w*_s = delta/dt (the level change of the free-surface solve, tried first): it is a continuity value, the
    # surface fluid then gets its new velocity without pressure, the non-hydrostatic pressure is too weak and the waves
    # too fast (1.0458 at k h = 1.2, 1.1601 at 2.0).
    # The top cell is never thinner than merge dz (except in a shallow column), so its coefficient 2/h_s stays bounded.
    top_unknown = cfg.wd_top_layer == "merged"
    n = 0
    fill!(idx, 0)
    @inbounds for i in 1:nx
        s = 0
        for k in nz:-1:1
            if wet(k, i)
                s = k
                break
            end
        end
        if top_unknown && s > 0
            # w^n at the surface: continuity integral of the discharges of time n over the column; theta predictor as at
            # the interior interfaces, with dz q^n = (0 - q_s^n)/(h_s/2)
            ws_n = 0.0
            for k in 1:s
                ws_n -= (wd.q[k, i + 1] - wd.q[k, i]) / dx
            end
            w_star[s + 1, i] = ws_n + theta * (1.0 - theta) * dt / rho0 * wd.q_nh[s, i] / (0.5 * h_nh[s, i])
        end
        for k in 1:(top_unknown ? s : s - 1)
            wet(k, i) || continue
            n += 1
            idx[k, i] = n
        end
    end
    # Pressure radiation at the open right face (nonhydro_open_face = "radiation", issue #86): the face pressure q_b of each
    # wet face layer obeys dq_b/dt + c (q_b - q_nx)/(dx/2) = 0 with the weight th of the level radiation (theta with
    # radiation_theta, else 1 = backward Euler):
    #     q_b^{n+1} (1 + th mu) = E + th mu q_nx^{n+1},   E = q_b^n - (1 - th) mu (q_b^n - q_nx^n),   mu = c dt/(dx/2),
    # c from _radiation_speed. The face then carries the correction flux -dt/rho0 (q_b - q_nx)/(dx/2) like an
    # interior face; in the system (unknown sol = theta^2 q) it adds a_b/(1 + th mu), a_b = 2 hf/dx^2, to the diagonal of
    # cell (k, nx) and a_b theta^2 E/(1 + th mu) to its right-hand side.
    if cfg.nonhydro_open_face == "radiation"
        @inbounds for k in 1:nz
            wd.q_right_explicit[k] = wd.q_nh[k, nx]   # Pa, q_nx^n (wd.q_nh still holds q^n here)
            wd.q_left_explicit[k] = wd.q_nh[k, 1]     # Pa, q_1^n (free left face)
        end
    end
    fill!(wd.q_nh, 0.0)
    # WHY: with dq/dx = 0 the open face is a wall for the non-hydrostatic pressure, which reflects the non-hydrostatic
    # part of an outgoing dispersive wave (docs/numerics_open_boundaries.md, section 7). Only the diagonal changes, so the
    # matrix stays symmetric positive definite (Cholesky and the column-line multigrid apply unchanged).
    radiate = cfg.nonhydro_open_face == "radiation" && _right_face_computed(model)
    mu_q = 0.0
    th_q = cfg.radiation_theta ? theta : 1.0   # -, time weight of the pressure radiation
    if radiate
        U_b, depth_b = _face_velocity_depth(model, nx + 1)
        # the current of this step's level radiation (_update_free_outflow_level!), so that both radiate together
        isnan(wd.u_right_mean) || cfg.radiation_period == 0.0 || (U_b = wd.u_right_mean)
        mu_q = _level_radiation_speed(cfg, max(_radiation_speed(cfg, U_b, depth_b), 0.0), U_b) * dt / (0.5 * dx)
        @inbounds for k in 1:nz
            # E: explicit part of q_b^{n+1} (1 + th mu), stored in q_right_explicit
            wd.q_right_explicit[k] = wd.q_right[k] - (1.0 - th_q) * mu_q * (wd.q_right[k] - wd.q_right_explicit[k])
        end
        _add_incoming_pressure!(model, wd, wd.char_right, wd.q_right_explicit, nx + 1, th_q, mu_q)
    end
    # free left face (issue #86): the same radiation towards -x, speed of the outgoing wave on the current -U
    radiate_left = cfg.nonhydro_open_face == "radiation" && wd.left_open !== nothing
    mu_q_left = 0.0
    if radiate_left
        U_l, depth_l = _face_velocity_depth(model, 1)
        U_out = (cfg.radiation_period > 0.0 && !isnan(wd.left_open.u_mean)) ? wd.left_open.u_mean : -U_l
        mu_q_left = _level_radiation_speed(cfg, max(_radiation_speed(cfg, U_out, depth_l), 0.0), U_out) * dt / (0.5 * dx)
        @inbounds for k in 1:nz
            wd.q_left_explicit[k] = wd.q_left[k] - (1.0 - th_q) * mu_q_left * (wd.q_left[k] - wd.q_left_explicit[k])
        end
        _add_incoming_pressure!(model, wd, wd.char_left, wd.q_left_explicit, 1, th_q, mu_q_left)
    end
    if n == 0
        model.last_nonhydro_pcorr_max = 0.0
        fill!(wd.q_right, 0.0)
        fill!(wd.q_left, 0.0)
        return nothing
    end
    rows, cols, vals, rhs = wd.nh_rows, wd.nh_cols, wd.nh_vals, wd.nh_rhs
    empty!(rows)
    empty!(cols)
    empty!(vals)
    resize!(rhs, n)
    # column-line multigrid (issue #71): its fine level gets the same coefficients as the triplets below
    use_mg = cfg.nonhydro_solver == "multigrid" || (cfg.nonhydro_solver == "auto" && n >= NH_MULTIGRID_MIN_UNKNOWNS)
    mg = nothing
    if use_mg
        wd.nh_multigrid === nothing && (wd.nh_multigrid = ColumnMultigrid(nz, nx))
        mg = wd.nh_multigrid
        L = mg.levels[1]
        fill!(L.active, false)
        fill!(L.dv, 0.0)
        fill!(L.dh, 0.0)
        fill!(L.cz, 0.0)
        fill!(L.cx, 0.0)
        fill!(mg.rhs, 0.0)
    end
    @inbounds for i in 1:nx, k in 1:nz
        P = idx[k, i]
        P == 0 && continue
        diag = 0.0
        rhs_radiation = 0.0   # known part of the pressure radiation at the open face (units of rhs[P])
        div = (h_nh[k, i] - wd.h[k, i]) / dt   # m/s
        for (f, j, sgn) in ((i, i - 1, -1.0), (i + 1, i + 1, 1.0))   # left and right face, neighbour column
            hf = wd.hf[k, f]
            hf > 0.0 || continue
            div += sgn * hf * (theta * st.u[k, f] + (1.0 - theta) * ws.u_old[k, f]) / dx
            if radiate && f == nx + 1
                a = 2.0 * hf / dx^2 / (1.0 + th_q * mu_q)
                diag += a
                rhs_radiation += a * theta * theta * wd.q_right_explicit[k]
                use_mg && (mg.levels[1].dh[k, i] += a)
                continue
            end
            if radiate_left && f == 1
                a = 2.0 * hf / dx^2 / (1.0 + th_q * mu_q_left)
                diag += a
                rhs_radiation += a * theta * theta * wd.q_left_explicit[k]
                use_mg && (mg.levels[1].dh[k, i] += a)
                continue
            end
            # boundary faces: no correction; at an open right boundary (water level or free outflow, issue #49) a
            # Neumann condition dq/dx = 0 (see _wd_nonhydrostatic_correction! docstring for why not q = 0)
            (1 < f <= nx) || continue
            a = hf / dx^2
            diag += a
            N = idx[k, j]
            if N > 0
                push!(rows, P); push!(cols, N); push!(vals, -a)
            end
            if use_mg
                mg.levels[1].dh[k, i] += a
                (j == i + 1 && N > 0) && (mg.levels[1].cx[k, i] = a)
            end
        end
        if k > 1 && wet(k - 1, i)   # interface k (below); w = 0 at the bed
            a = 1.0 / (0.5 * (h_nh[k - 1, i] + h_nh[k, i]))
            div -= w_star[k, i]
            diag += a
            N = idx[k - 1, i]
            if N > 0
                push!(rows, P); push!(cols, N); push!(vals, -a)
            end
            use_mg && (mg.levels[1].dv[k, i] += a)
        end
        # interface k+1 (above): between wet cells, since k lies below the surface cell ("cut"); for the top cell
        # ("merged") the surface, q = 0 at h_k/2 above the centre
        top = k == nz || !wet(k + 1, i)
        a = top ? 1.0 / (0.5 * h_nh[k, i]) : 1.0 / (0.5 * (h_nh[k, i] + h_nh[k + 1, i]))
        div += w_star[k + 1, i]
        diag += a
        N = top ? 0 : idx[k + 1, i]
        if N > 0
            push!(rows, P); push!(cols, N); push!(vals, -a)
        end
        push!(rows, P); push!(cols, P); push!(vals, diag)
        rhs[P] = -rho0 / dt * div + rhs_radiation
        if use_mg
            L1 = mg.levels[1]
            L1.dv[k, i] += a
            N > 0 && (L1.cz[k, i] = a)
            L1.active[k, i] = true
            mg.rhs[k, i] = rhs[P]
        end
    end
    # WHY 1/theta^2 below: the system makes the theta flux divergence-free with the correction -dt/rho0 grad(sol); the
    # pressure enters u^{n+1} with theta dt and the theta flux with theta^2 dt, so q = sol / theta^2 (Pa).
    # A lake at rest has rhs = 0 and gives q = 0 exactly with both solvers.
    solved = false
    if use_mg
        iterations, solved = @timeit model.timers "multigrid" begin
            _mg_setup!(mg)
            _mg_pcg!(mg; rtol=cfg.nonhydro_solver_tolerance, itmax=cfg.nonhydro_solver_max_iterations)
        end
        wd.nh_iterations = iterations
        wd.nh_iterations_max = max(wd.nh_iterations_max, iterations)
        wd.nh_multigrid_solves += 1
        if solved
            @inbounds for i in 1:nx, k in 1:nz
                idx[k, i] > 0 && (wd.q_nh[k, i] = mg.x[k, i] / (theta * theta))
            end
        else
            wd.nh_fallbacks += 1
            wd.nh_fallbacks == 1 && @warn @sprintf("non-hydrostatic multigrid: CG did not reach the relative residual %.1e in %d iterations; the direct solver takes over (warned once per model)",
                cfg.nonhydro_solver_tolerance, cfg.nonhydro_solver_max_iterations)
        end
    end
    if !solved
        # PERF: the coefficients change every step (layer thicknesses), so the matrix is assembled and factorized every
        # step (sparse, O(n) work and memory)
        same_pattern = @timeit model.timers "sparse_assembly" _wd_pressure_matrix!(wd, rows, cols, vals, n)
        F = @timeit model.timers "cholesky" _wd_pressure_factor!(wd, same_pattern)
        sol = @timeit model.timers "solve" F \ rhs
        @inbounds for i in 1:nx, k in 1:nz
            idx[k, i] > 0 && (wd.q_nh[k, i] = sol[idx[k, i]] / (theta * theta))
        end
    end

    # 4. the non-hydrostatic pressure gradient (weight theta) enters the predictor; the caller repeats the solve
    @inbounds for f in 2:nx, k in 1:nz
        wd.hf[k, f] > 0.0 || continue
        dq = wd.q_nh[k, f] - wd.q_nh[k, f - 1]
        dq == 0.0 || (ws.u_hat[k, f] -= theta * dt / rho0 * dq / dx)
    end
    if radiate
        @inbounds for k in 1:nz
            if wd.hf[k, nx + 1] > 0.0 && idx[k, nx] > 0
                wd.q_right[k] = (wd.q_right_explicit[k] + th_q * mu_q * wd.q_nh[k, nx]) / (1.0 + th_q * mu_q)
                ws.u_hat[k, nx + 1] -= theta * dt / rho0 * (wd.q_right[k] - wd.q_nh[k, nx]) / (0.5 * dx)
            else
                wd.q_right[k] = 0.0   # surface, empty or dry cell: q = 0 there
            end
        end
    else
        fill!(wd.q_right, 0.0)
    end
    if radiate_left
        @inbounds for k in 1:nz
            if wd.hf[k, 1] > 0.0 && idx[k, 1] > 0
                wd.q_left[k] = (wd.q_left_explicit[k] + th_q * mu_q_left * wd.q_nh[k, 1]) / (1.0 + th_q * mu_q_left)
                ws.u_hat[k, 1] -= theta * dt / rho0 * (wd.q_nh[k, 1] - wd.q_left[k]) / (0.5 * dx)
            else
                wd.q_left[k] = 0.0   # surface, empty or dry cell: q = 0 there
            end
        end
    else
        fill!(wd.q_left, 0.0)
    end
    model.last_nonhydro_pcorr_max = maximum(abs, wd.q_nh)
    return nothing
end

# Pressure matrix wd.nh_matrix (n x n) from the triplets (rows, cols, vals); returns whether the pattern is that of the
# last step. The pattern changes only when a cell becomes an unknown or stops being one (surface or dry cell), the
# values every step. Same pattern: the values are written into the stored matrix through the triplet positions
# (wd.nh_nonzero), without the sort of `sparse` (0.8 ms and 1.9 MiB per step on 11000 unknowns, issue #10). The
# triplets have no duplicates (one per cell pair), so each one has its own non-zero.
function _wd_pressure_matrix!(wd::WetDryState, rows::Vector{Int}, cols::Vector{Int}, vals::Vector, n::Int)
    if wd.nh_factor !== nothing && size(wd.nh_matrix, 1) == n && rows == wd.nh_pattern_rows && cols == wd.nh_pattern_cols
        nz = wd.nh_matrix.nzval
        @inbounds for t in eachindex(vals)
            nz[wd.nh_nonzero[t]] = vals[t]
        end
        return true
    end
    wd.nh_matrix = sparse(rows, cols, vals, n, n)
    # WHY the triplet numbers as values: sparse sorts the triplets into CSC order; its nzval then names the triplet of
    # each non-zero (exact in Float64 for fewer than 2^53 triplets)
    order = sparse(rows, cols, Float64.(eachindex(rows)), n, n)
    resize!(wd.nh_nonzero, length(rows))
    @inbounds for (position, t) in enumerate(order.nzval)
        wd.nh_nonzero[Int(t)] = position
    end
    copyto!(resize!(wd.nh_pattern_rows, length(rows)), rows)
    copyto!(resize!(wd.nh_pattern_cols, length(cols)), cols)
    return false
end

# Cholesky factor of wd.nh_matrix. WHY the reuse: the symbolic analysis (fill-reducing ordering, elimination tree) is
# half of the factorization (5.7 of 11.7 ms on 11000 unknowns, issue #10) and depends only on the pattern. Same
# pattern: the same ordering and the same numeric operations, the factor is bit-identical to a new one.
function _wd_pressure_factor!(wd::WetDryState, same_pattern::Bool)
    wd.nh_factorizations += 1
    same_pattern && return cholesky!(wd.nh_factor, Symmetric(wd.nh_matrix))
    wd.nh_factor = cholesky(Symmetric(wd.nh_matrix))
    wd.nh_symbolic_analyses += 1
    return wd.nh_factor
end

"""
    _step_forward_wetting_drying!(model)

One hydrostatic time step with wetting and drying (semi-implicit free surface after Casulli & Cheng 1992 with layer
thicknesses at the old time level, equation numbers not checked; wet/dry faces and momentum-conservative advection after
Stelling & Duinmeijer 2003). Steps:
1. layer thicknesses of cells and faces at time n; baroclinic pressure;
2. predictor u_hat on the wet face layers (advection, baroclinic pressure, body force, damping, implicit viscosity);
3. free-surface solve with exact volume positivity (Casulli 2009): V(eta) + T eta = b with the non-negative column
   volumes V_i(eta) = max(0, H + eta_i - z_b,i), by Newton's method (`_wd_free_surface_solve!`); a lake at rest has a
   zero residual, so eta does not change by round-off;
4. u = u_hat - g dt d(eta^{n+1})/dx on the wet face layers; in the non-hydrostatic mode the gradient of the
   non-hydrostatic pressure is added to u_hat and steps 3-4 are repeated (`_wd_nonhydrostatic_correction!`);
5. eta^{n+1} from the flux divergence (water volume conserved to round-off; equal to V(eta^{n+1}) >= 0 of the solve);
6. vertical fluxes W through the fixed interfaces from the layer continuity, with the new thicknesses;
7. conservative upwind transport of rho with the same fluxes and volumes (no global mass correction).
Stability: explicit advection, CFL u dt/dx <= 1 (the momentum update is capped; the density transport keeps its bounds
by clipping with redistribution). The free surface is implicit (no gravity-wave CFL limit). The Froude limiter
(`apply_froude_limiter`) and the horizontal viscosity are not applied in this mode.
"""
_step_forward_wetting_drying!(model::Model) = _step_forward_wetting_drying!(model, model.wetdry)

# WHY the second method: model.wetdry is untyped (Any); passing it as an argument is a function barrier, so the kernels
# below are compiled for the concrete WetDryState (otherwise every field access allocates).
function _step_forward_wetting_drying!(model::Model, wd::WetDryState)
    cfg = model.config
    st = model.state
    ws = model.ws
    nz, nx = cfg.nz, cfg.nx
    dt, dx, g = cfg.dt, model.dx, cfg.g
    model.vegetation === nothing || throw(ArgumentError("vegetation is not supported with wetting_drying=true"))

    copyto!(ws.eta_old, st.eta)
    copyto!(wd.u_old, st.u)
    fill!(wd.outflow, 0.0)
    fill!(wd.rhs, 0.0)   # residual of the free-surface solve (part of the depth tolerance)
    _wd_check_depth_positivity(model, wd)

    # 1. geometry at time n
    @timeit model.timers "wd/geometry" begin
        _wd_cell_thicknesses!(wd.h, model, wd)
        _wd_face_thicknesses!(model, wd)
        cfg.wd_top_layer == "merged" ? _wd_remap_face_layers!(model, wd) : _wd_initialize_new_face_layers!(model, wd)
        copyto!(ws.u_old, st.u)   # u^n with the newly opened face layers (theta part of the continuity flux)
        @. wd.q = wd.hf * st.u   # discharges of time n, for the advection
    end
    @timeit model.timers "wd/baroclinic_pressure" _wd_baroclinic_pressure!(model, wd;
        rho=_baroclinic_density(model, _wd_buoyant_density(model, wd); wet=wd.h, min_thickness=WD_EMPTY_CELL_THICKNESS))

    # 2. momentum predictor, with the explicit parts of the theta scheme (issue #34): -(1 - theta) g d(eta^n)/dx and,
    # non-hydrostatic, -(1 - theta)/rho0 dq^n/dx (wd.q_nh still holds q^n)
    theta = cfg.theta
    explicit_weight = 1.0 - theta
    nonhydro = !cfg.use_hydrostatic && nz > 1
    implicit = cfg.implicit_vertical_terms
    second_order_momentum = cfg.momentum_advection_scheme == "tvd2"
    damp = cfg.rayleigh_damping > 0.0 ? max(0.0, 1.0 - cfg.rayleigh_damping * dt) : 1.0
    # water-level boundary or free outflow (set_x_inflow_outflow!(...; right_level / right_free)): the boundary face
    # nx + 1 has its own momentum equation with its surface slope (_right_boundary_slope) and the baroclinic pressure
    # gradient of face nx
    open_bc = _right_face_computed(model)
    open_bc && _check_right_level_config(cfg)
    model.x_boundary_eta_left === nothing || _check_left_level_config(cfg)
    fhi = open_bc ? nx + 1 : nx
    radiate_q = open_bc && cfg.nonhydro_open_face == "radiation"   # pressure radiation at the open face (issue #86)
    # free left face (issue #86): face 1 has its own momentum equation, the mirror of the right one (surface slope
    # (eta_1 - eta_b)/(dx/2), _left_boundary_slope; baroclinic pressure gradient of face 2; ghost cell = first cell)
    left_open = wd.left_open !== nothing
    flo = left_open ? 1 : 2
    radiate_q_left = left_open && cfg.nonhydro_open_face == "radiation"
    ke = model.turbulence
    eddy_diffusion = ke !== nothing && ke.horizontal_diffusion
    # WHY hydrostatic only: the remapped form conserves momentum exactly for a flow without vertical structure; in the
    # non-hydrostatic mode the bore front has strong vertical structure and the remapped form gained 4.5 % momentum in
    # 10 layers (the column form lost 1.4 %), so there the column form stays (docs section 38)
    remap = cfg.layer_momentum_advection == "remapped" && cfg.use_hydrostatic && nz > 1
    fill!(wd.remap.active, false)
    @timeit model.timers "wd/momentum_predictor" @inbounds for f in flo:fhi
        depth_totals = _wd_advection_depth_totals(wd, f)
        # remapped advection (issue #80) on interior faces over a flat bed without an active gate; elsewhere the column form
        remapped = remap && 2 <= f <= nx && _wd_remap_flat_neighbourhood(wd, f)
        remapped && _wd_remapped_tendencies!(model, wd, f, second_order_momentum, !implicit)
        boundary = f == nx + 1
        left_boundary = f == 1
        surface_slope = boundary ? _right_boundary_slope(ws.eta_old, model, dx; old=true) :
            left_boundary ? _left_boundary_slope(ws.eta_old, model, dx; old=true) : (ws.eta_old[f] - ws.eta_old[f - 1]) / dx
        explicit_surface = explicit_weight * g * surface_slope   # m/s^2
        fp = boundary ? nx : (left_boundary ? 2 : f)   # face of the baroclinic pressure gradient (zero gradient at the boundary)
        for k in 1:nz
            if wd.hf[k, f] > 0.0
                adv = remapped ? wd.remap.tend[k] : _wd_advection_tendency(model, wd, k, f, second_order_momentum, depth_totals; vertical=!implicit)
                tend = adv - (wd.p_anom[k, fp] - wd.p_anom[k, fp - 1]) / (cfg.rho0 * dx) + cfg.body_force_x
                eddy_diffusion && (tend += _wd_horizontal_eddy_diffusion(model, wd, ke, k, f))
                # incoming wave entering with the current through a free face (issue #86)
                boundary && open_bc && (tend += _incoming_inflow_advection(model, wd, wd.char_right, k, f, 1.0))
                left_boundary && (tend += _incoming_inflow_advection(model, wd, wd.char_left, k, f, -1.0))
                if theta != 1.0
                    tend -= explicit_surface
                    # open right boundary face: dq/dx = 0 there (issue #49, _wd_nonhydrostatic_correction!)
                    nonhydro && !boundary && !left_boundary && (tend -= explicit_weight * (wd.q_nh[k, f] - wd.q_nh[k, f - 1]) / (cfg.rho0 * dx))
                    # pressure radiation (issue #86): the face pressure q_b of time n, half a cell from the adjacent centre
                    nonhydro && boundary && radiate_q && (tend -= explicit_weight * (wd.q_right[k] - wd.q_nh[k, nx]) / (cfg.rho0 * 0.5 * dx))
                    nonhydro && left_boundary && radiate_q_left && (tend -= explicit_weight * (wd.q_nh[k, 1] - wd.q_left[k]) / (cfg.rho0 * 0.5 * dx))
                end
                ws.u_hat[k, f] = damp * (st.u[k, f] + dt * tend)
            else
                ws.u_hat[k, f] = 0.0
            end
        end
        if cfg.sponge_length > 0.0 && cfg.sponge_rate > 0.0
            s = ((f - 1) * dx - (cfg.length - cfg.sponge_length)) / cfg.sponge_length
            if s > 0.0
                factor = 1.0 / (1.0 + dt * cfg.sponge_rate * min(s, 1.0)^2)
                for k in 1:nz
                    ws.u_hat[k, f] *= factor
                end
            end
        end
        implicit ? _wd_implicit_vertical!(model, wd, f) : _wd_vertical_viscosity!(model, wd, ws.u_hat, f)
    end
    _wd_boundary_velocities!(model, wd, ws.u_hat, wd.time + dt)
    # characteristic open faces (issue #86): explicit part of the natural condition, once per step
    _wd_characteristic_predictor!(model, wd)

    # 3.-4. free-surface solve with exact volume positivity, velocity correction
    @timeit model.timers "wd/free_surface_solve" _wd_free_surface_solve!(model, wd)

    # 4b. non-hydrostatic: pressure q from the projection of the free-surface velocities, its gradient added to the
    # predictor, and the free-surface solve repeated, so that the final velocities come from the positive solve (a
    # single layer has no vertical structure: hydrostatic)
    if !cfg.use_hydrostatic && nz > 1
        @timeit model.timers "wd/nonhydro_correction" _wd_nonhydrostatic_correction!(model, wd)
        @timeit model.timers "wd/free_surface_solve" _wd_free_surface_solve!(model, wd)
    end

    # 5. new water level from the flux divergence (equals V(eta^{n+1}) of the solve)
    @timeit model.timers "wd/level_and_w" _wd_update_level_from_fluxes!(model, wd)

    # 6. vertical fluxes through the fixed interfaces (m/s), integrated upwards from the bed
    @timeit model.timers "wd/level_and_w" @inbounds for i in 1:nx
        layer_thicknesses!(view(wd.h_new, :, i), _wd_level(model, i), wd.bed[i], model.dz; merge=_wd_merge(cfg))
    end
    max_mismatch = 0.0
    @timeit model.timers "wd/level_and_w" @inbounds for i in 1:nx
        st.w[1, i] = 0.0
        for k in 1:nz
            st.w[k + 1, i] = st.w[k, i] - ((wd.h_new[k, i] - wd.h[k, i]) / dt + (wd.q[k, i + 1] - wd.q[k, i]) / dx)
        end
        max_mismatch = max(max_mismatch, abs(st.w[nz + 1, i]))
        st.w[nz + 1, i] = 0.0
    end
    model.last_kinematic_mismatch = max_mismatch

    # 7. constituent transport; k-epsilon (issue #30): turbulent density diffusion with nu_t of time n, then k and eps on
    # the new geometry
    @timeit model.timers "wd/density_transport" _wd_transport_density!(model, wd)
    # suspended sediment (issue #64): with nu_t of time n, like the turbulent density diffusion; updates the mixture density
    wd.sediment === nothing || @timeit model.timers "wd/sediment" _wd_sediment_step!(model, wd, wd.sediment)
    if model.turbulence !== nothing
        @timeit model.timers "wd/density_transport" _wd_turbulent_density_diffusion!(model, wd, model.turbulence)
        @timeit model.timers "wd/k_epsilon" _wd_k_epsilon_step!(model, wd, model.turbulence)
    end
    cfg.wd_top_layer == "merged" && _wd_rebuild_interface_velocities!(model, wd)
    wd.time += dt

    # Hydrostatic and non-hydrostatic pressure components in the wet cells.
    copyto!(wd.h, wd.h_new)
    @timeit model.timers "wd/final_pressure" _wd_baroclinic_pressure!(model, wd)
    @timeit model.timers "wd/final_pressure" @inbounds for i in 1:nx, k in 1:nz
        p_hydro = wd.h[k, i] > 0.0 ? cfg.rho0 * g * (_wd_level(model, i) - (k - 0.5) * model.dz) + wd.p_anom[k, i] : 0.0
        p_corr = cfg.use_hydrostatic ? 0.0 : wd.q_nh[k, i]
        model.last_p_hydro[k, i] = p_hydro
        model.last_p_corr[k, i] = p_corr
        st.p[k, i] = p_hydro + p_corr
    end
    cfg.use_hydrostatic && (model.last_nonhydro_pcorr_max = 0.0)
    @views @. ws.rho_w = 0.5 * (st.rho[2:end, :] + st.rho[1:end-1, :])
    buoyancy!(model.last_buoy_w, ws.rho_w, model.rho_0_w_2d)
    return nothing
end

# Face value of rho for the transport: upwind1 rho_up; tvd2 the flux-limited Lax-Wendroff value
#   rho_f = rho_up + (1 - c)/2 * MC(rho_up - rho_upup, rho_down - rho_up),   c = min(1, |flux| dt / h_up)  (per cell volume),
# differences taken along the flow, first order where the cell behind the upwind cell is missing or empty (`has_upup`).
# Same scheme as the tvd2 transport of the other modes (_density_face_fluxes!; van Leer 1977, Sweby 1984), with the
# Courant number of the cut cell.
@inline function _wd_face_density(rho_up::Real, rho_upup::Real, rho_down::Real, courant::Real, has_upup::Bool, second::Bool)
    (second && has_upup) || return rho_up
    return rho_up + 0.5 * (1.0 - min(courant, 1.0)) * _monotonized_central_scalar(rho_up - rho_upup, rho_down - rho_up)
end

# Conservative transport of rho with the layer discharges q and interface fluxes W of the step:
#   (h rho)^{n+1} = (h rho)^n - dt/dx (F_{f+1} - F_f) - dt (G_{k+1} - G_k),  F = q rho_f, G = W rho_f (`_wd_face_density`).
# It is evaluated as the increment D = (h rho)^{n+1} - h^{n+1} rho^n (kg/m^2), rho^{n+1} = rho^n + D / h^{n+1}.
# WHY the increment: without fluxes and volume change D is exactly zero and rho does not change by round-off, which
# (h rho)/h would do; in a stratified lake at rest such round-off drives an internal motion.
# Empty cells (h^{n+1} below WD_EMPTY_CELL_THICKNESS) keep their density and pass D to the thickest cell of the column;
# values outside the initial bounds (possible in thin cells, where the outflow can exceed the old volume) are clipped and
# the clipped mass is redistributed over the cells with room for it. Both keep the constituent mass exact to round-off.
function _wd_transport_density!(model::Model, wd::WetDryState)
    _wd_transport_scalar!(model, wd, model.state.rho, model.x_boundary_rho_left, model.rho_min_init, model.rho_max_init)
    return nothing
end

# The transport above for any cell field `rho` (the density, or a suspended-sediment concentration, issue #64) with the
# inflow value `rho_in` at the left boundary face (nothing: the first cell, zero gradient; one value or one per layer),
# the value `rho_right` of water entering through the right boundary face (nothing: the last cell, zero gradient) and the
# bounds [rho_lo, rho_hi] of the clipping. Returns the amounts (units of rho times m^2, per unit width) that entered
# through the left and left through the right boundary face in the step, for the budgets.
function _wd_transport_scalar!(model::Model, wd::WetDryState, rho::AbstractMatrix{<:Real}, rho_in, rho_lo::Real,
    rho_hi::Real; rho_right=nothing)
    cfg = model.config
    st = model.state
    q, W = wd.q, st.w
    nz, nx = cfg.nz, cfg.nx
    dt, dx = cfg.dt, model.dx
    M = wd.mass   # the increment D
    @inbounds for i in 1:nx, k in 1:nz
        M[k, i] = (wd.h[k, i] - wd.h_new[k, i]) * rho[k, i]
    end
    second = cfg.transport_advection_scheme == "tvd2"
    wet(k, i) = wd.h[k, i] > WD_EMPTY_CELL_THICKNESS
    inflow = outflow = 0.0
    @inbounds for f in 1:(nx + 1), k in 1:nz
        qf = q[k, f]
        qf == 0.0 && continue
        up = qf > 0.0 ? (f > 1 ? f - 1 : 1) : (f <= nx ? f : nx)
        down = qf > 0.0 ? f : f - 1               # only used when has_upup (interior faces)
        upup = qf > 0.0 ? f - 2 : f + 1
        has = 1 <= upup <= nx && 1 <= down <= nx && 1 < f <= nx && wet(k, upup) && wet(k, up) && wet(k, down)
        courant = has ? abs(qf) * dt / (dx * wd.h[k, up]) : 0.0
        rho_f = (f == 1 && qf > 0.0 && rho_in !== nothing) ? _inflow_value(rho_in, k) :
            (f == nx + 1 && qf < 0.0 && rho_right !== nothing) ? _inflow_value(rho_right, k) :
            _wd_face_density(rho[k, up], has ? rho[k, upup] : 0.0, has ? rho[k, down] : 0.0, courant, has, second)
        flux = dt / dx * qf * rho_f
        f > 1 && (M[k, f - 1] -= flux)
        f <= nx && (M[k, f] += flux)
        f == 1 && (inflow += flux * dx)
        f == nx + 1 && (outflow += flux * dx)
    end
    @inbounds for i in 1:nx, k in 2:nz
        wk = W[k, i]
        wk == 0.0 && continue
        up, down, upup = wk > 0.0 ? (k - 1, k, k - 2) : (k, k - 1, k + 1)
        has = 1 <= upup <= nz && wet(upup, i) && wet(up, i) && wet(down, i)
        courant = has ? abs(wk) * dt / wd.h[up, i] : 0.0
        flux = dt * wk * _wd_face_density(rho[up, i], has ? rho[upup, i] : 0.0, has ? rho[down, i] : 0.0, courant, has, second)
        M[k - 1, i] -= flux
        M[k, i] += flux
    end

    pool = 0.0   # kg/m^2 (per dx): mass that could not be kept in place
    @inbounds for i in 1:nx
        residual = 0.0
        kmax = 0
        for k in 1:nz
            if wd.h_new[k, i] > WD_EMPTY_CELL_THICKNESS
                kmax = (kmax == 0 || wd.h_new[k, i] > wd.h_new[kmax, i]) ? k : kmax
            else
                residual += M[k, i]
            end
        end
        if kmax == 0
            pool += residual
            continue
        end
        M[kmax, i] += residual
        for k in 1:nz
            hk = wd.h_new[k, i]
            hk > WD_EMPTY_CELL_THICKNESS || continue
            r = rho[k, i] + M[k, i] / hk
            rc = clamp(r, rho_lo, rho_hi)
            pool += (r - rc) * hk
            rho[k, i] = rc
        end
    end
    if pool != 0.0
        # WHY slack-weighted: every cell takes a share proportional to its room up to the bound, so no bound is crossed
        room = 0.0
        @inbounds for i in 1:nx, k in 1:nz
            hk = wd.h_new[k, i]
            hk > WD_EMPTY_CELL_THICKNESS || continue
            room += (pool > 0.0 ? rho_hi - rho[k, i] : rho[k, i] - rho_lo) * hk
        end
        if room > 0.0
            share = pool / room
            @inbounds for i in 1:nx, k in 1:nz
                hk = wd.h_new[k, i]
                hk > WD_EMPTY_CELL_THICKNESS || continue
                rho[k, i] += share * (pool > 0.0 ? rho_hi - rho[k, i] : rho[k, i] - rho_lo)
            end
        end
    end
    return inflow, outflow
end


# k-epsilon in the wetting-drying step (issue #30): the closure of src/turbulence.jl (equations, constants, Patankar sinks,
# wall functions) on the cut cells of the new geometry h^{n+1}. Wet cells are thicker than WD_EMPTY_CELL_THICKNESS; in a
# column they are contiguous from the lowest (cut by the bed) to the top wet cell (cut by the surface).
# - Shear production from the velocities of time n+1: cell-centred u from the open faces of the step, du/dz between the
#   cell centres (one-sided in the lowest and the top wet cell), dw/dx between wet neighbours, and
#   2 (du/dx)^2 + 2 (dw/dz)^2 = 4 (du/dx)^2.
#   WHY dw/dz = -du/dx: the interface fluxes w of the cut cells include the volume change of the cell, so
#   (w_{k+1} - w_k) / h is no strain rate in a thin cut cell (it grows without bound as h -> 0).
# - Advection: conservative first-order upwind with the layer discharges q and interface fluxes w of the step, in the
#   increment form of _wd_transport_density!, so that a uniform field stays uniform while cells fill and empty.
# - Diffusion, sources and sinks: per column backward Euler in finite volumes with the cell thicknesses h_k and the centre
#   distances (h_k + h_{k+1}) / 2; first order in time, positive for any dt (as on the fixed grid).
# - Bed: with a roughness length z0 > 0 the log-law wall functions in the lowest wet cell, at the height
#   z1 = max(h / 2, WD_FRICTION_Z1_MIN_FACTOR z0) of its centre (as the bed friction of _wd_implicit_vertical!).
#   Surface: eps = C_mu^(3/4) k^(3/2) / (kappa 0.07 H) in the top wet cell, H the water depth of the column (Rodi 1993).
# Cells that are empty keep their k and eps (floors after drying) and have nu_t = 0.

# Eddy viscosity (m^2/s) of layer k at a u-face from its cells L and R: mean over the wet ones, 0 if both are empty.
@inline function _wd_face_eddy_viscosity(ke::KEpsilonState, wd::WetDryState, k::Int, L::Int, R::Int)
    wl = wd.h[k, L] > WD_EMPTY_CELL_THICKNESS
    wr = wd.h[k, R] > WD_EMPTY_CELL_THICKNESS
    (wl && wr) && return 0.5 * (ke.nu_t[k, L] + ke.nu_t[k, R])
    wl && return ke.nu_t[k, L]
    wr && return ke.nu_t[k, R]
    return 0.0
end

# Lowest and top wet cell of column i in the thicknesses h (0, 0: dry column).
@inline function _wd_wet_range(h::AbstractMatrix{<:Real}, i::Int)
    klo, khi = 0, 0
    @inbounds for k in axes(h, 1)
        if h[k, i] > WD_EMPTY_CELL_THICKNESS
            klo == 0 && (klo = k)
            khi = k
        end
    end
    return klo, khi
end

# Cell-centred horizontal velocity (m/s) of layer k in column i: mean of the open faces of the step.
@inline function _wd_cell_velocity(model::Model, wd::WetDryState, k::Int, i::Int)
    u = model.state.u
    ol, or = wd.hf[k, i] > 0.0, wd.hf[k, i + 1] > 0.0
    (ol && or) && return 0.5 * (u[k, i] + u[k, i + 1])
    ol && return u[k, i]
    or && return u[k, i + 1]
    return 0.0
end

# Implicit turbulent density diffusion K = nu_t / sigma_rho (m^2/s) per column on the wet cells of h^{n+1}, in finite
# volumes: h_k (rho_k - rho*_k) = dt [K_{k+1/2} (rho_{k+1} - rho_k) / d_{k+1/2} - K_{k-1/2} (rho_k - rho_{k-1}) / d_{k-1/2}],
# d = centre distance; zero flux at the bed and the surface. Conserves sum_k h_k rho_k (columns of the matrix sum to h).
function _wd_turbulent_density_diffusion!(model::Model, wd::WetDryState, ke::KEpsilonState)
    cfg = model.config
    rho, h, dt = model.state.rho, wd.h_new, cfg.dt
    lower, diag, upper, rhs, sol = ke.lower, ke.diag, ke.upper, ke.rhs, ke.sol
    @inbounds for i in 1:cfg.nx
        klo, khi = _wd_wet_range(h, i)
        n = khi - klo + 1
        (klo == 0 || n < 2) && continue
        for j in 1:n
            k = klo + j - 1
            kd = j > 1 ? dt * 0.5 * (ke.nu_t[k, i] + ke.nu_t[k - 1, i]) / KE_SIGMA_RHO / (0.5 * (h[k, i] + h[k - 1, i])) : 0.0
            ku = j < n ? dt * 0.5 * (ke.nu_t[k, i] + ke.nu_t[k + 1, i]) / KE_SIGMA_RHO / (0.5 * (h[k, i] + h[k + 1, i])) : 0.0
            lower[j], upper[j], diag[j], rhs[j] = -kd, -ku, h[k, i] + kd + ku, h[k, i] * rho[k, i]
        end
        _solve_tridiagonal_column!(sol, lower, diag, upper, rhs, n)
        for j in 1:n
            rho[klo + j - 1, i] = sol[j]
        end
    end
    return nothing
end

# Upwind advection of the cell field phi with the step fluxes (increment form of _wd_transport_density!), into `out`;
# the inflow at the left boundary carries `inflow[k]` (left_k / left_eps, issue #42), or without it the value of the first
# cell (zero gradient). With ke.horizontal_diffusion also
# the explicit horizontal eddy diffusion nu_t / sigma (sigma the Prandtl number of phi, -), conservative, through the open
# faces between wet cells; each face gets at most half of WD_HORIZONTAL_DIFFUSION_MAX of the thinner cell.
function _wd_advect_turbulence!(out::AbstractMatrix{<:Real}, phi::AbstractMatrix{<:Real}, model::Model, wd::WetDryState,
    floor_value::Real, sigma::Real; inflow::Union{Nothing,AbstractVector{<:Real}}=nothing)
    cfg = model.config
    nz, nx, dt, dx = cfg.nz, cfg.nx, cfg.dt, model.dx
    q, W, M = wd.q, model.state.w, wd.mass
    @inbounds for i in 1:nx, k in 1:nz
        M[k, i] = (wd.h[k, i] - wd.h_new[k, i]) * phi[k, i]
    end
    @inbounds for f in 1:(nx + 1), k in 1:nz
        qf = q[k, f]
        qf == 0.0 && continue
        up = qf > 0.0 ? max(f - 1, 1) : min(f, nx)
        value = (f == 1 && qf > 0.0 && inflow !== nothing) ? inflow[k] : phi[k, up]
        flux = dt / dx * qf * value
        f > 1 && (M[k, f - 1] -= flux)
        f <= nx && (M[k, f] += flux)
    end
    @inbounds for i in 1:nx, k in 2:nz
        wk = W[k, i]
        wk == 0.0 && continue
        flux = dt * wk * phi[wk > 0.0 ? k - 1 : k, i]
        M[k - 1, i] -= flux
        M[k, i] += flux
    end
    ke = model.turbulence
    if ke.horizontal_diffusion
        @inbounds for f in 2:nx, k in 1:nz
            hL, hR = wd.h_new[k, f - 1], wd.h_new[k, f]
            (wd.hf[k, f] > 0.0 && hL > WD_EMPTY_CELL_THICKNESS && hR > WD_EMPTY_CELL_THICKNESS) || continue
            d = dt / dx^2 * 0.5 * (ke.nu_t[k, f - 1] + ke.nu_t[k, f]) / sigma * wd.hf[k, f]   # m
            flux = min(d, 0.5 * WD_HORIZONTAL_DIFFUSION_MAX * min(hL, hR)) * (phi[k, f] - phi[k, f - 1])
            M[k, f - 1] += flux
            M[k, f] -= flux
        end
    end
    @inbounds for i in 1:nx, k in 1:nz
        hk = wd.h_new[k, i]
        out[k, i] = hk > WD_EMPTY_CELL_THICKNESS ? max(phi[k, i] + M[k, i] / hk, floor_value) : phi[k, i]
    end
    return out
end

function _wd_k_epsilon_step!(model::Model, wd::WetDryState, ke::KEpsilonState)
    cfg = model.config
    st = model.state
    nz, nx, dt, dx, dz = cfg.nz, cfg.nx, cfg.dt, model.dx, model.dz
    h = wd.h_new
    k, eps, nu_t = ke.k, ke.eps, ke.nu_t
    model.vegetation === nothing && fill!(ke.veg_work, 0.0)
    rho_b = _wd_buoyant_density(model, wd)   # kg/m^3, the mixture density with coupled sediment (issue #64)
    zc(kk, i) = max((kk - 1) * dz, wd.bed[i]) + 0.5 * h[kk, i]   # m, height of the wet-cell centre
    wet(kk, i) = 1 <= i <= nx && h[kk, i] > WD_EMPTY_CELL_THICKNESS
    wc(kk, i) = 0.5 * (st.w[kk, i] + st.w[kk + 1, i])
    # shear production and buoyancy with nu_t of time n
    @inbounds for i in 1:nx
        klo, khi = _wd_wet_range(h, i)
        klo == 0 && continue
        for kk in klo:khi
            kup, kdn = min(kk + 1, khi), max(kk - 1, klo)
            dudz = kup == kdn ? 0.0 : (_wd_cell_velocity(model, wd, kup, i) - _wd_cell_velocity(model, wd, kdn, i)) / (zc(kup, i) - zc(kdn, i))
            il, ir = wet(kk, i - 1) ? i - 1 : i, wet(kk, i + 1) ? i + 1 : i
            dwdx = ir == il ? 0.0 : (wc(kk, ir) - wc(kk, il)) / ((ir - il) * dx)
            dudx = (st.u[kk, i + 1] - st.u[kk, i]) / dx
            ke.strain2[kk, i] = (dudz + dwdx)^2 + 4 * dudx^2
            ke.production[kk, i] = nu_t[kk, i] * ke.strain2[kk, i]
            n2 = kup == kdn ? 0.0 : -(cfg.g / cfg.rho0) * (rho_b[kup, i] - rho_b[kdn, i]) / (zc(kup, i) - zc(kdn, i))
            ke.buoyancy[kk, i] = -(nu_t[kk, i] / KE_SIGMA_RHO) * n2
        end
    end
    _wd_advect_turbulence!(ke.k_adv, k, model, wd, KE_K_MIN, KE_SIGMA_K; inflow=wd.left_k)
    _wd_advect_turbulence!(ke.eps_adv, eps, model, wd, KE_EPS_MIN, KE_SIGMA_EPS; inflow=wd.left_eps)
    copyto!(k, ke.k_adv)
    copyto!(eps, ke.eps_adv)
    lower, diag, upper, rhs, sol = ke.lower, ke.diag, ke.upper, ke.rhs, ke.sol
    z0 = cfg.bed_roughness_length
    @inbounds for i in 1:nx
        klo, khi = _wd_wet_range(h, i)
        klo == 0 && continue
        depth = 0.0
        for kk in klo:khi
            depth += h[kk, i]
        end
        wall = z0 > 0.0
        equilibrium = wall && ke.wall_function === :equilibrium
        # wall cell kw: the lowest wet cell; with the equilibrium wall function the lowest one of at least
        # ke.wall_cell_min_fraction dz, and the shaved cells below it take its values
        kw = klo
        below = 0.0   # m, thickness of the shaved cells under the wall cell
        if equilibrium
            while kw < khi && h[kw, i] < ke.wall_cell_min_fraction * dz
                below += h[kw, i]
                kw += 1
            end
        end
        z1 = max(below + 0.5 * h[kw, i], WD_FRICTION_Z1_MIN_FACTOR * z0)   # m, height of the wall-cell centre
        ustar2 = (wall && !equilibrium) ? bed_friction_coefficient(z0, z1) * _wd_cell_velocity(model, wd, klo, i)^2 : 0.0   # m^2/s^2
        for (field, sigma) in ((:k, KE_SIGMA_K), (:eps, KE_SIGMA_EPS))
            phi = field === :k ? k : eps
            floor_value = field === :k ? KE_K_MIN : KE_EPS_MIN
            # equilibrium: k of the wall cell is solved with zero flux at the bed, eps of the cell is a Dirichlet value
            solve_wall_cell = !wall || (equilibrium && field === :k)
            first = solve_wall_cell ? kw : kw + 1
            if !solve_wall_cell
                # wall cell: wall-function value (Dirichlet), coupled through the interface above it
                if equilibrium
                    # eps = u_k^3 / (kappa z1), u_k = C_mu^(1/4) k^(1/2) from k solved just before (Launder & Spalding 1974)
                    bed_value = (KE_C_MU^0.25 * sqrt(k[kw, i]))^3 / (KE_KARMAN * z1)
                else
                    bed_value = field === :k ? ustar2 / sqrt(KE_C_MU) : sqrt(ustar2)^3 / (KE_KARMAN * z1)
                end
                phi[kw, i] = max(bed_value, floor_value)
            end
            n = khi - first + 1
            n >= 1 || continue
            for j in 1:n
                kk = first + j - 1
                hk = h[kk, i]
                ratio = eps[kk, i] / k[kk, i]   # 1/s, eps/k after the advection
                inv_t = 1 / _ke_time(ke, k[kk, i], eps[kk, i], ke.strain2[kk, i])   # 1/s, time-scale bound (>= eps/k)
                kd = kk > kw ? dt * (cfg.nu + 0.5 * (nu_t[kk, i] + nu_t[kk - 1, i]) / sigma) / (0.5 * (hk + h[kk - 1, i])) : 0.0
                ku = kk < khi ? dt * (cfg.nu + 0.5 * (nu_t[kk, i] + nu_t[kk + 1, i]) / sigma) / (0.5 * (hk + h[kk + 1, i])) : 0.0
                p, gb, pv = ke.production[kk, i], ke.buoyancy[kk, i], ke.veg_work[kk, i]
                if field === :k
                    source = p + max(gb, 0.0) + ke.cfk * pv
                    sink = ratio + max(-gb, 0.0) / k[kk, i]
                else
                    source, sink = _ke_eps_source_sink(ke, p, gb, pv, inv_t, eps[kk, i])
                end
                if equilibrium && field === :k && j == 1
                    # wall cell (Launder & Spalding 1974; Versteeg & Malalasekera 2007, sec. 9.11): velocity scale
                    # u_k = C_mu^(1/4) k^(1/2) instead of the local u*, shear stress tau = kappa u_k |U| / ln(z1/z0)
                    # (m^2/s^2), production tau dU/dz = tau^2 / (kappa u_k z1), dissipation u_k^3 / (kappa z1) as the sink
                    # u_k^3 / (kappa z1 k). WHY: u_k stays finite where the bed velocity changes sign (reattachment).
                    uk = KE_C_MU^0.25 * sqrt(k[kk, i])   # m/s
                    tau = KE_KARMAN * uk * abs(_wd_cell_velocity(model, wd, kk, i)) / log(z1 / z0)
                    source = tau^2 / (KE_KARMAN * uk * z1) + max(gb, 0.0) + ke.cfk * pv
                    sink = KE_C_MU^0.75 * sqrt(k[kk, i]) / (KE_KARMAN * z1) + max(-gb, 0.0) / k[kk, i]
                end
                lower[j] = -kd
                upper[j] = -ku
                diag[j] = hk * (1 + dt * sink) + kd + ku
                rhs[j] = hk * (phi[kk, i] + dt * source)
                if !solve_wall_cell && j == 1
                    # the Dirichlet cell below
                    lower[j] = 0.0
                    rhs[j] += kd * phi[kw, i]
                end
            end
            if field === :eps && ke.surface_condition === :rodi && khi >= first
                eps_s = KE_C_MU^0.75 * k[khi, i]^1.5 / (KE_KARMAN * KE_SURFACE_LENGTH * depth)
                lower[n], upper[n], diag[n], rhs[n] = 0.0, 0.0, 1.0, eps_s
            end
            _solve_tridiagonal_column!(sol, lower, diag, upper, rhs, n)
            for j in 1:n
                phi[first + j - 1, i] = max(sol[j], floor_value)
            end
        end
        for kk in klo:(kw - 1)
            k[kk, i] = k[kw, i]
            eps[kk, i] = eps[kw, i]
        end
    end
    @inbounds for i in 1:nx, kk in 1:nz
        nu_t[kk, i] = h[kk, i] > WD_EMPTY_CELL_THICKNESS ? KE_C_MU * k[kk, i] * _ke_time(ke, k[kk, i], eps[kk, i], ke.strain2[kk, i]) : 0.0
    end
    return nothing
end
