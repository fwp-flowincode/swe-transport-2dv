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
const WD_NEWTON_RTOL = 1e-13          # -, Newton stops when |F_i| <= WD_NEWTON_RTOL * (magnitudes in F_i) for all i
const WD_NEWTON_EXTRA_ITERATIONS = 10 # -, iterations allowed beyond the nx + 1 of the finite-termination bound
# -, safety factor on the round-off bound gamma_m S, m = 2 nz + 8, of a computed depth (docs/depth_positivity.md, section 5);
# a depth below -WD_POSITIVITY_ROUNDOFF_FACTOR gamma_m S stops the run
const WD_POSITIVITY_ROUNDOFF_FACTOR = 2

"""
    WetDryState

Bed, layer geometry, fluxes and work arrays of the wetting-drying time step (sizes for nz layers and nx columns).
- `bed` (m, nx): bed level per column; `bed_face` (m, nx+1): bed level at the u-faces (higher neighbour).
- `h`, `h_new` (m, nz x nx): layer thicknesses at the old and new time level.
- `hf` (m, nz x (nx+1)): layer thicknesses at the u-faces (upwind water level, face bed).
- `q` (m^2/s, nz x (nx+1)): layer discharges hf*u of the step.
- `p_anom` (Pa, nz x nx): baroclinic pressure g * integral from z to zeta of (rho - rho0), at the nominal layer centres.
- `time` (s): model time, advanced by every step (used by a time-dependent boundary velocity).
"""
mutable struct WetDryState{T<:Real}
    bed::Vector{T}
    bed_face::Vector{T}
    h::Matrix{T}
    h_new::Matrix{T}
    hf::Matrix{T}
    q::Matrix{T}
    p_anom::Matrix{T}
    mass::Matrix{T}
    a_face::Vector{T}
    U_hat::Vector{T}
    rhs::Vector{T}
    cp::Vector{T}
    dp::Vector{T}
    delta::Vector{T}
    eta_iter::Vector{T}   # m, Newton iterate of the free-surface solve
    outflow::Vector{T}    # m, outflow volume of each column in the last step (round-off scale of the depth check)
    tri_a::Vector{T}
    tri_b::Vector{T}
    tri_c::Vector{T}
    tri_d::Vector{T}
    # non-hydrostatic correction (use_hydrostatic=false)
    u_old::Matrix{T}           # m/s, nz x (nx+1): u of time n
    w_star::Matrix{T}          # m/s, (nz+1) x nx: predicted vertical velocity at the interfaces
    h_nh::Matrix{T}            # m, nz x nx: layer thicknesses of the provisional level of the free-surface solve
    q_nh::Matrix{T}            # Pa, nz x nx: non-hydrostatic pressure
    nh_index::Matrix{Int}      # unknown number of a cell of the pressure problem (0: surface, dry or empty cell)
    nh_rows::Vector{Int}
    nh_cols::Vector{Int}
    nh_vals::Vector{T}
    nh_rhs::Vector{T}
    time::Float64
    clamped_volume::Float64   # m^2 (per unit width): volume added by setting round-off negative depths to zero
    newton_iterations::Int        # Newton iterations of the last free-surface solve
    newton_iterations_max::Int    # largest number of Newton iterations so far
    left_velocity::Any   # nothing, or a function t (s) -> u (m/s, scalar or length nz) at the left boundary face
end

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
        zeros(T, nz, nx + 1), zeros(T, nz + 1, nx), zeros(T, nz, nx), zeros(T, nz, nx), zeros(Int, nz, nx),
        Int[], Int[], T[], T[], 0.0, 0.0, 0, 0, nothing)
end

"""
    layer_thicknesses!(h, zeta, zb, dz) -> h

Thicknesses (m) of the layers k = 1..nz (nz = length(h)) of a column with water level `zeta` (m) and bed level `zb` (m),
both above the grid bottom, on interfaces z_k = (k-1) dz (m):

    h_k = max(0, min(z_{k+1}, zeta) - max(z_k, zb)),   z_{nz+1} = +infinity (the top layer is open above).

sum(h) = max(zeta - zb, 0). For a flat bed and zeta in the top layer this is the variable top layer (dz + eta on top).
"""
function layer_thicknesses!(h::AbstractVector{<:Real}, zeta::Real, zb::Real, dz::Real)
    nz = length(h)
    @inbounds for k in 1:nz
        lo = max((k - 1) * dz, zb)
        hi = k == nz ? zeta : min(k * dz, zeta)
        h[k] = max(zero(eltype(h)), hi - lo)
    end
    return h
end

_wd_level(model::Model, i::Int) = model.config.depth + model.state.eta[i]   # m, water level zeta of column i

# Surface elevation eta (m) of a dry column: the smallest float with fl(depth + eta) >= zb, so that the computed depth
# zeta - zb is exactly >= 0 (eta = zb - depth alone can round to one ulp below the bed).
function _wd_dry_eta(depth::Real, zb::Real)
    eta = zb - depth
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
        st.eta[i] = levels[i] > wd.bed[i] ? levels[i] - cfg.depth : _wd_dry_eta(cfg.depth, wd.bed[i])
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
    wd.time = 0.0
    _refresh_density_metadata!(model)
    return nothing
end

# Clamps the water level of every column to its bed (a dry column has zeta = z_b), e.g. after a generic initializer.
function _wd_clamp_to_bed!(model::Model)
    wd = model.wetdry
    @inbounds for i in eachindex(model.state.eta)
        if model.config.depth + model.state.eta[i] < wd.bed[i]
            model.state.eta[i] = _wd_dry_eta(model.config.depth, wd.bed[i])
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
        layer_thicknesses!(view(h, :, i), _wd_level(model, i), wd.bed[i], model.dz)
    end
    return h
end

# Layer thicknesses at the u-faces: face bed z_b,f and upwind water level (sign of the depth-summed u of time n; the
# higher level for u = 0), Stelling & Duinmeijer (2003), Sec. 2 (equation numbers not checked). Faces with an upwind
# depth <= dry_depth_threshold are closed (hf = 0). Boundary faces use the adjacent column.
function _wd_face_thicknesses!(model::Model, wd::WetDryState)
    cfg = model.config
    u = model.state.u
    nx = cfg.nx
    @inbounds for f in 1:(nx + 1)
        if f == 1 || f == nx + 1
            zeta_f = _wd_level(model, f == 1 ? 1 : nx)
        else
            s = zero(eltype(u))
            for k in axes(u, 1)
                s += u[k, f]
            end
            zl, zr = _wd_level(model, f - 1), _wd_level(model, f)
            zeta_f = s > 0 ? zl : (s < 0 ? zr : max(zl, zr))
        end
        hf = view(wd.hf, :, f)
        if zeta_f - wd.bed_face[f] > cfg.dry_depth_threshold
            layer_thicknesses!(hf, zeta_f, wd.bed_face[f], model.dz)
        else
            fill!(hf, 0.0)
        end
    end
    return nothing
end

# Baroclinic pressure p' = g * integral_z^zeta (rho - rho0) dz' (Pa) at the nominal layer centres z_c = z_k + dz/2. Layers
# above the surface extrapolate with the density of the surface layer.
# WHY nominal centres (not the centroids of cut cells): a horizontally uniform stratification at rest then gives
# bit-identical p' in neighbouring columns whatever their bed, i.e. no spurious baroclinic force (lake at rest).
function _wd_baroclinic_pressure!(model::Model, wd::WetDryState)
    cfg = model.config
    rho = model.state.rho
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

# Upwind value at a cell centre or interface for the momentum advection: first order u_up, or (second) the MUSCL value
# u_up + s/2 with the MC slope s = MC(u_up - u_upup, u_down - u_up) of the upwind node (van Leer 1977), first order where
# the node behind it is not available (`has_upup` false: boundary or dry face). The MC slope satisfies |s/2| <=
# |u_down - u_up| with the same sign, so the value lies between u_up and u_down.
@inline function _wd_upwind_value(u_up::Real, u_upup::Real, u_down::Real, has_upup::Bool, second::Bool)
    (second && has_upup) || return u_up
    return u_up + 0.5 * _monotonized_central_scalar(u_up - u_upup, u_down - u_up)
end

# Advection tendency du/dt of face f, layer k: upwind, momentum conservative (Stelling & Duinmeijer 2003, Sec. 3,
# equation numbers not checked), horizontally with the cell discharges qbar and vertically with the interface fluxes W
# of time n:
#   du/dt = -(1/hbar) [ (qbar_R u*_R - qbar_L u*_L)/dx - u (qbar_R - qbar_L)/dx ] + (vertical analogue with W)
#         = [ qbar_L (u*_L - u) - qbar_R (u*_R - u) ] / (dx hbar) + [ W_bot (u*_bot - u) - W_top (u*_top - u) ] / hbar,
# with u* the value upwinded by the sign of each flux: the upwind velocity (upwind1) or its limited MUSCL extrapolation
# (tvd2, `_wd_upwind_value`; second order in space, forward Euler in time like the momentum tvd2 of the other modes).
# On an outflow side the upwind node is face f itself, so u* - u = 0 for upwind1 (the form reduces to a sum over the
# inflow sides, sum_j c_j (u_j - u), c_j >= 0) and u* - u = s_f/2 for tvd2.
# WHY the cap: with C = dt * (sum of the inflow coefficients c_j), the tendency is scaled by 1/C where C > 1; for upwind1
# this keeps the update a convex combination of the neighbours, so thin layers at the wet/dry front (large c_j) stay
# stable (they then take the inflow velocity). For tvd2 the outflow corrections are limited (|s_f/2| bounded by the
# neighbour differences) and are scaled with the same factor.
function _wd_advection_tendency(model::Model, wd::WetDryState, k::Int, f::Int, second::Bool)
    u = model.state.u
    W = model.state.w
    nz = model.config.nz
    nxp1 = model.config.nx + 1
    dx = model.dx
    L, R = f - 1, f
    hbar = 0.5 * (wd.h[k, L] + wd.h[k, R])
    hbar > WD_EMPTY_CELL_THICKNESS || return 0.0
    uf = u[k, f]
    open_h(ff) = 1 <= ff <= nxp1 && wd.hf[k, ff] > 0.0
    open_v(kk) = 1 <= kk <= nz && wd.hf[kk, f] > 0.0
    csum = 0.0   # 1/s, sum of the inflow coefficients
    tend = 0.0   # m/s^2
    @inbounds begin
        # horizontal: cell L (between faces f-1 and f) and cell R (between f and f+1)
        q_left = 0.5 * (wd.q[k, f - 1] + wd.q[k, f])
        if q_left > 0.0       # inflow from cell L: upwind node f-1, behind it f-2
            has = open_h(f - 2) && open_h(f - 1)
            c = q_left / (dx * hbar)
            csum += c
            tend += c * (_wd_upwind_value(u[k, f - 1], has ? u[k, f - 2] : 0.0, uf, has, second) - uf)
        elseif q_left < 0.0 && second   # outflow into cell L: upwind node f, behind it f+1
            has = open_h(f + 1) && open_h(f - 1)
            tend += q_left / (dx * hbar) * (_wd_upwind_value(uf, has ? u[k, f + 1] : 0.0, u[k, f - 1], has, true) - uf)
        end
        q_right = 0.5 * (wd.q[k, f] + wd.q[k, f + 1])
        if q_right < 0.0      # inflow from cell R: upwind node f+1, behind it f+2
            has = open_h(f + 2) && open_h(f + 1)
            c = -q_right / (dx * hbar)
            csum += c
            tend += c * (_wd_upwind_value(u[k, f + 1], has ? u[k, f + 2] : 0.0, uf, has, second) - uf)
        elseif q_right > 0.0 && second  # outflow into cell R: upwind node f, behind it f-1
            has = open_h(f - 1) && open_h(f + 1)
            tend -= q_right / (dx * hbar) * (_wd_upwind_value(uf, has ? u[k, f - 1] : 0.0, u[k, f + 1], has, true) - uf)
        end
        # vertical: interface k (below layer k) and k+1 (above)
        if open_v(k - 1)
            w_bot = 0.5 * (W[k, L] + W[k, R])
            if w_bot > 0.0    # inflow from below: upwind node k-1, behind it k-2
                has = open_v(k - 2)
                c = w_bot / hbar
                csum += c
                tend += c * (_wd_upwind_value(u[k - 1, f], has ? u[k - 2, f] : 0.0, uf, has, second) - uf)
            elseif w_bot < 0.0 && second   # outflow downwards: upwind node k, behind it k+1
                has = open_v(k + 1)
                tend += w_bot / hbar * (_wd_upwind_value(uf, has ? u[k + 1, f] : 0.0, u[k - 1, f], has, true) - uf)
            end
        end
        if open_v(k + 1)
            w_top = 0.5 * (W[k + 1, L] + W[k + 1, R])
            if w_top < 0.0    # inflow from above: upwind node k+1, behind it k+2
                has = open_v(k + 2)
                c = -w_top / hbar
                csum += c
                tend += c * (_wd_upwind_value(u[k + 1, f], has ? u[k + 2, f] : 0.0, uf, has, second) - uf)
            elseif w_top > 0.0 && second   # outflow upwards: upwind node k, behind it k-1
                has = open_v(k - 1)
                tend -= w_top / hbar * (_wd_upwind_value(uf, has ? u[k - 1, f] : 0.0, u[k + 1, f], has, true) - uf)
            end
        end
    end
    dt = model.config.dt
    return dt * csum > 1.0 ? tend / (dt * csum) : tend
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

# Boundary velocities of the predictor (wet layers only): left prescribed (vector or function of time) or wall; right
# prescribed, zero-gradient (open) or wall.
function _wd_boundary_velocities!(model::Model, wd::WetDryState, u::AbstractMatrix{<:Real}, t::Real)
    nz, nxp1 = size(u)
    left = wd.left_velocity !== nothing ? wd.left_velocity(t) : model.x_boundary_u_left_profile
    @inbounds for k in 1:nz
        wet_left = wd.hf[k, 1] > 0.0
        u[k, 1] = (left === nothing || !wet_left) ? 0.0 : (left isa Real ? left : left[k])
        wet_right = wd.hf[k, nxp1] > 0.0
        if model.x_boundary_u_right_profile !== nothing && wet_right
            u[k, nxp1] = model.x_boundary_u_right_profile[k]
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
# gamma_m = m eps / (1 - m eps), m = 2 nz + 8 (rounding steps of the continuity update) and
# S = depth + |eta^n| + |eta^{n+1}| + |z_b| + O_i (m) the magnitudes that enter it (Higham 2002, Lemma 3.1 and Sec. 4.2).
function _wd_depth_roundoff(model::Model, wd::WetDryState, i::Int)
    m = 2 * model.config.nz + 8
    gamma = m * eps(Float64) / (1 - m * eps(Float64))
    S = model.config.depth + abs(model.ws.eta_old[i]) + abs(model.state.eta[i]) + abs(wd.bed[i]) + wd.outflow[i]
    return abs(wd.rhs[i]) + WD_POSITIVITY_ROUNDOFF_FACTOR * gamma * S
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
(exact arithmetic); WD_NEWTON_RTOL is the round-off stopping test. The velocities are then corrected with the solved
eta (step 4), and their flux divergence reproduces V(eta^{n+1}) >= 0: depth positivity and volume conservation hold
for any dt without changing the solution afterwards.
For a lake at rest F(eta^n) = 0 exactly: no iteration, eta unchanged. The solved eta of a column that falls dry lies at
or below its bed (only V = 0 matters); `_wd_update_level_from_fluxes!` stores it as an exactly dry level.
"""
function _wd_free_surface_solve!(model::Model, wd::WetDryState)
    cfg = model.config
    st = model.state
    ws = model.ws
    nz, nx = cfg.nz, cfg.nx
    dt, dx, g, H = cfg.dt, model.dx, cfg.g, cfg.depth
    coeff = dt * dt * g / (dx * dx)   # 1/m
    a, eta, F, b = wd.a_face, wd.eta_iter, wd.rhs, wd.dp
    @inbounds for f in 1:(nx + 1)
        Hf = 0.0
        U = 0.0
        for k in 1:nz
            Hf += wd.hf[k, f]
            U += wd.hf[k, f] * ws.u_hat[k, f]
        end
        a[f] = (f == 1 || f == nx + 1) ? 0.0 : coeff * Hf
        wd.U_hat[f] = U
    end
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
            Tl = a[i] * (eta[i] - (i > 1 ? eta[i - 1] : 0.0))
            Tr = a[i + 1] * (eta[i] - (i < nx ? eta[i + 1] : 0.0))
            F[i] = V + Tl + Tr - b[i]
            scale = H + V + abs(b[i]) + abs(Tl) + abs(Tr)   # m, magnitudes that enter F_i (round-off scale)
            abs(F[i]) <= WD_NEWTON_RTOL * scale || (converged = false)
        end
        converged && break
        iterations += 1
        iterations <= max_iterations || throw(ErrorException(@sprintf(
            "free-surface Newton iteration did not converge in %d iterations (max |F| = %.3e m)", max_iterations, maximum(abs, F))))
        # Thomas algorithm for (P + T) d = -F: lower(i) = upper(i-1) = -a_i, main(i) = p_i + a_i + a_{i+1}
        @inbounds begin
            for i in 1:nx
                p = H + eta[i] >= wd.bed[i] ? 1.0 : 0.0
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
            st.u[k, f] = wd.hf[k, f] > 0.0 ? ws.u_hat[k, f] - g * dt * grad : 0.0
        end
    end
    @views st.u[:, 1] .= ws.u_hat[:, 1]
    @views st.u[:, nx + 1] .= ws.u_hat[:, nx + 1]
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
    @. wd.q = wd.hf * st.u
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
            st.eta[i] = _wd_dry_eta(cfg.depth, wd.bed[i])
            wd.clamped_volume += (_wd_level(model, i) - wd.bed[i] - depth) * dx
        end
        ws.eta_t[i] = (st.eta[i] - ws.eta_old[i]) / dt
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
WHY q = 0 in the surface cell (not at the surface itself): the surface cell is cut by the moving surface and can be
arbitrarily thin; the simple condition keeps the system SPD for every geometry. It is first order in the surface-cell
thickness (period error of a standing wave, test_validate_wd_nonhydro_standing_wave_full_dispersion_*).
For a lake at rest the right-hand side is exactly zero, so q = 0 and nothing changes.
"""
function _wd_nonhydrostatic_correction!(model::Model, wd::WetDryState)
    cfg = model.config
    st = model.state
    ws = model.ws
    nz, nx = cfg.nz, cfg.nx
    dt, dx, rho0 = cfg.dt, model.dx, cfg.rho0
    h_nh, w_star, idx = wd.h_nh, wd.w_star, wd.nh_index
    wet(k, i) = h_nh[k, i] > WD_EMPTY_CELL_THICKNESS

    # 1. provisional geometry
    @inbounds for i in 1:nx
        zeta = max(cfg.depth + ws.eta_old[i] + wd.delta[i], wd.bed[i])
        layer_thicknesses!(view(h_nh, :, i), zeta, wd.bed[i], model.dz)
    end

    # 2. w predictor at the interfaces k = 2..nz between two wet cells (k-1 below, k above)
    W = st.w
    u0 = wd.u_old
    fill!(w_star, 0.0)
    @inbounds for i in 1:nx, k in 2:nz
        (wet(k - 1, i) && wet(k, i)) || continue
        wk = W[k, i]
        ubar = 0.25 * (u0[k - 1, i] + u0[k - 1, i + 1] + u0[k, i] + u0[k, i + 1])   # m/s, u at the interface
        csum = 0.0
        tend = 0.0
        if ubar > 0.0 && i > 1 && wet(k - 1, i - 1) && wet(k, i - 1)
            c = ubar / dx
            csum += c
            tend += c * (W[k, i - 1] - wk)
        elseif ubar < 0.0 && i < nx && wet(k - 1, i + 1) && wet(k, i + 1)
            c = -ubar / dx
            csum += c
            tend += c * (W[k, i + 1] - wk)
        end
        w_below = 0.5 * (W[k - 1, i] + wk)   # m/s, at the centre of cell k-1
        if w_below > 0.0 && k > 2 && wet(k - 2, i)
            c = w_below / h_nh[k - 1, i]
            csum += c
            tend += c * (W[k - 1, i] - wk)
        end
        w_above = 0.5 * (wk + W[k + 1, i])   # m/s, at the centre of cell k
        if w_above < 0.0 && k < nz && wet(k + 1, i)
            c = -w_above / h_nh[k, i]
            csum += c
            tend += c * (W[k + 1, i] - wk)
        end
        w_star[k, i] = wk + (dt * csum > 1.0 ? tend / csum : dt * tend)
    end

    # 3. pressure problem: unknowns are the wet cells below the surface cell of each column
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
        for k in 1:(s - 1)
            wet(k, i) || continue
            n += 1
            idx[k, i] = n
        end
    end
    fill!(wd.q_nh, 0.0)
    if n == 0
        model.last_nonhydro_pcorr_max = 0.0
        return nothing
    end
    rows, cols, vals, rhs = wd.nh_rows, wd.nh_cols, wd.nh_vals, wd.nh_rhs
    empty!(rows)
    empty!(cols)
    empty!(vals)
    resize!(rhs, n)
    @inbounds for i in 1:nx, k in 1:nz
        P = idx[k, i]
        P == 0 && continue
        diag = 0.0
        div = (h_nh[k, i] - wd.h[k, i]) / dt   # m/s
        for (f, j, sgn) in ((i, i - 1, -1.0), (i + 1, i + 1, 1.0))   # left and right face, neighbour column
            hf = wd.hf[k, f]
            hf > 0.0 || continue
            div += sgn * hf * st.u[k, f] / dx
            (1 < f <= nx) || continue   # boundary faces: prescribed velocity, no correction
            a = hf / dx^2
            diag += a
            N = idx[k, j]
            if N > 0
                push!(rows, P); push!(cols, N); push!(vals, -a)
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
        end
        # interface k+1 (above): always between wet cells, since k lies below the surface cell
        a = 1.0 / (0.5 * (h_nh[k, i] + h_nh[k + 1, i]))
        div += w_star[k + 1, i]
        diag += a
        N = idx[k + 1, i]
        if N > 0
            push!(rows, P); push!(cols, N); push!(vals, -a)
        end
        push!(rows, P); push!(cols, P); push!(vals, diag)
        rhs[P] = -rho0 / dt * div
    end
    # PERF: the wet domain changes every step, so the matrix is assembled and factorized every step (sparse, O(n) work
    # and memory); a lake at rest has rhs = 0 and gives q = 0 exactly.
    A = sparse(rows, cols, vals, n, n)
    sol = cholesky(Symmetric(A)) \ rhs
    @inbounds for i in 1:nx, k in 1:nz
        idx[k, i] > 0 && (wd.q_nh[k, i] = sol[idx[k, i]])
    end

    # 4. the non-hydrostatic pressure gradient enters the predictor; the caller repeats the free-surface solve
    @inbounds for f in 2:nx, k in 1:nz
        wd.hf[k, f] > 0.0 || continue
        dq = wd.q_nh[k, f] - wd.q_nh[k, f - 1]
        dq == 0.0 || (ws.u_hat[k, f] -= dt / rho0 * dq / dx)
    end
    model.last_nonhydro_pcorr_max = maximum(abs, wd.q_nh)
    return nothing
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
    _wd_cell_thicknesses!(wd.h, model, wd)
    _wd_face_thicknesses!(model, wd)
    @. wd.q = wd.hf * st.u   # discharges of time n, for the advection
    _wd_baroclinic_pressure!(model, wd)

    # 2. momentum predictor
    second_order_momentum = cfg.momentum_advection_scheme == "tvd2"
    damp = cfg.rayleigh_damping > 0.0 ? max(0.0, 1.0 - cfg.rayleigh_damping * dt) : 1.0
    @inbounds for f in 2:nx
        for k in 1:nz
            if wd.hf[k, f] > 0.0
                tend = _wd_advection_tendency(model, wd, k, f, second_order_momentum) - (wd.p_anom[k, f] - wd.p_anom[k, f - 1]) / (cfg.rho0 * dx) + cfg.body_force_x
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
        _wd_vertical_viscosity!(model, wd, ws.u_hat, f)
    end
    _wd_boundary_velocities!(model, wd, ws.u_hat, wd.time + dt)

    # 3.-4. free-surface solve with exact volume positivity, velocity correction
    _wd_free_surface_solve!(model, wd)

    # 4b. non-hydrostatic: pressure q from the projection of the free-surface velocities, its gradient added to the
    # predictor, and the free-surface solve repeated, so that the final velocities come from the positive solve (a
    # single layer has no vertical structure: hydrostatic)
    if !cfg.use_hydrostatic && nz > 1
        _wd_nonhydrostatic_correction!(model, wd)
        _wd_free_surface_solve!(model, wd)
    end

    # 5. new water level from the flux divergence (equals V(eta^{n+1}) of the solve)
    _wd_update_level_from_fluxes!(model, wd)

    # 6. vertical fluxes through the fixed interfaces (m/s), integrated upwards from the bed
    @inbounds for i in 1:nx
        layer_thicknesses!(view(wd.h_new, :, i), _wd_level(model, i), wd.bed[i], model.dz)
    end
    max_mismatch = 0.0
    @inbounds for i in 1:nx
        st.w[1, i] = 0.0
        for k in 1:nz
            st.w[k + 1, i] = st.w[k, i] - ((wd.h_new[k, i] - wd.h[k, i]) / dt + (wd.q[k, i + 1] - wd.q[k, i]) / dx)
        end
        max_mismatch = max(max_mismatch, abs(st.w[nz + 1, i]))
        st.w[nz + 1, i] = 0.0
    end
    model.last_kinematic_mismatch = max_mismatch

    # 7. constituent transport
    _wd_transport_density!(model, wd)
    wd.time += dt

    # hydrostatic pressure (diagnostic): p = rho0 g (zeta - z_c) + p' in the wet cells
    copyto!(wd.h, wd.h_new)
    _wd_baroclinic_pressure!(model, wd)
    @inbounds for i in 1:nx, k in 1:nz
        st.p[k, i] = wd.h[k, i] > 0.0 ? cfg.rho0 * g * (_wd_level(model, i) - (k - 0.5) * model.dz) + wd.p_anom[k, i] : 0.0
    end
    copyto!(model.last_p_hydro, st.p)
    cfg.use_hydrostatic && (model.last_nonhydro_pcorr_max = 0.0)
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
    cfg = model.config
    st = model.state
    rho, q, W = st.rho, wd.q, st.w
    nz, nx = cfg.nz, cfg.nx
    dt, dx = cfg.dt, model.dx
    M = wd.mass   # the increment D
    @inbounds for i in 1:nx, k in 1:nz
        M[k, i] = (wd.h[k, i] - wd.h_new[k, i]) * rho[k, i]
    end
    second = cfg.transport_advection_scheme == "tvd2"
    wet(k, i) = wd.h[k, i] > WD_EMPTY_CELL_THICKNESS
    @inbounds for f in 1:(nx + 1), k in 1:nz
        qf = q[k, f]
        qf == 0.0 && continue
        up = qf > 0.0 ? (f > 1 ? f - 1 : 1) : (f <= nx ? f : nx)
        down = qf > 0.0 ? f : f - 1               # only used when has_upup (interior faces)
        upup = qf > 0.0 ? f - 2 : f + 1
        has = 1 <= upup <= nx && 1 <= down <= nx && 1 < f <= nx && wet(k, upup) && wet(k, up) && wet(k, down)
        courant = has ? abs(qf) * dt / (dx * wd.h[k, up]) : 0.0
        rho_f = _wd_face_density(rho[k, up], has ? rho[k, upup] : 0.0, has ? rho[k, down] : 0.0, courant, has, second)
        flux = dt / dx * qf * rho_f
        f > 1 && (M[k, f - 1] -= flux)
        f <= nx && (M[k, f] += flux)
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

    rho_lo, rho_hi = model.rho_min_init, model.rho_max_init
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
    return nothing
end
