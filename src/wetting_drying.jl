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
    theta::Vector{T}
    tri_a::Vector{T}
    tri_b::Vector{T}
    tri_c::Vector{T}
    tri_d::Vector{T}
    time::Float64
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
        zeros(T, nx), ones(T, nx), zeros(T, nz), zeros(T, nz), zeros(T, nz), zeros(T, nz), 0.0, nothing)
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
    @. st.eta = max(levels, wd.bed) - cfg.depth
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
        model.state.eta[i] = max(model.state.eta[i], wd.bed[i] - model.config.depth)
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

# Advection tendency du/dt of face f, layer k: first-order upwind, momentum conservative (Stelling & Duinmeijer 2003,
# Sec. 3, equation numbers not checked), horizontally with the cell discharges qbar and vertically with the interface
# fluxes W of time n. With the layer continuity, (1/hbar)[d(qbar u*)/dx - u d(qbar)/dx] reduces to a sum over the inflow
# sides, sum_j c_j (u_j - u), c_j >= 0.
# WHY the cap: dt * sum_j c_j <= 1 keeps the update a convex combination of the neighbours; in thin layers at the
# wet/dry front c_j can be large (stiff), and the cap makes the step stable there (it then takes the inflow velocity).
function _wd_advection_tendency(model::Model, wd::WetDryState, k::Int, f::Int)
    u = model.state.u
    W = model.state.w
    nz = model.config.nz
    dx = model.dx
    L, R = f - 1, f
    hbar = 0.5 * (wd.h[k, L] + wd.h[k, R])
    hbar > WD_EMPTY_CELL_THICKNESS || return 0.0
    uf = u[k, f]
    csum = 0.0
    tend = 0.0
    q_left = 0.5 * (wd.q[k, f - 1] + wd.q[k, f])
    if q_left > 0.0
        c = q_left / (dx * hbar)
        csum += c
        tend += c * (u[k, f - 1] - uf)
    end
    q_right = 0.5 * (wd.q[k, f] + wd.q[k, f + 1])
    if q_right < 0.0
        c = -q_right / (dx * hbar)
        csum += c
        tend += c * (u[k, f + 1] - uf)
    end
    if k > 1 && wd.hf[k - 1, f] > 0.0
        w_bot = 0.5 * (W[k, L] + W[k, R])
        if w_bot > 0.0
            c = w_bot / hbar
            csum += c
            tend += c * (u[k - 1, f] - uf)
        end
    end
    if k < nz && wd.hf[k + 1, f] > 0.0
        w_top = 0.5 * (W[k + 1, L] + W[k + 1, R])
        if w_top < 0.0
            c = -w_top / hbar
            csum += c
            tend += c * (u[k + 1, f] - uf)
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

"""
    _step_forward_wetting_drying!(model)

One hydrostatic time step with wetting and drying (semi-implicit free surface after Casulli & Cheng 1992 with layer
thicknesses at the old time level, equation numbers not checked; wet/dry faces and momentum-conservative advection after
Stelling & Duinmeijer 2003). Steps:
1. layer thicknesses of cells and faces at time n; baroclinic pressure;
2. predictor u_hat on the wet face layers (advection, baroclinic pressure, body force, damping, implicit viscosity);
3. free-surface solve for the increment d = eta^{n+1} - eta^n (tridiagonal, SPD):
       d_i + a_{i+1}(d_i - d_{i+1}) + a_i(d_i - d_{i-1}) = -dt/dx (Uhat_{i+1} - Uhat_i) + a_{i+1}(eta_{i+1} - eta_i) - a_i(eta_i - eta_{i-1}),
       a_f = dt^2 g H_f / dx^2, H_f = sum_k hf_{k,f} (0 at closed faces and the boundary faces);
   WHY the increment: for a lake at rest the right-hand side is exactly zero, so eta does not change by round-off;
4. u = u_hat - g dt d(eta^{n+1})/dx on the wet face layers;
5. positivity: the outflow of a column is scaled by theta_i = V_i / outflow_i where it exceeds the volume V_i;
   eta^{n+1} is then recomputed from the flux divergence (water volume conserved to round-off);
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

    # 1. geometry at time n
    _wd_cell_thicknesses!(wd.h, model, wd)
    _wd_face_thicknesses!(model, wd)
    @. wd.q = wd.hf * st.u   # discharges of time n, for the advection
    _wd_baroclinic_pressure!(model, wd)
    copyto!(ws.eta_old, st.eta)

    # 2. momentum predictor
    damp = cfg.rayleigh_damping > 0.0 ? max(0.0, 1.0 - cfg.rayleigh_damping * dt) : 1.0
    @inbounds for f in 2:nx
        for k in 1:nz
            if wd.hf[k, f] > 0.0
                tend = _wd_advection_tendency(model, wd, k, f) - (wd.p_anom[k, f] - wd.p_anom[k, f - 1]) / (cfg.rho0 * dx) + cfg.body_force_x
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

    # 3. free-surface increment
    coeff = dt * dt * g / (dx * dx)   # 1/m
    @inbounds for f in 1:(nx + 1)
        H = 0.0
        U = 0.0
        for k in 1:nz
            H += wd.hf[k, f]
            U += wd.hf[k, f] * ws.u_hat[k, f]
        end
        wd.a_face[f] = (f == 1 || f == nx + 1) ? 0.0 : coeff * H
        wd.U_hat[f] = U
    end
    eta = st.eta
    @inbounds for i in 1:nx
        flux_r = i < nx ? wd.a_face[i + 1] * (eta[i + 1] - eta[i]) : 0.0
        flux_l = i > 1 ? wd.a_face[i] * (eta[i] - eta[i - 1]) : 0.0
        wd.rhs[i] = -dt * (wd.U_hat[i + 1] - wd.U_hat[i]) / dx + flux_r - flux_l
    end
    # Thomas algorithm: lower(i) = upper(i-1) = -a_i, main(i) = 1 + a_i + a_{i+1}
    @inbounds begin
        wd.cp[1] = -wd.a_face[2] / (1.0 + wd.a_face[1] + wd.a_face[2])
        wd.dp[1] = wd.rhs[1] / (1.0 + wd.a_face[1] + wd.a_face[2])
        for i in 2:nx
            denom = 1.0 + wd.a_face[i] + wd.a_face[i + 1] + wd.a_face[i] * wd.cp[i - 1]
            wd.cp[i] = -wd.a_face[i + 1] / denom
            wd.dp[i] = (wd.rhs[i] + wd.a_face[i] * wd.dp[i - 1]) / denom
        end
        wd.delta[nx] = wd.dp[nx]
        for i in (nx - 1):-1:1
            wd.delta[i] = wd.dp[i] - wd.cp[i] * wd.delta[i + 1]
        end
    end

    # 4. velocity correction (eta^{n+1} of the solve is ws.eta_old + delta)
    @inbounds for f in 2:nx
        grad = ((ws.eta_old[f] + wd.delta[f]) - (ws.eta_old[f - 1] + wd.delta[f - 1])) / dx
        for k in 1:nz
            st.u[k, f] = wd.hf[k, f] > 0.0 ? ws.u_hat[k, f] - g * dt * grad : 0.0
        end
    end
    @views st.u[:, 1] .= ws.u_hat[:, 1]
    @views st.u[:, nx + 1] .= ws.u_hat[:, nx + 1]

    # 5. positivity of the column volumes and the new water level from the flux divergence
    @. wd.q = wd.hf * st.u
    @inbounds for i in 1:nx
        out = 0.0
        for k in 1:nz
            out += max(wd.q[k, i + 1], 0.0) + max(-wd.q[k, i], 0.0)
        end
        out *= dt / dx
        volume = max(_wd_level(model, i) - wd.bed[i], 0.0)
        wd.theta[i] = out > volume ? volume / out : 1.0
    end
    @inbounds for f in 1:(nx + 1), k in 1:nz
        q = wd.q[k, f]
        factor = q > 0.0 ? (f > 1 ? wd.theta[f - 1] : 1.0) : (q < 0.0 ? (f <= nx ? wd.theta[f] : 1.0) : 1.0)
        if factor < 1.0
            wd.q[k, f] = q * factor
            st.u[k, f] *= factor
        end
    end
    @inbounds for i in 1:nx
        div = 0.0
        for k in 1:nz
            div += wd.q[k, i + 1] - wd.q[k, i]
        end
        st.eta[i] = ws.eta_old[i] - dt * div / dx
        # WHY: a fully drained column can end a round-off below its bed
        st.eta[i] = max(st.eta[i], wd.bed[i] - cfg.depth)
        ws.eta_t[i] = (st.eta[i] - ws.eta_old[i]) / dt
    end

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
    model.last_nonhydro_pcorr_max = 0.0
    return nothing
end

# Conservative first-order upwind transport of rho with the layer discharges q and interface fluxes W of the step:
#   (h rho)^{n+1} = (h rho)^n - dt/dx (F_{f+1} - F_f) - dt (G_{k+1} - G_k),  F = q rho_up, G = W rho_up.
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
    @inbounds for f in 1:(nx + 1), k in 1:nz
        qf = q[k, f]
        qf == 0.0 && continue
        up = qf > 0.0 ? (f > 1 ? f - 1 : 1) : (f <= nx ? f : nx)
        flux = dt / dx * qf * rho[k, up]
        f > 1 && (M[k, f - 1] -= flux)
        f <= nx && (M[k, f] += flux)
    end
    @inbounds for i in 1:nx, k in 2:nz
        wk = W[k, i]
        wk == 0.0 && continue
        flux = dt * wk * (wk > 0.0 ? rho[k - 1, i] : rho[k, i])
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
