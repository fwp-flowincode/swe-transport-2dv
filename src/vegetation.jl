# Vegetation: drag of rigid and flexible stems on the horizontal flow (issue #8, docs/plan_issue_8_vegetation.md).
#
# Two representations share one stem model:
# - VegetationPatch: a canopy described by averaged quantities (stem height, diameter, stems per m², drag coefficient);
#   one representative stem per u-face column; the force per unit volume is the stem force times the stem density.
# - VegetationStem: an individual stem; in the 2DV model (per metre width) it represents a row of stems with lateral
#   spacing `lateral_spacing` (m).
#
# Stem mechanics (both): inextensible, clamped at the base, free tip, quasi-static large-deflection beam (elastica)
# under the normal drag of the horizontal flow and the net buoyancy (Alben, Shelley & Zhang 2002; Gosselin, de Langre &
# Machado-Almeida 2010; Luhar & Nepf 2011):
#     EI dθ/ds = T(s),   T(s) = sum of the moments of the loads beyond s about the point s,   θ(0) = 0,
# with θ the angle from the vertical (positive downstream), the normal drag per unit length
#     f_n = 1/2 rho0 C_D b u_n |u_n|,   u_n = u cos θ   (horizontal velocity u, normal component u cos θ),
# whose horizontal component per unit vertical extent is 1/2 rho0 C_D b u|u| cos²θ, and the net buoyancy per unit
# length B_l (N/m, upward). Rigid stems (bending stiffness EI = Inf) stay vertical.
#
# Discretisation: n_segments straight segments; segment angles at the segment midpoints, curvature at the nodes
# (θ_1 = κ_0 ds/2, θ_m = θ_{m-1} + κ_{m-1} ds; midpoint rule, second order). Each segment is split at the cell
# boundaries of the z grid; a piece in cell row k uses the velocity of that row (piecewise constant in z; for individual
# stems linear in x between the two neighbouring u-faces). The same pieces load the stem and, with the opposite sign,
# the fluid, so the momentum removed from the fluid equals the stem drag exactly (Newton's third law), and a rigid patch
# gives exactly 1/2 C_D a u|u| per unit volume with a = n b (cell fraction below the stem height at the canopy top).
# The posture is solved with Newton's method (finite-difference Jacobian, backtracking), warm-started from the last
# posture. The vertical drag component is not applied to w (no effect in hydrostatic mode; documented limitation), and
# stem parts above the free surface carry no load.
#
# The drag enters the u predictor explicitly (forward Euler); stable and monotone for dt * K <= 1 with the linearised
# drag rate K (1/s) per u-face, the sum of 1/2 C_D b |u| cos²θ (vertical extent) times the deposition weights of the
# pieces (rigid patch: K = 1/2 C_D a |u|); step_forward! checks this (VEGETATION_EXPLICIT_DRAG_LIMIT).

const VEGETATION_EXPLICIT_DRAG_LIMIT = 1.0     # -, max dt * |a_drag / u| of the explicit drag (monotone decay)
const VEGETATION_POSTURE_TOL = 1e-10           # rad, Newton convergence of the segment angles
const VEGETATION_POSTURE_MAX_ITER = 60         # -, Newton iterations per posture update
const VEGETATION_POSTURE_FD_STEP = 1e-7        # rad, finite-difference step of the Newton Jacobian
const VEGETATION_MAX_ANGLE = 0.5 * pi - 1e-3   # rad, angles are limited to just below horizontal

"""
    VegetationStemConfig(; base_x, height, base_z=0.0, diameter=0.01, drag_coefficient=1.0, bending_stiffness=Inf,
                          net_buoyancy=0.0, lateral_spacing=1.0, n_segments=20)

Individual stem: base position `base_x`, `base_z` (m), stem length `height` (m), frontal width `diameter` b (m),
`drag_coefficient` C_D (-), bending stiffness EI `bending_stiffness` (N m²; Inf = rigid), net buoyancy per unit length
`net_buoyancy` (N/m, upward positive; (rho_water - rho_stem) g A for a stem of cross-section A), lateral spacing of the
row of stems it represents `lateral_spacing` (m; the 2DV model is per metre width), number of beam segments.
"""
Base.@kwdef struct VegetationStemConfig
    base_x::Float64
    height::Float64
    base_z::Float64 = 0.0
    diameter::Float64 = 0.01
    drag_coefficient::Float64 = 1.0
    bending_stiffness::Float64 = Inf
    net_buoyancy::Float64 = 0.0
    lateral_spacing::Float64 = 1.0
    n_segments::Int = 20
end

"""
    VegetationStem(; kwargs...)

Individual stem with the configuration `VegetationStemConfig(; kwargs...)`, initially vertical. Fields: `config`,
`theta` (rad, segment angles from the vertical, positive downstream), `force_x` (N, horizontal drag on the stem at the
last update; the force on the fluid is minus this, spread over the row spacing).
"""
mutable struct VegetationStem
    config::VegetationStemConfig
    theta::Vector{Float64}
    force_x::Float64
end

function VegetationStem(; kwargs...)
    cfg = VegetationStemConfig(; kwargs...)
    cfg.n_segments >= 2 || throw(ArgumentError("n_segments must be at least 2"))
    cfg.height > 0 && cfg.diameter > 0 && cfg.lateral_spacing > 0 || throw(ArgumentError("height, diameter and lateral_spacing must be positive"))
    cfg.bending_stiffness > 0 || throw(ArgumentError("bending_stiffness must be positive (Inf for rigid)"))
    return VegetationStem(cfg, zeros(cfg.n_segments), 0.0)
end

"""
    VegetationPatch(; x_start, x_end, stem_height, stem_diameter, stem_density, drag_coefficient=1.0,
                     bending_stiffness=Inf, net_buoyancy=0.0, n_segments=20)

Canopy patch with averaged quantities over x_start <= x <= x_end (m, applied to the u-faces in this range): stem length
`stem_height` (m; emergent if it reaches the surface), frontal width `stem_diameter` b (m), `stem_density` n
(stems/m²), drag coefficient C_D (-), bending stiffness EI (N m²; Inf = rigid) and net buoyancy (N/m) of a stem. The
frontal area per unit volume is a = n b (1/m). Solid volume fraction and blockage are neglected (sparse canopies,
n pi b²/4 << 1).
"""
Base.@kwdef struct VegetationPatch
    x_start::Float64
    x_end::Float64
    stem_height::Float64
    stem_diameter::Float64
    stem_density::Float64
    drag_coefficient::Float64 = 1.0
    bending_stiffness::Float64 = Inf
    net_buoyancy::Float64 = 0.0
    n_segments::Int = 20
end

"""
    patch_frontal_area(patch) -> Float64

Frontal area per unit volume a = n b (1/m) of `patch`.
"""
patch_frontal_area(patch::VegetationPatch) = patch.stem_density * patch.stem_diameter

"""
    VegetationModel(stems::VegetationStem...; patches=VegetationPatch[])

Vegetation attached to a model: individual `stems` and canopy `patches`. `patch_theta[p]` holds the segment angles of
the representative stem of patch p at every u-face (n_segments x (nx+1); sized at the first update); `drag_rate` the
linearised drag rate per u-face (1/s) of the last update.
"""
mutable struct VegetationModel
    stems::Vector{VegetationStem}
    patches::Vector{VegetationPatch}
    patch_theta::Vector{Matrix{Float64}}
    drag_rate::Matrix{Float64}      # 1/s, linearised drag rate per u-face at the last update (stability check)
end

function VegetationModel(stems::VegetationStem...; patches::AbstractVector{VegetationPatch}=VegetationPatch[])
    for p in patches
        p.x_end > p.x_start || throw(ArgumentError("patch x_end must exceed x_start"))
        p.stem_height > 0 && p.stem_diameter > 0 && p.stem_density > 0 || throw(ArgumentError("patch stem height, diameter and density must be positive"))
        p.n_segments >= 2 || throw(ArgumentError("patch n_segments must be at least 2"))
    end
    return VegetationModel(collect(VegetationStem, stems), collect(VegetationPatch, patches), [zeros(p.n_segments, 0) for p in patches],
        zeros(0, 0))
end

# ---------------------------------------------------------------------------------------------------------------------
# Stem mechanics
# ---------------------------------------------------------------------------------------------------------------------

# Parameters of one posture/load evaluation (all SI): stem length, base, frontal width, C_D, EI, net buoyancy, rho0,
# cell height dz, number of cell rows nz, water level (m above the grid bottom).
struct _StemSetup
    length::Float64
    x0::Float64
    z0::Float64
    b::Float64
    cd::Float64
    ei::Float64
    buoyancy::Float64
    rho0::Float64
    dz::Float64
    nz::Int
    water_level::Float64
end

# Load pieces of a stem: segment index, cell row, midpoint (x, z), horizontal and vertical force (N per stem).
mutable struct _StemPieces
    segment::Vector{Int}
    row::Vector{Int}
    x::Vector{Float64}
    z::Vector{Float64}
    fx::Vector{Float64}
    fz::Vector{Float64}
    coef::Vector{Float64}           # N s/m, drag coefficient of the piece: fx = coef * u
    n::Int
end
_StemPieces() = _StemPieces(Int[], Int[], Float64[], Float64[], Float64[], Float64[], Float64[], 0)

function _push_piece!(p::_StemPieces, seg, row, x, z, fx, fz, coef)
    p.n += 1
    if p.n > length(p.segment)
        push!(p.segment, seg); push!(p.row, row); push!(p.x, x); push!(p.z, z); push!(p.fx, fx); push!(p.fz, fz); push!(p.coef, coef)
    else
        p.segment[p.n] = seg; p.row[p.n] = row; p.x[p.n] = x; p.z[p.n] = z; p.fx[p.n] = fx; p.fz[p.n] = fz; p.coef[p.n] = coef
    end
    return nothing
end

# Splits the stem with segment angles `theta` at the cell boundaries and evaluates the loads of every piece below the
# water level with the velocity `velocity(row, x)` (m/s). Returns the node positions (x, z), length n_segments + 1.
function _stem_loads!(pieces::_StemPieces, theta::AbstractVector{<:Real}, s::_StemSetup, velocity)
    nseg = length(theta)
    ds = s.length / nseg
    pieces.n = 0
    xa, za = s.x0, s.z0
    @inbounds for j in 1:nseg
        sn, cs = sincos(theta[j])
        xb, zb = xa + ds * sn, za + ds * cs
        # rows overlapped by this segment (row k spans ((k-1) dz, k dz); the top row reaches the water level)
        k_lo = clamp(floor(Int, za / s.dz) + 1, 1, s.nz)
        k_hi = clamp(floor(Int, zb / s.dz) + 1, 1, s.nz)
        for k in k_lo:k_hi
            top = k == s.nz ? s.water_level : k * s.dz
            lo = max(za, (k - 1) * s.dz)
            hi = min(zb, top, s.water_level)
            hi > lo || continue
            frac = cs > 0 ? ((lo + hi) / 2 - za) / (zb - za) : 0.5
            xm = xa + frac * (xb - xa)
            zm = (lo + hi) / 2
            u = velocity(k, xm)
            dzp = hi - lo                                  # vertical extent of the piece (m)
            coef = 0.5 * s.rho0 * s.cd * s.b * abs(u) * cs * cs * dzp
            fx = coef * u
            fz = -fx * (cs > 0 ? sn / cs : 0.0) + s.buoyancy * (cs > 0 ? dzp / cs : 0.0)
            _push_piece!(pieces, j, k, xm, zm, fx, fz, coef)
        end
        xa, za = xb, zb
    end
    return nothing
end

# Angles implied by the loads of the posture `theta` (fixed-point map G(θ) of the elastica), written to `out`.
function _posture_map!(out::AbstractVector{<:Real}, theta::AbstractVector{<:Real}, s::_StemSetup, velocity, pieces::_StemPieces,
    xn::Vector{Float64}, zn::Vector{Float64})
    nseg = length(theta)
    ds = s.length / nseg
    _stem_loads!(pieces, theta, s, velocity)
    xn[1], zn[1] = s.x0, s.z0
    @inbounds for j in 1:nseg
        sn, cs = sincos(theta[j])
        xn[j + 1] = xn[j] + ds * sn
        zn[j + 1] = zn[j] + ds * cs
    end
    # torque at node i (i = 0 .. nseg-1) from the pieces of segments i+1 .. nseg; positive bends downstream
    angle = 0.0
    @inbounds for i in 0:(nseg - 1)
        torque = 0.0
        for p in 1:pieces.n
            if pieces.segment[p] > i
                torque += (pieces.z[p] - zn[i + 1]) * pieces.fx[p] - (pieces.x[p] - xn[i + 1]) * pieces.fz[p]
            end
        end
        kappa = torque / s.ei
        angle += kappa * (i == 0 ? 0.5 * ds : ds)
        out[i + 1] = clamp(angle, -VEGETATION_MAX_ANGLE, VEGETATION_MAX_ANGLE)
    end
    return out
end

# Newton solve of θ = G(θ) in place (warm start: the incoming theta). Returns true when converged.
function _solve_posture!(theta::AbstractVector{<:Real}, s::_StemSetup, velocity, pieces::_StemPieces)
    nseg = length(theta)
    xn = zeros(nseg + 1); zn = zeros(nseg + 1)          # PERF: small (n_segments+1) buffers per solve, not per piece
    g = zeros(nseg); gp = zeros(nseg); r = zeros(nseg); trial = zeros(nseg); jac = zeros(nseg, nseg)
    residual(th, out) = (_posture_map!(gp, th, s, velocity, pieces, xn, zn); @. out = th - gp; out)
    residual(theta, r)
    norm_r = maximum(abs, r)
    for _ in 1:VEGETATION_POSTURE_MAX_ITER
        norm_r < VEGETATION_POSTURE_TOL && return true
        for c in 1:nseg                                  # finite-difference Jacobian of the residual
            copyto!(trial, theta)
            trial[c] += VEGETATION_POSTURE_FD_STEP
            residual(trial, g)
            @views @. jac[:, c] = (g - r) / VEGETATION_POSTURE_FD_STEP
        end
        step = jac \ r
        lambda = 1.0
        accepted = false
        for _ in 1:20                                    # backtracking on the max-norm of the residual
            @. trial = clamp(theta - lambda * step, -VEGETATION_MAX_ANGLE, VEGETATION_MAX_ANGLE)
            residual(trial, g)
            if maximum(abs, g) < norm_r
                copyto!(theta, trial); copyto!(r, g); norm_r = maximum(abs, g)
                accepted = true
                break
            end
            lambda /= 2
        end
        accepted || return norm_r < sqrt(VEGETATION_POSTURE_TOL)
    end
    return norm_r < VEGETATION_POSTURE_TOL
end

"""
    update_stem_posture!(theta, velocity; length, base_x=0.0, base_z=0.0, diameter, drag_coefficient, bending_stiffness,
                         net_buoyancy=0.0, rho0=1000.0, dz, nz, water_level) -> (converged, force_x, pieces)

Quasi-static posture of a stem (elastica, see the file header) in the horizontal velocity `velocity(row, x)` (m/s; cell
row k spans ((k-1) dz, k dz), `nz` rows, the top row up to `water_level`): updates the segment angles `theta` (rad) in
place and returns whether Newton converged, the horizontal drag on the stem `force_x` (N) and the load pieces (used to
deposit the opposite force on the fluid). Rigid stems (`bending_stiffness = Inf`) keep theta = 0.
"""
function update_stem_posture!(theta::AbstractVector{<:Real}, velocity; length::Real, base_x::Real=0.0, base_z::Real=0.0,
    diameter::Real, drag_coefficient::Real, bending_stiffness::Real, net_buoyancy::Real=0.0, rho0::Real=1000.0,
    dz::Real, nz::Int, water_level::Real, pieces::_StemPieces=_StemPieces())
    s = _StemSetup(length, base_x, base_z, diameter, drag_coefficient, bending_stiffness, net_buoyancy, rho0, dz, nz, water_level)
    converged = true
    if isfinite(bending_stiffness)
        converged = _solve_posture!(theta, s, velocity, pieces)
    else
        fill!(theta, 0.0)
    end
    _stem_loads!(pieces, theta, s, velocity)
    force_x = sum(@view pieces.fx[1:pieces.n]; init=0.0)
    return converged, force_x, pieces
end

"""
    vegetation_path(stem) -> (x, z)

Node positions (m) of `stem` (n_segments + 1 points from the base to the tip).
"""
function vegetation_path(stem::VegetationStem)
    cfg = stem.config
    ds = cfg.height / length(stem.theta)
    x = zeros(length(stem.theta) + 1); z = zeros(length(stem.theta) + 1)
    x[1], z[1] = cfg.base_x, cfg.base_z
    for j in eachindex(stem.theta)
        x[j + 1] = x[j] + ds * sin(stem.theta[j])
        z[j + 1] = z[j] + ds * cos(stem.theta[j])
    end
    return x, z
end

"""
    stem_tip_deflection(stem) -> Float64

Horizontal tip displacement (m) of `stem` relative to its base.
"""
stem_tip_deflection(stem::VegetationStem) = (x = first(vegetation_path(stem)); x[end] - x[1])

"""
    stem_deflected_height(stem) -> Float64

Height of the tip above the base (m) of `stem`.
"""
stem_deflected_height(stem::VegetationStem) = (z = last(vegetation_path(stem)); z[end] - z[1])

copy_vegetation(stem::VegetationStem) = VegetationStem(stem.config, copy(stem.theta), stem.force_x)
copy_vegetation(vegetation::VegetationModel) = VegetationModel([copy_vegetation(s) for s in vegetation.stems],
    copy(vegetation.patches), [copy(t) for t in vegetation.patch_theta], copy(vegetation.drag_rate))

# ---------------------------------------------------------------------------------------------------------------------
# Coupling to the flow
# ---------------------------------------------------------------------------------------------------------------------

# Fraction (0..1) of the control interval (x_face - dx/2, x_face + dx/2) of a u-face that lies inside the patch; with
# periodic_x the patch is also counted shifted by +- the domain length. WHY: weighting the faces by their control length
# makes the vegetated length exactly x_end - x_start (counting every face in the range at full weight overestimated it by
# one face, 5 % for a 4 m patch at dx = 0.2 m).
function _patch_face_weight(patch::VegetationPatch, x_face::Real, dx::Real, domain_length::Real, periodic_x::Bool)
    lo, hi = x_face - dx / 2, x_face + dx / 2
    overlap(a, b) = max(0.0, min(hi, b) - max(lo, a))
    w = overlap(patch.x_start, patch.x_end)
    if periodic_x
        w += overlap(patch.x_start - domain_length, patch.x_end - domain_length) + overlap(patch.x_start + domain_length, patch.x_end + domain_length)
    end
    return min(w / dx, 1.0)
end

"""
    vegetation_source!(accel, vegetation, u, eta, dx, dz, rho0; periodic_x=false) -> accel

Horizontal acceleration (m/s², nz x (nx+1), on the u-faces) of the fluid by the vegetation drag for the face velocities
`u` (m/s) and the surface elevation `eta` (m, per column; water level = nz dz + eta): updates the postures of all
flexible stems and patch columns (quasi-static, warm-started) and deposits minus the stem forces. Individual stems:
force / (rho0 dx dz lateral_spacing), split linearly between the two neighbouring u-faces. Patches: stem density times
the stem force per row / (rho0 dz) at every u-face, weighted by the fraction of the face control interval
(x_face +- dx/2) inside the patch. Returns `accel`; `vegetation.stems[i].force_x`
holds the stem drags.
"""
function vegetation_source!(accel::AbstractMatrix{<:Real}, vegetation::VegetationModel, u::AbstractMatrix{<:Real},
    eta::AbstractVector{<:Real}, dx::Real, dz::Real, rho0::Real; periodic_x::Bool=false)
    fill!(accel, 0.0)
    nz, nxp1 = size(u)
    size(vegetation.drag_rate) == size(u) || (vegetation.drag_rate = zeros(size(u)))
    rate = vegetation.drag_rate
    fill!(rate, 0.0)
    nx = nxp1 - 1
    domain_length = nx * dx
    pieces = _StemPieces()
    column(x) = clamp(floor(Int, x / dx) + 1, 1, nx)     # cell column containing x

    for stem in vegetation.stems
        cfg = stem.config
        water_level = nz * dz + eta[column(cfg.base_x)]
        function face_weights(x)
            xr = periodic_x ? mod(x, domain_length) : clamp(x, 0.0, domain_length)
            pos = xr / dx + 1.0
            i0 = clamp(floor(Int, pos), 1, nx)
            return i0, pos - i0
        end
        velocity(k, x) = (w = face_weights(x); (1.0 - w[2]) * u[k, w[1]] + w[2] * u[k, w[1] + 1])
        _, fx_total, pcs = update_stem_posture!(stem.theta, velocity; length=cfg.height, base_x=cfg.base_x, base_z=cfg.base_z,
            diameter=cfg.diameter, drag_coefficient=cfg.drag_coefficient, bending_stiffness=cfg.bending_stiffness,
            net_buoyancy=cfg.net_buoyancy, rho0=rho0, dz=dz, nz=nz, water_level=water_level, pieces=pieces)
        stem.force_x = fx_total
        scale = 1.0 / (rho0 * dx * dz * cfg.lateral_spacing)
        @inbounds for p in 1:pcs.n
            i0, t = face_weights(pcs.x[p])
            k = pcs.row[p]
            accel[k, i0] -= (1.0 - t) * pcs.fx[p] * scale
            accel[k, i0 + 1] -= t * pcs.fx[p] * scale
            rate[k, i0] += (1.0 - t) * pcs.coef[p] * scale
            rate[k, i0 + 1] += t * pcs.coef[p] * scale
        end
    end

    for (ip, patch) in enumerate(vegetation.patches)
        if size(vegetation.patch_theta[ip]) != (patch.n_segments, nxp1)
            vegetation.patch_theta[ip] = zeros(patch.n_segments, nxp1)
        end
        theta_all = vegetation.patch_theta[ip]
        faces = periodic_x ? (1:nx) : (1:nxp1)
        for i in faces
            x_face = (i - 1) * dx
            weight = _patch_face_weight(patch, x_face, dx, domain_length, periodic_x)
            weight > 0 || continue
            water_level = nz * dz + eta[clamp(i, 1, nx)]
            column_velocity(k, x) = u[k, i]
            theta = @view theta_all[:, i]
            update_stem_posture!(theta, column_velocity; length=patch.stem_height, base_x=x_face, diameter=patch.stem_diameter,
                drag_coefficient=patch.drag_coefficient, bending_stiffness=patch.bending_stiffness, net_buoyancy=patch.net_buoyancy,
                rho0=rho0, dz=dz, nz=nz, water_level=water_level, pieces=pieces)
            scale = weight * patch.stem_density / (rho0 * dz)
            @inbounds for p in 1:pieces.n
                accel[pieces.row[p], i] -= pieces.fx[p] * scale
                rate[pieces.row[p], i] += pieces.coef[p] * scale
            end
        end
    end

    if periodic_x
        # face nx+1 is face 1: fold its deposits onto face 1 and copy back
        @views accel[:, 1] .+= accel[:, end]
        @views accel[:, end] .= accel[:, 1]
        @views rate[:, 1] .+= rate[:, end]
        @views rate[:, end] .= rate[:, 1]
    end
    return accel
end

"""
    vegetation_drag_rate(vegetation) -> Float64

Largest linearised drag rate K (1/s) over the u-faces at the last `vegetation_source!`; the explicit drag is monotone
for dt * K <= VEGETATION_EXPLICIT_DRAG_LIMIT. WHY not |accel / u| per face: an individual stem deposits on both
neighbouring faces with the interpolated velocity, so a face with nearly zero velocity would show an arbitrarily large
ratio although the drag is not stiff.
"""
vegetation_drag_rate(vegetation::VegetationModel) = isempty(vegetation.drag_rate) ? 0.0 : maximum(vegetation.drag_rate)

function attach_vegetation!(model, vegetation::VegetationModel)
    model.vegetation = vegetation
    return model
end

function clear_vegetation!(model)
    model.vegetation = nothing
    return model
end
