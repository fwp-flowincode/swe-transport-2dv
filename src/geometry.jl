const BED_LEVEL_ALIGNMENT_TOL = 1e-9   # -, allowed deviation of bed_level/dz from a whole number of cells
const BED_MIN_WET_CELLS = 2            # -, wet cells required above the bed in every column

"""
    BedGeometry

Staircase bed on the z-level grid (bed level per column, issue #3). All masks use the grid indexing of the state
(k = 1 bottom, k = nz top; u-faces i = 1..nx+1; w-faces k = 1..nz+1).

Fields:
- `has_bed`: a bed level was given (`ModelConfig.bed_level !== nothing`); the masked code paths are used.
- `any_solid`: at least one cell lies below the bed (false for a zero bed).
- `bed_level` (m, length nx): bed height above the grid bottom, a whole number of cells.
- `k_bed` (length nx): index of the lowest wet cell of each column.
- `bed_face_level` (m, length nx+1): bed height at the u-faces, the higher of the two neighbouring columns (outer
  faces: the adjacent column).
- `k_bed_face` (length nx+1): lowest layer in which a u-face is open.
- `wet_cell` (nz x nx): cell lies above the bed.
- `u_open` (nz x (nx+1)): u-face between two wet cells (outer faces: adjacent cell wet); u may be non-zero.
- `u_active` (nz x (nx+1)): u-face next to at least one wet cell; walls (u = 0) take part in the advection stencils
  like the outer domain faces do.
- `w_open` ((nz+1) x nx): w-face above the bed of its column (incl. the free surface); w may be non-zero.
- `w_active` ((nz+1) x nx): `w_open` plus the bed face itself (w = 0).
"""
struct BedGeometry
    has_bed::Bool
    any_solid::Bool
    bed_level::Vector{Float64}
    k_bed::Vector{Int}
    bed_face_level::Vector{Float64}
    k_bed_face::Vector{Int}
    wet_cell::BitMatrix
    u_open::BitMatrix
    u_active::BitMatrix
    w_open::BitMatrix
    w_active::BitMatrix
end

"""
    BedGeometry(bed_level, nz, nx, dz; periodic_x=false)

Masks of a staircase bed. `bed_level` (m, length nx, or `nothing` for the flat bed) must be a whole number of cells
`dz` (m) high and leave at least BED_MIN_WET_CELLS wet cells in every column (the bed stays below the top layer, so the
variable top layer is unaffected; dry columns are not supported). `periodic_x=true` is not supported with a bed.
"""
function BedGeometry(bed_level::Union{Nothing,AbstractVector{<:Real}}, nz::Int, nx::Int, dz::Real; periodic_x::Bool=false)
    has_bed = bed_level !== nothing
    levels = has_bed ? Float64.(collect(bed_level)) : zeros(Float64, nx)
    if has_bed
        if length(levels) != nx
            throw(ArgumentError("bed_level must have nx = $(nx) entries, got $(length(levels))"))
        end
        if periodic_x
            throw(ArgumentError("bed_level is not supported with periodic_x=true"))
        end
    end

    k_bed = ones(Int, nx)
    @inbounds for i in 1:nx
        cells = levels[i] / dz
        nb = round(Int, cells)
        if levels[i] < 0.0 || abs(cells - nb) > BED_LEVEL_ALIGNMENT_TOL
            throw(ArgumentError(@sprintf("bed_level[%d] = %.6g m must be >= 0 and a whole number of cells (dz = %.6g m)", i, levels[i], dz)))
        end
        if nz - nb < BED_MIN_WET_CELLS
            throw(ArgumentError(@sprintf("bed_level[%d] = %.6g m leaves fewer than %d wet cells (depth = %d cells)", i, levels[i], BED_MIN_WET_CELLS, nz)))
        end
        k_bed[i] = nb + 1
    end

    bed_face_level = zeros(Float64, nx + 1)
    k_bed_face = ones(Int, nx + 1)
    bed_face_level[1], k_bed_face[1] = levels[1], k_bed[1]
    bed_face_level[nx + 1], k_bed_face[nx + 1] = levels[nx], k_bed[nx]
    @inbounds for i in 2:nx
        bed_face_level[i] = max(levels[i - 1], levels[i])
        k_bed_face[i] = max(k_bed[i - 1], k_bed[i])
    end

    wet_cell = falses(nz, nx)
    @inbounds for i in 1:nx, k in k_bed[i]:nz
        wet_cell[k, i] = true
    end

    u_open = falses(nz, nx + 1)
    u_active = falses(nz, nx + 1)
    @inbounds for i in 1:(nx + 1), k in 1:nz
        left = i >= 2 && wet_cell[k, i - 1]
        right = i <= nx && wet_cell[k, i]
        outer = i == 1 || i == nx + 1
        u_open[k, i] = outer ? (left || right) : (left && right)
        u_active[k, i] = left || right
    end

    w_open = falses(nz + 1, nx)
    w_active = falses(nz + 1, nx)
    @inbounds for i in 1:nx, k in k_bed[i]:(nz + 1)
        w_active[k, i] = true
        w_open[k, i] = k > k_bed[i]
    end

    return BedGeometry(has_bed, any(>(1), k_bed), levels, k_bed, bed_face_level, k_bed_face, wet_cell, u_open, u_active, w_open, w_active)
end
