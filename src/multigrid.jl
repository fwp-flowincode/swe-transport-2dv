# Column-line multigrid for the non-hydrostatic pressure of the wetting-drying step (issue #71): conjugate gradients with a
# V-cycle preconditioner built for the z-level grid. Line relaxation along the strongly coupled direction and
# semi-coarsening along the weakly coupled one (Trottenberg, Oosterlee & Schueller 2001, Multigrid, Academic Press,
# ch. 5, anisotropic operators; Dendy 1982, J. Comput. Phys. 48, 366-386): the vertical coupling of the cells (1/dz) is
# much stronger than the horizontal one (h/dx^2), so
# - the smoother solves each column exactly (tridiagonal) in a block Gauss-Seidel sweep over the columns;
# - the coarse levels merge neighbouring columns in pairs and keep the layers (x only), down to a single column;
# - the coarse operators are Galerkin products with piecewise-constant prolongation, the horizontal part halved: merging
#   two columns doubles the spacing, and the Galerkin product keeps the coupling of the fine face (hf/dx^2) where the
#   doubled spacing has half of it (measured: 13 iterations on 8.6e3 to 7e5 unknowns with the factor 1/2, 20 to 54
#   iterations with the plain Galerkin product).
# Forward sweep before and backward sweep after the coarse-grid correction make the V-cycle symmetric, so it preconditions
# CG. The operator is the 5-point stencil of the pressure system on the (nz x columns) grid of each level; inactive cells
# (surface cell, dry and empty cells) have the diagonal 1 and no couplings, so with a zero right-hand side they keep 0.

const MG_HORIZONTAL_FACTOR = 0.5   # -, horizontal part of a coarse operator relative to the Galerkin product (see above)
const NH_MULTIGRID_MIN_UNKNOWNS = 5000   # -, nonhydro_solver = "auto" uses the multigrid from this many unknowns on
const NH_MULTIGRID_MAX_ITERATIONS = 200  # -, default of ModelConfig.nonhydro_solver_max_iterations (normally 7-17 are needed)
const NH_MULTIGRID_TOLERANCE_MAX = 1e-2  # -, largest accepted nonhydro_solver_tolerance

mutable struct MultigridLevel
    active::BitMatrix          # nz x ncol: cell is an unknown
    dv::Matrix{Float64}        # 1/m, vertical part of the diagonal
    dh::Matrix{Float64}        # 1/m, horizontal part of the diagonal
    cz::Matrix{Float64}        # 1/m, coupling of (k, c) and (k+1, c) (the matrix entry is -cz)
    cx::Matrix{Float64}        # 1/m, coupling of (k, c) and (k, c+1)
    d::Matrix{Float64}         # 1/m, diagonal (1 in inactive cells)
    ipiv::Matrix{Float64}      # m, reciprocal pivots of the column tridiagonal blocks
    mult::Matrix{Float64}      # -, elimination multipliers cz[k-1] / pivot[k-1]
    klo::Vector{Int}           # lowest and highest active layer of each column (klo > khi: none)
    khi::Vector{Int}
    x::Matrix{Float64}         # work: correction, right-hand side, residual
    b::Matrix{Float64}
    r::Matrix{Float64}
end

function MultigridLevel(nz::Int, ncol::Int)
    z() = zeros(nz, ncol)
    return MultigridLevel(falses(nz, ncol), z(), z(), z(), z(), ones(nz, ncol), ones(nz, ncol), z(), fill(nz + 1, ncol),
        zeros(Int, ncol), z(), z(), z())
end

"""
    ColumnMultigrid(nz, nx)

Workspace of the column-line multigrid for an nz x nx grid: the levels nx, ceil(nx/2), ..., 1 columns and the CG vectors.
Allocated once; `_mg_setup!` refills it every step (the coefficients and the active cells change).
"""
struct ColumnMultigrid
    levels::Vector{MultigridLevel}
    x::Matrix{Float64}         # solution (sol of the pressure system, Pa * theta^2)
    rhs::Matrix{Float64}
    r::Matrix{Float64}
    z::Matrix{Float64}
    p::Matrix{Float64}
    Ap::Matrix{Float64}
end

function ColumnMultigrid(nz::Int, nx::Int)
    levels = [MultigridLevel(nz, nx)]
    ncol = nx
    while ncol > 1
        ncol = (ncol + 1) ÷ 2
        push!(levels, MultigridLevel(nz, ncol))
    end
    z() = zeros(nz, nx)
    return ColumnMultigrid(levels, z(), z(), z(), z(), z(), z())
end

# Diagonal, pivots and the active layer range of a level from dv, dh, cz.
function _mg_factor!(L::MultigridLevel)
    nz, ncol = size(L.active)
    @inbounds for c in 1:ncol
        lo, hi = nz + 1, 0
        for k in 1:nz
            if L.active[k, c]
                L.d[k, c] = L.dv[k, c] + L.dh[k, c]
                lo = min(lo, k)
                hi = k
            else
                L.d[k, c] = 1.0
            end
        end
        L.klo[c], L.khi[c] = lo, hi
        for k in lo:hi
            if k > lo
                L.mult[k, c] = L.cz[k - 1, c] * L.ipiv[k - 1, c]
                L.ipiv[k, c] = 1 / (L.d[k, c] - L.cz[k - 1, c] * L.mult[k, c])
            else
                L.mult[k, c] = 0.0
                L.ipiv[k, c] = 1 / L.d[k, c]
            end
        end
    end
    return L
end

# Coarse level C from the fine level F: columns 2C-1 and 2C merged, Galerkin with piecewise-constant prolongation, the
# horizontal part times MG_HORIZONTAL_FACTOR.
function _mg_coarsen!(C::MultigridLevel, F::MultigridLevel)
    nz, nf = size(F.active)
    nc = size(C.active, 2)
    @inbounds for J in 1:nc
        c1, c2 = 2J - 1, min(2J, nf)
        for k in 1:nz
            act, dv, dh, cz = false, 0.0, 0.0, 0.0
            for c in c1:c2   # one column for the last coarse column of an odd count
                F.active[k, c] || continue
                act = true
                dv += F.dv[k, c]
                dh += F.dh[k, c]
                cz += F.cz[k, c]
            end
            # the face inside the pair: its coupling cancels in the Galerkin product (cx = 0 unless both cells active)
            c1 != c2 && (dh -= 2 * F.cx[k, c1])
            C.active[k, J] = act
            C.dv[k, J] = dv
            C.dh[k, J] = MG_HORIZONTAL_FACTOR * dh
            C.cz[k, J] = cz
            C.cx[k, J] = J < nc ? MG_HORIZONTAL_FACTOR * F.cx[k, c2] : 0.0
        end
    end
    return _mg_factor!(C)
end

# All levels from the fine level (whose active, dv, dh, cz, cx are set).
function _mg_setup!(mg::ColumnMultigrid)
    _mg_factor!(mg.levels[1])
    for i in 2:length(mg.levels)
        _mg_coarsen!(mg.levels[i], mg.levels[i - 1])
    end
    return mg
end

# y = A x on the grid of level L (x zero in inactive cells)
function _mg_apply!(y::Matrix{Float64}, L::MultigridLevel, x::Matrix{Float64})
    nz, ncol = size(x)
    @inbounds for c in 1:ncol, k in 1:nz
        s = L.d[k, c] * x[k, c]
        k > 1 && (s -= L.cz[k - 1, c] * x[k - 1, c])
        k < nz && (s -= L.cz[k, c] * x[k + 1, c])
        c > 1 && (s -= L.cx[k, c - 1] * x[k, c - 1])
        c < ncol && (s -= L.cx[k, c] * x[k, c + 1])
        y[k, c] = s
    end
    return y
end

# One block Gauss-Seidel sweep over the columns (forward or backward), each column solved exactly (Thomas algorithm with
# the stored pivots).
function _mg_sweep!(L::MultigridLevel, forward::Bool)
    x, b, r = L.x, L.b, L.r
    ncol = size(x, 2)
    @inbounds for j in 1:ncol
        c = forward ? j : ncol + 1 - j
        lo, hi = L.klo[c], L.khi[c]
        lo > hi && continue
        left, right = c > 1, c < ncol
        prev = 0.0
        for k in lo:hi
            s = b[k, c]
            left && (s += L.cx[k, c - 1] * x[k, c - 1])
            right && (s += L.cx[k, c] * x[k, c + 1])
            prev = s + L.mult[k, c] * prev
            r[k, c] = prev
        end
        next = r[hi, c] * L.ipiv[hi, c]
        x[hi, c] = next
        for k in (hi - 1):-1:lo
            next = (r[k, c] + L.cz[k, c] * next) * L.ipiv[k, c]
            x[k, c] = next
        end
    end
    return L
end

# Symmetric V-cycle on level i: x = M^-1 b (b set by the caller)
function _mg_vcycle!(mg::ColumnMultigrid, i::Int)
    L = mg.levels[i]
    fill!(L.x, 0.0)
    _mg_sweep!(L, true)
    i == length(mg.levels) && return L
    _mg_apply!(L.r, L, L.x)
    @. L.r = L.b - L.r
    C = mg.levels[i + 1]
    fill!(C.b, 0.0)
    nz, ncol = size(L.x)
    @inbounds for c in 1:ncol, k in 1:nz
        L.active[k, c] && (C.b[k, (c + 1) ÷ 2] += L.r[k, c])
    end
    _mg_vcycle!(mg, i + 1)
    @inbounds for c in 1:ncol, k in 1:nz
        L.active[k, c] && (L.x[k, c] += C.x[k, (c + 1) ÷ 2])
    end
    _mg_sweep!(L, false)
    return L
end

"""
    _mg_pcg!(mg; rtol, itmax) -> (iterations, converged)

Preconditioned conjugate gradients for A x = rhs on the fine grid of `mg` (mg.rhs set, zero in inactive cells), from
x = 0, until |r| <= rtol |rhs| (Euclidean norms of the recursively updated residual; at rtol = 1e-8 it agrees with the
true residual, while far below round-off it keeps falling); the solution in mg.x. A zero right-hand side gives x = 0
exactly (lake at rest).
"""
function _mg_pcg!(mg::ColumnMultigrid; rtol::Real, itmax::Int)
    L = mg.levels[1]
    x, r, z, p, Ap = mg.x, mg.r, mg.z, mg.p, mg.Ap
    fill!(x, 0.0)
    copyto!(r, mg.rhs)
    nb = norm(r)
    nb == 0 && return (0, true)
    copyto!(L.b, r)
    _mg_vcycle!(mg, 1)
    copyto!(z, L.x)
    copyto!(p, z)
    rz = dot(r, z)
    for it in 1:itmax
        _mg_apply!(Ap, L, p)
        alpha = rz / dot(p, Ap)
        @. x += alpha * p
        @. r -= alpha * Ap
        norm(r) <= rtol * nb && return (it, true)
        copyto!(L.b, r)
        _mg_vcycle!(mg, 1)
        copyto!(z, L.x)
        rz_new = dot(r, z)
        @. p = z + (rz_new / rz) * p
        rz = rz_new
    end
    return (itmax, false)
end
