mutable struct PressureCorrectionSolver{T<:Real}
    shape::Tuple{Int, Int}
    dx::T
    dz::T
    method::String
    top_dirichlet::Bool
    periodic_x::Bool
    inv_lambda::Union{Matrix{T}, Nothing}
    poisson_matrix::Union{SparseMatrixCSC{T, Int}, Nothing}
    poisson_fact::Any
    rhs::Matrix{T}
    rhs_hat::Matrix{T}
    p_corr::Matrix{T}
    div_tmp::Matrix{T}
    div_z_tmp::Matrix{T}
    rhs_flat::Vector{T}
    p_corr_flat::Vector{T}
    dct_plan::Any
    idct_plan::Any
    timers::TimerOutput
    fftw_threads::Int
end

# `top_half_layer=false` (legacy): p = 0 at the centre of the top cell. `top_half_layer=true`: p at the top-cell centre is
# the weight of the upper half of the top layer, rho_top*g*dz/2 (Pa; rho in kg/m^3, g in m/s^2, dz in m).
# WHY: the legacy reference ignores the density of the upper half of the top layer, so the top row misses the
# baroclinic gradient (g*dz/2)*d(rho_top)/dx. The barotropic free-surface force g*d(eta)/dx is applied separately and
# uniformly, so the (small) eta-dependence of the half layer is not repeated here.
function solve_hydrostatic_pressure!(p_new::AbstractMatrix{<:Real}, rho::AbstractMatrix{<:Real}, g::Real, dz::Real; top_half_layer::Bool=false)
    nz, nx = size(rho)
    fill!(p_new, 0.0)
    if nz < 2
        return p_new
    end

    @inbounds for i in 1:nx
        p_top = top_half_layer ? 0.5 * rho[nz, i] * g * dz : 0.0
        p_new[nz, i] = p_top
        acc = p_top
        for k in (nz - 1):-1:1
            inc = 0.5 * (rho[k + 1, i] + rho[k, i]) * g * dz
            acc += inc
            p_new[k, i] = acc
        end
    end
    return p_new
end

function solve_hydrostatic_pressure(p::AbstractMatrix{<:Real}, rho::AbstractMatrix{<:Real}, g::Real, dz::Real; top_half_layer::Bool=false)
    p_new = similar(p)
    return solve_hydrostatic_pressure!(p_new, rho, g, dz; top_half_layer=top_half_layer)
end

"""
    build_poisson_matrix(shape, dx, dz; top_dirichlet=false)

Sparse 5-point discrete Laplacian for the non-hydrostatic pressure correction `p'` on an `nz x nx` cell-centred grid
(`shape = (nz, nx)`, k=1 bottom, k=nz top; `dx`, `dz` in m), Neumann (zero-flux) at the bottom and side walls in every
case.

`top_dirichlet=false` (legacy): Neumann at the top too, matching the spectral (DCT) solver's eigenbasis exactly (see
`test_pressure_solver_spectral_matches_sparse_reference`). This operator has a 1D null space (the constant mode); row 1
is pinned (`p'[1,1]=0`) for gauge-fixing, and callers must remove the mean of the RHS (solvability condition) and of the
solution, as `solve!` and `solve_nonhydrostatic_pressure` already do.

`top_dirichlet=true`: Dirichlet `p'=0` at the top FACE (physically: the free surface is the reference level for the
barotropic/hydrostatic split -- section 7 -- so the non-hydrostatic pressure it carries is exactly what remains after
that split, which the depth-varying, dispersive part of a linear surface wave requires to vanish there; see docs
section 15). Ghost-cell closure: with the face a half cell above the top cell centre, `p_ghost = -p[nz,i]` (linear
interpolation through `p[nz,i]` and `p_ghost` gives exactly 0 at the face); substituting into the standard stencil adds
`-2/dz^2` to the diagonal of the top row (not `-1/dz^2`: the ghost contributes both the usual "real neighbour exists"
decrement AND its own value, `+1/dz^2 * p_ghost = -1/dz^2 * p[nz,i]`, verified against the manufactured solution below).
A Dirichlet boundary removes the null space, so no row is pinned and callers must NOT remove any mean (`solve!` branches
on `solver.top_dirichlet`).

Verification (`test_verify_poisson_matrix_top_dirichlet_matches_manufactured_solution`): 1D-in-z manufactured solution
`p(z) = C/2*(z^2 - H^2)` solves `p'' = C` on `z in (0,H)` with `p'(0)=0` (Neumann bottom) and `p(H)=0` (Dirichlet top,
at the domain edge, matching the ghost closure above).
"""
function build_poisson_matrix(shape::Tuple{Int, Int}, dx::Real, dz::Real; top_dirichlet::Bool=false, periodic_x::Bool=false)
    nz, nx = shape
    n = nz * nx

    inv_dx2 = 1.0 / (dx * dx)
    inv_dz2 = 1.0 / (dz * dz)

    row = Int[]
    col = Int[]
    val = Float64[]
    sizehint!(row, 5 * n)
    sizehint!(col, 5 * n)
    sizehint!(val, 5 * n)

    idx(k, i) = (k - 1) * nx + i

    for k in 1:nz
        for i in 1:nx
            r = idx(k, i)
            diag = 0.0

            if periodic_x
                push!(row, r); push!(col, idx(k, mod1(i - 1, nx))); push!(val, inv_dx2)
                push!(row, r); push!(col, idx(k, mod1(i + 1, nx))); push!(val, inv_dx2)
                diag -= 2.0 * inv_dx2
            else
                if i > 1
                    push!(row, r); push!(col, idx(k, i - 1)); push!(val, inv_dx2)
                    diag -= inv_dx2
                end
                if i < nx
                    push!(row, r); push!(col, idx(k, i + 1)); push!(val, inv_dx2)
                    diag -= inv_dx2
                end
            end
            if k > 1
                push!(row, r); push!(col, idx(k - 1, i)); push!(val, inv_dz2)
                diag -= inv_dz2
            end
            if k < nz
                push!(row, r); push!(col, idx(k + 1, i)); push!(val, inv_dz2)
                diag -= inv_dz2
            elseif top_dirichlet
                diag -= 2.0 * inv_dz2
            end

            push!(row, r); push!(col, r); push!(val, diag)
        end
    end

    A = sparse(row, col, val, n, n)
    if !top_dirichlet
        A[1, :] .= 0.0
        A[1, 1] = 1.0
    end
    dropzeros!(A)
    return A
end

"""
    PressureCorrectionSolver(shape, dx, dz; method="spectral", top_dirichlet=false, T=Float64)

`top_dirichlet=true` requires `method="sparse"`: the spectral method's DCT-based fast transform represents only a pure
Neumann-Neumann eigenbasis in z and cannot express a Dirichlet top condition (a different, mixed-condition transform
would be needed; not implemented). See `build_poisson_matrix` for the boundary condition and docs section 15 for why a
Dirichlet top condition is needed for dispersive non-hydrostatic surface waves.
"""
function PressureCorrectionSolver(shape::Tuple{Int, Int}, dx::Real, dz::Real; method::String="spectral", top_dirichlet::Bool=false, periodic_x::Bool=false, T::Type{<:Real}=Float64)
    nz, nx = shape

    rhs = zeros(T, nz, nx)
    rhs_hat = zeros(T, nz, nx)
    p_corr = zeros(T, nz, nx)
    div_tmp = zeros(T, nz, nx)
    div_z_tmp = zeros(T, nz, nx)
    rhs_flat = zeros(T, nz * nx)
    p_corr_flat = zeros(T, nz * nx)

    if method == "spectral"
        if periodic_x
            throw(ArgumentError("periodic_x=true requires method=\"sparse\" for the pressure-correction solver"))
        end
        if top_dirichlet
            throw(ArgumentError("top_dirichlet=true requires method=\"sparse\" (the spectral solver cannot represent a Dirichlet top condition)"))
        end
        kx = collect(0:(nx - 1))
        kz = collect(0:(nz - 1))
        lam_x = [2.0 * (cos(pi * x / nx) - 1.0) / (dx * dx) for x in kx]
        lam_z = [2.0 * (cos(pi * z / nz) - 1.0) / (dz * dz) for z in kz]
        inv_lambda = zeros(T, nz, nx)
        @inbounds for i in 1:nx
            for k in 1:nz
                if i == 1 && k == 1
                    inv_lambda[k, i] = 0.0
                else
                    λ = lam_z[k] + lam_x[i]
                    inv_lambda[k, i] = 1.0 / λ
                end
            end
        end
        fftw_threads = max(1, Base.Threads.nthreads())
        FFTW.set_num_threads(fftw_threads)

        dct_plan = FFTW.plan_dct(rhs, (1, 2); flags=FFTW.MEASURE)
        idct_plan = FFTW.plan_idct(rhs_hat, (1, 2); flags=FFTW.MEASURE)

        return PressureCorrectionSolver(shape, T(dx), T(dz), method, top_dirichlet, periodic_x, inv_lambda, nothing, nothing, rhs, rhs_hat, p_corr, div_tmp, div_z_tmp, rhs_flat, p_corr_flat, dct_plan, idct_plan, TimerOutput(), fftw_threads)
    elseif method == "sparse"
        A = build_poisson_matrix(shape, dx, dz; top_dirichlet=top_dirichlet, periodic_x=periodic_x)
        F = lu(A)
        return PressureCorrectionSolver(shape, T(dx), T(dz), method, top_dirichlet, periodic_x, nothing, A, F, rhs, rhs_hat, p_corr, div_tmp, div_z_tmp, rhs_flat, p_corr_flat, nothing, nothing, TimerOutput(), 1)
    else
        throw(ArgumentError("Unsupported pressure solver method '$(method)'"))
    end
end

function solve_nonhydrostatic_pressure(
    u::AbstractMatrix{<:Real},
    w::AbstractMatrix{<:Real},
    p_corr::AbstractMatrix{<:Real},
    rho0::Real,
    dt::Real,
    dx::Real,
    dz::Real,
    poisson_matrix::SparseMatrixCSC{<:Real, Int},
)
    nz, nx = size(p_corr)
    div_u = div_x_u(u, dx) .+ div_z_w(w, dz)

    rhs = (rho0 / dt) .* div_u
    rhs_flat = zeros(eltype(rhs), nz * nx)
    idx = 1
    @inbounds for k in 1:nz
        for i in 1:nx
            rhs_flat[idx] = rhs[k, i]
            idx += 1
        end
    end
    rhs_flat .-= mean(rhs_flat)
    rhs_flat[1] = 0.0

    p_corr_flat = poisson_matrix \ rhs_flat
    p_corr_new = similar(p_corr)
    idx = 1
    @inbounds for k in 1:nz
        for i in 1:nx
            p_corr_new[k, i] = p_corr_flat[idx]
            idx += 1
        end
    end
    p_corr_new .-= mean(p_corr_new)
    return p_corr_new
end

function solve!(solver::PressureCorrectionSolver{T}, u::AbstractMatrix{<:Real}, w::AbstractMatrix{<:Real}, rho0::Real, dt::Real, dx::Real, dz::Real) where {T<:Real}
    nz, nx = solver.shape

    @timeit solver.timers "pressure/divergence_and_rhs" begin
        div_x_u!(solver.div_tmp, u, dx)
        div_z_w!(solver.div_z_tmp, w, dz)
        scale = rho0 / dt
        @inbounds for i in 1:nx
            for k in 1:nz
                solver.rhs[k, i] = scale * (solver.div_tmp[k, i] + solver.div_z_tmp[k, i])
            end
        end
    end

    # WHY: mean-removal / gauge-fixing is the solvability condition and null-space fix for a PURE NEUMANN operator
    # (1D null space, the constant mode). A Dirichlet top condition removes that null space, so the RHS mean must be
    # left as given and the solution's mean is a real, physical, non-arbitrary quantity that must not be zeroed.
    if !solver.top_dirichlet
        @timeit solver.timers "pressure/remove_rhs_mean" begin
            rhs_mean = mean(solver.rhs)
            @inbounds for i in 1:nx
                for k in 1:nz
                    solver.rhs[k, i] -= rhs_mean
                end
            end
        end
    end

    if solver.method == "spectral"
        @timeit solver.timers "pressure/dct_forward" mul!(solver.rhs_hat, solver.dct_plan, solver.rhs)

        @timeit solver.timers "pressure/diagonal_solve" begin
            @inbounds for i in 1:nx
                for k in 1:nz
                    solver.rhs_hat[k, i] *= solver.inv_lambda[k, i]
                end
            end
            solver.rhs_hat[1, 1] = 0.0
        end

        @timeit solver.timers "pressure/idct_inverse" mul!(solver.p_corr, solver.idct_plan, solver.rhs_hat)

        @timeit solver.timers "pressure/remove_pcorr_mean" begin
            p_mean = mean(solver.p_corr)
            @inbounds for i in 1:nx
                for k in 1:nz
                    solver.p_corr[k, i] -= p_mean
                end
            end
        end

        return solver.p_corr
    end

    @timeit solver.timers "pressure/flatten_rhs" begin
        idx = 1
        @inbounds for k in 1:nz
            for i in 1:nx
                solver.rhs_flat[idx] = solver.rhs[k, i]
                idx += 1
            end
        end
    end

    if !solver.top_dirichlet
        solver.rhs_flat[1] = 0.0   # matches the row-1 gauge pin in build_poisson_matrix (Neumann-Neumann only)
    end

    @timeit solver.timers "pressure/linear_solve" ldiv!(solver.p_corr_flat, solver.poisson_fact, solver.rhs_flat)

    @timeit solver.timers "pressure/unflatten_pcorr" begin
        idx = 1
        @inbounds for k in 1:nz
            for i in 1:nx
                solver.p_corr[k, i] = solver.p_corr_flat[idx]
                idx += 1
            end
        end
    end

    if !solver.top_dirichlet
        @timeit solver.timers "pressure/remove_pcorr_mean" begin
            p_mean = mean(solver.p_corr)
            @inbounds for i in 1:nx
                for k in 1:nz
                    solver.p_corr[k, i] -= p_mean
                end
            end
        end
    end

    return solver.p_corr
end
