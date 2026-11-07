# State of the wetting-drying time step (issue #31); the kernels are in wetting_drying.jl.
# WHY a file of its own, included before model.jl: Model holds it as the typed field wetdry::Union{Nothing,WetDryState{T}}
# (issue #10: the untyped field made every value read from it in the kernels outside the function barrier dynamic).

"""
    WetDryState

Bed, layer geometry, fluxes and work arrays of the wetting-drying time step (sizes for nz layers and nx columns).
- `bed` (m, nx): bed level per column; `bed_face` (m, nx+1): bed level at the u-faces (higher neighbour).
- `h`, `h_new` (m, nz x nx): layer thicknesses at the old and new time level.
- `hf` (m, nz x (nx+1)): layer thicknesses at the u-faces (upwind water level, face bed).
- `q` (m^2/s, nz x (nx+1)): layer discharges hf*u of the step.
- `p_anom` (Pa, nz x nx): baroclinic pressure g * integral from z to zeta of (rho - rho0), at the nominal layer centres.
- `time` (s): model time, advanced by every step (used by a time-dependent boundary velocity).
- `gate_edge` (m, nx+1), `gate_contraction` (-, nx+1): sluice gates at the u-faces (`set_sluice_gate!`).
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
    w_nh::Matrix{T}            # m/s, (nz+1) x nx: w^n at the interfaces from the continuity of u^n (theta < 1, issues #34, #38)
    nh_index::Matrix{Int}      # unknown number of a cell of the pressure problem (0: surface, dry or empty cell)
    nh_rows::Vector{Int}
    nh_cols::Vector{Int}
    nh_vals::Vector{T}
    nh_rhs::Vector{T}
    hf_prev::Matrix{T}         # m, nz x (nx+1): face layer thicknesses of the previous step (-1: none yet)
    time::Float64
    clamped_volume::Float64   # m^2 (per unit width): volume added by setting round-off negative depths to zero
    newton_iterations::Int        # Newton iterations of the last free-surface solve
    newton_iterations_max::Int    # largest number of Newton iterations so far
    left_velocity::Any   # nothing, or a function t (s) -> u (m/s, scalar or length nz) at the left boundary face
    gate_edge::Vector{T}         # m, nx+1: lower edge of a sluice gate above the grid bottom (Inf: no gate; issue #48)
    gate_contraction::Vector{T}  # -, nx+1: contraction coefficient of the gate opening (1: none)
    gate_active::Vector{Bool}    # nx+1: the gate cuts the flow area of the face in this step (upstream level above it)
    left_k::Any     # nothing (zero gradient), or k (m^2/s^2, length nz) of the inflow at the left boundary face (#42)
    left_eps::Any   # nothing (zero gradient), or eps (m^2/s^3, length nz) of the inflow at the left boundary face (#42)
    sediment::Any   # nothing, or the SedimentState of suspended sediment (attach_sediment!, issue #64)
    # non-hydrostatic pressure matrix (issue #10): the Cholesky factor of the last step and the triplet pattern it was
    # analysed for; while the pattern is unchanged only the numeric factorization is repeated
    nh_factor::Any                 # nothing, or the CHOLMOD factor
    nh_matrix::SparseMatrixCSC{T,Int}   # the pressure matrix of the last step
    nh_nonzero::Vector{Int}        # position in nh_matrix.nzval of each triplet of the pattern
    nh_pattern_rows::Vector{Int}
    nh_pattern_cols::Vector{Int}
    nh_symbolic_analyses::Int      # symbolic analyses (ordering and elimination tree) so far
    nh_factorizations::Int         # numeric factorizations so far
    # column-line multigrid (issue #71, ModelConfig.nonhydro_solver): workspace, CG iterations of the last solve and the
    # largest so far, multigrid solves and direct fallbacks (CG not converged)
    nh_multigrid::Union{Nothing,ColumnMultigrid}
    nh_iterations::Int
    nh_iterations_max::Int
    nh_multigrid_solves::Int
    nh_fallbacks::Int
end
