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
# Work arrays of the remapped momentum advection (issue #80, `_wd_remapped_tendencies!`): cumulative thickness and
# discharge of the open layers of the five faces f-2..f+2 (from the face bed upwards), their depths and layer counts, the
# tendencies of the layers of face f, and per face the vertical sigma-layer fluxes (interface below layer k) and the
# advection thicknesses that the implicit vertical solve uses on remapped faces.
struct RemapWork{T<:AbstractFloat}
    zc::Matrix{T}         # m, (nz+1) x 5
    qc::Matrix{T}         # m^2/s, (nz+1) x 5
    depth::Vector{T}      # m, 5
    count::Vector{Int}    # 5
    tend::Vector{T}       # m/s^2, nz
    omega::Matrix{T}      # m/s, nz x (nx+1): sigma-layer flux through the interface below layer k
    hadv::Matrix{T}       # m, nz x (nx+1): advection thickness phi_k Hbar_f of layer k
    active::Vector{Bool}  # nx+1: the face was advected with the remapped form in this step
end

RemapWork(nz::Int, nx::Int; T::Type{<:AbstractFloat}=Float64) = RemapWork{T}(zeros(T, nz + 1, 5), zeros(T, nz + 1, 5), zeros(T, 5),
    zeros(Int, 5), zeros(T, nz), zeros(T, nz, nx + 1), zeros(T, nz, nx + 1), fill(false, nx + 1))

"""
    OpenFaceState

Free (radiating) boundary face of the wetting-drying step whose state is not held by `Model` (the left face, issue #86;
the right face keeps its fields in `Model` for compatibility):
- `eta_b` (m): level of the face at time n (radiation condition, `_update_left_free_level!`);
- `eta_eff` (m), `weight` (-): its Dirichlet form for the free-surface solve of the step;
- `eta_target` (m), `relaxation_time` (s; Inf: none): relaxation of the level (relaxed free face);
- `u_mean` (m/s): depth-mean face velocity filtered over `radiation_period` (NaN until the first step).
"""
mutable struct OpenFaceState
    eta_b::Float64
    eta_eff::Float64
    weight::Float64
    eta_target::Float64
    relaxation_time::Float64
    u_mean::Float64
end

"""
    CharacteristicFaceState

Per-step data of an open face with `open_face_scheme="characteristic"` (issue #86, docs/numerics_open_boundaries.md
section 8), right or left:
- `active`: subcritical this step (false: not open, or supercritical outflow with the extrapolated level);
- `c_out`, `c_in` (m/s): absolute speeds of the outgoing and the incoming wave (long-wave or design-period phase speeds);
- `beta` (-): the weight of the natural condition this step (limited at a shallow boundary column);
- `d_old` (m): D^n = eta_b^n - eta_N^n, the face level minus the adjacent cell level;
- `phi` (m/s): theta g dt H_f/(dx/2) of the face;
- `level_weight` (-): eta_b^{n+1} = eta_N^{n+1} + level_weight (eta_eff - eta_N^{n+1}) (the flux weight differs for
  beta != 0);
- `target_discharge` (m^2/s, positive in +x): Q_T of the drift correction (relaxed face).
Incoming wave of an open face (`right_incoming_level` / `left_incoming_level`, both schemes; issue #86):
- `incoming`: nothing, or the level eta_in(t) (m) of the incoming wave at the face as a function of the time t (s);
- `e_b_old`, `e_b_new` (m): eta_in at the face at the start and the end of the step;
- `e_n_old`, `e_n_new` (m): the same at the adjacent cell centre, eta_in(t - (dx/2)/c_in) (the incoming wave arrives
  there half a cell later);
- `k_in` (rad/m): wave number of the incoming design wave (0 without `radiation_period`), for its non-hydrostatic
  pressure.
"""
mutable struct CharacteristicFaceState
    active::Bool
    c_out::Float64
    c_in::Float64
    beta::Float64
    d_old::Float64
    phi::Float64
    level_weight::Float64
    target_discharge::Float64
    incoming::Union{Nothing,Function}
    e_b_old::Float64
    e_b_new::Float64
    e_n_old::Float64
    e_n_new::Float64
    k_in::Float64
    u_out::Float64   # m/s, (filtered) current in the direction of the outgoing wave, U_n
    e_g::Float64     # m, eta_in one cell outside the face at time n (ghost node of the inflow advection)
end

CharacteristicFaceState() = CharacteristicFaceState(false, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, nothing, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

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
    remap::RemapWork{T}   # remapped momentum advection (issue #80)
    # Pa, nz: non-hydrostatic pressure of the open right boundary face (nonhydro_open_face = "radiation", issue #86)
    q_right::Vector{T}
    q_right_explicit::Vector{T}   # Pa, nz: work array (q_nx^n, then the explicit part of the theta-weighted radiation)
    # m/s: depth-mean velocity of the open right face, low-pass filtered over radiation_period (the current that
    # Doppler-shifts the radiation speed; NaN until the first step)
    u_right_mean::Float64
    # free (radiating) left boundary face (set_x_inflow_outflow!(...; left_free=true), issue #86): nothing, or its state
    left_open::Union{Nothing,OpenFaceState}
    q_left::Vector{T}            # Pa, nz: non-hydrostatic pressure of the open left face (pressure radiation)
    q_left_explicit::Vector{T}   # Pa, nz: work array (q_1^n, then the explicit part of the radiation)
    # characteristic open faces (open_face_scheme = "characteristic", issue #86)
    char_right::CharacteristicFaceState
    char_left::CharacteristicFaceState
end

