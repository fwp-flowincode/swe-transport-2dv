# WHY these bounds: the linear amplification factor of the theta scheme, |G|^2 = (1 + mu^2 (1-theta)^2)/(1 + mu^2 theta^2),
# is <= 1 for every time step only if theta >= 1/2 (docs/plan_issue_34_theta_scheme.md, section 3); theta > 1 only adds
# damping and is not a Casulli theta scheme.
const THETA_MIN = 0.5   # -, smallest allowed implicitness of the free surface
const THETA_MAX = 1.0   # -, largest allowed implicitness of the free surface

"""
    theta_froude_max(theta)

Largest Froude number `Fr = |U| / sqrt(g H)` (-) of a mean flow for which the theta scheme with explicit momentum
advection is linearly stable (docs/plan_issue_34_theta_scheme.md, section 3.3):

    Fr_max(theta) = (2 theta - 1) / (2 (1 - theta)),   equivalently   theta >= (1 + 2 Fr) / (2 (1 + Fr)).

`theta` (-) in [THETA_MIN, THETA_MAX]; `Fr_max(1/2) = 0`, `Fr_max(0.55) = 0.11`, `Fr_max(0.7) = 2/3`, `Fr_max(0.75) = 1`,
`Fr_max(1) = Inf`. Von Neumann analysis of the linearised 1D system about a uniform flow U (upwind or Lax-Wendroff
advection of u and eta with Courant number C <= 1, theta free surface with any gravity-wave Courant number): long
waves, where the explicit advection adds the Doppler phase U k dt without damping, give |lambda|^2 <= 1 iff
2 (1 - theta) Fr <= 2 theta - 1; the bound is sharp for Lax-Wendroff (the smooth limit of tvd2) and sufficient for
upwind1, whose numerical diffusion lowers the required theta. Above Fr_max the growth per step is weak (slowly growing
gravity waves on the mean flow), not an immediate blow-up. The bound concerns the theta coupling only; the explicit
momentum advection is stable up to the advective Courant number 1 (issue #37). In the full model the bound is sharp
with the variable top layer and conservative on the fixed grid (a periodic channel decayed up to 1.4 times the bound;
docs/TESTS_TECHNICAL_DOCUMENTATION.md, section 19). `step_forward!` warns once when the surface Froude number of the run
exceeds it (`surface_froude_number`, issue #39).
"""
theta_froude_max(theta::Real) = theta >= 1 ? Inf : (2 * theta - 1) / (2 * (1 - theta))

"""
    theta_for_froude(Fr)

Smallest implicitness theta (-) for which a mean flow of Froude number `Fr` (-, >= 0) is linearly stable with explicit
momentum advection, the inverse of `theta_froude_max`: theta = (1 + 2 Fr) / (2 (1 + Fr)), from 1/2 at Fr = 0 towards 1
for Fr -> Inf (0.7 for Fr = 2/3, 0.75 for Fr = 1, 0.83 for Fr = 2). Source: docs/plan_issue_34_theta_scheme.md,
section 3.3.
"""
function theta_for_froude(Fr::Real)
    Fr >= 0 || throw(ArgumentError("Fr must be >= 0, got $(Fr)"))
    return isinf(Fr) ? 1.0 : (1 + 2 * Fr) / (2 * (1 + Fr))
end

"""
    baroclinic_extrapolation_weight(theta)

Weight beta (-) of the density extrapolation rho^n + beta (rho^n - rho^{n-1}) in the baroclinic pressure (and the
non-hydrostatic buoyancy) of the momentum predictor (issue #23): beta = 2 (1 - theta), 0 at theta = 1.

WHY: the momentum step takes the density of time n, the transport moves it with u^{n+theta} = theta u^{n+1} +
(1 - theta) u^n (issue #34). For an internal wave of frequency omega, s = omega dt, the oscillator model
(lambda - 1) u = -s r, (lambda - 1) r = s (theta lambda + 1 - theta) u gives |lambda|^2 = 1 + (1 - theta) s^2: weakly
unstable for every theta < 1, fastest for the short hydrostatic internal waves (an internal seiche on 160 x 80 cells
grew at theta = 0.7). With the extrapolated density lambda (lambda - 1)^2 + s^2 ((1 + beta) lambda - beta) (theta lambda
+ 1 - theta) = 0, and the growth term of order s^2 becomes (1 - theta) - beta: beta = 1 - theta leaves a growth of order
s^4, beta = 2 (1 - theta) damps every wave up to s_max = 0.82 (theta = 1/2), 0.86 (0.55), 1.13 (0.7), 1.89 (0.9), i.e. an
internal Courant number c dt / dx up to about s_max / pi. The damping of a resolved wave is about pi (1 - theta) omega dt
per period (first order in dt, like the theta scheme itself). theta = 1 keeps the former scheme, bit for bit.
"""
baroclinic_extrapolation_weight(theta::Real) = 2 * (1 - theta)


Base.@kwdef mutable struct ModelConfig
    nz::Int = 20
    nx::Int = 100
    length::Float64 = 30.0
    depth::Float64 = 4.0
    g::Float64 = 9.81
    rho0::Float64 = 1000.0
    rho_left::Float64 = 1000.722
    rho_right::Float64 = 999.972
    nu::Float64 = 1e-6
    use_turbulence_model::Bool = false
    turbulence_mixing_length::Float64 = 0.0
    max_turbulent_viscosity::Float64 = 5e-4
    # Turbulence closure: "none", "mixing_length" (nu_t = l^2 |du/dz|, also selected by use_turbulence_model) or
    # "k_epsilon" (src/turbulence.jl: k-epsilon with buoyancy and vegetation terms and turbulent density diffusion;
    # fixed grid with a flat bed, and with wetting and drying; the automatic default of variable_top_layer is false with
    # it, an explicit variable_top_layer=true is allowed). Evaluate with turbulence_closure(cfg).
    turbulence_closure::String = "none"
    # Bed roughness length z0 (m): > 0 applies a log-law quadratic bed friction c_b u|u| on the lowest row
    # (c_b = (kappa/ln(z1/z0))^2, z1 = dz/2) and, with k-epsilon, the wall functions; 0 keeps the free-slip bed.
    bed_roughness_length::Float64 = 0.0
    # Chezy coefficient C (m^(1/2)/s; issue #14): > 0 applies the depth-averaged quadratic bed friction
    # tau_b / rho0 = g |U| U / C^2 (U the depth-averaged velocity of the u-face) as a stress on the lowest layer, so the
    # depth-integrated momentum balance has the Chezy law exactly, independently of the vertical viscosity; 0 = off.
    # Excludes bed_roughness_length (use chezy_roughness_length(C, depth) for the log-law equivalent, e.g. with
    # k-epsilon). WHY on the depth average: uniform channel flow then has U = C sqrt(H i_b) exactly; the vertical
    # profile needs a turbulent viscosity of the order kappa u_* H / 6 (with molecular nu the lowest layer reverses).
    chezy_coefficient::Float64 = 0.0
    dt::Float64 = 0.0025
    # Implicitness theta (-, THETA_MIN <= theta <= THETA_MAX) of the free surface (issue #34; Casulli & Cattani 1994): the
    # surface level in the pressure gradient and the velocity in the continuity flux are taken at n + theta (layer
    # thicknesses at n), and the transport uses the matching theta velocities. theta = 1/2 is second order in time without
    # damping, theta = 1 (the former scheme) first order with a damping of about 2 pi^2 / N per wave period (N steps per
    # period). Stability: theta >= 1/2 for any time step without mean flow; with a mean flow of Froude number Fr and the
    # explicit momentum advection theta >= (1 + 2 Fr) / (2 (1 + Fr)), i.e. Fr <= theta_froude_max(theta).
    # WHY 0.7: stable up to Fr = 2/3 (theta = 0.55 only up to Fr = 0.11: a periodic channel at Fr = 0.45 with tvd2 grew
    # by 36 in 50 s, and decayed from the predicted theta = 0.59 on), damping 0.8 pi^2 / N per period (2.5 times less
    # than theta = 1). See docs/plan_issue_34_theta_scheme.md, section 3.
    theta::Float64 = 0.7
    # Implicit vertical terms (issue #34, src/implicit_vertical.jl): vertical viscosity, vertical momentum advection
    # (first-order upwind with the interface velocities of time n) and bed friction (linearised, c_b |u^n|) in one
    # tridiagonal matrix per u-face, coupled to the free surface (Casulli & Cheng 1992); the explicit momentum advection
    # is then horizontal only. No time-step limit from these terms. Also allows bed_roughness_length with wetting_drying.
    # WHY false by default: keeps the results of the validated cases (explicit vertical terms) unchanged.
    implicit_vertical_terms::Bool = false
    nt::Int = 40000
    rayleigh_damping::Float64 = 0.01
    max_froude::Float64 = 0.9
    apply_froude_limiter::Bool = true
    # Advective Courant number |u| dt/dx (-) above which step_forward! warns once (issue #61); Inf: no warning. The
    # default ADVECTIVE_COURANT_WARNING = 0.8 keeps a margin to the stability limit 1 of the explicit advection.
    advective_courant_warning::Float64 = ADVECTIVE_COURANT_WARNING
    momentum_advection_scheme::String = "tvd2"
    # Momentum advection of the layers in the wetting-drying step (issue #80): "column" (the column-scaled advective
    # velocity of `_wd_advection_tendency`) or "remapped" (Kleptsova, Stelling & Pietrzak 2010, sec. 6.1: at each face
    # the neighbouring columns are divided into sub-layers with the layer shares of the face, so that a flow without
    # vertical structure is advected as in 1D in every layer; applied in the hydrostatic mode on faces over a flat bed
    # without an active sluice gate, elsewhere the column form). The column form loses momentum where a bore lifts the
    # surface through layer interfaces (0.9 % in 10 layers, the Stoker bore 1.7 % behind); "remapped" gives the 1D
    # result there. WHY "column" by default: the remapped form is exact only for flows without vertical structure; with
    # structure (a gate jet and its jump, a non-hydrostatic bore front) it gains momentum or lets a jump oscillate.
    # See docs/TESTS_TECHNICAL_DOCUMENTATION.md, section 38.
    layer_momentum_advection::String = "column"
    # Density advection: "upwind1" (first-order donor cell, monotone for CFL_x + CFL_z <= 1) or "tvd2" (flux-limited
    # Lax-Wendroff with the MC limiter of the momentum tvd2 scheme, monotone for CFL_x + CFL_z <= 1/2).
    # WHY "upwind1" default: keeps the results of the validated cases unchanged.
    transport_advection_scheme::String = "upwind1"
    use_hydrostatic::Bool = true
    # Periodic lateral domain; requires the sparse pressure solver and fixed-thickness layers.
    periodic_x::Bool = false
    pressure_solver_method::String = "spectral"
    write_plots::Bool = true
    write_animation::Bool = true
    write_pressure_buoyancy_diagnostics::Bool = true
    save_interval_seconds::Float64 = 2.0
    output_dir::String = "output_barotropic"
    wind_stress_x::Float64 = 0.0
    wind_mixing_diffusivity::Float64 = 0.0
    # Uniform horizontal body force (m/s^2) on the whole water column, e.g. g times a channel slope; drives steady flow
    # in a periodic channel (vegetation reference cases, issue #8).
    body_force_x::Float64 = 0.0
    # Absorbing sponge layer at the right end (e.g. for waves): over the last sponge_length (m) of the domain u is damped
    # with the rate sponge_rate * s^2 (1/s), s = 0 ... 1 the relative position in the layer (quadratic ramp, implicit).
    sponge_length::Float64 = 0.0
    sponge_rate::Float64 = 0.0
    # Radiation condition of the free (and relaxed) open faces: the right outflow (right_free) and, in the wetting-drying
    # step, the free left face (left_free) (issue #86, docs/numerics_open_boundaries.md sections 5 and 7):
    # - radiation_theta: weight the radiation condition with the theta of the interior (true) instead of backward
    #   Euler (false). WHY an option: false keeps the validated results of issues #46/#49/#54 bit for bit.
    # - radiation_period (s): 0 = long-wave speed U + sqrt(g h); > 0 = phase speed of a wave of this absolute period
    #   on the current U (in the direction of the outgoing wave), from (omega - k U)^2 = g k tanh(k h); requires the
    #   pressure radiation below.
    # - nonhydro_open_face: "neumann" (dq/dx = 0 at the open face, issue #49) or "radiation" (the non-hydrostatic pressure
    #   of the face radiates, implicit in the pressure problem; wetting-drying step only).
    radiation_theta::Bool = false
    radiation_period::Float64 = 0.0
    nonhydro_open_face::String = "neumann"
    # Scheme of the free open faces (issue #86, docs/numerics_open_boundaries.md section 8): "radiation" (the radiation
    # condition above, the default) or "characteristic" (wetting-drying step: natural condition, the face momentum plus
    # open_face_beta c_beta d(eta_b - eta_N)/dt, and the essential condition on the incoming characteristic variable,
    # with the drift correction of a relaxed face on that variable instead of the level).
    # - open_face_beta (-): weight of the natural condition; -1/2 (box form) reflects (k dx)^3/64 instead of (k dx)^2/16.
    # - open_face_wet_depth (m): below this depth of the boundary column (time n) beta is limited to -phi/c_beta, so
    #   that the face keeps the free-surface matrix an M-matrix even if the column falls dry (section 8.4).
    # - radiation_speed_factor (-): factor on the intrinsic phase speed of all radiation speeds (level, non-hydrostatic
    #   pressure, characteristic speeds), c_r = U + f (c - U). WHY: the remaining reflection of the open face is half the
    #   interior's phase-speed error (section 7.2); with f = c_num/c, the model's own phase speed of the design wave
    #   (`numerical_phase_speed_factor`), the face radiates the wave that the model actually carries.
    radiation_speed_factor::Float64 = 1.0
    # - open_face_relaxation: what a relaxed free face (right/left_relaxation_time) relaxes: "level" (eta_b towards the
    #   target level, the former relaxed outflow; reflects 1/sqrt(1 + 4 omega^2 tau^2)) or "incoming" (the incoming
    #   characteristic variable c_out eta - Q_n towards c_out eta_T - Q_T,n, which an outgoing linear wave does not
    #   change: no reflection at the design frequency; docs/numerics_open_boundaries.md section 8.1). The characteristic
    #   scheme always relaxes the incoming variable.
    open_face_relaxation::String = "level"
    open_face_scheme::String = "radiation"
    open_face_beta::Float64 = -0.5
    open_face_wet_depth::Float64 = 0.1
    # Variable-thickness top layer (hydrostatic mode only): the top layer has the thickness dz + eta instead of
    # dz, its volume change d(eta)/dt enters the top-layer budgets, and the free surface is a material boundary
    # (zero flux through it). nothing (default) = automatic: on for the hydrostatic mode, off for the non-hydrostatic
    # mode; true / false force it (true is rejected with use_hydrostatic=false). false keeps the legacy fixed-grid
    # treatment, which is needed when the surface leaves the top layer (|eta| >~ dz, e.g. large waves or the surface
    # dam break); its density transport uses the top-row advective form (docs section 4). Evaluate with
    # use_variable_top_layer(cfg). See docs section 13.
    variable_top_layer::Union{Nothing,Bool} = nothing
    # Smallest allowed top-layer thickness as a fraction of dz (-). The variable top layer cannot follow a surface
    # that leaves the top layer (no layer removal / wetting-drying), so the run stops if dz + eta < fraction * dz.
    top_layer_min_fraction::Float64 = 0.1
    # Non-hydrostatic mode only (rejected with use_hydrostatic=true): Dirichlet p'=0 at the top face of the
    # non-hydrostatic pressure-correction Poisson problem, instead of the legacy Neumann (zero-flux) condition there.
    # WHY: with the legacy Neumann condition, a pure free-surface gravity wave (no baroclinic forcing) does not become
    # dispersive -- its numerical period matches the shallow-water relation T=2L/sqrt(gH) at any k*H, not the full
    # relation omega^2=g*k*tanh(k*H) -- because the barotropic surface force is applied uniformly over depth, outside
    # the elliptic solve, and (for a barotropic-only wave) there is nothing left for a Neumann-Neumann correction to
    # do. Dirichlet p'=0 at the top face is the correct dynamic free-surface condition for the part of the pressure
    # NOT already carried by the barotropic/hydrostatic split (see docs section 15), and lets the elliptic solve
    # produce the depth-varying (dispersive) correction. Requires pressure_solver_method="sparse": the spectral (DCT)
    # method's fast transform represents only a Neumann-Neumann eigenbasis and cannot express this condition.
    nonhydro_surface_dirichlet::Bool = false
    # Solver of the non-hydrostatic pressure of the wetting-drying step (issue #71): "direct" (sparse Cholesky),
    # "multigrid" (CG with the column-line multigrid of src/multigrid.jl) or "auto" (multigrid from
    # NH_MULTIGRID_MIN_UNKNOWNS unknowns on, below it the direct solver is faster). WHY auto: measured on the plane jet
    # (issue #25), the multigrid takes 0.63, 0.54 and 0.38 of the direct time at 8.6e3, 7.7e4 and 7e5 unknowns, but 3 to 7
    # times the (sub-millisecond) direct time below about 5e3.
    nonhydro_solver::String = "auto"
    # Relative residual |r| / |rhs| at which the multigrid CG stops (-). The pressure solve makes the theta fluxes of the
    # cells below the surface divergence-free; the residual is a divergence error of this relative size. WHY 1e-8: the
    # pressure then differs from the direct solution by about 1e-9 relative, well below the splitting error of the step
    # (the repeated free-surface solve changes the fluxes by more).
    nonhydro_solver_tolerance::Float64 = 1e-8
    # CG iterations of the multigrid before the direct solver takes over for that step (normally 7-17).
    nonhydro_solver_max_iterations::Int = NH_MULTIGRID_MAX_ITERATIONS
    # Bed level per column (m above the grid bottom, length nx; issue #3), a staircase on the z-level grid: each entry
    # must be a whole number of cells and leave at least two wet cells (the bed stays below the top layer; no dry
    # columns). nothing (default) = flat bed, legacy code paths. An explicit vector (also all zeros) uses the masked
    # code paths; see BedGeometry in src/geometry.jl.
    bed_level::Union{Nothing,Vector{Float64}} = nothing
    # Wetting and drying (issue #31, src/wetting_drying.jl): time step on cells cut by the bed and the free surface, with
    # dry columns, hydrostatic or non-hydrostatic, momentum and transport advection "upwind1" or "tvd2"; the free-surface
    # solve keeps every depth >= 0 (docs/depth_positivity.md). bed_level may then take any value >= 0 (not only whole
    # cells) and nz = 1 (1D) is allowed. Requires a closed or prescribed lateral boundary (no periodic_x).
    wetting_drying::Bool = false
    # Wetting and drying: smallest upwind water depth (m) at which a u-face is open; shallower faces carry no flow.
    dry_depth_threshold::Float64 = 1e-6
end

"""
    use_variable_top_layer(cfg) -> Bool

Whether `cfg` runs with the variable-thickness top layer: the explicit `cfg.variable_top_layer` if set, otherwise
`cfg.use_hydrostatic` and no k-epsilon closure (automatic default; the k-epsilon closure takes the variable top layer
only with an explicit `variable_top_layer=true`). Evaluated on every use, so changing `use_hydrostatic` or
the closure on a config after creation (as `copy_config` workflows do) keeps the default consistent.
"""
use_variable_top_layer(cfg::ModelConfig) =
    cfg.wetting_drying ? false :   # WHY: the wet/dry step has its own cut-cell geometry, which contains the variable top layer
    cfg.variable_top_layer === nothing ? (cfg.use_hydrostatic && turbulence_closure(cfg) != "k_epsilon") : cfg.variable_top_layer

"""
    turbulence_closure(cfg) -> String

Effective turbulence closure: `cfg.turbulence_closure` if set, otherwise "mixing_length" when the legacy flag
`use_turbulence_model` is set, else "none".
"""
turbulence_closure(cfg::ModelConfig) = cfg.turbulence_closure != "none" ? cfg.turbulence_closure : (cfg.use_turbulence_model ? "mixing_length" : "none")

"""
    output_case_tag(cfg) -> String

Case tag used in the names of the tagged output files written by `run!`, e.g.
`hydro_mom-tvd2_trans-upwind1_dt0p0025_L30p0_H4p0_dx0p300_dz0p200`: pressure mode, momentum and transport advection
schemes, dt (s), length and depth (m), dx and dz (m); every '.' is replaced by 'p'.
"""
function output_case_tag(cfg::ModelConfig)
    mode_tag = cfg.use_hydrostatic ? "hydro" : "nonhydro"
    tag = replace(
        @sprintf(
            "%s_mom-%s_trans-%s_dt%.4f_L%.1f_H%.1f_dx%.3f_dz%.3f",
            mode_tag,
            cfg.momentum_advection_scheme,
            cfg.transport_advection_scheme,
            cfg.dt,
            cfg.length,
            cfg.depth,
            cfg.length / cfg.nx,
            cfg.depth / cfg.nz,
        ),
        "." => "p",
    )
    return cfg.periodic_x ? "$(tag)_xperiodic" : tag
end

function copy_config(cfg::ModelConfig)
    return ModelConfig(
        nz=cfg.nz,
        nx=cfg.nx,
        length=cfg.length,
        depth=cfg.depth,
        g=cfg.g,
        rho0=cfg.rho0,
        rho_left=cfg.rho_left,
        rho_right=cfg.rho_right,
        nu=cfg.nu,
        use_turbulence_model=cfg.use_turbulence_model,
        turbulence_mixing_length=cfg.turbulence_mixing_length,
        max_turbulent_viscosity=cfg.max_turbulent_viscosity,
        turbulence_closure=cfg.turbulence_closure,
        bed_roughness_length=cfg.bed_roughness_length,
        dt=cfg.dt,
        theta=cfg.theta,
        implicit_vertical_terms=cfg.implicit_vertical_terms,
        nt=cfg.nt,
        rayleigh_damping=cfg.rayleigh_damping,
        max_froude=cfg.max_froude,
        apply_froude_limiter=cfg.apply_froude_limiter,
        momentum_advection_scheme=cfg.momentum_advection_scheme,
        transport_advection_scheme=cfg.transport_advection_scheme,
        use_hydrostatic=cfg.use_hydrostatic,
        periodic_x=cfg.periodic_x,
        pressure_solver_method=cfg.pressure_solver_method,
        write_plots=cfg.write_plots,
        write_animation=cfg.write_animation,
        write_pressure_buoyancy_diagnostics=cfg.write_pressure_buoyancy_diagnostics,
        save_interval_seconds=cfg.save_interval_seconds,
        output_dir=cfg.output_dir,
        wind_stress_x=cfg.wind_stress_x,
        wind_mixing_diffusivity=cfg.wind_mixing_diffusivity,
        body_force_x=cfg.body_force_x,
        sponge_length=cfg.sponge_length,
        sponge_rate=cfg.sponge_rate,
        radiation_theta=cfg.radiation_theta,
        radiation_period=cfg.radiation_period,
        nonhydro_open_face=cfg.nonhydro_open_face,
        radiation_speed_factor=cfg.radiation_speed_factor,
        open_face_relaxation=cfg.open_face_relaxation,
        open_face_scheme=cfg.open_face_scheme,
        open_face_beta=cfg.open_face_beta,
        open_face_wet_depth=cfg.open_face_wet_depth,
        variable_top_layer=cfg.variable_top_layer,
        top_layer_min_fraction=cfg.top_layer_min_fraction,
        nonhydro_surface_dirichlet=cfg.nonhydro_surface_dirichlet,
        bed_level=cfg.bed_level === nothing ? nothing : copy(cfg.bed_level),
        wetting_drying=cfg.wetting_drying,
        dry_depth_threshold=cfg.dry_depth_threshold,
        advective_courant_warning=cfg.advective_courant_warning,
    )
end

mutable struct ModelWorkspace{T<:Real}
    adv_u::Matrix{T}
    diff_u::Matrix{T}
    nu_eff_u::Matrix{T}
    adv_w::Matrix{T}
    p_corr::Matrix{T}
    p_internal::Matrix{T}
    grad_p_x::Matrix{T}
    grad_p_z::Matrix{T}
    grad_pc_x::Matrix{T}
    grad_pc_z::Matrix{T}
    u_hat::Matrix{T}
    w_hat::Matrix{T}
    rho_w::Matrix{T}
    buoy::Matrix{T}
    div_u::Matrix{T}
    eta_old::Vector{T}
    eta_grad_x::Vector{T}
    eta_t::Vector{T}
    rho_new::Matrix{T}
    slack::Matrix{T}
    div_tmp::Matrix{T}
    advection_u_ws::AdvectionWorkspace{T}
    advection_w_ws::AdvectionWorkspace{T}
    interpolation_ws::InterpolationWorkspace{T}
    density_ws::DensityWorkspace{T}
    tridiag_dp::Vector{T}
    tridiag_x::Vector{T}
    h_top_face::Vector{T}   # m, top-layer thickness at the u-faces (variable top layer only)
    u_old::Matrix{T}        # m/s, nz x (nx+1): u of time level n
    u_theta::Matrix{T}      # m/s, nz x (nx+1): theta u^{n+1} + (1 - theta) u^n (continuity flux and transport)
    w_theta::Matrix{T}      # m/s, (nz+1) x nx: vertical velocity of the continuity of u_theta (transport)
    # implicit vertical terms (implicit_vertical_terms=true)
    r_vert::Matrix{T}       # -, nz x (nx+1): column response r = A^{-1} dZ per u-face (1 without implicit terms)
    H_face::Vector{T}       # m, nx+1: effective depth sum_k dZ_k r_k per u-face
    col::Matrix{T}          # work columns (nz+1) x 11: u, r, uhat, dZ, W, nu, cprime, dprime, rprime, h_adv, zprime
end

function ModelWorkspace(cfg::ModelConfig; T::Type{<:Real}=Float64)
    nz = cfg.nz
    nx = cfg.nx
    return ModelWorkspace(
        zeros(T, nz, nx + 1),
        zeros(T, nz, nx + 1),
        zeros(T, nz, nx + 1),
        zeros(T, nz + 1, nx),
        zeros(T, nz, nx),
        zeros(T, nz, nx),
        zeros(T, nz, nx + 1),
        zeros(T, nz + 1, nx),
        zeros(T, nz, nx + 1),
        zeros(T, nz + 1, nx),
        zeros(T, nz, nx + 1),
        zeros(T, nz + 1, nx),
        zeros(T, nz - 1, nx),
        zeros(T, nz - 1, nx),
        zeros(T, nz, nx),
        zeros(T, nx),
        zeros(T, nx + 1),
        zeros(T, nx),
        zeros(T, nz, nx),
        zeros(T, nz, nx),
        zeros(T, nz, nx),
        AdvectionWorkspace(nz, nx + 1; T=T),
        AdvectionWorkspace(nz + 1, nx; T=T),
        InterpolationWorkspace(nz, nx; T=T),
        DensityWorkspace(nz, nx; T=T),
        zeros(T, nx),
        zeros(T, nx),
        zeros(T, nx + 1),
        zeros(T, nz, nx + 1),
        zeros(T, nz, nx + 1),
        zeros(T, nz + 1, nx),
        ones(T, nz, nx + 1),
        zeros(T, nx + 1),
        zeros(T, nz + 1, 11),
    )
end

mutable struct Model{T<:Real}
    config::ModelConfig
    dx::T
    dz::T
    state::State{T}
    rho_min_init::T
    rho_max_init::T
    initial_constituent_mass::T
    last_kinematic_mismatch::T
    last_nonhydro_pcorr_max::T
    last_p_hydro::Matrix{T}
    last_p_corr::Matrix{T}
    last_buoy_w::Matrix{T}
    barotropic_solver::Any
    pressure_correction_solver::PressureCorrectionSolver{T}
    rho_0_w::Vector{T}
    rho_0_w_2d::Matrix{T}
    # Non-hydrostatic Dirichlet surface mode only (docs section 15): z-gradient of the hydrostatic pressure of the
    # REFERENCE density profile (the initial state's column 1, matching rho_0_w's reference), size (nz+1) x nx.
    # WHY: solve_hydrostatic_pressure! integrates the FULL density (not the anomaly rho-rho_ref), so grad_p_z carries
    # a large "trivial" reference-profile contribution alongside the (much smaller) genuine baroclinic one. That is
    # harmless for the legacy Neumann-Neumann projection (mean-removed; only grad_pc_x's gradient of an
    # x-independent field matters, which is exactly zero for a z-only reference). It is NOT harmless for the
    # Dirichlet-top projection (no mean removal; the absolute magnitude of p_corr is physically meaningful): the
    # reference contribution would dominate p_corr's magnitude (found empirically: ~1e4 Pa here vs. a genuine
    # dynamic-pressure signal ~1 Pa for a small-amplitude surface wave), risking floating-point cancellation and, for
    # a Dirichlet condition intended to mean "no dynamic pressure beyond the barotropic/hydrostatic split" (docs
    # section 15), applying it to the wrong (reference-contaminated) quantity in the first place. Subtracted from
    # ws.grad_p_z in step_forward! before the w_hat predictor, only when cfg.nonhydro_surface_dirichlet.
    grad_p_z_reference::Matrix{T}
    timers::TimerOutput
    ws::ModelWorkspace{T}
    vegetation::Any
    x_boundary_u_left_profile::Any
    x_boundary_u_right_open::Bool
    x_boundary_u_right_profile::Any
    # WHY typed (not Any) here and below: read in the kernels of every step; an untyped field made the values derived from
    # it dynamic and boxed (issue #10)
    x_boundary_eta_right::Union{Nothing,Float64}   # nothing, or the prescribed water level eta_R (m) at the right boundary face
    x_boundary_q_left::Any      # nothing, or the discharge q (m^2/s, or a function of t (s)) of the left boundary face (profile = shape)
    x_boundary_eta_left::Union{Nothing,Float64}    # nothing, or the level eta_L (m) of the left boundary face (supercritical inflow, #46)
    x_boundary_rho_left::Union{Nothing,Float64,Vector{Float64}}    # nothing, or the density (kg/m^3, scalar or one per layer) of the inflow at the left boundary face (#47, #42)
    x_boundary_right_free::Bool # free outflow at the right boundary face (issue #46)
    x_boundary_eta_free::Float64  # m, radiated level of the free outflow face (_update_free_outflow_level!)
    x_boundary_free_dirichlet::Tuple{Float64,Float64}   # (m, -), effective level and weight of the free outflow this step
    x_boundary_eta_target::Float64        # m, target level of a relaxed free outflow (issue #54)
    x_boundary_relaxation_time::Float64   # s, relaxation time of the free outflow to x_boundary_eta_target; Inf: none
    geometry::BedGeometry   # staircase bed masks (all cells wet for the flat bed)
    # WHY not Any: the kernels read it in their inner loops; an untyped field boxed every value computed from it
    # (about 10 MiB per wetting-drying step with k-epsilon on 200 x 56 cells, issue #10)
    turbulence::Union{Nothing,KEpsilonState{T}}   # KEpsilonState with turbulence_closure = "k_epsilon", else nothing
    wetdry::Union{Nothing,WetDryState{T}}   # WetDryState with wetting_drying = true, else nothing
    max_surface_froude::Float64   # -, largest surface_froude_number after any step so far (issue #39)
    froude_warning_issued::Bool   # the theta Froude-bound warning has been given (once per model)
    boundary_froude_warned::Bool  # the boundary-type Froude warning has been given (once per model, issue #46)
    max_advective_courant::Float64   # -, largest |u| dt/dx after any step so far (issue #61)
    courant_warning_issued::Bool     # the advective Courant warning has been given (once per model)
    reach_mode_warned::Bool          # the warning of the growing reach mode has been given (once per model)
    # Density extrapolation of the momentum predictor (baroclinic_extrapolation_weight, issue #23)
    rho_previous::Matrix{T}     # kg/m^3, nz x nx: (buoyant) density at the start of the last step; NaN in cells dry then
    rho_baroclinic::Matrix{T}   # kg/m^3, nz x nx: extrapolated density of this step's predictor (work array)
    rho_previous_valid::Bool    # false after a (re)initialisation: the next step does not extrapolate
    performance::ModelPerformance    # steps, their wall-clock time and allocations (print_performance_report, issue #10)
end

function _linspace(start::Real, stop::Real, n::Int)
    return collect(range(start, stop, length=n))
end

function _interp_linear(xout::AbstractVector{<:Real}, xin::AbstractVector{<:Real}, yin::AbstractVector{<:Real})
    yout = similar(xout)
    @inbounds for j in eachindex(xout)
        x = xout[j]
        if x <= xin[1]
            yout[j] = yin[1]
        elseif x >= xin[end]
            yout[j] = yin[end]
        else
            hi = searchsortedfirst(xin, x)
            lo = hi - 1
            t = (x - xin[lo]) / (xin[hi] - xin[lo])
            yout[j] = (1.0 - t) * yin[lo] + t * yin[hi]
        end
    end
    return yout
end

# Constituent mass (kg/m) of the current state; uses the variable top-layer thickness dz + eta when it is active.
function _constituent_mass(model::Model)
    model.wetdry === nothing || return _wd_constituent_mass(model)
    st = model.state
    if model.geometry.any_solid
        # PERF: the wet cells are gathered into a work array instead of sum(st.rho[wet_cell]), whose logical index
        # allocated a copy every step (issue #10). WHY the same gather and sum: the same values in the same order give
        # the same pairwise sum, bit for bit; this mass drives _correct_constituent_mass!.
        wet, values = model.geometry.wet_cell, model.geometry.wet_values
        j = 0
        @inbounds for idx in eachindex(st.rho, wet)
            if wet[idx]
                j += 1
                values[j] = st.rho[idx]
            end
        end
        mass = sum(values) * model.dx * model.dz
        if use_variable_top_layer(model.config)
            nz = size(st.rho, 1)
            @inbounds for i in eachindex(st.eta)
                mass += st.rho[nz, i] * st.eta[i] * model.dx
            end
        end
        return mass
    end
    if use_variable_top_layer(model.config)
        return constituent_mass(st.rho, model.dx, model.dz; eta=st.eta)
    end
    return constituent_mass(st.rho, model.dx, model.dz)
end

# WHY: the transport keeps rho within [rho_min_init, rho_max_init] (monotone schemes; clipped mass is redistributed);
# a prescribed inflow density (left_rho) outside the initial range is a legitimate new extreme and must not be clipped
function _include_inflow_density_in_bounds!(model::Model)
    rho_in = model.x_boundary_rho_left
    rho_in === nothing && return nothing
    model.rho_min_init = min(model.rho_min_init, minimum(rho_in))
    model.rho_max_init = max(model.rho_max_init, maximum(rho_in))
    return nothing
end

function _refresh_density_metadata!(model::Model)
    st = model.state
    cfg = model.config
    # WHY: a new or changed density field has no history; extrapolating against the old one would add a spurious force
    model.rho_previous_valid = false

    model.rho_min_init = minimum(st.rho)
    model.rho_max_init = maximum(st.rho)
    _include_inflow_density_in_bounds!(model)
    model.initial_constituent_mass = _constituent_mass(model)

    z_w = _linspace(0.0, cfg.depth, cfg.nz + 1)
    z_c = _linspace(model.dz / 2.0, cfg.depth - model.dz / 2.0, cfg.nz)
    rho_0_w = _interp_linear(z_w, z_c, @view st.rho[:, 1])
    copyto!(model.rho_0_w, rho_0_w)
    model.rho_0_w_2d .= reshape(@view(model.rho_0_w[2:end-1]), :, 1)

    if cfg.nonhydro_surface_dirichlet
        # See the Model struct's grad_p_z_reference field docstring for why this is needed. rho_ref is the SAME
        # column-1 profile rho_0_w is built from (cell-centered, not yet interpolated to faces), repeated across all
        # columns: p_ref is then exactly x-independent, so its OWN x-gradient (used elsewhere, e.g. grad_p_x) is
        # already exactly zero without any correction -- only the z-gradient needs this explicit subtraction.
        rho_ref = repeat(reshape(collect(@view(st.rho[:, 1])), :, 1), 1, cfg.nx)
        p_ref = solve_hydrostatic_pressure(zeros(eltype(st.rho), cfg.nz, cfg.nx), rho_ref, cfg.g, model.dz)
        grad_z_c!(model.grad_p_z_reference, p_ref, model.dz)
    else
        fill!(model.grad_p_z_reference, 0.0)
    end

    return nothing
end

function refresh_density_metadata!(model::Model)
    return _refresh_density_metadata!(model)
end

# Density (kg/m^3) for the baroclinic pressure and buoyancy of this step's momentum predictor: rho^n + beta (rho^n -
# rho^{n-1}) with beta = baroclinic_extrapolation_weight(theta) (issue #23); stores rho^n for the next step. `rho` is the
# density of time n (st.rho, or the buoyant mixture density with wetting and drying). With `wet` (cell thicknesses of time
# n, m) cells thinner than `min_thickness` are stored as NaN, and a cell that was dry at the last step is not
# extrapolated. WHY: its stored density is that of an empty cell, not of the water now in it.
# theta = 1 returns `rho` itself (the former scheme, bit for bit); the first step after a (re)initialisation does not
# extrapolate.
function _baroclinic_density(model::Model, rho::AbstractMatrix; wet::Union{Nothing,AbstractMatrix}=nothing,
        min_thickness::Real=0.0)
    beta = baroclinic_extrapolation_weight(model.config.theta)
    prev, out = model.rho_previous, model.rho_baroclinic
    valid = model.rho_previous_valid
    model.rho_previous_valid = true
    @inbounds for idx in eachindex(out, rho, prev)
        r = rho[idx]
        p = prev[idx]
        out[idx] = (valid && !isnan(p)) ? r + beta * (r - p) : r
        prev[idx] = (wet === nothing || wet[idx] > min_thickness) ? r : oftype(r, NaN)
    end
    return beta == 0 ? rho : out
end

# Clears the non-hydrostatic pressure of the previous step (ws.p_corr, Pa; wetting-drying q_nh, Pa, and w_nh, m/s), called
# by every initialize_*!. WHY: the incremental projection (van Kan 1986) carries q^n into the next predictor, so a model
# re-initialised after a run would otherwise feel the stale pressure gradient of the old flow.
function _reset_nonhydrostatic_history!(model::Model)
    model.rho_previous_valid = false   # the density history of the baroclinic extrapolation (issue #23) as well
    fill!(model.ws.p_corr, 0.0)
    if model.wetdry !== nothing
        fill!(model.wetdry.q_nh, 0.0)
        fill!(model.wetdry.w_nh, 0.0)
    end
    return nothing
end

function initialize_lock_exchange!(model::Model; rho_left::Real=model.config.rho_left, rho_right::Real=model.config.rho_right)
    cfg = model.config
    st = model.state

    half = fld(cfg.nx, 2)
    @views st.rho[:, 1:half] .= rho_left
    @views st.rho[:, (half + 1):end] .= rho_right
    fill!(st.u, 0.0)
    fill!(st.w, 0.0)
    fill!(st.eta, 0.0)
    fill!(st.p, 0.0)
    _reset_nonhydrostatic_history!(model)

    _refresh_density_metadata!(model)

    return nothing
end

function initialize_rayleigh_taylor!(
    model::Model;
    rho_top::Real,
    rho_bottom::Real,
    interface_height_fraction::Real=0.5,
    perturbation_amplitude_fraction::Real=0.02,
    perturbation_wavenumber::Int=1,
)
    cfg = model.config
    st = model.state

    if !(0.0 < interface_height_fraction < 1.0)
        throw(ArgumentError("interface_height_fraction must be in (0, 1)"))
    end
    if perturbation_amplitude_fraction < 0.0
        throw(ArgumentError("perturbation_amplitude_fraction must be non-negative"))
    end
    if perturbation_wavenumber < 1
        throw(ArgumentError("perturbation_wavenumber must be >= 1"))
    end

    interface_height = interface_height_fraction * cfg.depth
    perturbation_amplitude = perturbation_amplitude_fraction * cfg.depth

    x_centers = _linspace(model.dx / 2.0, cfg.length - model.dx / 2.0, cfg.nx)
    z_centers = _linspace(model.dz / 2.0, cfg.depth - model.dz / 2.0, cfg.nz)

    fill!(st.rho, rho_bottom)
    @inbounds for i in 1:cfg.nx
        phase = 2.0 * pi * perturbation_wavenumber * x_centers[i] / cfg.length
        z_interface = interface_height + perturbation_amplitude * cos(phase)
        for k in 1:cfg.nz
            if z_centers[k] >= z_interface
                st.rho[k, i] = rho_top
            end
        end
    end

    fill!(st.u, 0.0)
    fill!(st.w, 0.0)
    fill!(st.eta, 0.0)
    fill!(st.p, 0.0)
    _reset_nonhydrostatic_history!(model)

    _refresh_density_metadata!(model)

    return nothing
end

function initialize_rising_plume!(
    model::Model;
    rho_ambient::Real,
    rho_plume::Real,
    plume_center_x_fraction::Real=0.5,
    plume_center_z_fraction::Real=0.2,
    plume_sigma_x_fraction::Real=0.06,
    plume_sigma_z_fraction::Real=0.06,
)
    cfg = model.config
    st = model.state

    if !(0.0 < plume_center_x_fraction < 1.0)
        throw(ArgumentError("plume_center_x_fraction must be in (0, 1)"))
    end
    if !(0.0 < plume_center_z_fraction < 1.0)
        throw(ArgumentError("plume_center_z_fraction must be in (0, 1)"))
    end
    if plume_sigma_x_fraction <= 0.0 || plume_sigma_z_fraction <= 0.0
        throw(ArgumentError("plume sigma fractions must be > 0"))
    end

    xc = plume_center_x_fraction * cfg.length
    zc = plume_center_z_fraction * cfg.depth
    sx = plume_sigma_x_fraction * cfg.length
    sz = plume_sigma_z_fraction * cfg.depth

    x_centers = _linspace(model.dx / 2.0, cfg.length - model.dx / 2.0, cfg.nx)
    z_centers = _linspace(model.dz / 2.0, cfg.depth - model.dz / 2.0, cfg.nz)

    drho = rho_plume - rho_ambient
    @inbounds for i in 1:cfg.nx
        dxn = (x_centers[i] - xc) / sx
        for k in 1:cfg.nz
            dzn = (z_centers[k] - zc) / sz
            st.rho[k, i] = rho_ambient + drho * exp(-(dxn * dxn + dzn * dzn))
        end
    end

    fill!(st.u, 0.0)
    fill!(st.w, 0.0)
    fill!(st.eta, 0.0)
    fill!(st.p, 0.0)
    _reset_nonhydrostatic_history!(model)

    _refresh_density_metadata!(model)

    return nothing
end

function initialize_sinking_plume!(
    model::Model;
    rho_ambient::Real,
    rho_plume::Real,
    plume_center_x_fraction::Real=0.5,
    plume_center_z_fraction::Real=0.8,
    plume_sigma_x_fraction::Real=0.06,
    plume_sigma_z_fraction::Real=0.06,
)
    return initialize_rising_plume!(
        model;
        rho_ambient=rho_ambient,
        rho_plume=rho_plume,
        plume_center_x_fraction=plume_center_x_fraction,
        plume_center_z_fraction=plume_center_z_fraction,
        plume_sigma_x_fraction=plume_sigma_x_fraction,
        plume_sigma_z_fraction=plume_sigma_z_fraction,
    )
end

function initialize_jet_impulse!(
    model::Model;
    rho_uniform::Real,
    u0::Real,
    x_center_fraction::Real=0.5,
    z_center_fraction::Real=0.5,
    sigma_x_fraction::Real=0.08,
    sigma_z_fraction::Real=0.05,
)
    cfg = model.config
    st = model.state

    if !(0.0 < x_center_fraction < 1.0)
        throw(ArgumentError("x_center_fraction must be in (0, 1)"))
    end
    if !(0.0 < z_center_fraction < 1.0)
        throw(ArgumentError("z_center_fraction must be in (0, 1)"))
    end
    if sigma_x_fraction <= 0.0 || sigma_z_fraction <= 0.0
        throw(ArgumentError("sigma fractions must be > 0"))
    end

    fill!(st.rho, rho_uniform)
    fill!(st.w, 0.0)
    fill!(st.eta, 0.0)
    fill!(st.p, 0.0)
    _reset_nonhydrostatic_history!(model)

    x0 = x_center_fraction * cfg.length
    z0 = z_center_fraction * cfg.depth
    sx = sigma_x_fraction * cfg.length
    sz = sigma_z_fraction * cfg.depth

    x_u = _linspace(0.0, cfg.length, cfg.nx + 1)
    z_u = _linspace(model.dz / 2.0, cfg.depth - model.dz / 2.0, cfg.nz)

    @inbounds for i in 1:(cfg.nx + 1)
        dxn = (x_u[i] - x0) / sx
        for k in 1:cfg.nz
            dzn = (z_u[k] - z0) / sz
            st.u[k, i] = u0 * exp(-(dxn * dxn + dzn * dzn))
        end
    end

    if cfg.periodic_x
        @views st.u[:, end] .= st.u[:, 1]
    else
        @views st.u[:, 1] .= 0.0
        @views st.u[:, end] .= 0.0
    end

    _refresh_density_metadata!(model)

    return nothing
end

"""
    initialize_internal_seiche!(model; rho_top, rho_bottom, interface_height_fraction=0.5,
        interface_tilt_amplitude_fraction=0.05, cell_average=false)

Two layers at rest, densities `rho_top` above `rho_bottom` (kg/m^3), the interface at
z = depth (interface_height_fraction + interface_tilt_amplitude_fraction cos(2 pi x / length)) (m above the grid bottom;
one full wavelength in the basin, mode 2 of a closed basin). With `cell_average` each cell takes the volume-weighted
density of the part of the cell below and above the interface. WHY: the default step profile puts the interface on the
cell faces, so a displacement smaller than dz is lost or rounded to a whole cell; the cell average keeps the displaced
volume exactly (internal seiche period, issue #23).
"""
function initialize_internal_seiche!(
    model::Model;
    rho_top::Real,
    rho_bottom::Real,
    interface_height_fraction::Real=0.5,
    interface_tilt_amplitude_fraction::Real=0.05,
    cell_average::Bool=false,
)
    cfg = model.config
    st = model.state

    if !(0.0 < interface_height_fraction < 1.0)
        throw(ArgumentError("interface_height_fraction must be in (0, 1)"))
    end
    if interface_tilt_amplitude_fraction < 0.0
        throw(ArgumentError("interface_tilt_amplitude_fraction must be non-negative"))
    end

    interface_height = interface_height_fraction * cfg.depth
    amp = interface_tilt_amplitude_fraction * cfg.depth
    x_centers = _linspace(model.dx / 2.0, cfg.length - model.dx / 2.0, cfg.nx)
    z_centers = _linspace(model.dz / 2.0, cfg.depth - model.dz / 2.0, cfg.nz)

    @inbounds for i in 1:cfg.nx
        z_interface = interface_height + amp * cos(2.0 * pi * x_centers[i] / cfg.length)
        for k in 1:cfg.nz
            if cell_average
                below = clamp((z_interface - (k - 1) * model.dz) / model.dz, 0.0, 1.0)   # -, fraction below the interface
                st.rho[k, i] = below * rho_bottom + (1.0 - below) * rho_top
            else
                st.rho[k, i] = z_centers[k] >= z_interface ? rho_top : rho_bottom
            end
        end
    end

    fill!(st.u, 0.0)
    fill!(st.w, 0.0)
    fill!(st.eta, 0.0)
    fill!(st.p, 0.0)
    _reset_nonhydrostatic_history!(model)

    _refresh_density_metadata!(model)

    return nothing
end

function initialize_surface_dam_break!(
    model::Model;
    rho_uniform::Real,
    eta_jump::Real,
    split_fraction::Real=0.5,
)
    cfg = model.config
    st = model.state

    if !(0.0 < split_fraction < 1.0)
        throw(ArgumentError("split_fraction must be in (0, 1)"))
    end

    split_idx = clamp(round(Int, split_fraction * cfg.nx), 1, cfg.nx - 1)

    fill!(st.rho, rho_uniform)
    fill!(st.u, 0.0)
    fill!(st.w, 0.0)
    fill!(st.p, 0.0)
    @views st.eta[1:split_idx] .= 0.5 * eta_jump
    @views st.eta[(split_idx + 1):end] .= -0.5 * eta_jump
    _reset_nonhydrostatic_history!(model)

    _refresh_density_metadata!(model)

    return nothing
end

function initialize_kelvin_helmholtz!(
    model::Model;
    rho_top::Real,
    rho_bottom::Real,
    interface_height_fraction::Real=0.5,
    interface_perturbation_amplitude_fraction::Real=0.02,
    interface_perturbation_wavenumber::Int=2,
    u_top::Real=0.2,
    u_bottom::Real=-0.2,
)
    cfg = model.config
    st = model.state

    if !(0.0 < interface_height_fraction < 1.0)
        throw(ArgumentError("interface_height_fraction must be in (0, 1)"))
    end
    if interface_perturbation_amplitude_fraction < 0.0
        throw(ArgumentError("interface_perturbation_amplitude_fraction must be non-negative"))
    end
    if interface_perturbation_wavenumber < 1
        throw(ArgumentError("interface_perturbation_wavenumber must be >= 1"))
    end

    z0 = interface_height_fraction * cfg.depth
    amp = interface_perturbation_amplitude_fraction * cfg.depth
    x_centers = _linspace(model.dx / 2.0, cfg.length - model.dx / 2.0, cfg.nx)
    z_centers = _linspace(model.dz / 2.0, cfg.depth - model.dz / 2.0, cfg.nz)

    @inbounds for i in 1:cfg.nx
        z_interface = z0 + amp * cos(2.0 * pi * interface_perturbation_wavenumber * x_centers[i] / cfg.length)
        for k in 1:cfg.nz
            st.rho[k, i] = z_centers[k] >= z_interface ? rho_top : rho_bottom
        end
    end

    x_u = _linspace(0.0, cfg.length, cfg.nx + 1)
    z_u = _linspace(model.dz / 2.0, cfg.depth - model.dz / 2.0, cfg.nz)
    transition = max(2.0 * model.dz, 1e-6)
    @inbounds for i in 1:(cfg.nx + 1)
        xphase = 2.0 * pi * interface_perturbation_wavenumber * x_u[i] / cfg.length
        z_interface_u = z0 + amp * cos(xphase)
        for k in 1:cfg.nz
            s = tanh((z_u[k] - z_interface_u) / transition)
            st.u[k, i] = 0.5 * (u_top + u_bottom) + 0.5 * (u_top - u_bottom) * s
        end
    end

    fill!(st.w, 0.0)
    fill!(st.eta, 0.0)
    fill!(st.p, 0.0)
    _reset_nonhydrostatic_history!(model)
    # WHY: with periodic x the first and last u-faces are the same seam face and carry the shear like every other face
    # (x_u[1] = 0 and x_u[end] = L have the same perturbation phase); zeroing them would put a wall-like velocity
    # defect across the whole depth at the seam. Closed walls keep zero normal velocity.
    if cfg.periodic_x
        @views st.u[:, end] .= st.u[:, 1]
    else
        @views st.u[:, 1] .= 0.0
        @views st.u[:, end] .= 0.0
    end

    _refresh_density_metadata!(model)

    return nothing
end

"""
    initialize_shear_layer_mode!(model; velocity, thickness, density_difference=0.0, amplitude, wavenumber_count=1,
        z0=depth/2) -> (growth_rate, phase_speed)

Periodic tanh shear layer seeded with its unstable Kelvin-Helmholtz mode (issue #22): mean flow
u = U tanh((z - z0)/delta), density rho0 - (drho/2) tanh((z - z0)/delta) (the light fluid on top), plus the
Taylor-Goldstein eigenfunction of the wavenumber k = 2 pi `wavenumber_count` / length (`kelvin_helmholtz_tanh_mode`,
walls at the bed and the top) with the interface displacement `amplitude` (m) at z0. `velocity` U (m/s), `thickness`
delta (m), `density_difference` drho (kg/m^3, >= 0); the bulk Richardson number is J = g drho delta / (2 rho0 U^2) < 1/4.
Returns the growth rate (1/s) and the complex phase speed (m/s) of the mode.

The perturbation velocities come from the stream function psi = A Re(phi(z) e^{i k x}) at the cell corners,
u = d psi / dz on the u-faces and w = -d psi / dx on the w-interfaces, so the discrete divergence is zero; the density
perturbation is rho' = A Re(phi drho_bar/dz / (U - c) e^{i k x}) (linearised density equation). WHY the eigenfunction:
the mode grows at its rate from the first step, without the transient of an interface displacement alone. Requires
`periodic_x` and a flat bed.
"""
function initialize_shear_layer_mode!(model::Model; velocity::Real, thickness::Real, density_difference::Real=0.0,
        amplitude::Real, wavenumber_count::Int=1, z0::Real=model.config.depth / 2)
    cfg = model.config
    st = model.state
    cfg.periodic_x || throw(ArgumentError("initialize_shear_layer_mode! needs periodic_x = true"))
    model.geometry.has_bed && throw(ArgumentError("initialize_shear_layer_mode! needs a flat bed"))
    density_difference >= 0 || throw(ArgumentError("density_difference must be >= 0 (light fluid on top)"))
    wavenumber_count >= 1 || throw(ArgumentError("wavenumber_count must be >= 1"))
    U, delta, drho = float(velocity), float(thickness), float(density_difference)
    k = 2pi * wavenumber_count / cfg.length                      # 1/m
    J = cfg.g * drho * delta / (2 * cfg.rho0 * U^2)              # -, bulk Richardson number
    sigma, c, zs, phis = kelvin_helmholtz_tanh_mode(; k=k, velocity=U, thickness=delta, richardson=J, depth=cfg.depth, z0=z0)
    dzs = zs[2] - zs[1]
    phi(z) = begin   # linear interpolation of the eigenfunction (m^2/s per unit A)
        t = clamp(z / dzs, 0.0, length(phis) - 1.0)
        j = min(floor(Int, t), length(phis) - 2)
        (1 - (t - j)) * phis[j + 1] + (t - j) * phis[j + 2]
    end
    A = amplitude / abs(phi(z0) / (0.0 - c))                     # m/s, interface displacement amplitude at z0
    psi(x, z) = A * real(phi(z) * cis(k * x))
    dx, dz = model.dx, model.dz
    rho_bar(z) = cfg.rho0 - drho / 2 * tanh((z - z0) / delta)
    rho_bar_z(z) = -drho / (2delta) * (1 - tanh((z - z0) / delta)^2)
    @inbounds for i in 1:(cfg.nx + 1), kk in 1:cfg.nz
        zc = (kk - 0.5) * dz
        st.u[kk, i] = U * tanh((zc - z0) / delta) + (psi((i - 1) * dx, kk * dz) - psi((i - 1) * dx, (kk - 1) * dz)) / dz
    end
    @views st.u[:, end] .= st.u[:, 1]   # WHY: the periodic seam face, exactly (cis(k L) = 1 only to round-off)
    @inbounds for i in 1:cfg.nx, kk in 1:(cfg.nz + 1)
        st.w[kk, i] = -(psi(i * dx, (kk - 1) * dz) - psi((i - 1) * dx, (kk - 1) * dz)) / dx
    end
    @inbounds for i in 1:cfg.nx, kk in 1:cfg.nz
        zc = (kk - 0.5) * dz
        Ubar = U * tanh((zc - z0) / delta)
        st.rho[kk, i] = rho_bar(zc) + A * real(phi(zc) * rho_bar_z(zc) / (Ubar - c) * cis(k * (i - 0.5) * dx))
    end
    fill!(st.eta, 0.0)
    fill!(st.p, 0.0)
    _reset_nonhydrostatic_history!(model)
    _refresh_density_metadata!(model)
    return (sigma, c)
end

function Model(cfg::ModelConfig; T::Type{<:Real}=Float64)
    _check_model_config(cfg)
    dx = T(cfg.length / cfg.nx)
    dz = T(cfg.depth / cfg.nz)
    st = from_shape(cfg.nz, cfg.nx; T=T)

    rho_min_init = minimum(st.rho)
    rho_max_init = maximum(st.rho)
    last_p_hydro = zeros(T, cfg.nz, cfg.nx)
    last_p_corr = zeros(T, cfg.nz, cfg.nx)
    last_buoy_w = zeros(T, cfg.nz - 1, cfg.nx)

    # WHY flat masks with wetting and drying: the wet/dry step keeps its own bed (any level, dry columns allowed)
    geometry = cfg.wetting_drying ? BedGeometry(nothing, cfg.nz, cfg.nx, cfg.depth / cfg.nz; min_wet_cells=1) :
        BedGeometry(cfg.bed_level, cfg.nz, cfg.nx, cfg.depth / cfg.nz; periodic_x=cfg.periodic_x)
    initial_constituent_mass = geometry.any_solid ? sum(st.rho[geometry.wet_cell]) * dx * dz : constituent_mass(st.rho, dx, dz)
    barotropic_solver = BarotropicSolver(cfg, dx, dz; T=T, geometry=geometry)
    # WHY wet only with solid cells: a zero bed keeps the rectangular (and spectral-capable) Poisson problem unchanged
    pressure_solver = PressureCorrectionSolver((cfg.nz, cfg.nx), dx, dz; method=cfg.pressure_solver_method, top_dirichlet=cfg.nonhydro_surface_dirichlet, periodic_x=cfg.periodic_x, wet=geometry.any_solid ? geometry.wet_cell : nothing, T=T)

    z_w = _linspace(0.0, cfg.depth, cfg.nz + 1)
    z_c = _linspace(dz / 2.0, cfg.depth - dz / 2.0, cfg.nz)
    rho_0_w = T.(_interp_linear(z_w, z_c, @view st.rho[:, 1]))
    rho_0_w_2d = repeat(reshape(@view(rho_0_w[2:end-1]), :, 1), 1, cfg.nx)
    grad_p_z_reference = zeros(T, cfg.nz + 1, cfg.nx)   # filled by _refresh_density_metadata! below

    ws = ModelWorkspace(cfg; T=T)

    timers = TimerOutput()
    model = Model(
        cfg,
        dx,
        dz,
        st,
        rho_min_init,
        rho_max_init,
        initial_constituent_mass,
        zero(T),
        zero(T),
        last_p_hydro,
        last_p_corr,
        last_buoy_w,
        barotropic_solver,
        pressure_solver,
        rho_0_w,
        rho_0_w_2d,
        grad_p_z_reference,
        timers,
        ws,
        nothing,
        nothing,
        false,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        false,
        0.0,
        (0.0, 1.0),
        0.0,
        Inf,
        geometry,
        turbulence_closure(cfg) == "k_epsilon" ? KEpsilonState(cfg.nz, cfg.nx; T=T) : nothing,
        cfg.wetting_drying ? WetDryState(cfg; T=T) : nothing,
        0.0,
        false,
        false,
        0.0,
        false,
        false,
        zeros(T, cfg.nz, cfg.nx),
        zeros(T, cfg.nz, cfg.nx),
        false,
        _register_performance!(_performance_label(cfg), cfg.nx * cfg.nz, timers, pressure_solver.timers),
    )

    initialize_lock_exchange!(model; rho_left=cfg.rho_left, rho_right=cfg.rho_right)
    if cfg.wetting_drying
        _wd_clamp_to_bed!(model)
        _refresh_density_metadata!(model)
    end

    return model
end

"""
    set_x_inflow_outflow!(model, left_profile; left_discharge=nothing, left_level=nothing, left_rho=nothing,
                          open_right=true, right_profile=nothing, right_level=nothing, right_free=false,
                          right_relaxation_time=nothing, left_free=false, left_relaxation_time=nothing,
                          right_target_discharge=nothing, left_target_discharge=nothing,
                          right_incoming_level=nothing, left_incoming_level=nothing)

Lateral boundary conditions: velocity `left_profile` (m/s, one value per layer) at the left boundary face, or with
`left_discharge` q (m^2/s) the profile shape scaled to the discharge, u_k = q * left_profile_k / sum_j dZ_j left_profile_j
with the face layer thicknesses dZ of time level n (variable top layer: dz + eta_1; wetting and drying: the cut face
layers; dry layers carry no flow), so that the inflow is q whatever the level at the inlet. `left_level` eta_L (m, with
`left_discharge`; issue #46) also prescribes the level of the inlet face: the face layers are those of eta_L instead of
the level of the first cell, so the inflow has the depth and velocity of the prescribed state (supercritical inflow,
Fr > 1, needs both; for subcritical inflow prescribe the discharge only). Variable top layer or wetting and drying only
(the fixed grid has the depth H at every face). In the wetting-drying mode `left_discharge` can be a function q(t)
(issue #61; q in m^2/s, t in s, the model time of the end of the step), e.g. a ramp: against a fixed level an impulsive
start excites the mode of the reach (docs section 28). `left_rho` (kg/m^3; issue #47, wetting and drying only): density of the
inflowing water, one value or one per layer (issue #42; otherwise the inflow takes the density of the first cell); it
widens the bounds of the monotone transport, so a new extreme is not clipped. `left_k` (m^2/s^2) and `left_eps`
(m^2/s^3), one value per layer, both or neither (issue #42; wetting and drying with k-epsilon only): k and eps carried by
the inflow; without them the inflow carries the values of the first cell (zero gradient).

At the right boundary face a wall, a zero-gradient velocity (`open_right=true`), a prescribed velocity `right_profile`
(m/s, length nz), a prescribed water level `right_level` (m, surface elevation eta at the boundary face x = L) or a free
outflow (`right_free=true`, issue #46); the last two override `open_right`. Both together with
`right_relaxation_time` tau (s, > 0; issue #54): a free outflow whose level relaxes to the target `right_level`.

Water-level boundary (issue #14) and free outflow: the velocity of the boundary face is computed by its own momentum
equation: advection in an open control volume (`_open_right_face_velocity!`), the baroclinic pressure gradient and
horizontal diffusion of face nx (zero gradient), its own implicit vertical column and the surface slope
(eta_b - eta_nx)/(dx/2) between the last cell centre and the level eta_b of the boundary face, implicit in the
free-surface solve like an interior face (`_right_boundary_slope`). With the variable top layer (and with wetting and
drying) the top layer of the boundary face has the level eta_b; in the wetting-drying step its bed is extrapolated to
x = L (`_wd_face_bed`).
- water level: eta_b = eta_R. For subcritical outflow; it reflects outgoing waves completely.
- free outflow: eta_b from the interior (`_update_free_outflow_level!`): extrapolated for supercritical outflow, an
  implicit radiation condition for the outgoing long wave for subcritical outflow (reflects < 1 % of a long wave). No
  level is prescribed: with the fixed grid the mean level is then a neutral mode.
- relaxed free outflow: the radiation condition with the relaxation term -(eta_b - eta_R)/tau. Outgoing waves of
  angular frequency omega are reflected by |R| = 1/sqrt(1 + 4 omega^2 tau^2), and the mean level returns to eta_R
  within about tau (for tau >> L/sqrt(g H)). tau = Inf is `right_free`, tau -> 0 `right_level`.
Both require `implicit_vertical_terms=true` and no `periodic_x`; in the non-hydrostatic mode (issue #49) the open face
has dq/dx = 0 (`_check_right_level_config`, `_wd_nonhydrostatic_correction!`).
Dispersive waves (issue #86, docs/numerics_open_boundaries.md section 7; ModelConfig options): `radiation_theta`
weights the radiation condition with theta; in the wetting-drying step `nonhydro_open_face="radiation"` lets the
non-hydrostatic pressure of the face radiate as well, and `radiation_period` T gives both radiation conditions the
phase speed of period T on the (filtered) current instead of the long-wave speed. A regular wave of k h = 1.2 then
reflects 1.5 % instead of 6.9 %; other periods are reflected more (single-frequency condition). These options apply to
both open ends.
Characteristic open faces (`ModelConfig(open_face_scheme="characteristic")`, wetting-drying step, issue #86,
docs/numerics_open_boundaries.md section 8): the free faces get the natural condition (face momentum plus the box term
of `open_face_beta`) and the essential condition on the incoming characteristic variable c_out eta - Q_n (outward
discharge); a relaxed face corrects that variable towards c_out eta_T - Q_T,n, with the target level (`right_level`,
`left_level`) and `right_target_discharge` / `left_target_discharge` Q_T (m^2/s, positive in +x; default 0), which
must be the discharge that passes, since the mean state of the face lies on that line. The same relaxation of the
incoming variable for the radiation scheme: `ModelConfig(open_face_relaxation="incoming")`.
Incoming waves (issue #86): `right_incoming_level` / `left_incoming_level` = t -> eta_in(t) (m, t in s), the level of
the incoming wave at a free face; the face imposes it and radiates the rest (outgoing and reflected waves). Needs
`radiation_theta=true` (radiation scheme); with the pressure radiation and `radiation_period` the face also carries the
incoming wave's non-hydrostatic pressure (linear theory). `numerical_phase_speed_factor` gives the
`radiation_speed_factor` that matches the radiation speed to the model's own phase speed.

Free left face (`left_free=true`, wetting-drying step only, issue #86): the mirror of the right free outflow. Face 1
gets its own level (radiation condition for the wave travelling in -x, `_update_left_free_level!`) and momentum equation
(surface slope (eta_1 - eta_b)/(dx/2), baroclinic pressure gradient of face 2, ghost cell = first cell), and with
`nonhydro_open_face="radiation"` its own radiating non-hydrostatic pressure. With `left_level` and
`left_relaxation_time` the level relaxes to `left_level` (relaxed free face; without `left_free`, `left_level` keeps its
meaning of a supercritical inlet level). Water may leave or enter through it; `left_discharge` and `left_free` exclude
each other. `step_forward!` warns once when the
Froude number at a boundary does not fit its type (`_check_boundary_froude!`), and once when a velocity inlet against a
prescribed level can feed the mode of the reach (`_check_reach_mode_growth!`).
"""
function set_x_inflow_outflow!(
    model::Model,
    left_profile::AbstractVector{<:Real};
    left_discharge::Union{Nothing,Real,Function}=nothing,
    left_level::Union{Nothing,Real}=nothing,
    left_rho::Union{Nothing,Real,AbstractVector{<:Real}}=nothing,
    left_k::Union{Nothing,AbstractVector{<:Real}}=nothing,
    left_eps::Union{Nothing,AbstractVector{<:Real}}=nothing,
    open_right::Bool=true,
    right_profile::Union{Nothing,AbstractVector{<:Real}}=nothing,
    right_level::Union{Nothing,Real}=nothing,
    right_free::Bool=false,
    right_relaxation_time::Union{Nothing,Real}=nothing,
    left_free::Bool=false,
    left_relaxation_time::Union{Nothing,Real}=nothing,
    right_target_discharge::Union{Nothing,Real}=nothing,
    left_target_discharge::Union{Nothing,Real}=nothing,
    right_incoming_level::Union{Nothing,Function}=nothing,
    left_incoming_level::Union{Nothing,Function}=nothing,
)
    if left_free
        _set_left_free_boundary!(model, left_discharge, left_level, left_relaxation_time)
        left_level = nothing   # the target of a relaxed free face, not the level of a supercritical inlet
    else
        left_relaxation_time === nothing || throw(ArgumentError("left_relaxation_time needs left_free=true and the target level left_level"))
        model.wetdry === nothing || (model.wetdry.left_open = nothing)
    end
    if left_discharge !== nothing
        if left_discharge isa Function
            model.wetdry === nothing && throw(ArgumentError("a time-dependent left_discharge requires ModelConfig(wetting_drying=true)"))
            isfinite(left_discharge(model.wetdry.time)) || throw(ArgumentError("left_discharge must be finite"))
        else
            isfinite(left_discharge) || throw(ArgumentError("left_discharge must be finite"))
        end
        all(>=(0.0), left_profile) && any(>(0.0), left_profile) ||
            throw(ArgumentError("left_discharge needs a profile shape with values >= 0, not all zero"))
    end
    if left_level !== nothing
        left_discharge === nothing && throw(ArgumentError("left_level needs left_discharge (supercritical inflow: discharge and level)"))
        isfinite(left_level) || throw(ArgumentError("left_level must be finite"))
        _check_left_level_config(model.config)
    end
    model.wetdry === nothing || (model.wetdry.left_velocity = nothing)
    model.x_boundary_u_left_profile = collect(left_profile)
    model.x_boundary_q_left = left_discharge isa Real ? Float64(left_discharge) : left_discharge
    model.x_boundary_eta_left = left_level === nothing ? nothing : Float64(left_level)
    _set_inflow_density!(model, left_rho)
    _set_inflow_turbulence!(model, left_k, left_eps)
    _set_right_boundary!(model, open_right, right_profile, right_level, right_free, right_relaxation_time)
    _set_target_discharges!(model, right_target_discharge, right_relaxation_time, left_target_discharge,
        left_free ? left_relaxation_time : nothing)
    _set_incoming_levels!(model, right_incoming_level, right_free, left_incoming_level, left_free)
    # WHY: the discharge scaling uses the level of time n in ws.eta_old (set by every step); before the first step it
    # is the initial level
    copyto!(model.ws.eta_old, model.state.eta)
    _apply_x_boundary_conditions!(model)
    return model
end

"""
    set_x_inflow_outflow!(model, left_velocity::Function; open_right=false, right_profile=nothing, right_level=nothing,
                          right_free=false, right_relaxation_time=nothing, left_rho=nothing, left_k=nothing,
                          left_eps=nothing, right_target_discharge=nothing, right_incoming_level=nothing)

Wetting-drying mode: time-dependent velocity at the left boundary face, `left_velocity(t)` (m/s; scalar or one value per
layer) at the model time t (s) of the end of each step; dry layers of the face carry no flow. Right boundary, inflow
density and inflow k and eps as in the method with a velocity profile.
"""
function set_x_inflow_outflow!(
    model::Model,
    left_velocity::Function;
    open_right::Bool=false,
    right_profile::Union{Nothing,AbstractVector{<:Real}}=nothing,
    right_level::Union{Nothing,Real}=nothing,
    right_free::Bool=false,
    right_relaxation_time::Union{Nothing,Real}=nothing,
    left_rho::Union{Nothing,Real,AbstractVector{<:Real}}=nothing,
    left_k::Union{Nothing,AbstractVector{<:Real}}=nothing,
    left_eps::Union{Nothing,AbstractVector{<:Real}}=nothing,
    right_target_discharge::Union{Nothing,Real}=nothing,
    right_incoming_level::Union{Nothing,Function}=nothing,
)
    model.wetdry === nothing && throw(ArgumentError("a time-dependent boundary velocity requires ModelConfig(wetting_drying=true)"))
    model.wetdry.left_velocity = left_velocity
    model.x_boundary_u_left_profile = nothing
    model.x_boundary_q_left = nothing
    model.x_boundary_eta_left = nothing
    _set_inflow_density!(model, left_rho)
    _set_inflow_turbulence!(model, left_k, left_eps)
    _set_right_boundary!(model, open_right, right_profile, right_level, right_free, right_relaxation_time)
    _set_target_discharges!(model, right_target_discharge, right_relaxation_time, nothing, nothing)
    _set_incoming_levels!(model, right_incoming_level, right_free, nothing, false)
    return model
end

# Discharge q (m^2/s) of the left boundary face at time t (s).
function _left_discharge(model::Model, t::Real)
    q = model.x_boundary_q_left
    q isa Function || return q
    value = q(t)
    isfinite(value) || throw(DomainError(value, "left_discharge(t = $t) is not finite"))
    return value
end

# Density (kg/m^3) of the water entering through the left boundary face (left_rho, issue #47); wetting-drying step only.
# WHY only there: its transport has fluxes through the outer faces (upwind, the inflow takes the density of the first
# cell without left_rho); the transport of the other modes has none (all outer faces closed for the constituent).
function _set_inflow_density!(model::Model, left_rho)
    if left_rho !== nothing
        all(isfinite, left_rho) || throw(ArgumentError("left_rho must be finite"))
        left_rho isa AbstractVector && length(left_rho) != model.config.nz &&
            throw(ArgumentError("left_rho needs one value per layer (length nz)"))
        model.wetdry === nothing && throw(ArgumentError("left_rho needs wetting_drying=true (the other transport schemes have no flux through the outer faces)"))
    end
    model.x_boundary_rho_left = left_rho === nothing ? nothing : left_rho isa Real ? Float64(left_rho) : Float64.(collect(left_rho))
    _include_inflow_density_in_bounds!(model)
    return nothing
end

# Value of layer k of an inflow quantity given as one value or one value per layer.
_inflow_value(value::Real, k::Int) = value
_inflow_value(value::AbstractVector, k::Int) = value[k]

# k (m^2/s^2) and eps (m^2/s^3) of the water entering through the left boundary face (issue #42), one value per layer.
# WHY both or neither: a prescribed k with the eps of the first cell gives an arbitrary length scale k^(3/2)/eps.
function _set_inflow_turbulence!(model::Model, left_k, left_eps)
    (left_k === nothing) == (left_eps === nothing) || throw(ArgumentError("left_k and left_eps go together"))
    if left_k !== nothing
        (model.wetdry !== nothing && model.turbulence isa KEpsilonState) ||
            throw(ArgumentError("left_k and left_eps need wetting_drying=true and turbulence_closure=\"k_epsilon\""))
        nz = model.config.nz
        (length(left_k) == nz && length(left_eps) == nz) || throw(ArgumentError("left_k and left_eps need one value per layer (length nz)"))
        (all(v -> isfinite(v) && v > 0, left_k) && all(v -> isfinite(v) && v > 0, left_eps)) ||
            throw(ArgumentError("left_k and left_eps must be finite and > 0"))
    end
    if model.wetdry !== nothing
        model.wetdry.left_k = left_k === nothing ? nothing : Float64.(collect(left_k))
        model.wetdry.left_eps = left_eps === nothing ? nothing : Float64.(collect(left_eps))
    end
    return nothing
end

# Free (radiating) left face (issue #86): wetting-drying step only; no discharge (the face has its own momentum equation);
# with left_level and left_relaxation_time the level relaxes to left_level (relaxed free face), as at the right.
function _set_left_free_boundary!(model::Model, left_discharge, left_level, left_relaxation_time)
    wd = model.wetdry
    wd === nothing && throw(ArgumentError("left_free=true is implemented for wetting_drying=true only"))
    left_discharge === nothing || throw(ArgumentError("left_free and left_discharge exclude each other (the free face computes its velocity)"))
    model.config.nx >= 2 || throw(ArgumentError("left_free needs nx >= 2 (the slope of face 2)"))
    _check_right_level_config(model.config)   # the same requirements as an open right face
    relaxed = left_relaxation_time !== nothing
    if relaxed
        left_level === nothing && throw(ArgumentError("left_relaxation_time needs the target level left_level"))
        left_relaxation_time > 0 || throw(ArgumentError("left_relaxation_time must be > 0 (Inf: no relaxation)"))
        isfinite(left_level) || throw(ArgumentError("left_level must be finite"))
    else
        left_level === nothing || throw(ArgumentError("left_level and left_free exclude each other (unless left_relaxation_time is given: relaxed free face)"))
    end
    eta_b = _extrapolated_left_level(model.state.eta)
    wd.left_open = OpenFaceState(eta_b, eta_b, 1.0, relaxed ? Float64(left_level) : 0.0,
        relaxed ? Float64(left_relaxation_time) : Inf, NaN)
    fill!(wd.q_left, 0.0)
    return nothing
end

# Target discharges Q_T (m^2/s, positive in +x) of the drift correction of relaxed characteristic faces (issue #86,
# docs/numerics_open_boundaries.md section 8.1): the mean state of such a face lies on c_out eta - Q_n = c_out eta_T -
# Q_T,n, so the target must carry the discharge that passes (0 if not given).
function _set_target_discharges!(model::Model, right_q, right_tau, left_q, left_tau)
    for (q, tau, side) in ((right_q, right_tau, "right"), (left_q, left_tau, "left"))
        q === nothing && continue
        (model.wetdry !== nothing && (model.config.open_face_scheme == "characteristic" ||
                                      model.config.open_face_relaxation == "incoming")) ||
            throw(ArgumentError("$(side)_target_discharge needs wetting_drying=true and open_face_scheme=\"characteristic\" or open_face_relaxation=\"incoming\""))
        tau === nothing && throw(ArgumentError("$(side)_target_discharge needs a relaxed free face ($(side)_free, $(side)_level, $(side)_relaxation_time)"))
        isfinite(q) || throw(ArgumentError("$(side)_target_discharge must be finite"))
    end
    if model.wetdry !== nothing
        model.wetdry.char_right.target_discharge = right_q === nothing ? 0.0 : Float64(right_q)
        model.wetdry.char_left.target_discharge = left_q === nothing ? 0.0 : Float64(left_q)
    end
    return nothing
end

# Incoming waves of the free faces (issue #86): eta_in(t) (m) at the face, imposed while the outgoing waves radiate
# (`_radiation_face_dirichlet`, `_wd_characteristic_dirichlet!`, `_add_incoming_pressure!`).
function _set_incoming_levels!(model::Model, right_f, right_free::Bool, left_f, left_free::Bool)
    for (f, free, side) in ((right_f, right_free, "right"), (left_f, left_free, "left"))
        f === nothing && continue
        model.wetdry === nothing && throw(ArgumentError("$(side)_incoming_level needs wetting_drying=true"))
        free || throw(ArgumentError("$(side)_incoming_level needs a free face ($(side)_free=true)"))
        (model.config.open_face_scheme == "characteristic" || model.config.radiation_theta) ||
            throw(ArgumentError("$(side)_incoming_level needs radiation_theta=true (or open_face_scheme=\"characteristic\")"))
        isfinite(f(model.wetdry.time)) || throw(ArgumentError("$(side)_incoming_level must be finite"))
    end
    if model.wetdry !== nothing
        model.wetdry.char_right.incoming = right_f
        model.wetdry.char_left.incoming = left_f
    end
    return nothing
end

function _set_right_boundary!(model::Model, open_right::Bool, right_profile, right_level, right_free::Bool,
        right_relaxation_time)
    relaxed = right_relaxation_time !== nothing
    if relaxed
        (right_free && right_level !== nothing) ||
            throw(ArgumentError("right_relaxation_time needs right_free=true and the target level right_level"))
        right_relaxation_time > 0 || throw(ArgumentError("right_relaxation_time must be > 0 (Inf: no relaxation)"))
    end
    if right_level !== nothing
        right_profile === nothing || throw(ArgumentError("right_profile and right_level exclude each other"))
        right_free && !relaxed && throw(ArgumentError("right_level and right_free exclude each other (unless right_relaxation_time is given: relaxed free outflow)"))
        isfinite(right_level) || throw(ArgumentError("right_level must be finite"))
    end
    right_free && right_profile !== nothing && throw(ArgumentError("right_profile and right_free exclude each other"))
    (right_level !== nothing || right_free) && _check_right_level_config(model.config)
    right_free && model.config.nx < 2 && throw(ArgumentError("right_free needs nx >= 2 (the slope of face nx)"))
    model.x_boundary_u_right_open = right_level === nothing && !right_free && open_right
    model.x_boundary_u_right_profile = right_profile === nothing ? nothing : collect(right_profile)
    # relaxed free outflow: a free outflow (x_boundary_eta_right stays nothing) with the target level
    model.x_boundary_eta_right = (right_level === nothing || relaxed) ? nothing : Float64(right_level)
    model.x_boundary_right_free = right_free
    model.x_boundary_eta_target = relaxed ? Float64(right_level) : 0.0
    model.x_boundary_relaxation_time = relaxed ? Float64(right_relaxation_time) : Inf
    right_free && (model.x_boundary_eta_free = _extrapolated_right_level(model.state.eta))
    return nothing
end

# Configuration needed by a right boundary face with its own momentum equation (water level or free outflow).
function _check_right_level_config(cfg::ModelConfig)
    cfg.periodic_x && throw(ArgumentError("a water-level boundary or free outflow (right_level, right_free) is not possible with periodic_x=true"))
    # Non-hydrostatic mode (issue #49): dq/dx = 0 on the open face, so the pressure problem keeps the boundary
    # conditions of a closed basin and any pressure solver works; the boundary face gets no q correction.
    # WHY not q = 0 on the face: a dispersive wave (Gaussian of width 1 m in 1 m depth) leaving through the free
    # outflow reflected 4.4 % with q = 0 and 1.3 % with dq/dx = 0 (fixed grid, nonhydro_surface_dirichlet=true; with
    # the Neumann top both 0.2 %, the wave is then not dispersive); see also _wd_nonhydrostatic_correction!.
    # WHY implicit vertical terms: the boundary face needs its own vertical viscosity and friction (the explicit
    # diffusion_term! leaves the boundary points out); its implicit column provides them
    cfg.implicit_vertical_terms || throw(ArgumentError("a water-level boundary or free outflow (right_level, right_free) requires implicit_vertical_terms=true"))
    return nothing
end

function _check_left_level_config(cfg::ModelConfig)
    # WHY: the level of the inlet face only enters through its layer thicknesses; the fixed grid has dz in every layer
    (cfg.wetting_drying || use_variable_top_layer(cfg)) ||
        throw(ArgumentError("left_level needs the variable top layer or wetting_drying=true (the fixed grid has the depth H at every face)"))
    return nothing
end

# Whether the right boundary face has its own momentum equation (water-level boundary or free outflow).
_right_face_computed(model::Model) = model.x_boundary_eta_right !== nothing || model.x_boundary_right_free

function clear_x_inflow_outflow!(model::Model)
    model.wetdry === nothing || (model.wetdry.left_velocity = nothing)
    model.x_boundary_u_left_profile = nothing
    model.x_boundary_q_left = nothing
    model.x_boundary_eta_left = nothing
    model.x_boundary_rho_left = nothing
    if model.wetdry !== nothing
        model.wetdry.left_k = nothing
        model.wetdry.left_eps = nothing
        model.wetdry.left_open = nothing
    end
    model.x_boundary_u_right_open = false
    model.x_boundary_u_right_profile = nothing
    model.x_boundary_eta_right = nothing
    model.x_boundary_right_free = false
    model.x_boundary_relaxation_time = Inf
    return model
end

function _check_model_config(cfg::ModelConfig)
    if !(THETA_MIN <= cfg.theta <= THETA_MAX)
        throw(ArgumentError("theta must be in [$(THETA_MIN), $(THETA_MAX)] (theta < 1/2 is unstable), got $(cfg.theta)"))
    end
    cfg.advective_courant_warning > 0.0 ||
        throw(ArgumentError("advective_courant_warning must be > 0 (Inf: no warning), got $(cfg.advective_courant_warning)"))
    cfg.wetting_drying && _check_wetting_drying_config(cfg)
    cfg.nonhydro_open_face in ("neumann", "radiation") ||
        throw(ArgumentError("nonhydro_open_face must be \"neumann\" or \"radiation\", got \"$(cfg.nonhydro_open_face)\""))
    cfg.nonhydro_open_face == "radiation" && !cfg.wetting_drying &&
        throw(ArgumentError("nonhydro_open_face=\"radiation\" is implemented for wetting_drying=true only"))
    (isfinite(cfg.radiation_period) && cfg.radiation_period >= 0.0) ||
        throw(ArgumentError("radiation_period must be >= 0 (0: long-wave speed), got $(cfg.radiation_period)"))
    # WHY: with dq/dx = 0 at the face (or without a non-hydrostatic pressure) the face has the long-wave impedance, and a
    # level radiating with the dispersive phase speed reflects more than the long-wave speed (measured: 15.6 % instead of
    # 6.9 % at k h = 1.2; docs/numerics_open_boundaries.md, section 7)
    cfg.radiation_period > 0.0 && !(cfg.nonhydro_open_face == "radiation" && !cfg.use_hydrostatic) &&
        throw(ArgumentError("radiation_period > 0 needs use_hydrostatic=false and nonhydro_open_face=\"radiation\""))
    (isfinite(cfg.radiation_speed_factor) && RADIATION_SPEED_FACTOR_RANGE[1] <= cfg.radiation_speed_factor <= RADIATION_SPEED_FACTOR_RANGE[2]) ||
        throw(ArgumentError("radiation_speed_factor must lie in $(RADIATION_SPEED_FACTOR_RANGE), got $(cfg.radiation_speed_factor)"))
    cfg.open_face_relaxation in ("level", "incoming") ||
        throw(ArgumentError("open_face_relaxation must be \"level\" or \"incoming\", got \"$(cfg.open_face_relaxation)\""))
    cfg.open_face_relaxation == "incoming" && !cfg.wetting_drying &&
        throw(ArgumentError("open_face_relaxation=\"incoming\" is implemented for wetting_drying=true only"))
    cfg.open_face_scheme in ("radiation", "characteristic") ||
        throw(ArgumentError("open_face_scheme must be \"radiation\" or \"characteristic\", got \"$(cfg.open_face_scheme)\""))
    cfg.open_face_scheme == "characteristic" && !cfg.wetting_drying &&
        throw(ArgumentError("open_face_scheme=\"characteristic\" is implemented for wetting_drying=true only"))
    (isfinite(cfg.open_face_beta) && -1.0 <= cfg.open_face_beta <= 0.0) ||
        throw(ArgumentError("open_face_beta must lie in [-1, 0], got $(cfg.open_face_beta)"))
    (isfinite(cfg.open_face_wet_depth) && cfg.open_face_wet_depth >= 0.0) ||
        throw(ArgumentError("open_face_wet_depth must be >= 0 (m), got $(cfg.open_face_wet_depth)"))
    if cfg.periodic_x && use_variable_top_layer(cfg)
        throw(ArgumentError("periodic_x=true is not supported with the variable top layer"))
    end
    if cfg.variable_top_layer === true && !cfg.use_hydrostatic
        throw(ArgumentError("variable_top_layer=true is implemented for the hydrostatic mode only (use_hydrostatic=true): the non-hydrostatic pressure projection assumes a fixed geometry"))
    end
    if !(0.0 < cfg.top_layer_min_fraction < 1.0)
        throw(ArgumentError("top_layer_min_fraction must be in (0, 1)"))
    end
    if !(cfg.turbulence_closure in ("none", "mixing_length", "k_epsilon"))
        throw(ArgumentError("turbulence_closure must be \"none\", \"mixing_length\" or \"k_epsilon\", got \"$(cfg.turbulence_closure)\""))
    end
    if cfg.bed_roughness_length < 0.0 || cfg.bed_roughness_length >= 0.5 * cfg.depth / cfg.nz
        throw(ArgumentError("bed_roughness_length must be in [0, dz/2)"))
    end
    if (turbulence_closure(cfg) == "k_epsilon" || cfg.bed_roughness_length > 0.0) && cfg.bed_level !== nothing && !cfg.wetting_drying
        throw(ArgumentError("turbulence_closure=\"k_epsilon\" and bed_roughness_length are implemented for the flat bed only (bed_level=nothing)"))
    end
    if cfg.chezy_coefficient != 0.0
        cfg.chezy_coefficient > 0.0 || throw(ArgumentError("chezy_coefficient must be >= 0, got $(cfg.chezy_coefficient)"))
        cfg.bed_roughness_length == 0.0 || throw(ArgumentError("chezy_coefficient and bed_roughness_length exclude each other (one bed friction law)"))
        turbulence_closure(cfg) == "k_epsilon" && throw(ArgumentError("turbulence_closure=\"k_epsilon\" needs a roughness length for its wall functions: use bed_roughness_length=chezy_roughness_length(C, depth) instead of chezy_coefficient"))
        cfg.bed_level === nothing || cfg.wetting_drying || throw(ArgumentError("chezy_coefficient is implemented for the flat bed (bed_level=nothing) or with wetting_drying=true"))
    end
    if !(cfg.transport_advection_scheme in ("upwind1", "tvd2"))
        throw(ArgumentError("transport_advection_scheme must be \"upwind1\" or \"tvd2\", got \"$(cfg.transport_advection_scheme)\""))
    end
    if cfg.nonhydro_surface_dirichlet
        if cfg.use_hydrostatic
            throw(ArgumentError("nonhydro_surface_dirichlet=true is implemented for the non-hydrostatic mode only (use_hydrostatic=false): the hydrostatic branch has no non-hydrostatic pressure correction"))
        end
        if cfg.pressure_solver_method != "sparse"
            throw(ArgumentError("nonhydro_surface_dirichlet=true requires pressure_solver_method=\"sparse\" (the spectral solver cannot represent a Dirichlet top condition)"))
        end
    end
    return nothing
end

function _check_wetting_drying_config(cfg::ModelConfig)
    cfg.layer_momentum_advection in ("remapped", "column") ||
        throw(ArgumentError("layer_momentum_advection must be \"remapped\" or \"column\", got \"$(cfg.layer_momentum_advection)\""))
    cfg.nonhydro_solver in ("auto", "direct", "multigrid") ||
        throw(ArgumentError("nonhydro_solver must be \"auto\", \"direct\" or \"multigrid\", got \"$(cfg.nonhydro_solver)\""))
    0.0 < cfg.nonhydro_solver_tolerance < NH_MULTIGRID_TOLERANCE_MAX ||
        throw(ArgumentError("nonhydro_solver_tolerance must be in (0, $(NH_MULTIGRID_TOLERANCE_MAX)), got $(cfg.nonhydro_solver_tolerance)"))
    cfg.nonhydro_solver_max_iterations >= 1 ||
        throw(ArgumentError("nonhydro_solver_max_iterations must be >= 1, got $(cfg.nonhydro_solver_max_iterations)"))
    cfg.periodic_x && throw(ArgumentError("wetting_drying=true does not support periodic_x=true"))
    cfg.momentum_advection_scheme in ("upwind1", "tvd2") || throw(ArgumentError("momentum_advection_scheme must be \"upwind1\" or \"tvd2\", got \"$(cfg.momentum_advection_scheme)\""))
    turbulence_closure(cfg) in ("none", "k_epsilon") || throw(ArgumentError("wetting_drying=true supports turbulence_closure=\"none\" or \"k_epsilon\""))
    turbulence_closure(cfg) == "k_epsilon" && !cfg.implicit_vertical_terms &&
        throw(ArgumentError("wetting_drying=true with turbulence_closure=\"k_epsilon\" requires implicit_vertical_terms=true (the eddy viscosity enters the implicit vertical column)"))
    (cfg.bed_roughness_length == 0.0 && cfg.chezy_coefficient == 0.0) || cfg.implicit_vertical_terms ||
        throw(ArgumentError("wetting_drying=true supports bed_roughness_length and chezy_coefficient only with implicit_vertical_terms=true (the cut layers at the bed can be arbitrarily thin)"))
    cfg.wind_stress_x == 0.0 || throw(ArgumentError("wetting_drying=true does not support wind_stress_x"))
    cfg.dry_depth_threshold > 0.0 || throw(ArgumentError("dry_depth_threshold must be > 0"))
    if cfg.bed_level !== nothing
        length(cfg.bed_level) == cfg.nx || throw(ArgumentError("bed_level must have nx = $(cfg.nx) entries, got $(length(cfg.bed_level))"))
        all(b -> isfinite(b) && b >= 0.0, cfg.bed_level) || throw(ArgumentError("bed_level must be finite and >= 0"))
    end
    return nothing
end

# Stops the run if the free surface has left the top layer (dz + eta < top_layer_min_fraction * dz): the variable top
# layer has no layer removal or wetting/drying, so continuing would divide by a vanishing or negative thickness.
function _check_top_layer_thickness(model::Model, eta::AbstractVector{<:Real})
    cfg = model.config
    eta_min = minimum(eta)
    if model.dz + eta_min < cfg.top_layer_min_fraction * model.dz
        error(@sprintf(
            "variable top layer: min(eta) = %.4e m leaves the top layer (dz = %.4e m, top_layer_min_fraction = %.2f); layer removal / wetting-drying is not supported",
            eta_min,
            model.dz,
            cfg.top_layer_min_fraction,
        ))
    end
    return nothing
end

# Applies the lateral velocity boundary conditions to `u` (state.u or a provisional u_hat, m/s, size nz x (nx+1)).
function _apply_x_boundary_conditions!(model::Model, u::AbstractMatrix{<:Real})
    if model.config.periodic_x
        @views u[:, end] .= u[:, 1]
        return nothing
    end

    if _left_face_computed(model)
        # free left face (issue #86): u of the boundary face comes from its own momentum equation
    elseif model.x_boundary_u_left_profile === nothing
        @views u[:, 1] .= 0.0
    else
        left_profile = model.x_boundary_u_left_profile
        if length(left_profile) != size(u, 1)
            throw(ArgumentError("left inflow profile length must match nz"))
        end
        @views u[:, 1] .= left_profile
        if model.x_boundary_q_left !== nothing
            # discharge boundary: scale the shape with the face layer thicknesses of level n (ws.eta_old)
            # WHY the wetting-drying branch: the step scales with the cut face layers itself
            # (_wd_boundary_velocities!), but u of the boundary face set here is u^n of the next step and carries
            # (1 - theta) of its inflow; scaled with the full layers of the flat grid the inflow over a raised bed was
            # 0.6 % too small whenever the discharge was reset every step (issue #14)
            transport = model.wetdry === nothing ? _left_face_transport(model, left_profile) :
                _wd_left_face_transport(model, left_profile)   # m^2/s per unit of the shape
            transport > 0.0 || throw(DomainError(transport, "left_discharge: the inflow face is dry"))
            # WHY the wetdry time: a function q(t) exists only in the wetting-drying mode
            @views u[:, 1] .*= _left_discharge(model, model.wetdry === nothing ? 0.0 : model.wetdry.time) / transport
        end
    end

    if _right_face_computed(model)
        # water-level boundary or free outflow: u of the boundary face comes from its own momentum equation
    elseif model.x_boundary_u_right_profile !== nothing
        right_profile = model.x_boundary_u_right_profile
        if length(right_profile) != size(u, 1)
            throw(ArgumentError("right outflow profile length must match nz"))
        end
        @views u[:, end] .= right_profile
    elseif model.x_boundary_u_right_open
        @views u[:, end] .= u[:, end - 1]
    else
        @views u[:, end] .= 0.0
    end

    return nothing
end

_apply_x_boundary_conditions!(model::Model) = _apply_x_boundary_conditions!(model, model.state.u)

# Transport (m^2/s per unit of the shape) of the profile `shape` (length nz) at the left boundary face: layers dz above the
# bed face, the top layer dz + eta with the variable top layer (eta of level n in ws.eta_old, or the prescribed inlet
# level, left_level).
function _left_face_transport(model::Model, shape::AbstractVector{<:Real})
    nz, dz = model.config.nz, model.dz
    eta_face = model.x_boundary_eta_left === nothing ? model.ws.eta_old[1] : model.x_boundary_eta_left
    h_top = use_variable_top_layer(model.config) ? dz + eta_face : dz
    kb = model.geometry.has_bed ? model.geometry.k_bed_face[1] : 1
    transport = 0.0
    for k in kb:nz
        transport += (k == nz ? h_top : dz) * shape[k]
    end
    return transport
end

# Zeroes u (m/s, nz x (nx+1)) on all faces that are not open: walls of the bed and faces inside it (variable bed).
function _mask_u!(u::AbstractMatrix{<:Real}, geo::BedGeometry)
    @inbounds for idx in eachindex(u)
        if !geo.u_open[idx]
            u[idx] = 0.0
        end
    end
    return u
end

# Zeroes w (m/s, (nz+1) x nx) on the bed face of each column and below it (variable bed).
function _mask_w!(w::AbstractMatrix{<:Real}, geo::BedGeometry)
    @inbounds for idx in eachindex(w)
        if !geo.w_open[idx]
            w[idx] = 0.0
        end
    end
    return w
end

# Smallest water depth (m) of a u-face that counts for surface_froude_number. WHY: at a wetting-drying shoreline U/sqrt(g h)
# of a film of water is meaningless for the stability of the theta scheme (the gravity-wave speed there is ~0), and the
# wet/dry front is limited by the positivity of the solve, not by the Froude bound (issue #38).
const SURFACE_FROUDE_MIN_DEPTH = 0.01
# Margin (-) by which the surface Froude number must exceed theta_froude_max before the warning. WHY: at theta = 1/2 the
# bound is 0, so the orbital velocities of any small wave (Fr ~ 1e-5, no mean flow) would warn, and runs at the bound
# (Fr = 2/3 at theta = 0.7) warn on round-off; an excess below 0.01 needs theta at most 0.005 higher (d theta / d Fr =
# 1/(2 (1 + Fr)^2) <= 1/2), the growth is negligible.
const SURFACE_FROUDE_WARNING_MARGIN = 0.01

"""
    surface_froude_number(model) -> (Fr, face)

Largest Froude number Fr = |U| / sqrt(g h) (-) of the depth-averaged flow over the u-faces, and the face where it occurs
(0 if no face counts). U (m/s) is the depth-averaged velocity and h (m) the water depth of the face:
- flat or staircase bed: h = depth - bed level of the face + eta averaged to the face, U = sum_k dZ_k u_k / sum_k dZ_k
  over the open layers (top layer dz + eta with the variable top layer, dz on the fixed grid);
- wetting and drying: the cut face layers hf of the last step, h = sum_k hf_k, U = sum_k hf_k u_k / h; faces shallower
  than SURFACE_FROUDE_MIN_DEPTH are left out.
The surface (barotropic) Froude number of the theta bound `theta_froude_max`; `froude_number` is the internal one.
PERF: O(nz nx) without allocation; evaluated after every step by the Froude-bound check.
"""
function surface_froude_number(model::Model)
    fr_max = 0.0
    face = 0
    for f in 1:(model.config.nx + 1)
        fr = _face_froude(model, f)
        if fr > fr_max   # NaN (too shallow) compares false
            fr_max = fr
            face = f
        end
    end
    return fr_max, face
end

# Surface Froude number |U| / sqrt(g h) (-) of u-face f (see surface_froude_number); NaN if the face is shallower than
# SURFACE_FROUDE_MIN_DEPTH.
function _face_froude(model::Model, f::Int)
    U, depth = _face_velocity_depth(model, f)
    depth > SURFACE_FROUDE_MIN_DEPTH || return NaN
    return abs(U) / sqrt(model.config.g * depth)
end

# Depth-averaged velocity U (m/s) and water depth h (m) of u-face f (see surface_froude_number). A boundary face with a
# known level (right_level or free outflow, left_level) has that level. h <= 0 for a dry face (U = 0 then).
function _face_velocity_depth(model::Model, f::Int)
    cfg = model.config
    st = model.state
    nz, nx, dz = cfg.nz, cfg.nx, model.dz
    wd = model.wetdry
    if wd !== nothing
        depth = 0.0       # m
        transport = 0.0   # m^2/s
        @inbounds for k in 1:nz
            depth += wd.hf[k, f]
            transport += wd.hf[k, f] * st.u[k, f]
        end
        return (depth > 0.0 ? transport / depth : 0.0), depth
    end
    geo = model.geometry
    iL = f == 1 ? (cfg.periodic_x ? nx : 1) : f - 1
    iR = f == nx + 1 ? (cfg.periodic_x ? 1 : nx) : f
    eta_f = 0.5 * (st.eta[iL] + st.eta[iR])
    if !cfg.periodic_x
        f == nx + 1 && _right_boundary_level(model) !== nothing && (eta_f = _right_boundary_level(model))
        f == 1 && model.x_boundary_eta_left !== nothing && (eta_f = model.x_boundary_eta_left)
    end
    depth = cfg.depth - geo.bed_face_level[f] + eta_f
    variable_top = use_variable_top_layer(cfg)
    kb = geo.has_bed ? geo.k_bed_face[f] : 1
    transport = 0.0
    thickness = 0.0
    @inbounds for k in kb:nz
        dZ = (k == nz && variable_top) ? dz + eta_f : dz
        transport += dZ * st.u[k, f]
        thickness += dZ
    end
    return transport / thickness, depth
end

# Margin (-) around Fr = 1 for the boundary-type warning. WHY: near-critical boundary flow fluctuates about 1 in a
# transient; only a clear mismatch (a level prescribed at an outflow with Fr > 1.05, a level at an inflow with Fr < 0.95)
# means the boundary type is wrong.
const BOUNDARY_FROUDE_MARGIN = 0.05
const RADIATION_NEWTON_ITERATIONS = 50   # -, Newton steps of the Doppler dispersion relation (_radiation_speed)
const RADIATION_NEWTON_RTOL = 1e-12      # -, relative step at which that Newton iteration stops
const RADIATION_SPEED_FACTOR_RANGE = (0.5, 2.0)   # -, admissible radiation_speed_factor (c_num/c is within a few %)
# Calibration of the model's phase speed (numerical_phase_speed_factor): standing wave in a closed basin
const PHASE_CALIBRATION_AMPLITUDE = 1e-3   # -, wave amplitude / depth (linear)
const PHASE_CALIBRATION_PERIODS = 6        # -, periods simulated; the zero crossings of all but the first are used
const PHASE_CALIBRATION_MIN_CELLS = 8      # -, fewest cells per wavelength of the calibration basin

# Warns once per model when the Froude number at an open boundary does not fit its type (issue #46): a prescribed level
# at a supercritical outflow over-specifies it (use right_free=true); a prescribed level at a subcritical inflow does too
# (prescribe the discharge only).
function _check_boundary_froude!(model::Model)
    model.boundary_froude_warned && return nothing
    message = nothing
    if model.x_boundary_eta_right !== nothing
        fr = _face_froude(model, model.config.nx + 1)
        fr > 1 + BOUNDARY_FROUDE_MARGIN && (message = @sprintf("supercritical outflow (Fr = %.2f) at a water-level boundary (right_level): the level over-specifies it; use right_free=true", fr))
    end
    if message === nothing && model.x_boundary_eta_left !== nothing
        fr = _face_froude(model, 1)
        fr < 1 - BOUNDARY_FROUDE_MARGIN && (message = @sprintf("subcritical inflow (Fr = %.2f) with a prescribed inlet level (left_level): discharge and level over-specify it; prescribe the discharge only", fr))
    end
    if message !== nothing
        model.boundary_froude_warned = true
        @warn message * ". Warned once per model."
    end
    return nothing
end

# Advective Courant number (-) of the warning (issue #61). WHY 0.8: explicit advection is stable up to 1 (section 20), but
# the jet of the runs of section 28 stays smooth only up to 0.76; from about 0.85 the peak velocity feeds itself and it
# then breaks within about 0.5 s. The warning comes before that.
const ADVECTIVE_COURANT_WARNING = 0.8

# Records the largest |u| dt/dx of the run and warns once above cfg.advective_courant_warning.
function _check_advective_courant!(model::Model)
    # PERF: O(nz nx) without allocation, as the Froude check
    courant = maximum(abs, model.state.u) * model.config.dt / model.dx
    model.max_advective_courant = max(model.max_advective_courant, courant)
    limit = model.config.advective_courant_warning
    if !model.courant_warning_issued && courant > limit
        model.courant_warning_issued = true
        @warn @sprintf("advective Courant number %.2f exceeds %.2f: explicit advection is stable up to 1. Reduce dt. Against a fixed level check the swing of the discharge (docs section 28). Warned once per model.",
            courant, limit)
    end
    return nothing
end

# Factor (-) above which the mode of the reach counts as growing (issue #61). WHY not 1: friction and the head loss at
# obstacles are not in the estimate, and a uniform channel has exactly 1.
const REACH_MODE_GAIN_WARNING = 1.1

"""
    reach_mode_gain(U_left, h_left, h_right; g) -> Float64

Factor (-) by which the quarter-wave mode of a reach changes per period between a prescribed velocity at the left end
and a prescribed level at the right end, for linear long waves on a mean flow (docs section 28):

    G = (c1 + U1) / (c1 - U1) * (c2 - U2) / (c2 + U2),   c = sqrt(g h),   U2 = U1 h1 / h2

`U_left` (m/s, > 0 is inflow) and the depths `h_left`, `h_right` (m) are of the two ends, `g` in m/s^2. U2 comes from the
continuity of the steady flow. WHY not the velocity of the outlet: in a transient it lags the inlet, and a uniform
channel would seem to grow. Friction and the head loss at obstacles are not included, so G > 1 means the mode can grow.
Without a mean flow G = 1; a prescribed discharge at the inlet gives the second factor alone. NaN unless both ends are
subcritical with inflow.
"""
function reach_mode_gain(U_left::Real, h_left::Real, h_right::Real; g::Real=9.81)
    (U_left > 0 && h_left > 0 && h_right > 0) || return NaN
    c1, c2 = sqrt(g * h_left), sqrt(g * h_right)
    U2 = U_left * h_left / h_right
    (U_left < c1 && U2 < c2) || return NaN
    return (c1 + U_left) / (c1 - U_left) * (c2 - U2) / (c2 + U2)
end

# Warns once when a prescribed velocity at the inlet against a prescribed level can feed the mode of the reach.
function _check_reach_mode_growth!(model::Model)
    model.reach_mode_warned && return nothing
    model.x_boundary_eta_right === nothing && return nothing
    model.x_boundary_q_left === nothing || return nothing
    velocity_inlet = model.wetdry === nothing ? model.x_boundary_u_left_profile !== nothing :
        (model.wetdry.left_velocity !== nothing || model.x_boundary_u_left_profile !== nothing)
    velocity_inlet || return nothing
    U1, h1 = _face_velocity_depth(model, 1)
    _, h2 = _face_velocity_depth(model, model.config.nx + 1)
    gain = reach_mode_gain(U1, h1, h2; g=model.config.g)
    if gain > REACH_MODE_GAIN_WARNING   # NaN compares false
        model.reach_mode_warned = true
        @warn @sprintf("a velocity inlet against a fixed level (right_level) can feed the mode of the reach, factor %.2f per period: its swing can grow until the run fails. Prescribe left_discharge or use the relaxed outflow (docs section 28). Warned once per model.", gain)
    end
    return nothing
end

# Records the largest surface Froude number of the run and warns once when it exceeds the theta bound by more than
# SURFACE_FROUDE_WARNING_MARGIN (issue #39).
function _check_surface_froude!(model::Model)
    fr, face = surface_froude_number(model)
    model.max_surface_froude = max(model.max_surface_froude, fr)
    theta = model.config.theta
    if !model.froude_warning_issued && fr > theta_froude_max(theta) + SURFACE_FROUDE_WARNING_MARGIN
        model.froude_warning_issued = true
        @warn @sprintf("surface Froude number %.3f (u-face %d) exceeds theta_froude_max(theta = %.3g) = %.3f: gravity waves on the mean flow can grow slowly. Use theta >= theta_for_froude(Fr) = %.3f. The bound is sharp with the variable top layer and conservative on the fixed grid (docs/TESTS_TECHNICAL_DOCUMENTATION.md, section 19). Warned once per model.",
            fr, face, theta, theta_froude_max(theta), theta_for_froude(fr))
    end
    _check_boundary_froude!(model)
    _check_advective_courant!(model)
    _check_reach_mode_growth!(model)
    return nothing
end

function compute_diagnostics(model::Model)
    cfg = model.config
    st = model.state

    cfl_x, cfl_z, cfl_total = cfl_numbers(st.u, st.w, cfg.dt, model.dx, model.dz)
    div = div_x_u(st.u, model.dx) .+ div_z_w(st.w, model.dz)

    return Dict(
        "cfl_x" => cfl_x,
        "cfl_z" => cfl_z,
        "cfl_total" => cfl_total,
        "froude" => froude_number(st.u, st.rho, cfg.rho0, cfg.g, cfg.depth),
        "surface_froude" => surface_froude_number(model)[1],
        "surface_froude_max" => model.max_surface_froude,
        "advective_courant_max" => model.max_advective_courant,
        "water_volume" => model.wetdry === nothing ? water_volume(cfg.depth, st.eta, model.dx) - sum(model.geometry.bed_level) * model.dx : sum(water_depth(model)) * model.dx,
        "constituent_mass" => _constituent_mass(model),
        "boundary_flux" => model.wetdry === nothing ?
            boundary_normal_flux(st.u, st.w, model.dx, model.dz) :
            -sum(@view model.wetdry.q[:, 1]) + sum(@view model.wetdry.q[:, end]),
        "div_max" => maximum(abs, div),
        "div_mean_abs" => mean(abs, div),
        "rho_min" => minimum(st.rho),
        "rho_max" => maximum(st.rho),
        "kinematic_fs_mismatch" => model.last_kinematic_mismatch,
        "nonhydro_pcorr_max" => model.last_nonhydro_pcorr_max,
    )
end

function _correct_constituent_mass!(model::Model)
    st = model.state
    ws = model.ws
    cell_area = model.dx * model.dz
    target = model.initial_constituent_mass

    for _ in 1:4
        current = _constituent_mass(model)
        diff = target - current
        if abs(diff) < 1e-12
            break
        end

        delta_per_cell = diff / cell_area
        if delta_per_cell > 0.0
            @. ws.slack = model.rho_max_init - st.rho
        else
            @. ws.slack = st.rho - model.rho_min_init
        end
        if model.geometry.has_bed
            ws.slack .*= model.geometry.wet_cell   # solid cells keep their (unused) density
        end

        total_slack = sum(ws.slack)
        if use_variable_top_layer(model.config)
            # WHY: the top-layer cells have the volume dx*(dz + eta), so a density change d(rho) there carries the
            # mass d(rho)*(1 + eta/dz)*cell_area; weighting the slack by the volume makes the correction add up to `diff`.
            nz_local = size(st.rho, 1)
            @inbounds for i in eachindex(st.eta)
                total_slack += ws.slack[nz_local, i] * st.eta[i] / model.dz
            end
        end
        if total_slack <= 0.0
            break
        end

        @. st.rho = st.rho + delta_per_cell * (ws.slack / total_slack)

        nz, nx = size(st.rho)
        @inbounds for i in 1:nx
            for k in 1:nz
                st.rho[k, i] = clamp(st.rho[k, i], model.rho_min_init, model.rho_max_init)
            end
        end
    end

    return nothing
end

function _apply_wind_surface_density_mixing!(rho::AbstractMatrix{<:Real}, cfg::ModelConfig, dz::Real)
    if cfg.wind_stress_x == 0.0 || cfg.wind_mixing_diffusivity <= 0.0 || size(rho, 1) < 2
        return nothing
    end

    # Conservative two-cell exchange at the free surface to mimic wind-driven entrainment.
    gamma = cfg.wind_mixing_diffusivity * cfg.dt / (dz * dz)
    gamma = min(max(gamma, 0.0), 0.5)
    if gamma == 0.0
        return nothing
    end

    top = size(rho, 1)
    below = top - 1
    nx = size(rho, 2)
    @inbounds for i in 1:nx
        delta = rho[top, i] - rho[below, i]
        mix_flux = gamma * delta
        rho[top, i] -= mix_flux
        rho[below, i] += mix_flux
    end

    return nothing
end

function pressure_buoyancy_components(model::Model)
    st = model.state
    cfg = model.config

    if model.wetdry !== nothing
        return copy(model.last_p_hydro), copy(model.last_p_corr), copy(model.last_buoy_w)
    end

    p_hydro = solve_hydrostatic_pressure(st.p, st.rho, cfg.g, model.dz; top_half_layer=use_variable_top_layer(cfg))
    p_corr = cfg.use_hydrostatic ? zeros(size(st.p)) : (st.p .- p_hydro)

    rho_w = 0.5 .* (st.rho[2:end, :] .+ st.rho[1:end-1, :])
    buoy_w = buoyancy(rho_w, model.rho_0_w_2d)

    return p_hydro, p_corr, buoy_w
end

function _compute_effective_viscosity_u!(nu_eff_u::AbstractMatrix{<:Real}, model::Model)
    cfg = model.config
    st = model.state
    nz, nxp1 = size(st.u)

    fill!(nu_eff_u, cfg.nu)

    if model.turbulence !== nothing
        # WHY no cap (max_turbulent_viscosity belongs to the mixing-length model): k-epsilon sets nu_t itself
        turbulent_viscosity_u!(nu_eff_u, model.turbulence; periodic_x=cfg.periodic_x)
        @. nu_eff_u += cfg.nu
        return nu_eff_u
    end

    if turbulence_closure(cfg) == "mixing_length" && cfg.turbulence_mixing_length > 0.0
        ml2 = cfg.turbulence_mixing_length * cfg.turbulence_mixing_length
        # WHY rows 1 and nz (one-sided shear): the flux-form diffusion_term! couples the top and bottom rows to their
        # neighbours with the face viscosity 0.5*(nu[k] + nu[k +- 1]); without a turbulent value there, the coupling of
        # a wind-driven top row would be halved.
        @inbounds for i in 1:nxp1
            for k in 1:nz
                k_up = min(k + 1, nz)
                k_down = max(k - 1, 1)
                shear = abs((st.u[k_up, i] - st.u[k_down, i]) / ((k_up - k_down) * model.dz))
                nu_eff_u[k, i] += ml2 * shear
            end
        end
    end

    max_nu = cfg.nu + max(cfg.max_turbulent_viscosity, 0.0)
    @inbounds for idx in eachindex(nu_eff_u)
        nu_eff_u[idx] = min(nu_eff_u[idx], max_nu)
    end

    return nu_eff_u
end

# Surface slope d(eta)/dx (1) at the u-faces: grad[i] between cells i and i+1 (i = 1 ... nx-1), or with periodic_x
# grad[i] at face i (i = 1 ... nx+1, face nx+1 = face 1).
function _surface_gradient!(grad::AbstractVector{<:Real}, eta::AbstractVector{<:Real}, dx::Real, periodic_x::Bool)
    nx = length(eta)
    if periodic_x
        grad[1] = (eta[1] - eta[end]) / dx
        @inbounds for i in 2:nx
            grad[i] = (eta[i] - eta[i - 1]) / dx
        end
        grad[nx + 1] = grad[1]
    else
        @inbounds for i in 1:(nx - 1)
            grad[i] = (eta[i + 1] - eta[i]) / dx
        end
    end
    return grad
end

# Surface slope (1) at the right boundary face of a water-level boundary: the level eta_right (m) lies on the face itself,
# half a cell from the centre of the last cell. WHY at the face (not in a ghost cell at L + dx/2): eta_right is then the
# level at the end of the domain, x = L, as a measured or normal-depth level is specified.
_right_level_slope(eta::AbstractVector{<:Real}, eta_right::Real, dx::Real) = (eta_right - eta[end]) / (0.5 * dx)

# Level (m) of the right boundary face at time n: the prescribed water level, the radiated level of a free outflow, or
# nothing (layer thicknesses of the face, explicit part of its surface slope).
function _right_boundary_level(model::Model)
    model.x_boundary_eta_right === nothing || return model.x_boundary_eta_right
    return model.x_boundary_right_free ? model.x_boundary_eta_free : nothing
end

# Level eta_eff (m) and weight w (-) of the boundary face in the free-surface solve: the slope of time n + 1 at the face is
# w (eta_eff - eta_nx^{n+1}) / (dx/2), a Dirichlet condition with the coefficient w 2 theta^2 dt^2 g H / dx^2. Prescribed
# level: (eta_R, 1). Free outflow: from _update_free_outflow_level!.
function _right_boundary_dirichlet(model::Model)
    model.x_boundary_eta_right === nothing || return (Float64(model.x_boundary_eta_right), 1.0)
    return model.x_boundary_free_dirichlet
end

# Surface slope (1) at the right boundary face with its own momentum equation: time n (explicit part; old=true) with its
# level of time n, or time n + 1 (implicit part; old=false) from the solved level eta_nx^{n+1} (_right_boundary_dirichlet).
function _right_boundary_slope(eta::AbstractVector{<:Real}, model::Model, dx::Real; old::Bool=false)
    old && return _right_level_slope(eta, _right_boundary_level(model), dx)
    eta_eff, weight = _right_boundary_dirichlet(model)
    return weight * _right_level_slope(eta, eta_eff, dx)
end

# Level (m) at x = L extrapolated linearly from the last two cell centres, eta_nx + (eta_nx - eta_{nx-1}) / 2.
# WHY linear: uniform flow over a real bed slope has a linear surface; a constant extrapolation would put the level of
# the last centre at x = L and take the pressure force off the boundary face.
_extrapolated_right_level(eta::AbstractVector{<:Real}) = 1.5 * eta[end] - 0.5 * eta[end - 1]

"""
    _update_free_outflow_level!(model)

Free outflow (issue #46): advances the level eta_b (m) of the right boundary face, which then acts as the level of a
water-level boundary for this step (`_right_boundary_level`), from the state of time n. U (m/s) and h (m) are the
depth-averaged velocity and the depth of the boundary face.

- Supercritical outflow (U >= sqrt(g h)): both characteristics leave the domain, the level comes from the interior,
  eta_b = 1.5 eta_nx - 0.5 eta_{nx-1} (linear extrapolation to x = L).
- Subcritical outflow: outgoing-wave (Sommerfeld) radiation condition with the speed c = U + sqrt(g h) of the outgoing
  long-wave characteristic, on the deviation from a surface parallel to the bed,
      deta_b/dt + c [(eta_b - eta_nx)/(dx/2) - s_bed] = 0,   s_bed = (z_b,nx - z_b,nx-1)/dx,
  backward Euler, eta_b^{n+1} = (eta_b^n + mu (eta_nx^{n+1} + s_bed dx/2)) / (1 + mu), mu = c dt/(dx/2) (a convex
  combination, stable for any dt). Implicit in the free-surface solve: the slope of the boundary face at n + 1 is
  (eta_b^{n+1} - eta_nx^{n+1})/(dx/2) = w (eta_eff - eta_nx^{n+1})/(dx/2) with w = 1/(1 + mu), eta_eff = eta_b^n + mu s_bed dx/2,
  i.e. the Dirichlet condition of a water-level boundary with the coefficient w (`_right_boundary_dirichlet`);
  `_finish_free_outflow_level!` then sets eta_b^{n+1}. WHY implicit: with eta_nx of time n the lag of one step (c dt = dx
  at the gravity-wave Courant number 1) reflected 41 % of an outgoing wave at every resolution.
  WHY the gradient to the last cell centre: a wave leaving the domain then carries its level
  out, eta_b follows it with the lag of half a cell; relaxing eta_b to the level extrapolated to the face itself instead
  gives the boundary face the slope of face nx, so it moves exactly as face nx, the last cell keeps its level and the
  wave is reflected as at a fixed level (measured: the same reflection as `right_level`). WHY s_bed: uniform flow has a
  surface parallel to the bed, also at the outlet (exact with a real bed slope); without it the boundary face would lose
  its pressure force there. Other steady outflows (e.g. a backwater curve at the outlet) are pulled towards the
  bed-parallel continuation: prescribe the level (`right_level`) when the outlet level is known.
- Relaxed free outflow (issue #54, `right_relaxation_time` tau): the subcritical radiation condition gains the
  relaxation to the target level eta_R,
      deta_b/dt + c [(eta_b - eta_nx)/(dx/2) - s_bed] = -(eta_b - eta_R)/tau,
  backward Euler with r = dt/tau: eta_b^{n+1} (1 + mu + r) = eta_b^n + mu (eta_nx^{n+1} + s_bed dx/2) + r eta_R, the same
  Dirichlet form with w = (1 + r)/(1 + mu + r) and eta_eff = (eta_b^n + mu s_bed dx/2 + r eta_R)/(1 + r) (r = 0: the
  plain free outflow, bit for bit). For a wave exp(i(k x - omega t)) with c = omega/k the continuous condition reflects
  R = -1/(1 - 2 i omega tau) (Blayo & Debreu 2005): periods much longer than tau are imposed, shorter ones radiated. In
  a basin of length L closed at x = 0 a level offset decays as exp(-t/tau) for tau >> L/c (lowest mode,
  eta_t + (L/c) eta_tt = -eta/tau). Supercritical outflow ignores the relaxation (no information enters).

The mean level is not fixed by the boundary: with the fixed grid (depth H at every face, variable_top_layer=false) and no
other level control it is a neutral mode. Source: Orlanski (1976), J. Comput. Phys. 21, 251-269 (radiation condition;
equation numbers not checked); relaxation: Chapman (1985), J. Phys. Oceanogr. 15, 1060-1075; Marchesiello, McWilliams
& Shchepetkin (2001), Ocean Modelling 3, 1-20; Blayo & Debreu (2005), Ocean Modelling 9, 231-252.
"""
function _update_free_outflow_level!(model::Model)
    cfg = model.config
    nx = cfg.nx
    eta = model.state.eta
    U, depth = _face_velocity_depth(model, nx + 1)
    U_rad = _update_radiation_current!(model, U)   # m/s, current of the radiation speed (U itself for period 0)
    celerity = depth > 0.0 ? sqrt(cfg.g * depth) : 0.0   # m/s, long-wave speed (0 for a dry face)
    _prepare_characteristic_face!(model, :right, U >= celerity, U_rad, depth)
    if U >= celerity
        # supercritical: the level of the face is the interior one, extrapolated (time n)
        model.x_boundary_eta_free = _extrapolated_right_level(eta)
        model.x_boundary_free_dirichlet = (model.x_boundary_eta_free, 1.0)
        return nothing
    end
    bed = model.wetdry === nothing ? model.geometry.bed_level : model.wetdry.bed
    s_bed = (bed[nx] - bed[nx - 1]) / model.dx   # -, slope of a bed-parallel surface (eta rises with the bed)
    # radiation speed: U + sqrt(g h), or with radiation_period > 0 the phase speed of that period (issue #86; the
    # same speed as the pressure radiation of the face, so that the hydrostatic and the non-hydrostatic part of the
    # outgoing wave leave together)
    char = model.wetdry === nothing ? nothing : model.wetdry.char_right
    model.x_boundary_free_dirichlet = _radiation_face_dirichlet(model, char, model.x_boundary_eta_free, eta[nx], s_bed,
        U * depth, U_rad, depth, model.x_boundary_relaxation_time, model.x_boundary_eta_target, 1.0)
    return nothing
end

"""
    _radiation_face_dirichlet(model, char, eta_b, eta_N, s_bed, Q_out, U_out, depth, tau, eta_T, n) -> (eta_eff, w)

Dirichlet form of the radiation condition of a free face (right n = 1, left n = -1) at time n: face level eta_b (m),
adjacent cell level eta_N (m), bed-parallel slope s_bed (-), outward discharge Q_out = Q_n^n (m^2/s), current U_out (m/s)
in the direction of the outgoing wave, face depth (m), relaxation time tau (s; Inf: none) and target level eta_T (m).
The radiation speed is c_out = `_radiation_speed`(U_out) (mu = c_out dt/(dx/2)).
- Without incoming wave and with `open_face_relaxation="level"`: `_radiation_dirichlet` as before (bit for bit).
- Incoming wave eta_in(t) (`right_incoming_level` / `left_incoming_level`, issue #86): the face radiates the
  deviation from it, eta - eta_in, at the face and at the adjacent cell centre (`_prepare_incoming!`), so that it imposes
  the incoming wave and radiates the rest; a relaxed level then relaxes towards eta_T + eta_in.
- `open_face_relaxation="incoming"`: instead of the level, the incoming characteristic variable w_in = c_out eta - Q_n
  is relaxed, explicitly: eta_eff -= (dt/tau) (w_in^n - w_T)/c_out with w_T = c_out eta_T - Q_T,n + (c_out + c_in)
  eta_in (Q_T: `right_target_discharge` / `left_target_discharge`). An outgoing wave has Q_n' = c_out eta' (with
  c_out the model's phase speed, `radiation_speed_factor`), so w_in and the relaxation do not see it
  (docs/numerics_open_boundaries.md section 8.1).
"""
function _radiation_face_dirichlet(model::Model, char::Union{Nothing,CharacteristicFaceState}, eta_b::Real, eta_N::Real,
        s_bed::Real, Q_out::Real, U_out::Real, depth::Real, tau::Real, eta_T::Real, n::Real)
    cfg = model.config
    half = 0.5 * model.dx
    c_out = max(_radiation_speed(cfg, U_out, depth), 0.0)   # >= 0 (U > -celerity: not a supercritical inflow)
    mu = _level_radiation_speed(cfg, c_out, U_out) * cfg.dt / half
    incoming_relax = cfg.open_face_relaxation == "incoming"
    r = incoming_relax ? 0.0 : cfg.dt / tau   # -, relaxation of the level (0: none)
    (char === nothing || (char.incoming === nothing && !incoming_relax)) &&
        return _radiation_dirichlet(cfg, eta_b, eta_N, s_bed, half, mu, r, eta_T)
    c_in = _radiation_speed(cfg, -U_out, depth)
    char.u_out = U_out
    _prepare_incoming!(model, char, c_in)
    eta_eff, weight = _radiation_dirichlet(cfg, eta_b, eta_N, s_bed, half, mu, r, eta_T;
        e_b_old=char.e_b_old, e_b_new=char.e_b_new, e_n_old=char.e_n_old, e_n_new=char.e_n_new)
    if incoming_relax && isfinite(tau) && c_out > 0.0
        w_target = c_out * eta_T - n * char.target_discharge + (c_out + c_in) * char.e_b_old
        eta_eff -= (cfg.dt / tau) * (c_out * eta_b - Q_out - w_target) / c_out
    end
    return eta_eff, weight
end

"""
    _level_radiation_speed(cfg, c_out, U_out) -> m/s

Speed with which the level (and the non-hydrostatic pressure) of a free face radiates, from the absolute speed c_out
(m/s) of the outgoing wave and the current U_out (m/s) in its direction. With `radiation_theta=true` and a current that
enters through the face (U_out < 0): the intrinsic speed c_out - U_out. WHY: the face momentum has no advection then
(the outer face carries the zero-gradient flux of the inflow), so the face does not carry the wave with the current;
the characteristic form of the face (docs/numerics_open_boundaries.md section 8) gives the intrinsic speed for that
face momentum. Measured with a current of 0.16 m/s entering (k h = 0.3, hydrostatic): 4.0 % reflection with U + c,
0.22 % with c. Otherwise c_out (the former speed; outflow: the face momentum advects with the current).
"""
_level_radiation_speed(cfg::ModelConfig, c_out::Real, U_out::Real) =
    (cfg.radiation_theta && U_out < 0.0) ? c_out - U_out : c_out

"""
    _prepare_incoming!(model, char, c_in)

Incoming wave of an open face for this step (issue #86): eta_in (m) at the face at the start and the end of the step,
and at the adjacent cell centre, where the incoming wave of speed c_in (m/s, absolute, into the domain) arrives
(dx/2)/c_in later: eta_in(t - (dx/2)/c_in). The wave number of the incoming design wave, k_in = omega/c_in (rad/m;
0 without `radiation_period`), gives its non-hydrostatic pressure (`_incoming_pressure_profile`). Zero without an
incoming wave.
"""
function _prepare_incoming!(model::Model, char::CharacteristicFaceState, c_in::Real)
    cfg = model.config
    f = char.incoming
    if f === nothing
        char.e_b_old = char.e_b_new = char.e_n_old = char.e_n_new = char.e_g = char.k_in = 0.0
        return nothing
    end
    t, dt = model.wetdry.time, cfg.dt
    lag = c_in > 0.0 ? 0.5 * model.dx / c_in : 0.0   # s
    char.e_b_old, char.e_b_new = f(t), f(t + dt)
    char.e_n_old, char.e_n_new = f(t - lag), f(t + dt - lag)
    char.e_g = f(t + 2.0 * lag)   # m, one cell outside (the ghost node of the inflow advection), time n
    char.c_in = c_in
    all(isfinite, (char.e_b_old, char.e_b_new, char.e_n_old, char.e_n_new)) ||
        throw(DomainError(t, "the incoming level of an open face is not finite at t = $t s"))
    char.k_in = (cfg.radiation_period > 0.0 && c_in > 0.0) ? 2pi / cfg.radiation_period / c_in : 0.0
    return nothing
end

"""
    _radiation_dirichlet(cfg, eta_b, eta_in, s_bed, half, mu, r, eta_target) -> (eta_eff, w)

Dirichlet form of the radiation condition of an open face (both ends, issues #46, #54, #86): the face level eta_b (m) at
half = dx/2 (m) from the adjacent cell centre with level eta_in (m), the bed-parallel slope s_bed (-, towards the face),
mu = c dt/(dx/2) (-) and r = dt/tau (-) of the relaxation to eta_target (m):
- backward Euler: eta_b^{n+1} (1 + mu + r) = eta_b^n + mu (eta_in^{n+1} + s_bed dx/2) + r eta_target;
- theta-weighted (`radiation_theta`, issue #86): eta_b^{n+1} - eta_b^n = -mu [theta (eta_b - eta_in - s_bed dx/2)^{n+1}
  + (1 - theta) (eta_b - eta_in - s_bed dx/2)^n] - r (eta_b^{n+1} - eta_target).
Both read eta_b^{n+1} = eta_in^{n+1} + w (eta_eff - eta_in^{n+1}) with w = (1 + r)/(1 + mu + r), resp.
(1 + r)/(1 + theta mu + r) > 0 (`_right_boundary_dirichlet`). WHY theta: the interior and the face momentum act at
n + theta; a radiation condition at n + 1 shifts the boundary level by (1 - theta) omega dt and reflects
|R| ~ (1 - theta)/2 Cr k dx; weighted like the interior it reflects tan^2(k dx/4) (docs/numerics_open_boundaries.md,
section 5). The backward-Euler arithmetic is that of issues #46/#54 term for term (bit for bit).
"""
function _radiation_dirichlet(cfg::ModelConfig, eta_b::Real, eta_in::Real, s_bed::Real, half::Real, mu::Real, r::Real,
        eta_target::Real; e_b_old::Real=0.0, e_b_new::Real=0.0, e_n_old::Real=0.0, e_n_new::Real=0.0)
    if cfg.radiation_theta
        th = cfg.theta
        if e_b_old == e_b_new == e_n_old == e_n_new == 0.0
            explicit = eta_b - (1.0 - th) * mu * (eta_b - eta_in - s_bed * half)
            return ((explicit + th * mu * s_bed * half + r * eta_target) / (1.0 + r), (1.0 + r) / (1.0 + th * mu + r))
        end
        # incoming wave (issue #86): the radiation condition for the deviation eta - eta_in, with eta_in = e_b at the face
        # and e_n at the cell centre; the level relaxes towards eta_target + e_b
        explicit = eta_b + (e_b_new - e_b_old) - (1.0 - th) * mu * ((eta_b - e_b_old) - (eta_in - e_n_old) - s_bed * half) +
                   th * mu * (e_b_new - e_n_new)
        return ((explicit + th * mu * s_bed * half + r * (eta_target + e_b_new)) / (1.0 + r), (1.0 + r) / (1.0 + th * mu + r))
    end
    (e_b_old == e_b_new == e_n_old == e_n_new == 0.0) ||
        throw(ArgumentError("an incoming wave at an open face needs radiation_theta=true"))
    return ((eta_b + mu * s_bed * half + r * eta_target) / (1.0 + r), (1.0 + r) / (1.0 + mu + r))
end

"""
    _update_left_free_level!(model)

Free (radiating) left face of the wetting-drying step (`set_x_inflow_outflow!(...; left_free=true)`, issue #86): the
mirror of `_update_free_outflow_level!`. The outgoing wave travels in -x with the speed c = -U + sqrt(g h), or with
`radiation_period` the phase speed of that period on the current -U (U the depth-mean face velocity, positive into the
domain; filtered over the period). Supercritical outflow to the left (-U >= sqrt(g h)): the level extrapolated linearly
to x = 0. Otherwise the radiation condition of `_radiation_dirichlet` towards the first cell centre, with the bed-parallel
slope s_bed = (z_b,1 - z_b,2)/dx and the relaxation to the target level of a relaxed free face.
"""
function _update_left_free_level!(model::Model)
    cfg = model.config
    face = model.wetdry.left_open
    eta = model.state.eta
    U, depth = _face_velocity_depth(model, 1)
    U_out = -U   # m/s, velocity in the direction of the outgoing wave
    if cfg.radiation_period > 0.0
        r_f = cfg.dt / cfg.radiation_period
        face.u_mean = isnan(face.u_mean) ? U_out : (face.u_mean + r_f * U_out) / (1.0 + r_f)
        U_out = face.u_mean
    end
    celerity = depth > 0.0 ? sqrt(cfg.g * depth) : 0.0
    _prepare_characteristic_face!(model, :left, -U >= celerity, U_out, depth)
    if -U >= celerity
        face.eta_b = _extrapolated_left_level(eta)
        face.eta_eff, face.weight = face.eta_b, 1.0
        return nothing
    end
    bed = model.wetdry.bed
    s_bed = (bed[1] - bed[2]) / model.dx   # -, slope of a bed-parallel surface towards x = 0
    face.eta_eff, face.weight = _radiation_face_dirichlet(model, model.wetdry.char_left, face.eta_b, eta[1], s_bed,
        -U * depth, U_out, depth, face.relaxation_time, face.eta_target, -1.0)
    return nothing
end

# Free left face after the free-surface solve: eta_b^{n+1} = eta_1^{n+1} + w (eta_eff - eta_1^{n+1}); for a
# characteristic face with its level weight (_wd_characteristic_dirichlet!).
function _finish_left_free_level!(model::Model)
    face = model.wetdry.left_open
    eta_1 = model.state.eta[1]
    char = model.wetdry.char_left
    weight = _characteristic_active(model, char) ? char.level_weight : face.weight
    face.eta_b = eta_1 + weight * (face.eta_eff - eta_1)
    return nothing
end

"""
    _prepare_characteristic_face!(model, side, supercritical, U_out, depth)

Characteristic open face (`open_face_scheme="characteristic"`, issue #86, docs/numerics_open_boundaries.md section 8),
before the step: whether the face is active (subcritical; a supercritical outflow keeps the extrapolated level of the
radiation scheme), and the absolute speeds of its outgoing and incoming waves, c_out = `_radiation_speed`(U_out) and
c_in = `_radiation_speed`(-U_out) (m/s): U_out (m/s) is the (filtered) current in the direction of the outgoing wave,
depth (m) that of the face. Long waves: c_out = U_out + sqrt(g h), c_in = sqrt(g h) - U_out; with `radiation_period`
the phase speeds of that period. The Dirichlet form of the face is set in the free-surface solve
(`_wd_characteristic_dirichlet!`), since it needs the predictor of the step.
"""
function _prepare_characteristic_face!(model::Model, side::Symbol, supercritical::Bool, U_out::Real, depth::Real)
    cfg = model.config
    (cfg.open_face_scheme == "characteristic" && model.wetdry !== nothing) || return nothing
    char = side === :right ? model.wetdry.char_right : model.wetdry.char_left
    char.c_out = _radiation_speed(cfg, U_out, depth)
    char.c_in = _radiation_speed(cfg, -U_out, depth)
    char.u_out = U_out
    # WHY c_in > 0: the incoming characteristic must enter (subcritical); a supercritical inflow is not supported
    char.active = !supercritical && depth > 0.0 && char.c_in > 0.0
    return nothing
end

# Whether an open face uses the characteristic scheme in this step (subcritical, scheme selected).
_characteristic_active(model::Model, char::CharacteristicFaceState) =
    model.config.open_face_scheme == "characteristic" && char.active

# Level (m) at x = 0 extrapolated linearly from the first two cell centres (mirror of _extrapolated_right_level).
_extrapolated_left_level(eta::AbstractVector{<:Real}) = 1.5 * eta[1] - 0.5 * eta[2]

# Whether the left boundary face is a free (radiating) face with its own momentum equation (wetting-drying step).
_left_face_computed(model::Model) = model.wetdry !== nothing && model.wetdry.left_open !== nothing

# Surface slope (1) at the free left face: time n (old=true) from the face level of time n, or the implicit slope of
# time n + 1, w (eta_1^{n+1} - eta_eff)/(dx/2) (the mirror of _right_boundary_slope).
function _left_boundary_slope(eta::AbstractVector{<:Real}, model::Model, dx::Real; old::Bool=false)
    face = model.wetdry.left_open
    old && return (eta[1] - face.eta_b) / (0.5 * dx)
    return face.weight * (eta[1] - face.eta_eff) / (0.5 * dx)
end

"""
    _update_radiation_current!(model, U) -> U_rad (m/s)

Current that Doppler-shifts the radiation speed of the open right face (issue #86). With `radiation_period` T > 0
(wetting-drying step): the face depth-mean velocity U low-pass filtered over T, implicit,
U_rad^{n+1} = (U_rad^n + (dt/T) U)/(1 + dt/T), started at U. WHY: the instantaneous U contains the orbital velocity of the
wave itself (about +-a omega/(k h), +-0.2 m/s for Klopman's waves), which does not shift the wave's own phase speed and
would modulate it at the wave frequency. With T = 0: U (the long-wave characteristic speed U + sqrt(g h) is nonlinear in U
by design).
"""
function _update_radiation_current!(model::Model, U::Real)
    T = model.config.radiation_period
    wd = model.wetdry
    (T > 0.0 && wd !== nothing) || return U
    r = model.config.dt / T
    wd.u_right_mean = isnan(wd.u_right_mean) ? U : (wd.u_right_mean + r * U) / (1.0 + r)
    return wd.u_right_mean
end

"""
    _radiation_speed(cfg, U, depth) -> c (m/s)

Speed of the outgoing wave at the open right face for the radiation of the non-hydrostatic pressure (issue #86): the
long-wave speed U + sqrt(g h) for `radiation_period = 0`, otherwise the absolute phase speed omega/k of a wave of
period T on the depth-mean current U (m/s) in depth h (m), with k from the Doppler-shifted linear dispersion relation
(omega - k U)^2 = g k tanh(k h), omega = 2 pi / T (Newton; e.g. Jonsson 1990, eq. not checked). Valid for |U| below the
blocking velocity of the wave; for U >= sqrt(g h) the long-wave speed is returned.
"""
function _radiation_speed(cfg::ModelConfig, U::Real, depth::Real)
    depth > 0.0 || return 0.0
    c0 = sqrt(cfg.g * depth)
    factor = cfg.radiation_speed_factor   # -, c_num/c of the model (WHY == 1 apart: the former arithmetic bit for bit)
    (cfg.radiation_period > 0.0 && U < c0) || return factor == 1.0 ? U + c0 : U + factor * c0
    omega = 2pi / cfg.radiation_period
    k = omega / (U + c0)   # rad/m, long-wave start: (omega - k U)^2 = g k tanh(k h) has the root k >= this one
    for _ in 1:RADIATION_NEWTON_ITERATIONS
        sigma = omega - k * U
        f = sigma^2 - cfg.g * k * tanh(k * depth)
        df = -2.0 * U * sigma - cfg.g * tanh(k * depth) - cfg.g * k * depth / cosh(k * depth)^2
        step = f / df
        k -= step
        abs(step) <= RADIATION_NEWTON_RTOL * k && break
    end
    return factor == 1.0 ? omega / k : U + factor * (omega / k - U)
end

"""
    numerical_phase_speed_factor(cfg; current=0.0) -> f (-)

Ratio of the model's intrinsic phase speed to the linear-theory one for the wave of period `cfg.radiation_period` (s)
in depth `cfg.depth` (m) on a uniform `current` (m/s, in the direction of the wave; the absolute speed is then
U + f (c - U)), with the vertical and horizontal resolution, time step, theta and pressure options of `cfg`
(wetting-drying step). For a face that absorbs waves travelling against the current, pass that (negative) current: the
waves are shorter there and the model's dispersion error larger. Pass it as `radiation_speed_factor`: the open faces then radiate the wave that the model
carries, and their remaining reflection is no longer the interior's dispersion error (issue #86,
docs/numerics_open_boundaries.md section 7.2: at k h = 1.2 with 10 layers 1.49 % with c, 0.15 % with c_num).

Method: a standing wave eta = a cos(k x) of one wavelength in a closed basin with the grid spacing dx = length/nx of
`cfg` (the wavelength rounded to whole cells, at least PHASE_CALIBRATION_MIN_CELLS; amplitude
PHASE_CALIBRATION_AMPLITUDE times the depth, flat bed, uniform density), run over PHASE_CALIBRATION_PERIODS periods. The
period follows from the zero crossings of the level in the first column (linear interpolation between steps), and
c_num = omega_num / k is compared with sqrt(g tanh(k h)/k) at the same k. A standing wave in a closed basin is an exact
discrete mode of the interior scheme (walls at the faces), so the factor is that of a progressive wave of the same k;
it is applied as a factor on the intrinsic phase speed, also on a current. Cost: one small run (one wavelength).
"""
function numerical_phase_speed_factor(cfg::ModelConfig; current::Real=0.0)
    cfg.wetting_drying || throw(ArgumentError("numerical_phase_speed_factor needs wetting_drying=true"))
    cfg.radiation_period > 0.0 || throw(ArgumentError("numerical_phase_speed_factor needs radiation_period > 0 (the design wave)"))
    h, g = cfg.depth, cfg.g
    omega = 2pi / cfg.radiation_period
    lin = copy_config(cfg)
    lin.radiation_speed_factor = 1.0
    isfinite(current) && abs(current) < sqrt(g * h) || throw(ArgumentError("current must be subcritical, got $(current) m/s"))
    c_abs = _radiation_speed(lin, current, h)   # m/s, linear-theory absolute phase speed on the current
    k = omega / c_abs   # rad/m, its wave number (Doppler-shifted)
    dx = cfg.length / cfg.nx
    ncell = max(PHASE_CALIBRATION_MIN_CELLS, round(Int, 2pi / k / dx))
    L = ncell * dx
    kc = 2pi / L   # rad/m, wave number of the calibration basin
    cal = copy_config(lin)
    cal.nx, cal.length = ncell, L
    cal.bed_level = zeros(ncell)
    cal.write_plots = false
    cal.sponge_length = 0.0
    cal.body_force_x = 0.0
    model = Model(cal)
    fill!(model.state.rho, cal.rho0)
    refresh_density_metadata!(model)
    xc = ((1:ncell) .- 0.5) .* dx
    initialize_water_level!(model, h .+ PHASE_CALIBRATION_AMPLITUDE * h .* cos.(kc .* xc))
    set_x_inflow_outflow!(model, zeros(cal.nz); open_right=false)
    T_guess = 2pi / (kc * sqrt(g * tanh(kc * h) / kc))
    nsteps = round(Int, PHASE_CALIBRATION_PERIODS * T_guess / cal.dt)
    crossings = Float64[]
    prev = PHASE_CALIBRATION_AMPLITUDE * h
    for n in 1:nsteps
        step_forward!(model)
        cur = water_depth(model)[1] - h
        if (prev > 0.0) != (cur > 0.0)
            push!(crossings, (n - 1 + prev / (prev - cur)) * cal.dt)
        end
        prev = cur
    end
    length(crossings) >= 4 || throw(ErrorException("numerical_phase_speed_factor: too few zero crossings ($(length(crossings)))"))
    # WHY skip the first crossing: the start excites small parasitic modes that have decayed by then (theta damping)
    half_periods = diff(crossings[2:end])
    T_num = 2.0 * sum(half_periods) / length(half_periods)
    ratio = (2pi / T_num) / (kc * sqrt(g * tanh(kc * h) / kc))   # -, 1 + eps: sigma_num/sigma at the fixed k
    # WHY the conversion: the radiation condition needs the phase speed at the fixed (absolute) frequency of the design
    # wave, c_num = omega/k_num with omega - k_num U = sigma_num(k_num). With sigma_num(k) = (1 + eps) sigma_lin(k) near k
    # (the Doppler term k U taken as exact), k_num solves (omega - k U)^2 = (1 + eps)^2 g k tanh(k h) (Newton from k);
    # about 1 + eps c/c_g on still water: 1.39 eps at k h = 1.2, where the factor at the fixed k alone left 0.35 %
    # instead of 0.15 %. The factor applies to the intrinsic speed: U + f (c - U) = omega/k_num.
    U = Float64(current)
    kn = k
    for _ in 1:RADIATION_NEWTON_ITERATIONS
        sigma = omega - kn * U
        res = sigma^2 - ratio^2 * g * kn * tanh(kn * h)
        dres = -2.0 * U * sigma - ratio^2 * g * (tanh(kn * h) + kn * h / cosh(kn * h)^2)
        step = res / dres
        kn -= step
        abs(step) <= RADIATION_NEWTON_RTOL * kn && break
    end
    return (omega / kn - U) / (c_abs - U)
end

# Free outflow after the free-surface solve: the level of the boundary face at n + 1 from its Dirichlet condition,
# eta_b^{n+1} = eta_nx^{n+1} + w (eta_eff - eta_nx^{n+1}) (_update_free_outflow_level!).
function _finish_free_outflow_level!(model::Model)
    eta_eff, weight = model.x_boundary_free_dirichlet
    wd = model.wetdry
    wd !== nothing && _characteristic_active(model, wd.char_right) && (weight = wd.char_right.level_weight)
    eta_nx = model.state.eta[end]
    model.x_boundary_eta_free = eta_nx + weight * (eta_eff - eta_nx)
    return nothing
end

# u[k, f] += factor * slope (* r[k, f], the column response of the implicit vertical terms, if given) on the interior
# u-faces (all faces with periodic_x); factor in m/s per unit slope.
function _add_surface_gradient!(u::AbstractMatrix{<:Real}, grad::AbstractVector{<:Real}, factor::Real, periodic_x::Bool;
    r::Union{Nothing,AbstractMatrix{<:Real}}=nothing)
    nz, nxp1 = size(u)
    ilo, ihi = periodic_x ? (1, nxp1) : (2, nxp1 - 1)
    @inbounds for i in ilo:ihi
        grad_index = periodic_x ? i : i - 1
        for k in 1:nz
            u[k, i] += r === nothing ? factor * grad[grad_index] : factor * grad[grad_index] * r[k, i]
        end
    end
    return u
end

# Column transports U_f = sum_k dZ_k u[k, f] (m^2/s) of the u-faces: dZ = dz, and the top-layer face thickness for the
# variable top layer (faces below the bed carry u = 0).
function _face_transport!(U::AbstractVector{<:Real}, u::AbstractMatrix{<:Real}, model::Model, variable_top::Bool)
    nz = size(u, 1)
    @inbounds for f in axes(u, 2)
        s = 0.0
        for k in 1:(nz - 1)
            s += u[k, f]
        end
        top = variable_top ? model.ws.h_top_face[f] : model.dz
        U[f] = s * model.dz + top * u[nz, f]
    end
    return U
end

# Implicit vertical terms of the fixed grid (implicit_vertical_terms=true) on every interior u-face: viscosity nu_eff_u,
# upwind vertical advection in flux form with w^n averaged to the face, log-law bed friction c_b |u^n_1| (flat bed,
# z1 = dz/2) or Chezy friction g |U^n| / C^2 on the depth average. Replaces ws.u_hat by A^{-1}(dZ .* u_hat) on the open
# layers and writes the column response ws.r_vert and the effective depth ws.H_face (src/implicit_vertical.jl).
function _implicit_vertical_terms!(model::Model, variable_top::Bool)
    cfg = model.config
    st = model.state
    ws = model.ws
    geo = model.geometry
    nz, nx = cfg.nz, cfg.nx
    dz, dt = model.dz, cfg.dt
    col = ws.col
    cb = cfg.bed_roughness_length > 0.0 ? bed_friction_coefficient(cfg.bed_roughness_length, 0.5 * dz) : 0.0
    cf = cfg.chezy_coefficient > 0.0 ? cfg.g / cfg.chezy_coefficient^2 : 0.0   # -, Chezy friction coefficient g / C^2
    # the boundary face of a water-level boundary or free outflow has its own column (set_x_inflow_outflow!)
    flo, fhi = cfg.periodic_x ? (1, nx) : (2, _right_face_computed(model) ? nx + 1 : nx)
    @inbounds for f in flo:fhi
        iL = cfg.periodic_x ? mod1(f - 1, nx) : f - 1   # cells left and right of face f
        iR = cfg.periodic_x ? mod1(f, nx) : min(f, nx)  # boundary face: w of the last cell (zero gradient)
        kb = geo.has_bed ? geo.k_bed_face[f] : 1
        m = nz - kb + 1
        for j in 1:m
            k = kb + j - 1
            col[j, 3] = ws.u_hat[k, f]
            col[j, 4] = (variable_top && k == nz) ? ws.h_top_face[f] : dz
            col[j, 5] = 0.5 * (st.w[k, iL] + st.w[k, iR])   # interface below layer k
            col[j, 6] = j > 1 ? 0.5 * (ws.nu_eff_u[k - 1, f] + ws.nu_eff_u[k, f]) : 0.0
        end
        gamma = cb * abs(ws.u_old[kb, f])   # m/s
        gamma_mean = 0.0                    # m/s, Chezy: c_f |U^n| with the depth-averaged velocity of time n
        if cf > 0.0
            transport = 0.0
            depth = 0.0
            for j in 1:m
                transport += col[j, 4] * ws.u_old[kb + j - 1, f]
                depth += col[j, 4]
            end
            gamma_mean = cf * abs(transport / depth)
        end
        # WHY the variable-top-layer correction -u_top div(v) (_top_row_advective_form!, including its vertical part
        # u_top W/h_top, issue #14) stays explicit in u_hat and is not added to the top diagonal: A stays in pure flux
        # form (1'A = dZ'), so without friction the effective depth sum(dZ r) equals the water depth exactly (header of
        # src/implicit_vertical.jl). A top-diagonal term breaks that identity by O(dt W/dz) (-3 % at dt W/dz = 0.1,
        # issue #34 review). The explicit term is stable for dt |W|/h_top < 2, W ~ d(eta)/dt at the top interface: the
        # surface would have to move two top-layer thicknesses in one step, far outside the accuracy range of the
        # variable top layer (and a fall below top_layer_min_fraction * dz stops the run, _check_top_layer_thickness).
        implicit_vertical_column!(view(col, :, 1), view(col, :, 2), view(col, :, 3), view(col, :, 4), view(col, :, 5),
            view(col, :, 6), gamma, dt, m, view(col, :, 7), view(col, :, 8), view(col, :, 9); gamma_mean=gamma_mean,
            zprime=view(col, :, 11))
        H = 0.0
        for k in 1:(kb - 1)
            ws.r_vert[k, f] = 0.0
        end
        for j in 1:m
            k = kb + j - 1
            ws.u_hat[k, f] = col[j, 1]
            ws.r_vert[k, f] = col[j, 2]
            H += col[j, 4] * col[j, 2]
        end
        ws.H_face[f] = H
    end
    if cfg.periodic_x
        @views ws.u_hat[:, nx + 1] .= ws.u_hat[:, 1]
        @views ws.r_vert[:, nx + 1] .= ws.r_vert[:, 1]
        ws.H_face[nx + 1] = ws.H_face[1]
    end
    return nothing
end

# Vertical velocity w (m/s, (nz+1) x nx) of the layer continuity of u (m/s), integrated upwards from w = 0 at the bed.
# Fixed grid: every layer has the thickness dz, so w[nz+1] is the column divergence. Variable top layer: the layers below
# the top layer have the thickness dz, and the top layer closes its own budget with its face thickness of level n,
#   w[nz+1] = w[nz] - d(h_face u_top)/dx,
# which equals d(eta)/dt when u moved the surface (eta^{n+1} solves the barotropic equation with the same thicknesses).
function _integrate_w_from_continuity!(w::AbstractMatrix{<:Real}, u::AbstractMatrix{<:Real}, model::Model, variable_top::Bool)
    ws = model.ws
    geo = model.geometry
    nz, nx = model.config.nz, model.config.nx
    dz, dx = model.dz, model.dx
    div_x_u!(ws.div_u, u, dx)
    @views w[1, :] .= 0.0
    if geo.has_bed
        # w = 0 on the bed face of each column and below it
        @inbounds for i in 1:nx, k in 1:geo.k_bed[i]
            w[k, i] = 0.0
        end
    end
    @inbounds for i in 1:nx
        ktop = variable_top ? nz - 1 : nz
        for k in geo.k_bed[i]:ktop
            w[k + 1, i] = w[k, i] - dz * ws.div_u[k, i]
        end
        if variable_top
            div_hu_top = (ws.h_top_face[i + 1] * u[nz, i + 1] - ws.h_top_face[i] * u[nz, i]) / dx
            w[nz + 1, i] = w[nz, i] - div_hu_top
        end
    end
    return w
end

# Grid and mode of a model for the performance overview, e.g. "200 x 56, wetting-drying, non-hydrostatic, k-epsilon".
function _performance_label(cfg::ModelConfig)
    mode = (cfg.use_hydrostatic || cfg.nz == 1) ? "hydrostatic" : "non-hydrostatic"
    closure = turbulence_closure(cfg) == "k_epsilon" ? ", k-epsilon" : ""
    return "$(cfg.nx) x $(cfg.nz), $(cfg.wetting_drying ? "wetting-drying" : "fixed grid"), $(mode)$(closure)"
end

"""
    step_forward!(model)

Advances the model by one time step dt. The wall-clock time and the allocations of the step are added to
`model.performance` (print_performance_report).
"""
function step_forward!(model::Model)
    t0 = time_ns()
    b0 = Base.gc_bytes()
    _step_forward!(model)
    _record_step!(model.performance, time_ns() - t0, Base.gc_bytes() - b0)
    return nothing
end

function _step_forward!(model::Model)
    cfg = model.config
    st = model.state
    ws = model.ws

    _check_model_config(cfg)   # the flags are mutable after construction
    if cfg.wetting_drying != (model.wetdry !== nothing)
        throw(ArgumentError("wetting_drying cannot be changed after Model construction"))
    end
    # free outflow: radiated level of the boundary face for this step, from the state of time n
    model.x_boundary_right_free && _update_free_outflow_level!(model)
    _left_face_computed(model) && _update_left_free_level!(model)   # free left face (issue #86)
    if cfg.wetting_drying
        _step_forward_wetting_drying!(model)
        model.x_boundary_right_free && _finish_free_outflow_level!(model)
        _left_face_computed(model) && _finish_left_free_level!(model)
        _check_surface_froude!(model)
        return nothing
    end
    if (model.turbulence !== nothing) != (turbulence_closure(cfg) == "k_epsilon")
        throw(ArgumentError("turbulence_closure cannot be changed across the k_epsilon boundary after Model construction"))
    end
    if cfg.periodic_x != model.pressure_correction_solver.periodic_x
        throw(ArgumentError("periodic_x cannot be changed after Model construction"))
    end
    # WHY the type assertion: the Model field is untyped; a concrete solver type avoids boxing and dynamic dispatch
    bsolver = model.barotropic_solver::BarotropicSolver{eltype(st.eta)}
    # WHY the conversion: the solver stores theta in the state precision; Float32(0.7) != 0.7 in Float64
    if typeof(bsolver.theta)(cfg.theta) != bsolver.theta
        throw(ArgumentError("theta cannot be changed after Model construction (the free-surface matrix depends on it)"))
    end
    geo = model.geometry
    masked = geo.has_bed
    if geo.any_solid && !cfg.use_hydrostatic && cfg.pressure_solver_method != "sparse"
        throw(ArgumentError("the non-hydrostatic mode over a bed with solid cells requires pressure_solver_method=\"sparse\" (the spectral solver needs a rectangular domain)"))
    end
    if masked && model.vegetation !== nothing
        throw(ArgumentError("bed_level is not supported together with vegetation"))
    end
    variable_top = use_variable_top_layer(cfg)
    # right boundary face with its own momentum equation (set_x_inflow_outflow!): level eta_R (m), prescribed or the
    # radiated level of a free outflow
    eta_right = _right_boundary_level(model)
    open_bc = _right_face_computed(model)
    open_bc && _check_right_level_config(cfg)
    eta_left = model.x_boundary_eta_left   # m, prescribed inlet level (supercritical inflow), or nothing
    eta_left === nothing || _check_left_level_config(cfg)

    copyto!(ws.eta_old, st.eta)
    if variable_top
        _check_top_layer_thickness(model, ws.eta_old)
        top_layer_face_thickness!(ws.h_top_face, ws.eta_old, model.dz)
        # the level of a boundary face with a prescribed level is known
        for (eta_face, f, name) in ((eta_right, cfg.nx + 1, "right_level"), (eta_left, 1, "left_level"))
            eta_face === nothing && continue
            model.dz + eta_face >= cfg.top_layer_min_fraction * model.dz ||
                throw(ArgumentError("$(name) = $(eta_face) m leaves the top layer (dz = $(model.dz) m)"))
            ws.h_top_face[f] = model.dz + eta_face
        end
    end

    if masked
        _mask_u!(st.u, geo)
        _mask_w!(st.w, geo)
    end
    copyto!(ws.u_old, st.u)
    theta = cfg.theta

    implicit = cfg.implicit_vertical_terms
    # WHY top_row_advective_form=true also on the fixed grid: the grid top carries w = d(eta)/dt but no momentum flux,
    # so the flux form alone advects the top row at about (nz + 1) U (issue #37, _top_row_advective_form!)
    @timeit model.timers "step/advection_u" advection_term_u!(ws.adv_u, st.u, st.w, model.dx, model.dz, cfg.momentum_advection_scheme, ws.advection_u_ws, ws.interpolation_ws; top_row_advective_form=true, periodic_x=cfg.periodic_x, active=masked ? geo.u_active : nothing, dt=cfg.dt, vertical=!implicit, open_right=open_bc, h_top_vertical=(implicit && variable_top) ? ws.h_top_face : nothing)
    @timeit model.timers "step/advection_w" advection_term_w!(ws.adv_w, st.u, st.w, model.dx, model.dz, cfg.momentum_advection_scheme, ws.advection_w_ws, ws.interpolation_ws, cfg.periodic_x; active=masked ? geo.w_active : nothing, dt=cfg.dt)

    # density of the predictor's baroclinic pressure and buoyancy, extrapolated for theta < 1 (issue #23)
    # WHY outside @timeit: its try block would make rho_b local to it
    rho_b = _baroclinic_density(model, st.rho)
    @timeit model.timers "step/hydrostatic_predictor" begin
        solve_hydrostatic_pressure!(ws.p_internal, rho_b, cfg.g, model.dz; top_half_layer=variable_top)
        copyto!(model.last_p_hydro, ws.p_internal)
    end

    @timeit model.timers "step/u_hat_predictor" begin
        grad_x_c!(ws.grad_p_x, ws.p_internal, model.dx; periodic_x=cfg.periodic_x)
        _compute_effective_viscosity_u!(ws.nu_eff_u, model)
        diffusion_term!(ws.diff_u, st.u, ws.nu_eff_u, model.dx, model.dz; periodic_x=cfg.periodic_x, active=masked ? geo.u_active : nothing, vertical=!implicit)

        nz, nxp1 = size(st.u)
        if open_bc
            # WHY the tendencies of face nx: grad_x_c! and diffusion_term! leave the boundary points out (wall values);
            # an outflow boundary assumes zero gradients instead, i.e. the baroclinic pressure gradient and horizontal
            # diffusion of the last interior face (the advection has its own open control volume, open_right)
            @views ws.grad_p_x[:, nxp1] .= ws.grad_p_x[:, nxp1 - 1]
            @views ws.diff_u[:, nxp1] .= ws.diff_u[:, nxp1 - 1]
        end
        @inbounds for i in 1:nxp1
            for k in 1:nz
                ws.u_hat[k, i] = st.u[k, i] + cfg.dt * (-ws.adv_u[k, i] - (1.0 / cfg.rho0) * ws.grad_p_x[k, i] + ws.diff_u[k, i] + cfg.body_force_x)
            end
        end
        if !cfg.use_hydrostatic
            # Non-hydrostatic pressure q at n + theta (Casulli 1999) in incremental form (van Kan 1986): the predictor
            # carries the full -dt/rho0 dq^n/dx (ws.p_corr still holds q^n), the projection solves for the increment
            # dq = q^{n+1} - q^n with theta dt, so the momentum equation has theta q^{n+1} + (1 - theta) q^n.
            # WHY incremental: with only (1 - theta) q^n in the predictor the free-surface solve misses most of q and
            # the splitting error is first order in time (dispersive standing wave at theta = 1/2: observed order
            # 1.49 -> 1.27); the incremental form gives order 2.0 (test_verify_theta_nonhydro_dispersive_temporal_order).
            grad_x_c!(ws.grad_pc_x, ws.p_corr, model.dx; periodic_x=cfg.periodic_x)
            # WHY no term on an open right boundary face (issue #49): dq/dx = 0 there (grad_x_c! is zero on the
            # boundary faces), see _check_right_level_config
            @. ws.u_hat -= cfg.dt / cfg.rho0 * ws.grad_pc_x
        end

        if cfg.wind_stress_x != 0.0
            # Wind stress tau_x (Pa) as a body force on the top layer, du = dt * tau_x / (rho0 * h_top), with the
            # top-layer thickness h_top = dz (fixed grid) or dz + eta at the u-face (variable top layer).
            # WHY in the predictor (it used to be added to st.u after the barotropic solve and the non-hydrostatic
            # projection): the free-surface solve and the projection then see the wind transport, as in the
            # semi-implicit scheme of Casulli & Cheng (1992), where the surface stress enters the explicit part of the
            # top-layer momentum equation. Added afterwards, the depth-integrated divergence of the final u no longer
            # matched -d(eta)/dt in the two wall columns and the non-hydrostatic u was not divergence-free there.
            wlo, whi = cfg.periodic_x ? (1, nxp1) : (2, nxp1 - 1)
            @inbounds for i in wlo:whi
                h_top = variable_top ? ws.h_top_face[i] : model.dz
                ws.u_hat[nz, i] += cfg.dt * cfg.wind_stress_x / (cfg.rho0 * h_top)
            end
        end

        if cfg.rayleigh_damping > 0.0
            damp = max(0.0, 1.0 - cfg.rayleigh_damping * cfg.dt)
            @. ws.u_hat = ws.u_hat * damp
        end
        if cfg.bed_roughness_length > 0.0 && !implicit
            # Log-law bed friction on the lowest row (explicit): du/dt = -c_b |u1| u1 / dz.
            cb = bed_friction_coefficient(cfg.bed_roughness_length, 0.5 * model.dz)
            @inbounds for i in 1:nxp1
                rate = cb * abs(st.u[1, i]) / model.dz
                cfg.dt * rate > 1.0 && throw(ArgumentError("bed friction too stiff for the explicit step: dt * c_b |u| / dz = $(cfg.dt * rate) > 1"))
                ws.u_hat[1, i] -= cfg.dt * rate * st.u[1, i]
            end
        end
        if cfg.chezy_coefficient > 0.0 && !implicit
            # Chezy bed friction (explicit) as a stress on the lowest layer: du_1/dt = -c_f |U| U / h_1, c_f = g / C^2,
            # U = sum_k h_k u_k / H the depth-averaged velocity of time n (flat bed). Depth-integrated it is
            # dU/dt = -c_f |U| U / H, forward Euler stable for dt * 2 c_f |U| / H < 2.
            cf = cfg.g / cfg.chezy_coefficient^2   # -
            @inbounds for i in 1:nxp1
                h_top = variable_top ? ws.h_top_face[i] : model.dz
                depth = (nz - 1) * model.dz + h_top
                s = 0.0
                for k in 1:(nz - 1)
                    s += st.u[k, i]
                end
                U = (s * model.dz + h_top * st.u[nz, i]) / depth
                cfg.dt * cf * abs(U) / depth > 1.0 && throw(ArgumentError("Chezy friction too stiff for the explicit step: dt * c_f |U| / H = $(cfg.dt * cf * abs(U) / depth) > 1"))
                ws.u_hat[1, i] -= cfg.dt * cf * abs(U) * U / (nz == 1 ? h_top : model.dz)
            end
        end
        if cfg.sponge_length > 0.0 && cfg.sponge_rate > 0.0
            # WHY quadratic ramp and implicit factor: a smooth increase of the damping over a layer of one or more
            # wavelengths absorbs waves with little reflection; 1/(1 + dt*rate) is stable for any rate.
            x_sponge = cfg.length - cfg.sponge_length
            @inbounds for i in 1:nxp1
                s = ((i - 1) * model.dx - x_sponge) / cfg.sponge_length
                s > 0.0 || continue
                factor = 1.0 / (1.0 + cfg.dt * cfg.sponge_rate * min(s, 1.0)^2)
                for k in 1:nz
                    ws.u_hat[k, i] *= factor
                end
            end
        end

        if masked
            _mask_u!(ws.u_hat, geo)
        end
        if cfg.apply_froude_limiter
            fr_hat = froude_number(ws.u_hat, st.rho, cfg.rho0, cfg.g, cfg.depth)
            if isfinite(fr_hat) && fr_hat > cfg.max_froude && fr_hat > 0.0
                fac = cfg.max_froude / fr_hat
                @. ws.u_hat = ws.u_hat * fac
            end
        end
    end

    if model.vegetation !== nothing
        @timeit model.timers "step/vegetation_coupling" begin
            # Explicit (forward Euler) drag from the velocities of time level n, like the other explicit terms.
            # WHY explicit: an implicit factor applied here, before the free-surface solve, would make the steady state
            # depend on dt (the surface-slope correction below is not divided by it); see src/vegetation.jl.
            vegetation_source!(ws.adv_u, model.vegetation, st.u, model.dx, model.dz, cfg.rho0;
                periodic_x=cfg.periodic_x, h_top_face=variable_top ? ws.h_top_face : nothing)
            drag_rate = vegetation_drag_rate(model.vegetation)
            if cfg.dt * drag_rate > VEGETATION_EXPLICIT_DRAG_LIMIT
                throw(ArgumentError("vegetation drag too stiff for the explicit step: dt * |a/u| = $(cfg.dt * drag_rate) > $(VEGETATION_EXPLICIT_DRAG_LIMIT); reduce dt"))
            end
            if model.turbulence !== nothing
                # vegetation production P_v = work of the drag per unit mass (m^2/s^3), averaged from the u-faces to the cells
                work = model.turbulence.veg_work
                @inbounds for i in 1:cfg.nx, k in 1:cfg.nz
                    work[k, i] = max(0.0, -0.5 * (ws.adv_u[k, i] * st.u[k, i] + ws.adv_u[k, i + 1] * st.u[k, i + 1]))
                end
            end
            @. ws.u_hat = ws.u_hat + cfg.dt * ws.adv_u
        end
    end

    @timeit model.timers "step/buoyancy_predictor" begin
        @views @. ws.rho_w = 0.5 * (rho_b[2:end, :] + rho_b[1:end-1, :])
        buoyancy!(ws.buoy, ws.rho_w, model.rho_0_w_2d)
        copyto!(model.last_buoy_w, ws.buoy)
        grad_z_c!(ws.grad_p_z, ws.p_internal, model.dz)
        if cfg.nonhydro_surface_dirichlet
            # See the Model struct's grad_p_z_reference field docstring: removes the reference-density profile's
            # "trivial" contribution before this feeds the Dirichlet-top projection's right-hand side.
            ws.grad_p_z .-= model.grad_p_z_reference
        end
    end

    @timeit model.timers "step/w_hat_predictor" begin
        copyto!(ws.w_hat, st.w)
        @views @. ws.w_hat[2:end-1, :] = st.w[2:end-1, :] + cfg.dt * (-ws.adv_w[2:end-1, :] - (1.0 / cfg.rho0) * ws.grad_p_z[2:end-1, :] + ws.buoy)
        if !cfg.use_hydrostatic
            grad_z_c!(ws.grad_pc_z, ws.p_corr, model.dz)   # incremental form, see the u_hat predictor
            @views @. ws.w_hat[2:end-1, :] = ws.w_hat[2:end-1, :] - cfg.dt / cfg.rho0 * ws.grad_pc_z[2:end-1, :]
        end
        if masked
            _mask_w!(ws.w_hat, geo)
        end
    end

    nx = cfg.nx

    # WHY (order): the free surface is advanced from the MOMENTUM-PREDICTOR transport (u_hat), before any
    # non-hydrostatic correction, in both modes -- moved here (was after the projection). In the earlier ordering,
    # the projection made u_hat exactly divergence-free cell-by-cell; summed over depth that forced the transport
    # divergence used by the free-surface equation to equal minus the OLD, stale top-face w_hat (carried over from
    # the previous step's w_hat predictor copy above, never touched by the projection's interior-only correction),
    # i.e. eta^{n+1} ~= eta^n + dt*eta_t^n: a homogeneous recursion with eta=0 as a self-consistent fixed point. That
    # silently made the non-hydrostatic mode a rigid lid (eta stayed at round-off regardless of real internal
    # forcing; see docs/TESTS_TECHNICAL_DOCUMENTATION.md section 14) even though the same barotropic solve and
    # surface-slope correction as the hydrostatic branch were already being computed. Moving them here (before the
    # projection, exactly as the hydrostatic branch already had them relative to its own -- absent -- projection)
    # decouples eta from the projection in both modes; nothing changes for the hydrostatic branch, whose projection
    # block below is a no-op.
    @timeit model.timers "step/barotropic_solve" begin
        # WHY (both branches): the solve uses the boundary-face transport that the final velocities will have, so
        # that eta^{n+1} - eta^n = -dt * sum_k div(h*u^{n+1}) holds exactly (and the top-layer budgets close). The
        # predictor above also fills the wall faces (advection and diffusion evaluated from interior neighbours); on
        # the fixed grid this used to enter the eta equation as a spurious flux through closed walls, so the water
        # volume drifted (seen in the non-hydrostatic wind-shear case, issue #4).
        # In a wall splash it also forced the projection to accommodate fictitious wall transport in the top cells,
        # driving the surface jet of issue #5. Enforce the walls before BOTH solves, not just on the final state.u.
        if theta != 1.0
            # explicit part of the surface pressure gradient, -(1 - theta) g dt d(eta^n)/dx (Casulli & Cattani 1994)
            _surface_gradient!(ws.eta_grad_x, ws.eta_old, model.dx, cfg.periodic_x)
            _add_surface_gradient!(ws.u_hat, ws.eta_grad_x, -(1.0 - theta) * cfg.g * cfg.dt, cfg.periodic_x)
            if open_bc
                slope = _right_boundary_slope(ws.eta_old, model, model.dx; old=true)
                @views ws.u_hat[:, nx + 1] .-= (1.0 - theta) * cfg.g * cfg.dt * slope
            end
        end
        _apply_x_boundary_conditions!(model, ws.u_hat)
        if masked
            _mask_u!(ws.u_hat, geo)
        end
        # The continuity flux takes u^{n+theta} = theta u^{n+1} + (1 - theta) u^n with the thicknesses of level n; with
        # u^{n+1} = u_hat - theta g dt d(eta^{n+1})/dx the solve sees the transport of theta u_hat + (1 - theta) u^n and the
        # coefficients theta^2 dt^2 g H / dx^2 (BarotropicSolver). theta = 1 gives u_theta = u_hat exactly.
        if implicit
            # vertical viscosity, vertical advection and bed friction implicit: u_hat <- A^{-1}(dZ .* u_hat), r = A^{-1} dZ
            _implicit_vertical_terms!(model, variable_top)
        end
        @. ws.u_theta = theta * ws.u_hat + (1.0 - theta) * ws.u_old
        if implicit
            _face_transport!(bsolver.U_hat, ws.u_theta, model, variable_top)
            eta_eff, weight = open_bc ? _right_boundary_dirichlet(model) : (0.0, 1.0)
            solve_effective_depth!(bsolver, st.eta, ws.eta_old, ws.H_face; periodic_x=cfg.periodic_x,
                right_level=open_bc ? eta_eff : nothing, right_weight=weight)
            variable_top && _check_top_layer_thickness(model, st.eta)
        elseif variable_top
            solve_variable_thickness!(bsolver, st.eta, ws.u_theta, ws.eta_old, ws.h_top_face)
            _check_top_layer_thickness(model, st.eta)
        else
            solve!(bsolver, st.eta, ws.u_theta, ws.eta_old)
        end
    end

    @timeit model.timers "step/free_surface_slope" begin
        _surface_gradient!(ws.eta_grad_x, st.eta, model.dx, cfg.periodic_x)
        # WHY: corrects ws.u_hat in place (not yet written to st.u) so a following non-hydrostatic projection acts on
        # this surface-corrected predictor, exactly as it already acted on the (pressure-predictor-only) u_hat before.
        _add_surface_gradient!(ws.u_hat, ws.eta_grad_x, -cfg.g * cfg.dt * theta, cfg.periodic_x; r=implicit ? ws.r_vert : nothing)
        if open_bc
            slope = _right_boundary_slope(st.eta, model, model.dx)
            @views @. ws.u_hat[:, nx + 1] -= cfg.g * cfg.dt * theta * slope * ws.r_vert[:, nx + 1]
        end

        if masked
            _mask_u!(ws.u_hat, geo)
        end

        @inbounds for i in 1:nx
            ws.eta_t[i] = (st.eta[i] - ws.eta_old[i]) / cfg.dt
        end
    end

    if !cfg.use_hydrostatic
        @timeit model.timers "step/nonhydro_projection" begin
            # WHY: ws.w_hat's top face still carries the value copied from the OLD st.w at the top of this function
            # (the w_hat predictor above only updates interior rows 2:end-1); refresh it to the just-computed eta_t so
            # the projection's divergence right-hand side reflects the true, kinematically consistent free-surface
            # velocity (the free surface's own kinematic condition, enforced through the barotropic solve above and
            # the w reconstruction below) rather than a one-step-stale one. This is done the same way whether the
            # solver's top condition is the legacy Neumann one or the (docs section 15) Dirichlet p'=0 condition:
            # either way, the top face's velocity is treated as externally known here, and the elliptic solve's job is
            # to find the pressure (and correct the INTERIOR flow) consistent with it and with the chosen top
            # pressure condition.
            # Theta scheme: the surface moves with u^{n+theta}, eta_t = theta w_s^{n+1} + (1 - theta) w_s^n, so the surface
            # value of w^{n+1} is (eta_t - (1 - theta) w_s^n) / theta, with w_s^n the column divergence of u^n with the
            # thicknesses of level n (as in the surface solve); it equals the column divergence of the surface-corrected
            # u_hat, so the projection keeps the column transports.
            # WHY not the stored st.w[end, :]: it need not be the divergence of u^n (initial state, variable top layer
            # integrated with the thicknesses of level n - 1); the mismatch went into a depth-uniform part of q with a
            # slope, which the incremental form keeps for ever (steady wind set-up: surface slope -3 times the exact one
            # at theta = 0.7). ws.w_theta is free here (it is computed after the projection).
            if theta == 1.0
                @views ws.w_hat[end, :] .= ws.eta_t
            else
                _integrate_w_from_continuity!(ws.w_theta, ws.u_old, model, variable_top)
                @views @. ws.w_hat[end, :] = (ws.eta_t - (1.0 - theta) * ws.w_theta[end, :]) / theta
            end

            # increment dq = q^{n+1} - q^n with the weight theta in the momentum equation: the projection corrects with
            # theta dt (incremental form, see the u_hat predictor)
            dt_q = theta * cfg.dt
            solve!(model.pressure_correction_solver, ws.u_hat, ws.w_hat, cfg.rho0, dt_q, model.dx, model.dz)
            dq = model.pressure_correction_solver.p_corr
            grad_x_c!(ws.grad_pc_x, dq, model.dx; periodic_x=cfg.periodic_x)
            grad_z_c!(ws.grad_pc_z, dq, model.dz)
            ws.p_corr .+= dq   # q^{n+1}

            @. ws.u_hat = ws.u_hat - dt_q * (1.0 / cfg.rho0) * ws.grad_pc_x   # open right boundary face: dq/dx = 0
            # WHY: only INTERIOR w faces (2:end-1) are corrected, same as the legacy Neumann case. With the
            # homogeneous-Neumann condition this correction is exactly zero at the boundary faces (0 and end), so
            # leaving them uncorrected changes nothing there. With the Dirichlet condition the (unapplied) top-face
            # correction is generally non-zero, but it would only ever affect ws.w_hat, which nothing downstream
            # reads: the model's actual w (state.w) is always reconstructed from state.u's divergence (w_reconstruction
            # below), never copied from ws.w_hat. The Dirichlet condition's effect on the physics is carried entirely
            # through the pressure field's effect on ws.u_hat (grad_pc_x, above) and hence on state.u.
            @views @. ws.w_hat[2:end-1, :] = ws.w_hat[2:end-1, :] - dt_q * (1.0 / cfg.rho0) * ws.grad_pc_z[2:end-1, :]
            if masked
                # WHY: the gradient between a wet cell and a solid cell (p' = 0 there) is not a physical force on a wall
                _mask_u!(ws.u_hat, geo)
                _mask_w!(ws.w_hat, geo)
            end

            model.last_nonhydro_pcorr_max = maximum(abs, ws.p_corr)
            copyto!(model.last_p_corr, ws.p_corr)
        end
    else
        model.last_nonhydro_pcorr_max = 0.0
        fill!(model.last_p_corr, 0.0)
        fill!(ws.p_corr, 0.0)
    end

    @timeit model.timers "step/free_surface_u_update" begin
        ilo, ihi = cfg.periodic_x ? (1, nx + 1) : (2, open_bc ? nx + 1 : nx)
        @inbounds for i in ilo:ihi
            for k in 1:cfg.nz
                st.u[k, i] = ws.u_hat[k, i]
            end
        end

        _apply_x_boundary_conditions!(model)
        if masked
            _mask_u!(st.u, geo)
        end
        @. ws.u_theta = theta * st.u + (1.0 - theta) * ws.u_old
    end

    @timeit model.timers "step/w_reconstruction" begin
        # w^{n+theta} from the continuity of u^{n+theta}, the velocities that moved the surface: its surface value equals
        # eta_t up to round-off (kinematic condition, checked here) and it transports the constituent.
        _integrate_w_from_continuity!(ws.w_theta, ws.u_theta, model, variable_top)
        max_mismatch = 0.0
        @inbounds for i in 1:nx
            max_mismatch = max(max_mismatch, abs(ws.w_theta[end, i] - ws.eta_t[i]))
        end
        model.last_kinematic_mismatch = max_mismatch
        @views ws.w_theta[end, :] .= ws.eta_t   # velocity of the fluid at the surface (d eta/dt)
        if theta == 1.0
            copyto!(st.w, ws.w_theta)
        else
            # w^{n+1} of the continuity of u^{n+1} (momentum advection of the next step, output); its surface value is
            # the column divergence of u^{n+1}, not eta_t.
            _integrate_w_from_continuity!(st.w, st.u, model, variable_top)
        end
    end

    @timeit model.timers "step/density_transport" begin
        # WHY u_theta, w_theta: the transport velocities must be those of the volume change (eta^{n+1} - eta^n uses
        # u^{n+theta}), otherwise a uniform constituent does not stay uniform and the mass is not conserved.
        if variable_top
            advect_density_top_layer!(ws.rho_new, st.rho, ws.u_theta, ws.w_theta, cfg.dt, model.dx, model.dz, ws.density_ws, ws.h_top_face, ws.eta_old, st.eta; scheme=cfg.transport_advection_scheme, active=masked ? geo.wet_cell : nothing, rho_min=model.rho_min_init, rho_max=model.rho_max_init)
        else
            # WHY: fixed grid (variable top layer off, incl. every non-hydrostatic run): advective form in the top row removes
            # the spurious source of the moving top face, see _density_top_row_advective_form! (docs section 4).
            advect_density!(ws.rho_new, st.rho, ws.u_theta, ws.w_theta, cfg.dt, model.dx, model.dz, ws.density_ws; scheme=cfg.transport_advection_scheme, top_row_advective_form=true, periodic_x=cfg.periodic_x, active=masked ? geo.wet_cell : nothing, rho_min=model.rho_min_init, rho_max=model.rho_max_init)
        end
        _apply_wind_surface_density_mixing!(ws.rho_new, cfg, model.dz)
        if model.turbulence !== nothing
            # turbulent density diffusion K = nu_t / sigma_rho at the w-faces (implicit, conservative), with the top-row
            # thickness dz + eta^{n+1} of the variable top layer (issue #30)
            ke = model.turbulence
            variable_top && (@. ke.h_top = model.dz + st.eta)
            implicit_vertical_diffusion!(ws.rho_new, (k, i) -> 0.5 * (ke.nu_t[k, i] + ke.nu_t[k + 1, i]) / KE_SIGMA_RHO,
                (k, i) -> (variable_top && k == cfg.nz) ? ke.h_top[i] : model.dz, cfg.dt, model.dz, ke;
                top_thickness=variable_top ? ke.h_top : nothing)
        end
        copyto!(st.rho, ws.rho_new)
        _correct_constituent_mass!(model)
    end

    if model.turbulence !== nothing
        model.vegetation === nothing && fill!(model.turbulence.veg_work, 0.0)
        @timeit model.timers "step/k_epsilon" k_epsilon_step!(model.turbulence, st.u, st.w, st.rho, cfg.dt, model.dx, model.dz, cfg.nu,
            cfg.g, cfg.rho0; periodic_x=cfg.periodic_x, bed_roughness=cfg.bed_roughness_length,
            top_thickness=variable_top ? model.turbulence.h_top : nothing)
    end

    @timeit model.timers "step/final_pressure" begin
        solve_hydrostatic_pressure!(ws.p_internal, st.rho, cfg.g, model.dz; top_half_layer=variable_top)
        if cfg.use_hydrostatic
            copyto!(st.p, ws.p_internal)
            fill!(model.last_p_corr, 0.0)
        else
            @. st.p = ws.p_internal + ws.p_corr
            copyto!(model.last_p_corr, ws.p_corr)
        end
        copyto!(model.last_p_hydro, ws.p_internal)
    end

    @timeit model.timers "step/final_buoyancy" begin
        @views @. ws.rho_w = 0.5 * (st.rho[2:end, :] + st.rho[1:end-1, :])
        buoyancy!(model.last_buoy_w, ws.rho_w, model.rho_0_w_2d)
    end

    model.x_boundary_right_free && _finish_free_outflow_level!(model)
    @timeit model.timers "step/froude_check" _check_surface_froude!(model)
    return nothing
end

function run!(model::Model)
    history = State{Float64}[copy_state(model.state)]
    history_eta_prev = Vector{Vector{Float64}}([copy(model.state.eta)])
    history_time = Float64[0.0]
    vegetation_history = model.vegetation === nothing ? nothing : Any[copy_vegetation(model.vegetation)]

    start_time = time()
    reset_timer!(model.timers)
    reset_timer!(model.pressure_correction_solver.timers)
    save_interval_seconds = model.config.save_interval_seconds
    save_interval = max(1, Int(round(save_interval_seconds / model.config.dt)))
    effective_save_interval = save_interval * model.config.dt
    eta_prev_for_final_plot = copy(model.state.eta)

    initial_diag = compute_diagnostics(model)
    initial_mean_rho = mean(model.state.rho)
    @printf("Initial mean density: %.6f\n", initial_mean_rho)
    pressure_mode = model.config.use_hydrostatic ? "hydrostatic" : "non-hydrostatic"
    println("Pressure mode: $(pressure_mode)")
    @printf(
        "Initial diagnostics | Fr=%.3f | CFL=%.3f | Vol=%.6e | Constituent=%.6e\n",
        initial_diag["froude"],
        initial_diag["cfl_total"],
        initial_diag["water_volume"],
        initial_diag["constituent_mass"],
    )
    @printf(
        "Output cadence: every %.3f s (%d steps, requested %.3f s)\n",
        effective_save_interval,
        save_interval,
        save_interval_seconds,
    )

    for i in 1:model.config.nt
        copyto!(eta_prev_for_final_plot, model.state.eta)
        @timeit model.timers "run/step_forward_total" step_forward!(model)

        if mod(i, save_interval) == 0
            @timeit model.timers "run/history_and_diagnostics" begin
                push!(history, copy_state(model.state))
                push!(history_eta_prev, copy(eta_prev_for_final_plot))
                push!(history_time, i * model.config.dt)
                if vegetation_history !== nothing
                    push!(vegetation_history, copy_vegetation(model.vegetation))
                end

                st = model.state
                cfg = model.config
                diag = compute_diagnostics(model)
                mean_rho = mean(st.rho)
                nh_part = cfg.use_hydrostatic ? "" : @sprintf(" | NH_pcorr_max=%.2e", diag["nonhydro_pcorr_max"])

                @printf(
                    "Step %04d/%d | CFL: %.3f (u=%.3f, w=%.3f) | Fr: %.3f | Max Vel: u=%.2e, w=%.2e | Mean Rho: %.6f (err=%.2e) | Div: max=%.2e, mean=%.2e | Wtop-Eta_t=%.2e | VolErr=%.2e | ConstituentErr=%.2e%s\n",
                    i - 1,
                    cfg.nt,
                    diag["cfl_total"],
                    diag["cfl_x"],
                    diag["cfl_z"],
                    diag["froude"],
                    maximum(abs, st.u),
                    maximum(abs, st.w),
                    mean_rho,
                    mean_rho - initial_mean_rho,
                    diag["div_max"],
                    diag["div_mean_abs"],
                    diag["kinematic_fs_mismatch"],
                    diag["water_volume"] - initial_diag["water_volume"],
                    diag["constituent_mass"] - initial_diag["constituent_mass"],
                    nh_part,
                )
            end
        end
    end

    elapsed = time() - start_time
    @printf("Simulation finished in %.2f seconds.\n", elapsed)

    output_dir = isempty(strip(model.config.output_dir)) ? "output_barotropic" : model.config.output_dir
    if !isdir(output_dir)
        mkpath(output_dir)
    end

    default_final = joinpath(output_dir, "final_state.png")
    default_anim = joinpath(output_dir, "animation.gif")
    default_pb_plot = joinpath(output_dir, "pressure_buoyancy_diagnostics.png")
    default_pb_npz = joinpath(output_dir, "pressure_buoyancy_diagnostics.npz")

    @timeit model.timers "run/output_cleanup" begin
        for p in (default_final, default_anim, default_pb_plot, default_pb_npz)
            if isfile(p)
                rm(p; force=true)
            end
        end
    end

    final_diag = compute_diagnostics(model)
    title = @sprintf(
        "Final State (t=%.2fs, L=%.1fm, H=%.1fm, dx=%.3fm, dz=%.3fm, Fr=%.3f, VolErr=%.2e, ConErr=%.2e)",
        model.config.nt * model.config.dt,
        model.config.length,
        model.config.depth,
        model.dx,
        model.dz,
        final_diag["froude"],
        final_diag["water_volume"] - initial_diag["water_volume"],
        final_diag["constituent_mass"] - initial_diag["constituent_mass"],
    )

    if model.config.write_plots
        @timeit model.timers "run/write_final_plot" plot_state(model.state, model.config; title=title, filename=default_final, eta_prev=eta_prev_for_final_plot, dt=model.config.dt, vegetation=model.vegetation)
    end

    if model.config.write_animation
        @timeit model.timers "run/write_animation" create_animation(history, model.config; filename=default_anim, save_interval=save_interval, eta_prev_history=history_eta_prev, frame_times=history_time, vegetation_history=vegetation_history)
    end

    p_hydro_final, p_corr_final, buoy_w_final = pressure_buoyancy_components(model)
    if model.config.write_pressure_buoyancy_diagnostics
        @timeit model.timers "run/write_pressure_buoyancy" begin
            plot_pressure_buoyancy_diagnostics(model.state, model.config, p_hydro_final, p_corr_final, buoy_w_final; filename=default_pb_plot)
            npzwrite(default_pb_npz, Dict(
                "p_total" => model.state.p,
                "p_hydro" => p_hydro_final,
                "p_corr" => p_corr_final,
                "buoy_w" => buoy_w_final,
            ))
        end
    end

    tag = output_case_tag(model.config)

    if model.config.write_plots
        @timeit model.timers "run/tag_final_plot" begin
            tagged_final = joinpath(output_dir, "final_state_$(tag).png")
            cp(default_final, tagged_final; force=true)
            println("Saved final plot: $(abspath(default_final))")
            println("Saved tagged final plot: $(abspath(tagged_final))")
        end
    end

    if model.config.write_animation
        @timeit model.timers "run/tag_animation" begin
            tagged_anim = joinpath(output_dir, "animation_$(tag).gif")
            cp(default_anim, tagged_anim; force=true)
            println("Saved animation: $(abspath(default_anim))")
            println("Saved tagged animation: $(abspath(tagged_anim))")
        end
    end

    if model.config.write_pressure_buoyancy_diagnostics
        @timeit model.timers "run/tag_pressure_buoyancy" begin
            tagged_pb_plot = joinpath(output_dir, "pressure_buoyancy_diagnostics_$(tag).png")
            tagged_pb_npz = joinpath(output_dir, "pressure_buoyancy_diagnostics_$(tag).npz")
            cp(default_pb_plot, tagged_pb_plot; force=true)
            cp(default_pb_npz, tagged_pb_npz; force=true)
            println("Saved pressure/buoyancy diagnostics plot: $(abspath(default_pb_plot))")
            println("Saved pressure/buoyancy diagnostics data: $(abspath(default_pb_npz))")
            println("Saved tagged pressure/buoyancy diagnostics plot: $(abspath(tagged_pb_plot))")
            println("Saved tagged pressure/buoyancy diagnostics data: $(abspath(tagged_pb_npz))")
        end
    end

    println("\nTimer summary (sorted by wall time):")
    print_timer(model.timers; allocations=true, sortby=:time, linechars=:ascii, title="Model Hotspots")

    if !model.config.use_hydrostatic
        println("\nPressure-correction substage summary (sorted by wall time):")
        print_timer(model.pressure_correction_solver.timers; allocations=true, sortby=:time, linechars=:ascii, title="Pressure-Correction Hotspots")
    end

    return nothing
end

mutable struct BarotropicSolver{T<:Real}
    config::ModelConfig
    dx::T
    dz::T
    nx::Int
    periodic_factorization::Any
    H::T
    g::T
    dt::T
    theta::T   # -, implicitness of the free surface; the coefficients a_f carry the factor theta^2
    lower::Vector{T}
    main::Vector{T}
    upper::Vector{T}
    cp::Vector{T}
    inv_denom::Vector{T}
    U_hat::Vector{T}
    rhs::Vector{T}
    dp::Vector{T}
    x::Vector{T}
    # Coefficients of the variable-thickness system, rebuilt every step (they depend on eta^n); the constant-thickness
    # factorisation above is left untouched so both solves can be used on the same solver object.
    lower_var::Vector{T}
    main_var::Vector{T}
    upper_var::Vector{T}
    cp_var::Vector{T}
    inv_denom_var::Vector{T}
    k_bed_face::Vector{Int}   # lowest open layer per u-face (all ones for the flat bed)
    cyclic_z::Vector{T}       # work vector of the cyclic (periodic) solve with per-step coefficients
end

function BarotropicSolver(cfg::ModelConfig, dx::Real, dz::Real; T::Type{<:Real}=Float64, geometry::Union{Nothing,BedGeometry}=nothing)
    nx = cfg.nx
    H = T(cfg.depth)
    g = T(cfg.g)
    dt = T(cfg.dt)
    theta = T(cfg.theta)
    k_bed_face = geometry === nothing ? ones(Int, nx + 1) : copy(geometry.k_bed_face)

    # WHY theta^2: u^{n+1} carries theta g dt d(eta^{n+1})/dx and enters the continuity flux with the weight theta
    alpha = theta * theta * dt * dt * g * H / (dx * dx)
    periodic_factorization = nothing
    if cfg.periodic_x
        rows = Int[]
        cols = Int[]
        vals = T[]
        sizehint!(rows, 3 * nx)
        sizehint!(cols, 3 * nx)
        sizehint!(vals, 3 * nx)
        for i in 1:nx
            push!(rows, i); push!(cols, i); push!(vals, one(T) + 2 * alpha)
            push!(rows, i); push!(cols, mod1(i - 1, nx)); push!(vals, -alpha)
            push!(rows, i); push!(cols, mod1(i + 1, nx)); push!(vals, -alpha)
        end
        periodic_factorization = lu(sparse(rows, cols, vals, nx, nx))
    end
    lower = fill(T(-alpha), nx - 1)
    main = fill(T(1.0 + 2.0 * alpha), nx)
    upper = fill(T(-alpha), nx - 1)
    main[1] = T(1.0 + alpha)
    main[end] = T(1.0 + alpha)
    if geometry !== nothing && geometry.has_bed
        # Water depth per u-face, H_f = H - bed_face_level (m); a_f = dt^2 g H_f / dx^2, closed outer faces a = 0.
        # WHY the evaluation order: for a zero bed a_f + a_f == 2 * alpha and 1 + (0 + a_f) == 1 + alpha exactly, so the
        # masked path reproduces the flat-bed coefficients bit for bit.
        a_face = zeros(T, nx + 1)
        for f in 2:nx
            a_face[f] = theta * theta * dt * dt * g * (H - geometry.bed_face_level[f]) / (dx * dx)
        end
        for i in 1:nx
            main[i] = T(1.0 + (a_face[i] + a_face[i + 1]))
        end
        for i in 1:(nx - 1)
            lower[i] = -a_face[i + 1]
            upper[i] = -a_face[i + 1]
        end
    end

    cp = zeros(T, nx - 1)
    inv_denom = zeros(T, nx)
    inv_denom[1] = 1.0 / main[1]

    if nx > 1
        cp[1] = upper[1] * inv_denom[1]
    end
    for i in 2:(nx - 1)
        denom = main[i] - lower[i - 1] * cp[i - 1]
        inv_denom[i] = 1.0 / denom
        cp[i] = upper[i] * inv_denom[i]
    end
    if nx > 1
        denom_last = main[end] - lower[end] * cp[end]
        inv_denom[end] = 1.0 / denom_last
    end

    U_hat = zeros(T, nx + 1)
    rhs = zeros(T, nx)
    dp = zeros(T, nx)
    x = zeros(T, nx)

    return BarotropicSolver(
        cfg, T(dx), T(dz), nx, periodic_factorization, H, g, dt, theta, lower, main, upper, cp, inv_denom, U_hat, rhs, dp, x,
        zeros(T, nx - 1), zeros(T, nx), zeros(T, nx - 1), zeros(T, nx - 1), zeros(T, nx), k_bed_face, zeros(T, nx),
    )
end

"""
    solve_variable_thickness!(solver, eta, u_hat, eta_old, h_top_face)

Semi-implicit free-surface update for a variable-thickness top layer. `u_hat` (m/s, `nz x (nx+1)`) is the provisional
velocity, `eta_old` (m) the surface elevation at the old time level and `h_top_face` (m, length `nx+1`) the top-layer
thickness at the u-faces built from `eta_old` (`top_layer_face_thickness!`). The new elevation `eta` (m) solves

    eta_i - a_{i+1}(eta_{i+1} - eta_i) + a_i(eta_i - eta_{i-1}) = eta_old_i - dt * (Uhat_{i+1} - Uhat_i)/dx,
    Uhat_i = dz * sum_{k<nz} uhat_{k,i} + h_top_face_i * uhat_{nz,i},
    a_i    = dt^2 * g * H_i / dx^2,   H_i = (nz-1)*dz + h_top_face_i   (a_1 = a_{nx+1} = 0, closed/prescribed boundary faces),

i.e. the continuity equation `eta_t + d(sum_k h_k u_k)/dx = 0` with `u^{n+1} = uhat - g*dt*d(eta^{n+1})/dx` in every layer
and the thicknesses taken at the old time level (linear in `eta^{n+1}`). For `eta_old = 0` it reduces to the constant-depth
system of `solve!`. The tridiagonal matrix is symmetric positive definite and is solved by the Thomas algorithm.
Source: Casulli & Cheng (1992), Int. J. Numer. Methods Fluids 15, 629-648 (semi-implicit free surface; equation numbers not checked).
"""
function solve_variable_thickness!(
    solver::BarotropicSolver,
    eta::AbstractVector{<:Real},
    u_hat::AbstractMatrix{<:Real},
    eta_old::AbstractVector{<:Real},
    h_top_face::AbstractVector{<:Real},
)
    if solver.periodic_factorization !== nothing
        throw(ArgumentError("solve_variable_thickness! does not support periodic_x=true"))
    end
    nx = solver.nx
    nz = size(u_hat, 1)
    dx = solver.dx
    dz = solver.dz
    dt = solver.dt
    g = solver.g
    coeff = solver.theta * solver.theta * dt * dt * g / (dx * dx)   # 1/m; times H_i (m) it gives the dimensionless a_i

    @inbounds for i in 1:(nx + 1)
        s = zero(eltype(solver.U_hat))
        for k in 1:(nz - 1)
            s += u_hat[k, i]
        end
        solver.U_hat[i] = s * dz + h_top_face[i] * u_hat[nz, i]
    end

    @inbounds for i in 1:nx
        # water depth per face: (nz - k_bed_face) layers of dz below the top layer plus the top-layer thickness
        a_left = i == 1 ? zero(coeff) : coeff * ((nz - solver.k_bed_face[i]) * dz + h_top_face[i])
        a_right = i == nx ? zero(coeff) : coeff * ((nz - solver.k_bed_face[i + 1]) * dz + h_top_face[i + 1])
        solver.main_var[i] = 1.0 + a_left + a_right
        if i >= 2
            solver.lower_var[i - 1] = -a_left
        end
        if i <= nx - 1
            solver.upper_var[i] = -a_right
        end
        solver.rhs[i] = eta_old[i] - dt * (solver.U_hat[i + 1] - solver.U_hat[i]) / dx
    end

    solver.inv_denom_var[1] = 1.0 / solver.main_var[1]
    solver.cp_var[1] = solver.upper_var[1] * solver.inv_denom_var[1]
    @inbounds for i in 2:(nx - 1)
        denom = solver.main_var[i] - solver.lower_var[i - 1] * solver.cp_var[i - 1]
        solver.inv_denom_var[i] = 1.0 / denom
        solver.cp_var[i] = solver.upper_var[i] * solver.inv_denom_var[i]
    end
    solver.inv_denom_var[nx] = 1.0 / (solver.main_var[nx] - solver.lower_var[nx - 1] * solver.cp_var[nx - 1])

    solver.dp[1] = solver.rhs[1] * solver.inv_denom_var[1]
    @inbounds for i in 2:nx
        solver.dp[i] = (solver.rhs[i] - solver.lower_var[i - 1] * solver.dp[i - 1]) * solver.inv_denom_var[i]
    end
    solver.x[nx] = solver.dp[nx]
    @inbounds for i in (nx - 1):-1:1
        solver.x[i] = solver.dp[i] - solver.cp_var[i] * solver.x[i + 1]
    end

    copyto!(eta, solver.x)
    return eta
end

function _solve_tridiagonal!(solver::BarotropicSolver, rhs::AbstractVector{<:Real}, dp::AbstractVector{<:Real}, x::AbstractVector{<:Real})
    dp[1] = rhs[1] * solver.inv_denom[1]
    for i in 2:solver.nx
        dp[i] = (rhs[i] - solver.lower[i - 1] * dp[i - 1]) * solver.inv_denom[i]
    end

    x[end] = dp[end]
    for i in (solver.nx - 1):-1:1
        x[i] = dp[i] - solver.cp[i] * x[i + 1]
    end

    return x
end

function solve!(solver::BarotropicSolver, eta::AbstractVector{<:Real}, u_hat::AbstractMatrix{<:Real}, eta_old::AbstractVector{<:Real})
    nx = solver.nx

    nz = size(u_hat, 1)
    @inbounds for i in 1:(nx + 1)
        s = zero(eltype(solver.U_hat))
        for k in 1:nz
            s += u_hat[k, i]
        end
        solver.U_hat[i] = s * solver.dz
    end

    @inbounds for i in 1:nx
        div_U_hat = (solver.U_hat[i + 1] - solver.U_hat[i]) / solver.dx
        solver.rhs[i] = eta_old[i] - solver.dt * div_U_hat
    end

    if solver.periodic_factorization === nothing
        _solve_tridiagonal!(solver, solver.rhs, solver.dp, solver.x)
    else
        copyto!(solver.x, solver.periodic_factorization \ solver.rhs)
    end
    copyto!(eta, solver.x)

    return eta
end

# Thomas algorithm for the tridiagonal system with sub-diagonal `lower` (lower[i] couples x[i] to x[i-1], i >= 2),
# diagonal `diag` and super-diagonal `upper` (upper[i] couples x[i] to x[i+1]); `cp`, `dp` work vectors. Writes x.
function _thomas!(x::AbstractVector, lower::AbstractVector, diag::AbstractVector, upper::AbstractVector,
    rhs::AbstractVector, cp::AbstractVector, dp::AbstractVector, n::Int)
    cp[1] = upper[1] / diag[1]
    dp[1] = rhs[1] / diag[1]
    @inbounds for i in 2:n
        denom = diag[i] - lower[i] * cp[i - 1]
        cp[i] = upper[i] / denom
        dp[i] = (rhs[i] - lower[i] * dp[i - 1]) / denom
    end
    x[n] = dp[n]
    @inbounds for i in (n - 1):-1:1
        x[i] = dp[i] - cp[i] * x[i + 1]
    end
    return x
end

"""
    solve_effective_depth!(solver, eta, eta_old, H_face; periodic_x=false, right_level=nothing)

Free-surface solve with per-face effective depths `H_face` (m, length nx+1; the implicit vertical terms give
H_f = sum_k dZ_k r_k, src/implicit_vertical.jl) and the face transports already in `solver.U_hat` (m^2/s):

    eta_i + a_{i+1}(eta_i - eta_{i+1}) + a_i(eta_i - eta_{i-1}) = eta_old_i - dt (U_{i+1} - U_i)/dx,
    a_f = theta^2 dt^2 g H_f / dx^2   (a = 0 at the outer faces unless periodic_x; with periodic_x face 1 = face nx+1).

`right_level` (m): water-level boundary at face nx+1, eta_{nx+1} = right_level at half a cell from the last centre, so
a_{nx+1} = right_weight 2 theta^2 dt^2 g H_{nx+1} / dx^2 and a_{nx+1} right_level moves to the right-hand side
(`_right_level_slope`). `right_weight` (-, default 1): a free outflow passes its effective level and the weight
1/(1 + mu) of its implicit radiation condition (`_right_boundary_dirichlet`).

Symmetric, diagonally dominant (H_f >= 0): Thomas algorithm, or with periodic_x the cyclic system by the
Sherman-Morrison formula (Press et al. 2007, Numerical Recipes, 3rd ed., section 2.7.2).
PERF: coefficients rebuilt every step (H_f depends on the friction and the flow), O(nx), no allocation.
"""
function solve_effective_depth!(solver::BarotropicSolver, eta::AbstractVector{<:Real}, eta_old::AbstractVector{<:Real},
    H_face::AbstractVector{<:Real}; periodic_x::Bool=false, right_level::Union{Nothing,Real}=nothing,
    right_weight::Real=1.0)
    nx = solver.nx
    dt, dx = solver.dt, solver.dx
    coeff = solver.theta * solver.theta * dt * dt * solver.g / (dx * dx)   # 1/m
    diag = solver.main_var
    # WHY these arrays: lower_var/upper_var have nx-1 entries, the cyclic solve needs sub- and super-diagonals of length
    # nx (the corner entries are moved out before the solve); solver.dp and solver.cyclic_z hold them
    sub, sup = solver.dp, solver.cyclic_z
    @inbounds for i in 1:nx
        a_left = (i == 1 && !periodic_x) ? 0.0 : coeff * H_face[i]
        a_right = (i == nx && !periodic_x) ? 0.0 : coeff * H_face[i + 1]
        diag[i] = 1.0 + a_left + a_right
        sub[i] = -a_left
        sup[i] = -a_right
        solver.rhs[i] = eta_old[i] - dt * (solver.U_hat[i + 1] - solver.U_hat[i]) / dx
    end
    if right_level !== nothing
        periodic_x && throw(ArgumentError("right_level is not possible with periodic_x=true"))
        a_bc = right_weight * 2.0 * coeff * H_face[nx + 1]   # WHY 2: the level lies half a cell from the last centre
        diag[nx] += a_bc
        solver.rhs[nx] += a_bc * right_level
    end
    if !periodic_x
        sub[1] = 0.0
        sup[nx] = 0.0
        # cp/dp work: inv_denom_var (nx) and cp_var (nx-1) are too short for _thomas!, use eta as dp storage
        _thomas!(solver.x, sub, diag, sup, solver.rhs, solver.inv_denom_var, eta, nx)
        copyto!(eta, solver.x)
        return eta
    end
    # cyclic: corners A[nx,1] = A[1,nx] = -a_1 (face 1 = face nx+1)
    alpha = sub[1]
    beta = sup[nx]
    gamma = -diag[1]
    diag[1] -= gamma
    diag[nx] -= alpha * beta / gamma
    sub[1] = 0.0
    sup[nx] = 0.0
    _thomas!(solver.x, sub, diag, sup, solver.rhs, solver.inv_denom_var, eta, nx)   # T x = rhs
    z = solver.U_hat   # WHY: the transports are no longer needed once rhs is built; reused as the second right-hand side
    fill!(z, 0.0)
    z[1] = gamma
    z[nx] = alpha
    _thomas!(z, sub, diag, sup, z, solver.inv_denom_var, eta, nx)                   # T z = (gamma, 0, ..., alpha)
    factor = (solver.x[1] + beta * solver.x[nx] / gamma) / (1.0 + z[1] + beta * z[nx] / gamma)
    @inbounds for i in 1:nx
        eta[i] = solver.x[i] - factor * z[i]
    end
    return eta
end
