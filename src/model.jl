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
momentum advection has its own limits (advective Courant number, see the plan, section 3.3).
"""
theta_froude_max(theta::Real) = theta >= 1 ? Inf : (2 * theta - 1) / (2 * (1 - theta))


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
    # flat bed and fixed top layer only: the automatic default of variable_top_layer is false with it, an explicit
    # variable_top_layer=true is rejected). Evaluate with turbulence_closure(cfg).
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
    momentum_advection_scheme::String = "tvd2"
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
`cfg.use_hydrostatic` and no k-epsilon closure (automatic default; the k-epsilon closure assumes layers of equal
thickness and rejects an explicit `variable_top_layer=true`). Evaluated on every use, so changing `use_hydrostatic` or
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
        variable_top_layer=cfg.variable_top_layer,
        top_layer_min_fraction=cfg.top_layer_min_fraction,
        nonhydro_surface_dirichlet=cfg.nonhydro_surface_dirichlet,
        bed_level=cfg.bed_level === nothing ? nothing : copy(cfg.bed_level),
        wetting_drying=cfg.wetting_drying,
        dry_depth_threshold=cfg.dry_depth_threshold,
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
    x_boundary_eta_right::Any   # nothing, or the prescribed water level eta_R (m) at the right boundary face
    x_boundary_q_left::Any      # nothing, or the discharge q (m^2/s) of the left boundary face (profile = shape)
    geometry::BedGeometry   # staircase bed masks (all cells wet for the flat bed)
    turbulence::Any         # KEpsilonState with turbulence_closure = "k_epsilon", else nothing
    wetdry::Any             # WetDryState with wetting_drying = true, else nothing
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
        mass = sum(st.rho[model.geometry.wet_cell]) * model.dx * model.dz
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

function _refresh_density_metadata!(model::Model)
    st = model.state
    cfg = model.config

    model.rho_min_init = minimum(st.rho)
    model.rho_max_init = maximum(st.rho)
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

# Clears the non-hydrostatic pressure of the previous step (ws.p_corr, Pa; wetting-drying q_nh, Pa, and w_nh, m/s), called
# by every initialize_*!. WHY: the incremental projection (van Kan 1986) carries q^n into the next predictor, so a model
# re-initialised after a run would otherwise feel the stale pressure gradient of the old flow.
function _reset_nonhydrostatic_history!(model::Model)
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

function initialize_internal_seiche!(
    model::Model;
    rho_top::Real,
    rho_bottom::Real,
    interface_height_fraction::Real=0.5,
    interface_tilt_amplitude_fraction::Real=0.05,
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
            st.rho[k, i] = z_centers[k] >= z_interface ? rho_top : rho_bottom
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
        TimerOutput(),
        ws,
        nothing,
        nothing,
        false,
        nothing,
        nothing,
        nothing,
        geometry,
        turbulence_closure(cfg) == "k_epsilon" ? KEpsilonState(cfg.nz, cfg.nx; T=T) : nothing,
        cfg.wetting_drying ? WetDryState(cfg; T=T) : nothing,
    )

    initialize_lock_exchange!(model; rho_left=cfg.rho_left, rho_right=cfg.rho_right)
    if cfg.wetting_drying
        _wd_clamp_to_bed!(model)
        _refresh_density_metadata!(model)
    end

    return model
end

"""
    set_x_inflow_outflow!(model, left_profile; left_discharge=nothing, open_right=true, right_profile=nothing,
                          right_level=nothing)

Lateral boundary conditions: velocity `left_profile` (m/s, one value per layer) at the left boundary face, or with
`left_discharge` q (m^2/s) the profile shape scaled to the discharge, u_k = q * left_profile_k / sum_j dZ_j left_profile_j
with the face layer thicknesses dZ of time level n (variable top layer: dz + eta_1; wetting and drying: the cut face
layers; dry layers carry no flow), so that the inflow is q whatever the level at the inlet. At the right boundary face a
wall, a zero-gradient velocity (`open_right=true`), a prescribed velocity `right_profile` (m/s, length nz) or a
prescribed water level `right_level` (m, surface elevation eta at the boundary face x = L; overrides `open_right`).

Water-level boundary (issue #14): the velocity of the boundary face is computed by its own momentum equation: advection
in an open control volume (`_open_right_face_velocity!`), the baroclinic pressure gradient and horizontal diffusion of
face nx (zero gradient), its own implicit vertical column and the surface slope (eta_R - eta_nx)/(dx/2) between the last
cell centre and the boundary, which enters the free-surface solve implicitly like an interior face. Requires the
hydrostatic mode, `implicit_vertical_terms=true` and no `periodic_x`. With the variable top layer (and with wetting and
drying) the top layer of the boundary face has the level eta_R.
"""
function set_x_inflow_outflow!(
    model::Model,
    left_profile::AbstractVector{<:Real};
    left_discharge::Union{Nothing,Real}=nothing,
    open_right::Bool=true,
    right_profile::Union{Nothing,AbstractVector{<:Real}}=nothing,
    right_level::Union{Nothing,Real}=nothing,
)
    if left_discharge !== nothing
        isfinite(left_discharge) || throw(ArgumentError("left_discharge must be finite"))
        all(>=(0.0), left_profile) && any(>(0.0), left_profile) ||
            throw(ArgumentError("left_discharge needs a profile shape with values >= 0, not all zero"))
    end
    model.wetdry === nothing || (model.wetdry.left_velocity = nothing)
    model.x_boundary_u_left_profile = collect(left_profile)
    model.x_boundary_q_left = left_discharge === nothing ? nothing : Float64(left_discharge)
    _set_right_boundary!(model, open_right, right_profile, right_level)
    # WHY: the discharge scaling uses the level of time n in ws.eta_old (set by every step); before the first step it
    # is the initial level
    copyto!(model.ws.eta_old, model.state.eta)
    _apply_x_boundary_conditions!(model)
    return model
end

"""
    set_x_inflow_outflow!(model, left_velocity::Function; open_right=false, right_profile=nothing, right_level=nothing)

Wetting-drying mode: time-dependent velocity at the left boundary face, `left_velocity(t)` (m/s; scalar or one value per
layer) at the model time t (s) of the end of each step; dry layers of the face carry no flow. Right boundary as in the
method with a velocity profile.
"""
function set_x_inflow_outflow!(
    model::Model,
    left_velocity::Function;
    open_right::Bool=false,
    right_profile::Union{Nothing,AbstractVector{<:Real}}=nothing,
    right_level::Union{Nothing,Real}=nothing,
)
    model.wetdry === nothing && throw(ArgumentError("a time-dependent boundary velocity requires ModelConfig(wetting_drying=true)"))
    model.wetdry.left_velocity = left_velocity
    model.x_boundary_u_left_profile = nothing
    model.x_boundary_q_left = nothing
    _set_right_boundary!(model, open_right, right_profile, right_level)
    return model
end

function _set_right_boundary!(model::Model, open_right::Bool, right_profile, right_level)
    if right_level !== nothing
        right_profile === nothing || throw(ArgumentError("right_profile and right_level exclude each other"))
        isfinite(right_level) || throw(ArgumentError("right_level must be finite"))
        _check_right_level_config(model.config)
    end
    model.x_boundary_u_right_open = right_level === nothing && open_right
    model.x_boundary_u_right_profile = right_profile === nothing ? nothing : collect(right_profile)
    model.x_boundary_eta_right = right_level === nothing ? nothing : Float64(right_level)
    return nothing
end

function _check_right_level_config(cfg::ModelConfig)
    cfg.periodic_x && throw(ArgumentError("a water-level boundary (right_level) is not possible with periodic_x=true"))
    cfg.use_hydrostatic || throw(ArgumentError("a water-level boundary (right_level) is implemented for the hydrostatic mode only"))
    # WHY implicit vertical terms: the boundary face needs its own vertical viscosity and friction (the explicit
    # diffusion_term! leaves the boundary points out); its implicit column provides them
    cfg.implicit_vertical_terms || throw(ArgumentError("a water-level boundary (right_level) requires implicit_vertical_terms=true"))
    return nothing
end

function clear_x_inflow_outflow!(model::Model)
    model.wetdry === nothing || (model.wetdry.left_velocity = nothing)
    model.x_boundary_u_left_profile = nothing
    model.x_boundary_q_left = nothing
    model.x_boundary_u_right_open = false
    model.x_boundary_u_right_profile = nothing
    model.x_boundary_eta_right = nothing
    return model
end

function _check_model_config(cfg::ModelConfig)
    if !(THETA_MIN <= cfg.theta <= THETA_MAX)
        throw(ArgumentError("theta must be in [$(THETA_MIN), $(THETA_MAX)] (theta < 1/2 is unstable), got $(cfg.theta)"))
    end
    cfg.wetting_drying && _check_wetting_drying_config(cfg)
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
    if turbulence_closure(cfg) == "k_epsilon" && use_variable_top_layer(cfg)
        throw(ArgumentError("turbulence_closure=\"k_epsilon\" does not support variable_top_layer=true"))
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
    cfg.periodic_x && throw(ArgumentError("wetting_drying=true does not support periodic_x=true"))
    cfg.momentum_advection_scheme in ("upwind1", "tvd2") || throw(ArgumentError("momentum_advection_scheme must be \"upwind1\" or \"tvd2\", got \"$(cfg.momentum_advection_scheme)\""))
    turbulence_closure(cfg) == "none" || throw(ArgumentError("wetting_drying=true does not support a turbulence closure"))
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

    if model.x_boundary_u_left_profile === nothing
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
            @views u[:, 1] .*= model.x_boundary_q_left / transport
        end
    end

    if model.x_boundary_eta_right !== nothing
        # water-level boundary: u of the boundary face comes from its own momentum equation (step_forward!)
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
# bed face, the top layer dz + eta of level n (ws.eta_old) with the variable top layer.
function _left_face_transport(model::Model, shape::AbstractVector{<:Real})
    nz, dz = model.config.nz, model.dz
    h_top = use_variable_top_layer(model.config) ? dz + model.ws.eta_old[1] : dz
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
    # the boundary face of a water-level boundary has its own column (set_x_inflow_outflow!(...; right_level))
    flo, fhi = cfg.periodic_x ? (1, nx) : (2, model.x_boundary_eta_right === nothing ? nx : nx + 1)
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

function step_forward!(model::Model)
    cfg = model.config
    st = model.state
    ws = model.ws

    _check_model_config(cfg)   # the flags are mutable after construction
    if cfg.wetting_drying != (model.wetdry !== nothing)
        throw(ArgumentError("wetting_drying cannot be changed after Model construction"))
    end
    cfg.wetting_drying && return _step_forward_wetting_drying!(model)
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
    # water-level boundary at the right (set_x_inflow_outflow!(...; right_level)): eta_R (m) at the boundary face
    eta_right = model.x_boundary_eta_right
    level_bc = eta_right !== nothing
    level_bc && _check_right_level_config(cfg)

    copyto!(ws.eta_old, st.eta)
    if variable_top
        _check_top_layer_thickness(model, ws.eta_old)
        top_layer_face_thickness!(ws.h_top_face, ws.eta_old, model.dz)
        if level_bc
            model.dz + eta_right >= cfg.top_layer_min_fraction * model.dz ||
                throw(ArgumentError("right_level = $(eta_right) m leaves the top layer (dz = $(model.dz) m)"))
            ws.h_top_face[cfg.nx + 1] = model.dz + eta_right   # the level of the boundary face is known
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
    @timeit model.timers "step/advection_u" advection_term_u!(ws.adv_u, st.u, st.w, model.dx, model.dz, cfg.momentum_advection_scheme, ws.advection_u_ws, ws.interpolation_ws; top_row_advective_form=true, periodic_x=cfg.periodic_x, active=masked ? geo.u_active : nothing, dt=cfg.dt, vertical=!implicit, open_right=level_bc, h_top_vertical=(implicit && variable_top) ? ws.h_top_face : nothing)
    @timeit model.timers "step/advection_w" advection_term_w!(ws.adv_w, st.u, st.w, model.dx, model.dz, cfg.momentum_advection_scheme, ws.advection_w_ws, ws.interpolation_ws, cfg.periodic_x; active=masked ? geo.w_active : nothing, dt=cfg.dt)

    @timeit model.timers "step/hydrostatic_predictor" begin
        solve_hydrostatic_pressure!(ws.p_internal, st.rho, cfg.g, model.dz; top_half_layer=variable_top)
        copyto!(model.last_p_hydro, ws.p_internal)
    end

    @timeit model.timers "step/u_hat_predictor" begin
        grad_x_c!(ws.grad_p_x, ws.p_internal, model.dx; periodic_x=cfg.periodic_x)
        _compute_effective_viscosity_u!(ws.nu_eff_u, model)
        diffusion_term!(ws.diff_u, st.u, ws.nu_eff_u, model.dx, model.dz; periodic_x=cfg.periodic_x, active=masked ? geo.u_active : nothing, vertical=!implicit)

        nz, nxp1 = size(st.u)
        if level_bc
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
        @views @. ws.rho_w = 0.5 * (st.rho[2:end, :] + st.rho[1:end-1, :])
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
            if level_bc
                slope = _right_level_slope(ws.eta_old, eta_right, model.dx)
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
            solve_effective_depth!(bsolver, st.eta, ws.eta_old, ws.H_face; periodic_x=cfg.periodic_x, right_level=eta_right)
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
        if level_bc
            slope = _right_level_slope(st.eta, eta_right, model.dx)
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

            @. ws.u_hat = ws.u_hat - dt_q * (1.0 / cfg.rho0) * ws.grad_pc_x
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
        ilo, ihi = cfg.periodic_x ? (1, nx + 1) : (2, level_bc ? nx + 1 : nx)
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
            # turbulent density diffusion K = nu_t / sigma_rho at the w-faces (implicit, conservative; k-epsilon runs on the
            # fixed grid only, so every layer has the thickness dz)
            ke = model.turbulence
            implicit_vertical_diffusion!(ws.rho_new, (k, i) -> 0.5 * (ke.nu_t[k, i] + ke.nu_t[k + 1, i]) / KE_SIGMA_RHO,
                (k, i) -> model.dz, cfg.dt, model.dz, ke)
        end
        copyto!(st.rho, ws.rho_new)
        _correct_constituent_mass!(model)
    end

    if model.turbulence !== nothing
        model.vegetation === nothing && fill!(model.turbulence.veg_work, 0.0)
        @timeit model.timers "step/k_epsilon" k_epsilon_step!(model.turbulence, st.u, st.w, st.rho, cfg.dt, model.dx, model.dz, cfg.nu,
            cfg.g, cfg.rho0; periodic_x=cfg.periodic_x, bed_roughness=cfg.bed_roughness_length)
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
a_{nx+1} = 2 theta^2 dt^2 g H_{nx+1} / dx^2 and a_{nx+1} right_level moves to the right-hand side (`_right_level_slope`).

Symmetric, diagonally dominant (H_f >= 0): Thomas algorithm, or with periodic_x the cyclic system by the
Sherman-Morrison formula (Press et al. 2007, Numerical Recipes, 3rd ed., section 2.7.2).
PERF: coefficients rebuilt every step (H_f depends on the friction and the flow), O(nx), no allocation.
"""
function solve_effective_depth!(solver::BarotropicSolver, eta::AbstractVector{<:Real}, eta_old::AbstractVector{<:Real},
    H_face::AbstractVector{<:Real}; periodic_x::Bool=false, right_level::Union{Nothing,Real}=nothing)
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
        a_bc = 2.0 * coeff * H_face[nx + 1]   # WHY 2: the level lies half a cell from the last centre
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
