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
    dt::Float64 = 0.0025
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
end

"""
    use_variable_top_layer(cfg) -> Bool

Whether `cfg` runs with the variable-thickness top layer: the explicit `cfg.variable_top_layer` if set, otherwise
`cfg.use_hydrostatic` and no k-epsilon closure (automatic default; the k-epsilon closure assumes layers of equal
thickness and rejects an explicit `variable_top_layer=true`). Evaluated on every use, so changing `use_hydrostatic` or
the closure on a config after creation (as `copy_config` workflows do) keeps the default consistent.
"""
use_variable_top_layer(cfg::ModelConfig) =
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
    geometry::BedGeometry   # staircase bed masks (all cells wet for the flat bed)
    turbulence::Any         # KEpsilonState with turbulence_closure = "k_epsilon", else nothing
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

    geometry = BedGeometry(cfg.bed_level, cfg.nz, cfg.nx, cfg.depth / cfg.nz; periodic_x=cfg.periodic_x)
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
        geometry,
        turbulence_closure(cfg) == "k_epsilon" ? KEpsilonState(cfg.nz, cfg.nx; T=T) : nothing,
    )

    initialize_lock_exchange!(model; rho_left=cfg.rho_left, rho_right=cfg.rho_right)

    return model
end

function set_x_inflow_outflow!(
    model::Model,
    left_profile::AbstractVector{<:Real};
    open_right::Bool=true,
    right_profile::Union{Nothing,AbstractVector{<:Real}}=nothing,
)
    model.x_boundary_u_left_profile = collect(left_profile)
    model.x_boundary_u_right_open = open_right
    model.x_boundary_u_right_profile = right_profile === nothing ? nothing : collect(right_profile)
    _apply_x_boundary_conditions!(model)
    return model
end

function clear_x_inflow_outflow!(model::Model)
    model.x_boundary_u_left_profile = nothing
    model.x_boundary_u_right_open = false
    model.x_boundary_u_right_profile = nothing
    return model
end

function _check_model_config(cfg::ModelConfig)
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
    if (turbulence_closure(cfg) == "k_epsilon" || cfg.bed_roughness_length > 0.0) && cfg.bed_level !== nothing
        throw(ArgumentError("turbulence_closure=\"k_epsilon\" and bed_roughness_length are implemented for the flat bed only (bed_level=nothing)"))
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
    end

    if model.x_boundary_u_right_profile !== nothing
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
        "water_volume" => water_volume(cfg.depth, st.eta, model.dx) - sum(model.geometry.bed_level) * model.dx,
        "constituent_mass" => _constituent_mass(model),
        "boundary_flux" => boundary_normal_flux(st.u, st.w, model.dx, model.dz),
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

function step_forward!(model::Model)
    cfg = model.config
    st = model.state
    ws = model.ws

    _check_model_config(cfg)   # the flags are mutable after construction
    if (model.turbulence !== nothing) != (turbulence_closure(cfg) == "k_epsilon")
        throw(ArgumentError("turbulence_closure cannot be changed across the k_epsilon boundary after Model construction"))
    end
    if cfg.periodic_x != model.pressure_correction_solver.periodic_x
        throw(ArgumentError("periodic_x cannot be changed after Model construction"))
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

    copyto!(ws.eta_old, st.eta)
    if variable_top
        _check_top_layer_thickness(model, ws.eta_old)
        top_layer_face_thickness!(ws.h_top_face, ws.eta_old, model.dz)
    end

    if masked
        _mask_u!(st.u, geo)
        _mask_w!(st.w, geo)
    end

    @timeit model.timers "step/advection_u" advection_term_u!(ws.adv_u, st.u, st.w, model.dx, model.dz, cfg.momentum_advection_scheme, ws.advection_u_ws, ws.interpolation_ws; top_row_advective_form=variable_top, periodic_x=cfg.periodic_x, active=masked ? geo.u_active : nothing, dt=cfg.dt)
    @timeit model.timers "step/advection_w" advection_term_w!(ws.adv_w, st.u, st.w, model.dx, model.dz, cfg.momentum_advection_scheme, ws.advection_w_ws, ws.interpolation_ws, cfg.periodic_x; active=masked ? geo.w_active : nothing, dt=cfg.dt)

    @timeit model.timers "step/hydrostatic_predictor" begin
        solve_hydrostatic_pressure!(ws.p_internal, st.rho, cfg.g, model.dz; top_half_layer=variable_top)
        copyto!(model.last_p_hydro, ws.p_internal)
    end

    @timeit model.timers "step/u_hat_predictor" begin
        grad_x_c!(ws.grad_p_x, ws.p_internal, model.dx; periodic_x=cfg.periodic_x)
        _compute_effective_viscosity_u!(ws.nu_eff_u, model)
        diffusion_term!(ws.diff_u, st.u, ws.nu_eff_u, model.dx, model.dz; periodic_x=cfg.periodic_x, active=masked ? geo.u_active : nothing)

        nz, nxp1 = size(st.u)
        @inbounds for i in 1:nxp1
            for k in 1:nz
                ws.u_hat[k, i] = st.u[k, i] + cfg.dt * (-ws.adv_u[k, i] - (1.0 / cfg.rho0) * ws.grad_p_x[k, i] + ws.diff_u[k, i] + cfg.body_force_x)
            end
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
        if cfg.bed_roughness_length > 0.0
            # Log-law bed friction on the lowest row (explicit): du/dt = -c_b |u1| u1 / dz.
            cb = bed_friction_coefficient(cfg.bed_roughness_length, 0.5 * model.dz)
            @inbounds for i in 1:nxp1
                rate = cb * abs(st.u[1, i]) / model.dz
                cfg.dt * rate > 1.0 && throw(ArgumentError("bed friction too stiff for the explicit step: dt * c_b |u| / dz = $(cfg.dt * rate) > 1"))
                ws.u_hat[1, i] -= cfg.dt * rate * st.u[1, i]
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
    # forcing; see test/TESTS_TECHNICAL_DOCUMENTATION.md section 14) even though the same barotropic solve and
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
        _apply_x_boundary_conditions!(model, ws.u_hat)
        if masked
            _mask_u!(ws.u_hat, geo)
        end
        if variable_top
            solve_variable_thickness!(model.barotropic_solver, st.eta, ws.u_hat, ws.eta_old, ws.h_top_face)
            _check_top_layer_thickness(model, st.eta)
        else
            solve!(model.barotropic_solver, st.eta, ws.u_hat, ws.eta_old)
        end
    end

    @timeit model.timers "step/free_surface_slope" begin
        if cfg.periodic_x
            ws.eta_grad_x[1] = (st.eta[1] - st.eta[end]) / model.dx
            @inbounds for i in 2:nx
                ws.eta_grad_x[i] = (st.eta[i] - st.eta[i - 1]) / model.dx
            end
            ws.eta_grad_x[nx + 1] = ws.eta_grad_x[1]
        else
            @inbounds for i in 1:(nx - 1)
                ws.eta_grad_x[i] = (st.eta[i + 1] - st.eta[i]) / model.dx
            end
        end

        # WHY: corrects ws.u_hat in place (not yet written to st.u) so a following non-hydrostatic projection acts on
        # this surface-corrected predictor, exactly as it already acted on the (pressure-predictor-only) u_hat before.
        ilo, ihi = cfg.periodic_x ? (1, nx + 1) : (2, nx)
        @inbounds for i in ilo:ihi
            for k in 1:cfg.nz
                grad_index = cfg.periodic_x ? i : i - 1
                ws.u_hat[k, i] -= cfg.g * cfg.dt * ws.eta_grad_x[grad_index]
            end
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
            @views ws.w_hat[end, :] .= ws.eta_t

            solve!(model.pressure_correction_solver, ws.u_hat, ws.w_hat, cfg.rho0, cfg.dt, model.dx, model.dz)
            copyto!(ws.p_corr, model.pressure_correction_solver.p_corr)

            grad_x_c!(ws.grad_pc_x, ws.p_corr, model.dx; periodic_x=cfg.periodic_x)
            grad_z_c!(ws.grad_pc_z, ws.p_corr, model.dz)

            @. ws.u_hat = ws.u_hat - cfg.dt * (1.0 / cfg.rho0) * ws.grad_pc_x
            # WHY: only INTERIOR w faces (2:end-1) are corrected, same as the legacy Neumann case. With the
            # homogeneous-Neumann condition this correction is exactly zero at the boundary faces (0 and end), so
            # leaving them uncorrected changes nothing there. With the Dirichlet condition the (unapplied) top-face
            # correction is generally non-zero, but it would only ever affect ws.w_hat, which nothing downstream
            # reads: the model's actual w (state.w) is always reconstructed from state.u's divergence (w_reconstruction
            # below), never copied from ws.w_hat. The Dirichlet condition's effect on the physics is carried entirely
            # through the pressure field's effect on ws.u_hat (grad_pc_x, above) and hence on state.u.
            @views @. ws.w_hat[2:end-1, :] = ws.w_hat[2:end-1, :] - cfg.dt * (1.0 / cfg.rho0) * ws.grad_pc_z[2:end-1, :]
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
        ilo, ihi = cfg.periodic_x ? (1, nx + 1) : (2, nx)
        @inbounds for i in ilo:ihi
            for k in 1:cfg.nz
                st.u[k, i] = ws.u_hat[k, i]
            end
        end

        _apply_x_boundary_conditions!(model)
        if masked
            _mask_u!(st.u, geo)
        end

        div_x_u!(ws.div_u, st.u, model.dx)
    end

    @timeit model.timers "step/w_reconstruction" begin
        @views st.w[1, :] .= 0.0
        if masked
            # w = 0 on the bed face of each column and below it; continuity is integrated upwards from the bed
            @inbounds for i in 1:nx, k in 1:geo.k_bed[i]
                st.w[k, i] = 0.0
            end
        end
        max_mismatch = 0.0
        if variable_top
            # Layers below the top layer have the fixed thickness dz; the top layer closes its own budget
            #   d(eta)/dt + d(h_face*u_top)/dx = w[nz],   i.e. w[nz] = eta_t + div(h_face*u_top),
            # which holds because eta^{n+1} solves the barotropic equation with the same thicknesses.
            @inbounds for i in 1:nx
                for k in geo.k_bed[i]:(cfg.nz - 1)
                    st.w[k + 1, i] = st.w[k, i] - model.dz * ws.div_u[k, i]
                end
                div_hu_top = (ws.h_top_face[i + 1] * st.u[cfg.nz, i + 1] - ws.h_top_face[i] * st.u[cfg.nz, i]) / model.dx
                m = abs(st.w[cfg.nz, i] - div_hu_top - ws.eta_t[i])
                max_mismatch = max(max_mismatch, m)
            end
        else
            @inbounds for i in 1:nx
                for k in geo.k_bed[i]:cfg.nz
                    st.w[k + 1, i] = st.w[k, i] - model.dz * ws.div_u[k, i]
                end
            end

            @inbounds for i in 1:nx
                m = abs(st.w[end, i] - ws.eta_t[i])
                max_mismatch = max(max_mismatch, m)
            end
        end
        model.last_kinematic_mismatch = max_mismatch
        @views st.w[end, :] .= ws.eta_t   # velocity of the fluid at the surface (d eta/dt)
    end

    @timeit model.timers "step/density_transport" begin
        if variable_top
            advect_density_top_layer!(ws.rho_new, st.rho, st.u, st.w, cfg.dt, model.dx, model.dz, ws.density_ws, ws.h_top_face, ws.eta_old, st.eta; scheme=cfg.transport_advection_scheme, active=masked ? geo.wet_cell : nothing, rho_min=model.rho_min_init, rho_max=model.rho_max_init)
        else
            # WHY: fixed grid (variable top layer off, incl. every non-hydrostatic run): advective form in the top row removes
            # the spurious source of the moving top face, see _density_top_row_advective_form! (docs section 4).
            advect_density!(ws.rho_new, st.rho, st.u, st.w, cfg.dt, model.dx, model.dz, ws.density_ws; scheme=cfg.transport_advection_scheme, top_row_advective_form=true, periodic_x=cfg.periodic_x, active=masked ? geo.wet_cell : nothing, rho_min=model.rho_min_init, rho_max=model.rho_max_init)
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
end

function BarotropicSolver(cfg::ModelConfig, dx::Real, dz::Real; T::Type{<:Real}=Float64, geometry::Union{Nothing,BedGeometry}=nothing)
    nx = cfg.nx
    H = T(cfg.depth)
    g = T(cfg.g)
    dt = T(cfg.dt)
    k_bed_face = geometry === nothing ? ones(Int, nx + 1) : copy(geometry.k_bed_face)

    alpha = dt * dt * g * H / (dx * dx)
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
            a_face[f] = dt * dt * g * (H - geometry.bed_face_level[f]) / (dx * dx)
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
        cfg, T(dx), T(dz), nx, periodic_factorization, H, g, dt, lower, main, upper, cp, inv_denom, U_hat, rhs, dp, x,
        zeros(T, nx - 1), zeros(T, nx), zeros(T, nx - 1), zeros(T, nx - 1), zeros(T, nx), k_bed_face,
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
    coeff = dt * dt * g / (dx * dx)   # 1/m; times H_i (m) it gives the dimensionless a_i

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
