using SweTransport2DV
using Printf

include(joinpath(@__DIR__, "common.jl"))

function run_case!(label::String, model, prefix::String, title::String)
    @printf("Running wave-damping case: %s\n", label)
    t0 = time()
    history, eta_prev, frame_time, veg_hist, eta_prev_plot, save_interval = run_custom_case!(model)
    outputs = save_custom_case!(prefix, title, model, history, eta_prev, frame_time, veg_hist, eta_prev_plot, save_interval)
    t1 = time()
    amp_final = maximum(abs, model.state.eta)
    amp_peak = maximum(maximum(abs.(st.eta)) for st in history)
    @printf("CASE=%s FULL_WALL_SECONDS=%.2f\n", label, t1 - t0)
    return (; outputs, history, amp_final, amp_peak)
end

function initialize_wave_case!(model, amplitude)
    fill!(model.state.rho, model.config.rho0)
    fill!(model.state.u, 0.0)
    fill!(model.state.w, 0.0)
    fill!(model.state.p, 0.0)

    c = sqrt(model.config.g * model.config.depth)
    x0 = 0.2 * model.config.length
    sigma = 0.1 * model.config.length
    wavelength = 0.4 * model.config.length
    k = 2.0 * pi / wavelength

    x_centers = collect(range(model.dx / 2.0, model.config.length - model.dx / 2.0, length=model.config.nx))
    for i in 1:model.config.nx
        phase = x_centers[i] - x0
        envelope = exp(-0.5 * (phase / sigma)^2)
        model.state.eta[i] = amplitude * envelope * cos(k * phase)
    end

    # Linear barotropic relation for a predominantly right-going packet: u ~= (c/H) * eta.
    ubar = (c / model.config.depth) .* model.state.eta
    for i in 2:model.config.nx
        model.state.u[:, i] .= ubar[i - 1]
    end

    model.state.u[:, 1] .= 0.0
    model.state.u[:, end] .= 0.0
    refresh_density_metadata!(model)
end

base_cfg = ModelConfig(
    length=20.0,
    depth=4.0,
    nx=200,
    nz=48,
    dt=0.0025,
    nt=1000,
    use_hydrostatic=true,
    # WHY: variable_top_layer (default true for hydrostatic mode) requires the free surface to stay inside the top
    # layer. wave_amplitude below is 1.0 m against dz = depth/nz = 0.083 m, so this case needs the legacy fixed grid.
    variable_top_layer=false,
    momentum_advection_scheme="upwind1",
    rayleigh_damping=0.0,
    apply_froude_limiter=false,
    write_plots=true,
    write_animation=true,
    write_pressure_buoyancy_diagnostics=true,
    save_interval_seconds=0.10,
)

reference_cfg = copy_config(base_cfg)
reference_cfg.output_dir = vegetation_output_dir(joinpath("wave_damping", "reference"))
high_canopy_off_cfg = copy_config(base_cfg)
high_canopy_off_cfg.output_dir = vegetation_output_dir(joinpath("wave_damping", "high_canopy_turb_off"))
high_canopy_off_cfg.use_turbulence_model = false
high_canopy_on_cfg = copy_config(base_cfg)
high_canopy_on_cfg.output_dir = vegetation_output_dir(joinpath("wave_damping", "high_canopy_turb_on"))
high_canopy_on_cfg.use_turbulence_model = true
high_canopy_on_cfg.turbulence_mixing_length = 0.08
high_canopy_on_cfg.vegetation_turbulence_strength = 0.35
high_canopy_on_cfg.vegetation_turbulence_spread_cells = 2
high_canopy_on_cfg.max_turbulent_viscosity = 1e-4

reference_model = Model(reference_cfg)
high_canopy_off_model = Model(high_canopy_off_cfg)
high_canopy_on_model = Model(high_canopy_on_cfg)

wave_amplitude = 1.0
initialize_wave_case!(reference_model, wave_amplitude)
initialize_wave_case!(high_canopy_off_model, wave_amplitude)
initialize_wave_case!(high_canopy_on_model, wave_amplitude)

function attach_high_canopy!(model)
    cfg = model.config
    centers = [0.45, 0.50, 0.55] .* cfg.length
    stems = [
        VegetationStem(
            base_x=x,
            height=0.85 * cfg.depth,
            diameter=0.06,
            drag_coefficient=2.8,
            stiffness=0.10,
            damping=0.05,
            mass_per_length=0.40,
            sample_points=18,
            source_strength=3.2,
        ) for x in centers
    ]
    attach_vegetation!(model, VegetationModel(stems...))
end

attach_high_canopy!(high_canopy_off_model)
attach_high_canopy!(high_canopy_on_model)

ref = run_case!(
    "vegetation_wave_damping_reference",
    reference_model,
    "vwd_ref",
    "Right-going free-surface wave packet reference without vegetation",
)
off = run_case!(
    "vegetation_wave_damping_high_canopy_turb_off",
    high_canopy_off_model,
    "vwd_hc_off",
    "Right-going free-surface wave packet with high canopy effect (turbulence OFF)",
)
on = run_case!(
    "vegetation_wave_damping_high_canopy_turb_on",
    high_canopy_on_model,
    "vwd_hc_on",
    "Right-going free-surface wave packet with high canopy effect (turbulence ON)",
)

amp0 = maximum(abs.(ref.history[1].eta))
@printf("Reference eta amplitude initial/final = %.3f / %.3f m\n", amp0, ref.amp_final)
@printf("High-canopy OFF eta amplitude initial/final = %.3f / %.3f m\n", amp0, off.amp_final)
@printf("High-canopy ON eta amplitude initial/final = %.3f / %.3f m\n", amp0, on.amp_final)
@printf("Wave damping final amplitude ratio high-canopy OFF/reference = %.3f\n", off.amp_final / max(ref.amp_final, eps()))
@printf("Wave damping final amplitude ratio high-canopy ON/reference = %.3f\n", on.amp_final / max(ref.amp_final, eps()))
@printf("Wave damping final amplitude ratio high-canopy ON/OFF = %.3f\n", on.amp_final / max(off.amp_final, eps()))
@printf("Wave damping peak amplitude ratio high-canopy OFF/reference = %.3f\n", off.amp_peak / max(ref.amp_peak, eps()))
@printf("Wave damping peak amplitude ratio high-canopy ON/reference = %.3f\n", on.amp_peak / max(ref.amp_peak, eps()))
@printf("Wave damping peak amplitude ratio high-canopy ON/OFF = %.3f\n", on.amp_peak / max(off.amp_peak, eps()))

print_case_outputs(ref.outputs; title="Wave-damping reference outputs")
print_case_outputs(off.outputs; title="Wave-damping high-canopy turbulence-OFF outputs")
print_case_outputs(on.outputs; title="Wave-damping high-canopy turbulence-ON outputs")