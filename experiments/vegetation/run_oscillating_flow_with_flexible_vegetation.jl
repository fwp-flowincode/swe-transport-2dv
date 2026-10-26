using SweTransport2DV
using Printf

include(joinpath(@__DIR__, "common.jl"))

# Ported to the vegetation model of issue #8 (physical stem parameters: bending stiffness EI, lateral spacing of the
# row of stems; no tuning factors). Qualitative only: the stems are quasi-static (no mass, added mass or damping); the
# dynamic stem model and a quantitative reference are step 3 of docs/plan_issue_8_vegetation.md (separate issue).

cfg = ModelConfig(
    length=12.0,
    depth=2.5,
    nx=120,
    nz=20,
    dt=0.01,
    nt=1200,
    use_hydrostatic=true,
    momentum_advection_scheme="upwind1",
    rayleigh_damping=0.0,
    apply_froude_limiter=false,
    write_plots=true,
    write_animation=true,
    write_pressure_buoyancy_diagnostics=true,
    save_interval_seconds=0.5,
    output_dir=vegetation_output_dir("oscillating_flow"),
)

model = Model(cfg)
fill!(model.state.rho, cfg.rho0)
fill!(model.state.u, 0.0)
fill!(model.state.w, 0.0)
fill!(model.state.eta, 0.0)
fill!(model.state.p, 0.0)
refresh_density_metadata!(model)

stem = VegetationStem(
    base_x=0.5 * cfg.length,
    height=0.85 * cfg.depth,
    diameter=0.03,              # m
    drag_coefficient=1.2,       # -, circular cylinder
    bending_stiffness=0.05,     # N m^2
    lateral_spacing=0.2,        # m, row of stems
)
attach_vegetation!(model, VegetationModel(stem))

period = 3.0
forcing_amplitude = 1.5

history, eta_prev_history, history_time, vegetation_history, eta_prev_for_plot, save_interval = run_custom_case!(model; on_step=(m, i, t) -> begin
    m.config.wind_stress_x = forcing_amplitude * sin(2.0 * pi * t / period)
end)

max_tip_deflection = maximum(abs(stem_tip_deflection(frame.stems[1])) for frame in vegetation_history)

tag = case_tag(cfg)
@printf("Running vegetation oscillating-flow case | tag=%s\n", tag)
t0 = time()
outputs = save_custom_case!("vegetation_oscillating_flow", "Time-oscillating flow with a single flexible stem", model, history, eta_prev_history, history_time, vegetation_history, eta_prev_for_plot, save_interval)
t1 = time()
@printf("CASE=vegetation_oscillating_flow FULL_WALL_SECONDS=%.2f\n", t1 - t0)
print_case_outputs(outputs; title=@sprintf("Vegetation oscillating-flow outputs (peak tip deflection=%.3f m)", max_tip_deflection))