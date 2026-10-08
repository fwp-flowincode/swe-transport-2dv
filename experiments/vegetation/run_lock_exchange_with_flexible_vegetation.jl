using SweTransport2DV
using Printf

include(joinpath(@__DIR__, "common.jl"))

cfg = ModelConfig(
    nx=600,
    nz=120,
    dt=0.01,
    nt=12000,
    use_hydrostatic=false,
    momentum_advection_scheme="tvd2",
    rayleigh_damping=0.01,
    apply_froude_limiter=true,
    max_froude=0.9,
    write_plots=true,
    write_animation=true,
    write_pressure_buoyancy_diagnostics=true,
    pressure_solver_method="spectral",
    save_interval_seconds=2.0,
    output_dir=vegetation_output_dir("lock_exchange"),
)

model = Model(cfg)
initialize_lock_exchange!(model)

stem = VegetationStem(
    base_x=17.0,
    height=0.9 * cfg.depth,
    diameter=0.04,
    drag_coefficient=1.8,
    stiffness=0.60,
    damping=0.22,
    mass_per_length=0.45,
    sample_points=18,
    source_strength=0.7,
)
attach_vegetation!(model, VegetationModel(stem))

tag = case_tag(cfg)
@printf("Running vegetation lock-exchange case | tag=%s\n", tag)
t0 = time()
run!(model)
t1 = time()
final_tip_deflection = model.vegetation.stems[1].deflection[end]
outputs = copy_case_outputs!("vegetation_lock_exchange", tag; output_dir=cfg.output_dir)
@printf("CASE=vegetation_lock_exchange FULL_WALL_SECONDS=%.2f\n", t1 - t0)
print_case_outputs(outputs; title=@sprintf("Vegetation lock-exchange outputs (final tip deflection=%.3f m)", final_tip_deflection))