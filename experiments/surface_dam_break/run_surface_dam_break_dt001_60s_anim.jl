using SweTransport2DV
using Printf

include(joinpath(@__DIR__, "..", "common", "helpers.jl"))

# Dam-break long-wave references:
# - Stoker (1957), Water Waves
# - Whitham (1974), Linear and Nonlinear Waves

cfg = ModelConfig(
    length=20.0,
    depth=4.0,
    nx=800,
    nz=120,
    dt=0.01,
    nt=6000,
    use_hydrostatic=true,
    # WHY: variable_top_layer (default true for hydrostatic mode, see docs section 13) requires the free surface to
    # stay inside the top layer. Here dz = depth/nz = 0.033 m but eta_jump/2 = 0.04 m already exceeds it before the
    # first step, so the run would stop immediately; the surface dam break needs the legacy fixed-grid treatment.
    variable_top_layer=false,
    momentum_advection_scheme="upwind1",
    rayleigh_damping=0.0,
    apply_froude_limiter=true,
    max_froude=0.9,
    write_plots=true,
    write_animation=true,
    write_pressure_buoyancy_diagnostics=true,
    save_interval_seconds=0.5,
    pressure_solver_method="spectral",
    output_dir=joinpath("output_barotropic", "surface_dam_break"),
)

c_ref = sqrt(cfg.g * cfg.depth)
@printf("Surface dam-break reference speed | c_ref=%.4f m/s\n", c_ref)

model = Model(cfg)
initialize_surface_dam_break!(
    model;
    rho_uniform=1000.0,
    eta_jump=0.08,
    split_fraction=0.5,
)

t0 = time()
run!(model)
t1 = time()

tag = case_tag(cfg)
outputs = copy_case_outputs!("surface_dam_break", tag; output_dir=cfg.output_dir)
@printf("CASE=surface_dam_break_hydro_upwind1_dx0p025_dz0p033_dt0p01_60s FULL_WALL_SECONDS=%.2f\n", t1 - t0)
print_case_outputs(outputs; title="Surface dam-break outputs")
