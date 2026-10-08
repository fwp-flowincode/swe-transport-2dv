using SweTransport2DV
using Printf

include(joinpath(@__DIR__, "common.jl"))

cfg = ModelConfig(
    nx=120,
    nz=24,
    dt=0.005,
    nt=14400,
    use_hydrostatic=true,
    momentum_advection_scheme="upwind1",
    rayleigh_damping=0.01,
    apply_froude_limiter=true,
    max_froude=0.9,
    write_plots=true,
    write_animation=true,
    write_pressure_buoyancy_diagnostics=true,
    save_interval_seconds=2.0,
    output_dir=vegetation_output_dir("uniform_flow"),
)

cfg.rho_left = cfg.rho0
cfg.rho_right = cfg.rho0

model = Model(cfg)
fill!(model.state.rho, cfg.rho0)
fill!(model.state.w, 0.0)
fill!(model.state.eta, 0.0)
fill!(model.state.p, 0.0)

z_centers = collect(range(model.dz / 2.0, cfg.depth - model.dz / 2.0, length=cfg.nz))
z0 = 0.05 * cfg.depth
profile = log.(1 .+ 20.0 .* (z_centers .- z0) ./ (cfg.depth - z0))
profile ./= maximum(profile)
u_ref = 0.08
for i in 1:(cfg.nx + 1)
    model.state.u[:, i] .= u_ref .* profile
end
refresh_density_metadata!(model)
set_x_inflow_outflow!(model, u_ref .* profile; open_right=false, right_profile=u_ref .* profile)

stem = VegetationStem(
    base_x=0.5 * cfg.length,
    height=0.8 * cfg.depth,
    diameter=0.03,
    drag_coefficient=2.0,
    stiffness=0.55,
    damping=0.22,
    mass_per_length=0.45,
    sample_points=20,
    source_strength=0.5,
)
attach_vegetation!(model, VegetationModel(stem))

tag = case_tag(cfg)
@printf("Running vegetation uniform-flow case | tag=%s\n", tag)
t0 = time()
run!(model)
t1 = time()
outputs = copy_case_outputs!("vegetation_uniform_flow", tag; output_dir=cfg.output_dir)
@printf("CASE=vegetation_uniform_flow FULL_WALL_SECONDS=%.2f\n", t1 - t0)
print_case_outputs(outputs; title="Vegetation uniform-flow outputs")