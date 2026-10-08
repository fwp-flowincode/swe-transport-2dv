using SweTransport2DV
using Printf

include(joinpath(@__DIR__, "..", "common", "helpers.jl"))

# Stratified impulse/shear references:
# - Turner (1973), Buoyancy Effects in Fluids
# - Kundu, Cohen, Dowling (2015), Fluid Mechanics

cfg = ModelConfig(
    length=8.0,
    depth=4.0,
    nx=400,
    nz=200,
    dt=0.005,
    nt=4000,
    use_hydrostatic=false,
    momentum_advection_scheme="tvd2",
    rayleigh_damping=0.005,
    apply_froude_limiter=true,
    max_froude=0.9,
    write_plots=true,
    write_animation=true,
    write_pressure_buoyancy_diagnostics=true,
    save_interval_seconds=0.1,
    pressure_solver_method="spectral",
    output_dir=joinpath("output_barotropic", "jet_impulse"),
)

model = Model(cfg)
initialize_jet_impulse!(
    model;
    rho_uniform=1000.0,
    u0=0.25,
    x_center_fraction=0.5,
    z_center_fraction=0.45,
    sigma_x_fraction=0.08,
    sigma_z_fraction=0.05,
)

u_scale = 0.25
t_adv = cfg.length / max(u_scale, 1e-12)
@printf("Jet impulse setup | U0=%.3f m/s | L/U0=%.2f s\n", u_scale, t_adv)

t0 = time()
run!(model)
t1 = time()

tag = case_tag(cfg)
outputs = copy_case_outputs!("jet_impulse", tag; output_dir=cfg.output_dir)
@printf("CASE=jet_impulse_nonhydro_tvd2_dx0p02_dz0p02_dt0p005_20s FULL_WALL_SECONDS=%.2f\n", t1 - t0)
print_case_outputs(outputs; title="Jet impulse outputs")
