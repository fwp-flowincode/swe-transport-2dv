using SweTransport2DV
using Printf

include(joinpath(@__DIR__, "..", "common", "helpers.jl"))

# Buoyant plume references:
# - Morton, Taylor, Turner (1956), Proc. Royal Soc. A, doi:10.1098/rspa.1956.0011
# - Turner (1973), Buoyancy Effects in Fluids

cfg = ModelConfig(
    length=8.0,
    depth=4.0,
    nx=400,
    nz=200,
    dt=0.01,
    nt=4000,
    use_hydrostatic=false,
    momentum_advection_scheme="tvd2",
    rayleigh_damping=0.0,
    apply_froude_limiter=true,
    max_froude=0.9,
    write_plots=true,
    write_animation=true,
    write_pressure_buoyancy_diagnostics=true,
    save_interval_seconds=0.2,
    pressure_solver_method="spectral",
    output_dir=joinpath("output_barotropic", "sinking_plume"),
)

drho = 1000.50 - 1000.00
b0 = cfg.g * drho / cfg.rho0
@printf("Sinking plume setup | Delta_rho=%.3f kg/m^3 | b0=%.4e m/s^2\n", drho, b0)

model = Model(cfg)
initialize_sinking_plume!(
    model;
    rho_ambient=1000.00,
    rho_plume=1000.50,
    plume_center_x_fraction=0.5,
    plume_center_z_fraction=0.8,
    plume_sigma_x_fraction=0.06,
    plume_sigma_z_fraction=0.05,
)

t0 = time()
run!(model)
t1 = time()

tag = case_tag(cfg)
outputs = copy_case_outputs!("sinking_plume", tag; output_dir=cfg.output_dir)
@printf("CASE=sinking_plume_nonhydro_tvd2_dx0p02_dz0p02_dt0p01_40s FULL_WALL_SECONDS=%.2f\n", t1 - t0)
print_case_outputs(outputs; title="Sinking plume outputs")
