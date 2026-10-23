using SweTransport2DV
using Printf

include(joinpath(@__DIR__, "..", "common", "helpers.jl"))

# Kelvin-Helmholtz references:
# - Kelvin (1871), Philosophical Magazine
# - Helmholtz (1868), Monatsberichte der Koniglichen Akademie der Wissenschaften
# - Chandrasekhar (1981), Hydrodynamic and Hydromagnetic Stability

cfg = ModelConfig(
    length=8.0,
    depth=4.0,
    nx=400,
    nz=200,
    dt=0.01,
    nt=3000,
    use_hydrostatic=false,
    periodic_x=true,
    momentum_advection_scheme="tvd2",
    rayleigh_damping=0.001,
    apply_froude_limiter=true,
    max_froude=0.9,
    write_plots=true,
    write_animation=true,
    write_pressure_buoyancy_diagnostics=true,
    save_interval_seconds=0.2,
    pressure_solver_method="sparse",
    output_dir=joinpath("output_barotropic", "kelvin_helmholtz"),
)

du = 0.40
lambda = cfg.length / 2.0
k = 2.0 * pi / lambda
sigma_est = 0.5 * k * du
@printf("KH setup estimate | k=%.3f 1/m | sigma~%.3f 1/s\n", k, sigma_est)
println("KH lateral boundary condition | periodic x")

model = Model(cfg)
initialize_kelvin_helmholtz!(
    model;
    rho_top=999.95,
    rho_bottom=1000.05,
    interface_height_fraction=0.5,
    interface_perturbation_amplitude_fraction=0.02,
    interface_perturbation_wavenumber=2,
    u_top=0.20,
    u_bottom=-0.20,
)

t0 = time()
run!(model)
t1 = time()

tag = case_tag(cfg)
outputs = copy_case_outputs!("kelvin_helmholtz", tag; output_dir=cfg.output_dir)
@printf("CASE=kelvin_helmholtz_nonhydro_tvd2_dx0p02_dz0p02_dt0p01_30s FULL_WALL_SECONDS=%.2f\n", t1 - t0)
print_case_outputs(outputs; title="Kelvin-Helmholtz outputs")
