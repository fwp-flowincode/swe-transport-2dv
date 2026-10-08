using SweTransport2DV
using Printf

include(joinpath(@__DIR__, "..", "common", "helpers.jl"))

# Two-layer seiche references:
# - Gill (1982), Atmosphere-Ocean Dynamics
# - Cushman-Roisin and Beckers (2011), Introduction to Geophysical Fluid Dynamics

cfg = ModelConfig(
    length=30.0,
    depth=4.0,
    nx=600,
    nz=120,
    dt=0.05,
    nt=8000,
    use_hydrostatic=false,
    momentum_advection_scheme="tvd2",
    rayleigh_damping=0.001,
    apply_froude_limiter=true,
    max_froude=0.9,
    write_plots=true,
    write_animation=true,
    write_pressure_buoyancy_diagnostics=true,
    save_interval_seconds=1.0,
    pressure_solver_method="spectral",
    output_dir=joinpath("output_barotropic", "internal_seiche"),
)

const RHO_TOP = 999.90
const RHO_BOTTOM = 1000.60
const INTERFACE_HEIGHT_FRACTION = 0.50
const INTERFACE_TILT_AMPLITUDE_FRACTION = 0.03

h1 = (1.0 - INTERFACE_HEIGHT_FRACTION) * cfg.depth
h2 = INTERFACE_HEIGHT_FRACTION * cfg.depth
gp = cfg.g * (RHO_BOTTOM - RHO_TOP) / cfg.rho0
c = sqrt(gp * h1 * h2 / (h1 + h2))
T_theory = 2.0 * cfg.length / c
@printf("Internal seiche estimate | c=%.4f m/s | T_theory=%.2f s\n", c, T_theory)

model = Model(cfg)
initialize_internal_seiche!(
    model;
    rho_top=RHO_TOP,
    rho_bottom=RHO_BOTTOM,
    interface_height_fraction=INTERFACE_HEIGHT_FRACTION,
    interface_tilt_amplitude_fraction=INTERFACE_TILT_AMPLITUDE_FRACTION,
)

t0 = time()
run!(model)
t1 = time()

tag = case_tag(cfg)
outputs = copy_case_outputs!("internal_seiche", tag; output_dir=cfg.output_dir)
@printf("CASE=internal_seiche_nonhydro_tvd2_dx0p05_dz0p033_dt0p05_400s FULL_WALL_SECONDS=%.2f\n", t1 - t0)
print_case_outputs(outputs; title="Internal seiche outputs")
