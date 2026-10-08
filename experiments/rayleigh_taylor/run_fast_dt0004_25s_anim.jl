using SweTransport2DV
using Printf

include(joinpath(@__DIR__, "..", "common", "helpers.jl"))

# Faster RTI-growth setup references:
# - Rayleigh (1883), Proc. London Math. Soc., doi:10.1112/plms/s1-14.1.170
# - Taylor (1950), Proc. Royal Soc. A, doi:10.1098/rspa.1950.0052
# - Sharp (1984), Physica D, doi:10.1016/0167-2789(84)90510-4

const RHO_TOP = 1002.50
const RHO_BOTTOM = 998.50
const INTERFACE_HEIGHT_FRACTION = 0.82
const PERTURBATION_AMPLITUDE_FRACTION = 0.03
const PERTURBATION_WAVENUMBER = 3

cfg = ModelConfig(
    length=8.0,
    depth=4.0,
    nx=400,
    nz=160,
    dt=0.004,
    nt=6250,
    use_hydrostatic=false,
    momentum_advection_scheme="tvd2",
    rayleigh_damping=0.0,
    apply_froude_limiter=true,
    max_froude=0.9,
    write_plots=true,
    write_animation=true,
    write_pressure_buoyancy_diagnostics=true,
    save_interval_seconds=0.1,
    pressure_solver_method="spectral",
    output_dir=joinpath("output_barotropic", "rayleigh_taylor_fast"),
)

atwood = (RHO_TOP - RHO_BOTTOM) / (RHO_TOP + RHO_BOTTOM)
top_thickness = (1.0 - INTERFACE_HEIGHT_FRACTION) * cfg.depth
bottom_thickness = INTERFACE_HEIGHT_FRACTION * cfg.depth
@printf("RTI fast setup | Atwood=%.6e | top_thickness=%.3f m | bottom_thickness=%.3f m | top/bottom=%.3f\n", atwood, top_thickness, bottom_thickness, top_thickness / bottom_thickness)

model = Model(cfg)
initialize_rayleigh_taylor!(
    model;
    rho_top=RHO_TOP,
    rho_bottom=RHO_BOTTOM,
    interface_height_fraction=INTERFACE_HEIGHT_FRACTION,
    perturbation_amplitude_fraction=PERTURBATION_AMPLITUDE_FRACTION,
    perturbation_wavenumber=PERTURBATION_WAVENUMBER,
)

t0 = time()
run!(model)
t1 = time()

tag = case_tag(cfg)
outputs = copy_case_outputs!("rayleigh_taylor_fast", tag; output_dir=cfg.output_dir)
@printf("CASE=rayleigh_taylor_fast_nonhydro_tvd2_dx0p02_dz0p025_dt0p004_25s FULL_WALL_SECONDS=%.2f\n", t1 - t0)
print_case_outputs(outputs; title="Rayleigh-Taylor fast outputs")
