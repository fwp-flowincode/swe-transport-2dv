# Kelvin-Helmholtz shear-layer instability, 800 x 400 cells (dx = dz = 0.01 m), dt = 0.0025 s, 120 s production run
# with the built-in run! outputs (final-state plot, animation every 1 s, pressure/buoyancy diagnostics in
# output_barotropic/kelvin_helmholtz/). Settings and references: cases.jl; documented figures: plot_kelvin_helmholtz.jl
# and README.md.

using SweTransport2DV
using Printf

include(joinpath(@__DIR__, "..", "common", "helpers.jl"))
include(joinpath(@__DIR__, "cases.jl"))

cfg = kelvin_helmholtz_config(;
    t_end=120.0,
    write_plots=true,
    write_animation=true,
    write_pressure_buoyancy_diagnostics=true,
    # WHY 1 s (formerly 0.2 s): run! keeps every saved state for the animation; 601 states of 800 x 400 cells
    # would need ~6 GB of memory, 121 states ~1.2 GB.
    save_interval_seconds=1.0,
    output_dir=joinpath("output_barotropic", "kelvin_helmholtz"),
)

lambda = cfg.length / KH_PERTURBATION_WAVENUMBER          # m
k = 2.0 * pi / lambda                                     # 1/m
sigma_est = 0.5 * k * (KH_U_TOP - KH_U_BOTTOM)            # 1/s, vortex-sheet growth rate (Chandrasekhar 1981)
@printf("KH setup estimate | k=%.3f 1/m | sigma~%.3f 1/s\n", k, sigma_est)
println("KH lateral boundary condition | periodic x")

model = Model(cfg)
initialize_kelvin_helmholtz_case!(model)

t0 = time()
run!(model)
t1 = time()

tag = case_tag(cfg)
outputs = copy_case_outputs!("kelvin_helmholtz", tag; output_dir=cfg.output_dir)
@printf("CASE=kelvin_helmholtz_%s FULL_WALL_SECONDS=%.2f\n", tag, t1 - t0)
print_case_outputs(outputs; title="Kelvin-Helmholtz outputs")
