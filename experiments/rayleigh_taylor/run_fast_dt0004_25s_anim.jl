# Rayleigh-Taylor instability, fast-growth case (Atwood 2.0e-3, m = 3), 25 s production run with the built-in run!
# outputs (final-state plot, animation, pressure/buoyancy diagnostics in output_barotropic/rayleigh_taylor_fast/).
# Case definition, settings and references: cases.jl. Documented figures: plot_rayleigh_taylor.jl and README.md.

using SweTransport2DV
using Printf

include(joinpath(@__DIR__, "..", "common", "helpers.jl"))
include(joinpath(@__DIR__, "cases.jl"))

const CASE_NAME = "fast"
const T_END_S = 25.0   # s

cfg = rayleigh_taylor_config(
    CASE_NAME;
    nt=round(Int, T_END_S / RT_CASES[CASE_NAME].dt),
    write_plots=true,
    write_animation=true,
    write_pressure_buoyancy_diagnostics=true,
    save_interval_seconds=1.0,   # s, history and animation frame interval (26 frames; 0.1 s gave 251 frames, ~20 min rendering)
    output_dir=joinpath("output_barotropic", "rayleigh_taylor_fast"),
)

case = RT_CASES[CASE_NAME]
atwood = (case.rho_top - case.rho_bottom) / (case.rho_top + case.rho_bottom)
@printf("RTI fast setup | Atwood=%.6e | top_thickness=%.3f m | bottom_thickness=%.3f m | linear growth rate=%.4f 1/s\n",
    atwood, (1.0 - case.interface_height_fraction) * cfg.depth, case.interface_height_fraction * cfg.depth, linear_growth_rate(CASE_NAME; g=cfg.g))

model = Model(cfg)
initialize_rayleigh_taylor_case!(model, CASE_NAME)

t0 = time()
run!(model)
t1 = time()

tag = case_tag(cfg)
outputs = copy_case_outputs!("rayleigh_taylor_fast", tag; output_dir=cfg.output_dir)
@printf("CASE=rayleigh_taylor_fast_%s FULL_WALL_SECONDS=%.2f\n", tag, t1 - t0)
print_case_outputs(outputs; title="Rayleigh-Taylor fast outputs")
