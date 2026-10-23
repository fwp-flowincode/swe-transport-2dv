# Lock exchange, non-hydrostatic, 1500 x 200 cells (dx = dz = 0.02 m), dt = 0.01 s, 100 s production run with the
# built-in run! outputs without animation (final-state plot, pressure/buoyancy diagnostics in
# output_barotropic/lock_exchange/). Settings and references: cases.jl; documented figures: README.md.

using SweTransport2DV
using Printf

include(joinpath(@__DIR__, "..", "common", "helpers.jl"))
include(joinpath(@__DIR__, "cases.jl"))

cfg = lock_exchange_config(;
    use_hydrostatic=false,
    nx=1500,
    nz=200,
    dt=0.01,
    t_end=100.0,
    write_plots=true,
    write_animation=false,
    write_pressure_buoyancy_diagnostics=true,
    save_interval_seconds=1.0,   # s, interval of the diagnostics printout
    output_dir=joinpath("output_barotropic", "lock_exchange"),
)

@printf("Lock-exchange reference speed | U_ref=%.4f m/s\n", lock_exchange_reference_speed(cfg))

t0 = time()
model = Model(cfg)
run!(model)
t1 = time()

tag = case_tag(cfg)
outputs = copy_case_outputs!("lock_exchange", tag; output_dir=cfg.output_dir)
@printf("CASE=lock_exchange_%s FULL_WALL_SECONDS=%.2f\n", tag, t1 - t0)
print_case_outputs(outputs; title="Lock-exchange outputs")
