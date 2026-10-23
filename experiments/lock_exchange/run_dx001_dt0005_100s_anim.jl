# Lock exchange, non-hydrostatic, 3000 x 400 cells (dx = dz = 0.01 m), dt = 0.005 s, 100 s production run with the
# built-in run! outputs (final-state plot, animation every 1 s, pressure/buoyancy diagnostics in
# output_barotropic/lock_exchange/). Settings and references: cases.jl; documented figures: README.md.
#
# WHY dt = 0.005 s (formerly 0.05 s): the time-step rule U dt/min(dx, dz) <= 0.05 gives dt <= 0.0058 s on dx = 0.01 m.

using SweTransport2DV
using Printf

include(joinpath(@__DIR__, "..", "common", "helpers.jl"))
include(joinpath(@__DIR__, "cases.jl"))

cfg = lock_exchange_config(;
    use_hydrostatic=false,
    nx=3000,
    nz=400,
    dt=0.005,
    t_end=100.0,
    write_plots=true,
    write_animation=true,
    write_pressure_buoyancy_diagnostics=true,
    save_interval_seconds=1.0,   # s, history and animation frame interval (101 frames; about 4 GB of history in memory)
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
