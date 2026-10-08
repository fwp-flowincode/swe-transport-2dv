using SweTransport2DV
using Printf

include(joinpath(@__DIR__, "..", "common", "helpers.jl"))

# Lock-exchange gravity-current references:
# - Benjamin (1968), J. Fluid Mech., doi:10.1017/S0022112068002708
# - Shin et al. (2004), J. Fluid Mech., doi:10.1017/S0022112004009172
# - Jankowski dissertation notes in ../../literature/lock_exchange_jaceks_diss.md

cfg = ModelConfig(
    nx=1500,
    nz=200,
    dt=0.01,
    nt=10000,
    use_hydrostatic=false,
    momentum_advection_scheme="tvd2",
    rayleigh_damping=0.01,
    apply_froude_limiter=true,
    max_froude=0.9,
    write_plots=true,
    write_animation=true,
    write_pressure_buoyancy_diagnostics=true,
    save_interval_seconds=1.0,
    pressure_solver_method="spectral",
    output_dir=joinpath("output_barotropic", "lock_exchange"),
)

gprime = cfg.g * (cfg.rho_left - cfg.rho_right) / cfg.rho0
u_ref = 0.5 * sqrt(gprime * cfg.depth)
@printf("Lock-exchange reference speed | g'=%.4e m/s^2 | U_ref=%.4f m/s\n", gprime, u_ref)

t0 = time()
model = Model(cfg)
run!(model)
t1 = time()

tag = case_tag(cfg)
outputs = copy_case_outputs!("lock_exchange", tag; output_dir=cfg.output_dir)
@printf("CASE=lock_exchange_nonhydro_tvd2_dx0p02_dt0p01_100s_anim FULL_WALL_SECONDS=%.2f\n", t1 - t0)
print_case_outputs(outputs; title="Lock-exchange outputs")
