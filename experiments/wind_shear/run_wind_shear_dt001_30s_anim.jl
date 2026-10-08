using SweTransport2DV
using Printf
using Statistics

include(joinpath(@__DIR__, "..", "common", "helpers.jl"))

# Wind-driven upper-ocean/shear-mixing references:
# - Price, Weller, and Pinkel (1986), J. Geophys. Res., doi:10.1029/JC091iC07p08411
# - Large and Pond (1981), J. Phys. Oceanogr., doi:10.1175/1520-0485(1981)011<0324:OOMFWT>2.0.CO;2
# - Kundu, Cohen, and Dowling (2015), Fluid Mechanics

cfg = ModelConfig(
    length=20.0,
    depth=4.0,
    nx=400,
    nz=160,
    dt=0.01,
    nt=3000,
    use_hydrostatic=true,
    momentum_advection_scheme="upwind1",
    rayleigh_damping=0.0005,
    apply_froude_limiter=true,
    max_froude=0.9,
    write_plots=true,
    write_animation=true,
    write_pressure_buoyancy_diagnostics=true,
    save_interval_seconds=0.2,
    pressure_solver_method="spectral",
    wind_stress_x=0.2,
    wind_mixing_diffusivity=0.02,
    output_dir=joinpath("output_barotropic", "wind_shear"),
)

rho_top = 999.70
rho_bottom = 1000.60
u_ref_surface_cell = cfg.wind_stress_x * (cfg.nt * cfg.dt) / (cfg.rho0 * (cfg.depth / cfg.nz))
@printf(
    "Wind-shear setup | tau_x=%.3f N/m^2 | Kz=%.3f m^2/s | u_ref_surface_cell(%.1fs)=%.4f m/s\n",
    cfg.wind_stress_x,
    cfg.wind_mixing_diffusivity,
    cfg.nt * cfg.dt,
    u_ref_surface_cell,
)

model = Model(cfg)
initialize_internal_seiche!(
    model;
    rho_top=rho_top,
    rho_bottom=rho_bottom,
    interface_height_fraction=0.5,
    interface_tilt_amplitude_fraction=0.0,
)

varz0 = mean([var(@view model.state.rho[:, i]) for i in 1:cfg.nx])

t0 = time()
run!(model)
t1 = time()

u_top_mean = mean(@view model.state.u[cfg.nz, 2:cfg.nx])
varz1 = mean([var(@view model.state.rho[:, i]) for i in 1:cfg.nx])
@printf(
    "Wind-shear diagnostics | u_top_mean=%.4f m/s | varz0=%.6e | varz1=%.6e | delta_varz=%.6e\n",
    u_top_mean,
    varz0,
    varz1,
    varz0 - varz1,
)

tag = case_tag(cfg)
outputs = copy_case_outputs!("wind_shear", tag; output_dir=cfg.output_dir)
@printf("CASE=wind_shear_hydro_upwind1_dx0p05_dz0p025_dt0p01_30s FULL_WALL_SECONDS=%.2f\n", t1 - t0)
print_case_outputs(outputs; title="Wind-shear outputs")
