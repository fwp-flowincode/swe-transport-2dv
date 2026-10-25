using SweTransport2DV
using Printf
using Statistics

# Wind-driven shear/mixing references:
# - Price, Weller, and Pinkel (1986), J. Geophys. Res., doi:10.1029/JC091iC07p08411
# - Large and Pond (1981), J. Phys. Oceanogr., doi:10.1175/1520-0485(1981)011<0324:OOMFWT>2.0.CO;2
#
# Reference checks used here:
# 1) Surface-cell acceleration scaling: u ~ tau_x * t / (rho0 * dz)
# 2) Wind-induced surface mixing reduces vertical density variance.

cfg = ModelConfig(
    length=20.0,
    depth=4.0,
    nx=200,
    nz=80,
    dt=0.01,
    nt=3000,
    use_hydrostatic=true,
    momentum_advection_scheme="upwind1",
    rayleigh_damping=0.0005,
    apply_froude_limiter=true,
    max_froude=0.9,
    write_plots=false,
    write_animation=false,
    write_pressure_buoyancy_diagnostics=false,
    pressure_solver_method="spectral",
    wind_stress_x=0.2,
    wind_mixing_diffusivity=0.02,
    output_dir=joinpath("output_barotropic", "validation", "wind_shear_transport_mixing"),
)

model = Model(cfg)
initialize_internal_seiche!(
    model;
    rho_top=999.7,
    rho_bottom=1000.6,
    interface_height_fraction=0.5,
    interface_tilt_amplitude_fraction=0.0,
)

varz0 = mean([var(@view model.state.rho[:, i]) for i in 1:cfg.nx])
for n in 1:cfg.nt
    step_forward!(model)
end

u_top_mean = mean(@view model.state.u[cfg.nz, 2:cfg.nx])
u_ref_surface_cell = cfg.wind_stress_x * (cfg.nt * cfg.dt) / (cfg.rho0 * (cfg.depth / cfg.nz))
ratio = u_top_mean / u_ref_surface_cell

varz1 = mean([var(@view model.state.rho[:, i]) for i in 1:cfg.nx])
delta_varz = varz0 - varz1

@printf(
    "VALIDATION=wind_shear_transport_mixing u_top=%.4f u_ref=%.4f ratio=%.3f varz0=%.6e varz1=%.6e delta_varz=%.6e\n",
    u_top_mean,
    u_ref_surface_cell,
    ratio,
    varz0,
    varz1,
    delta_varz,
)

# Calibrated against this model setup with mild damping and resolved vertical structure.
if !(0.70 <= ratio <= 1.10)
    error(@sprintf("Wind-shear transport validation failed: ratio %.3f outside [0.70, 1.10]", ratio))
end

# WHY (threshold 5e-4, after 5e-3 and 1.5e-3): this case runs with the (default) variable top layer, under which the
# interior mixing measured here is the genuine wind-driven overturning circulation only. With the legacy fixed top
# face, a moving surface's zero top-face flux acted as an additional, non-physical density source (see
# test/TESTS_TECHNICAL_DOCUMENTATION.md section 13) that inflated delta_varz well past 5e-3 at this same
# duration/forcing. A second, smaller artefact of the same kind remained until issue #4: the wind increment was added
# to u after the barotropic solve, so the top-layer budget did not close in the two wall columns (docs section 10);
# it raised delta_varz from about 7.0e-4 to 2.1e-3. Extending the run or raising the wind stress to recover more
# separation runs into the u ~ tau*t/(rho0*dz) scaling above breaking down (the ratio bound), so the threshold is
# lowered instead, with headroom (about 1.4x) below the value observed on this grid.
const DELTA_VARZ_MIN = 5e-4   # (kg/m^3)^2
if delta_varz <= DELTA_VARZ_MIN
    error(@sprintf("Wind-shear mixing validation failed: variance reduction %.6e <= %.1e", delta_varz, DELTA_VARZ_MIN))
end

println("Wind-shear transport/mixing validation passed.")
