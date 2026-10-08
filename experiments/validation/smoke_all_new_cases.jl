using SweTransport2DV
using Printf

# Smoke coverage basis:
# - RTI: Rayleigh (1883), Taylor (1950)
# - Seiche: Gill (1982), Cushman-Roisin and Beckers (2011)
# - Dam-break: Stoker (1957), Whitham (1974)
# - Lock exchange: Benjamin (1968)
# - Wind shear: Price, Weller, and Pinkel (1986)

function make_cfg(; length=8.0, depth=4.0, nx=120, nz=60, dt=0.01, nt=40, use_hydrostatic=false, scheme="tvd2", damping=0.001, output_dir="output_barotropic")
    return ModelConfig(
        length=length,
        depth=depth,
        nx=nx,
        nz=nz,
        dt=dt,
        nt=nt,
        use_hydrostatic=use_hydrostatic,
        momentum_advection_scheme=scheme,
        rayleigh_damping=damping,
        apply_froude_limiter=true,
        max_froude=0.9,
        write_plots=false,
        write_animation=false,
        write_pressure_buoyancy_diagnostics=false,
        pressure_solver_method="spectral",
        output_dir=output_dir,
    )
end

function run_smoke_case!(name::AbstractString, cfg::ModelConfig, initializer::Function)
    cfg_case = copy_config(cfg)
    cfg_case.output_dir = joinpath("output_barotropic", "validation", "smoke", name)
    model = Model(cfg_case)
    initializer(model)
    run!(model)
    @printf("SMOKE=%s OK nt=%d\n", name, cfg_case.nt)
    return nothing
end

run_smoke_case!(
    "lock_exchange",
    make_cfg(),
    m -> initialize_lock_exchange!(m; rho_left=1000.5, rho_right=999.5),
)

run_smoke_case!(
    "rayleigh_taylor",
    make_cfg(damping=0.0),
    m -> initialize_rayleigh_taylor!(
        m;
        rho_top=1000.8,
        rho_bottom=999.2,
        interface_height_fraction=0.8,
        perturbation_amplitude_fraction=0.02,
        perturbation_wavenumber=2,
    ),
)

run_smoke_case!(
    "rising_plume",
    make_cfg(length=10.0, nx=140, nt=50),
    m -> initialize_rising_plume!(
        m;
        rho_ambient=1000.0,
        rho_plume=999.0,
        plume_center_x_fraction=0.5,
        plume_center_z_fraction=0.2,
        plume_sigma_x_fraction=0.08,
        plume_sigma_z_fraction=0.08,
    ),
)

run_smoke_case!(
    "sinking_plume",
    make_cfg(length=10.0, nx=140, nt=50),
    m -> initialize_sinking_plume!(
        m;
        rho_ambient=1000.0,
        rho_plume=1001.0,
        plume_center_x_fraction=0.5,
        plume_center_z_fraction=0.8,
        plume_sigma_x_fraction=0.08,
        plume_sigma_z_fraction=0.08,
    ),
)

run_smoke_case!(
    "jet_impulse",
    make_cfg(length=10.0, nx=140, nt=50),
    m -> initialize_jet_impulse!(
        m;
        rho_uniform=1000.0,
        u0=0.30,
        x_center_fraction=0.5,
        z_center_fraction=0.5,
        sigma_x_fraction=0.10,
        sigma_z_fraction=0.08,
    ),
)

run_smoke_case!(
    "internal_seiche",
    make_cfg(length=8.0, nx=160, nz=80, dt=0.02, nt=80, use_hydrostatic=true, scheme="upwind1", damping=0.0002),
    m -> initialize_internal_seiche!(
        m;
        rho_top=998.5,
        rho_bottom=1001.5,
        interface_height_fraction=0.5,
        interface_tilt_amplitude_fraction=0.03,
    ),
)

run_smoke_case!(
    "surface_dam_break",
    make_cfg(length=20.0, nx=200, dt=0.01, nt=60, use_hydrostatic=true, scheme="upwind1", damping=0.0002),
    m -> initialize_surface_dam_break!(
        m;
        rho_uniform=1000.0,
        eta_jump=0.08,
        split_fraction=0.5,
    ),
)

run_smoke_case!(
    "kelvin_helmholtz",
    make_cfg(length=10.0, nx=160, nz=80, dt=0.005, nt=80, damping=0.0),
    m -> initialize_kelvin_helmholtz!(
        m;
        rho_top=999.5,
        rho_bottom=1000.5,
        interface_height_fraction=0.5,
        interface_perturbation_amplitude_fraction=0.02,
        interface_perturbation_wavenumber=2,
        u_top=0.20,
        u_bottom=-0.20,
    ),
)

run_smoke_case!(
    "wind_shear",
    make_cfg(length=20.0, depth=4.0, nx=200, nz=80, dt=0.01, nt=80, use_hydrostatic=true, scheme="upwind1", damping=0.0005),
    m -> begin
        m.config.wind_stress_x = 0.2
        m.config.wind_mixing_diffusivity = 0.02
        initialize_internal_seiche!(
            m;
            rho_top=999.7,
            rho_bottom=1000.6,
            interface_height_fraction=0.5,
            interface_tilt_amplitude_fraction=0.0,
        )
    end,
)

println("SMOKE_ALL_NEW_CASES PASSED")
