# Lock exchange: model settings shared by the scripts in this folder (plot_density_hydro_vs_nonhydro_100s.jl and the
# run_*.jl production scripts). The rationale of each setting is documented in the header of
# plot_density_hydro_vs_nonhydro_100s.jl and in README.md.
#
# Lock-exchange gravity-current references:
# - Benjamin (1968), J. Fluid Mech., doi:10.1017/S0022112068002708
# - Shin et al. (2004), J. Fluid Mech., doi:10.1017/S0022112004009172
# - Jankowski dissertation notes in ../../literature/lock_exchange_jaceks_diss.md

using SweTransport2DV

const LE_RHO_HEAVY = 1000.722               # kg/m^3, left (heavy) water, ModelConfig default rho_left
const LE_RHO_LIGHT = 999.972                # kg/m^3, right (light) water, ModelConfig default rho_right
const LE_RAYLEIGH_DAMPING_PER_S = 0.002     # 1/s, minimal Rayleigh damping (removes hydrostatic streaks, see plot script)

"""
    lock_exchange_config(; use_hydrostatic, nx, nz, dt, t_end, kwargs...) -> ModelConfig

Model configuration of the lock exchange in the default 30 m x 4 m channel (gate at 15 m) with `nx` x `nz` cells,
time step `dt` (s) and simulated time `t_end` (s). Keyword arguments override or add `ModelConfig` fields.

Settings (case description): inviscid (nu = 0), reference density rho0 = mean of the two densities, Rayleigh damping
LE_RAYLEIGH_DAMPING_PER_S, `tvd2` momentum and density advection, Froude limiter at 0.9, spectral pressure solver.
The limiter rescales the whole velocity field when max|u| > 0.9 sqrt(g' H) = 0.154 m/s. In the non-hydrostatic runs
(1500 x 200 and 3000 x 400, `tvd2` transport) it is active from about 37 s on, in 16 of 100 one-second samples, where
it caps the velocity in the billows (measured from the production-run logs). The hydrostatic mode uses the variable-thickness top layer (library
default, set explicitly for clarity); the non-hydrostatic mode uses the fixed grid.
Time step: U dt/min(dx, dz) <= 0.05 and, hydrostatic, |w| dt/dz <= 0.5 (time_step_bounds in
../../analysis/lock_exchange/lock_exchange.py), e.g. dt = 0.01 s non-hydrostatic on 1500 x 200, 0.005 s on 3000 x 400,
0.01 s hydrostatic on 600 x 80.
"""
function lock_exchange_config(; use_hydrostatic::Bool, nx::Int, nz::Int, dt::Real, t_end::Real, kwargs...)
    return ModelConfig(;
        nx=nx,
        nz=nz,
        dt=dt,
        nt=round(Int, t_end / dt),
        use_hydrostatic=use_hydrostatic,
        variable_top_layer=use_hydrostatic,
        momentum_advection_scheme="tvd2",
        transport_advection_scheme="tvd2",
        rayleigh_damping=LE_RAYLEIGH_DAMPING_PER_S,
        nu=0.0,                                         # m^2/s
        rho_left=LE_RHO_HEAVY,
        rho_right=LE_RHO_LIGHT,
        rho0=0.5 * (LE_RHO_HEAVY + LE_RHO_LIGHT),       # kg/m^3
        apply_froude_limiter=true,
        max_froude=0.9,
        pressure_solver_method="spectral",
        kwargs...,
    )
end

"""
    lock_exchange_reference_speed(cfg) -> Float64

Bulk front-speed scale U = 0.5 sqrt(g' H) (m/s) with g' = g (rho_left - rho_right) / rho0 (Benjamin 1968).
"""
lock_exchange_reference_speed(cfg::ModelConfig) = 0.5 * sqrt(cfg.g * (cfg.rho_left - cfg.rho_right) / cfg.rho0 * cfg.depth)
