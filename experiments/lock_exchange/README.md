Lock exchange
=============

A gravity current after the removal of a gate between heavy and light water in a
closed channel. The heavy water runs along the bottom and the light water along the
top in opposite directions. The front speed is set by the reduced gravity
g' = g (rho1 - rho2)/rho0 and the depth H: U = Fr sqrt(g' H) with Fr ~ 0.5 (Benjamin 1968).
In non-hydrostatic runs, Kelvin-Helmholtz billows form on the interface between the two layers.

Example
-------

Density 100 s after the removal of the gate between heavy (yellow) and light
(purple) water:
- 30 m x 4 m channel, gate at 15 m, rho1 = 1000.722 kg/m3, rho2 = 999.972 kg/m3;
- 1500 x 200 grid (dx = dz = 0.02 m);
- inviscid (nu = 0), Rayleigh damping r = 0.002 1/s;
- dt = 0.01 s (non-hydrostatic) and 0.002 s (hydrostatic).

Script: `plot_density_hydro_vs_nonhydro_100s.jl`.

**Non-hydrostatic:** Kelvin-Helmholtz billows on the interface behind the front.

![Lock exchange, non-hydrostatic, animation](figures/lock_exchange_nonhydro.gif)

![Lock exchange, non-hydrostatic, t = 100 s](figures/lock_exchange_nonhydro_t100s.png)

**Hydrostatic:** no billows. The fronts are steep and near-vertical, with a sloping,
diffuse interface between them.

![Lock exchange, hydrostatic, t = 100 s](figures/lock_exchange_hydro_t100s.png)

Scripts
-------

- `plot_density_hydro_vs_nonhydro_100s.jl`: the example above (density plots, animations, `.npz` fields).
- `run_dx002_dt001_100s.jl`, `run_dx002_dt001_100s_anim.jl`, `run_dx001_dt005_anim.jl`: production runs.
- `../validation/validate_lock_exchange_front_speed.jl`: front speed compared with theory.
- `../../analysis/lock_exchange/`: Python tools for the bulk reference values and for evaluating
  model runs (front positions, Froude number, nose angle, time-step limits).

Time step
---------

The default keeps U dt/min(dx, dz) <= 0.05. Hydrostatic runs must also keep |w| dt/dz <= 0.5,
because the diagnosed vertical velocity grows with the vertical resolution
(|w| ~ 0.25 U H/dz; 2.5-5 m/s on the 1500 x 200 grid). Larger steps produce spikes and spurious
mixing behind the fronts and, on fine grids, slow the fronts down. See `time_step_bounds` in
`../../analysis/lock_exchange/lock_exchange.py`.
