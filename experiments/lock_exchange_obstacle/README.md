Lock exchange with obstacle
===========================

A density-stratified version of the classical dam break with an obstacle (issue #3). In the
original two-phase case a water column collapses in air, hits a small obstacle, is thrown over it
and hits the opposite wall (Martin & Moyce 1952; Koshizuka & Oka 1996; Ubbink 1997; Robert et al.).
Here the column is heavy water and the rest of the tank is filled with light water up to a free
surface. The column collapses under the reduced gravity g' = g (rho_heavy - rho_light)/rho0 and
the light water has to flow back over it: the result is a gravity current that runs into the
obstacle, overflows it, and climbs the far wall. The obstacle is part of the bed (`bed_level`, a
bed level per column; see the plan in `docs/plan_issue_3_lock_exchange_obstacle.md`).

Setup
-----

Geometry of Robert et al. (section 3.1 and Figure 2 of the paper), a = 0.146 m, d = 0.024 m:

| | |
|---|---|
| tank | 4a x 4a = 0.584 m x 0.584 m, closed walls, free surface on top |
| heavy column | a wide, 2a high, at the left wall |
| obstacle | d wide, 2d high, on the bed at x = 2a ... 2a + d (0.292-0.316 m) |
| densities | light water 1000 kg/m³ (= rho0); heavy water 1000 + drho |
| documented case | drho = 10 kg/m³, 584 x 584 cells (1 mm), dt = 0.00075 s |
| density sweep | drho = 1, 10, 50 kg/m³ on 292 x 292 cells (2 mm) |

Settings (`cases.jl`):
- non-hydrostatic; sparse pressure solver, because the bed has solid cells;
- `tvd2` momentum and density advection;
- free-slip walls, as in the reference case;
- nu = 1e-6 m²/s; no Rayleigh damping, no Froude limiter.
- The time step scales with the cell size and with 1/sqrt(drho), so all runs have similar CFL numbers. The largest
  |u| dt/dx + |w| dt/dz is 0.30-0.47, below the `tvd2` limit 0.5.
- The results are compared in dimensionless time t* = t sqrt(g'/a). The time scale sqrt(a/g') is 1.22 s for 10 kg/m³
  (3.86 s for 1 kg/m³, 0.55 s for 50 kg/m³). The original's 0.6 s (in full gravity) corresponds to t* = 4.9.
- Simulated time: t* = 12 (14.6 s for 10 kg/m³), including the wall splash and return flow. The committed figures
  above are from the original runs ending at t* = 7.

Results
-------

![Lock exchange with obstacle, density animation, 10 kg/m3, 1 mm](figures/lock_exchange_obstacle.gif)

![Density at t* = 0, 1, 2, 3, 5 and 7](figures/lock_exchange_obstacle_snapshots.png)

- **t* = 1:** the column collapses; the heavy water forms a nose that runs towards the obstacle.
- **t* = 1.9:** the front reaches the obstacle; the force on it peaks. Part of the current is lifted over it, while the light water flows
  back above the current. Kelvin-Helmholtz billows form on the interface.
- **t* = 3-5:** the current continues behind the obstacle as a turbulent head; the column behind it has drained to a
  thin layer.
- **t* = 6.3:** the front reaches the far wall (x = 3.9a); by t* = 7 the heavy water climbs the wall.

Compared with the two-phase original, the flow is a gravity current rather than a free jet. The light water has to make
room, so there is no tongue thrown over the obstacle through air, and the time scale is sqrt(a/g') instead of
sqrt(a/g).

**Density difference and Boussinesq scaling.** The model is Boussinesq, so in dimensionless form all three density
differences are the same flow, apart from the Reynolds number and the coupling to the free surface. The front positions
and the horizontal pressure force on the obstacle collapse when plotted against t*:

![Front position against t* for 1, 10 and 50 kg/m3](figures/lock_exchange_obstacle_front.png)

![Force on the obstacle against t*](figures/lock_exchange_obstacle_force.png)

| drho (kg/m³) | front at obstacle (t*) | front at 3.9a (t*) | peak force F/(rho0 g' a²) | peak force (N/m) |
|---|---|---|---|---|
| 1 | 1.87 | 6.30 | 0.382 at t* = 1.87 | 0.080 |
| 10 | 1.92 | 6.30 | 0.377 at t* = 1.97 | 0.79 |
| 50 | 1.92 | 6.50 | 0.378 at t* = 1.92 | 4.0 |
| 10 (1 mm grid) | 1.89 | 6.27 | 0.390 at t* = 1.94 | |

The dimensionless front positions agree within about 3 % and the peak forces within about 1.5 %; the finer grid
confirms them. The largest deviation is the slightly slower front for 50 kg/m³. The force is the horizontal pressure
force per unit width: the total pressure (hydrostatic and non-hydrostatic part plus the free-surface pressure
rho0 g eta, which the model applies separately) integrated over the obstacle walls. The small oscillations around
t* < 1 come from free-surface waves.

How large the density difference can be: the solver itself has no limit. The pressure solves are direct, and all three
runs are stable at the same CFL numbers. The limit is physical: the Boussinesq approximation neglects terms of relative
size drho/rho0. It is generally accepted up to a few per cent, so 50 kg/m³ (5 %) is at its edge. For a real
(non-Boussinesq) fluid the heavy and the light current would no longer be symmetric. The documented case uses
10 kg/m³ (1 %).

Limitations
-----------

The original runs stopped at t* = 7 because a spurious surface jet appeared after the wall splash (issue #5).
The cause was predictor transport through closed side walls: the fixed-grid branch advanced the free surface and
projected pressure before imposing the wall velocities. Imposing them only on the final velocity changed the transport
after both solves, leaving a continuity defect in the top cells. The wall enforcement before both solves, already
introduced with the wind-shear correction in PR #7, removes this source; a variable-thickness non-hydrostatic layer is
not needed to fix this defect.

An isolated reversal of that wall-ordering correction reproduces the report on the 2 mm grid with 10 kg/m³:
at t* = 8.85 the top speed is 0.303 m/s and kinetic/released potential energy is 0.972; at t* = 10.82 they reach
0.593 m/s and 1.773. With the correction, the corresponding top speeds are 0.0049 and 0.0080 m/s, and the energy
ratios are 0.747 and 0.740. These comparisons retain the same viscosity, transport and pressure solver.

The script now runs through t* = 12 and records the issue's energy diagnostic: kinetic energy
0.5 rho0 sum(u_c² + w_c²) dx dz against released potential energy g sum((rho - rho0) z) dx dz (initial minus current),
both over wet cells. Progress output reports their ratio; NPZ files contain `kinetic_J_per_m` and
`released_potential_J_per_m` alongside `t_star`. The automated regression covers obstacle and unobstructed wall
return on a 4 mm grid, including energy, top speed, volume, mass, density bounds and continuity. This is a regression
against artificial energy generation, not validation of the detailed late-time mixing or of a finite-amplitude moving
surface: the non-hydrostatic grid still has fixed-thickness layers.

Scripts
-------

- `plot_lock_exchange_obstacle.jl sweep`: drho = 1, 10, 50 kg/m³ on 2 mm, front and force figures (about 2 min per run).
- `plot_lock_exchange_obstacle.jl documented`: drho = 10 kg/m³ on 1 mm, GIF, snapshots and final density
  (about 20 min).
- `cases.jl`: geometry, settings, time step, initial condition and diagnostics (front position, obstacle force).

Output goes to `output_barotropic/lock_exchange_obstacle_figures/`; the figures here are copies.

References
----------

- Martin & Moyce (1952), Phil. Trans. R. Soc. Lond. A 244, 312-324
- Koshizuka & Oka (1996), Nucl. Sci. Eng. 123, 421-434
- Ubbink (1997), Numerical prediction of two fluid systems with sharp interfaces, PhD thesis, Imperial College London
- Robert, Leroyer, Wackers, Visonneau & Queutey, Time-Splitting Procedure for the Resolution of the Volume Fraction for
  Unsteady Flows, LHEEA, École Centrale de Nantes (the paper linked in issue #3)
- Benjamin (1968), J. Fluid Mech. 31, 209-248
