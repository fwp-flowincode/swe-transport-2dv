Flow over obstacles (issue #47)
==============================

Steady, frictionless, hydrostatic flow over a bump and over a broad-crested weir, sub-, trans- and supercritical, with a
hydraulic jump. Runs in 1D (wetting-drying step with one layer) and 2DV (8 or 16 layers); the obstacles are real bed
levels (cut cells). Theory and implementation: [docs/TESTS_TECHNICAL_DOCUMENTATION.md](../../docs/TESTS_TECHNICAL_DOCUMENTATION.md),
section 23; tests in `test/test_obstacles.jl`.

Run: `julia --project experiments/obstacles/plot_obstacles.jl [product ...]` (products: `profiles convergence sections
weir shock tracer`, default all; about 10 min for all).

Cases
-----

**SWASHES bump** (Delestre et al. 2013, section 3.1.1). 25 m channel with the bed 0.2 - 0.05 (x - 10)² for 8 < x < 12 m.
The exact solutions are in `src/reference_solutions.jl` (`bump_flow_depth`, `hydraulic_jump_position`):

| case | q (m²/s) | boundaries | exact solution |
|---|---|---|---|
| subcritical | 4.42 | discharge in, level 2 m out | Bernoulli, subcritical everywhere |
| transcritical | 1.53 | discharge in, free outflow | critical depth (q²/g)^(1/3) at the crest, supercritical downstream |
| shock | 0.18 | discharge in, level 0.33 m out | transcritical, hydraulic jump at x = 11.666 m (Bélanger conjugate depths) |

**Broad-crested weir.** Trapezoid of height 0.5 m (crest 18-22 m, 3 m ramps) in a 40 m channel, q = 1 m²/s, and
tailwater depths from 0.85 to 1.4 m:
- free flow: critical depth on the crest, upstream energy head H = 3/2 h_c above the crest, i.e. the weir law
  q = (2/3)^(3/2) sqrt(g) H^(3/2), independent of the tailwater;
- submerged flow: once the tailwater energy exceeds that head, the flow is subcritical everywhere and the upstream
  energy equals the tailwater energy.

**Tracer.** The inflow carries the density 1000.001 kg/m³ (`set_x_inflow_outflow!(...; left_rho)`; its buoyancy is
negligible). The front passes the bump with the steady flow.

Results
-------

![Bump profiles](figures/obstacles_bump_profiles.png)

1D and 2DV lie on the exact solutions in all three regimes (nx = 250).

![Convergence](figures/obstacles_bump_convergence.png)

First-order convergence of the depth: the error halves with dx. In the shock case it is measured at least 0.5 m away
from the jump, and the error of the jump position falls below dx/2.

| nx (dx) | subcritical | transcritical | shock, away from the jump | jump position |
|---|---|---|---|---|
| 125 (0.2 m) | 9.0 mm | 22 mm | 10 mm | 52 mm |
| 250 (0.1 m) | 4.5 mm | 12 mm | 5.4 mm | 17 mm |
| 500 (0.05 m) | 2.2 mm | 6.1 mm | 2.9 mm | 12 mm |

![Sections](figures/obstacles_bump_sections.png)

2DV velocity sections (16 layers). The transcritical flow accelerates over the crest. Under the jump of the shock case
the hydrostatic model forms a roller with backflow at the bed. With nu = 1e-4 m²/s it does not become steady (the
discharge varies by 3 %); with nu = 1e-3 m²/s it does, and the jump then sits about 0.15 m upstream of the
depth-averaged one. The cut cells are drawn at full height.

![Weir](figures/obstacles_weir.png)

The broad-crested weir:
- free flow: the weir law gives the discharge from the model head within 1.25 % at dx = 0.4 m and 0.45 % at 0.2 m, and
  the upstream energy is the same for every free-flow tailwater (modular flow);
- above the modular limit (tailwater energy above 3/2 h_c + p) the upstream energy follows the tailwater energy, as in
  frictionless theory.

![Hydraulic jump](figures/obstacles_shock.gif)

The jump forms on the downstream slope of the bump: the lake at the outlet level is pushed downstream until the
supercritical stream from the crest meets the tailwater at the conjugate depths.

![Tracer](figures/obstacles_tracer.gif)

A tracer front over the bump (2DV, transcritical). The transport stays within [0, 1] (monotone, `tvd2`). Its content
changes every step exactly by the boundary fluxes (`test_verify_tracer_through_bump_monotone_and_conserved`).

Limits
------

- Hydrostatic only. The sharp-crested (thin-plate) weir of issue #47 needs the non-hydrostatic pressure (a curved
  nappe); the open boundaries of the non-hydrostatic mode now exist (issue #49), the weir case is still to be set up.
- A prescribed inflow density (`left_rho`) exists only in the wetting-drying step; the transport of the other modes has
  no flux through the outer faces.
- Frictionless with an open outlet: a lake carrying the discharge is a steady state as well. The transcritical case
  therefore starts with water only upstream of the crest, so that the crest takes control.
