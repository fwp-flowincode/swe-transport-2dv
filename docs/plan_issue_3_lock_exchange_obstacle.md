# Plan: lock exchange with obstacle (issue #3)

Status: approved 2026-10-24 with the decisions in section 9.

## 1. Goal

Reproduce, with two water masses of different density, the classical two-phase benchmark in which a collapsing water
column hits a small obstacle, flows over it and hits the opposite wall (Koshizuka & Oka 1996; the paper linked in the
issue uses the same case). Requirements from the issue:

- non-hydrostatic mode, `tvd2` for momentum and transport, a sufficiently fine grid;
- find out how large the density difference can be without causing problems;
- first implement a variable bed geometry (the obstacle is part of the bed).

## 2. Reference case and its density-stratified counterpart

Geometry from Robert et al. (section 3.1 and Figure 2 of the paper linked in the issue, after Ubbink 1997), with
a = 0.146 m and d = 0.024 m:

| | two-phase original | this model |
|---|---|---|
| tank | square, 4a x 4a = 0.584 m x 0.584 m, solid walls, top open to the air | same, closed side walls and bed, free surface on top |
| water column | a = 0.146 m wide, 2a = 0.292 m high, at the left wall | heavy water, same size |
| rest of the tank | air | light water, filling the whole tank to the top |
| obstacle | d = 0.024 m wide, 2d = 0.048 m high, on the bed, a gap of a after the column (x = 0.292-0.316 m) | same block, as part of the bed |
| walls | slip condition on all solid walls | free-slip, as the model's outer walls |
| simulated time | 0.6 s (table at 0.2 s) | see section 7 |

The paper's text gives the sizes in cm (146 cm, 584 cm). The classical case is in mm, and the paper's own timing
confirms that: all times scale with sqrt(a/g), and the falling column reaches the obstacle after about 0.1 s (its
Figure 3a), which fits a = 0.146 m. With a ten times larger a everything would happen sqrt(10) = 3.2 times later.
The cm in the text are taken as a typo.

All edges (0.146, 0.292, 0.316, 0.048 and 0.584 m) lie on cell faces for dx = dz = 2 mm (292 x 292 cells; obstacle
12 x 24 cells) and for 1 mm (584 x 584 cells).

The physics differs from the original in one important way. In the original, air is about 800 times lighter than
water, so the water column collapses under full gravity g and the air hardly matters. Here the column is surrounded by
water of almost the same density. It collapses under the reduced gravity g' = g (rho_heavy - rho_light) / rho0 and must
displace the light water, which flows back over it. The result is a gravity current (a partial-depth lock release)
that runs into the obstacle, partly overflows it and reflects at the far wall, with Kelvin-Helmholtz billows on the
interface. The time scale is sqrt(a/g') instead of sqrt(a/g).

## 3. How large can the density difference be?

The model is Boussinesq: momentum is divided by the constant reference density rho0, and density differences only
enter through the buoyancy and the hydrostatic pressure. Two separate limits follow:

1. **Physical validity.** The Boussinesq approximation neglects terms of relative size (rho_heavy - rho_light)/rho0.
   It is commonly accepted up to a few per cent. Beyond that the model still runs, but the results are those of a
   Boussinesq fluid, not of the real one (for example, the front speeds of the heavy and the light current become
   unequal in reality, but stay equal in the model).
2. **Numerical limits.** The pressure solvers are direct (FFT or sparse LU), so there is no iterative convergence that
   could fail. What changes with the density difference is the velocity scale U ~ sqrt(g' H) and therefore the
   admissible time step (`tvd2` transport needs |u| dt/dx + |w| dt/dz <= 0.5). In Boussinesq scaling all runs are the
   same flow on the time scale sqrt(a/g'), except for the Reynolds number (viscous effects) and the numerical
   diffusion at a given CFL number.

Proposal: run the case for rho_heavy - rho_light = 1, 10 and 50 kg/m³ (0.1 %, 1 %, 5 %) with the time step scaled to
keep the same CFL number. Compare the runs in dimensionless time t sqrt(g'/a): if they collapse, the model behaves as
expected and any of them can be used; the README then states the Boussinesq limit. Default for the documented
case: 1 % (10 kg/m³), open question Q4.

## 4. Current state of the code

The model assumes a flat bed and a rectangular domain everywhere:

- `ModelConfig.depth` is a single number; every column has `nz` cells of thickness `dz`.
- The barotropic free-surface solve (`BarotropicSolver`) uses one depth H for all faces.
- The non-hydrostatic pressure correction uses the spectral (DCT/FFT) solver by default, which only works on a
  rectangle; the sparse solver builds the 5-point Laplacian for the full rectangle.
- The w reconstruction integrates continuity upwards from w = 0 at k = 1.
- The hydrostatic pressure integrates downwards through all cells.
- Advection, diffusion and density transport treat only the outer domain faces as walls.
- Diagnostics (water volume, constituent mass, CFL, Froude number) and plots use all cells.

## 5. Design: variable bed on the z-level grid

**Representation.** A bed level per column, `bed_level[i]` (m above the grid bottom), rounded to whole cells
(staircase). Internally the model stores cell and face masks:

- `k_bed[i]`: index of the lowest wet cell in column i (cells below are solid);
- wet cells, wet u-faces (both neighbouring cells wet) and the bed w-face of each column (w = 0).

A bed level per column covers bathymetry (slopes, sills, steps) and the obstacle. It cannot represent overhangs or
floating bodies; the internal masks would allow that later without changing the solvers again (open question Q2).
The obstacle edges are placed on cell faces by choosing dx and dz accordingly; partial (shaved) cells are out of scope.

**Restriction.** The bed must stay below the top layer (k_bed[i] < nz), so the variable top layer of the hydrostatic
mode is unaffected. Dry columns (wetting and drying) are out of scope.

**Changes per component:**

| Component | Change |
|---|---|
| `ModelConfig`, `Model` | new option `bed_level` (default: flat bed, results unchanged); masks built at construction; checks (bed below top layer, `bed_level` length nx) |
| momentum predictor | u = 0 on dry and wall faces; advection fluxes through solid faces zero; `tvd2` slopes one-sided next to walls (as at the outer walls today); diffusion only between wet points (free-slip at the bed and obstacle walls, as at the outer walls) |
| hydrostatic pressure | integrate only through wet cells; baroclinic gradient only on wet u-faces |
| barotropic solve | depth per u-face, H_f = depth - max(bed_left, bed_right); transport summed over wet layers; tridiagonal coefficients per face (the variable-thickness solve already has per-face coefficients) |
| w reconstruction | start at w = 0 on the bed face of each column |
| non-hydrostatic projection | sparse solver on the wet cells only (Neumann at solid faces, gauge pin and mean removal on wet cells); the spectral solver is rejected when a bed is set (it needs a rectangle) |
| density transport | zero fluxes through solid faces; solid cells keep a fixed value and are excluded from bounds, mass and mass correction |
| diagnostics, plots | volume and mass over wet cells; solid cells drawn grey (NaN) in the heatmaps |

**Pressure-solver cost.** The sparse LU is factorised once. For 584 x 584 cells (1 mm) that is about 340,000
unknowns, probably a few hundred MB and a fraction of a second per solve. If this is too slow, a later step can use a
conjugate-gradient solver preconditioned with the existing spectral solver (open question Q6).

## 6. Verification (tests)

Following the repository conventions (`test_verify_*` for code verification, `test_validate_*` for physics):

1. `test_verify_flat_bed_reproduces_legacy_results`: `bed_level = 0` gives bit-identical results to the current code
   (hydrostatic and non-hydrostatic).
2. `test_verify_raised_flat_bed_equals_shallower_domain`: a uniform bed level d (a whole number of cells) gives the
   same results, to round-off, as a domain of depth H - d with fewer layers (lock exchange, both modes). This checks
   the masks, the barotropic depth, the w reconstruction and the projection together.
3. `test_verify_rest_state_over_obstacle`: uniform and horizontally stratified density at rest over an obstacle stay
   at rest (no spurious currents); both modes.
4. `test_verify_masked_projection_is_divergence_free`: after the projection the discrete divergence vanishes in every
   wet cell, and the masked Laplacian satisfies a manufactured solution on an L-shaped domain.
5. `test_validate_seiche_over_raised_bed`: surface seiche period over a uniformly raised bed matches
   T = 2L / sqrt(g (H - d)).
6. Conservation with an obstacle: water volume and constituent mass to round-off in a lock exchange over the
   obstacle.

## 7. Experiment

New folder `experiments/lock_exchange_obstacle/` following the existing pattern (`cases.jl`, plot script, README,
small figures):

- geometry of section 2, heavy column at the left wall;
- non-hydrostatic, `tvd2`/`tvd2`, sparse pressure solver, no Froude limiter;
- grid: 2 mm first (292 x 292), then 1 mm (584 x 584) for the documented figures;
- density sweep 1 / 10 / 50 kg/m³ (section 3), compared in dimensionless time t sqrt(g'/a);
- simulated time: the original's 0.6 s corresponds to t sqrt(g/a) = 4.9. With g' instead of g that is
  0.6 s x sqrt(g/g'), e.g. 6 s for 10 kg/m³. Because the light water has to flow back over the column, the flow is
  slower than the air-water case, so the runs go to about twice that (t sqrt(g'/a) ~ 10);
- figures: density GIF, snapshot panel at times comparable to the paper's Figure 3, front position against
  dimensionless time for the three density differences;
- optional: horizontal pressure force on the obstacle against time, the quantity the paper reports (Fx_max).

Outcome (implemented): the runs stop at t* = 7. From t* ~ 7.5 a spurious jet forms along the free surface after the
splash at the far wall (top cell of the fixed grid in the non-hydrostatic mode), also without the obstacle; it is tracked
in issue #5. The investigation added the Lax-Wendroff time correction to the `tvd2` momentum advection, which
removed an earlier energy source. See `experiments/lock_exchange_obstacle/README.md`.

## 8. Phases and commits

1. Geometry option, masks and checks; flat bed unchanged (test 1).
2. Hydrostatic mode and barotropic solve with a variable bed (tests 2, 3, 5 for the hydrostatic mode).
3. Non-hydrostatic projection on the masked domain (tests 2-4 for the non-hydrostatic mode).
4. Transport, diffusion and diagnostics at solid faces; plots with the mask (test 6).
5. Experiment, density sweep, figures and README.

Each phase is one commit on this branch, with the full test suite passing (apart from the three known vegetation
failures on `main`).

## 9. Decisions (review 2026-10-24)

- **Q1 Geometry:** from the paper (section 2); the cm in its text are read as mm.
- **Q2 Representation:** bed level per column (staircase).
- **Q3 Restriction:** the simplest variant for now: the bed stays below the top layer and dry columns are not
  supported. This may be extended later.
- **Q4 Density:** sweep 1 / 10 / 50 kg/m³, documented case 10 kg/m³ (1 %).
- **Q5 Top boundary:** free surface, light water filling the tank to the top.
- **Q6 Solver:** sparse LU first; an iterative solver only if it is too slow.
- **Q7 Walls:** free-slip at the bed and the obstacle, as in the paper and at the model's outer walls.

## References

- Martin & Moyce (1952), Phil. Trans. R. Soc. Lond. A 244, 312-324 (collapse of a liquid column)
- Koshizuka & Oka (1996), Nucl. Sci. Eng. 123, 421-434 (moving-particle semi-implicit method; dam break with obstacle)
- Robert, Leroyer, Wackers, Visonneau & Queutey, Time-Splitting Procedure for the Resolution of the Volume Fraction for
  Unsteady Flows (LHEEA, École Centrale de Nantes; the paper linked in issue #3, section 3)
- Ubbink (1997), Numerical prediction of two fluid systems with sharp interfaces, PhD thesis, Imperial College London
- Benjamin (1968), J. Fluid Mech. 31, 209-248 (gravity currents)
