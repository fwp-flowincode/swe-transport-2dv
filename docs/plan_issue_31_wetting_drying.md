Plan: issue #31, wetting and drying
==================================

Issue #31 asks for wetting and drying with these acceptance criteria:
- lake at rest: stagnant water stays stagnant to machine precision, in 1D and 2DV;
- mass conservation of water and constituents with wetting and drying, in 1D and 2DV;
- Ritter dam break on a dry bed: robust and sufficiently accurate, in 1D and 2DV;
- wave runup on a sloping beach against an analytical solution: robust and sufficiently accurate;
- new experiments with README, figures and animations.

"1D" is the single-layer model (`nz = 1`), "2DV" the multi-layer model (`nz > 1`), both on the same code path.

1 Current state
---------------

- z-level grid; only the top layer has a variable thickness dz + eta (hydrostatic mode). The run stops when the
  surface leaves the top layer (`_check_top_layer_thickness`).
- Variable bed (issue #3): a staircase of whole cells, at least two wet cells per column, no dry columns.
- `nz = 1` fails at construction (`BED_MIN_WET_CELLS = 2`).
- Constituent mass is enforced by a global correction (`_correct_constituent_mass!`), which would hide a
  non-conservative transport.

2 Approach
----------

A separate, opt-in time step for `ModelConfig(wetting_drying=true)` in `src/wetting_drying.jl`. The validated paths of
all existing cases stay unchanged. Hydrostatic mode only in this issue.

### 2.1 Geometry: cut cells at the bed and at the surface

- Bed level `bed_level` (m above the grid bottom, per column): any value >= 0, no longer whole cells; dry columns allowed.
- Water level zeta = depth + eta (m above the grid bottom); dry columns have zeta = bed level.
- Layer k (interfaces z_k = (k-1) dz) holds the thickness h_k = max(0, min(z_{k+1}, zeta) - max(z_k, z_b)); the top
  layer is open above (z_{nz+1} = infinity). For a flat bed and zeta in the top layer this is the existing variable
  top layer.
- Face thicknesses use the bed of the face z_b,f = max(z_b,L, z_b,R) and the upwind water level (u > 0: left, u < 0:
  right, u = 0: the higher one), as in Stelling & Duinmeijer (2003). A face is wet if zeta_f - z_b,f exceeds
  `dry_depth_threshold`; otherwise u = 0 there.

### 2.2 Time step (semi-implicit, Casulli & Cheng 1992 with variable layer thicknesses)

1. Momentum predictor on the wet face layers: advection, baroclinic pressure, viscosity, body force, damping.
   - Advection: upwind, momentum conservative (Stelling & Duinmeijer 2003), horizontal and vertical; first order
     (`upwind1`: a sum over inflow terms) or with limited MUSCL values (`tvd2`); capped so that thin layers stay stable.
   - Baroclinic pressure: p' = g * integral from z to zeta of (rho - rho0), evaluated at the nominal layer centres, so
     that a horizontally uniform stratification at rest has exactly no gradient (lake at rest with stratification).
2. Free-surface solve in the volume formulation of Casulli (2009): V(eta) + T eta = b with the non-negative column
   volumes V_i = max(0, H + eta_i - z_b,i), face depths at time n, Newton's method of Brugnano & Casulli (2008) started
   from eta^n (a lake at rest has a zero residual and stays exact). Depth positivity holds for any dt by construction
   of the solve; proof, existence and convergence, floating-point bound and run-time check: `docs/depth_positivity.md`.
   (Step 1 used a linear solve and an outflow limiter afterwards; replaced in step 2 because the limited velocities no
   longer solved the discrete momentum equations.)
3. Velocity correction u = u_hat - g dt d(eta)/dx on the wet face layers. Non-hydrostatic mode: pressure q from a
   projection on the cells below the surface cell, its gradient added to u_hat, steps 2-3 repeated.
4. eta^{n+1} from the flux divergence (volume conserved to round-off; equal to V(eta^{n+1}) of the solve).
5. Vertical fluxes through the fixed interfaces from the layer continuity, using the new layer thicknesses.
6. Constituent transport: conservative finite volumes with the same fluxes and layer volumes (`upwind1`, or `tvd2`:
   flux-limited Lax-Wendroff with the Courant number of the cut cell). Cells
   that empty pass their residual mass to the column; values outside the initial bounds (possible in thin cells)
   are clipped and the clipped mass is redistributed, so the transport stays exactly conservative. No global mass
   correction.

### 2.3 Boundaries

Closed walls, a prescribed left velocity (also time-dependent, `set_x_inflow_outflow!(model, t -> u)`) for the runup
forcing.

### 2.4 Not supported with wetting and drying (rejected with a clear error)

Periodic x, vegetation, k-epsilon, bed roughness, wind stress. Candidates for later steps, see section 6.
(Non-hydrostatic mode and `tvd2`: supported since step 2.)

3 Reference solutions and tests (`test/test_wetting_drying.jl`)
---------------------------------------------------------------

| # | Case | Reference | Kind |
|---|---|---|---|
| W1 | Layer thickness kernel | sum of layers = max(zeta - z_b, 0); reduces to the variable top layer | `test_verify_*` |
| W2 | Lake at rest over a beach and an island, 1D and 2DV (also stratified) | u = 0 and zeta = const to machine precision, dry cells stay dry | `test_verify_*` |
| W3 | Sloshing over a beach with drying and flooding, 1D and 2DV | water volume and constituent mass conserved to round-off; depth >= 0; density within its bounds | `test_verify_*` |
| W4 | Ritter dam break on a dry bed, 1D and 2DV | Ritter (1892) solution: L1 depth error, front position, convergence; 2DV close to 1D | `test_validate_*` |
| W5 | Reference solutions | Ritter conserves volume; Carrier & Greenspan satisfies the nonlinear SWE (residual) | `test_verify_*` |
| W6 | Standing wave on a planar beach | Carrier & Greenspan (1958): shoreline motion and surface elevation over one period | `test_validate_*` |

4 Experiments
-------------

`experiments/wetting_drying/`: lake at rest, Ritter dam break (1D and 2DV), Carrier-Greenspan runup; README, figures and
animations, like the other experiment folders.

5 Status (step 1, 2026-10-28)
----------------------------

Implemented test-first: `src/wetting_drying.jl`, `src/reference_solutions.jl` and `test/test_wetting_drying.jl` (90
checks, about 25 s). Results:

| Case | 1D (nz = 1) | 2DV |
|---|---|---|
| W2 lake at rest, 500 steps | u = 0, level unchanged (exact) | same, also with a two-layer stratification |
| W3 sloshing with flooding/drying, 1000 steps | volume and mass to round-off, depth >= 0, rho within bounds | same (nz = 10) |
| W4 Ritter, dx = 0.25 m, t = 5 s | L1 depth error 0.9 %, front (h = 1 cm) 11 % behind | same (nz = 12); depth within 1 % of the 1D run |
| W6 Carrier-Greenspan, A = 0.4, one period | shoreline 9.2 mm (9 % of the runup amplitude) at dx = 0.05 m; first order | 9.4 mm (nz = 15) |

Findings during the implementation:
- `rho = (h rho)/h` changes rho by round-off even without flow; in a stratified lake at rest this drove an internal
  motion of 5e-9 m/s. The transport is therefore written as an increment (exactly zero without flow).
- The untyped `Model.wetdry` field made the step allocate 6 MB; a function barrier brought it to zero allocations
  (0.13 ms per step for 400 x 12 cells).
- The dry front of the Ritter case converges slowly (first-order upwinding; 12.7 %, 11.3 %, 7.5 % at dx = 0.5, 0.25,
  0.125 m), while the depth converges at first order. The test accepts 12 % at dx = 0.25 m and requires convergence.

- 2DV Ritter: the velocity is not uniform over the depth where the surface has dropped through several layers (upper
  layers up to 6 % slower at t = 5 s). Likely cause: in the draining surface layer the upwind face thickness exceeds the
  averaged cell thickness, which over-weights the inflow of slower fluid in the momentum-conservative form; the vertical
  exchange carries the deficit down. To be fixed with the momentum-advection work of step 2.

Open questions for review:
- **Q1, accuracy targets:** are 12 % front lag (Ritter, dx = 0.25 m) and 10 % shoreline error (Carrier-Greenspan,
  dx = 0.05 m) "sufficiently accurate"? Tighter targets need second-order (limited) momentum advection and depth
  reconstruction near the front (section 6).
- **Q2, non-hydrostatic wetting and drying:** needed in this issue, or a separate one?

Experiments of section 4: done, `experiments/wetting_drying/` (README, figures, animations).

Decided in review (2026-10-28): non-hydrostatic wetting and drying and the `tvd2` momentum and transport schemes are
part of this issue (step 2). Still to do: the documentation in `test/TESTS_TECHNICAL_DOCUMENTATION.md`.

6 Status (step 2, 2026-10-28)
----------------------------

- **Depth positivity inside the solve** (review request): the free surface is solved in the volume formulation of
  Casulli (2009) with the Newton method of Brugnano & Casulli (2008); the outflow limiter of step 1 is removed. Proof
  in `docs/depth_positivity.md`. In all validation cases the solve takes one Newton iteration per step (the former linear
  solve, results unchanged); at advective Courant numbers >= 6 in the dam-break positivity test, 2-4 iterations, depths
  >= 0, clamped round-off volume <= 3e-16 m^2.
- **tvd2** for momentum (limited MUSCL values in the momentum-conservative form, including the outflow-side terms,
  which the first-order form does not have) and transport (flux-limited Lax-Wendroff). Ritter at dx = 0.25 m: 1D L1
  0.51 % (upwind1 0.91 %), front 5.6 % (11.3 %); 2DV L1 0.80 % (0.89 %), layer-velocity spread 0.18 m/s (0.29 m/s), front
  11.3 % (10.3 %, one cell). Carrier-Greenspan shoreline 13.6 mm (14.6 mm) at dx = 0.1 m; surface error 3.8 mm (3.3 mm).
- **Non-hydrostatic** (`use_hydrostatic=false`): w predictor, pressure q on the wet cells below the surface cell (q = 0
  there), sparse Cholesky each step, gradient added to the predictor and the free-surface solve repeated. Standing wave
  at k H = pi/2: period error 1.75 %, 0.89 %, 0.45 % on 40 x 20, 80 x 40, 160 x 80 cells (first order) against
  omega^2 = g k tanh(k H) (the shallow-water period is 24 % shorter). Ritter and Carrier-Greenspan in 2DV within the same
  targets as hydrostatic. Lake at rest exact, conservation to round-off.
- **2DV layer-velocity spread** (open): converges at first order in dx (mean 0.075, 0.043, 0.023, 0.012 m/s for dx = 0.5
  ... 0.0625 m, upwind1) and grows with nz. A column-consistent weighting of the advection (each layer with the 1D
  Stelling-Duinmeijer weight) was tried and was much worse (spread 3.8 m/s); not adopted.

7 Out of scope (proposed for later)
-----------------------------------

- Second-order momentum advection at the wet/dry front itself (tvd2 falls back to first order next to dry faces).
- A non-hydrostatic surface condition at the free surface itself instead of q = 0 in the surface cell (first order in
  the surface-cell thickness).
- Bed friction (Manning / log law) on the lowest wet layer, needed for laboratory runup cases.
- Energy-head conserving advection at contractions (Stelling & Duinmeijer 2003), for flow over weirs.

References
----------

- Casulli & Cheng (1992), Int. J. Numer. Methods Fluids 15, 629-648
- Stelling & Duinmeijer (2003), Int. J. Numer. Methods Fluids 43, 1329-1354
- Ritter (1892), Z. Ver. Deutsch. Ing. 36, 947-954
- Carrier & Greenspan (1958), J. Fluid Mech. 4, 97-109
- Delestre et al. (2013), SWASHES, Int. J. Numer. Methods Fluids 72, 269-300 (collection of analytical solutions)
