Issue #88: amplitude-dependent wave damping in the non-hydrostatic wetting-drying step
=====================================================================================

0 Status
--------

- Reproduction: `test_verify_wd_nonhydro_standing_wave_energy_amplitude_independent` (test/test_wetting_drying.jl,
  `@test_broken` until fixed): standing Klopman wave (h = 0.5 m, T = 1.44 s, k = 2.35 1/m) in a closed basin of one
  wavelength, 60 x 20 cells, N = 200 steps per period, θ = 0.55. Energy rate at a = 0.057 m minus that at a = 0.0057 m
  ("excess"): 0.0322 1/s; θ part 2·4π²(θ − ½)/(N T) = 0.0137 1/s.
- Part 1 (implemented, not committed, not validated): `wd_face_level = "limited"` (second-order limited level of the
  face thicknesses). Excess 0.0322 → 0.0227 1/s at 60 columns, 0.0218 at 120 columns.
- Part 2: the remaining 0.022 1/s, growing with nz: w at interfaces activated or removed by the moving surface
  (sections 5-6); design in section 7.

1 Measurements so far (basin unless stated)
-------------------------------------------

| change | excess (1/s) |
|---|---|
| baseline (upwind face level) | 0.0322 |
| N = 400 / 800 | 0.0330 / 0.0351 |
| q = 0 at the surface instead of in the surface cell | 0.0323 |
| fully linear (no advection, geometry frozen) | 0.0000 |
| only hf nonlinear (no advection) | 0.0126 (0.0074 at 120 columns) |
| all geometry nonlinear, no advection | 0.065 |
| advection, geometry frozen | 0.054 |
| `wd_face_level = "limited"` | 0.0227 (0.0218 at 120 columns) |

Flume (#85): the same extra decay with k-ε or without closure, with `upwind1` (+0.003 1/m), with nz = 40 (no decrease),
with Δx = 22.5 mm (face-to-station loss 28 → 26 %). Two-solve split: final against projected interface fluxes 1e-3 rms,
energy of the mismatch 3e-4 of the loss.

2 Discrete kinetic-energy budget of the momentum advection
----------------------------------------------------------

Momentum control volume of face f, layer k: thickness $\bar h = \tfrac12(h_{k,L} + h_{k,R})$ (cells L, R of the face),
width Δx. The advection of `_wd_advection_tendency` (Stelling & Duinmeijer 2003, upwind1 for the algebra; tvd2 replaces
$u_j$ by a limited value between $u_j$ and $u$) is

$$\frac{du}{dt} = \sum_{j\in\text{in}} c_j\,(u_j - u),\qquad c_j \ge 0,$$

with the inflow sides j: horizontal $c = |a_C|/\Delta x$, vertical $c = |\bar W|/\bar h$ ($\bar W$ the interface flux
averaged to the face). Write $F_j = \bar h c_j$ (m/s) for the inflow mass flux per unit area. With the identity
$u(u_j - u) = \tfrac12(u_j^2 - u^2) - \tfrac12(u_j - u)^2$ the kinetic energy $e = \tfrac12\bar h u^2$ obeys

$$\frac{de}{dt} = \underbrace{\sum_{\text{in}} F_j \tfrac12 u_j^2 - \sum_{\text{out}} F_j \tfrac12 u^2}_{\text{conservative KE flux}}
\;+\; \underbrace{\tfrac12 u^2\Bigl(\frac{d\bar h}{dt} - M\Bigr)}_{S:\ \text{continuity residual}}
\;-\; \underbrace{\tfrac12\sum_{\text{in}} F_j (u_j - u)^2}_{D \ \ge\ 0:\ \text{upwind dissipation}}
\;+\; \bar h u\,(\text{pressure and other terms}),$$

where $M = \sum_\text{in}F_j - \sum_\text{out}F_j$ is the net inflow of mass into the control volume implied by the
advective weights. Three kinds of error follow:

**(S) continuity residual.** The cell continuities averaged to the face give
$d\bar h/dt = (\bar q_L - \bar q_R)/\Delta x + \bar W_\text{bot} - \bar W_\text{top}$ ($\bar q_C$ the mean layer discharge of
the two faces of cell C). The code uses $a_C = (\bar q_C/\tilde h_C)(\tilde H_C/\bar H_f)$, so $a_C\bar h = \gamma_C\bar q_C$,
$\gamma_C = (\bar h/\tilde h_C)(\tilde H_C/\bar H_f)$, and

$$\frac{d\bar h}{dt} - M = -\frac{(\gamma_L - 1)\bar q_L - (\gamma_R - 1)\bar q_R}{\Delta x}.$$

$\bar h - \tilde h_C$ (cell against face thickness of the layer) is the difference between the cell levels and the
upwind face levels, $O(\Delta x\,\partial_x\zeta)$ in absolute terms, and $\tilde H_C - \bar H_f$ likewise. Hence
$(\gamma_C - 1)\bar q_C = O(\Delta x\,\partial_x\zeta\, u)$ and $S = O(\Delta x)$: it vanishes with Δx, also in a thin
cut layer (there the *relative* error $\Delta x\,\partial_x\zeta/h_k$ is O(1), the absolute one is not).

**(D) upwind dissipation.** Horizontal: $F = \bar q$-like, $(u_j - u)^2 = (\Delta x\,u_x)^2$, so $D_h = O(\Delta x)$
(upwind1), smaller for tvd2 except at extrema. Vertical between full layers: $\tfrac12|W|(\Delta z\,u_z)^2$ per interface,
$O(\Delta z)$ summed over the column. At the interface below the surface cell the distance of the layer centres is
$(\Delta z + h_s)/2$, still $O(\Delta z)$, unless the surface layer velocity differs from the layer below by O(1)
(see 3c).

**(P) pressure work.** The free-surface gradient does work $-g\eta_x\bar h u$ on the face layer, while the potential
energy changes with the continuity flux $h_f u$. The mismatch per face layer is $g\,\eta_x\,u\,(h_f - \bar h)$; summed
over the column it is concentrated in the cut surface layer. With the upwind level
$h_f - \bar h = -\tfrac12\Delta x\,|\zeta_x|\,\mathrm{sign}(u)$-like, i.e. a dissipation $\tfrac12 g\Delta x|u|\eta_x^2$:
exactly the Δx-dependent Part 1 (removed to second order by the limited level). Where the surface crosses an interface
between L and R the per-layer mismatches of layers s and s+1 have opposite signs; with vertical shear they leave
$g\eta_x(u_s - u_{s+1})\delta$, $\delta = O(\Delta x\zeta_x)$: $O(\Delta x\,\Delta z)$. The non-hydrostatic pressure work
has the same structure.

**Conclusion of the smooth analysis.** Every term of the energy budget of the continuous-in-time scheme vanishes with Δx,
Δz or both, as long as the layer thicknesses are smooth functions of x and t. The measured remainder (0.022 1/s,
4 % lower at half Δx, not lower at nz = 40 in the flume, not lower at smaller Δt) is not explained by any of them.
What the smooth analysis leaves out are the events where a thickness is not smooth:

3 Non-smooth events (candidates for the remainder)
--------------------------------------------------

a) **Opening and closing of face layers.** Each period the surface sweeps through 2–4 layers (a = 0.057 m, Δz = 25 mm);
   at every face a layer opens when the upwind level rises past its lower interface and takes the velocity of the open
   layer below (`_wd_initialize_new_face_layers!`), and closes (hf = 0) when the level falls. The momentum of a closing
   layer and the stale velocity it keeps are discarded; the number of events per period is a/Δz per face, each acting on a
   layer of thickness → 0, so the energy per event is small, but the *advective inflow* of the next step into the
   newly opened layer from the neighbours (closed-face rule: velocity of the nearest open layer below) is not.
b) **The closed-face rule and the cap.** Inflow into a layer whose far face is closed takes the velocity of a lower layer
   (`kin != k`); the cap $dt\sum c > 1$ turns the update into a convex combination in thin layers. Both are inelastic
   (energy lost $\tfrac12F(\Delta u)^2$ with $\Delta u$ the layer-to-layer velocity difference, O(Δz u_z), times a mass
   flux that does *not* scale with Δz: the volume swept through the interface per period, about a per unit area).
   Estimate: $\tfrac12\rho\,(a\omega)\,(\Delta z\,u_z)^2$ per unit area, $u_z \approx k u$: 1000/2 · 0.25 · (0.025 · 2.35 ·
   0.3)² = 0.04 W/m², against the measured loss of 0.18 W/m² (0.022 1/s · E, E = ρ g a²/4 = 8 J/m²). Same order, and
   it would scale with Δz (not tested in the basin).
c) **Velocity jump at the surface cell.** If the surface-layer velocity differs from the layer below by more than
   O(Δz u_z), the vertical upwind dissipation at that interface, $\tfrac12|W_s|(u_s - u_{s-1})^2$, is independent of the
   grid ($|W_s| \approx |\partial_t\eta|$). A jump of 0.06 m/s (20 % of the orbital velocity) would explain the whole
   remainder. q = 0 in the surface cell would cause such a jump (no non-hydrostatic gradient in the surface layer), but
   moving q = 0 to the surface did not change the excess, which argues against it.

5 Measured energy budget (refutes section 3)
-------------------------------------------

Probe with hooks in patched copies of the step (no source change): per step the change of
E = sum 1/2 rho hbar u^2 dx + sum 1/2 rho g (zeta - H)^2 dx + sum 1/2 rho h w_c^2 dx is split into the stages of the step.
Basin as in section 0, `wd_face_level = "limited"`, excess = rate at a = 0.057 m minus rate at a = 0.0057 m (1/s).
The budget totals agree with the test (0.0146 / 0.0230 / 0.0274 against 0.0151 / 0.0227 / 0.0270 at nz = 10 / 20 / 40).

| term (excess, 1/s) | nz = 10 | nz = 20 | nz = 40 |
|---|---|---|---|
| total | 0.0146 | 0.0230 | 0.0274 |
| new face layers, horizontal advection, closed-face rule, cap | 0.0002 | 0.0002 | -0.0003 |
| implicit vertical u step + geometry 1/2 u^2 dhbar (cancel each other) | 0.0023 | 0.0013 | 0.0008 |
| horizontal pressure work, hydrostatic g eta_x u (hf - hbar r) | 0.0001 | 0.0003 | 0.0002 |
| horizontal pressure work, non-hydrostatic q_x u (hf - hbar) / rho0 | 0.0000 | 0.0000 | 0.0000 |
| rest (non-hydrostatic correction, w side) | 0.0120 | 0.0214 | 0.0269 |

- The u side (advection, cut-cell rules, horizontal pressure work) is energy consistent to < 0.003 1/s: section 3 is refuted.
  The upwind face level (Part 1) appears in the rest (0.039 instead of 0.029 at nz = 20), as expected from section 2 (P).
- The excess grows with nz (basin test: 0.0151 / 0.0227 / 0.0270): not a Δz-scaling inelastic loss.
- w advection of the correction: switching it off lowers the excess by 0.004 (nz = 20) and 0.003 (nz = 40).
- w between steps: the projected w^{n+1} is discarded and rebuilt from the continuity of u. Energy of the jump at
  continuity interfaces -0.002 / -0.001 (gain), at fallback interfaces 0; at interfaces that appear or vanish (the
  surface crossing an interface) a loss of 0.0084 (nz = 20) and 0.0125 (nz = 40): the only measured term that grows
  with nz. Diagnostic only (the w of the energy is the continuity w, not the projected one).

6 Switch tests: w at a newly activated interface
-----------------------------------------------

Where the surface rises past the top of the surface cell s, interface s+1 enters the pressure problem. Its predictor
value w^n is the stored `st.w` there (`_wd_nonhydrostatic_correction!`, step 2, `full_below` false), which the continuity
of step 6 makes about 0 (the whole volume change of the column goes into the surface cell): the interface starts at rest.
Probe variants of that value (excess, 1/s, limited face level; u face layers closing: < 0.001 in all variants):

| w^n at the new interface | nz = 20 | nz = 40 |
|---|---|---|
| stored st.w (about 0), current code | +0.0230 | +0.0274 |
| 0.5 x kinematic | +0.0120 | +0.0083 |
| kinematic w_s - h_s (u_R - u_L)/dx (layer velocities of time n) | -0.0026 | -0.0132 |
| continuity of u^n everywhere (`full_below = k < nz`) | -0.0025 | -0.0133 |
| zero crossing (linear fit) | 0.90 x kinematic | 0.67 x kinematic |

- The whole nz-dependent excess comes from the crossing events (0.7 / 2 crossings per period and column at nz = 20 / 40
  for a = 0.057 m, Δz = 25 / 12.5 mm).
- Starting the interface at rest loses energy, starting it with the surface velocity gains energy, both growing with
  nz; the flux through the sloping surface (kinematic against continuity value) does not matter. No single factor
  works for both nz: the error is structural, not a missing time-centring.

7 Design: step on the grid of time n, conservative remap to n+1, merged thin top layer
--------------------------------------------------------------------------------------

**Diagnosis in terms of the grid.** The step mixes three geometries: the momentum predictor, the free-surface solve and
the face thicknesses use the grid of time n (layer thicknesses from ζ^n, Casulli-Cheng); the non-hydrostatic correction
uses a provisional geometry h_nh (from ζ^n + δ); the final continuity (step 6) uses h^{n+1}. Where the surface crosses a
grid level, layers and interfaces exist in one geometry and not in the other: the w predictor of an interface that
exists only in h_nh has no value of time n (it is set to st.w ≈ 0), and the fluxes computed on one grid are used on
another without mapping. Sections 5-6 measure the result: a loss per crossing, independent of how the missing value
is chosen.

**Principle.** The whole step is computed on one grid G^n (layer and face thicknesses of time n); the fluxes of the step
belong to G^n. At the end of the step the state is mapped conservatively from G^n to G^{n+1}. Layers are not created or
removed inside the step; their appearance or disappearance is a change from G^n to G^{n+1}, handled by the remap.

**Grid G^n with a merged top layer.**
- Cell column: the z-levels below the surface; the top computational layer is [z_s, ζ] if h_s = ζ - z_s >= α Δz, else
  the merged layer [z_{s-1}, ζ] (thickness between α Δz and (1 + α) Δz). The same at the faces with the face level ζ_f
  and the face thicknesses.
- Interfaces: only between computational layers; the interface z_s inside a merged top layer does not exist.
- Pressure: q unknown in every computational cell including the top one, q = 0 at ζ (distance to the centre of the top
  layer at least α Δz / 2, so the coefficients stay bounded).
- α to be chosen (0.1 ... 0.5); the step must keep the surface inside the top computational layer, |ζ^{n+1} - ζ^n| < α Δz
  per step for smooth waves (for bores the remap of several levels in one step is still conservative, only less
  accurate).

**Step on G^n.** Predictor, free-surface solve, non-hydrostatic correction and the continuity for w all on G^n: the
thickness change of the step is that of the top computational layer only (ζ^{n+1} - ζ^n), the layers below are fixed.
No provisional geometry h_nh, no fallback for w: every interface of G^n lies between two computational layers of G^n,
and w^n there is the continuity integral of u^n on G^n.

**Conservative remap G^n → G^{n+1}** (only where the top layer changes: crossing of a level, merge or split):
- u at a face: layer momentum hf u. New layer k gets the overlap integral of the old piecewise-constant profile, the top
  layer extended to ζ_f^{n+1} (the fluid that rose belongs to the top layer):
  u_k^{n+1} = Σ_j |[z_k, z_{k+1}] ∩ [z_j, z_{j+1}]| u_j / hf_k^{n+1}. Split: both parts keep u (momentum and energy
  conserved). Merge: mass-weighted mean (momentum conserved; energy loss 1/2 m Δu^2, small with the shear across Δz).
- Discharges and fluxes of the step stay on G^n (continuity and transport are closed on G^n); the remap moves only the
  state (u, constituents by the same overlap rule, conservative).
- w: not remapped as a state; w^{n+1} is the continuity integral of the remapped u on G^{n+1} (no memory of w, as now).
- q: the pressure of the merged/split top layer as initial guess only (solved again each step).

**Relation to the current code.** `_wd_initialize_new_face_layers!` is the copy part of the remap for u (a new face layer
at zero thickness); it becomes the overlap remap with merging. The provisional geometry h_nh of
`_wd_nonhydrostatic_correction!` and the `full_below` fallback disappear. The top-cell rule "q = 0 in the surface cell"
becomes "q = 0 at ζ" with the top computational cell as an unknown.

**Acceptance.** Probe budget excess < 0.002 1/s at nz = 10 / 20 / 40 and at Δx/2;
`test_verify_wd_nonhydro_standing_wave_energy_amplitude_independent` passes (`@test_broken` → `@test`).

**Risk and cost.** Changes every wetting-drying result (hydrostatic too, through the merged top layer and the face
thicknesses); all wetting-drying tests and the flume are rerun. Extra unknowns: one pressure unknown per column (top
cell). The thickness range of the top layer [α Δz, (1 + α) Δz] removes the thin-layer cases that the cap, the
closed-face rule and the floatmax advection thickness exist for; they stay as safeguards.

8 Implementation and measurements (step 1 and 2)
------------------------------------------------

Decisions (review of section 7): α = 0.1 as default (seen in practice), 0.25 and 0.5 compared; the remap over several
layers in one step is accepted (fast changes need a small time step anyway); the option may become the default.

Implemented behind `wd_top_layer = "cut" | "merged"` and `wd_merge_fraction` (default 0.1); "cut" is unchanged (energy
probe identical to all printed digits):
- `layer_thicknesses!(...; merge)`: top layer thinner than α Δz joined with the layer below (cells, faces, h^{n+1}).
- `_wd_remap_face_layers!`: overlap remap of u at the faces whose layering changed (replaces the copy of
  `_wd_initialize_new_face_layers!` in merged mode).
- `_wd_rebuild_interface_velocities!`: after the transport, the state w is rebuilt from the theta fluxes on G^{n+1}
  (w = 0 at the top of the top layer and above); the transport keeps the flux-consistent w with the remap volumes.
- Non-hydrostatic correction on G^n (h_nh = h^n: no divergence source from a layering change, w^n always from the
  continuity of u^n) and the top cell as unknown with q = 0 at ζ (Dirichlet at h_s/2, w*_s = δ/dt).

Energy-rate excess (1/s), a = 0.057 minus a = 0.0057 m, limited face level, nx = 60, N = 200, θ = 0.55
(probe_energy_test.jl; budget probe in brackets where run):

| configuration                                   | nz 10   | nz 20            | nz 40    |
|-------------------------------------------------|---------|------------------|----------|
| cut (Part 1 only)                               | 0.0151  | 0.0227           | 0.0270   |
| merged top layer + u remap only                 | 0.167   | 0.238            | 0.033    |
| + state w rebuilt on G^{n+1}                    |         | (0.059)          |          |
| + pressure problem on G^n                       | 0.0153  | 0.0106           | 0.0082   |
| (α = 0.25 / 0.5 at nz 20)                       |         | 0.0109 / 0.0118  |          |
| + top cell unknown, q = 0 at ζ                  | 0.0194  | 0.0063           | 0.0040   |
| same, w advection switched off (budget)         | (0.009) | (-0.004)         | (-0.006) |
| + second-order limited w advection (merged)     | 0.0101  | -0.0017          | -0.0025  |
| same at nx = 120 (Δx/2)                         | 0.0125  | -0.0005          |          |

Findings: the merged layer alone is much worse, because the remap volume (α Δz/dt at a split) stayed in the state w and
in the divergence of the provisional pressure geometry; both are grid motion, not fluid velocity. With the whole step on
G^n the excess decreases with nz (it grew with nz before). The largest remaining term is the explicit first-order upwind
w advection (0.007 to 0.014 1/s, amplitude-proportional numerical dissipation); without it the remainder is within
±0.009 and of either sign.

w advection (merged mode): the upwind difference plus the anti-diffusive correction of flux-limited Lax-Wendroff
face values (MC limiter), horizontally and vertically, first order where a neighbour interface is missing and where the
first-order step is capped. Small-wave rates move towards the pure θ rate (nz 20: 0.0150 to 0.0140, θ part 0.0137).
Tried and dropped: the surface velocity dζ/dt of the last step as the upper stencil point of the vertical advection
(excess 0.021 / 0.0074 / 0.0043, both amplitudes lose more). The remainder at nz = 10 (Δz = 50 mm < a = 57 mm) does not
change with Δx, so it is a vertical-resolution effect; at nz ≥ 20 the criterion |excess| ≤ 0.002 1/s holds (nz = 40:
0.0025, marginal).

Surface velocity of the top-cell unknown. The first form predicted w*_s = δ/dt (the level change of the free-surface
solve); the open-boundary tests then showed a model phase speed 4.6 % too fast at k h = 1.2 (10 layers; "cut" 3.3 %),
16 % at k h = 2. δ/dt is a continuity value: the surface fluid gets its new velocity without pressure, so q is too weak.
Predicting w_s from the vertical momentum like the interior interfaces (w^n_s = continuity integral of the discharges
of time n over the column, θ split of q with dz q = -q_s/(h_s/2)) gives phase speed / linear theory (40 cells per
wavelength, θ = 0.7, numerical_phase_speed_factor):

| layers | k h = 0.6 cut / merged | k h = 1.2 cut / merged | k h = 2.0 cut / merged |
|--------|------------------------|------------------------|------------------------|
| 10     | 1.0072 / 0.9981        | 1.0328 / 0.9971        | 1.0845 / 0.9950        |
| 20     | 1.0028 / 0.9983        | 1.0157 / 0.9981        | 1.0417 / 0.9984        |

(`copy_config` did not copy wd_face_level, wd_top_layer, wd_merge_fraction, so numerical_phase_speed_factor dropped
them; fixed. Still missing there, older: chezy_coefficient, layer_momentum_advection, nonhydro_solver,
nonhydro_solver_tolerance, nonhydro_solver_max_iterations.)

Energy-rate excess with it (1/s): nz 10 / 20 / 40: 0.0043 / 0.0033 / 0.0034; nz 20 at nx = 120: 0.0034, at N = 400:
0.0032; at a = 0.0285 m: 0.0002. The small wave now loses exactly the θ part (0.01355 against 0.0137). The remainder
does not change with Δx, Δt or Δz and grows much faster than a^2 (factor 15 for twice the amplitude): open.

Incoming wave profile at open faces ("merged"): zero at the still level (q = 0 at ζ), evaluated at the still-water
centres of the layers. Open-boundary test files with merged + limited as defaults: 86 of 89 pass (generated amplitude
within 1 % in all cases); failing: the factor range (1.025, 1.040), which encodes the dispersion error of "cut" (merged
0.997); the ratio of the residual reflection at k h = 2 to the interior dispersion error (the error is now 0.5 %, the
reflection about 0.005 instead of about 0.04); the reflection while generating on one current, 0.0066 > 0.005.

Investigation of the remainder (merged + limited, nx = 60, nz = 20, N = 200 unless noted; excess in 1/s):
- not a beating: over 30 periods the energy ratio large/small falls steadily (0.995 after 3 periods, 0.856 after 29),
  the loss slows as the amplitude decays;
- amplitude law at nz = 20: a = 0.0285 / 0.040 / 0.057 / 0.070 m: 0.0002 / 0.0011 / 0.0033 / 0.0058 (local exponent
  4.7 falling to 2.7: an onset, not a power law); at a = 0.0285: nz 10 / 20 / 40: 0.0011 / 0.0002 / 0.0004;
- not the limiter of the w advection: MC 0.0033, unlimited Lax-Wendroff 0.0033, Fromm 0.0034;
- not the face level: with the upwind face level 0.0140 (nx 60) and 0.0090 (nx 120), first order in Δx, extrapolated
  0.004 at Δx -> 0, the same remainder;
- budget (nz 20): w advection 0.0025 (0.0021 left without it), vertical step + thickness change 0.0013 net (-0.0273 and
  +0.0286), interfaces appearing/vanishing -0.0008.
Further tests (nz 20 unless noted):
- merge loss of the remap and the horizontal u advection are small in the budget (init 0.00005, advh 0.0002);
- explicit instead of implicit vertical terms: 0.0060 (nz 40: 0.0049), worse;
- switching single advection terms off breaks the balance instead of isolating a loss: without the vertical u
  advection -0.050 (the large wave gains energy), without the horizontal u advection 0.0124, without any advection
  (u horizontal, u vertical, w) 0.067. The free-surface geometry alone gives a large amplitude-dependent imbalance that
  the advection nearly cancels; the remainder is the residual of that cancellation, not one dissipative term.
Next: a discrete energy analysis of the scheme (kinetic energy of the face control volumes with the continuity of the
flux thicknesses and the advective fluxes, per term pair), to find the pair that does not cancel.

Flume #WMN (waves only, `klopman_model(:WMN)`, 760 x 20 cells, N = 100, no turbulence closure, 60 periods; envelope
of the first harmonic of the level in m, the wave enters at the right):

| θ    | scheme           | x = 33.5 m | 16 m   | 0.5 m  | spatial decay (1/m) | θ part (estimate) |
|------|------------------|------------|--------|--------|---------------------|-------------------|
| 0.70 | cut + upwind     | 0.0578     | 0.0262 | 0.0136 | 0.044               | 0.041             |
| 0.70 | merged + limited | 0.0590     | 0.0319 | 0.0173 | 0.039               | 0.041             |
| 0.55 | cut + upwind     | 0.0576     | 0.0393 | 0.0274 | 0.022               | 0.010             |
| 0.55 | merged + limited | 0.0605     | 0.0522 | 0.0474 | 0.0078              | 0.010             |

(decay fitted between x = 0.5 and 32 m; θ part 4π²(θ - 1/2)/(N T c_g), c_g = 1.345 m/s, linear estimate.) With merged +
limited the decay is the θ damping; the amplitude-dependent loss of "cut + upwind" (0.012 1/m at θ = 0.55) is gone.
The calibrated radiation-speed factor is 0.9987 (cut 1.0158). The remaining decay in the flume is set by θ and N of the
case (KL_THETA = 0.7, KL_STEPS_PER_PERIOD = 100), not by issue #88.

9 Next steps
------------

1. Validation of "merged": wetting-drying test file (merged variants of the lake-at-rest, conservation and positivity
   tests; the energy test passes for merged + limited), open-boundary incoming pressure profile (it assumes q = 0 in the
   surface cell, `_incoming_pressure_profile`), flume #WMN; then the defaults.
2. Open: the nz = 10 remainder (0.010 1/s at a > Δz).
