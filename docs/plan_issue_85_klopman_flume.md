Plan: Klopman's flume, waves following and opposing a current (issue #85)
=========================================================================

Issue: https://github.com/fwp-flowincode/swe-transport-2dv/issues/85. Validation case 2DV-13. The open faces it needs
(generation and absorption on a current) came with #86 (`docs/numerics_open_boundaries.md` sections 7.4–7.6).
Scope: the wetting-drying (cut-cell) step, non-hydrostatic, k-epsilon with a rough bed.

0 Status
--------

1. Done: reference profiles digitised (section 2).
2. In progress: current only (#SP) through two free faces (section 3.2) and the first waves-only run (#WMN, section
   3.3); then following (#CMP) and opposing (#CMN).
3. Changed (2026-11-10): the laminar streaming verification against Longuet-Higgins (1953) is replaced by a
   verification of the wave streaming stress that does not need the Stokes layer resolved (section 5). The laminar
   case moves to a separate issue together with a stretched vertical grid.

1 Model set-up
--------------

| Item | Choice | Why |
|---|---|---|
| Domain | the flat test section, x = 6.4–40.6 m of the flume (34.2 m), flat bed, still-water depth 0.500 m | the LDV station (x = 22.5 m) is 16 m from either ramp; the ramps and the wave boards are replaced by the open faces of #86 |
| Faces | both free (`left_free`, `right_free`), radiation scheme with `radiation_theta`, `nonhydro_open_face="radiation"`, `radiation_period = 1.44 s`, calibrated `radiation_speed_factor` (`numerical_phase_speed_factor(cfg; current)`) | 0.2–0.4 % reflection at the design period on a current of 0.16 m/s (#86) |
| Current | both faces relaxed with `open_face_relaxation="incoming"`, target discharge q = 0.0805 m²/s, target level 0 | the relaxation of the incoming variable does not reflect the waves; the target discharge is the total transport (gap 3 of the issue): the mean flux through a face includes the wave mass flux above the trough |
| Waves | #CMP: `left_incoming_level`; #CMN and #WMN: `right_incoming_level`; linear, T = 1.44 s, ramped over 3 periods | as in the flume (current always +x) |
| Amplitude | tuned until the first harmonic at x = 22.5 m is 0.060 m | Klopman tuned to the measured incident wave, not the board signal |
| Turbulence | `turbulence_closure="k_epsilon"`, `bed_roughness_length = 4e-5` m (Klopman's fit z₀ ≈ 0.04 mm), ν = 1.2e-6 m²/s (11–16 °C) | report section 4.4 |
| Grid | Δx ≈ L/60 (0.045 m, 760 columns), nz = 20 and 40 (Δz = 25 and 12.5 mm), Δt = T/100 | two grid levels for the sign criterion |

Cost (measured): 14 ms per step at 760 × 20 cells, non-hydrostatic with k-epsilon, i.e. 1.4 s per wave period.

Open point: the friction slope. A uniform current needs a mean surface slope $u_*^2/(gh)\approx1.1\cdot10^{-5}$,
i.e. 0.37 mm over the domain. With the target level 0 at both faces the relaxed incoming variable then sets the
discharge to $q-c\,\Delta\eta/2$, about 0.5 % low. Check in the current-only run; if it matters, set the target levels to
$\pm\Delta\eta/2$ from that run.

2 Reference data
----------------

Digitised from the scanned report (H840.30 Part II) into `experiments/validation/klopman/references/`, one CSV per test
with the figure, the method and the 7 % calibration note in the header:

| File | Figures |
|---|---|
| `klopman1994_mean_u_{SP,WMN,CMP,CMN}.csv` | mean U(z): 4.2/4.3, 4.6/4.9, 4.7/4.10, 4.8/4.11 (below z = 15 mm the log-scale figure) |
| `klopman1994_u1_amplitude_{WMN,CMP,CMN}.csv` | first-harmonic amplitude of u: 4.12–4.14 |

Method (`digitize_klopman.py`): 300 dpi render, grid lines as the axis calibration, markers by normalised
cross-correlation with one isolated marker. Error about 0.001 m/s in u. The linear and log figures agree to 0.001 m/s
where they overlap (#CMN at z = 0.10 m: 0.1235 and 0.1240 m/s). Spot checks of the issue at z = 0.40 m: opposing
0.231 m/s, following 0.123 m/s. Overlapping markers of the two LDVs near z = 0.25 m are partly merged.

Not digitised (yet): Fig. 4.4/4.5 (current only, σ_u, σ_w, −u'w'), the log-scale amplitude Fig. 4.15–4.17, and the model
curves of Groeneweg & Klopman (1998).

3 Steps
-------

### 3.1 Case module

`experiments/validation/klopman/cases.jl`: constants (named, SI units, report section) and
`klopman_model(case; nx, nz, amplitude)` for #SP, #WMN, #CMP, #CMN; the period-averaged diagnostics at the station
(Eulerian mean u at fixed heights, the complex first harmonic, ν_t, k) and the volume budget.

### 3.2 Current only (#SP)

- Run to a steady profile (time scale $h^2/\nu_t\approx10^3$ s).
- Compare U(z) with Fig. 4.2/4.3; fit $u_*$ and $z_0$ from the log layer as Klopman did (u* ≈ 7.3 mm/s).
- Check that the free faces keep the target discharge and that the profile is developed at the station.
- Grid levels nz = 20, 40.

Run of 2400 s (nz = 20): steady after about 500 s; the profile at 600 s and 2400 s agree to 1e-4 m/s. Discharge
0.07933 m²/s at both faces and the station (1.45 % below the target), levels +0.30 mm (left) and −0.08 mm (right).
The deficit is larger than the estimate $c\Delta\eta/2$ (0.5 %). Remedy: the steady face levels of the current-only
run become the target levels of all runs; the steady state then lies on both relaxation lines with Q = Q_T. The profile lies within about 3 % of the measured one scaled by 1.07 (the calibration bias);
the measured profile integrates to about 0.075 m²/s, 7 % below the pumped 0.0805, which confirms the bias.

### 3.3 Waves only (#WMN)

- Incident amplitude at the station (tune to 0.060 m), first-harmonic profile against Fig. 4.12, undertow against
  Fig. 4.6.
- ν_t under the waves away from the bed (gap 4 of the issue). If k-epsilon overproduces (Larsen & Fuhrman 2018), a
  separate issue for a stabilised closure.
- The wave friction the model actually has (section 5.3), reported alongside the profiles; one run with Δz = 5 mm
  (nz = 100) as the near-bed resolution check against Fig. 4.9.

First run (60 periods, nz = 20, incoming amplitude 0.060 m): first harmonic of η at the station only 0.0245 m (41 %);
ν_t at the station 4·10⁻⁶–2·10⁻⁵ m²/s, but the largest ν_t in the domain grew from 4.5·10⁻⁵ (period 10) to
1.2·10⁻³ m²/s (period 60).

Diagnosis (η₁(x) along the flume, with k-epsilon and without a closure):
- **The wave decays along the flume**, from 0.058 m at the generating face to 0.031 m at the station and 0.013 m at
  the far face: a spatial amplitude decay of 0.044 1/m, the same with and without k-epsilon (to 1 %). It is the
  damping of the θ-scheme, about $4\pi^2(\theta-\tfrac12)/N$ per period (N steps per period; 0.079 at θ = 0.7,
  N = 100), i.e. a spatial rate $4\pi^2(\theta-\tfrac12)/(N T c_g)\approx0.042$ 1/m. Bed friction (f_w ≈ 0.01)
  gives only 3·10⁻⁴ 1/m.
- **This is not only an amplitude problem.** A wave that decays carries a radiation-stress gradient
  $dS_{xx}/dx\approx2\cdot0.044\,S_{xx}\approx1.6$ N/m², 30 times the bed stress of the current (0.05 N/m²). The
  depth-uniform part goes into a set-up, but its vertical distribution drives a spurious mean flow. Klopman's own
  damping was small (incident amplitude tuned at the station, reflection 7–8 %). The numerical damping over the
  18 m from the face to the station must therefore be a few percent at most.
- **ν_t grows at the generating face** (x = 34 m, mid-depth, 1.1·10⁻³ m²/s at period 60, still growing) and spreads
  downstream (1.9·10⁻⁴ at x = 25 m, 3.2·10⁻⁵ at the station). Where the wave is largest. Whether it is the face
  (incoming wave against the zero-gradient k) or the overproduction of k-epsilon under waves (Larsen & Fuhrman
  2018) is open; with less damping the wave will be large everywhere, which will tell.

Options for the damping (to be decided):
- θ closer to ½ and more steps per period: the stability limit with a mean flow is
  θ ≥ (1 + 2Fr)/(2(1 + Fr)), with Fr including the orbital velocity about 0.15, so θ ≥ 0.57 (waves only,
  Fr ≈ 0.08: θ ≥ 0.54). Decay over the 18.1 m from the face to the station (model x = 16.1 m, c_g = 1.35 m/s):
  θ = 0.6 with N = 400 about 9 % (4 times the cost, 6 s per period), with N = 800 about 4.5 % (8 times, 11 s per
  period); θ = 0.55 halves these. Bed friction (3·10⁻⁴ 1/m) cannot replace the θ-damping.
- A time integration of the free surface without the first-order damping (second order with θ = ½ and a stable
  advection): a model change, outside #85.

**θ sweep, #WMN with k-ε, 60 periods** (`klopman_model(:WMN; theta, steps_per_period)`, amplitude 0.060 m at the
face; envelope of |η₁| fitted over model x = 4–32 m):

| θ | N | Δt (s) | decay (1/m) | |η₁| at station (m) | loss face → station | run (s) |
|---|---|---|---|---|---|---|
| 0.7 | 100 | 0.0144 | 0.044 | 0.031 | 47 % | — |
| 0.6 | 200 | 0.0072 | 0.0241 | 0.037 | 34 % | 316 |
| 0.6 | 400 | 0.0036 | 0.0198 | 0.039 | 31 % | 522 |
| 0.55 | 200 | 0.0072 | 0.0186 | 0.040 | 29 % | 197 |
| 0.55 | 400 | 0.0036 | 0.0165 | 0.040 | 28 % | 334 |

- θ = 0.55 is stable without a current (no growth in 60 periods), as the bound θ ≥ 0.54 for Fr ≈ 0.08 says.
  With the current (#CMP/#CMN, θ ≥ 0.57) it is not covered.
- The decay fits rate = a (θ − ½)/N + b with a ≈ 17 1/m (θ-damping estimate 4π²/(T c_g) = 20 1/m) and
  **b ≈ 0.015 1/m independent of θ and Δt**. The θ-damping is under control (0.0025 1/m at θ = 0.55, N = 400);
  the remaining damping is not in the time integration.
- ν_t reaches 1.3·10⁻³ m²/s at the generating face in all runs; a wave-averaged ν_t of order 10⁻³ damps
  at about 2ν_t k²/c_g ≈ 0.008 1/m, the order of b. Bed friction (3·10⁻⁴ 1/m) is too small.
- The mean u at the station differs from run to run by up to 0.015 m/s with only the time step changed: it is
  driven by the decay, not by the streaming. The |u₁| shape (scaled to 0.060 m) is the same in all runs and
  about 10 % above the data at the bed.
- **Without a closure** (θ = 0.55, N = 400, `turbulence_closure="none"`): decay 0.0177 1/m, |η₁| = 0.040 m at
  the station, the same as with k-ε (0.0165). **k-ε is not the cause**; the ν_t of 10⁻³ at the face is real but
  its effect is within the scatter of the fit. The mean u (0.010–0.014 m/s, nearly uniform) is then the return
  flow of the wave's mass transport (E/(ρc h) ≈ 0.009 m/s at a = 0.04 m) with no turbulent shear.
- The remaining 0.015 1/m is in the spatial discretisation or the free-surface treatment. Candidates:
  (a) the surface moves through 2–4 of the 25 mm layers every period (trough −0.05 m): wetting and drying of the
  top cells in the cut-cell step; (b) the TVD limiter of `momentum_advection_scheme="tvd2"` clipping the orbital
  velocity at crests and troughs; (c) the horizontal resolution (60 cells per wavelength) of the non-hydrostatic
  pressure. (a) and (b) depend on the amplitude, (c) does not.
- **Small amplitude** (no closure, θ = 0.55, N = 400, a = 0.006 m): decay 0.0033 1/m, |η₁| 0.0060 → 0.0057 m
  from the face to the station (5 %). That is the θ-damping (0.0025) plus little else: **the extra
  0.015 1/m is an amplitude effect** — (a), (b), or a transfer of first-harmonic energy into higher harmonics
  (ka ≈ 0.14 at the face; a free second harmonic would give a recurrence along the flume, not a steady decay).
- **Advection scheme and layers** (a = 0.060 m, no closure, θ = 0.55, N = 400):

  | run | decay (1/m) | |η₁| station (m) | |η₂| face / station (m) |
  |---|---|---|---|
  | tvd2, nz = 20 | 0.0177 | 0.041 | — |
  | upwind1, nz = 20 | 0.0204 | 0.039 | 0.014 / 0.005 |
  | tvd2, nz = 40 | 0.0203 | 0.041 | 0.015 / 0.005 |

  Neither the limiter (upwind1 adds only 0.003 1/m) nor the number of layers crossed by the surface (nz = 40 does
  not reduce it) is the cause. nz = 40 warns: surface Froude number 0.121 at the generating face above
  `theta_froude_max(0.55) = 0.111`; it stayed bounded over 60 periods.
- **Second harmonic:** 0.014–0.015 m at the face, about twice the Stokes bound harmonic
  k a² (3 − σ²)/(4σ³) = 0.008 m (σ = tanh kh = 0.83), so the linear incoming level also excites a free second
  harmonic; at the station 0.005 m against the bound 0.004 m. Its energy is (a₂/a₁)² ≈ 6 % of the first
  harmonic: not the drain.
- **Candidate: the upwind surface-layer thickness at the u-faces** (`_wd_face_thicknesses!`, the upwind water
  level by the sign of the depth-summed u, Stelling & Duinmeijer 2003). The mass flux then carries the cut layer at
  the upwind level: a first-order upwind flux of η with diffusion D = |u_s| Δx/2 (u_s the velocity of the cut
  layer). For η_t + h u_x = D η_xx with u_t + g η_x = 0 the wave decays at D k²/2 in time, D k²/(2 c_g) in space.
  With the phase-weighted mean ⟨|cos| cos²⟩/⟨cos²⟩ = 8/(3π) ≈ 0.85 of û_s ≈ 0.22 m/s: D ≈ 4·10⁻³ m²/s, rate
  ≈ 0.009 1/m. It is proportional to the amplitude (a = 0.006 m: about 0.001 1/m, observed 0.0033 − 0.0025),
  independent of the advection scheme, of nz and of the closure, and proportional to Δx. The order matches the
  observed 0.015 1/m within a factor 2.
- Test: #WMN without closure, θ = 0.55, N = 400, a = 0.060 m at nx = 1520 (Δx = 22.5 mm, about 2 times the cost):
  the extra decay should halve. If confirmed, the remedy is a model change (a second-order limited surface level at
  the u-faces, keeping the upwind level near dry cells), a separate issue before #85 continues.
- **Result nx = 1520** (Δx = 22.5 mm, otherwise as tvd2/nz = 20): |η₁| 0.0566 → 0.0418 m from the face to the
  station (26 %), against 0.0571 → 0.0413 m (28 %) at Δx = 45 mm. **The upwind face level is not the cause** (it
  would have halved). Left of x = 12 m the envelope drops to 0.019 m (0.031 at Δx = 45 mm) and the mean u is
  −0.009 m/s instead of +0.014: something else is wrong near the left face at this resolution (not analysed).
- Status: the extra decay (about 0.015 1/m) is proportional to the amplitude but does not change with Δx, Δt
  (beyond the θ part), nz, the advection scheme or the closure. The first harmonic loses about half its energy
  flux over 18 m; the second harmonic holds only about 6 %, and the mean flow far less, so the energy is
  dissipated. Breaking is excluded (H/L = 0.045 against 0.14 tanh kh = 0.11).
- **Fractional steps:** the step (`_step_forward_wetting_drying!`) splits the pressure correction (free-surface
  solve, non-hydrostatic correction, solve again), freezes the layer thicknesses at time n (Casulli & Cheng), and
  lags the advection, w, transport and k-ε. All of these are O(Δt). Fitting rate = a(θ − ½)/N + b/N + c to the four
  θ/N runs gives b/N ≈ −0.0001 1/m at N = 400: no first-order splitting term; the extra decay is Δt-independent.
- **Closed basin, standing wave** (`probe_basin.jl`: wetting-drying step, non-hydrostatic, tvd2, nz = 20,
  Δx = 45 mm, no closure, θ = 0.55, N = 400; L = 4 wavelengths, k = 2.35 1/m; no open faces, no forcing, no
  relaxation). Mode amplitude over 40 periods:
  - a = 0.057 m: 0.057 → 0.049 m in 6 s (0.027 1/s), 0.025 → 0.0245 m at 53–55 s (0.010 1/s); fit 0.0137 1/s.
  - a = 0.0057 m: 0.0045 1/s, the θ part 0.0034 1/s plus 0.001.
  The amplitude-dependent damping is **in the interior equations or their discretisation**, not at the faces or in
  the forcing. Its early rate at a ≈ 0.05 m (0.025 1/s) matches the flume (0.015 1/m × c_g = 0.020 1/s). The basin
  runs take about 70 s, so the remaining diagnosis can be done there.
- **Basin, energy and step variants** (a = 0.057 m, 40 periods):

  | step | decay of the mode (1/s) | E/E₀ at 55 s | (A/A₀)² at 55 s |
  |---|---|---|---|
  | wetting-drying, non-hydrostatic | 0.0137 | 0.186 | 0.185 |
  | standard (fixed grid), non-hydrostatic | 0.0307 | 0.044 (noisy) | 0.033 |
  | wetting-drying, hydrostatic | 0.0383 | 0.016 | 0.005 |

  The total energy follows the mode amplitude: the energy is dissipated, not moved into harmonics. The
  standard step dissipates more (and warns Fr = 0.121 > `theta_froude_max(0.55)`), so the cut-cell geometry of the
  wetting-drying step is not the cause. The hydrostatic run steepens into bores (no dispersion at kh = 1.18) and
  loses energy at the fronts, as expected; not comparable.
- Common to all variants: the nonlinear momentum advection. **Basin without horizontal momentum advection**
  (`_wd_advection_tendency` redefined to zero in the probe script, no source change): decay 0.0228 1/s, E/E₀ = 0.125
  at 55 s, more than with it. Not a clean test (without advection the continuum equations do not conserve energy
  either), but the advection term is not the source.
- Checked and excluded: newly opened face layers take the velocity of the open layer below
  (`_wd_initialize_new_face_layers!`), so they do not start at rest; the basin has no bed friction, sponge or
  Rayleigh damping.
- **New candidate: the water above the grid top.** `klopman_model` and the basin use a grid height `depth` equal to
  the still-water depth (0.5 m), so above z = 0.5 m all water of a crest (up to a = 0.057 m, more than 2 dz) sits in
  the one top layer, whose thickness grows to dz + η. That part is not refined by nz, does not depend on Δx or Δt,
  and grows with the amplitude. Test: basin with grid height 0.6 m (nz = 24, the same dz = 25 mm), water level 0.5 m,
  so the surface stays inside the layered grid.
- **Result grid 0.6 m:** decay 0.0241 1/s, E/E₀ = 0.048 at 55 s: *more* damping than with the grid top at the
  still-water level (0.0137, 0.186). Refuted; the damping depends on where the surface lies in the layers, which
  points at the treatment of the cut surface cell (with the grid top at z = h the still surface sits on the top
  interface and the crest water in one thick layer; with 0.6 m the surface cuts the layers on both sides).
- **Two-solve split** (`probe_split.jl`, basin a = 0.057 m, periods 2–10): the theta interface flux of the
  projection, w_proj = w* − Δt/ρ₀ θ² ∂q/∂z on the h_nh geometry, against the final flux from the layer continuity
  after the repeated free-surface solve: rms difference 9.6·10⁻⁴ of rms w (max 7·10⁻³ against 0.31 m/s), at
  a = 0.0057 m 5.5·10⁻⁷. The energy of the mismatch, Σ ½ρ₀ h (w_final² − w_proj²) Δx summed over the steps, is
  +0.010 J/m, against the energy loss of −33.7 J/m over the same 9 periods. **The hydrostatic prediction with a
  non-hydrostatic correction and a second solve is not the cause**; a total-pressure prediction and correction
  (one coupled solve for η and q) would not remove the loss.
- **Remaining candidate: q = 0 in the whole surface cell** (`_wd_nonhydrostatic_correction!` docstring: first order
  in the surface-cell thickness; validated for the period of a standing wave, not for its damping). With the grid
  top at z = h the crest water lies in one surface cell up to dz + a thick, so q = 0 sits up to (dz + a)/2 below the
  surface whatever nz is. Next: q = 0 at the actual surface (the surface cell as an unknown with the Dirichlet value
  at distance h_s/2 above its centre, or a ghost value), in a separate issue.
- **Status:** an amplitude-dependent energy loss of the non-hydrostatic step, reproducible in a closed basin in
  90 s, not from the faces, forcing, closure, advection scheme, horizontal advection, Δx, Δt or nz, and
  sensitive to the position of the surface in the layers. It is a model defect, not a Klopman set-up issue, and
  blocks the wave cases of #85 (#WMN, #CMP, #CMN); #SP is unaffected.
- Earlier proposal: repeat at a = 0.060 m with `momentum_advection_scheme="upwind1"` (more damping if (b)) and with nz = 40
  (more cells crossed per period if (a)).

### 3.4 Following (#CMP) and opposing (#CMN)

- Mean U(z) below trough − 35 mm against Fig. 4.7/4.8, ΔU(z) relative to #SP, first-harmonic amplitude against
  Fig. 4.13/4.14.
- The primary criterion: the upper-half shear decreases (and turns negative) for following waves and increases for
  opposing waves, on both grid levels.

### 3.4a First full run (nz = 20, after #88)

Settings: merged top layer + limited face level (#88), θ = 0.6, N = 400, k-ε, z₀ = 0.04 mm. #SP spun up 600 s at
N = 100, its face levels (+0.30 / −0.09 mm) as target levels, 300 s more; the wave cases start from that state (#WMN
from rest), a 30-period tuning run scales the incoming amplitude, then 150 periods, averages over the last 50
(`run_klopman_cases.jl`, `validate_klopman_wave_current.jl`; 83 min in all).

- #SP: Q = 0.0797 m²/s at both faces and the station (1.0 % below 0.0805; 1.45 % with target levels 0). Mean u RMS
  0.047 (0.038 against the data × 1.07); upper-half shear 0.069 1/s, data 0.069.
- Station amplitude after tuning: #WMN 0.0525, #CMP 0.0567, #CMN 0.0578 m (the production runs end lower than the
  tuning runs). Incoming amplitudes 0.063, 0.065, 0.077 m: the opposing wave decays by 25 % over the 18 m to the station.
- Steady: mean-u drift between the two halves of the window 0.1–1.3 mm/s.
- Mean u RMS/0.16: #WMN 0.059, #CMP 0.056 (0.103 against × 1.07), #CMN 0.222.
- **Primary criterion** (upper-half shear, 0.25 m ≤ z ≤ trough − 35 mm): following waves decrease it (model 0.021 vs
  0.069 1/s, data −0.137: the decrease is right, too weak, no sign change); **opposing waves do not increase it**
  (model 0.019, data +0.392). The opposing profile is nearly uniform at 0.165 m/s above z = 0.1 m and too fast near
  the bed; the data grow linearly to 0.23 m/s at z = 0.40 m.
- First-harmonic amplitude of u: unscaled model against data RMS/0.16 0.015 (#WMN, #CMP), 0.106 (#CMN).

### 3.4b Wave effects on turbulence, bed friction and streaming

The model has no explicit mechanism for them; it gets them from the resolved orbital motion, which goes wrong in three
ways: (1) k-ε is unstable in the nearly potential flow under waves (Larsen & Fuhrman 2018) and overproduces ν_t;
(2) the steady log law at the first cell centre gives a wave friction factor of 30–40 % of Swart's (section 5.3);
(3) the streaming stress follows from (2) and is not tested. Handled in this issue, in this order.

**(1) Stabilised k-ε** (`ModelConfig(ke_wave_stabilization = λ₂)`, `ke_eddy_viscosity`, both steps; default 0):
ν_t = C_μ k²/max(ε, λ₂ C p_rat ε), p_rat = S_ijS_ij/Ω_ijΩ_ij, C = C₂/C₁ (Larsen & Fuhrman 2018, in the k-ε form of
OpenFOAM's multiphaseStabilizedTurbulence). Tests: `test_verify_k_epsilon_wave_stabilization_kernel`,
`test_verify_k_epsilon_wave_stabilization_stops_growth` (friction-free basin: ν_t at mid-depth grows ×4.1 over periods
12–20 without, ×1.0 with λ₂ = 0.1, ×1.4 with 0.05).

Flume, nz = 20, λ₂ = 0.1 (#SP, #WMN, #CMN; #CMP not yet rerun):
- ν_t at mid-depth: #WMN 2·10⁻⁶ m²/s (1.6·10⁻³ without); #CMN 1.8·10⁻⁴ at z = 0.16 m, 2·10⁻⁶ at 0.36 m (the
  current's turbulence is suppressed in the upper half as well); #SP unchanged.
- Mean u RMS/0.16: #WMN 0.024 (0.059 without), #CMN 0.284 (0.222).
- Upper-half shear: #WMN 0.049 (data 0.090); **#CMN 0.100 against #SP 0.069: the opposing criterion now holds**
  (data 0.392: the shear is a quarter of the measured one, its increase over #SP 0.031 against 0.323, about 10 %).
- #CMN is 0.03–0.05 m/s too fast over the whole depth and has almost no near-bed deficit (model 0.135 m/s at
  z = 12 mm, data 0.03–0.10 m/s below 20 mm): the current feels too little bed resistance under the waves, which is
  item (2) (the wave-enhanced apparent roughness the log law at z₁ = 12.5 mm does not have).
- #WMN is not steady yet (mean-u drift 18 mm/s between the halves of the window; without turbulence the mean flow
  adjusts slowly).
- #CMP rerun with λ₂ = 0.1: upper-half shear −0.011 1/s against #SP 0.069 (data −0.137): **the following criterion
  now holds including the sign change**; mean u RMS 0.085. #WMN over 300 periods: still a drift of 18 mm/s between the
  halves (RMS 0.052): the mean flow without turbulence does not settle.
- Status of the primary criterion at nz = 20 with λ₂ = 0.1: both directions right (following −0.011, opposing +0.100
  against 0.069), both weaker than measured (−0.137, +0.392).

**(2) Wave-enhanced bed friction** (`ModelConfig(wave_bed_friction = "swart")`, `wave_friction_filter_time`;
`_wd_wave_friction_factor!`): friction coefficient max(c_b, f_w/2) with Swart's f_w for the orbital excursion of the
resolved bed velocity (running variances of u₁ and du₁/dt), in the momentum step and the k-ε wall cell. No phase lead.
Test `test_verify_wd_wave_bed_friction_kernel`. Klopman: f_w = 0.035, factor 3.4 on c_b at z₁ = 12.5 mm.

#### Results of (2) and (3), nz = 20, λ₂ = 0.1

**(2) Swart friction**, first version: c = max(c_b, f_w/2) raised the friction of the current-only case by 2.3–5.5
(small fluctuations of the current give a short apparent excursion). Corrected: c = c_b + max(f_w/2 − c_b, 0) r with
the wave share r = var u/(var u + ū²) of the bed velocity; current only then unchanged (factor 1.00). Run with it:

| case | RMS λ₂ only | RMS + Swart | shear λ₂ only | shear + Swart | shear data |
|---|---|---|---|---|---|
| #SP | 0.047 | 0.047 | +0.069 | +0.069 | +0.069 |
| #WMN | 0.052 | 0.029 | +0.052 | +0.053 | +0.088 |
| #CMP | 0.085 | 0.179 | −0.011 | +0.014 | −0.135 |
| #CMN | 0.284 | 0.339 | +0.100 | +0.142 | +0.403 |

Following waves lose the sign change (the lower half too slow), opposing waves gain shear but the profile is faster
still. The simple form changes the friction alike for both directions; the data differ strongly near the bed. Not
adopted; λ₂ = 0.1 with the log law stays the baseline. A wave-current boundary-layer model (Grant & Madsen 1979, with
the phase lead) would be the next step if the bed friction has to improve.

**(3) Streaming stress** (probe, #WMN, 40 periods, averages over the last 10 and over one wavelength around the
station): ⟨ũw̃⟩ at the top of the first cell −4.7·10⁻⁶ m²/s² against D_f/c = ⟨ũ_b τ̃_b⟩/c = −6.6·10⁻⁶ (ratio 0.72),
in the wave direction; near-bed mean velocity −0.033 m/s (streaming in the wave direction). The partial standing wave
makes ⟨ũw̃⟩ vary between −2.9·10⁻⁵ and +1.8·10⁻⁵ within the wavelength, so the ratio is uncertain by about ±0.2. The
mechanism is present and about consistent with the model's own bed friction; its size follows the too-small f_w of
the log law at z₁ = 12.5 mm.

**Mean flux** (mean u over 0.02–0.38 m; (q − M)/h with the linear wave mass transport M = g a²/(2c) at a = 0.060 m):

| case | model | data (× 1.07) | (q − M)/h |
|---|---|---|---|
| #SP | 0.157 | 0.152 (0.162) | 0.161 |
| #WMN | 0.019 | 0.012 (0.013) | 0.019 |
| #CMP | 0.129 | 0.140 (0.150) | 0.144 |
| #CMN | 0.200 | 0.159 (0.170) | 0.183 |

Current only and waves only agree with the mass balance. With a current the model is about 10 % off the linear
estimate, low for following and high for opposing waves; the data of the opposing case lie 7–13 % below the estimate
themselves (3D circulation of the flume, Klopman 1997, or the calibration). The window excludes the near-bed layer
and the part above the trough, so these are indications, not budgets.

### 3.5 Products

`validate_klopman_wave_current.jl` (figures and a table), `README.md`, a fast `test_validate_klopman_*` on a coarse grid,
a 2DV-13 row in `docs/validation_cases.md`, a section in `docs/TESTS_TECHNICAL_DOCUMENTATION.md`.

### 3.6 Status 2026-11-15: work paused, results under review

The results of 3.4a/3.4b are **not reliable yet**. Two problems were found; they are mixed in every comparison so far.
All numbers below: λ₂ = 0.1, upper-half shear (1/s) by least squares over 0.25 m ≤ z ≤ z_max (validator) or
0.25–0.36 m (marked *), station mean u over 0.02–0.36 m. Result files in `experiments/validation/klopman/results/`
(tags `_t<θ>n<N>` and `_nx<nx>`, driver switches `KL_THETA_N` and `KL_NX`).

**nz = 40 (θ = 0.6, N = 400, 760 columns):** SP +0.069, CMP −0.021 (data −0.135), CMN +0.167 (data +0.403); both signs
right, about 44 % and 29 % of the measured change. WMN station amplitude only 0.046 m (tuning missed).

**Problem 1: the wave effect depends on the time step and shrinks as Δt → 0.**

| grid | θ | N | CMP shear | CMN shear |
|---|---|---|---|---|
| 760 × 20 | 0.6 | 400 | −0.011 | +0.100 |
| 760 × 20 | 0.55 | 400 | −0.026 | +0.105 |
| 760 × 20 | 0.55 | 200 | −0.058 | +0.144 |
| 380 × 20 | 0.55 | 200 | −0.069* | +0.164* |
| 380 × 20 | 0.55 | 400 | −0.031* | +0.134* |
| 380 × 20 | 0.55 | 800 | −0.007* | +0.090* |

(SP +0.072* on both grids.) On 380 columns the CMP discharge stays the same for N = 200/400/800 (mean u 0.074–0.076
m/s), so the shift is a genuine Δt effect; the steps shrink only by a factor of about 1.6 per halving, and the
extrapolation to Δt → 0 removes most of the following-wave effect. Findings so far:
- not θ: θ 0.6 → 0.55 at N = 400 changes little;
- not Δt/Δx (horizontal advection): the shift is the same on 380 and 760 columns;
- not the k-ε dynamics: with ν_t, k, ε frozen at the SP state (k-ε step overridden by a no-op, scratch probe) the
  shift remains (CMP −0.056 at N = 200, −0.023 at N = 400);
- not the pressure solver tolerance (1e-8).
Remaining candidates: the merged top layer and remap at the moving surface (it moves up to aωΔt per step), vertical
advection near the surface, the explicit relaxation of the open faces.

Cost note (for the search): the 380-column grid takes about 6 ms per step; CMP + CMN at N = 200 take 7 min. Larger
steps are limited by θ ≥ (1 + 2Fr)/(2(1 + Fr)) (the model's Froude bound): θ = 0.525 is not allowed with the current;
θ = 0.55 is borderline (instantaneous Fr 0.12 at the faces).

**Problem 2: the model does not keep the flume discharge with waves.** The open faces relax the combination
c·η − Q towards c·η_T − q (τ = 20 s), not the discharge itself; a hard discharge inflow would reflect the waves leaving
through the face. Without waves the discharge is right (0.0797 against 0.0805 m²/s); with waves it drifts:

| run | station mean u (m/s) | expected (q − M)/h |
|---|---|---|
| CMP 760 × 20, N 400 / 200 | 0.136 / 0.162 | 0.144 |
| CMN 760 × 20, N 400 / 200 | 0.178 / 0.202 | 0.183 |
| CMP 380 × 20 | 0.073–0.076 | 0.144 |
| CMP 380 × 20, k-ε frozen | 0.017 (mean Q 0.016 m²/s) | 0.144 |

The shear depends on the current strength, so part of every Δt difference on 760 columns is a discharge difference.
The size of the drift is not understood: through the face condition alone it needs mean face levels about 3 cm below
the target, which seems too large.

**Next steps when resumed:**
1. Diagnostic run (380 columns, about 10 min): period-mean level and discharge at both faces and at the station. Does
   the face condition hold its combination (then the level shifted and a slow "pump" correction of the inflow target
   level is needed), or does the face misbehave (then fix the face first)?
2. Control the discharge (fix of 1) and repeat the Δt comparison (N = 200/400) on 380 columns.
3. If the Δt effect remains: switch off the remaining candidates one at a time (scratch probe with method overrides).
4. Then repeat the 20/40-layer runs and update the README, the 2DV-13 row, the slow test and the tests document.

4 Acceptance tolerances
-----------------------

Set after the current-only run (the issue's procedure): the RMS of $U_\text{model}-U_\text{meas}$ over
0.02 m ≤ z ≤ 0.40 m, normalised by 0.16 m/s, with the 7 % calibration bias of the data stated alongside.

5 Wave streaming at the bed
---------------------------

### 5.1 Why the laminar check is replaced

The issue proposed laminar bottom streaming under progressive waves against Longuet-Higgins (1953) (data Mei, Liu &
Carter 1972: h = 0.13 m, T = 0.81 s). It does not fit the model:
- **No laminar bed.** `bed_roughness_length = 0` is a free-slip bed, `> 0` the log-law friction; a no-slip viscous bed
  would be a new feature.
- **Resolution.** Longuet-Higgins' result is the mean drift at the edge of the Stokes layer, δ = √(2ν/ω) ≈ 0.5 mm.
  It needs Δz ≈ 0.1 mm at the bed, about 1300 uniform layers. With 2–10 mm the layer lies inside the first cell and
  the drift above it follows from the friction law, not from the laminar layer.
- **Spin-up.** The steady laminar mean profile in that tank needs about h²/ν ≈ 1.7·10⁴ s (5 hours).

The laminar case moves to a separate issue together with a stretched vertical grid (or a 1DV bed-layer model).

### 5.2 What matters: the streaming stress

The mean stress that the waves transfer to the mean flow through the bed layer depends only on the energy dissipated
at the bed (from continuity and the momentum balance of the bed layer; Longuet-Higgins 1953, Deigaard & Fredsøe 1989):

```math
\frac{\tau_s}{\rho}=\overline{\tilde u\,\tilde w}\Big|_{\text{top of the bed layer}}=\frac{\overline{u_b\,\tau_b}}{\rho\,c}=\frac{D_f}{\rho\,c},
```

with u_b the orbital velocity above the bed layer, τ_b the bed stress and c the phase speed. The internal structure of
the layer does not enter, which is why the laminar result does not depend on ν. The model gets the streaming stress
right if (1) its bed stress has the right magnitude and phase and (2) the vertical momentum advection carries the wave
stress $\overline{\tilde u\tilde w}$ consistently. Both can be checked without resolving the layer.

Estimate for #WMN: u_b ≈ 0.18 m/s, orbital excursion A ≈ 0.04 m, A/k_s ≈ 34 (rough turbulent), wave friction factor
(Swart 1974) f_w ≈ 0.035, bed-layer thickness 1–5 mm (κu*_w/ω ≈ 2 mm). Then D_f/(ρc) ≈ 2·10⁻⁵ m²/s², about 40 % of the
current-only bed stress (5·10⁻⁵ m²/s²); a large part of the following/opposing effect, in line with Walstra et al.
(2000). Order of magnitude only: in the turbulent case the streaming can be weaker or reversed by wave asymmetry
(Trowbridge & Madsen 1984).

### 5.3 Wave friction of the wall function

The wall function applies the steady log law at the first cell centre z₁ = Δz/2, i.e. for the oscillating flow a
friction factor f_w = 2 (κ / ln(z₁/z₀))²:

| Δz | z₁ | f_w | of Swart's 0.035 |
|---|---|---|---|
| 25 mm (nz = 20) | 12.5 mm | 0.010 | 30 % |
| 12.5 mm (nz = 40) | 6.25 mm | 0.013 | 40 % |
| 10 mm | 5 mm | 0.014 | 40 % |
| 5 mm | 2.5 mm | 0.020 | 55 % |
| 2.5 mm | 1.25 mm | 0.028 | 80 % |

On the planned grids the wave dissipation and the streaming stress are therefore 2.5–3.5 times too small, a grid
effect (the current part is fine: the steady log law holds at any z₁ in the log layer). 2–5 mm puts the first cell at
or inside the turbulent bed layer, where a quasi-steady log law is reasonable, and resolves the measured near-bed
structure of #WMN (+0.01 m/s below 1 mm, reversal up to 15–20 mm, Fig. 4.9). Uniform 5 mm (nz = 100) costs about 5
times nz = 20; 2 mm everywhere (nz = 250) is too costly, which again points to a stretched grid.

### 5.4 Actions

1. `test_verify_wave_streaming_stress_budget` (new): progressive waves over the quadratic log-law bed, no current,
   Δz between 2 and 10 mm, wetting-drying step, generated at the left free face and absorbed at the right one. Checks,
   with named tolerances:
   - the wave-averaged stress at the top of the first cell, $\overline{\tilde u\tilde w}$, equals
     $\overline{u_b\tau_b}/(\rho c)$ (u_b, τ_b the model's own first-cell velocity and bed stress);
   - the amplitude decays along the channel as $d(E c_g)/dx = -D_f$, with $E=\tfrac12\rho g a^2$ and the model's
     $D_f=\overline{u_b\tau_b}$;
   - with the quadratic law, $D_f = \tfrac{2}{3\pi}\rho f_w u_b^3$ with f_w from the table above (analytical, for a
     sinusoidal u_b).
   These identities hold for any bed law, so they test the mechanism that matters in Klopman's flume. Interior
   viscosity small (molecular), the decay length long against the channel, so a and u_b vary slowly.
2. In the Klopman runs: report the wave friction the model has (from the decay or $\overline{u_b\tau_b}$ at the
   station) next to the profiles; one #WMN run at Δz = 5 mm (nz = 100) against Fig. 4.9.
3. If the f_w deficit of the coarse grids dominates the mean profiles: decide (outside #85) between a finer or
   stretched vertical grid and a wave-aware bed stress (a wall function with the phase lead and the f_w of a thin
   bed layer). A model change, so a separate decision.
4. Documentation: section in `docs/TESTS_TECHNICAL_DOCUMENTATION.md` for the new test; a short note in
   `docs/validation_cases.md` (2DV-13) on the wave friction of the grid.

References for this section: Longuet-Higgins (1953); Deigaard, R. & Fredsøe, J. (1989), Shear stress distribution in
dissipative water waves, *Coastal Eng.* 13, 357–378; Swart, D. H. (1974), Offshore sediment transport and equilibrium
beach profiles, Delft Hydraulics Publ. 131; Trowbridge, J. & Madsen, O. S. (1984), Turbulent wave boundary layers 2,
*J. Geophys. Res.* 89(C5), 7999–8007.
