Plan: relaxation zones for waves on a current (issue #86)
=======================================================

Issue: https://github.com/fwp-flowincode/swe-transport-2dv/issues/86. Prerequisite of the Klopman validation (#85).
Scope: the wetting-drying (cut-cell) time step `_step_forward_wetting_drying!` (`src/wetting_drying.jl`), hydrostatic and
non-hydrostatic, with and without k-epsilon. The fixed grid is out of scope.

0 Revised order and status
--------------------------

After review the order changed: first make the open boundary itself absorb dispersive waves (a characteristic-type
condition that fits the existing radiation outflow), then generation at both ends, and relaxation zones only for what
remains.

1. **Done: absorbing right face for dispersive waves.** `radiation_theta`, `nonhydro_open_face="radiation"`,
   `radiation_period` (`docs/numerics_open_boundaries.md` sections 5 and 7.2; tests in
   `test/test_open_boundary_radiation.jl`). Regular waves: 1.5 % at $kH=1.2$ instead of 6.9 %; the remainder is the
   interior's dispersion error. A simpler alternative to the 3D generating-absorbing conditions of Wellens & Borsboom
   (2020) and Borsboom & Jacobsen (2021), exact for one frequency.
2. **Next: generation with the characteristic open face** (Borsboom 2023, Mooiman 2024). The formulation is in
   `docs/numerics_open_boundaries.md` section 8. In short:
   - a natural condition (outgoing information): the face momentum plus a term with weight $\beta$;
   - an essential condition (incoming information) in increment form, with the incoming wave $\Delta F$ and a drift
     correction on the incoming characteristic variable instead of the level.

   It replaces the earlier idea of radiating the deviation from a target, which it contains. With $\beta=-\tfrac12$ (box
   form) the linear reflection drops from $(k\Delta x)^2/16$ to $(k\Delta x)^3/64$. Sub-steps:
   1. Done: the formulation (section 8).
   2. Done: normal-mode analysis of the discrete face (`characteristic_reflection` in
      `docs/figures/numerics_figures.jl`, section 8.3, Figure 6). Decisions:
      - $\beta=-\tfrac12$ with the coefficient $c-|U_n|$;
      - no third point;
      - the open face takes the upwind level as thickness. The present $\eta_b$ reflects to first order for outflow,
        $0.27\,Fr\,k\Delta x$;
      - $\beta$ is limited only below `open_face_wet_depth`;
      - the box form for $q_b$ becomes a model variant in step 3.
   3. Implementation, absorbing only ($\Delta F=0$), at both faces as option `open_face_scheme="characteristic"`. The
      existing tests stay unchanged; new tests check the reflection against Table 2 and the mean flow on a current with
      $(\eta_T,Q_T)$.
   4. Generation (given level and given discharge, the incoming $q_{in}$ in the non-hydrostatic mode). Test: a
      $kH=1.2$ wave on a 0.16 m/s current, generated at one face and absorbed at the other; check the amplitude
      against the target and the remaining reflection.
3. **Done: the free left face** (`left_free=true`), the mirror of the right one with the same options
   (`docs/numerics_open_boundaries.md` section 7.3); exact mirror symmetry verified. For the opposing case of #85 it still
   needs the generation of step 2 (the current enters there).
4. **Relaxation zones (sections 2.1-2.5 below)** only if the reflection tests of 2 and 3 need them.

1 Current state
---------------

| Need | What exists | Why it is not enough |
|---|---|---|
| absorb waves | `sponge_length`, `sponge_rate`: implicit damping $\hat u\leftarrow\hat u/(1+\Delta t\,r\,s^2)$ in the momentum predictor, right end only ([wetting_drying.jl:1451](../src/wetting_drying.jl)) | relaxes towards $u=0$, so it also removes a current; right end only |
| absorb waves at an open end | radiation / relaxed outflow (`docs/numerics_open_boundaries.md`) | long-wave speed: a component with $kh\approx1.2$ is reflected by about 9 % |
| generate waves | time-dependent inflow `left_velocity(t)` at the face $x=0$ | a velocity face reflects returning waves with $R=+1$; left end only |
| waves travelling upstream | none | the current enters at the left, so the opposing case needs generation at the right and absorption at the left |

The water level is never set directly in this step: it follows from the flux divergence
(`_wd_update_level_from_fluxes!`), which makes the volume exact and, with the free-surface solve, keeps every depth
$\ge0$ (`docs/depth_positivity.md`). The design keeps that property.

2 Method
--------

### 2.1 Velocity relaxation (stage A, the default)

In a zone of length $L_r$ next to a boundary, the momentum predictor gets a nudging term towards a target velocity
$u_T(x,z,t)$, implicit like the existing sponge:

```math
\hat u_{k,f} \leftarrow \frac{\hat u_{k,f} + \Delta t\,r(\xi_f)\,u_T(x_f,z_k,t^{n+1})}{1 + \Delta t\,r(\xi_f)},
\qquad r(\xi) = r_{\max}\,\beta(\xi),
```

with $\xi\in[0,1]$ the relative position in the zone (0 at the inner edge, 1 at the boundary) and the weight

```math
\beta(\xi) = \frac{\exp(\xi^{p}) - 1}{\exp(1) - 1},\qquad p = 3.5
```

(Jacobsen, Fuhrman & Fredsøe 2012, eq. 13 in their convention $\alpha_R=1-\beta$; equation number to be checked).
Non-hydrostatic: the same nudging of the predicted vertical velocity $w^*$ towards $w_T$ before the pressure problem
(`_wd_nonhydrostatic_correction!`, step 2).

WHY a rate instead of the blend $\phi\leftarrow\alpha\phi+(1-\alpha)\phi_T$ of Jacobsen et al.: that blend is applied once
per step, so its strength depends on $\Delta t$; the rate form converges to a time-continuous damping term, is
unconditionally stable (implicit), and is the existing sponge for $u_T=0$, $p=2$.

WHY velocities only: the level then still follows from the flux divergence. Volume conservation (to round-off in a
closed domain) and depth positivity hold unchanged, and the constituent transport and k-epsilon need no change. A
progressive-wave target $u_T$ imposes the matching level through continuity,
$\partial_t\eta=-\partial_x\int u\,dz$.

WHY $\hat u$ before the free-surface solve: the surface gradient of the solve is added afterwards, so the nudging acts
as a forcing term, not as a hard constraint. With $\Delta t\,r_{\max}\gg1$ the boundary end of the zone approaches
$u=u_T$.

### 2.2 Level relaxation (stage B, only if stage A fails the reflection test)

A volume source at the surface, $s_i(\eta_T-\eta_i^{n+1})$, implicit in the free-surface system:

```math
F_i(\eta) = V_i(\eta) + (T\eta)_i + \Delta t\,s_i\,(\eta_i - \eta_{T,i}) - b_i = 0 .
```

It adds $\Delta t\,s_i\ge0$ to the diagonal (the Newton matrix stays an M-matrix, $F$ stays convex). To be shown before
implementing: the positivity argument of `docs/depth_positivity.md` with this term. Also needed: the source volume in
`_wd_update_level_from_fluxes!`, the surface flux $w_{nz+1}$ in step 6, the volume budget, and the transport (the
source water carries the local cell value, so concentrations do not change).

### 2.3 Targets

`LinearWaveOnCurrent` (named parameters, SI units):
- amplitude $a$ (m), absolute period $T$ (s), direction $\pm1$, still-water depth $h$ (m), ramp time $t_r$ (s, smooth
  start $\tfrac12(1-\cos\pi t/t_r)$);
- current profile $U(z)$ (m/s): a function of $z$, or the log profile $U(z)=\frac{u_*}{\kappa}\ln(z/z_0)$ scaled to a
  discharge; uniform as a special case;
- wave number from the Doppler relation $(\omega-kU_e)^2=gk\tanh kh$ (Newton), with $U_e$ the depth mean or the
  weighted current $U_e=\frac{2k}{\sinh 2kh}\int_{-h}^0U\cosh 2k(z+h)\,dz$ of Kirby & Chen (1989, eq. to be checked);
- wave velocities (first order, intrinsic frequency $\sigma=\omega-kU_e$):
  $u_w=a\sigma\frac{\cosh k(z+h)}{\sinh kh}\cos(kx-\omega t)$, $w_w=a\sigma\frac{\sinh k(z+h)}{\sinh kh}\sin(kx-\omega t)$;
  optional second-order Stokes terms (Dean & Dalrymple 1991, ch. 11; bound second harmonic, no correction for the
  current, like Klopman's wave board);
- layers between the still-water level and the crest: the value at $z=0$ (constant extrapolation; WHY: the linear
  profile grows exponentially above $z=0$, and Wheeler stretching changes the target's mass flux);
- **mass-flux option** `transport = :eulerian | :total`. With `:total`, the current is scaled to
  $q-M$, with $M=\tfrac12 g a^2/c$ the Stokes transport (m²/s) of the target wave, so that the mean total transport is
  $q$. Klopman's flume (#85) needs `:total`.

Absorption zone: the same target with $a=0$ (current only). With $U=0$ and $p=2$ it reproduces the existing sponge.

### 2.4 Boundary faces

The zone's outer edge is the domain boundary:
- **left, generation:** the inflow face gets the target at $x=0$ (current plus wave), via the existing
  `left_velocity(t)`, so face and zone agree. Left, absorption: the discharge inlet as now.
- **right, absorption:** relaxed outflow to the mean level (`right_level` + `right_relaxation_time`); the zone removes
  the waves before they reach the face.
- **right, generation (opposing case):** the outflow face needs the target too. To be decided in the implementation:
  a time-dependent `right_level` or a prescribed right profile; check what `set_x_inflow_outflow!` supports.

### 2.5 Interface

The attach pattern of the vegetation and sediment models:
`attach_relaxation_zones!(model, [RelaxationZone(side=:left, length=..., rate=..., target=...), ...])`, stored in
`WetDryState` (a typed field, like `remap`). The target is evaluated once per step and face column.
PERF: the target needs `cosh`/`sinh` per face layer and step, about $10^3$ columns × 40 layers × 2 zones; the
depth profiles are precomputed once (they do not depend on time), and only $\cos(kx-\omega t)$ is evaluated per step.

3 Verification and validation
-----------------------------

Tests in `test/test_relaxation_zones.jl` (new file), all in the wetting-drying step:

| Test | Set-up | Criterion |
|---|---|---|
| `test_verify_relaxation_still_water_at_rest` | zones with $a=0$, $U=0$, closed ends | $u$, $\eta$ stay zero to round-off |
| `test_verify_relaxation_current_passes_unchanged` | uniform flow, target = inflow profile, both pressure modes | steady state to round-off; the zones are invisible |
| `test_verify_relaxation_is_sponge_for_zero_target` | $u_T=0$, $p=2$ | identical to `sponge_length`/`sponge_rate` |
| `test_verify_relaxation_volume_conserved` | waves, closed domain | volume to round-off (stage A) |
| `test_validate_generated_wave_height_and_phase` | waves only, then on $\pm U$ | first harmonic just inside the zone within a named tolerance of $a$ and the target phase |
| `test_validate_wave_number_on_current` | $kh\approx1.2$, $U=-0.16,0,+0.16$ m/s | measured $k$ (phase along the flume) within a named tolerance of the Doppler relation |
| `test_validate_relaxation_zone_reflection` | $kh=0.5$ to $1.5$, $\lvert U\rvert/c\le0.1$, zone 0.5 to 2 wavelengths | $\lvert R\rvert$ below a named limit (target 2 % at 1.5 wavelengths); also reported against $L_r$ and $r_{\max}$ |
| `test_verify_relaxation_time_step_independent` | halve $\Delta t$ | reflection and wave height change by less than a named tolerance |

Reflection: from the envelope of the first harmonic along the flume, $R=(H_{\max}-H_{\min})/(H_{\max}+H_{\min})$, or
the two-gauge separation of Goda & Suzuki (1976). Measurement window after the ramp, before any long-wave return.

Hydrostatic mode: the same tests at $kh\le0.3$ (long waves), since the hydrostatic mode has no dispersion.

4 Work order
------------

1. Target module (`src/relaxation_zones.jl`): Doppler wave number, profiles, mass-flux option. Unit tests against hand
   values (no model run).
2. Stage A in the momentum predictor and $w^*$; the still-water, current, sponge-equivalence and volume tests.
3. Wave tests: height, phase, wave number, reflection; tune the defaults of $L_r$, $r_{\max}$, $p$ from the reflection
   study.
4. Right-hand generation (opposing case) and its boundary face.
5. Stage B only if step 3 misses the reflection target.
6. Documentation: a section in `docs/numerics_open_boundaries.md` (method, weight, reflection against zone length,
   limitations) and in `docs/TESTS_TECHNICAL_DOCUMENTATION.md`.

5 Open points for review
------------------------

- Default zone length and $r_{\max}$: to be set from the reflection study (expected: about 1–2 wavelengths,
  $r_{\max}$ of a few $\omega$).
- Whether the second-order Stokes target is needed for #85 ($ak\approx0.14$); first order first, then measure the free
  second harmonic it produces.
- The right-hand generation face (2.4).
- Whether the existing sponge should later become a special case of the zones (one code path) or stay as it is.

References
----------

- Jacobsen, N. G., Fuhrman, D. R. & Fredsøe, J. (2012). A wave generation toolbox for the open-source CFD library:
  OpenFoam®. *Int. J. Numer. Meth. Fluids* 70, 1073–1088.
- Mayer, S., Garapon, A. & Sørensen, L. S. (1998). A fractional step method for unsteady free-surface flow with
  applications to non-linear wave dynamics. *Int. J. Numer. Meth. Fluids* 28, 293–315.
- Israeli, M. & Orszag, S. A. (1981). Approximation of radiation boundary conditions. *J. Comput. Phys.* 41, 115–135.
- Kirby, J. T. & Chen, T.-M. (1989). Surface waves on vertically sheared flows: approximate dispersion relations.
  *J. Geophys. Res.* 94(C1), 1013–1027.
- Dean, R. G. & Dalrymple, R. A. (1991). *Water Wave Mechanics for Engineers and Scientists*. World Scientific.
- Goda, Y. & Suzuki, Y. (1976). Estimation of incident and reflected waves in random wave experiments. *Proc. 15th ICCE*,
  828–845.
- Casulli, V. (2009). A high-resolution wetting and drying algorithm for free-surface hydrodynamics. *Int. J. Numer.
  Meth. Fluids* 60, 391–408.
