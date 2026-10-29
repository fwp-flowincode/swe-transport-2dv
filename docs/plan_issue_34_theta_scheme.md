Plan: theta scheme for the time integration (issue #34)
======================================================

Issue: https://github.com/fwp-flowincode/swe-transport-2dv/issues/34. Theta-implicitness in the Casulli sense:

- the velocity in the mass flux of the continuity equation at $n+\theta$ (layer thicknesses at level $n$);
- the free-surface level in the pressure gradient of the momentum equation at $n+\theta$ (thicknesses at level $n$);
- non-hydrostatic: the non-hydrostatic pressure probably also at $n+\theta$;
- bed friction fully implicit ($n+1$), unless an analysis shows that $\theta$ is better;
- horizontal advection explicit at level $n$; other horizontal terms case by case;
- vertical advection implicit (with or without $\theta$), guided by robustness;
- a $\theta$-dependency analysis and tests of the temporal accuracy (wave propagation).

1 Current state
---------------

Both time steps are the $\theta=1$ case of the scheme below.

| Term | Fixed grid (`step_forward!`, flat or masked bed, variable top layer, periodic) | Wetting and drying (`_step_forward_wetting_drying!`) |
|---|---|---|
| free surface in the pressure gradient | implicit, $\eta^{n+1}$ | implicit, $\eta^{n+1}$ |
| velocity in the continuity flux | $u^{n+1}$, thicknesses of level $n$ | $u^{n+1}$, cut-cell thicknesses of level $n$ |
| horizontal advection | explicit (`upwind1`/`tvd2`, unsplit with the vertical part) | explicit (`_wd_advection_tendency`) |
| vertical advection | explicit, inside the same unsplit kernel | explicit, capped |
| vertical viscosity | explicit (`diffusion_term!`, both directions) | implicit, applied to $\hat u$ before the solve |
| bed friction | explicit, stops if $\Delta t~c_b\lvert u\rvert/\Delta z>1$ | not supported (rejected) |
| non-hydrostatic pressure | projection after the surface solve, full $\Delta t$ | projection, gradient added to $\hat u$, solve repeated |
| transport | $u^{n+1}$, $w^{n+1}$ | $u^{n+1}$, $w^{n+1}$ |

The implicit free surface with $\theta=1$ damps gravity waves by about $2\pi^2/N$ per period ($N$ steps per period),
for example 5 % per period at $N=400$ (seen in the standing waves of issue #31).

2 Scheme (Casulli & Cattani 1994; Casulli & Cheng 1992)
------------------------------------------------------

Per u-face column (layers $k$, thicknesses $\Delta Z^n$ of level $n$), with $u^{n+\theta}=\theta u^{n+1}+(1-\theta)u^n$ and
$\eta^{n+\theta}$ likewise:

```math
A^n u^{n+1} = G^n - g\,\theta\,\frac{\Delta t}{\Delta x}\,(\eta^{n+1}_{i+1}-\eta^{n+1}_i)\,\Delta Z^n,
```

```math
G^n = \Delta Z^n\bigl[u^n + \Delta t\,(\text{explicit terms})\bigr] - g(1-\theta)\frac{\Delta t}{\Delta x}(\eta^n_{i+1}-\eta^n_i)\,\Delta Z^n,
```

where the explicit terms are horizontal advection, horizontal viscosity, baroclinic pressure, body force, wind (as now),
and $A^n$ is the tridiagonal matrix of the implicit vertical terms:

```math
A^n = \mathrm{diag}(\Delta Z^n) + \Delta t\,\bigl(\text{vertical viscosity} + \text{upwind vertical advection with } W^n + \gamma^n e_1e_1^\top\bigr),
```

with the linearised bed friction $\gamma^n = c_b\lvert u^n_1\rvert$ (fully implicit). Upwind vertical advection in advective
form gives row sums $\Delta Z_k$ plus the friction term, so $A^n$ is a strictly diagonally dominant M-matrix:
$(A^n)^{-1}\ge0$, and $\Delta Z^\top (A^n)^{-1}\Delta Z>0$.

Continuity with the $\theta$-velocity and the level-$n$ thicknesses:

```math
\eta^{n+1}_i = \eta^n_i - \frac{\Delta t}{\Delta x}\Bigl[(\Delta Z^\top u^{n+\theta})_{i+1/2} - (\Delta Z^\top u^{n+\theta})_{i-1/2}\Bigr].
```

Eliminating $u^{n+1}$ gives the symmetric tridiagonal system (positive definite, M-matrix)

```math
\eta^{n+1}_i - g\theta^2\frac{\Delta t^2}{\Delta x^2}\Bigl[a_{i+1/2}(\eta^{n+1}_{i+1}-\eta^{n+1}_i) - a_{i-1/2}(\eta^{n+1}_i-\eta^{n+1}_{i-1})\Bigr] = \eta^n_i - \frac{\Delta t}{\Delta x}\Bigl[\theta\,(\Delta Z^\top A^{-1}G)_{i+1/2} + (1-\theta)(\Delta Z^\top u^n)_{i+1/2} - (\ldots)_{i-1/2}\Bigr],
```

with $a_f = \Delta Z^\top A^{-1}\Delta Z$ (equal to the water depth $H_f$ when $A=\mathrm{diag}(\Delta Z)$, the present scheme).

- **Wetting and drying:** the same with $V_i(\eta)$ instead of $\eta_i$ (Casulli 2009). The positivity proof
  (`docs/depth_positivity.md`) carries over: $a_f\ge0$ still holds, $T$ keeps its structure, the flux-derived depth
  still equals $V_i(\eta^{n+1})\ge0$, and condition (C) is unchanged (the explicit $(1-\theta)$ fluxes telescope within a
  component). Proof and document to be updated.
- **Non-hydrostatic pressure** (Casulli 1999; Casulli & Zanolli 2002): $q$ at $n+\theta$, i.e.
  $-\frac{\Delta t}{\rho_0}\nabla(\theta q^{n+1} + (1-\theta) q^n)$; the $(1-\theta)q^n$ part goes into $G^n$ and the projection
  corrects with $\theta\Delta t$. Local continuity holds for $u^{n+1}$ and therefore for $u^{n+\theta}$.
  Implemented in incremental form (van Kan 1986): the predictor carries the full $\nabla q^n$, the projection solves for
  $q^{n+1}-q^n$. Its top velocity is $(\eta_t-(1-\theta)w_s^n)/\theta$, where $w_s^n$ is the column divergence of $u^n$
  with the thicknesses of level $n$. Taking the stored surface value of $w$ instead (not the divergence of $u^n$ after
  the initialisation, nor with the variable top layer) put a depth-uniform $q$ with a slope into the first steps; the
  incremental form never removes it. In the steady wind set-up the surface slope then came out at $-3$ times the exact
  one at $\theta=0.7$ (found with the default $\theta<1$; fixed).
- **Transport:** with $u^{n+\theta}$ and the matching $w^{n+\theta}$ (from the layer continuity), so that the transport is
  consistent with the volume change (a uniform constituent stays uniform, mass is conserved).
- **Vertical velocity output:** $w$ from the continuity of $u^{n+\theta}$; the kinematic condition
  $w_{top}=(\eta^{n+1}-\eta^n)/\Delta t$ then closes exactly for every $\theta$.

3 Theta analysis
----------------

### 3.1 Gravity waves (no mean flow)

Linear shallow water on the C-grid, Fourier mode with $\mu = 2\frac{c~\Delta t}{\Delta x}\sin\frac{k\Delta x}{2}$,
$c=\sqrt{gH}$. The scheme gives $(G-1)^2 = -\mu^2(\theta G + 1-\theta)^2$, so

```math
G = \frac{1 \pm i\mu(1-\theta)}{1 \mp i\mu\theta},\qquad \lvert G\rvert^2 = \frac{1+\mu^2(1-\theta)^2}{1+\mu^2\theta^2}.
```

- $\theta\ge\frac12$: $\lvert G\rvert\le1$ for every $\Delta t$ (unconditionally stable); $\theta<\frac12$ is unstable for
  large $\mu$ and is rejected (`THETA_MIN`).
- $\theta=\frac12$: $\lvert G\rvert=1$ (no damping), second order in time; $\theta>\frac12$: first order.
- Damping per period ($N$ steps, $\mu\approx2\pi/N$): $\lvert G\rvert^N\approx\exp\bigl(-4\pi^2(\theta-\frac12)/N\bigr)$, i.e.
  $2\pi^2/N$ at $\theta=1$, $0.8\pi^2/N$ at $\theta=0.7$ and $0.2\pi^2/N$ at $\theta=0.55$.
- Phase: $\arg G = \arctan(\mu(1-\theta)) + \arctan(\mu\theta)$, compared with $\omega\Delta t$; the period error is
  $O(\mu^2)$ at $\theta=\frac12$ and $O(\mu)$ otherwise.

Verified: the standing waves of `test_verify_theta_standing_wave_amplification` match the exact $G$ in every mode.

### 3.2 Bed friction and vertical advection

Bed friction: implicit $\gamma u^{n+1}$ is unconditionally stable and keeps the sign of $u$ (no overshoot to reversed
flow at any $\Delta t$); $\theta$-weighted friction is second order but can reverse $u$ when
$\Delta t~\gamma/\Delta z>1/(1-\theta)$. Implemented: fully implicit, as the issue suggests.

Vertical advection: implicit upwind with $W^n$ (first order in $z$, M-matrix, no time-step limit from $W\Delta t/\Delta z$).
A $\theta$-centred implicit form would be second order but no M-matrix (oscillations in thin cut layers), so upwind is
implemented (robustness).

### 3.3 Stability with a mean flow (explicit momentum advection)

Section 3.1 assumes $U=0$. With a mean flow, the explicitly advected velocity and the $\theta$-implicit gravity waves
interact, and $\theta$ close to $\frac12$ becomes unstable. Linearised 1D system about a uniform flow $U>0$, with the
explicit advection factor $\alpha$ of one step for $u$, and $\beta=\alpha$ (surface advected: variable top layer, wetting
and drying) or $\beta=1$ (fixed layer thicknesses) for $\eta$. Per Fourier mode the eigenvalues $\lambda$ solve

```math
(\alpha-\lambda)(\beta-\lambda) + \mu^2\,(1-\theta+\theta\lambda)^2 = 0 .
```

For $\alpha=\beta$, this gives $\lambda = (\alpha \mp i\mu(1-\theta))/(1 \pm i\mu\theta)$. Long waves are advected
without damping, $\alpha\approx e^{-i\psi}$ with $\psi=U k\Delta t$, so
$\lvert\alpha - i\mu(1-\theta)\rvert^2 = 1 + 2\mu(1-\theta)\sin\psi + \mu^2(1-\theta)^2$. This is at most
$1+\mu^2\theta^2$ for every wavenumber exactly when

```math
\mathrm{Fr} = \frac{\lvert U\rvert}{\sqrt{gH}} \;\le\; \mathrm{Fr}_{\max}(\theta) = \frac{2\theta-1}{2(1-\theta)},
\qquad\text{equivalently}\qquad \theta \;\ge\; \frac{1+2\,\mathrm{Fr}}{2(1+\mathrm{Fr})}.
```

Interpretation: the explicit advection shifts the phase by $U k\Delta t$ before the implicit gravity step, and the
explicit part $(1-\theta)$ of the gravity term turns this phase shift into growth for the wave travelling with the
flow; the implicit part $\theta$ has to damp it.

| $\theta$ | 0.5 | 0.55 | 0.6 | 0.65 | **0.7** | 0.75 | 0.8 | 0.9 | 1 |
|---|---|---|---|---|---|---|---|---|---|
| $\mathrm{Fr_{max}}$ | 0 | 0.11 | 0.25 | 0.43 | **0.67** | 1 | 1.5 | 4 | $\infty$ |

The full eigenvalue problem was evaluated numerically (`experiments/theta_scheme/stability_analysis.jl`) for
advective Courant numbers $C=U\Delta t/\Delta x\in(0,1]$, all gravity-wave Courant numbers and all wavenumbers:

- Lax-Wendroff advection (the smooth limit of the MC-limited `tvd2`), surface advected: the minimal $\theta$ equals
  the closed form for every $C$; the bound is sharp (spectral radius $1+2.5\cdot10^{-8}$ at $1.05~\mathrm{Fr_{max}}$).
- Fixed layer thicknesses ($\beta=1$): the minimal $\theta$ is lower (for example 0.655 instead of 0.75 at Fr = 1).
- `upwind1`: its numerical diffusion lowers the minimal $\theta$ (to $\frac12$ for small $C$ and large Fr); it reaches the
  closed form at $C=1$.
- For $\mathrm{Fr}\le\mathrm{Fr_{max}}(\theta)$ the spectral radius is $\le1$ in all four combinations (largest value
  $1-1.8\cdot10^{-14}$), so the closed form is a sufficient condition for both schemes and both continuity forms.

Above $\mathrm{Fr_{max}}$ the growth is slow (for example a factor of about 1.04 per step at $\theta=0.55$, Fr = 0.45),
so short runs or runs with physical damping (bed friction, viscosity) may not show it. Consequences for the
default:

- $\theta=0.55$ (the first default) allows only Fr ≤ 0.11. Model check: a hydrostatic periodic channel (fixed layer
  thicknesses, $U=1$ m/s, $H=0.5$ m, Fr = 0.45, `tvd2`, $C=0.2$); the analysis for fixed layer thicknesses gives a
  minimal $\theta$ of 0.59 there. With the consistent momentum fluxes of issue #37 (16 cells per wavelength, nz = 2 and
  8, identical results) the surface perturbation grows by 36 in 50 s at $\theta=0.55$ and by 2 at 0.57, and decays from
  $\theta=0.59$ on (factor 0.11 at 0.59, 0.026 at 0.6). Before issue #37 the growth was about $10^7$ (4 cells per
  wavelength), mostly from the advection defect; it also decayed from $\theta=0.6$ on. With `upwind1` every $\theta$ decays (its
  numerical diffusion hides the growth). Above `theta_froude_max` the periodic channel still decays (checked up to
  1.4 times the bound): with fixed layer thicknesses the bound is conservative, it is sharp for an advected surface. The k-ε open channel (Fr = 0.43) stays stable at $\theta=0.55$, because its
  bed friction and turbulent viscosity damp the weak growth; its failure during development came from the
  non-hydrostatic top-velocity error described in section 2.
- $\theta=0.7$ (default, decided 2026-10-28) allows Fr ≤ 2/3, which covers subcritical river and estuary flows. It
  keeps $2.5$ times less wave damping than $\theta=1$.
- Supercritical flows with Fr > 1 need $\theta\ge\frac{1+2\mathrm{Fr}}{2(1+\mathrm{Fr})}$, for example
  $\theta\ge0.8$ at Fr = 1.5. $\theta=1$ is stable for every Fr.

The Froude bound concerns the coupling with $\theta$ only. The explicit momentum advection also has limits of its own,
independent of $\theta$ (already present with $\theta=1$, the former scheme). Measured in a periodic channel with a
uniform flow and a small surface perturbation ($H=0.5$ m, 16 cells per wavelength, hydrostatic):

- `upwind1`, $\theta=1$, Fr = 0.2: stable up to $C\approx0.45$ with 2 layers and $C\approx0.3$ with 8 layers. The limit
  depends on the layer thickness, so the vertical part of the unsplit advection (or its top boundary) is involved.
- `tvd2`, $\theta=1$, Fr = 0.5, 2 layers: the perturbation grows even at $C=0.1$ (factor $1.8\cdot10^4$ in 200 s; at
  $\theta=0.7$ $9\cdot10^6$), though linear Lax-Wendroff theory predicts stability. The MC limiter is nonlinear (it
  acts on the perturbation for every amplitude), so this is not covered by the linear analysis.

These limits are not part of issue #34; they are to be analysed separately (follow-up issue).

Resolved in issue #37: both limits came from inconsistent discrete momentum fluxes, not from the limiter. The fixed grid
had no momentum flux through its top (where $w=\partial_t\eta$), and the vertical velocity at the u control volumes was
smoothed over three interfaces, so the u control volume was not divergence-free. With the top-row advective form on the
fixed grid and the C-grid face velocities, the cases above decay up to $C=0.9$ for nz = 2 ... 16, both schemes,
$\theta=0.7$ and 1 (`experiments/advection_stability/README.md`).

4 Configuration (decided 2026-10-28)
------------------------------------

- `theta::Float64 = 0.7` (dimensionless, $\frac12\le\theta\le1$, smaller values rejected): free surface, continuity
  flux, non-hydrostatic pressure. Decided first as 0.55. Changed to 0.7 (decision 2026-10-28) after the mean-flow
  analysis of section 3.3: 0.55 is stable only up to Fr = 0.11, 0.7 up to Fr = 2/3. `theta_froude_max(theta)` returns
  the bound.
- `implicit_vertical_terms::Bool = false` (decision: behind a switch, so that the validated cases stay unchanged):
  vertical viscosity, vertical advection (first-order upwind with $W^n$, decision) and bed friction in the matrix $A^n$
  with the $\Delta Z^\top A^{-1}\Delta Z$ coupling. With the switch on, `tvd2` momentum advection is horizontal only.
- `bed_roughness_length > 0` with wetting and drying: possible with the implicit friction (currently rejected).

5 Tests (TDD)
-------------

- `test_verify_theta_*`
  - linear standing wave: amplitude ratio per period and period against the analytical $G$ for several $\theta$ and
    $\Delta t$ (fixed grid, variable top layer, masked bed, wetting and drying, 1D and 2DV);
  - temporal order: errors against a run with a much smaller $\Delta t$ on the same grid; observed order 2 at
    $\theta=\frac12$, 1 at $\theta=1$;
  - volume and constituent conservation, kinematic closure and uniform-density preservation for $\theta\ne1$;
  - lake at rest exact for every $\theta$ (also wetting and drying);
  - $\theta=1$ reproduces the present free surface (where the vertical terms are unchanged);
  - $\theta<\frac12$ rejected.
- implicit vertical terms:
  - bed friction: steady state independent of $\Delta t$, stable far beyond the old explicit limit, decay of a uniform
    flow against the exact linear decay;
  - vertical advection: stable for $W\Delta t/\Delta z\gg1$, consistent (MMS on a prescribed $W$);
  - $a_f=\Delta Z^\top A^{-1}\Delta Z$ equals $H_f$ without friction (unchanged results).
- wetting and drying: depth positivity for $\theta=\frac12$ at large $\Delta t$; Ritter and runup with $\theta$.
- non-hydrostatic: dispersion of standing waves for $\theta=\frac12$ and $1$.

6 Experiments (`experiments/theta_scheme/`)
--------------------------------------------

- amplitude and period error against $\theta$ and $N$ (steps per period): numerical against analytical $G$;
- temporal convergence for $\theta=\frac12, 0.55, 1$;
- stability with a mean flow: minimal $\theta$ against Courant and Froude number (`stability_analysis.jl`, done);
- nonlinear and wet/dry: Ritter dam break and solitary runup for several $\theta$ (robustness, oscillations);
- non-hydrostatic standing wave for several $\theta$;
- README with figures, as for the other experiments.

7 Steps
-------

1. Fixed grid, hydrostatic: $\theta$ in the surface solve, continuity flux, $w$ and transport (flat, masked, periodic,
   variable top layer).
2. Implicit vertical terms with the $A^{-1}$ coupling: vertical viscosity, upwind vertical advection, implicit bed
   friction (fixed grid; the wet/dry viscosity moves into the same matrix).
3. Wetting and drying with $\theta$, positivity proof updated.
4. Non-hydrostatic with $q$ at $n+\theta$ (fixed grid and wetting and drying).
5. Analysis, experiments, technical documentation.

References
----------

- Casulli (1990), Semi-implicit finite difference methods for the two-dimensional shallow water equations, J. Comput.
  Phys. 86, 56-74
- Casulli & Cheng (1992), Int. J. Numer. Methods Fluids 15, 629-648
- Casulli & Cattani (1994), Stability, accuracy and efficiency of a semi-implicit method for three-dimensional shallow
  water flow, Comput. Math. Appl. 27(4), 99-112
- Casulli (1999), Int. J. Numer. Methods Fluids 30, 425-440 (non-hydrostatic pressure)
- Casulli & Zanolli (2002), Semi-implicit numerical modeling of nonhydrostatic free-surface flows for environmental
  problems, Math. Comput. Model. 36, 1131-1149
- Casulli (2009), Int. J. Numer. Methods Fluids 60, 391-408 (wetting and drying, volume formulation)
