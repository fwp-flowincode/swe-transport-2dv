Depth positivity of the wetting-drying scheme
=============================================

This note proves that the wetting-drying time step (`ModelConfig(wetting_drying=true)`, `src/wetting_drying.jl`,
issue #31) never produces a negative water depth, states the conditions under which this holds, and describes the
run-time check. Notation follows the plan `docs/plan_issue_31_wetting_drying.md`.

Summary
-------

1. **Positivity is enforced inside the free-surface solve, not afterwards.** Following Casulli (2009), the unknown water
   volume of a column is the non-negative function $V_i(\eta) = \max(0, H + \eta_i - z_{b,i})$ of its level, and the
   free-surface equation becomes the mildly nonlinear system $V(\eta) + T\eta = b$. Its solution gives velocities whose
   flux divergence reproduces $V(\eta^{n+1}) \ge 0$ exactly (Theorem 1). Nothing is corrected after the solve: the
   final velocities satisfy the discrete momentum equations that produced the surface.
2. **No time-step restriction.** Theorem 1 holds for every $\Delta t > 0$. The Newton iteration of Brugnano & Casulli
   (2008) converges for every $\Delta t$ in at most $n_x + 1$ steps (Theorem 2).
3. **The only condition** is that a prescribed outflow through a domain boundary does not exceed the water volume that is
   connected to that boundary (section 4); otherwise no non-negative solution exists at all, and the run stops. Closed
   boundaries and prescribed inflow always satisfy it.
4. **Floating point:** the computed depth equals $V_i(\eta^{n+1})$ up to the final residual of the solve and the round-off
   of the continuity sums, bounded in section 5. A depth below zero within this bound is stored as an exactly dry level;
   a larger negative depth (or one found at the start of a step) stops the run with a `DomainError`.
5. **Non-hydrostatic mode:** the non-hydrostatic pressure gradient enters the momentum predictor and the free-surface solve
   is repeated, so the final velocities again come from the positive solve (section 6).
6. In the validation cases (Ritter, Carrier-Greenspan, sloshing; 1D and 2DV) the solve takes exactly one Newton
   iteration per step (the former linear solve). The nonlinear iteration acts only where a column would otherwise be
   drained below its bed, e.g. at advective Courant numbers of 6 and more in the dam break of the positivity test
   (2-4 iterations).

History: a first version (commit 7d00172) solved the linear semi-implicit system and scaled the outflow of columns
that would run dry afterwards. Positivity held, but the scaled velocities were no longer the solution of the discrete
momentum equations where the limiter acted. The present formulation replaces that limiter.

1 Setting
---------

Columns $i = 1,\dots,n_x$ of width $\Delta x$, layers $k = 1,\dots,n_z$ on the fixed interfaces $z_k = (k-1)\Delta z$; the top
layer is open above. Column $i$ has the bed level $z_{b,i}$ and the level $\zeta_i = H + \eta_i$ ($H$ = grid depth). Its
water volume per unit area (its depth) is

$$V_i(\eta_i) = \max(0,\ H + \eta_i - z_{b,i}) \ \ge 0,$$

and layer $k$ holds $h_{k,i} = \max\bigl(0, \min(z_{k+1}, \zeta_i) - \max(z_k, z_{b,i})\bigr) \ge 0$, with
$\sum_k h_{k,i} = V_i$. Face $f = i+1/2$ has the bed $z_{b,f} = \max(z_{b,i}, z_{b,i+1})$ and layer thicknesses $h_{k,f} \ge 0$
from the upwind level of time $n$ (zero if the upwind depth is below `dry_depth_threshold`), and the total
$H_f = \sum_k h_{k,f}$.

**Layer thicknesses are non-negative by construction**, for cells and faces, whatever $\eta$ is. The only quantity whose
sign is at stake is the column volume; the layer thicknesses are derived from the level. Positivity of the column depth
is therefore the complete statement.

2 The free-surface system
-------------------------

The momentum predictor gives $\hat u_{k,f}$ on the wet face layers (and the boundary velocities). In every wet face layer
the new velocity is

$$u_{k,f}^{n+1} = \hat u_{k,f} - g\thinspace\Delta t\thinspace\frac{\eta_{i+1} - \eta_i}{\Delta x}, \qquad f = i+1/2 \text{ (interior)},$$

so the column transport is $Q_f = \sum_k h_{k,f}u_{k,f}^{n+1} = \hat U_f - g\thinspace\Delta t\thinspace H_f(\eta_{i+1}-\eta_i)/\Delta x$ with
$\hat U_f = \sum_k h_{k,f}\hat u_{k,f}$ (boundary faces: prescribed, no gradient term). The continuity of column $i$,

$$V_i(\eta_i^{n+1}) = V_i(\eta^n_i) - \frac{\Delta t}{\Delta x}\bigl(Q_{i+1/2} - Q_{i-1/2}\bigr),$$

is the system

$$F(\eta) := V(\eta) + T\eta - b = 0,\qquad
(T\eta)_i = a_{i+1/2}(\eta_i - \eta_{i+1}) + a_{i-1/2}(\eta_i - \eta_{i-1}),\qquad
b_i = V_i(\eta^n_i) - \frac{\Delta t}{\Delta x}\bigl(\hat U_{i+1/2} - \hat U_{i-1/2}\bigr),$$

with $a_f = g\thinspace\Delta t^2 H_f/\Delta x^2 \ge 0$ ($a = 0$ at closed faces and at the domain boundary faces). Casulli (2009)
introduced this volume formulation for sub-grid bathymetry; here each column has one bed level, so $V_i$ has a single kink.

Properties used below:

- **(T)** $T$ is symmetric, $T_{ij} \le 0$ for $i \ne j$, and every row sums to zero: $T$ is a positive semidefinite
  Z-matrix. Call two columns *connected* if they share a face with $a_f > 0$; this splits the columns into components $C$.
  Within a component $T$ is irreducible, and $\sum_{i\in C}(T\eta)_i = 0$ for every $\eta$.
- **(M)** If $D$ is a non-negative diagonal matrix with $D_{ii} > 0$ for at least one column of every component, then $D + T$
  is an irreducibly diagonally dominant Z-matrix on each component, hence a non-singular M-matrix with
  $(D+T)^{-1} \ge 0$ elementwise (Varga 2000, Sec. 3.5; Berman & Plemmons 1994, ch. 6).
- **(V)** Each $V_i$ is convex, non-decreasing and piecewise linear. With the right derivative
  $p_i(\eta) = 1$ if $H + \eta_i \ge z_{b,i}$, else $0$, and $P(\eta) = \mathrm{diag}(p_i)$, convexity gives the subgradient
  inequality $V(y) \ge V(x) + P(x)(y - x)$ for all $x, y$ (componentwise).

3 Positivity and conservation
-----------------------------

**Theorem 1.** Let $\eta^{\ast}$ solve $F(\eta^{\ast}) = 0$ and let $u^{n+1}$ be the velocities of section 2 with $\eta = \eta^{\ast}$. Then
the new depth from the flux divergence,

$$V_i^{n+1} := V_i(\eta^n_i) - \frac{\Delta t}{\Delta x}\sum_k\bigl(h_{k,i+1/2}u_{k,i+1/2}^{n+1} - h_{k,i-1/2}u_{k,i-1/2}^{n+1}\bigr),$$

equals $V_i(\eta_i^{\ast}) \ge 0$ for every column, for every $\Delta t > 0$; and $\sum_i V_i^{n+1}\Delta x$ changes only by the
boundary fluxes.

*Proof.* By section 2 and the definition of $T$,

$$\frac{\Delta t}{\Delta x}\bigl(Q_{i+1/2} - Q_{i-1/2}\bigr) = \frac{\Delta t}{\Delta x}\bigl(\hat U_{i+1/2} - \hat U_{i-1/2}\bigr) - \frac{g\Delta t^2}{\Delta x^2}\bigl[H_{i+1/2}(\eta_{i+1}^{\ast}-\eta_i^{\ast}) - H_{i-1/2}(\eta_i^{\ast}-\eta_{i-1}^{\ast})\bigr] = \frac{\Delta t}{\Delta x}\bigl(\hat U_{i+1/2} - \hat U_{i-1/2}\bigr) + (T\eta^{\ast})_i.$$

Hence
$V_i^{n+1} = b_i - (T\eta^{\ast})_i = V_i(\eta_i^{\ast}) \ge 0$, by $F(\eta^{\ast}) = 0$ and the definition of $V_i$. Summing the flux form over
$i$ telescopes to the boundary fluxes. $\blacksquare$

The positivity is a property of the solution of the system; no time-step condition enters. What remains is that the
system has a solution and that the solver finds it (Theorem 2).

**Induction.** The initial depths are $\ge 0$ (`initialize_water_level!` sets dry columns to their bed; the `Model`
constructor clamps generic initial states to the bed). By Theorem 1 the depths stay $\ge 0$ in every step.

4 Existence and convergence of the Newton iteration
---------------------------------------------------

The solver (`_wd_free_surface_solve!`) is Newton's method with the right derivative (Brugnano & Casulli 2008):

$$\bigl(P(\eta^m) + T\bigr)(\eta^{m+1} - \eta^m) = -F(\eta^m), \qquad \eta^0 = \eta^n.$$

**Condition (C).** Every component $C$ that contains at least one face with $a_f > 0$ satisfies $\sum_{i\in C} b_i > 0$.

Interpretation: faces between components are closed ($H_f = 0$, so $\hat U_f = 0$), hence
$\sum_{i\in C} b_i = \sum_{i \in C}V_i(\eta^n_i) + \frac{\Delta t}{\Delta x}(\text{net prescribed inflow through the domain boundary faces of } C)$.
A component with an open face holds water ($V > 0$ upwind of that face), so (C) holds for closed boundaries and for
prescribed inflow, for every $\Delta t$. It fails only if a prescribed boundary *outflow* removes at least the whole volume of
its component within one step; then the continuity equations have no solution with $V \ge 0$ at all (they would require
a negative volume), and no scheme can satisfy them.

**Theorem 2.** Under (C), every Newton matrix $P(\eta^m) + T$ is non-singular, the iterates satisfy
$\eta^{m+1} \le \eta^m$ for $m \ge 1$, and the iteration reaches $F(\eta^m) = 0$ after at most $n_x + 1$ steps (exact arithmetic).

*Proof.*

(a) *First step.* The depths at time $n$ are $\ge 0$, so $H + \eta^n_i \ge z_{b,i}$ and $p_i(\eta^0) = 1$ for all $i$
(right derivative at the kink). $I + T$ is non-singular by (M).

(b) *Non-negative residual.* For $m \ge 1$, the Newton equation of step $m-1$ and (V) with $x = \eta^{m-1}$, $y = \eta^m$ give
$$V(\eta^m) \ge V(\eta^{m-1}) + P(\eta^{m-1})(\eta^m - \eta^{m-1}) = b - T\eta^m,$$
i.e. $F(\eta^m) \ge 0$.

(c) *Non-singularity* for $m \ge 1$. Suppose $P(\eta^m) + T$ were singular. By (M) there is a component $C$ with $p_i(\eta^m) = 0$
for all $i \in C$ (all columns of $C$ dry in the iterate). Then $V_i(\eta^m) = 0$ on $C$ and, summing $F$ over $C$ with (T),
$\sum_{i\in C}F_i(\eta^m) = -\sum_{i\in C} b_i$. By (b) the left side is $\ge 0$, so $\sum_C b_i \le 0$. For a component with an open
face this contradicts (C). A single column without open faces has $T_{ii} = 0$, $b_i = V_i(\eta^n_i)$ and
$F_i(\eta^0) = 0$; its Newton row reads $p_i\thinspace\delta_i = -F_i$ with $p_i(\eta^0) = 1$, so $\delta_i = 0$, its level never changes
and $p_i = 1$ throughout. Hence every matrix is non-singular and, by (M), $(P(\eta^m)+T)^{-1} \ge 0$.

(d) *Monotonicity.* For $m \ge 1$: $(P(\eta^m)+T)(\eta^{m+1}-\eta^m) = -F(\eta^m) \le 0$, and the inverse is non-negative, so
$\eta^{m+1} \le \eta^m$.

(e) *Finite termination.* $p_i$ can only switch from 1 to 0: at $m = 0$ all $p_i = 1$, and for $m \ge 1$ the levels do not
increase while $p_i = 1 \iff \eta_i \ge z_{b,i} - H$. So $P$ changes at most $n_x$ times. If $P(\eta^m) = P(\eta^{m-1})$ for some
$m \ge 1$, then $\eta^m$ and $\eta^{m-1}$ lie on the same linear piece of every $V_i$ (both at or above the kink, where $V_i$ has
slope 1, or both below it, where $V_i = 0$), so $V(\eta^m) = V(\eta^{m-1}) + P(\eta^{m-1})(\eta^m - \eta^{m-1})$ holds with
equality, and the Newton equation of step $m-1$ gives $F(\eta^m) = 0$. Therefore a solution is reached after at most
$n_x + 1$ steps. In particular a solution exists. $\blacksquare$

*Remarks.*
- The volumes $V_i(\eta^{\ast})$ are unique: if $\eta^{\ast}, \tilde\eta$ both solve the system,
  $0 = (V(\eta^{\ast}) - V(\tilde\eta))^\top(\eta^{\ast}-\tilde\eta) + (\eta^{\ast}-\tilde\eta)^\top T(\eta^{\ast}-\tilde\eta)$, both terms are
  $\ge 0$ (monotone $V$, semidefinite $T$), so $V(\eta^{\ast}) = V(\tilde\eta)$ wherever the columns are wet. The level of a column
  that ends dry is not unique (any level at or below its bed); it only enters the velocities at faces whose fluxes it
  balances, and the stored level is reset to the bed (section 5).
- A lake at rest gives $F(\eta^n) = 0$ exactly (zero predictor transport, equal levels across every open face), so the
  iteration stops before the first step and $\eta$ does not change by round-off.
- Without drying in the step, $P(\eta^1) = I$ and the solution is reached after one step: the linear semi-implicit
  scheme of Casulli & Cheng (1992). This is the case in all validation runs.

5 Floating-point arithmetic and the run-time check
--------------------------------------------------

The iteration stops when $|F_i| \le$ `WD_NEWTON_RTOL` $\cdot\thinspace s_i$ for all $i$ ($s_i$ = sum of the magnitudes in $F_i$,
`WD_NEWTON_RTOL` $= 10^{-13}$; by (e) the exact solution is reached in finitely many steps, so the final residual is
round-off). The new level is then computed from the flux divergence (`_wd_update_level_from_fluxes!`), which conserves
the volume to round-off. By Theorem 1 the computed depth is

$$\widehat V_i^{n+1} = V_i(\eta^m_i) - F_i(\eta^m) + \text{(rounding)}, \qquad
|\text{rounding}| \le \gamma_{2n_z+8}\thinspace S_i,$$

with $\gamma_k = k\varepsilon/(1-k\varepsilon)$ (Higham 2002, Lemma 3.1 and Sec. 4.2: a sum of $2n_z$ discharges and a few further
roundings) and $S_i = H + |\eta^n_i| + |\eta_i^{n+1}| + |z_{b,i}| + O_i$, where $O_i$ is the outflow volume of the step. Hence

$$\widehat V_i^{n+1} \ \ge\ -\bigl(|F_i| + \gamma_{2n_z+8}\thinspace S_i\bigr).$$

The code (`_wd_depth_roundoff`) uses this bound with the safety factor `WD_POSITIVITY_ROUNDOFF_FACTOR = 2` on the
rounding term:

- $-(|F_i| + 2\gamma S_i) \le \widehat V_i^{n+1} < 0$: round-off; the column is stored as exactly dry and the volume added is
  accumulated in `model.wetdry.clamped_volume` (m²; zero in all validation cases, $\le 3\cdot10^{-16}$ m² in the
  large-time-step tests);
- $\widehat V_i^{n+1} < -(|F_i| + 2\gamma S_i)$: the solve did not satisfy its equations; the run stops with a `DomainError`.

**Exactly dry levels.** A dry column stores the smallest float $\eta$ with $\mathrm{fl}(H + \eta) \ge z_b$ (`_wd_dry_eta`);
$\eta = z_b - H$ alone can round to one ulp below the bed. IEEE subtraction gives $\mathrm{fl}(a - b) \ge 0 \iff a \ge b$, so
the computed depth of a dry column is exactly $\ge 0$, and with it every depth the model computes.

**Start-of-step check.** `_wd_check_depth_positivity` stops the run if a column enters a step with a negative depth
beyond round-off; this can only happen if the state was modified from outside (test
`test_verify_wd_negative_depth_is_rejected`). It also guarantees the premise of Theorem 2 (a): all $p_i(\eta^0) = 1$.

**Singular matrix.** If (C) is violated, step (c) of the proof fails and a pivot of the tridiagonal elimination becomes
$\le 0$; the solver then stops with a `DomainError` naming the column.

6 Non-hydrostatic mode
----------------------

The non-hydrostatic pressure $q$ is obtained from a projection of the velocities of a first free-surface solve on the
wet cells below the surface cell of each column ($q = 0$ in the surface cell). Its gradient $-\frac{\Delta t}{\rho_0}\partial_x q$ is
then **added to the predictor** $\hat u$, and the free-surface solve of section 2 is repeated
(`_wd_nonhydrostatic_correction!`). The final velocities are those of this second solve, so Theorems 1 and 2 apply
unchanged. Correcting the velocities directly after the solve (as in Casulli 1999) would change the column transports
after the solve and void Theorem 1. The price is that the continuity of the cells below the surface holds only
approximately (the two solves differ by a depth-uniform change of the surface gradient); the vertical velocity then
follows from the exact layer continuity.

7 Time-step conditions
----------------------

**(a) Depth positivity and volume conservation:** none (Theorems 1 and 2), apart from condition (C) for prescribed
boundary outflow.

**(b) Other conditions of the step** (not related to the depth, for completeness):
- explicit momentum advection: stable without condition because the update is capped (`_wd_advection_tendency`), but
  accurate only for $|u|\Delta t/\Delta x \lesssim 1$;
- constituent transport: conservative for every $\Delta t$; monotone (no new extrema) only if the outflow of each cell does
  not exceed its old volume, which thin cut cells can violate at any $\Delta t$; values outside the initial bounds are then
  clipped and the clipped mass redistributed. The density bounds are enforced, not proven;
- free surface: implicit, no gravity-wave Courant limit (Casulli & Cheng 1992).

8 Tests
-------

| Test | Checks |
|---|---|
| `test_verify_wd_depth_positivity_large_time_step_{1d,2dv}_{hydro,nonhydro}_{upwind1,tvd2}` | dam break onto a dry bed and beach at advective Courant number 6: the Newton iteration is needed (columns fall dry within the solve), depth $\ge 0$ in every step, clamped volume $\le 10^{-14}$ of the total, volume and mass conserved to round-off |
| `test_verify_wd_negative_depth_is_rejected` | a state with a negative depth stops the step with a `DomainError` |
| `test_verify_wd_volume_and_constituent_conservation_*` | depth $\ge 0$ over 1000 steps of sloshing with flooding and drying |
| `test_verify_wd_lake_at_rest_*` | $F(\eta^n) = 0$: no iteration, level and velocity unchanged exactly |

References
----------

- Casulli & Cheng (1992), Int. J. Numer. Methods Fluids 15, 629-648
- Casulli (1999), A semi-implicit finite difference method for non-hydrostatic, free-surface flows, Int. J. Numer.
  Methods Fluids 30, 425-440
- Brugnano & Casulli (2008), Iterative solution of piecewise linear systems, SIAM J. Sci. Comput. 30, 463-472
- Casulli (2009), A high-resolution wetting and drying algorithm for free-surface hydrodynamics, Int. J. Numer. Methods
  Fluids 60, 391-408
- Varga (2000), Matrix Iterative Analysis, 2nd ed., Springer (irreducibly diagonally dominant M-matrices)
- Berman & Plemmons (1994), Nonnegative Matrices in the Mathematical Sciences, SIAM (M-matrices, monotone inverse)
- Higham (2002), Accuracy and Stability of Numerical Algorithms, 2nd ed., SIAM (summation error)
- Stelling & Duinmeijer (2003), Int. J. Numer. Methods Fluids 43, 1329-1354
