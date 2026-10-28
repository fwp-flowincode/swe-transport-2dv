Depth positivity of the wetting-drying scheme
=============================================

This note proves that the wetting-drying time step (`ModelConfig(wetting_drying=true)`, `src/wetting_drying.jl`,
issue #31) never produces a negative water depth, states the time-step conditions that are, and are not, needed, and
describes the run-time check. Notation follows the plan `docs/plan_issue_31_wetting_drying.md`.

Summary
-------

1. **Positivity holds for every time step $\Delta t > 0$** (Theorem 1). It is guaranteed by the outflow limiter in the
   final continuity step, not by the semi-implicit free-surface solve (which can go below the bed, section 4). The proof uses only the structure of that last
   step, so it holds for any way the velocities $u^{n+1}$ are computed (hydrostatic or non-hydrostatic, first- or
   second-order advection), as long as the step ends with the limited continuity update.
2. **The free-surface solve alone is not positive.** Without the limiter, the semi-implicit scheme with face depths at
   time level $n$ can produce negative depths when $\Delta t$ is large or the flow reverses at a face (section 4). This is
   flagged on purpose: positivity is bought by modifying the outflow where the limiter acts.
3. **The limiter is inactive** (the scheme is the unmodified semi-implicit scheme) if the flow does not reverse at
   the outflow faces of a column during the step and the Courant-type condition
   $\frac{\Delta t}{\Delta x}\max_k\left(u^{+}_{k,i+1/2} + u^{-}_{k,i-1/2}\right) \le 1$ holds (Lemma 2). In all validation
   cases (Ritter, Carrier-Greenspan, sloshing; 1D and 2DV) it never acted.
4. **Floating point:** a computed depth can fall below zero by round-off only, bounded by
   $\gamma_{2n_z+8}\,S_i$ (section 5). Such a depth is set to an exact zero; anything below the bound stops the run with a
   `DomainError`, as does a negative depth found at the start of a step.

1 Setting
---------

Columns $i = 1,\dots,n_x$ of width $\Delta x$, layers $k = 1,\dots,n_z$ on the fixed interfaces $z_k = (k-1)\Delta z$; the top
layer is open above. Column $i$ has the bed level $z_{b,i}$ and the water level $\zeta_i = H + \eta_i$ ($H$ = grid depth). Its
depth (water volume per unit area) is

$$V_i = \zeta_i - z_{b,i},$$

and layer $k$ holds the thickness

$$h_{k,i} = \max\bigl(0,\ \min(z_{k+1}, \zeta_i) - \max(z_k, z_{b,i})\bigr) \ge 0, \qquad \sum_k h_{k,i} = \max(V_i, 0).$$

Face $f = i+1/2$ between columns $i$ and $i+1$ has the bed $z_{b,f} = \max(z_{b,i}, z_{b,i+1})$ and the upwind level
$\zeta_f$: $\zeta_i$ if $\sum_k u^n_{k,f} > 0$, $\zeta_{i+1}$ if $< 0$, $\max(\zeta_i,\zeta_{i+1})$ if $= 0$. Its layer thicknesses
$h_{k,f}$ follow from the same formula with $(\zeta_f, z_{b,f})$, and are set to zero if $\zeta_f - z_{b,f}$ does not exceed
`dry_depth_threshold`. Boundary faces use the adjacent column.

**Layer thicknesses are non-negative by construction**, for cells and faces, whatever $\zeta$ is. The only quantity whose
sign is at stake is the column depth $V_i$; the layer thicknesses are derived from it. Positivity of $V_i$ is therefore the
complete statement.

2 The final continuity step
---------------------------

After the velocities $u^{n+1}_{k,f}$ have been computed (momentum predictor, free-surface solve, velocity correction; in
the non-hydrostatic variant also the pressure correction), the step ends with `_wd_limit_fluxes_and_update_level!`:

1. Layer discharges $q_{k,f} = h_{k,f}\,u^{n+1}_{k,f}$ (m²/s).
2. Outflow volume of column $i$ during the step (m):
   $$O_i = \frac{\Delta t}{\Delta x}\sum_{k=1}^{n_z}\Bigl(q^{+}_{k,i+1/2} + q^{-}_{k,i-1/2}\Bigr), \qquad a^{+} = \max(a,0),\ a^{-} = \max(-a,0).$$
3. Limiter, with $V^n_i = \max(\zeta^n_i - z_{b,i}, 0)$:
   $$\theta_i = \begin{cases} 1, & O_i \le V^n_i,\\ V^n_i / O_i, & O_i > V^n_i,\end{cases} \qquad 0 \le \theta_i \le 1,\quad \theta_i O_i \le V^n_i.$$
4. Limited discharges: an outflowing face-layer discharge is multiplied by the limiter of the column it leaves,
   $$\tilde q_{k,f} = \begin{cases}\theta_{L(f)}\,q_{k,f}, & q_{k,f} > 0,\\ \theta_{R(f)}\,q_{k,f}, & q_{k,f} < 0,\end{cases}$$
   with $L(f), R(f)$ the columns left and right of $f$. A discharge entering through a boundary face from outside the
   domain is not limited ($\theta := 1$ there); one leaving through a boundary face is limited by the adjacent column.
   The velocities are scaled in the same way, $\tilde u_{k,f} = \tilde q_{k,f}/h_{k,f}$.
5. New level from the flux divergence:
   $$\eta^{n+1}_i = \eta^n_i - \frac{\Delta t}{\Delta x}\sum_{k}\bigl(\tilde q_{k,i+1/2} - \tilde q_{k,i-1/2}\bigr).$$

The same limited discharges are used for the vertical fluxes and the constituent transport, so the step is
conservative: summing step 5 over $i$ telescopes to the boundary fluxes.

3 Proof of positivity
---------------------

**Theorem 1.** Let $V^n_i \ge 0$ for all $i$. Then, in exact arithmetic, $V^{n+1}_i = \zeta^{n+1}_i - z_{b,i} \ge 0$ for all
$i$, for every $\Delta t > 0$, every velocity field $u^{n+1}$ and every bed.

*Proof.* Fix a column $i$. Split the sum in step 5 into the discharges that leave $i$ and those that enter it. A face-layer
discharge leaves $i$ through its right face if $q_{k,i+1/2} > 0$ and through its left face if $q_{k,i-1/2} < 0$; in both
cases column $i$ is the column it leaves, so it is multiplied by $\theta_i$. All other non-zero discharges at the faces of
$i$ enter $i$; they are multiplied by the limiter of the neighbour (or by 1 at an inflow boundary). Hence

$$V^{n+1}_i = V^n_i - \theta_i O_i + I_i, \qquad
I_i = \frac{\Delta t}{\Delta x}\sum_k\Bigl(\theta_{i-1}\,q^{+}_{k,i-1/2} + \theta_{i+1}\,q^{-}_{k,i+1/2}\Bigr) \ge 0,$$

since all $\theta \ge 0$ and $q^{\pm} \ge 0$ (with $\theta_0 = \theta_{n_x+1} := 1$ for unlimited boundary inflow). Because
$V^n_i \ge 0$, the max in the definition of $V^n_i$ is inactive and $V^n_i$ is the true depth. By step 3, $\theta_i O_i \le V^n_i$
in both cases: if $O_i \le V^n_i$ then $\theta_i O_i = O_i \le V^n_i$; otherwise $\theta_i O_i = V^n_i$. Therefore

$$V^{n+1}_i \ge V^n_i - V^n_i + I_i = I_i \ge 0. \qquad\blacksquare$$

**Corollary (induction).** The initial state has $V^0_i \ge 0$: `initialize_water_level!` sets a dry column to its bed and
the `Model` constructor clamps generic initial states to the bed. By Theorem 1, $V^n_i \ge 0$ for all $n$.

*Remarks.*
- Nothing in the proof refers to how $u^{n+1}$ was computed. Any variant of the step (non-hydrostatic correction,
  second-order advection, other forcing) keeps positivity provided it ends with steps 1-5 and uses the face
  thicknesses $h_{k,f} \ge 0$. This is a structural invariant of the code: every wetting-drying step must end with
  `_wd_limit_fluxes_and_update_level!`.
- The limiter is a one-pass, local scaling; it uses only $V^n_i$ and ignores the inflow $I_i$ that would partly
  compensate the outflow. It is therefore conservative in the sense of being sufficient, not minimal.
- A column that is limited ($\theta_i < 1$) ends with $V^{n+1}_i = I_i$: it is emptied by its outflow, apart from what flows in.

4 What is not positive: the free-surface solve alone
----------------------------------------------------

The semi-implicit free-surface equation (step 3 of the time step, Casulli & Cheng 1992) solves

$$\eta^{n+1}_i - \eta^n_i + \frac{\Delta t}{\Delta x}\Bigl(\hat U_{i+1/2} - \hat U_{i-1/2}\Bigr)
- \frac{g\,\Delta t^2}{\Delta x^2}\Bigl(H_{i+1/2}(\eta^{n+1}_{i+1}-\eta^{n+1}_i) - H_{i-1/2}(\eta^{n+1}_i-\eta^{n+1}_{i-1})\Bigr) = 0,$$

with the face depths $H_f = \sum_k h_{k,f}$ taken at time level $n$. Its solution need not satisfy $\zeta^{n+1}_i \ge z_{b,i}$:

- the system is linear in $\eta^{n+1}$ with depths frozen at time $n$; nothing couples the outflow to the depth that is
  left at time $n+1$. A column with a small depth and a large implicit gravity-driven outflow (large $\Delta t$, steep
  surface gradient) is drained below its bed. The test `test_verify_wd_depth_positivity_large_time_step_*` runs a dam
  break at an advective Courant number of 3, where the limiter acts ($\min\theta < 1$). Measured in that case: the level of
  the free-surface solve itself falls $6.8\cdot10^{-8}$ m below the bed in 1D (the limited step keeps it at or above the
  bed); in 2DV it stays at or above the bed in this case. The counterexample is small but real, so positivity cannot be
  claimed for the solve alone;
- the upwind direction of the face depth is taken from $u^n$; if the flow reverses during the step, the face depth comes
  from the column the water now flows *into*, and the outflow of the other column is computed with a depth that it may
  not have.

Casulli (2009) obtains positivity within the solve by a nonlinear (Newton-type) system for the wet volume; the present
scheme instead restores it after the solve with the limiter. **Consequence:** where the limiter acts, the final
velocities differ from those of the free-surface solve (they are scaled down), so momentum and the surface gradient are
locally inconsistent for that step. This is a first-order local modification, confined to columns that would otherwise
run dry within one step.

5 Floating-point arithmetic
---------------------------

In floating point, $\theta_i O_i \le V^n_i$ and the cancellation in step 5 hold only up to rounding. With unit roundoff
$\varepsilon$ and $\gamma_m = m\varepsilon/(1-m\varepsilon)$ (Higham 2002, Lemma 3.1 and Sec. 4.2):

- $O_i$ is a sum of $2n_z$ non-negative terms times $\Delta t/\Delta x$: $\hat O_i = O_i(1+\delta_1)$, $|\delta_1| \le \gamma_{2n_z+1}$;
- $\hat\theta_i = (V^n_i/\hat O_i)(1+\delta_2)$, $|\delta_2| \le \varepsilon$, hence the limited outflow is
  $\hat\theta_i O_i \le V^n_i(1+\gamma_{2n_z+3})$;
- the divergence is a sum of $2n_z$ terms; forming $\eta^{n+1}$ and $\zeta^{n+1} - z_b$ adds a few roundings.

Altogether the computed depth satisfies

$$\widehat{V}^{n+1}_i \ \ge\ -\,\gamma_{m}\,S_i, \qquad m = 2n_z + 8, \qquad S_i = H + |\eta^n_i| + |\eta^{n+1}_i| + |z_{b,i}| + O_i,$$

i.e. it can be negative by a few ulps of the magnitudes involved, never more. The code (`_wd_depth_roundoff`) uses this
bound with the safety factor `WD_POSITIVITY_ROUNDOFF_FACTOR = 2`:

- $-2\gamma_m S_i \le \widehat V^{n+1}_i < 0$: round-off; the column is set to an exactly dry level and the (ulp-sized)
  volume added is accumulated in `model.wetdry.clamped_volume` (m²; zero in all validation cases);
- $\widehat V^{n+1}_i < -2\gamma_m S_i$: violates Theorem 1, so it indicates a defect; the run stops with a `DomainError`.

**Exactly dry levels.** A dry column stores $\eta_i = z_{b,i} - H$; in floating point $H + \eta_i$ can round to one ulp below
$z_{b,i}$, i.e. a computed depth of $-10^{-16}$ m. `_wd_dry_eta` therefore stores the smallest float $\eta$ with
$\mathrm{fl}(H+\eta) \ge z_b$, so that the computed depth of a dry column is exactly $\ge 0$ (IEEE subtraction gives
$\mathrm{fl}(a-b) \ge 0 \iff a \ge b$). With this, every depth the model computes is $\ge 0$, not only $\ge -$round-off.

**Start-of-step check.** `_wd_check_depth_positivity` stops the run if a column enters a step with a depth below
$-2\gamma_m S_i$; this can only happen if the state was modified from outside (test
`test_verify_wd_negative_depth_is_rejected`).

6 Time-step conditions
----------------------

**(a) Positivity:** none. Theorem 1 holds for every $\Delta t > 0$.

**(b) Limiter inactive** (the scheme is the unmodified semi-implicit scheme with upwind face depths).

*Lemma 2.* Suppose that at every outflow face of column $i$ the face depth is upwinded from $i$ (the flow direction of the
step agrees with that of $u^n$ used for the upwinding). Then

$$\frac{\Delta t}{\Delta x}\max_k\Bigl(u^{n+1,+}_{k,i+1/2} + u^{n+1,-}_{k,i-1/2}\Bigr) \le 1 \quad\Longrightarrow\quad \theta_i = 1.$$

*Proof.* For an outflow face $f$ of $i$ the upwind level is $\zeta_f = \zeta_i$, and $z_{b,f} \ge z_{b,i}$ (the face bed is
the higher neighbour). Hence for every layer
$h_{k,f} = \max(0, \min(z_{k+1},\zeta_i) - \max(z_k, z_{b,f})) \le \max(0, \min(z_{k+1},\zeta_i) - \max(z_k, z_{b,i})) = h_{k,i}$.
Then

$$O_i = \frac{\Delta t}{\Delta x}\sum_k\bigl(h_{k,i+1/2}u^{+}_{k,i+1/2} + h_{k,i-1/2}u^{-}_{k,i-1/2}\bigr)
\le \frac{\Delta t}{\Delta x}\sum_k h_{k,i}\bigl(u^{+}_{k,i+1/2} + u^{-}_{k,i-1/2}\bigr)
\le \frac{\Delta t}{\Delta x}\max_k\bigl(u^{+}_{k,i+1/2} + u^{-}_{k,i-1/2}\bigr)\,V^n_i \le V^n_i. \qquad\blacksquare$$

This is the usual advective Courant condition, written for a column with outflow on both sides. It is **not** required
for positivity; it only tells when the limiter cannot act. If the flow reverses at a face during the step, the face
depth is taken from the other column and the limiter may act at any $\Delta t$ (rarely, and only at the front).
Measured: in the Ritter (Courant number 0.06), Carrier-Greenspan and sloshing runs, 1D and 2DV, $\theta_i = 1$ in every
column and step, and no depth was clamped.

**(c) Other conditions of the step** (not related to depth positivity, listed for completeness):
- explicit momentum advection: stable without condition because the update is capped to a convex combination
  (`_wd_advection_tendency`), but accurate only for $|u|\Delta t/\Delta x \lesssim 1$;
- constituent transport: conservative for every $\Delta t$; it is monotone (no new extrema) only if the outflow of each
  cell does not exceed its old volume, $\Delta t\,(\text{outflow of cell } (k,i)) \le h^n_{k,i}\Delta x$, which thin cut cells can
  violate at any $\Delta t$; values outside the initial bounds are then clipped and the clipped mass redistributed
  (plan, section 2.2). Unlike the depth, the density bounds are therefore enforced, not proven;
- free surface: implicit, no gravity-wave Courant limit (Casulli & Cheng 1992).

7 Tests
-------

| Test | Checks |
|---|---|
| `test_verify_wd_depth_positivity_large_time_step_{1d,2dv}` | dam break onto a dry bed and beach at Courant number 3: limiter acts, depth $\ge 0$ in every step, clamped volume $\le 10^{-14}$ of the total, volume and mass conserved to round-off |
| `test_verify_wd_negative_depth_is_rejected` | a state with a negative depth stops the step with a `DomainError` |
| `test_verify_wd_volume_and_constituent_conservation_{1d,2dv}` | depth $\ge 0$ over 1000 steps of sloshing with flooding and drying |

References
----------

- Casulli & Cheng (1992), Int. J. Numer. Methods Fluids 15, 629-648
- Casulli (2009), A high-resolution wetting and drying algorithm for free-surface hydrodynamics, Int. J. Numer. Methods
  Fluids 60, 391-408
- Stelling & Duinmeijer (2003), Int. J. Numer. Methods Fluids 43, 1329-1354
- Higham (2002), Accuracy and Stability of Numerical Algorithms, 2nd ed., SIAM (Lemma 3.1, Sec. 4.2: summation error)
