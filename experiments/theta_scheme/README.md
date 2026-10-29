Theta scheme (issue #34)
========================

Analysis and experiments for the theta time integration of the free surface (`ModelConfig(theta=...)`, default 0.7).
Derivations: [docs/plan_issue_34_theta_scheme.md](../../docs/plan_issue_34_theta_scheme.md), section 3.

Allowed theta range
-------------------

| Condition | Required | Source |
|---|---|---|
| gravity waves, any time step | $\frac12\le\theta\le1$ (enforced by `ModelConfig`) | amplification factor, plan section 3.1 |
| mean flow with Froude number Fr, explicit momentum advection | $\mathrm{Fr}\le\frac{2\theta-1}{2(1-\theta)}$ = `theta_froude_max(theta)` | von Neumann analysis, plan section 3.3 |

| theta | 0.5 | 0.55 | 0.6 | 0.65 | **0.7** | 0.75 | 0.8 | 0.9 | 1 |
|---|---|---|---|---|---|---|---|---|---|
| largest stable Fr | 0 | 0.11 | 0.25 | 0.43 | **0.67** | 1 | 1.5 | 4 | any |

Fr is the depth-averaged velocity over the long-wave speed, $\lvert U\rvert/\sqrt{gH}$, of the fastest part of the flow.
The model computes it after every step (`surface_froude_number(model)`) and warns once per run when it exceeds the
bound by more than 0.01, with the theta it needs (`theta_for_froude(Fr)`; issue #39).
Above the bound gravity waves on the current grow slowly; with first-order upwind momentum advection the scheme's
numerical diffusion often hides this (the bound is sharp for the smooth limit of `tvd2`). The explicit momentum
advection had Courant-number limits of its own, independent of theta (plan section 3.3); they were removed in issue #37
([../advection_stability/README.md](../advection_stability/README.md)).

Scripts
-------

- `stability_analysis.jl`: minimal stable theta against the advective Courant number and Fr for `upwind1` and the
  Lax-Wendroff limit of `tvd2`, with and without advection of the surface; checks that the closed form is sufficient
  and sharp. Run with `julia --project experiments/theta_scheme/stability_analysis.jl` (a few seconds).
