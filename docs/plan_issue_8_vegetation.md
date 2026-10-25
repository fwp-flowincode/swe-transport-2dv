Plan: issue #8, review and update the vegetation tests
=====================================================

Issue #8 asks for vegetation tests that cover rigid and flexible vegetation, submerged and emergent vegetation, patches
with averaged (representative) quantities as well as individual stems, and comparisons with reference or analytical
solutions where possible. This plan reviews the current vegetation model, proposes the model changes the tests need,
and lists the reference solutions. Status: reviewed; decisions in section 7.

1 Current state
---------------

**Model** (`src/vegetation.jl`, coupling in `step_forward!`):

- Only individual stems (`VegetationStem`), always flexible. No patch/canopy representation with averaged quantities,
  no rigid option (only a large `stiffness`).
- No stem density or lateral spacing. The 2DV model is per metre width, and the stem force F (N) is deposited as
  F / (rho0 dx dz), i.e. implicitly one stem per metre of width.
- The force on the fluid is not the force on the stem:
  - it is multiplied by `VEGETATION_FEEDBACK_RELAXATION = 0.35`;
  - it is multiplied again by a per-stem `source_strength` (2.4 in one experiment);
  - it is smoothed along the stem.
- The stem mechanics is not a beam:
  - "bending" is a second-derivative term (a tensioned string), scaled by a gain `VEGETATION_RESPONSE_GAIN = 8`;
  - velocities and node steps are clamped;
  - the two lowest nodes are fixed;
  - stem buoyancy, which matters for flexible vegetation (Luhar & Nepf 2011), is missing.
- The drag is explicit: dense canopies need small time steps.
- An optional "vegetation turbulence" adds nu = strength * d * |u| near the stems (ad hoc).

**Tests** (`test/test_vegetation.jl`):

- Two tests have errored for a long time because of a dispatch bug: `vegetation_source!` without a pressure passes
  `p = nothing`, and no method accepts it.
- The other two check only signs: the stem bends, a wave is damped more than without the stem, and a lock-exchange
  front is slower. There is no comparison with a reference.

**Experiments** (`experiments/vegetation/`): uniform flow (flexible; stiff short with tuned `source_strength`),
oscillating flow, wave damping, lock exchange with one stem. None has a README or a reference.

2 Proposed model changes
------------------------

### 2.1 Canopy patches with averaged quantities (new)

A patch is described by its x-range [x0, x1], stem height h_v (m), stem diameter or blade width b (m), stem density n
(stems/m²) and drag coefficient C_D. The frontal area per unit volume is a = n b (1/m).

- **Drag:** the force per unit volume on the fluid is F = 1/2 rho0 C_D a u|u|, applied for z < h_v (the whole depth
  if emergent). The top cell of a submerged canopy is weighted by the fraction of the cell below h_v.
- **Implicit treatment:** the drag enters the u predictor linearised implicitly,
  u* = u_explicit / (1 + dt 1/2 C_D a |u|). This is unconditionally stable for dense canopies.
- **Rigid patch:** h_v is fixed.
- **Flexible patch:** the effective (reconfigured) height follows from the local velocity with the effective-length
  model of Luhar & Nepf (2011):
  - effective length: l_e/l = 1 - (1 - 0.9 Ca^(-1/3)) / (1 + Ca^(-3/2) (8 + B^(3/2)));
  - Cauchy number Ca = rho C_D b U² l³ / (2 EI) and buoyancy parameter B = Delta rho g b t l³ / (EI), for blade
    thickness t;
  - inputs: bending stiffness EI and stem density.
- **Porosity** (solid volume fraction phi = n pi b²/4) is neglected at first; this is valid for sparse canopies,
  phi < about 0.1, and will be documented.

### 2.2 Individual stems, rebuilt on physical grounds

- **Lateral spacing:** a stem represents a row of stems with lateral spacing Delta_y (m), with default 1 m (the
  current implicit value). The force per volume becomes F / (rho0 dx dz Delta_y).
- **Momentum:** the force on the fluid equals minus the force on the stem (Newton's third law). The relaxation factor,
  `source_strength`, the load smoothing and the gains are removed.
- **Rigid stem:** a straight cylinder with drag along its submerged length, no dynamics.
- **Flexible stem:** a quasi-static, large-deflection inextensible beam (elastica) with drag and buoyancy. Its
  posture is solved each step from EI dθ/ds etc. with distributed drag on the projected (normal) velocity. This is the
  model behind Luhar & Nepf (2011). It is valid when the stem responds faster than the flow changes; for waves see
  section 5.
- The current stem API (`VegetationStem`, `VegetationModel`, `attach_vegetation!`) is kept where possible, but the
  tuning parameters (`stiffness` as a string tension, `damping`, `mass_per_length`, `source_strength`) are replaced by
  EI, density and spacing.

### 2.3 Small additions needed for the reference cases

- `body_force_x` (m/s²) in `ModelConfig`: a uniform pressure gradient (channel slope) for steady flow in a periodic
  channel. It is the simplest way to obtain uniform, fully developed flow through a canopy.
- The dispatch bug (`p = nothing`) is fixed.

3 Reference solutions and tests
-------------------------------

| # | Case | Reference | Kind |
|---|---|---|---|
| R1 | Patch drag kernel | discrete force balance: the momentum removed from the fluid equals the integral of 1/2 rho C_D a u² | `test_verify_*` |
| R2 | Emergent rigid canopy, periodic channel, body force G | uniform velocity U = sqrt(2G/(C_D a)) (exact; the free-slip bed carries no stress) | `test_validate_*` |
| R3 | Submerged rigid canopy, periodic channel, constant ν | 1D steady momentum equation ν u'' = 1/2 C_D a(z) u² - G, solved independently to high accuracy (semi-analytical): fast flow above the canopy, slow flow inside, shear layer at the top (Nepf 2012) | `test_validate_*` |
| R4 | Patch vs individual stems | rows of rigid stems with Delta_y = 1/(n dx_stem) give the same mean drag and velocity as the equivalent patch | `test_verify_*` |
| R5 | Single flexible stem, small deflection | cantilever under uniform load: tip deflection q l⁴/(8 EI) (beam theory) | `test_verify_*` |
| R6 | Flexible stem/patch, large deflection | effective length and drag against Ca from Luhar & Nepf (2011); drag growing as U^(4/3) at large Ca (Vogel exponent -2/3) | `test_validate_*` |
| R7 | Emergent patch in a channel with inflow/outflow | head loss across the patch from the momentum balance g H Delta_eta = 1/2 C_D a L U² H | `test_validate_*` |
| R8 | Wave damping by a rigid emergent canopy | wave-height decay H(x) = H0 / (1 + alpha x) of Dalrymple, Kirby & Hwang (1984) / Mendez & Losada (2004) | experiment (+ test if robust) |
| R9 | Lock exchange through an emergent canopy | drag-dominated front propagation of Tanino, Nepf & Tse (2005) | experiment |

Coverage of the issue items:
- rigid: R2, R3, R7, R8;
- flexible: R5, R6;
- emergent: R2, R7, R8, R9;
- submerged: R3, R6;
- patches: R1-R3, R6-R9;
- individual stems: R4-R6.

The existing qualitative tests are replaced by these, or kept only if they still add something; the dispatch bug is
fixed either way.

4 Experiments and documentation
-------------------------------

`experiments/vegetation/` is reorganised like the other experiments: `cases.jl`, plot scripts, a README with figures,
and `figures/`. The experiments:
1. emergent and submerged rigid canopies in a channel (R2, R3, R7), including the profile figure with the shear layer;
2. flexible canopy reconfiguration (R6);
3. wave damping (R8);
4. lock exchange through a canopy (R9).

The old run scripts are replaced. The new parameters and models are documented in
`test/TESTS_TECHNICAL_DOCUMENTATION.md`.

5 Out of scope (proposed for later)
-----------------------------------

- Dynamic flexible stems (mass, added mass, damping), needed for swaying under waves (Luhar & Nepf 2016). A stem
  oscillating with the waves needs this; the quasi-static model gives the posture only.
- Vegetation-generated turbulence (Nepf 1999) as a physical model instead of the current ad hoc viscosity.
- Porosity and blockage effects of dense canopies.
- Vegetation over a variable bed (currently rejected by `step_forward!`).

6 Questions for review
----------------------

- **Q1, phasing:** do it in two steps?
  - step 1: patches (rigid and flexible), the rebuilt stems, R1-R7, the channel experiments;
  - step 2: waves (R8) and the lock exchange (R9).
- **Q2, flexible stems:** quasi-static elastica (recommended), or a dynamic beam now?
- **Q3, API:** may the old stem parameters (`stiffness` as string tension, `damping`, `mass_per_length`,
  `source_strength`) and the old experiment scripts be replaced (breaking change), or should they stay for
  compatibility?
- **Q4, `body_force_x`:** may it be added to `ModelConfig` for the periodic channel cases?
- **Q5, vegetation turbulence:** keep the current ad hoc term, remove it, or replace it in this issue?
- **Q6, references:** do you have specific lab data or papers you want to compare with (e.g. a BAW case)?

7 Decisions (review of 2026-10-25) and resulting changes
--------------------------------------------------------

- **Q1:** two steps as proposed. Step 1: patches, rebuilt stems, R1-R7, channel experiments. Step 2: waves (R8) and the
  lock exchange (R9).
- **Q2:** quasi-static elastica for flexible stems.
- **Q3:** old parameters and experiments may be replaced once their results are reproduced or improved. The old run
  scripts are therefore ported to the new model, not dropped.
- **Q4:** `body_force_x` may be added.
- **Q5:** left to the implementation. Proposal: the ad hoc vegetation turbulence (`vegetation_turbulence_strength`)
  is removed in step 1, because it has no reference. Turbulence comes from the existing models (constant nu or
  mixing length). A physical vegetation turbulence model (Nepf 1999) is a candidate for step 2.
- **Q6:** no own data; search for published data, see section 8.

Changes to section 2 that follow from these decisions and from checking the references:

- **Flexible patches use the elastica too**, instead of the Luhar & Nepf (2011) effective-length fit. A representative
  stem per column is bent by the local velocity profile, and the drag on its deflected shape, times the stem density,
  is the force on the fluid. This makes individual stems and patches consistent and keeps unverified fit formulas out
  of the model. The theory of reconfiguration then serves as an independent check (R5, R6): small deflection against
  the cantilever formula; large deflection against R = l_e/l proportional to Ca^(-1/3), i.e. drag proportional to
  U^(4/3) (Alben et al. 2002; Gosselin et al. 2010; Luhar & Nepf 2011).
- **Explicit drag instead of implicit:** implicit drag applied to the predictor before the free-surface solve would
  make the steady state depend on dt (the surface-slope correction is not divided by the same factor). A consistent
  implicit treatment would have to enter the barotropic system (Casulli & Cheng 1992). Explicit drag is exact in the
  steady state. It is stable for dt 1/2 C_D a |u| <= 1, e.g. dt <= 0.4 s for a = 10 1/m, C_D = 1, |u| = 0.5 m/s. The
  model checks this criterion.
- **Fluid forces:** the vertical component of the normal drag on bent stems is not applied to w (it has no effect in
  hydrostatic mode and is small for the reconfigured stems; this is documented).

8 Published data and test setups (search of 2026-10-25)
--------------------------------------------------------

- **Dunn, Lopez & Garcia (1996)**, University of Illinois Hydraulic Engineering Series 51 (handle 2142/12229): flume
  runs with rigid (runs 1-12) and flexible (13-16) submerged cylinders, mean-velocity and turbulence profiles. This is
  the standard validation set for submerged canopies (used e.g. by Lopez & Garcia 2001 and Huthoff et al. 2007). The
  repository blocks automated downloads.
- **Huthoff, Augustijn & Hulscher (2007)**, Water Resour. Res. 43, W06413, doi:10.1029/2006WR005625: analytical
  depth-averaged velocity for submerged rigid cylinders, validated with compiled flume data (University of Twente
  repository; also blocked for automated downloads).
- **Baptist et al. (2007)**, J. Hydraul. Res. 45(4), 435-450: resistance formula for submerged vegetation fitted to
  1DV k-epsilon results and checked against flume experiments; widely used (Delft3D trachytopes).
- **Dalrymple et al. (1984)** wave damping: confirmed through the SWAN technical documentation (cylinder approach,
  dissipation proportional to C_D b_v N_v and to the sinh terms of the vegetation height).

Data comparison: if the PDFs of Dunn et al. (1996) and Huthoff et al. (2007) are put into `docs/` (gitignored), the run
tables can be digitised and the submerged-canopy experiment compared run by run (bulk velocity; profiles where the
turbulence model allows).

References
----------

- Dalrymple, Kirby & Hwang (1984), J. Waterway, Port, Coastal, Ocean Eng. 110(1), 67-79
- Mendez & Losada (2004), Coastal Eng. 51, 103-118
- Nepf (1999), Water Resour. Res. 35(2), 479-489
- Nepf (2012), Annu. Rev. Fluid Mech. 44, 123-142
- Luhar & Nepf (2011), Limnol. Oceanogr. 56(6), 2003-2017
- Luhar & Nepf (2016), J. Fluids Struct. 61, 20-41
- Tanino, Nepf & Tse (2005), Water Resour. Res. 41, W12402
- Timoshenko & Gere, Mechanics of Materials (cantilever deflection)
- Alben, Shelley & Zhang (2002), Nature 420, 479-481 (drag reduction of flexible fibres)
- Gosselin, de Langre & Machado-Almeida (2010), J. Fluid Mech. 650, 319-341 (reconfiguration regimes)
- Gosselin (2019), Mechanics of a plant in fluid flow, arXiv:1905.08055 (review)
- Huthoff, Augustijn & Hulscher (2007), Water Resour. Res. 43, W06413
- Baptist et al. (2007), J. Hydraul. Res. 45(4), 435-450
- Dunn, Lopez & Garcia (1996), Hydraulic Engineering Series 51, University of Illinois
