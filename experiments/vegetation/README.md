Vegetation
==========

Flow through and over aquatic vegetation: rigid and flexible stems, emergent and submerged canopies, described either as
patches with averaged quantities or as individual stems (issue #8; plan and decisions in
`docs/plan_issue_8_vegetation.md`). This folder holds step 1 of the plan: the vegetation model with its references in
steady channel flow. Waves and gravity currents through canopies with quantitative references follow in step 2.

Model
-----

- **Patch** (`VegetationPatch`): stem length h_v, diameter b, stem density n (stems/m²) and drag coefficient C_D over
  an x-range. The frontal area per unit volume is a = n b. A representative stem per u-face column carries the drag of
  its column, and the force per unit volume is the stem force times n. A rigid patch removes 1/2 C_D a u|u| per unit
  volume below the stem height.
- **Individual stem** (`VegetationStem`): base position, length, diameter, C_D. In the 2DV model (per metre width) a
  stem stands for a row of stems with lateral spacing Δy (`lateral_spacing`).
- **Rigid or flexible**: the bending stiffness EI is `Inf` (rigid) or finite. Flexible stems take the quasi-static
  posture of an inextensible beam (elastica) under the normal drag 1/2 ρ C_D b (u cos θ)² and their net buoyancy. This
  is the model behind the reconfiguration studies of Alben et al. (2002), Gosselin et al. (2010) and Luhar & Nepf
  (2011). The posture is solved with Newton's method every time step.
- **Momentum:** the drag is split exactly at the cell boundaries; the momentum removed from the fluid equals the stem
  drag (Newton's third law).
- **Time stepping:** explicit drag; `step_forward!` stops if dt times the linearised drag rate exceeds 1.
- **Not included (yet):** stem dynamics (mass, swaying under waves), vegetation-generated turbulence, porosity of dense
  canopies, the vertical drag component on w, and vegetation over a variable bed.

The old stem model was replaced. It passed only 0.35 times the stem force to the fluid, had no stem density, and used
tuning factors (a gain of 8, a per-stem `source_strength`) and a string-like bending term. In the configurations of the
old tests its stems changed the wave amplitude by about 10⁻¹⁵ and the lock-exchange front position by less than one
cell. The uniform-flow scripts of the old model are replaced by the channel and patch experiments below. The
oscillating-flow, wave-damping and lock-exchange scripts are ported to the new stem parameters (EI, lateral spacing);
their quantitative references follow in step 2.

Reference cases
---------------

**Single flexible stem in a uniform current** (`plot_vegetation_reconfiguration.jl`). The reconfiguration number
R = F / F_rigid (the effective length l_e/l of Luhar & Nepf 2011) depends only on the Cauchy number
Ca = ρ C_D b U² l³ / (2 EI) and the buoyancy number B = B_l l³ / EI.

![Reconfiguration number against Cauchy number, and stem shapes](figures/vegetation_reconfiguration.png)

- **Rigid limit:** at small Ca the stem stays upright, R → 1 (R = 0.9997 at Ca = 0.1).
- **Small deflection:** against the cantilever under uniform load, tip deflection q l⁴/(8EI), the solver converges at
  exactly second order (relative errors 1.0·10⁻², 2.5·10⁻³ and 6.3·10⁻⁴ for 10, 20 and 40 segments).
- **Large deflection:** R ∝ Ca^(−1/3), i.e. drag ∝ U^(4/3) (Vogel exponent −2/3; Alben et al. 2002; Gosselin et al.
  2010). The slope between Ca = 300 and 3000 is −0.333, and R Ca^(1/3) = 0.89. This is consistent with the asymptotic
  effective length of about 0.9 Ca^(−1/3) in Luhar & Nepf (2011); the coefficient is still to be checked against the
  paper.
- **Buoyancy:** it keeps the stem upright until Ca exceeds roughly B and leaves the large-Ca asymptote unchanged, as
  described by Luhar & Nepf (2011).

**Canopies in a periodic channel** (`plot_vegetation_channel.jl`): the flow is driven by a body force G (a channel slope,
`body_force_x`), with constant viscosity ν = 10⁻³ m²/s and a = 4 1/m.

![Velocity profiles through emergent, submerged and flexible canopies](figures/vegetation_channel_profiles.png)

- **Emergent rigid canopy:** with a free-slip bed the velocity is uniform and 1/2 C_D a U² = G exactly. The model
  reproduces it to round-off in both modes.
- **Submerged rigid canopy** (h_v = H/2): compared with the steady 1D equation ν u″ = 1/2 C_D a u|u| − G, solved
  independently on 4000 cells. The flow is slow in the canopy and fast above it, with a shear layer at the canopy top
  (Nepf 2012). The model converges at second order to the reference, in both modes, which give identical results:

  ![Grid convergence of the submerged canopy](figures/vegetation_channel_convergence.png)

  The errors are 1.35·10⁻³, 3.51·10⁻⁴, 8.87·10⁻⁵ and 2.22·10⁻⁵ m/s for nz = 10, 20, 40 and 80.
- **Submerged flexible canopy** (EI = 2·10⁻⁵ N m², Ca about 170 at the canopy-top velocity): the stems bend to
  0.28 m of their 0.50 m length. The canopy offers less drag, so the depth-mean velocity is 0.067 m/s instead of
  0.027 m/s for the rigid canopy. In the steady state the total drag of a flexible patch balances the driving force
  (test `test_verify_flexible_patch_momentum_balance`: to 10⁻⁶ after 1400 s).
- **Individual stems vs. patch:** 25 rigid stems at random positions, with the lateral spacing of the patch density,
  give the patch velocity to 10⁻⁶ (test `test_verify_individual_stems_match_patch`).

**Patch in a channel with inflow and outflow** (`plot_vegetation_patch.jl`):

![Water level along the channel with an emergent patch](figures/vegetation_patch_head_loss.png)

The water level drops across an emergent rigid patch (a = 1 1/m, length 4 m, U = 0.1 m/s) by the head loss of the
momentum balance, g Δη = 1/2 C_D a L U². The model gives 2.038·10⁻³ m against 2.039·10⁻³ m (−0.03 %; the depth change
across the patch is neglected in the balance), the same in both modes.

![Submerged flexible patch: velocity and stem shapes](figures/vegetation_patch_flexible.gif)

A submerged flexible patch (stems of 0.4 m, b = 5 mm, 1000 stems/m², EI = 10⁻⁴ N m²) in a 0.2 m/s channel flow is
run non-hydrostatic with mixing-length turbulence. The figure shows u, with every tenth representative stem in white.
The stems bend to about half their length, the flow inside the canopy slows down, and a fast layer skims over it.

![Velocity field and profiles at 60 s](figures/vegetation_patch_flexible.png)

The profiles show the typical inflection near the canopy top. This case is qualitative; it has no reference solution.
After 60 s the flow is still adjusting, and near the outlet the prescribed uniform outflow distorts the flow.

Tests
-----

`test/test_vegetation.jl` checks:
- the patch drag per unit volume, including the canopy-top cell and the patch edges;
- Newton's third law for individual stems;
- the cantilever (order 2) and the reconfiguration law;
- the emergent canopy (exact);
- the submerged canopy against the 1D reference (order 2);
- individual stems against the patch;
- the momentum balance of a flexible patch;
- the head loss;
- the explicit-drag stability check;
- damping of a standing wave and delay of a lock-exchange front by a patch.

Details: section 16 of `test/TESTS_TECHNICAL_DOCUMENTATION.md`.

Published data
--------------

The standard laboratory data sets for submerged rigid and flexible canopies are not available here in machine-readable
form. These are Dunn, López & García (1996), University of Illinois Hydraulic Engineering Series 51, and the
compilation used by Huthoff, Augustijn & Hulscher (2007). Their repositories block automated downloads. With the PDFs
in `docs/` (gitignored), the run tables can be digitised and compared in step 2 (bulk velocity; profiles as far as the
turbulence model allows).

Scripts
-------

- `plot_vegetation_reconfiguration.jl`: single flexible stem, R(Ca); a few seconds.
- `plot_vegetation_channel.jl`: periodic channel references and grid convergence; about 3 min.
- `plot_vegetation_patch.jl`: head loss and the flexible-patch animation; about 1 min.
- `cases.jl`: shared settings, the 1D canopy reference and the reconfiguration helper (also used by the tests).
- `run_oscillating_flow_with_flexible_vegetation.jl`, `run_wave_damping_with_flexible_vegetation.jl`,
  `run_lock_exchange_with_flexible_vegetation.jl`: ported to the new model, references in step 2.

Output goes to `output_barotropic/vegetation_figures/`; the figures here are copies, the GIF reduced to every second
frame and 640 px width.

    julia --project=. experiments/vegetation/plot_vegetation_reconfiguration.jl
    julia --project=. experiments/vegetation/plot_vegetation_channel.jl
    julia --project=. experiments/vegetation/plot_vegetation_patch.jl

References
----------

- Alben, Shelley & Zhang (2002), Nature 420, 479-481
- Gosselin, de Langre & Machado-Almeida (2010), J. Fluid Mech. 650, 319-341
- Gosselin (2019), Mechanics of a plant in fluid flow, arXiv:1905.08055
- Luhar & Nepf (2011), Limnol. Oceanogr. 56(6), 2003-2017
- Nepf (2012), Annu. Rev. Fluid Mech. 44, 123-142
- Dunn, López & García (1996), Hydraulic Engineering Series 51, University of Illinois
- Huthoff, Augustijn & Hulscher (2007), Water Resour. Res. 43, W06413
- Timoshenko & Gere, Mechanics of Materials (cantilever deflection)
