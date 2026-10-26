# Validation case catalogue

This document proposes a validation programme for the physical and numerical
capabilities represented in SweTransport2DV. It is a roadmap, not a claim that
every proposed case already exists. The model is a 2DV (vertical x-z plane)
Boussinesq solver; “1D” below means a physically one-dimensional limit represented
by fields invariant in the other coordinate, not a separate 1D solver.

## Scope and categories

Use three complementary layers of evidence:

1. **Verification** checks that a discrete operator or implementation reproduces
   an exact solution, conservation law, or manufactured solution.
2. **Physics validation** compares resolved flow measures with analytical,
   semi-analytical, or published experimental results.
3. **Regression/smoke coverage** checks that representative configurations run,
   remain bounded, and preserve their documented invariants. A smoke run alone
   is not a physics validation.

The case families below span the current model: density transport and buoyancy,
hydrostatic and non-hydrostatic pressure, free-surface waves and bores, wind and
viscous mixing, periodic shear, variable bed, and flexible vegetation.

## Proposed 1D validation cases

These cases isolate one-dimensional limits of the 2DV equations and numerical
methods. They are useful both as small fast tests and as references for the
coupled 2DV cases.

| ID / family | Reference | Test description | Expected outcome |
|---|---|---|---|
| 1D-01 Scalar advection | Exact translation of a smooth profile; LeVeque (2002), ch. 6 | Advect a smooth density profile with uniform horizontal velocity, then with uniform vertical velocity. Compare `upwind1` and `tvd2` at fixed CFL under grid refinement. | The profile translates at the imposed velocity; mass is conserved. `tvd2` converges at approximately second order for smooth data, while `upwind1` is first order. Neither scheme creates new extrema under its documented CFL restriction. |
| 1D-02 Hydrostatic column | Hydrostatic balance, \(dp/dz=-\rho g\) | Initialize a horizontally uniform, possibly stratified column at rest and compare pressure to the vertical integral of density. Advance for many steps. | Pressure matches the discrete hydrostatic integral and the column remains at rest to round-off; no spurious buoyancy-driven velocity develops. |
| 1D-03 Vertical viscous adjustment | Exact steady wind profile for constant viscosity; Gill (1982) | Apply constant surface stress to a uniform-density closed basin, and compare the equilibrated vertical velocity profile with the steady parabolic solution. Repeat on successively finer vertical grids. | Surface stress and zero-stress bed conditions are satisfied; profile error decreases at second order with vertical spacing. |
| 1D-04 Long surface-wave limit | Stoker (1957); Whitham (1974) | Initialize a small-amplitude standing surface wave with uniform density and measure its period. Run the hydrostatic/default non-hydrostatic configuration and the opt-in non-hydrostatic Dirichlet-top configuration separately. | The shallow-water branch approaches \(T=2L/\sqrt{gH}\). With `nonhydro_surface_dirichlet=true` and the sparse solver, the period converges toward \(\omega^2=gk\tanh(kH)\). The default non-hydrostatic mode is expected to retain its documented shallow-water limitation. |
| 1D-05 Surface dam-break front | Stoker (1957); Whitham (1974) | Use a uniform-density initial surface step; track the leading bore/rarefaction before wall reflections and compare its speed with the shallow-water reference \(c=\sqrt{gH}\). | The front propagates at the expected gravity-wave scale and approaches the reference as resolution is refined; mass/volume budgets close. |

Existing coverage includes scalar-advection convergence and conservation
(`test/test_advection.jl`), hydrostatic pressure and wind profiles
(`test/test_pressure.jl`, `test/test_model.jl`), standing-wave period and
refinement (`test/test_variable_top_layer.jl`), and a surface dam-break front
script (`experiments/validation/validate_surface_dam_break_front_speed.jl`).
The case-level validation scripts also cover lock-exchange front speed,
internal-seiche period, and wind-driven transport/mixing; see
`experiments/validation/`. Rayleigh–Taylor growth, periodic-boundary behavior,
bed geometry, and vegetation have test or experiment coverage, while the
catalogue above proposes more systematic reference-based and grid-convergence
criteria for those families.

## Proposed 2DV validation cases

### Density-driven instability, currents, and plumes

| ID / family | Reference | Test description | Expected outcome |
|---|---|---|---|
| 2DV-01 Lock exchange / gravity current | Benjamin (1968); Shin et al. (2004) | Release two density classes from opposite halves of a closed basin. Measure left/right front positions and speeds over a stated fit interval; compare hydrostatic and non-hydrostatic runs and repeat at multiple resolutions. | The dense and light fronts move in opposite directions. For the documented front-speed definition and fit interval, compare the Froude coefficient with the literature-informed range (about 0.44–0.59); density remains bounded and constituent mass and water volume are conserved. Hydrostatic and non-hydrostatic results may differ where vertical accelerations matter. |
| 2DV-02 Rayleigh–Taylor instability | Rayleigh (1883); Taylor (1950); Chandrasekhar (1981) | Place a heavier layer above a lighter layer with a small sinusoidal interface perturbation. Measure the dominant-mode interface amplitude during the linear phase, then document overturning and mixing. | The unstable mode grows from the perturbation; its early growth rate approaches linear stability theory under refinement before nonlinear saturation. The stable heavy-below-light control remains quiescent. |
| 2DV-03 Kelvin–Helmholtz instability | Chandrasekhar (1981); Sharp (1984) | Initialize a periodically bounded, stratified shear layer with a small interface perturbation. Measure early growth, billow wavelength, and post-billow density variance. | The shear instability produces billows at the seeded/dominant wavelength; growth and mixing are resolved and converge as the grid is refined. Periodic seam fluxes and total constituent mass match the interior behavior. |
| 2DV-04 Rising and sinking plumes | Morton, Taylor & Turner (1956); Turner (1973) | Release a localized light or dense Gaussian anomaly in an ambient fluid. Track plume centroid, vertical velocity, width, and integrated anomaly over time. Include a sign-reversed pair. | Light anomalies rise and dense anomalies sink; the two cases reverse consistently when the density anomaly is reversed. The integral density anomaly is conserved, and plume trajectories/widths become grid-converged over the resolved comparison interval. |
| 2DV-05 Gravity current over a bed obstacle | Koshizuka & Oka (1996); Ubbink (1997); Benjamin (1968) | Release a dense column into lighter water in a tank with a staircase obstacle. Compare snapshots, front positions, and (where practical) horizontal obstacle force with published reference data; include flat-bed and raised-flat-bed controls. Treat the original air-water dam-break results as a geometry/flow-pattern reference, not a direct quantitative validation of this Boussinesq density-current counterpart. | The current reaches and interacts with the obstacle, with overflow/reflection and interface billows. Wet-cell mass and volume are conserved, solid-face velocities remain zero, and grid-refined front/force measures are reproducible. Interpret later-time surface jets cautiously (see `experiments/lock_exchange_obstacle/README.md`). |

### Waves, free surface, and boundary forcing

| ID / family | Reference | Test description | Expected outcome |
|---|---|---|---|
| 2DV-06 Internal seiche | Gill (1982); Cushman-Roisin & Beckers (2011) | Displace a two-layer interface in a closed basin and measure the dominant internal-wave period. Use the two-layer long-wave phase speed \(c=\sqrt{g'h_1h_2/(h_1+h_2)}\) as the reference: \(T=2L/c\) for mode 1. The existing mode-2 initializer uses \(T=L/c\). | The measured period approaches the mode-specific long-wave estimate under refinement; interface and free-surface phase remain coherent, with mass and volume budgets closed. |
| 2DV-07 Wind-driven stratified basin | Price, Weller & Pinkel (1986); Gill (1982) | Apply a uniform surface wind stress to a closed two-layer basin. Compare the equilibrated velocity profile and interface slope against the steady two-layer solution; track circulation and density-variance change. | Downwind surface flow and the expected counter-rotating two-layer circulation develop. Profiles and interface slope approach the analytical solution; basin volume and constituent mass are conserved. |
| 2DV-08 Free-surface and continuity closure | Casulli & Cheng (1992); Casulli & Walters (2000) | In both pressure modes, test a closed basin with internal forcing and a separate wind-forced case. Check the free-surface volume budget, top kinematic condition, and discrete divergence, including the variable-top-layer option where supported. | \(\eta\) responds to genuine forcing; closed-wall water volume and the applicable kinematic/continuity identities close to the stated numerical tolerance. Test fixed- and variable-thickness top-layer formulations against their respective budget equations, not against each other’s identities. |

### Geometry, transport, and vegetation

| ID / family | Reference | Test description | Expected outcome |
|---|---|---|---|
| 2DV-09 Variable-bed hydrostatic balance and projection | Manufactured solutions; Neumann Poisson problem | Use flat, raised-flat, and stepped-bed domains. Compare a raised flat bed with an equivalent shallower flat domain, then solve a manufactured pressure-correction field on a masked domain. | Flat-bed behavior is unchanged; equivalent-depth cases agree to round-off; the masked pressure solution converges at second order. Resting stratification produces no spurious current and the non-hydrostatic projection leaves negligible divergence in wet cells. |
| 2DV-10 Flexible vegetation | Vogel (1984); Nepf (1999); Luhar & Nepf (2011) | Compare uniform and oscillating flows with rigid stems, flexible stems, and no stems. Measure stem deflection, velocity attenuation, and free-surface-wave damping; then add vegetation to a lock exchange. | Stem bending follows load direction and remains bounded. Vegetation extracts momentum and damps wave/current motion relative to the no-stem control; trends respond consistently to stiffness and flow amplitude. |
| 2DV-11 Transport/mixing and numerical controls | Sweby (1984); LeVeque (2002); model conservation laws | Run a divergence-free cellular flow with a smooth tracer, and a separate mixing case with molecular/turbulent diffusivity. Check constituent mass, bounds, variance decay, and sensitivity to scheme/CFL. | Closed-domain mass is conserved; density `tvd2` remains bounded at \(CFL_x+CFL_z\le 1/2\) and converges for smooth transport. Mixing reduces tracer variance without creating mass; changing the scheme or diffusivity produces a measurable, physically consistent change. |
| 2DV-12 Impulse jet / transient shear | Turner (1973); Kundu, Cohen & Dowling (2015); grid-refined reference run | Initialize a localized horizontal-velocity impulse in uniform-density water. Track the velocity field, kinetic energy, jet width, and circulation over a fixed pre-reflection interval; compare resolutions and the no-advection/diffusion controls where useful. | The initial symmetry and expected shear-layer development are reproduced, and the measured fields converge under refinement. The uniform density remains unchanged; report energy changes alongside viscosity and damping rather than treating the impulsive flow as an energy-conserving case. |

## Suggested repository and test structure

Keep short deterministic checks in `test/` and case-level runs/plots under
`experiments/validation/`. For a family that grows beyond one script, use a
subdirectory rather than accumulating unrelated setup in a single runner:

```text
test/
  test_<numerical_area>.jl          # fast verification/regression assertions
experiments/validation/
  <family>/
    cases.jl                        # shared parameters and initial conditions
    validate_<case>.jl              # physics metrics and reference comparison
    README.md                       # setup, citations, fit interval, tolerances
    references/                     # only small, attributable reference data
```

Each validation case should state: (1) physical question and modeled
assumptions; (2) source/reference and the exact quantity compared; (3) initial
and boundary conditions, units, density contrast, and dimensional scales;
(4) grid, time step/CFL, schemes, pressure mode/solver, and stabilizers; (5)
measurement window and exclusions such as wall reflections; (6) pass criterion,
including tolerance or grid-convergence requirement; and (7) command and output
location. Keep plots diagnostic: the pass/fail decision should be based on
reported numerical metrics, not visual inspection alone.

Use `test_verify_*` for implementation/discrete-equation checks and
`test_validate_*` for comparisons with physical references, consistent with the
existing suite. Start with small automated runs and quantitative metrics; keep
expensive/high-resolution or animation runs as reproducible experiment scripts.
Record known limitations alongside results. In particular, the Boussinesq
approximation is intended for small relative density differences; the model does
not support wetting/drying; variable-top-layer and non-hydrostatic
Dirichlet-surface options have mode/solver restrictions; and late-time
non-hydrostatic lock-exchange runs can develop the documented top-cell surface
jet.

## Reference list

- Benjamin, T. B. (1968). Gravity currents and related phenomena. *Journal of Fluid Mechanics*, 31, 209–248.
- Casulli, V. & Cheng, R. T. (1992). Semi-implicit finite difference methods for three-dimensional shallow water flow. *International Journal for Numerical Methods in Fluids*, 15, 629–648.
- Casulli, V. & Walters, R. A. (2000). An unstructured grid, three-dimensional model based on the shallow water equations. *International Journal for Numerical Methods in Fluids*, 32, 331–348.
- Chandrasekhar, S. (1981). *Hydrodynamic and Hydromagnetic Stability*. Dover.
- Cushman-Roisin, B. & Beckers, J.-M. (2011). *Introduction to Geophysical Fluid Dynamics: Physical and Numerical Aspects*. Academic Press.
- Gill, A. E. (1982). *Atmosphere-Ocean Dynamics*. Academic Press.
- Koshizuka, S. & Oka, Y. (1996). Moving-particle semi-implicit method for fragmentation of incompressible fluid. *Nuclear Science and Engineering*, 123, 421–434.
- Kundu, P. K., Cohen, I. M. & Dowling, D. R. (2015). *Fluid Mechanics* (6th ed.). Academic Press.
- LeVeque, R. J. (2002). *Finite Volume Methods for Hyperbolic Problems*. Cambridge University Press.
- Luhar, M. & Nepf, H. M. (2011). Flow-induced reconfiguration of buoyant and flexible aquatic vegetation. *Limnology and Oceanography*, 56(6), 2003–2017.
- Morton, B. R., Taylor, G. I. & Turner, J. S. (1956). Turbulent gravitational convection from maintained and instantaneous sources. *Proceedings of the Royal Society A*, 234, 1–23.
- Nepf, H. M. (1999). Drag, turbulence, and diffusion in flow through emergent vegetation. *Water Resources Research*, 35(2), 479–489.
- Price, J. F., Weller, R. A. & Pinkel, R. (1986). Diurnal cycling: Observations and models of the upper ocean response to diurnal heating, cooling, and wind mixing. *Journal of Geophysical Research: Oceans*, 91(C7), 8411–8427.
- Rayleigh, Lord (1883). Investigation of the character of the equilibrium of an incompressible heavy fluid of variable density. *Proceedings of the London Mathematical Society*, 14, 170–177.
- Sharp, D. H. (1984). An overview of Rayleigh-Taylor instability. *Physica D: Nonlinear Phenomena*, 12(1–3), 3–18.
- Stoker, J. J. (1957). *Water Waves: The Mathematical Theory with Applications*. Interscience.
- Sweby, P. K. (1984). High-resolution schemes using flux limiters for hyperbolic conservation laws. *SIAM Journal on Numerical Analysis*, 21(5), 995–1011.
- Taylor, G. I. (1950). The instability of liquid surfaces when accelerated in a direction perpendicular to their planes. *Proceedings of the Royal Society A*, 201, 192–196.
- Turner, J. S. (1973). *Buoyancy Effects in Fluids*. Cambridge University Press.
- Ubbink, O. (1997). *Numerical Prediction of Two Fluid Systems with Sharp Interfaces*. PhD thesis, Imperial College London.
- Vogel, S. (1984). Drag and reconfiguration of broad leaves in high winds. *Journal of Experimental Botany*, 35(11), 941–948.
- Whitham, G. B. (1974). *Linear and Nonlinear Waves*. Wiley.
