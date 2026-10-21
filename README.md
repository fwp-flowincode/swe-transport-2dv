SweTransport2DV
=================

2DV (vertical plane) shallow-water and transport model with hydrostatic and
non-hydrostatic modes, in Julia. It computes reference solutions for cases
without an analytical solution, e.g. the lock exchange (gravity current
after removal of a gate between two fluids of different density).

Julia 1:1 port of the current Python package in 2dv_swe_transport, including:
- State, diagnostics, physics, pressure, visualization, and model driver.
- Matching regression tests (ported from test_model.py).
- Structured experiment suites (lock exchange, RTI, plumes, jet, seiche, dam-break, KH).
- Backward-compatible legacy top-level experiment wrappers.

Folder Layout
-------------
- src/: Julia package source.
- test/runtests.jl: full translated test suite.
- experiments/: translated benchmark/profile/run scripts.
- analysis/lock_exchange/: Python tools: bulk reference values of the lock exchange and
  evaluation of model runs (see its README).

Install
-------
From this folder:

```pwsh
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

Run tests
---------

```pwsh
julia --project=. -e "using Pkg; Pkg.test()"
```

Technical test documentation:
- `test/TESTS_TECHNICAL_DOCUMENTATION.md`

Run translated experiments
--------------------------

```pwsh
julia --project=. experiments/benchmarks/bench_speedup_dx001_vs_dx002.jl
julia --project=. experiments/benchmarks/profile_step_dx002.jl
julia --project=. experiments/lock_exchange/run_dx001_dt005_anim.jl
julia --project=. experiments/lock_exchange/run_dx002_dt001_100s.jl
julia --project=. experiments/lock_exchange/run_dx002_dt001_100s_anim.jl
julia --project=. experiments/rayleigh_taylor/run_reference_dt0005_25s_anim.jl
julia --project=. experiments/rayleigh_taylor/run_fast_dt0004_25s_anim.jl
julia --project=. experiments/plumes/run_rising_plume_dt001_40s_anim.jl
julia --project=. experiments/plumes/run_sinking_plume_dt001_40s_anim.jl
julia --project=. experiments/jet/run_jet_impulse_dt0005_20s_anim.jl
julia --project=. experiments/internal_seiche/run_internal_seiche_dt005_400s_anim.jl
julia --project=. experiments/surface_dam_break/run_surface_dam_break_dt001_60s_anim.jl
julia --project=. experiments/kelvin_helmholtz/run_kelvin_helmholtz_dt001_30s_anim.jl
julia --project=. experiments/wind_shear/run_wind_shear_dt001_30s_anim.jl
julia --project=. experiments/vegetation/run_uniform_flow_with_flexible_vegetation.jl
julia --project=. experiments/vegetation/run_oscillating_flow_with_flexible_vegetation.jl
julia --project=. experiments/vegetation/run_wave_damping_with_flexible_vegetation.jl
julia --project=. experiments/vegetation/run_lock_exchange_with_flexible_vegetation.jl
julia --project=. experiments/validation/validate_surface_dam_break_front_speed.jl
julia --project=. experiments/validation/validate_internal_seiche_period.jl
julia --project=. experiments/validation/validate_lock_exchange_front_speed.jl
julia --project=. experiments/validation/validate_wind_shear_transport_mixing.jl
julia --project=. experiments/validation/audit_hydrostatic_continuity_free_surface.jl
julia --project=. experiments/validation/smoke_all_new_cases.jl
```

Free-surface treatment (variable top layer)
-------------------------------------------

By default, the hydrostatic mode (`use_hydrostatic=true`) gives the top layer the thickness dz + eta instead of a fixed dz: its
volume change enters the top-layer budgets and the free surface is a material boundary. This removes a zero-flux inconsistency at
a moving surface that otherwise acts as a spurious density source in the top row. The non-hydrostatic mode keeps the legacy fixed
(dz) grid for this bookkeeping; see the section below for what does and does not move there.

The surface must stay inside the top layer (`top_layer_min_fraction`; the run stops with a clear error otherwise — there is no
layer removal or wetting/drying). Set `ModelConfig(variable_top_layer=false)` explicitly for hydrostatic cases with a large
surface excursion relative to dz, such as the surface dam break or a large-amplitude forced wave; the affected experiment/
validation scripts already do this (see the `variable_top_layer=false` comments there). Force it on for a specific case with
`variable_top_layer=true` (rejected for `use_hydrostatic=false`). Details, the full per-step formulation and limitations: section
13 of `test/TESTS_TECHNICAL_DOCUMENTATION.md`.

Non-hydrostatic free surface
-----------------------------

The non-hydrostatic branch's free surface used to stay pinned at round-off regardless of real internal forcing (an incidental
rigid-lid coupling through the order momentum, the barotropic solve and the non-hydrostatic pressure projection ran in — not a
deliberate choice). This is fixed: eta now responds to genuine internal (baroclinic) forcing in non-hydrostatic mode, exactly as
it already did in hydrostatic mode, with no change to either mode's continuity/kinematic guarantees.

By default this does NOT make free-surface gravity waves dispersive: a pure surface wave (uniform density, so no baroclinic
forcing) still follows the shallow-water relation `T = 2L/sqrt(gH)`, not the full `omega^2 = g*k*tanh(k*H)` relation, because the
barotropic surface force is applied outside the elliptic (non-hydrostatic) solve rather than coupled into it as a Dirichlet
condition. Details: section 14 of `test/TESTS_TECHNICAL_DOCUMENTATION.md`.

Set `ModelConfig(nonhydro_surface_dirichlet=true, pressure_solver_method="sparse")` (non-hydrostatic mode only) to fix this: the
pressure correction's top condition becomes Dirichlet `p'=0` instead of Neumann, and the resulting surface waves are genuinely
dispersive (validated against `omega^2=g*k*tanh(k*H)`, converging under grid refinement). Requires the `sparse` pressure solver (the
spectral one cannot represent a Dirichlet condition) and is implemented for the fixed grid only (not combined with
`variable_top_layer`). It is noticeably more expensive at large grid sizes than the default spectral solver — on this machine, a
1500x200 grid took about 1.9 s to factorize once and about 0.095 s per step for the pressure solve alone, comparable to or exceeding
a full spectral-method step's total cost. Details, the physical derivation, and a second defect found and fixed while validating
this (a "trivial" reference-density pressure contribution that otherwise dominates the correction's magnitude): section 15 of
`test/TESTS_TECHNICAL_DOCUMENTATION.md`.

Archive legacy root outputs (safe cleanup)
------------------------------------------

To keep only per-case subfolders in output_barotropic and archive old root-level files
without deleting them:

```pwsh
julia --project=. experiments/maintenance/archive_legacy_root_outputs.jl
```

This moves root-level files into a timestamped folder:
- output_barotropic/_legacy_root_archive/<timestamp>/

Plotting backend note
---------------------

The plotting code is configured for safer batch/headless execution:
- Uses non-interactive GR workstation defaults when not explicitly overridden.
- Avoids NaN-delimited vector segments in overlays.
- Applies finite color limits and robust y-limits to reduce plotting-range warnings.

Performance notes
-----------------
The port was written with in-place kernels and reusable workspaces in computationally intensive paths:
- Advection and limiter operations in src/physics.jl.
- Step orchestration and major temporaries in src/model.jl.
- Pressure correction solver buffers in src/pressure.jl.

This avoids repeated per-step allocations in hotspot functions and is intended to provide substantial speedup over the original pure NumPy path.

License
-------

MIT, see [LICENSE](LICENSE).
