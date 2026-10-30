Experiment Layout
=================

This folder is organized by case type and includes dedicated validation scripts and
shared helpers.

Each case folder can hold a `README.md` (description, settings, results) and a `figures/`
subfolder with small images and animations for it (keep each file well below 1 MB). Full
model output goes to `output_barotropic/`, which is not version controlled.

Structure
---------
- benchmarks/: profiling and speed benchmarks.
- lock_exchange/: lock-exchange runs and density plots ([README](lock_exchange/README.md)).
- lock_exchange_obstacle/: density-stratified dam break with obstacle over a variable bed ([README](lock_exchange_obstacle/README.md)).
- rayleigh_taylor/: RTI reference and fast-growth runs, growth vs. linear theory ([README](rayleigh_taylor/README.md)).
- plumes/: rising/sinking buoyant plume tests.
- jet/: impulse jet shear test.
- internal_seiche/: two-layer standing-wave test.
- surface_dam_break/: barotropic dam-break test.
- kelvin_helmholtz/: stratified shear-layer instability, periodic in x ([README](kelvin_helmholtz/README.md)).
- wind_shear/: wind-driven double gyre in a closed two-layer basin ([README](wind_shear/README.md)).
- vegetation/: rigid and flexible canopies and stems ([README](vegetation/README.md)).
- theta_scheme/: stability and accuracy of the theta time integration ([README](theta_scheme/README.md)).
- advection_stability/: stability of the momentum advection with a mean flow, issue #37 ([README](advection_stability/README.md)).
- sloped_channel/: uniform and gradually varied flow with Chezy friction, discharge and water-level boundaries, issue #14 ([README](sloped_channel/README.md)).
- obstacles/: flow over a bump and a broad-crested weir, sub-, trans- and supercritical, hydraulic jump, tracer, issue #47 ([README](obstacles/README.md)).
- open_boundaries/: waves at the free outflow, a prescribed level and a wall, hydrostatic and non-hydrostatic (dispersive), issues #46 and #49 ([README](open_boundaries/README.md)).
- validation/: scripts that compare selected cases to analytic/semianalytic references.
- common/: helper utilities used by case scripts.

Validation scripts currently available
--------------------------------------
- validation/validate_surface_dam_break_front_speed.jl
- validation/validate_internal_seiche_period.jl
- validation/validate_lock_exchange_front_speed.jl
- validation/validate_wind_shear_transport_mixing.jl
- validation/audit_hydrostatic_continuity_free_surface.jl
- validation/smoke_all_new_cases.jl
