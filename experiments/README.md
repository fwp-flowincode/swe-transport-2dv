Experiment Layout
=================

This folder is organized by case type and includes dedicated validation scripts and
shared helpers.

Structure
---------
- benchmarks/: profiling and speed benchmarks.
- lock_exchange/: lock-exchange production runs.
- rayleigh_taylor/: RTI reference and fast-growth runs.
- plumes/: rising/sinking buoyant plume tests.
- jet/: impulse jet shear test.
- internal_seiche/: two-layer standing-wave test.
- surface_dam_break/: barotropic dam-break test.
- kelvin_helmholtz/: stratified shear-layer instability test.
- wind_shear/: wind-driven free-surface transport and mixing case.
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

Reference documentation
-----------------------
- ../literature/additional_validation_references.md

Backward compatibility
----------------------
Legacy top-level scripts are kept and now include the corresponding new script,
so previous commands still work.
