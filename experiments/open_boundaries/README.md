Open boundaries (issues #46, #49 and #54)
=========================================

A wave runs against the right boundary of a channel: free outflow (`right_free`, radiation condition), prescribed
level (`right_level`), wall, or a free outflow relaxed to a target level
(`right_relaxation_time`). Hydrostatic long wave and non-hydrostatic dispersive wave, fixed grid and wetting-drying
step; uniform channel flow through a discharge inlet and a level outlet in the non-hydrostatic mode. Theory and
implementation: [docs/TESTS_TECHNICAL_DOCUMENTATION.md](../../docs/TESTS_TECHNICAL_DOCUMENTATION.md), sections 22, 24
and 25; the set-ups are those of `test/test_open_boundaries.jl`.

Run: `julia --project experiments/open_boundaries/plot_open_boundaries.jl [product ...]` (products: `wave pressure
reflection xt uniform relaxed`, default all; about 3 min for all).

Cases
-----

- **Long wave** (hydrostatic): Gaussian of width 50 m and amplitude 1 cm in 4 m depth, right-going, 1000 m channel,
  dx = 10 m, dt = dx / sqrt(gH).
- **Dispersive wave** (non-hydrostatic): Gaussian hump of width 1 m (k H ~ 1) and amplitude 1 cm in 1 m depth, at rest
  at x = 30 m in a 40 m channel; dx = 0.2 m, 10 layers, dt = 0.02 s. It splits; the right half reaches the outlet after
  about 3 s. The reference is a closed channel of twice the length, so nothing comes back in the run time.
  On the fixed grid with the Dirichlet top (`nonhydro_surface_dirichlet=true`). With the default Neumann top, surface
  waves on the fixed grid do not disperse (docs section 14), and the hump would travel as a long wave.
- **Uniform flow**: the channel of issue #14 (H = 4 m, i_b = 1e-4, C = 50), non-hydrostatic, for 300 steps of 10 s.

theta = 0.7 in all wave runs; it damps the reflected waves somewhat.

Figures
-------

- `open_boundaries_wave.gif`: surface of the dispersive wave for the three boundaries and the reference, fixed grid
  (top) and wetting-drying step (bottom).
- `open_boundaries_pressure.gif`, `open_boundaries_pressure.png` (t = 3 s): non-hydrostatic pressure q near the free
  outflow. q is largest at the bed and zero at the surface, of opposite sign under crest and trough. The open face has
  dq/dx = 0.
- `open_boundaries_reflection.png`: top, a gauge at x = 36 m; bottom, max_x |eta - eta_ref| / A over time.
- `open_boundaries_xt.png`: x-t diagrams of the surface, long wave (top) and dispersive wave (bottom, fixed grid).
- `open_boundaries_uniform.png`: deviation of the uniform flow, fixed grid (Neumann and Dirichlet top) and
  wetting-drying step.
- `open_boundaries_relaxed.png`: relaxed free outflow (issue #54). Left: |R| from the spectral ratio of the reflected
  records (4 km channel, long wave, tau = 2 to 50 s) against 1/sqrt(1 + 4 omega^2 tau^2). Middle: return of a 1 cm
  level offset in a closed 1 km basin against the lowest mode; the free outflow alone keeps the offset. Right: gauge
  records for the prescribed level, tau = 5 and 20 s, and the free outflow.

Results
-------

| reflection, max_x \|eta - eta_ref\| / A | free outflow | prescribed level / wall |
|---|---|---|
| long wave, fixed grid and wetting-drying step | 0.7 % | complete (inverted / upright) |
| dispersive wave, fixed grid (Dirichlet top), largest / at t = 12 s | 1.3 % / 0.7 % | 18 % at t = 12 s (after the theta damping) |
| dispersive wave, wetting-drying step, largest / at t = 12 s | 1.2 % / 0.6 % | 18 % at t = 12 s |

- The free outflow lets both waves out. The remaining reflection of the dispersive wave comes from the radiation
  condition, which uses the long-wave speed sqrt(gH). The short components behind the crest are slower, and a
  component with phase speed c is reflected by about (sqrt(gH) - c) / (sqrt(gH) + c).
- A prescribed level reflects with inverted sign, a wall with the same sign. The two differ only in sign, so their
  curves in the deviation panel lie on top of each other.
- Both grids give nearly the same dispersive wave and pressure field.
- Uniform flow stays at round-off (wetting-drying step 1e-12, Dirichlet top 1e-15). With the Neumann top there is
  an offset of 6e-10 m in the level (sparse solver; 4e-13 m with the spectral one), which appears in the first steps and then stays constant. With that projection, q holds the
  x-independent hydrostatic part of the reference density (1.7e4 Pa). It exerts no force, also on the open face.

Design note (issue #49)
-----------------------

On the open face, the first version had q = 0 on the fixed grid. These figures showed that it had been checked only
with the Neumann top, where the wave is not dispersive. With the Dirichlet top, q = 0 reflected 4.4 % of the dispersive
wave, against 1.3 % with dq/dx = 0. Both grids now use dq/dx = 0; on the fixed grid this also lets the spectral
pressure solver be used.
