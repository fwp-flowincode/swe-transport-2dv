Sloped channel with Chezy friction (issue #14)
==============================================

Uniform and gradually varied flow in the channel of issue #14:

| Quantity | Value |
|---|---|
| length L | 1000 m |
| depth H | 4 m |
| bed slope i_b | 1e-4 (bed falls in +x) |
| discharge q | 4 m²/s |
| Chezy coefficient C | 50 m^(1/2)/s |
| vertical viscosity | 0.02 m²/s (constant, of the order kappa u_* H / 6 = 0.017 m²/s) |

The normal flow is U = C sqrt(H i_b) = 1 m/s, so q = U H = 4 m²/s: the data of the issue are consistent (Fr = 0.16).
Theory and implementation: [docs/TESTS_TECHNICAL_DOCUMENTATION.md](../../docs/TESTS_TECHNICAL_DOCUMENTATION.md),
section 21; tests in `test/test_sloped_channel.jl`.

Model features used:
- `ModelConfig(chezy_coefficient=C)`: bed stress g |U| U / C² on the depth-averaged velocity, on the lowest layer.
- `set_x_inflow_outflow!(model, shape; left_discharge=q, right_level=eta_R)`: discharge at the inlet, water level at the
  outlet. The water level needs the hydrostatic mode with `implicit_vertical_terms=true`.
- The slope enters as the body force g i_b (`body_force_x`) on a flat bed, or (case 4) as a real bed in the
  wetting-drying step.

Run: `julia --project experiments/sloped_channel/run_cases.jl` (about 30 s).

Results
-------

**1. Spin-up of the periodic channel from rest.** Depth-averaged velocity at t = 1000 s against U_eq tanh(t/T),
T = 1019 s (0.753498 m/s). The explicit friction is first order in time; the implicit one, c_f |U^n| U^(n+1), is second
order (the product is the geometric mean of the two time levels).

| friction | dt (s) | error (m/s) | order |
|---|---|---|---|
| explicit | 2 | 3.56e-04 |  |
| explicit | 1 | 1.78e-04 | 1.00 |
| explicit | 0.5 | 8.89e-05 | 1.00 |
| implicit | 20 | 5.44e-05 |  |
| implicit | 10 | 1.36e-05 | 2.00 |
| implicit | 5 | 3.40e-06 | 2.00 |

**2. Open channel**, q at the inlet and eta = 0 at the outlet, started at 0.8 times the normal flow (nx = 50, nz = 10,
dt = 10 s). Maximum deviation from the normal flow (m/s):

| grid | t = 5000 s | 10000 s | 15000 s | 20000 s |
|---|---|---|---|---|
| variable top layer | 2.0e-06 | 3.7e-11 | 1.6e-15 | 1.6e-15 |
| wetting-drying step | 2.0e-06 | 3.5e-11 | 5.0e-13 | 6.0e-13 |

**3. Backwater (M1) and drawdown (M2) curves.** Outlet level 0.3 m above or below the normal depth, nz = 8. Steady
depth against the gradually-varied-flow equation (`chezy_backwater_depth`, RK4). The differences of 1e-5 to 5e-5 m
(0.003-0.02 % of the level change) include the momentum-flux coefficient of the sheared velocity profile, which the
reference (depth-uniform velocity) neglects.

| outlet level eta_R (m) | grid | nx | h(0) model (m) | h(0) reference (m) | max abs(h - h_ref) (m) |
|---|---|---|---|---|---|
| +0.3 | variable top layer | 25 | 4.28105 | 4.28102 | 2.7e-05 |
| +0.3 | variable top layer | 50 | 4.28086 | 4.28084 | 2.9e-05 |
| +0.3 | variable top layer | 100 | 4.28076 | 4.28074 | 3.0e-05 |
| +0.3 | wetting-drying | 25 | 4.28108 | 4.28102 | 5.1e-05 |
| +0.3 | wetting-drying | 50 | 4.28087 | 4.28084 | 3.8e-05 |
| +0.3 | wetting-drying | 100 | 4.28077 | 4.28074 | 3.4e-05 |
| -0.3 | variable top layer | 25 | 3.72534 | 3.72535 | 1.2e-05 |
| -0.3 | variable top layer | 50 | 3.72559 | 3.72559 | 1.0e-05 |
| -0.3 | variable top layer | 100 | 3.72572 | 3.72572 | 1.0e-05 |
| -0.3 | wetting-drying | 25 | 3.72527 | 3.72535 | 7.5e-05 |
| -0.3 | wetting-drying | 50 | 3.72556 | 3.72559 | 3.3e-05 |
| -0.3 | wetting-drying | 100 | 3.72570 | 3.72572 | 2.2e-05 |

Before this issue the variable top layer was 3.3e-4 m off (5.9e-4 m with 16 layers). The cause was the vertical part
of the explicit top-row correction, which used dz where the implicit column uses the top-layer thickness
(docs section 21).

**4. Real bed slope**, wetting-drying cut cells, no body force. The bed falls by i_b L = 0.1 m, a quarter of a layer;
the inlet shape is the steady parabola; the outlet has the normal depth. Deviation from the normal flow:

| nx | max abs(depth - H) (m) | U(L/2) - U_eq (m/s) | q(L/2) (m²/s) |
|---|---|---|---|
| 25 | 3.4e-05 | -5.3e-06 | 4.000000 |
| 50 | 1.3e-05 | -2.2e-06 | 4.000000 |
| 100 | 6.3e-06 | -7.3e-07 | 4.000000 |

**5. Log-law friction with the roughness length of the Chezy coefficient.** Here
z0 = `chezy_roughness_length(50, 4)` = 2.1 mm, with k-epsilon (periodic, 20000 s). The converted roughness gives a flow
4-5 % faster than the Chezy equilibrium. The conversion assumes a logarithmic velocity profile over the whole depth,
which the k-epsilon profile follows only approximately; within a few percent it is the equivalent roughness.

| nz | U (m/s) | U / U_eq - 1 |
|---|---|---|
| 10 | 1.0427 | +0.043 |
| 20 | 1.0478 | +0.048 |
| 40 | 1.0503 | +0.050 |

Limits
------

- The variable top layer is accurate for abs(eta) up to about dz. A level step of 0.5 m at the outlet of the normal
  flow (nz = 10, dz = 0.4 m) sends a bore that reflects at the discharge inlet and reaches 0.8 m, and the run fails.
  For such cases use the fixed grid or the wetting-drying step. The backwater cases above start from the reference
  profile.
- The water-level boundary is hydrostatic only and needs `implicit_vertical_terms=true`.
