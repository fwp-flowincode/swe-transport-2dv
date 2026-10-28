# 2DV SWE Test Technical Documentation

This document provides technical background, governing equations, and literature context for all test blocks of the test suite.
`test/runtests.jl` includes one file per functional area (this document lives in `docs/`). Shared packages and tolerances live in `test/test_common.jl`, which
every file includes, so each file also runs on its own (`julia --project=. test/test_advection.jl`):

| File | Area |
|---|---|
| `test_operators_and_diagnostics.jl` | state layout, grad/div operators, viscous diffusion (section 5), buoyancy, diagnostics |
| `test_advection.jl` | density (transport) and momentum advection, `upwind1` and `tvd2` (sections 3, 4) |
| `test_pressure.jl` | hydrostatic integration, Poisson matrix, non-hydrostatic pressure solvers (sections 6, 7) |
| `test_free_surface.jl` | barotropic solve, kinematic surface condition, continuity closure, wind forcing budgets (sections 8-10) |
| `test_model.jl` | full-model runs: conservation, stabilisation, configuration, initial conditions, dense front, wind-driven double gyre |
| `test_vegetation.jl` | vegetation: rigid and flexible patches and individual stems (section 16) |
| `test_turbulence.jl` | k-epsilon closure, bed wall functions, turbulent density diffusion (section 17) |
| `test_nonhydrostatic.jl` | non-hydrostatic branch and free surface (sections 14, 15) |
| `test_variable_top_layer.jl` | variable-thickness top layer (section 13) |
| `test_bed_geometry.jl` | variable bed (staircase, issue #3) |
| `test_wetting_drying.jl` | wetting and drying, reference solutions (section 18) |

## Governing model

The model advances incompressible 2DV flow with a free surface and active density transport:

```math
\frac{\partial u}{\partial t} + \mathcal{A}_u = -\frac{1}{\rho_0}\frac{\partial p}{\partial x},
```

```math
\frac{\partial w}{\partial t} + \mathcal{A}_w = -\frac{1}{\rho_0}\frac{\partial p}{\partial z} + b,
```

```math
\frac{\partial u}{\partial x} + \frac{\partial w}{\partial z} = 0,
```

```math
\frac{\partial \eta}{\partial t} + \frac{\partial}{\partial x}\int_{-H}^{\eta} u\,dz = 0,
```

```math
\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{u}) = 0.
```

Hydrostatic and non-hydrostatic branches split pressure as:

```math
p = p_h + p', \qquad \frac{\partial p_h}{\partial z} = -\rho g.
```

## Core references

- Stoker (1957), *Water Waves*.
- Whitham (1974), *Linear and Nonlinear Waves*.
- Benjamin (1968), JFM lock-exchange/gravity currents.
- Gill (1982), *Atmosphere-Ocean Dynamics*.
- Cushman-Roisin and Beckers (2011), *Introduction to Geophysical Fluid Dynamics*.
- Rayleigh (1883), Taylor (1950), and Sharp (1984) for instability references.
- Chandrasekhar (1981), *Hydrodynamic and Hydromagnetic Stability*.

## Implemented numerical methods

This section documents the numerical methods currently implemented in `src/physics.jl`, `src/pressure.jl`, and `src/model.jl`.

### 1) Staggered-grid spatial discretization

State variables are stored on a 2D staggered layout:

```math
u_{k,i+1/2},\quad w_{k+1/2,i},\quad p_{k,i},\ \rho_{k,i},\ \eta_i.
```

Discrete operators use centered first differences on their natural staggered locations:

```math
(\nabla_x p)_{k,i+1/2} = \frac{p_{k,i+1}-p_{k,i}}{\Delta x},
\qquad
(\nabla_z p)_{k+1/2,i} = \frac{p_{k+1,i}-p_{k,i}}{\Delta z},
```

```math
(\nabla\cdot\mathbf{u})_{k,i} = \frac{u_{k,i+1/2}-u_{k,i-1/2}}{\Delta x} + \frac{w_{k+1/2,i}-w_{k-1/2,i}}{\Delta z}.
```

### 2) Split time integration used in `step_forward!`

Per time step, the model uses a split predictor/free-surface/projection sequence (steps 4 and 5 were swapped from an
earlier ordering that had the free-surface solve AFTER the non-hydrostatic projection; see section 14):

1. Compute momentum advection tendencies for $u$ and $w$.
2. Build hydrostatic pressure predictor $p_h$.
3. Form provisional velocities $(u^{\ast},w^{\ast})$ from advection, pressure gradients, viscous diffusion (section 5),
   buoyancy and the wind stress (top row, section 10).
4. Enforce wall boundary values on $u^{\ast}$, advance free surface with depth-integrated transport using a barotropic solve
   (uses the un-projected $u^{\ast}$ in both modes), and correct $u^{\ast}$ with the free-surface slope.
5. Optionally apply non-hydrostatic projection to the (now surface-corrected) $(u^{\ast},w^{\ast})$ (when `use_hydrostatic=false`;
   section 14).
6. Enforce wall boundary values on $u$.
7. Reconstruct $w$ from continuity and enforce top kinematic condition $w_{top}=\eta_t$.
8. Advect density with conservative fluxes (`transport_advection_scheme`: `upwind1` or `tvd2`; fixed grid: top row in advective form; section 4).
9. Apply wind-driven surface mixing parameterization (if configured).
10. Apply global conservative density-mass correction within bounds.
11. Recompute final pressure fields ($p_h$ and optionally $p'=p-p_h$).

### 3) Momentum advection schemes (`upwind1`, `tvd2`)

For each advected component $\phi\in\{u,w\}$, the code computes conservative flux divergence:

```math
\mathcal{A}(\phi)=\frac{F^x_{i+1/2}-F^x_{i-1/2}}{\Delta x}+\frac{F^z_{k+1/2}-F^z_{k-1/2}}{\Delta z}.
```

`upwind1` (first-order upwind):

```math
F_{f}=v_f\,\phi_{upwind},
```

with donor-cell choice based on the sign of face velocity $v_f$.

`tvd2` (MC-limited reconstruction with Lax-Wendroff time correction, the same scheme as the density transport in
section 4):

```math
\phi^L_{i+1/2}=\phi_i+\frac{1}{2}(1-C_f)\,s_i,\qquad
\phi^R_{i+1/2}=\phi_{i+1}-\frac{1}{2}(1-C_f)\,s_{i+1},\qquad C_f=\frac{|v_f|\,\Delta t}{\Delta x},
```

```math
s_i = \mathrm{MC}(\Delta^-\phi_i,\Delta^+\phi_i),
```

```math
\mathrm{MC}(a,b)=\mathrm{minmod}\!\left(\mathrm{minmod}(2a,2b),\frac{a+b}{2}\right).
```

The face flux is then upwinded between $(\phi^L,\phi^R)$ using the face velocity sign; $v_f$ is the average of the two
neighbouring point velocities. The factor $(1-C_f)$ was added for issue #3: the momentum equations are advanced with one
forward-Euler step, which adds kinetic energy of order $(\Delta t^2/2)|\mathcal{A}|^2$ per step. Without the correction the
dense current of the lock exchange with obstacle created kinetic energy after hitting the side wall (up to 1.5 times the
released potential energy) and blew up; with it the energy budget stays consistent until $t\sqrt{g'/a}\approx9$.
`_advective_flux_divergence!` with `dt = 0` gives the former semi-discrete MUSCL flux.

### 4) Density transport method (`upwind1`, `tvd2`)

Density is advanced by one forward-Euler step of conservative face fluxes $F_f=v_f\,\rho_f$ on the C-grid velocities,
with zero flux through all outer faces:

```math
\rho^{n+1}=\rho^n-\Delta t\,\nabla\cdot(\rho\mathbf{u}).
```

The face value $\rho_f$ is selected by `ModelConfig.transport_advection_scheme` (default `upwind1`, which reproduces the
earlier results bit for bit):

- `upwind1`: donor cell, $\rho_f=\rho_{upwind}$. Monotone for $C_x+C_z\le1$, with $C_x=|u|\Delta t/\Delta x$ and
  $C_z=|w|\Delta t/\Delta z$.
- `tvd2`: flux-limited Lax-Wendroff with the MC slope of the momentum `tvd2` scheme (section 3). For $v_f>0$ at face $i-1/2$,

```math
\rho_f=\rho_{i-1}+\tfrac12\,(1-C_f)\,s_{i-1},\qquad C_f=\frac{|v_f|\,\Delta t}{\Delta x},\qquad s_i=\mathrm{MC}(\Delta^-\rho_i,\Delta^+\rho_i),
```

  mirrored for $v_f<0$, with $s=0$ in the boundary cells. It contains the time-correction factor $(1-C_f)$ (since issue #3
  also used for momentum, section 3). The reason is that the update is a single forward-Euler step: without that factor, MUSCL with forward
  Euler is first order in time and TVD only for $C\le1/2$ in 1D; with it the scheme is second order in time and TVD up to
  $C=1$ in 1D (MC lies in Sweby's TVD region $0\le\psi(r)\le\min(2r,2)$). The unsplit 2D update satisfies a local maximum
  principle for $C_x+C_z\le1/2$ (each face coefficient is at most $C(2-C)\le2C$). It has no corner-transport
  (cross-derivative) term, so for flow oblique to the grid the time error is $O(\Delta t\,|uw|\,\partial_{xz}\rho)$: first
  order in time but second order in space. Measured L1 orders for a Gaussian at $C=0.4$: about 2.2 for grid-aligned flow,
  1.0 for diagonal flow (error still about 4 times below `upwind1`), and 1.8 for diagonal flow at $C=0.02$.

References: van Leer (1977) for the MC limiter; Sweby (1984), SIAM J. Numer. Anal. 21, 995-1011, for flux-limited
Lax-Wendroff and its TVD region; LeVeque (2002), ch. 6.

Fixed-grid top row (variable top layer off, i.e. every non-hydrostatic run and hydrostatic runs with `variable_top_layer=false`):
the top face carries no flux although its velocity is $\eta_t$, so the top-row control volume has
$\nabla\cdot\mathbf{u}=-\eta_t/\Delta z\neq0$. The top row is therefore advanced in advective form,

```math
\mathcal{A}_{top}(\rho)=\nabla\cdot(\rho\mathbf{u})-\rho\,\nabla\cdot\mathbf{u}=\mathbf{u}\cdot\nabla\rho,
```

with $\nabla\cdot\mathbf{u}$ built from the same face velocities and zero-flux conventions as the fluxes
(`_density_top_row_advective_form!`, option `top_row_advective_form` of `advect_density!`). A uniform density stays uniform.
The fixed-grid mass $\sum\rho\,\Delta x\Delta z$ changes only by terms proportional to the top-row density anomaly (the uniform
part cancels because $\sum_i\eta_t=0$ in a closed basin); the global mass correction (section 11) removes that residual.

Optional bound-preserving clamping is applied during update:

```math
\rho^{n+1}\leftarrow\mathrm{clip}(\rho^{n+1},\rho_{min},\rho_{max}).
```

### 5) Viscous diffusion of u (`diffusion_term!`)

The viscous term of the horizontal momentum equation uses the viscosity $\nu$ (molecular `nu` plus the optional
mixing-length eddy viscosity $\nu_t=l^2|\partial u/\partial z|$, capped at `max_turbulent_viscosity`) in flux form with face viscosities $\nu_f=\tfrac12(\nu_a+\nu_b)$:

```math
\mathcal{D}(u)_{k,i}=\frac{F_{k+1/2}-F_{k-1/2}}{\Delta z}+\frac{G_{i+1/2}-G_{i-1/2}}{\Delta x},
\qquad F_{k+1/2}=\nu_f\frac{u_{k+1}-u_k}{\Delta z},\quad G_{i+1/2}=\nu_f\frac{u_{i+1}-u_i}{\Delta x}.
```

Zero flux through the bed and the free surface, $F_{1/2}=F_{n_z+1/2}=0$ (free slip at the bed, no stress at the top; the
wind stress enters the top row as a body force, section 10, which is the same stress condition). Over a variable bed the
face below the lowest wet row is treated as the bed; at the walls (outer and obstacle) $u=0$ enters the horizontal flux.
Second order in space, conservative in $z$, and the 5-point Laplacian for constant $\nu$ in the interior. Explicit
forward-Euler stability: $2\nu\Delta t(1/\Delta x^2+1/\Delta z^2)\le1$. Tests:
`test_verify_diffusion_flux_form_second_order_convergence` (manufactured solution with $\nu(z)$, order 2 in the max norm
including the top and bottom rows) and `test_verify_diffusion_flux_form_conserves_column_momentum`.

The former kernel $\nu\,\nabla^2u$ skipped the top and bottom rows: the top row had no viscous coupling to the row below,
so a wind stress stayed in the top cell. At $\Delta z=0.025$ m the resulting surface jet kept accelerating and the
hydrostatic downwelling at the downwind wall broke the run down (issue #4). The mixing-length $\nu_t$ uses one-sided shear
in the top and bottom rows, so that their face viscosity is not halved.

### 6) Hydrostatic pressure integration

Hydrostatic pressure uses vertical integration with top reference $p_h=0$:

```math
\frac{\partial p_h}{\partial z}=-\rho g,
```

implemented discretely as

```math
p_{h,k}=p_{h,k+1}+\frac{1}{2}(\rho_{k+1}+\rho_k)g\Delta z,
```

integrated from the top cell downward.

### 7) Non-hydrostatic pressure correction

The projection step solves

```math
\nabla^2 p' = \frac{\rho_0}{\Delta t}\nabla\cdot\mathbf{u}^{\ast}.
```

Two implementations are available:

1. `spectral`: DCT-based diagonal solve with Neumann-type eigenvalues.
2. `sparse`: finite-difference sparse Poisson matrix with LU solve.

Both implementations remove RHS mean and pressure-correction mean (gauge handling):

```math
\mathrm{mean}(\mathrm{RHS})\to 0,\qquad \mathrm{mean}(p')\to 0.
```

For the sparse branch, one row is pinned to avoid singularity (gauge fixation).

### 8) Barotropic free-surface solver

The free surface is updated from depth-integrated provisional transport $\hat U$:

```math
\eta_t + \partial_x U = 0,
\qquad
U=\int u\,dz.
```

The implemented tridiagonal Helmholtz-like update uses

```math
\alpha = \frac{gH\Delta t^2}{\Delta x^2},
```

with interior coefficients equivalent to

```math
-\alpha\eta_{i-1}^{n+1} + (1+2\alpha)\eta_i^{n+1} - \alpha\eta_{i+1}^{n+1} = \eta_i^n - \Delta t\,\partial_x\hat U_i,
```

and one-sided edge coefficients at boundaries.

### 9) Boundary treatment and kinematic closure

Implemented boundary behavior includes:

1. No-normal-flow walls for horizontal velocity (`u[:,1]=u[:,end]=0`), applied to the provisional $u^{\ast}$ before the
   barotropic solve as well as to the final $u$. The predictor also fills the wall faces (advection and diffusion evaluated
   from interior neighbours); on the fixed grid these values used to enter $\partial_x\hat U$ as a flux through the closed
   walls, so the water volume drifted (issue #4; `test_verify_wind_forcing_closes_volume_and_kinematic_budgets`).
2. Zero normal advective flux at outer faces for upwind/TVD fluxes.
3. Bottom vertical velocity reconstruction anchored at $w_{bottom}=0$.
4. Top kinematic enforcement by replacement:

```math
w_{top} \leftarrow \frac{\eta^{n+1}-\eta^n}{\Delta t}.
```

### 10) Stabilization and control methods

Rayleigh damping on provisional velocity:

```math
u^{\ast} \leftarrow (1-r\Delta t)u^{\ast}.
```

Froude limiter (if enabled):

```math
\mathbf{u}^{\ast} \leftarrow \mathbf{u}^{\ast}\cdot \frac{Fr_{max}}{Fr^{\ast}},\quad \text{if } Fr^{\ast}\gt Fr_{max}.
```

with

```math
Fr = \frac{\max|u|}{\sqrt{g'H}},\qquad g'=g\,\frac{\rho_{max}-\rho_{min}}{\rho_0}.
```

Wind stress forcing at the top $u$ row, part of the provisional velocity $u^{\ast}$ (step 3 of section 2):

```math
\Delta u_{wind}=\Delta t\,\frac{\tau_x}{\rho_0 h_{top}},
\qquad h_{top}=\Delta z\ \text{(fixed grid)}\ \text{or}\ \Delta z+\eta_{face}\ \text{(variable top layer)}.
```

The barotropic solve and the non-hydrostatic projection therefore include the wind transport, as in the semi-implicit
scheme of Casulli & Cheng (1992), where the surface stress enters the explicit part of the top-layer momentum equation. It
used to be added to the final $u$ after both; the depth-integrated divergence of $u$ then differed from $-\eta_t$ in the two
wall columns by $\Delta u_{wind}\Delta z/\Delta x$ per step, and the non-hydrostatic $u$ was not divergence-free there
(issue #4).

Wind-driven near-surface density exchange (conservative two-layer mixing):

```math
\gamma = \min\!\left(\max\!\left(\frac{K_z\Delta t}{\Delta z^2},0\right),0.5\right),
```

```math
\rho_{top}\leftarrow \rho_{top}-\gamma(\rho_{top}-\rho_{below}),
\qquad
\rho_{below}\leftarrow \rho_{below}+\gamma(\rho_{top}-\rho_{below}).
```

### 11) Conservative density-mass correction within bounds

After advection/mixing, constituent mass is nudged to initial mass target while preserving density bounds via slack-weight redistribution:

```math
\Delta M = M_{target}-M_{current},
\qquad
\Delta\rho_{cell}=\frac{\Delta M}{\Delta x\Delta z}.
```

Using slack

```math
s =
\begin{cases}
\rho_{max}^{init}-\rho, & \Delta M\gt 0 \\
\rho-\rho_{min}^{init}, & \Delta M\lt 0
\end{cases}
```

the update is

```math
\rho \leftarrow \rho + \Delta\rho_{cell}\,\frac{s}{\sum s},
```

then clamped to $[\rho_{min}^{init},\rho_{max}^{init}]$.

### 12) Implemented initial-condition construction methods

The following case initializers are implemented numerically and used by experiments/tests:

1. Lock exchange: piecewise two-density partition at domain midline.
2. Rayleigh-Taylor: heavy-over-light layers with sinusoidal interface perturbation.
3. Rising/sinking plume: Gaussian density anomaly in ambient field.
4. Jet impulse: Gaussian horizontal velocity patch with quiescent density field.
5. Internal seiche: two-layer density field with tilted sinusoidal interface.
6. Surface dam break: step in free surface $\eta$ over split index.
7. Kelvin-Helmholtz: hyperbolic-tangent shear layer with perturbed density interface.

### 13) Variable-thickness top layer (hydrostatic mode: on by default)

`ModelConfig.variable_top_layer` is `Union{Nothing,Bool}`, default `nothing`; `use_variable_top_layer(cfg)` resolves it to
`cfg.use_hydrostatic` when `nothing` (automatic: on for the hydrostatic mode, off for the non-hydrostatic mode), otherwise to
the explicit value. `variable_top_layer=true` with `use_hydrostatic=false` raises `ArgumentError` (see limitations below).

`variable_top_layer=false` (legacy treatment, forced by the automatic default when `use_hydrostatic=false`, and available as an
explicit override for the hydrostatic mode too — see limitations below): all layers have the fixed thickness $\Delta z$, the free
surface $\eta$ is a separate field that enters only through the barotropic solve (section 8), the surface-slope force
$-g\,\partial_x\eta$ and the top-face velocity $w_{top}=\eta_t$ (section 9). Density and momentum advection set the flux through
the top face to zero. In flux form that is consistent only if $w_{top}=0$ (rigid lid). With a moving surface a zero top flux is
a source of $\rho\,\eta_t\,\Delta t/\Delta z$ per step in the top row, and $\rho\approx 10^3\,\mathrm{kg/m^3}$ makes it large; this
was the original motivation for section 13 (see the wind-shear test note below). The surface moves in both modes: since the
reordering of section 14 the non-hydrostatic $\eta$ is driven by the momentum predictor, so the earlier assumption that the
non-hydrostatic projection keeps $\eta$ at round-off no longer holds. In the Rayleigh-Taylor experiment the source reached
1 kg/m³ per step (26 % of the density contrast) and lowered the linear growth rate to about 60 % of theory.
The fixed-grid density transport therefore advances the top row in advective form (`_density_top_row_advective_form!`, section 4),
which keeps a uniform density uniform; it is the $h\to\Delta z$ limit of the variable-top-layer budget below. The fixed-grid
momentum advection is unchanged.

`variable_top_layer=true` (hydrostatic mode; the default) lets the top layer follow the surface: layers $k\lt n_z$ keep $\Delta z$,
the top layer has $h=\Delta z+\eta$, and the free surface is a material boundary (zero flux through it). Per step:

1. Face thickness (old level): $h^{n}_{i+1/2}=\Delta z+\tfrac12(\eta^n_i+\eta^n_{i+1})$ (`top_layer_face_thickness!`).
2. Semi-implicit free surface (`solve_variable_thickness!`), same thicknesses in the right-hand side and the matrix:

```math
\eta_i^{n+1}-a_{i+1}\left(\eta_{i+1}^{n+1}-\eta_i^{n+1}\right)+a_i\left(\eta_i^{n+1}-\eta_{i-1}^{n+1}\right)=\eta_i^n-\Delta t\,\frac{\hat U_{i+1}-\hat U_i}{\Delta x},
```

```math
\hat U_i=\Delta z\sum_{k\lt n_z}\hat u_{k,i}+h^n_i\,\hat u_{n_z,i},\qquad
a_i=\frac{g\Delta t^2}{\Delta x^2}\left((n_z-1)\Delta z+h^n_i\right),
```

with $a=0$ at the boundary faces; the boundary-face values of $\hat u$ are set to the boundary condition first, so the solve sees the
transport the final velocities have. Then $u^{n+1}=\hat u-g\Delta t\,\partial_x\eta^{n+1}$ in every layer, which gives
$\eta^{n+1}-\eta^n=-\Delta t\sum_k\partial_x(h_k u_k^{n+1})$ exactly.
3. Continuity: layers below the top layer as before, $w_{k+1}=w_k-\Delta z\,(\partial_x u)_k$; the top layer closes its own budget

```math
\frac{\partial\eta}{\partial t}+\frac{\partial}{\partial x}\left(h\,u_{n_z}\right)=w_{n_z},
```

so $w_{n_z}=\eta_t+\partial_x(h u_{n_z})$. `last_kinematic_mismatch` now measures the residual of this identity (round-off level). The
stored surface value $w_{top}=\eta_t$ is unchanged.
4. Density (`advect_density_top_layer!`, face values from `transport_advection_scheme`, section 4): plain flux form for the layers below the top layer; the top layer uses the
thickness-integrated budget

```math
\rho^{n+1}_{top}=\frac{h^n\rho^n_{top}-\Delta t\left(\partial_x(h^n u\rho)-w_{n_z}\rho_{bot}\right)}{h^{n+1}},\qquad h^{n+1}=\Delta z+\eta^{n+1},
```

which preserves a uniform density exactly and conserves $\sum\rho h\,\Delta x$ to round-off. The constituent mass, the mass
correction (section 11) and the diagnostics use the actual cell volumes.
5. Momentum: the top row uses the advective form, $\mathcal{A}(u)\leftarrow\partial_x(v u)+\partial_z(w u)-u\,\nabla\!\cdot v$, with
$\nabla\!\cdot v$ built from the same face velocities and zero-flux boundary conventions as the flux form. It supplies the missing
$u\,\eta_t/h$ term and makes a uniform $u$ advection-free. Below the top row the validated flux form is unchanged.
6. Hydrostatic pressure: the reference at the top-cell centre is the weight of the upper half layer, $p=\rho_{top}g\Delta z/2$
(`top_half_layer=true`), instead of 0, so the top row sees the baroclinic gradient $(g\Delta z/2)\,\partial_x\rho_{top}$.

Limitations: the surface must stay inside the top layer, $\Delta z+\eta\ge$ `top_layer_min_fraction`$\cdot\Delta z$ (the run stops
with a clear error otherwise; there is no layer removal or wetting/drying). Cases with a large surface excursion relative to
$\Delta z$ must set `variable_top_layer=false` explicitly: the surface dam break (`initialize_surface_dam_break!`, $|\eta|$ set
directly by `eta_jump`) and any case forcing a large-amplitude free-surface wave (e.g. the vegetation wave-damping experiment,
amplitude $\mathcal{O}(1\,\mathrm{m})$ against $\Delta z=\mathcal{O}(0.1\,\mathrm{m})$); see the `variable_top_layer=false` comments
at their `ModelConfig` construction. The horizontal momentum fluxes of the top row are not weighted with the face thickness (an
$O(\eta/\Delta z)$ effect). Vegetation coupling assumes fixed geometry. The non-hydrostatic projection is not adapted (see above),
so `variable_top_layer=true` is rejected with `use_hydrostatic=false`.

Legacy-scheme-specific tests/scripts that pin `variable_top_layer=false` explicitly, because they check an invariant of the fixed
grid that the variable top layer replaces with a different one (cross-referenced above): `test_verify_hydrostatic_continuity_and_surface_closure_residuals`
and `experiments/validation/audit_hydrostatic_continuity_free_surface.jl` (top-row continuity residual; see
`test_verify_variable_top_layer_closes_kinematic_and_volume_budgets` for the variable-top-layer analogue).

References: Casulli and Cheng (1992), Int. J. Numer. Methods Fluids 15, 629-648 (semi-implicit free surface); Casulli and Walters (2000), Int. J.
Numer. Methods Fluids 32, 331-348 (z-layer model with a variable top layer). Equation numbers of these papers were not checked.

### 14) Non-hydrostatic free surface: rigid-lid coupling fix, and a remaining dispersion limitation

Before this fix, the non-hydrostatic branch's free surface $\eta$ stayed pinned at round-off ($\sim10^{-13}\,\mathrm{m}$)
regardless of real internal (baroclinic) forcing, effectively a rigid lid, even though the SAME barotropic solve (section 8) and
surface-slope correction (section 9) as the hydrostatic branch were being computed every step. Cause: in `step_forward!`, the
barotropic solve used to run AFTER the non-hydrostatic projection, on the projection's OUTPUT $\hat u$. The projection makes
$(\hat u,\hat w)$ exactly divergence-free cell-by-cell (Neumann-Neumann Poisson solve, section 7); summed over depth this forces
the transport divergence the barotropic solve sees, $\sum_k\partial_x\hat u_k$, to equal $-\hat w_{top}$ — but $\hat w_{top}$ is
never touched by the projection (it only corrects interior faces $2{:}n_z$) and still carries the value copied from the OLD
$w_{top}$ (the previous step's $\eta_t$) at the top of the function. The recursion $\eta^{n+1}\approx\eta^n+\Delta t\,\eta_t^n$ is
homogeneous in $\eta$, with $\eta\equiv0$ a self-consistent fixed point — an incidental consequence of the step ordering, not a
deliberate rigid-lid choice.

Fix: the barotropic solve and the surface-slope correction (applied to `ws.u_hat` in place, not yet written to `state.u`) now run
BEFORE the non-hydrostatic projection, in both modes — exactly where they already were relative to the (absent) projection on the
hydrostatic branch. This decouples $\eta$ from the projection: it is now driven directly by the momentum-predictor transport, the
same physics as the hydrostatic branch. Nothing changes for the hydrostatic branch (its projection block is a no-op). The
projection's own top-face value (`ws.w_hat[end,:]`) is also refreshed to the just-computed $\eta_t$ immediately before the solve
(previously stale by one step), so its divergence right-hand side is internally consistent; because the top condition is
homogeneous Neumann, the projection's correction there is exactly zero, so this refreshed value is left standing afterward, the
same way the (always exactly 0) bottom face already is.

Verified (`test_verify_nonhydrostatic_free_surface_responds_to_internal_forcing`): a lock exchange (uniform-density regions in
hydrostatic equilibrium give $\eta\equiv0$; the forcing here is purely baroclinic) run in non-hydrostatic mode now reaches
$\max|\eta|\sim7\times10^{-4}\,\mathrm{m}$ after 400 steps ($\Delta t=5\times10^{-4}\,\mathrm{s}$), several orders of magnitude
above round-off, while the kinematic closure and interior divergence stay at round-off (the reordering does not degrade the
existing continuity guarantees). No other test in the suite changed outcome from this fix (full-suite regression check).

**Limitation of the DEFAULT (legacy Neumann) mode: non-hydrostatic surface waves are non-dispersive.** The fix above makes $\eta$
respond to real (baroclinic) internal dynamics, but a pure free-surface gravity wave — uniform density, so zero baroclinic
pressure gradient and zero buoyancy forcing — is governed only by the barotropic surface-slope force $-g\,\partial_x\eta$, applied
UNIFORMLY over depth OUTSIDE the elliptic (non-hydrostatic) solve. With uniform density the momentum predictor's $w$ has no
forcing either, so the field entering the projection is already very close to the continuity-consistent (depth-uniform $u$,
linear-in-depth $w$) state, and the projection has essentially nothing left to correct. Consequence, verified by
`test_verify_nonhydrostatic_surface_wave_period_matches_shallow_water_not_full_dispersion` (a domain with $k H\approx1.57$, chosen so the
two relations differ by about 31%): the non-hydrostatic branch's surface-wave period still matches the shallow-water relation
$T=2L/\sqrt{gH}$, not the full linear dispersion relation $\omega^2=gk\tanh(kH)$. **This is fixed for a new, opt-in mode**
(`ModelConfig.nonhydro_surface_dirichlet=true`): section 15.

References: same free-surface literature as section 13 (Casulli and Cheng, 1992; Casulli and Walters, 2000); Stoker (1957), *Water
Waves*, and Whitham (1974), *Linear and Nonlinear Waves*, for the linear surface-gravity-wave dispersion relation
$\omega^2=gk\tanh(kH)$ and its shallow-water limit $\omega^2\to gk^2H$ as $kH\to0$.

### 15) Genuine non-hydrostatic surface-wave dispersion: Dirichlet top condition (opt-in)

`ModelConfig.nonhydro_surface_dirichlet` (default `false`, non-hydrostatic mode only, rejected with `use_hydrostatic=true`)
switches the non-hydrostatic pressure correction's top boundary from the legacy Neumann (zero-flux) condition to a Dirichlet
condition, $p'=0$ at the top face, fixing the section 14 dispersion limitation. Requires `pressure_solver_method="sparse"`: the
spectral (DCT-based) solver's fast transform represents only a Neumann-Neumann eigenbasis in $z$ and cannot express a mixed
Dirichlet(top)/Neumann(bottom) condition.

**Physical motivation.** The model splits total pressure as $p=p_{hydro}+p'$ (section 7), with the barotropic surface-slope force
$-g\,\partial_x\eta$ (a REFERENCE-density, $\rho_0$-based, depth-UNIFORM approximation of the surface's dynamic pressure) applied
separately. For a linear surface wave, potential-flow theory gives the exact dynamic pressure
$p_{dyn}(z)=\rho_0 g\,\eta\,\cosh(k(z+H))/\cosh(kH)$: equal to the barotropic approximation $\rho_0 g\eta$ exactly AT the surface
($z=0$: $\cosh(kH)/\cosh(kH)=1$) but decaying with depth for $kH$ not small. The remaining piece the barotropic term misses,
$p'(z)=p_{dyn}(z)-\rho_0 g\eta$, is therefore exactly 0 AT the top face and non-zero below — i.e. $p'=0$ at the top is the correct
dynamic free-surface condition for the part of the pressure not already carried by the barotropic/hydrostatic split, and letting the
elliptic solve represent that depth-decay is what makes the resulting flow dispersive.

**Discretization (`build_poisson_matrix(shape, dx, dz; top_dirichlet=true)`).** The top face sits half a cell above the top cell
centre; the Dirichlet closure uses a ghost cell $p_{ghost}=-p_{nz,i}$ (linear interpolation through $p_{nz,i}$ and $p_{ghost}$ gives
exactly 0 at the face). Substituting into the standard 5-point stencil adds $-2/\Delta z^2$ to the top row's diagonal (not
$-1/\Delta z^2$: the ghost contributes both the "a neighbour exists" decrement every real neighbour gives AND its own value,
$+1/\Delta z^2\cdot p_{ghost}=-1/\Delta z^2\cdot p_{nz,i}$; verified in `test_verify_poisson_matrix_top_dirichlet_matches_manufactured_solution`
against $p(z)=C/2(z^2-H^2)$, the solution of $p''=C$ with $p'(0)=0$, $p(H)=0$, converging at 2nd order). A Dirichlet condition
removes the pure-Neumann operator's 1D null space (the constant mode), so — unlike the legacy case — no row is pinned and the RHS
and solution means must NOT be removed (`PressureCorrectionSolver.top_dirichlet`; `solve!` branches on it).

**A second, independent defect found while validating this** (`test_verify_nonhydro_dirichlet_reference_pressure_removes_uniform_density_offset`):
`solve_hydrostatic_pressure!` integrates the FULL density $\rho$, not the anomaly $\rho-\rho_{ref}$ relative to a reference profile.
For the legacy Neumann-Neumann projection this is harmless — the projection's role there is only to correct $u$ via
$\partial_x p'$ (section 7), and the "trivial", $x$-independent reference-profile contribution to $p'$ has zero $x$-gradient
regardless of its (possibly large) magnitude, and the mean-removal step additionally erases whatever of it is left. It is NOT
harmless with the Dirichlet condition, where $p'$'s absolute magnitude is now physically meaningful (no mean removal): empirically,
for a $\rho\equiv\rho_0$ test case the reference contribution alone gave $|p'|\sim3.7\times10^4\,\mathrm{Pa}$, dwarfing the genuine
dynamic-pressure signal ($\rho_0 g\eta\sim1\,\mathrm{Pa}$ for a $10^{-4}\,\mathrm{m}$ wave) by four orders of magnitude — besides the
conceptual problem of applying a "$p'=0$ at the surface" condition to a quantity dominated by something that is not $p'$ at all.
Fix: `Model.grad_p_z_reference` caches $\partial_z p_{ref}$, where $p_{ref}$ is the hydrostatic pressure of the reference profile
(the initial state's column 1, matching `rho_0_w`'s reference, repeated across $x$ so it is exactly $x$-independent — its own
$x$-gradient is therefore already exactly 0 without correction); it is subtracted from `ws.grad_p_z` before the $\hat w$ predictor,
only when `nonhydro_surface_dirichlet` is active. After the fix, the same test case's $|p'|$ dropped to $\approx0.44\,\mathrm{Pa}$,
the correct order of magnitude.

**Validation** (`test_validate_nonhydrostatic_dirichlet_surface_wave_matches_full_dispersion`): the same $kH\approx1.57$ domain as
section 14's shallow-water test, with `nonhydro_surface_dirichlet=true`. The numerical period converges toward the exact relation
$T_{exact}=2\pi/\sqrt{gk\tanh(kH)}$ under grid refinement (observed: nx=nz=80, dt=0.002 gives a relative error of $-0.84\%$;
nx=80, nz=160, dt=0.001 gives $-0.42\%$ — refining $n_z$, the dominant error source, roughly halves the error, consistent with the
first-order-accurate `upwind1` advection scheme limiting the overall rate) and ends up an order of magnitude closer to the exact
relation than to the shallow-water one (which differs from the exact relation by 31% on this domain) — a decisive, convergent
confirmation of genuine dispersion, not a coincidental single-grid match.

**Scope and limitations.** Implemented for the FIXED (`variable_top_layer=false`) grid only — combining a moving, variable-thickness
top layer (section 13) with the Dirichlet condition would need a per-step-varying elliptic operator (the top layer's thickness
changes every step) and is not attempted; the two are mutually exclusive in practice anyway (`variable_top_layer=true` is
hydrostatic-mode-only, `nonhydro_surface_dirichlet=true` is non-hydrostatic-mode-only). The `sparse` (LU-factorized) pressure solver
is markedly more expensive at production grid sizes than the default `spectral` one: measured on this machine, a 1500x200 grid
(lock-exchange scale) took about 1.9 s to factorize once at construction and about 0.095 s per step for the pressure solve alone —
comparable to or exceeding a full spectral-method step's total cost, so a production run at that scale would run several times
slower overall. The kinematic top-face velocity fed into the projection's right-hand side (`ws.w_hat[end,:] .= eta_t`, refreshed from
the barotropic solve) is treated the same way for both the Dirichlet and Neumann top conditions — see the `step/nonhydro_projection`
comment in `src/model.jl` for why: the free surface's own kinematic condition is enforced through the barotropic solve and the
$w$-reconstruction step (section 9), not through this elliptic solve, whose role is only to determine the pressure (and correct the
interior/horizontal flow) consistent with that prescribed top velocity and the chosen top pressure condition.

References: same free-surface literature as sections 13-14; Lamb (1932), *Hydrodynamics*, and Dean and Dalrymple (1991), *Water Wave
Mechanics for Engineers and Scientists*, for the potential-flow derivation of $p_{dyn}(z)$. Equation numbers of all cited works were
not checked.

### Numerical-method references

- LeVeque (2002), *Finite Volume Methods for Hyperbolic Problems*.
- Harten (1983), High resolution schemes for hyperbolic conservation laws.
- van Leer (1977), Towards the ultimate conservative difference scheme (limiter framework).
- Sweby (1984), High resolution schemes using flux limiters for hyperbolic conservation laws, SIAM J. Numer. Anal. 21, 995-1011.
- Ferziger and Peric (2002), *Computational Methods for Fluid Dynamics*.
- Press et al. (2007), *Numerical Recipes* (elliptic solves, transforms, tridiagonal systems).

### 16) Vegetation (`src/vegetation.jl`, issue #8)

Two representations share one stem model. A **patch** (`VegetationPatch`) has the stem length $h_v$, diameter $b$,
stem density $n$ (stems/m²) and drag coefficient $C_D$ over an x-range; a representative stem per u-face column
carries the drag of its column, and the force per unit volume is $n$ times the stem force per unit height. The u-faces
are weighted by the fraction of their control interval $x\pm\Delta x/2$ inside the patch. An **individual stem**
(`VegetationStem`) stands for a row of stems with lateral spacing $\Delta y$; its force is deposited as
$F/(\rho_0\Delta x\,h\,\Delta y)$, split linearly between the two neighbouring u-faces. Here $h$ is the layer thickness
of the momentum equation: $\Delta z$, and in the top row with the variable top layer its u-face thickness. Stems are
loaded up to the top of the momentum layers: $n_z\Delta z$ on the fixed grid (linear free surface, the top layer keeps
$\Delta z$) and $(n_z-1)\Delta z+h_{top}$ with the variable top layer (individual stems: interpolated to the base). Earlier
the loads reached $n_z\Delta z+\eta$ while the force was divided by $\Delta z$, which scaled the top-row drag by
$(\Delta z+\eta)/\Delta z$.

Stems are rigid ($EI=\infty$) or flexible: inextensible, clamped at the base, quasi-static (elastica; Alben et al. 2002;
Gosselin et al. 2010; Luhar & Nepf 2011),

```math
EI\,\frac{d\theta}{ds}=T(s),\qquad f_n=\tfrac12\rho_0 C_D b\,u_n|u_n|,\quad u_n=u\cos\theta,
```

with $T(s)$ the moment of the loads beyond $s$ (normal drag and net buoyancy per length $B_l$) and $\theta$ the angle
from the vertical. Per unit vertical extent the horizontal drag is $\tfrac12\rho_0C_Db\,u|u|\cos^2\theta$. Discretisation:
$N$ straight segments, angles at the segment midpoints, curvature at the nodes ($\theta_1=\kappa_0\Delta s/2$,
$\theta_m=\theta_{m-1}+\kappa_{m-1}\Delta s$, second order). The segments are split at the cell boundaries; a piece in row
$k$ uses the velocity of that row, and the same pieces load the stem and, with the opposite sign, the fluid (exact
momentum exchange; a rigid patch gives exactly $\tfrac12C_Da\,u|u|$ per unit volume, $a=nb$). The posture is solved
with Newton's method (finite-difference Jacobian, backtracking), warm-started every step. If Newton fails (e.g. from an
upright stem at $Ca\gtrsim3000$), the load is applied in stages (velocity times $\sqrt{\lambda}$, $\lambda=1/n\ldots1$, with
$n=8, 32, 128$) from the start posture; postures that still do not converge are counted in
`VegetationModel.posture_failures`. Stem parts above the free
surface carry no load; the vertical drag component is not applied to $w$.

The drag enters the u predictor explicitly, from the velocities of time level n. It is monotone for
$\Delta t\,K\le1$ with the linearised drag rate $K$ (rigid patch: $\tfrac12C_Da|u|$); `step_forward!` checks this.
WHY explicit: an implicit factor applied to the predictor before the free-surface solve would make the steady state
depend on $\Delta t$, because the surface-slope correction is not divided by it.

`ModelConfig.body_force_x` (m/s²) adds a uniform body force (channel slope) for steady periodic-channel flow.
`sponge_length` (m) and `sponge_rate` (1/s) add an absorbing layer at the right end: over the last `sponge_length` the
u predictor is multiplied by $1/(1+\Delta t\,r\,s^2)$, $s$ the relative position in the layer (quadratic ramp, implicit,
stable for any rate); used for the wave cases. The
former ad hoc vegetation viscosity (`vegetation_turbulence_strength`) and the old stem model (force relaxation 0.35,
response gain 8, `source_strength`, string-like bending) were removed; in the old tests that model changed the wave
amplitude by about $10^{-15}$ and the lock-exchange front by less than one cell.

### 17) k-epsilon turbulence closure (`src/turbulence.jl`, issue #8)

Selected with `turbulence_closure = "k_epsilon"` (flat bed and fixed top layer only: the closure assumes layers of
equal thickness, so the automatic default of `variable_top_layer` is false with it and an explicit
`variable_top_layer=true` is rejected; the closure cannot be switched on or off after the model is built). Standard high-Reynolds-number k-epsilon (Launder &
Spalding 1974; Rodi 1993) on the cell centres:

```math
\frac{Dk}{Dt}=\partial_z\!\left(\Big(\nu+\frac{\nu_t}{\sigma_k}\Big)\partial_z k\right)+P+G+C_{fk}P_v-\varepsilon,\qquad
\frac{D\varepsilon}{Dt}=\partial_z\!\left(\Big(\nu+\frac{\nu_t}{\sigma_\varepsilon}\Big)\partial_z\varepsilon\right)
+\frac{\varepsilon}{k}\left(C_1(P+C_{f\varepsilon}P_v)+C_3\max(G,0)-C_2\varepsilon\right),\qquad \nu_t=C_\mu\frac{k^2}{\varepsilon},
```

with $P=\nu_t[(\partial_zu+\partial_xw)^2+2(\partial_xu)^2+2(\partial_zw)^2]$, the buoyancy term
$G=-(\nu_t/\sigma_\rho)N^2$, $N^2=-(g/\rho_0)\partial_z\rho$ ($C_3=0$ stable, 1 unstable; Rodi 1987), and the vegetation
production $P_v$, the work of the drag per unit mass (from the u-face accelerations of `vegetation_source!` times $u$,
averaged to the cells; $\tfrac12C_Da|u|^3$ for a rigid canopy). Constants: $C_\mu=0.09$, $C_1=1.44$, $C_2=1.92$,
$\sigma_k=1.0$, $\sigma_\varepsilon=1.3$, $\sigma_\rho=0.7$; vegetation weights $C_{fk}=1.0$, $C_{f\varepsilon}=1.33$
(Lopez & Garcia 1997, WES Technical Report WRP-CP-10, Eqs. 26-27 and p. 58; adjustable per `KEpsilonState`).

Numerics after each flow and density update: explicit first-order upwind advection (conservative, closed walls,
periodic or zero-gradient inflow), then per column a backward-Euler vertical diffusion with the sinks linearised
implicitly (Patankar: $\varepsilon/k^\ast\,k$, $C_2\varepsilon^\ast/k^\ast\,\varepsilon$, negative buoyancy); positive for
any $\Delta t$, first order in time. Bed: with `bed_roughness_length` $z_0>0$ a log-law quadratic friction
$c_b|u_1|u_1$ on the lowest row, $c_b=(\kappa/\ln(z_1/z_0))^2$, $z_1=\Delta z/2$, and the wall functions
$k_1=u_*^2/\sqrt{C_\mu}$, $\varepsilon_1=u_*^3/(\kappa z_1)$ (Rodi 1993); otherwise free slip and zero flux. Surface:
zero flux for $k$; for $\varepsilon$ the value $\varepsilon_s=C_\mu^{3/4}k_s^{3/2}/(\kappa\,0.07H)$ (Rodi 1993; default, as
in Lopez & Garcia 1997) or zero flux.
The momentum equation uses $\nu+\nu_t$ (u-faces: mean of the neighbouring cells; the mixing-length cap does not apply).
Density gets the turbulent diffusivity $\nu_t/\sigma_\rho$ (implicit, conservative with the layer thicknesses, top layer
$\Delta z+\eta$).

Known limitations: first order in time; in a homogeneous emergent canopy the level of k is not fixed by the
vegetation terms (below).

**Vegetation terms.** $C_1$ multiplies the vegetation term in the $\varepsilon$ equation (Lopez & Garcia 1997, Eq. 27).
In a homogeneous canopy without shear the k equation then drives $\varepsilon\to C_{fk}P_v$ and the $\varepsilon$
equation $\varepsilon\to(C_1C_{f\varepsilon}/C_2)P_v$; with $C_{f\varepsilon}=(C_2/C_1)C_{fk}$ (their p. 24, after Burke
1982) both agree, and $C_{f\varepsilon}=1.33$ gives $0.9975$ (test). The level of k is neutral, so it drifts slowly in an
emergent canopy; a wake length scale would fix it. An earlier version without $C_1$ gave $\varepsilon/P_v=0.693$ and
unbounded k.

**Comparison with Lopez & Garcia (1997) and Dunn et al. (1996)** (`experiments/vegetation/plot_vegetation_dunn_k_epsilon.jl`,
`plot_vegetation_manning.jl`). Flexible runs are computed as rigid canopies of the deflected height, as Lopez & Garcia
did. Against their computed k-epsilon profiles (runs 1, 9, 13, 17, digitized from Figs. 6.1, 6.2, 6.4, 6.5): velocity
0.92-1.03 of theirs, stress peak 0.97-1.05, stress in the lower canopy 0.68-1.35. Against the measurements: the depth-mean
velocity is 1.09-1.29 times the measured Q/(bH) (rigid runs, measured $C_{DB}$; 1.17-1.31 with $C_D=1$), and the measured
stress at the canopy top is only 0.12-0.83 of the 2D momentum balance $gS(H-h_p)$. Lopez & Garcia's model overpredicts
in the same way; they attribute both to secondary currents in the 0.91 m wide flume. The shape of the velocity profile in
the canopy, $u/u(h_p)$, agrees with the measurements (median deviation 4.8 %, largest 21 %). In their plant-density
series (Figs. 6.16, 6.18) the momentum transfer to the bed agrees within a few per cent, but over dense canopies in deep
water (H/h_p = 6-9) our model gives more resistance: Manning n up to 1.25 (160 stems/m²) and 1.53 (1581 stems/m²) times
theirs at their depth. Our profile above the canopy then corresponds to a roughness length of about $h_p/2$; with the
vegetation terms switched off the resistance is higher still. Converged in time and grid; not resolved.

### 18) Wetting and drying (`src/wetting_drying.jl`, issue #31)

`ModelConfig(wetting_drying=true)` selects a separate time step (`_step_forward_wetting_drying!`) on z-level cells cut
by the bed and by the free surface; the other time steps are not used then. Hydrostatic or non-hydrostatic,
`upwind1` or `tvd2` for momentum and transport; `nz = 1` gives the 1D (depth-averaged) model on the same code path.
Plan and decisions: `docs/plan_issue_31_wetting_drying.md`; depth-positivity proof: `docs/depth_positivity.md`;
experiments: `experiments/wetting_drying/README.md`.

**Geometry.** Interfaces $z_k=(k-1)\Delta z$; the top layer is open above. Column $i$ has the bed $z_{b,i}$ (any value
$\ge0$, `bed_level`) and the level $\zeta_i=H+\eta_i$; a dry column has $\zeta_i=z_{b,i}$ (stored as the smallest float with
$\mathrm{fl}(H+\eta)\ge z_b$). Layer thicknesses

```math
h_{k,i}=\max\bigl(0,\ \min(z_{k+1},\zeta_i)-\max(z_k,z_{b,i})\bigr),\qquad \textstyle\sum_k h_{k,i}=\max(\zeta_i-z_{b,i},0),
```

which for a flat bed and $\zeta$ in the top layer is the variable top layer of section 13. Faces use the bed of the
higher neighbour, $z_{b,f}=\max(z_{b,i},z_{b,i+1})$, and the upwind level of time $n$ (sign of $\sum_k u_{k,f}$; the higher
level for zero flow), as in Stelling & Duinmeijer (2003); a face whose upwind depth does not exceed
`dry_depth_threshold` (default $10^{-6}$ m) is closed.

**Time step.**
1. Cell and face thicknesses at time $n$; a face layer that opens takes the velocity of the nearest open layer below
   (see "depth-uniform velocity" below); baroclinic pressure
   $p'(z)=g\int_z^{\zeta}(\rho-\rho_0)\,dz'$ at the *nominal* layer centres $z_k+\Delta z/2$, layers above the surface
   extrapolated with the surface-layer density. WHY nominal centres: a horizontally uniform stratification at rest then
   gives bit-identical $p'$ in neighbouring columns whatever their bed (exact lake at rest, also stratified).
2. Momentum predictor on the wet face layers: advection (below), baroclinic pressure, body force, Rayleigh damping, sponge,
   implicit vertical viscosity with zero stress at bed and surface.
3. Free surface in the volume formulation of Casulli (2009): $V(\eta)+T\eta=b$ with $V_i=\max(0,H+\eta_i-z_{b,i})$,
   $(T\eta)_i=a_{i+1/2}(\eta_i-\eta_{i+1})+a_{i-1/2}(\eta_i-\eta_{i-1})$, $a_f=g\Delta t^2H_f/\Delta x^2$,
   $b_i=V_i(\eta^n)-\frac{\Delta t}{\Delta x}(\hat U_{i+1/2}-\hat U_{i-1/2})$; Newton's method of Brugnano & Casulli (2008)
   from $\eta^n$. Every depth stays $\ge0$ for any $\Delta t$; the iteration ends after at most $n_x+1$ steps; a lake at rest
   has a zero residual and is not iterated (proof and conditions: `docs/depth_positivity.md`). In the validation cases
   the solve takes one iteration per step (the linear semi-implicit solve of Casulli & Cheng 1992).
4. $u=\hat u-g\Delta t\,\partial_x\eta^{n+1}$ on the wet face layers; non-hydrostatic correction (below).
5. $\eta^{n+1}$ from the flux divergence (volume conserved to round-off); a depth below zero by more than the bound of
   `docs/depth_positivity.md`, section 5, stops the run (`DomainError`), as does a negative depth at the start of a step.
6. Vertical fluxes through the fixed interfaces from the layer continuity with the new thicknesses, integrated upwards
   from the bed.
7. Conservative transport of $\rho$ with the same discharges and cell volumes, in increment form
   $\rho^{n+1}=\rho^n+D/h^{n+1}$, $D=(h\rho)^{n+1}-h^{n+1}\rho^n$. WHY the increment: $(h\rho)/h$ changes $\rho$ by round-off even
   without flow, which drove an internal motion of $5\cdot10^{-9}$ m/s in a stratified lake at rest. Face values:
   `upwind1`, or `tvd2` = the flux-limited Lax-Wendroff value of section 4 with the Courant number of the cut cell
   $|q|\Delta t/(\Delta x\,h_{up})$. Empty cells ($h<10^{-12}$ m) pass their increment to the thickest cell of the column.
   The scheme is monotone only if no cell loses more than its old volume in a step, which thin cut cells can violate;
   values outside the initial bounds are clipped and the clipped mass is redistributed over the cells with room for it
   (slack-weighted), so the mass stays exact. There is no global mass correction (section 11) in this mode.

**Momentum advection.** Momentum-conservative upwind form of Stelling & Duinmeijer (2003), per layer with the
discharges $q_{k,f}=h_{k,f}u_{k,f}$ and the interface fluxes $W$ of time $n$:

```math
\frac{du_{k,f}}{dt}=\frac{a_L(u^*_L-u)-a_R(u^*_R-u)}{\Delta x}+\frac{\bar W_{bot}(u^*_{bot}-u)-\bar W_{top}(u^*_{top}-u)}{\bar h_{k}},
```

with $u^*$ the value upwinded by the sign of each flux: the upwind velocity (`upwind1`; on an outflow side $u^*=u$, so
the form is a sum over the inflow sides with non-negative coefficients) or its limited MUSCL extrapolation with the MC
slope (`tvd2`, forward Euler like the momentum `tvd2` of section 3; the outflow sides then contribute $s_f/2$). The
update is capped: with $C=\Delta t\sum$(inflow coefficients) the tendency is scaled by $1/C$ where $C>1$, so for `upwind1`
the update stays a convex combination (thin layers at the front are stable).

**Depth-uniform velocity: analysis and the advective velocities $a_L,a_R$.** Hydrostatic theory keeps a depth-uniform
velocity depth-uniform: in every layer $u_t=-uu_x-g\zeta_x$. A first version used the layer-wise weights
$a_C=\bar q_{k,C}/\bar h_{k}$ ($\bar q$ the cell discharge, $\bar h$ the cell thickness averaged to the face), the direct
layer-wise transcription of the 1D scheme. In the 2DV Ritter dam break the layers then drifted apart (upper layers up to
0.29 m/s, 6 %, slower than the bed layer at $t=5$ s, $\Delta x=0.25$ m), and the draining surface layers carried the
slower fluid to the front.

*Manufactured solution.* To isolate the operator, `test_verify_wd_advection_depth_uniform_mms_*` (and
`experiments/wetting_drying/plot_wetting_drying.jl mms`) evaluates the tendency for a prescribed state: a depth-uniform
$u(x)=1+2\tanh((x-50)/15)$ m/s (changes sign) under a surface $\zeta(x)=0.6+0.3(1-\tanh((x-50)/15))$ m that crosses six
layer interfaces ($\Delta z=0.1$ m), with $W$ from the discrete layer continuity (the column continuity gives $\zeta_t$, the
surface layer takes the volume change). The exact tendency is $-uu_x$ in every layer. Results (maximum error over all
layers, m/s²; $|uu_x|\le0.27$):

| $\Delta x$ (m) | 1 | 0.5 | 0.25 | 0.125 | 0.0625 |
|---|---|---|---|---|---|
| 1D `upwind1` | 7.8e-3 | 3.9e-3 | 1.9e-3 | 9.7e-4 | 4.8e-4 |
| 2DV, first version (layer-wise weights) | 0.51 | 0.46 | 0.87 | 0.39 | 0.17 |
| 2DV `upwind1` and `tvd2`, present | 1.2e-2 | 5.3e-3 | 2.6e-3 | 1.3e-3 | 5.7e-4 |
| 1D `tvd2` | 2.0e-3 | 1.1e-3 | 5.4e-4 | 2.7e-4 | 1.4e-4 |

The first version did not converge in the maximum norm (its mean error did, at first order: the defects are confined to
about one cell per interface crossing). The harness located three defects; each is fixed:

1. *Weight in thinly cut layers.* With $\bar q_{k,L}=\tfrac12(h_{k,f-1}u_{f-1}+h_{k,f}u_f)$ and upwinded face thicknesses,
   $\bar q_{k,L}/\bar h_k\approx u\,r_k$ with $r_k-1\approx-\Delta x\,\partial_x h_k/h_k$. For full layers $\partial_xh_k=0$; for
   the layer that the surface cuts, $\partial_x h_k=\partial_x\zeta$, so the relative error is $\Delta x|\zeta_x|/h_k$, which is
   O(1) wherever the cut layer is thinner than the surface drop over one cell, i.e. at every interface crossing, on every
   grid. At the worst face of the harness ($\Delta x=0.25$ m) the cut layer was 1.3 mm thick, $r=5.3$ and the tendency 5
   times the exact one. The present weights are the layer velocity times the column factor of the 1D scheme,

   ```math
   a_C=\frac{\bar q_{k,C}}{\tilde h_{k,C}}\;\frac{\tilde H_C}{\bar H_f},\qquad \tilde h_{k,C}=\tfrac12(h_{k,f'}+h_{k,f''}),\quad
   \tilde H_C=\sum_k\tilde h_{k,C},\quad \bar H_f=\sum_k\bar h_k,
   ```

   ($f',f''$ the faces of cell $C$): $\bar q_{k,C}/\tilde h_{k,C}$ is the velocity of the layer in the cell, and
   $\tilde H_C/\bar H_f$ the factor that makes the 1D scheme momentum conservative. For one layer this is exactly the 1D
   weight of Stelling & Duinmeijer; for a depth-uniform flow every layer gets the 1D advective velocity. Per-layer momentum
   conservation is given up in thin cut layers (the column-integrated form of the 1D scheme is kept).
2. *Closed far face.* If the far face of the inflow cell is closed for the layer (its upwind level lies below the layer),
   the form counted it as inflow with $u=0$ and dragged the face towards rest (the remaining O(1) errors after fix 1, at
   faces where the cut layer ends upstream). The water entering the layer there comes from the nearest open layer below at
   that face and rises; its velocity is now the upwind value (no inflow if all layers below are closed, e.g. a dry cell).
3. *Inconsistent tvd2 stencil.* Next to a closed face, `tvd2` combined a second-order inflow value with a first-order
   outflow value, which halved the tendency. Second order is now used in a direction only if the whole stencil (faces
   $f-2\ldots f+2$, layers $k-2\ldots k+2$) is open. Consequence: the 2DV maximum error of `tvd2` is first order (set by the
   faces where layers end), the mean error second order.

A fourth effect appears only in time-dependent runs: a face layer that opens when the upwind level rises past its lower
interface carried $u=0$ from the time it was closed. The layer-wise weights adjusted it almost instantly (the thin
$\bar h$ made the coefficient huge, which hid the problem); with consistent weights it adjusts at the physical rate
$u/\Delta x$ and lagged for many steps. A new face layer now takes the velocity of the nearest open layer below (or above,
for a face that opens from the bed upwards), the common practice of z-layer models.

Effect in the 2DV Ritter dam break (12 layers, $t=5$ s): the spread of the layer velocities dropped from 0.29 m/s to
0.0013 m/s (`upwind1`) and from 0.18 to 0.0028 m/s (`tvd2`), converging at first order (0.0080, 0.0036, 0.0013, 0.0005
m/s for $\Delta x=1\ldots0.125$ m, `upwind1`); depth error and front position of 2DV `upwind1` equal those of 1D to the
printed digits, and the `tvd2` front lag fell from 11.3 % to 6.6 % (1D: 5.6 %). Carrier-Greenspan 2DV: surface error
3.4 → 2.1 mm.

**Non-hydrostatic mode** (`use_hydrostatic=false`, $n_z>1$). Pressure split $p=p_{hydrostatic}+q$ (Casulli 1999; Casulli
& Zanolli 2002): after the free-surface solve, (i) a predictor of the vertical velocity at the interfaces between two wet
cells, $w^*=w^n-\Delta t\,\mathrm{adv}(w)$ (first-order upwind, advective form, capped); (ii) a Poisson problem for $q$ on
the wet cells below the surface cell of each column ($q=0$ in the surface cell, the dynamic surface condition, and in
empty cells): every such cell satisfies its continuity with $u=u_{fs}-\frac{\Delta t}{\rho_0}\partial_xq$ and
$w=w^*-\frac{\Delta t}{\rho_0}\partial_zq$ ($\partial_z$ over $\tfrac12(h_{k-1}+h_k)$, $w=0$ at the bed), which gives a symmetric
positive definite system, assembled and solved with a sparse Cholesky factorization every step (the wet domain
changes); (iii) $-\frac{\Delta t}{\rho_0}\partial_xq$ is added to the predictor $\hat u$ and the free-surface solve is repeated.
WHY (iii) instead of correcting $u$ directly (Casulli 1999): the direct correction changes the column transports after
the positive solve and voids the positivity proof; with the repeated solve the final velocities come from it. The price
is that the continuity of the cells below the surface holds only approximately (the two solves differ by a depth-uniform
change of the surface gradient); $w$ then follows from the exact layer continuity (step 6). $q=0$ in the surface cell is
first order in the surface-cell thickness. Standing wave at $kH=\pi/2$ (L = 8 m, H = 4 m) against
$\omega^2=gk\tanh(kH)$ (the shallow-water period is 24 % shorter): period error 1.75 %, 0.89 %, 0.45 % on 40 x 20, 80 x 40 and
160 x 80 cells (first order), both schemes. A lake at rest gives a zero right-hand side and $q=0$ exactly.

**Boundaries and restrictions.** Closed walls; a prescribed left velocity, constant (`set_x_inflow_outflow!` with a
profile) or time-dependent (`set_x_inflow_outflow!(model, t -> u)`); right: wall, prescribed or zero-gradient. Rejected
with wetting and drying: `periodic_x`, vegetation, k-epsilon, `bed_roughness_length`, `wind_stress_x`. Not applied: the
Froude limiter and the horizontal viscosity.

**Reference solutions** (`src/reference_solutions.jl`).
- Ritter (1892) dam break on a dry bed: $h=(2c_0-\xi)^2/(9g)$, $u=\tfrac23(\xi+c_0)$ for $-c_0<\xi<2c_0$,
  $\xi=(x-x_0)/t$, $c_0=\sqrt{gh_0}$.
- Carrier & Greenspan (1958) standing wave on a planar beach. Dimensionless with length $L$, elevation $sL$, velocity
  $\sqrt{gsL}$ and time $T=\sqrt{L/(gs)}$ ($s$ the slope; $x$ offshore, bed $-x$, $u$ offshore): from the Riemann
  invariants $u\pm2c-t$ ($c^2=x+\eta$), with $\sigma=4c$, $\lambda=2(t-u)$,
  $u=\phi_\sigma/\sigma$, $\eta=\phi_\lambda/4-u^2/2$, $x=\sigma^2/16-\eta$, $t=\lambda/2+u$, $(\sigma\phi_\sigma)_\sigma=\sigma\phi_{\lambda\lambda}$;
  the periodic solution is $\phi=AJ_0(\sigma)\cos\lambda$ (non-breaking for $A<1$; period $\pi T$). Points $(x,t)$ are mapped to
  $(\sigma,\lambda)$ by Newton's method; the shoreline is $\sigma=0$. Bessel functions by the trapezoid rule on their
  periodic integral representation (64 points, exponential convergence). The sign convention was derived here and is
  verified by the residual test below; equation numbers of the original were not checked.

**Order at the wet/dry front.** Both schemes are first order at the front, for three reasons: the depth has a kink there
(linear vanishing on a beach, quadratic in the Ritter rarefaction), so any limited reconstruction clips; in runup the
flow reverses at maximum runup and rundown (extremum of the velocity at the tip, switch of the upwind face depths); and
each column has one bed level, so the shoreline moves cell by cell with a staircase error of order $\Delta x\tan\beta$ in
elevation. The domain-integrated errors are hardly affected (O($\Delta x$) errors over an O($\Delta x$) region of small depth),
but the front position and the runup converge at first order, the same for `upwind1` and `tvd2` (solitary runup
0.80 ... 1.00 of the law for $\Delta x=0.2\ldots0.0125$ m, both schemes). A sloping bed within each column (sub-grid
bathymetry, Casulli 2009: $V_i(\eta)$ piecewise quadratic, still convex and non-decreasing, so the Newton solve and the
positivity proof carry over; Casulli & Zanolli 2012 for the iteration) would move the shoreline continuously and is the
candidate to reduce the staircase error.

References: Casulli & Cheng (1992), Int. J. Numer. Methods Fluids 15, 629-648; Casulli (1999), Int. J. Numer. Methods
Fluids 30, 425-440; Casulli & Zanolli (2002), Math. Comput. Model. 36, 1131-1149; Stelling & Duinmeijer (2003), Int. J.
Numer. Methods Fluids 43, 1329-1354; Brugnano & Casulli (2008), SIAM J. Sci. Comput. 30, 463-472; Casulli (2009), Int. J.
Numer. Methods Fluids 60, 391-408; Ritter (1892), Z. Ver. Deutsch. Ing. 36, 947-954; Carrier & Greenspan (1958), J.
Fluid Mech. 4, 97-109; Synolakis (1987), J. Fluid Mech. 185, 523-545; Casulli & Zanolli (2012), SIAM J. Sci.
Comput. 34, A2040-A2059; Delestre et al. (2013), Int. J. Numer. Methods Fluids 72, 269-300 (SWASHES). Equation numbers were
not checked.

## Test-by-test documentation

### test_verify_state_shapes

Purpose: verify staggered-array storage layout for $u$, $w$, $p$, $\rho$, and $\eta$.

Numerical basis: C-grid style staggering requires:

```math
u \in \mathbb{R}^{n_z \times (n_x+1)},\quad
w \in \mathbb{R}^{(n_z+1) \times n_x},\quad
p,\rho \in \mathbb{R}^{n_z \times n_x},\quad
\eta \in \mathbb{R}^{n_x}.
```

Validation section: shape consistency ensures discrete divergence/gradient operators are conformable.

### test_verify_grad_and_div_shapes_and_consistency

Purpose: verify centered gradient operators and combined divergence structure.

Reference equations:

```math
\nabla_h p = \left(\frac{\partial p}{\partial x}, \frac{\partial p}{\partial z}\right),
\qquad
\nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x} + \frac{\partial w}{\partial z}.
```

Validation section: for a linear $p(x)$, discrete $\partial p/\partial x$ is constant and $\partial p/\partial z=0$.

### test_verify_density_advection_zero_for_uniform_density

Purpose: confirm density advection operator vanishes for spatially uniform tracer.

Reference equation:

```math
\nabla\cdot(\rho\mathbf{u}) = \rho\,\nabla\cdot\mathbf{u} + \mathbf{u}\cdot\nabla\rho.
```

For constant $\rho$ and divergence-free velocity, RHS is zero.

Validation section: exact discrete near-zero tendency threshold is used.

### test_verify_density_advection_conservative_closed_box

Purpose: verify closed-box conservative transport of constituent mass.

Reference invariant:

```math
M_\rho = \int_\Omega \rho\, d\Omega
```

must remain constant with impermeable boundaries.

Validation section: compares $M_\rho^{n+1}$ vs $M_\rho^n$ at tight tolerance.

### test_verify_periodic_density_advection_conservative

Purpose: verify that density transport through the periodic seam conserves constituent mass.

Setup: a sinusoidal density profile in x, uniform velocity 0.15 m/s, `periodic_x=true`, schemes `upwind1` and `tvd2`.

Validation section: the domain integral of the advective tendency vanishes ($<10^{-10}$) and one transport step keeps
$M_\rho$ to round-off.

### test_verify_hydrostatic_pressure_monotone_with_depth

Purpose: validate hydrostatic pressure solver monotonicity and free-surface reference condition.

Reference equation:

```math
\frac{\partial p_h}{\partial z} = -\rho g,
```

with pressure fixed at the top cell in the implemented convention.

Validation section: checks top boundary and monotone increase with depth.

### test_verify_barotropic_solver_zero_flow_preserves_eta

Purpose: ensure barotropic free-surface solve preserves constant elevation when depth-integrated transport is zero.

Reference equation:

```math
\frac{\partial \eta}{\partial t} + \frac{\partial U}{\partial x}=0,
\quad U=\int u\,dz.
```

Validation section: if $U=0$, then $\eta$ remains unchanged.

### test_verify_model_step_conservation_and_bounds_short_run

Purpose: short-run system conservation and boundedness checks.

Checked invariants:

```math
\Delta V \approx 0, \quad \Delta M_\rho \approx 0, \quad \Phi_{\partial\Omega} \approx 0.
```

Validation section: includes lower/upper bounds against initial density extrema.

### test_verify_froude_diagnostic_behaviour

Purpose: verify finite positive internal Froude diagnostic.

Reference scaling:

```math
Fr = \frac{U}{c_i}, \quad c_i \sim \sqrt{g' H}.
```

Validation section: positivity and finiteness in stratified flow state.

### test_verify_diagnostics_boundary_flux_closed_streamfunction_field

Purpose: check zero net normal flux for streamfunction-generated closed flow.

Reference identity:

```math
\mathbf{u} = \left(\frac{\partial \psi}{\partial z}, -\frac{\partial \psi}{\partial x}\right)
\Rightarrow
\nabla\cdot\mathbf{u}=0.
```

Validation section: integrated boundary-normal flux is near zero.

### test_verify_poisson_matrix_and_nonhydrostatic_zero_rhs

Purpose: verify Poisson operator structure and zero-RHS pressure-correction behavior.

Reference equation:

```math
\nabla^2 p' = \frac{\rho_0}{\Delta t}\nabla\cdot\mathbf{u}^{\ast}.
```

Validation section: matrix diagonals for boundary/topology and zero solution for divergence-free predictor field.

### test_verify_pressure_solver_spectral_matches_sparse_reference

Purpose: cross-validate spectral and sparse solvers.

Reference requirement: both methods should produce equivalent pressure gradients for same RHS.

Validation section: compares $\nabla p'$ components with tolerance.

### test_verify_periodic_pressure_spectral_matches_sparse_reference

Purpose: cross-validate the spectral and sparse pressure-correction solvers with `periodic_x=true`.

Validation section: random periodic velocities; both solvers give the same $\nabla p'$ (within $10^{-7}$), and the sparse
Poisson matrix couples the first and last columns with $1/\Delta x^2$ (periodic seam).

### test_verify_hydrostatic_pressure_vectorized_matches_reference_loop

Purpose: ensure vectorized hydrostatic integration matches loop baseline exactly.

Reference equation:

```math
p_k = p_{k+1} + \frac{1}{2}(\rho_{k+1}+\rho_k)g\Delta z.
```

Validation section: elementwise equality at tight absolute/relative tolerances.

### test_verify_buoyancy_sign_heavy_down_light_up

Purpose: confirm buoyancy sign convention.

Reference equation:

```math
b = -g\frac{\rho-\rho_{ref}}{\rho_{ref}}.
```

Validation section: heavy parcels produce negative buoyancy, light positive.

### test_verify_pressure_component_decomposition_matches_total_pressure

Purpose: ensure pressure decomposition consistency.

Reference relation:

```math
p = p_h + p'.
```

Validation section: reconstructed total pressure equals model stored pressure.

### test_verify_model_froude_limiter_short_run

Purpose: verify limiter constrains Froude to configured maximum.

Reference policy:

```math
Fr_{new} \le Fr_{max}.
```

Validation section: tracks maximum diagnostic Froude through run.

### test_verify_requested_domain_grid_dimensions

Purpose: enforce requested geometric discretization.

Validation section:

```math
\Delta x = L/n_x,\quad \Delta z = H/n_z.
```

### test_verify_requested_salinity_based_initial_densities

Purpose: enforce lock-exchange initial density partition from salinity-derived target values.

Validation section: left/right halves match configured $\rho_{left}$ and $\rho_{right}$.

### test_verify_initialize_rayleigh_taylor_layering_and_reset

Purpose: verify RT initializer layering and state reset, and switch-back lock-exchange initializer.

Reference setup: heavy-over-light with optional perturbation (Rayleigh-Taylor canonical).

Validation section: layer values and zero initial velocities/free-surface checks.

### test_verify_initialize_rayleigh_taylor_argument_validation

Purpose: input-domain safety checks for RT initializer parameters.

Validation section: invalid parameter ranges must raise `ArgumentError`.

### test_verify_free_surface_kinematic_bc_consistency

Purpose: enforce kinematic free-surface closure.

Reference condition:

```math
w_{top} = \frac{\partial \eta}{\partial t}.
```

Validation section: compares reconstructed top vertical velocity to finite-difference $\eta_t$ each step.

### test_verify_hydrostatic_continuity_and_surface_closure_residuals

Purpose: verify that the LEGACY fixed-grid hydrostatic branch (`variable_top_layer=false`, pinned explicitly since
section 13's variable top layer is the default for `use_hydrostatic=true`) preserves discrete continuity and
top-closure residuals at machine-precision scale.

Reference conditions:

```math
\frac{\partial u}{\partial x} + \frac{\partial w}{\partial z} = 0,
\qquad
w_{top} = \frac{\partial \eta}{\partial t}.
```

Validation section: tracks maximum domain divergence, top-row divergence, and kinematic mismatch over multiple steps, with strict residual thresholds. The analogous identity for the variable top layer (a different closure in the top row; see section 13) is `test_verify_variable_top_layer_closes_kinematic_and_volume_budgets`.

### test_verify_wind_forcing_closes_volume_and_kinematic_budgets

Purpose: verify that the wind forcing keeps the water volume and the kinematic surface condition closed in all three
free-surface treatments (hydrostatic with variable top layer, hydrostatic fixed grid, non-hydrostatic).

Setup: closed 10 m x 4 m basin, 40 x 20 cells, two layers at rest, wind stress 0.2 Pa, $\nu=10^{-3}$ m²/s (non-zero
diffusion at the predictor's wall faces), 200 steps.

Validation section: $|\sum\eta\,\Delta x|<10^{-12}$ m² and the kinematic mismatch $|w_{top}^{rec}-\eta_t|$ below
$10^{-10}$ m/s every step, and a non-zero set-up. Before the fixes of issue #4 the volume drifted by about $10^{-8}$ m² on
the fixed grid (wall faces of $u^{\ast}$ in the barotropic solve) and the mismatch was about $8\cdot10^{-6}$ m/s in all
three treatments (wind added after the solve); see sections 9 and 10.

### test_verify_long_run_conservation_and_froude_control

Purpose: long-run integration stability and conservation.

Validation section: volume/mass conservation and limiter compliance over extended steps.

### test_verify_periodic_kelvin_helmholtz_boundaries_and_conservation

Purpose: verify the periodic seam of the Kelvin-Helmholtz initial state and model steps (non-hydrostatic, sparse solver).

Validation section: after `initialize_kelvin_helmholtz!` the seam faces 1 and nx+1 carry the same velocity and density
and the full shear profile (not a wall with u = 0); they stay identical over three steps; constituent mass is conserved
to round-off; the Poisson matrix couples the first and last columns.

### test_verify_momentum_advection_zero_for_uniform_velocity_fields

Purpose: verify zero self-advection tendency for uniform velocity fields.

Reference equation:

```math
\mathcal{A}(\mathbf{u}) = (\mathbf{u}\cdot\nabla)\mathbf{u} = 0
```

for spatially uniform $\mathbf{u}$.

Validation section: upwind1 and tvd2 both near-zero on interior.

### test_verify_tvd2_momentum_flux_equals_transport_scheme

Purpose: with a constant velocity ($C_x=0.14$, $C_z=0.16$) the `tvd2` momentum tendency (`_advective_flux_divergence!` with
`dt`) equals the `tvd2` density tendency (`advection_term_rho` with the same `dt`) to round-off for a smooth field with a
step, i.e. both use the same flux-limited Lax-Wendroff scheme. Contrast: the semi-discrete momentum flux (`dt = 0`)
differs.

### test_verify_momentum_advection_conservative_in_closed_box

Purpose: ensure net momentum-advection tendency integrates to near zero in closed domain.

Validation section: integral of advection tendencies for $u$ and $w$ near zero for both schemes.

### test_verify_model_runs_with_tvd2_momentum_advection

Purpose: smoke/stability check for TVD2 branch with full step coupling.

Validation section: finite diagnostic metrics after integration.

### test_validate_baroclinic_pressure_gradient_accelerates_dense_to_right

Purpose: ensure pressure-gradient sign from baroclinic density contrast is physically consistent.

Reference relation:

```math
a_x = -\frac{1}{\rho_0}\frac{\partial p}{\partial x}.
```

Validation section: gradient sign and acceleration sign in expected direction.

### test_validate_short_time_dense_front_moves_right_with_low_damping

Purpose: verify short-time lock-exchange front propagation direction.

Reference expectation: dense underflow front moves into lighter-fluid side.

Validation section: rightmost density-threshold front position advances in +x.

### test_validate_wind_driven_double_gyre

Purpose: validate that a wind stress building up over a closed, stratified basin drives the double gyre of issue #4 in both
the hydrostatic and the non-hydrostatic mode: a clockwise cell in the light upper layer (downwind at the surface, upwind
just above the interface) and, through the interfacial shear stress, a counter-clockwise cell in the heavy lower layer
(upwind just below the interface, downwind along the bed) (Monismith 1986; Spigel & Imberger 1980).

Diagnostic: streamfunction of the mean $u$ over the middle half of the basin, averaged in time over $300\le t\le600$ s,

```math
\psi(z)=\int_0^z \bar u\,dz',
\qquad \psi<0 \text{ in the upper layer},\quad \psi>0 \text{ in the lower layer}.
```

Setup: closed 10 m x 4 m basin, 80 x 40 cells, two layers at rest (999 over 1001 kg/m³, interface at mid-depth), wind
stress ramped linearly from zero to 0.2 Pa over 20 s, 600 s, `tvd2` momentum and density advection, mixing-length eddy
viscosity ($l=0.2$ m, capped at $5\cdot10^{-3}$ m²/s), no Rayleigh damping, no Froude limiter. With the molecular viscosity
alone the wind momentum stays near the surface and no stress acts across the interface: the lower layer then only sloshes
with the internal seiche and $\psi$ in the lower layer changes sign with its phase. The time average removes the seiche,
which remains on top of the gyres.

Validation section: $\psi_{upper}<-0.005$ m²/s (simulated $-0.0184$ in both modes), $\psi_{lower}>0.0015$ m²/s ($+0.0054$
hydrostatic, $+0.0057$ non-hydrostatic), mean surface velocity above 0.02 m/s (0.045) and mean bed velocity downwind above
0.0015 m/s (0.0057, 0.0060). The interface jump (2 kg/m³) and the vertical density variance decrease (by 1.05 and
1.26 kg/m³; the density has no diffusivity, so this is numerical diffusion in the sheared, seiching interface, not a
physical entrainment rate). Constituent mass and water volume are conserved and the CFL number stays below the `tvd2`
limit 0.5.

The predecessor (Copilot, 120 s, molecular viscosity) asserted upwind flow in the bottom quarter, i.e. a single overturning
cell; this contradicts the counter-rotating lower cell and failed in both modes (bed flow +0.0017 and +0.0010 m/s). The
experiment `experiments/wind_shear/` runs the same physics in a 20 m basin at 400 x 160 cells.

### test_verify_steady_wind_profile_uniform_density_second_order

Purpose: verify the wind forcing and the viscous term against the exact steady wind-driven flow in a closed basin of
uniform density with a constant viscosity and a free-slip bed (`wind_steady_uniform` in
`experiments/wind_shear/cases.jl`, which gives the derivation). Far from the end walls

```math
\nu u''=G,\quad u'(0)=0,\quad \nu u'(H)=\tau/\rho_0,\quad \int_0^H u\,dz=0
\;\Rightarrow\;
u(z)=\frac{s}{2H}\left(z^2-\frac{H^2}{3}\right),\quad s=\frac{\tau}{\rho_0\nu},\qquad \frac{\partial\eta}{\partial x}=\frac{\tau}{\rho_0 gH}.
```

Setup: 40 m x 4 m basin, 40 columns, $\nu=10^{-2}$ m²/s, $\tau=0.2$ Pa, started from the exact state (u, $\eta$), 600 s,
mean u over $x=15$-$25$ m averaged over the last 200 s, $n_z=5,10,20$, both modes.

Validation section: observed order of the maximum profile error above 1.8 (observed 1.98 and 1.88; errors
$1.3\cdot10^{-4}$, $3.4\cdot10^{-5}$, $9.2\cdot10^{-6}$ m/s, which equal the midpoint-rule offset $s\,\Delta z^2/(24H)$ of
the discrete zero-flux condition) and surface slope within 1 % (observed 0.07 %). On finer grids the error levels off at
about $3\cdot10^{-6}$ m/s ($10^{-4}$ of the surface velocity), independent of the run length (not investigated further).

### test_validate_steady_two_layer_wind_double_gyre

Purpose: validate the double gyre against the exact steady two-layer solution for one constant viscosity
(`wind_steady_two_layer`): a parabola per layer, continuous velocity and stress at the interface, zero flux per layer.
With lower-layer thickness $a$, upper $b$, $\zeta=z-a$:

```math
u_{lower}=\frac{g_1}{2}\left(z^2-\frac{a^2}{3}\right),\quad
u_{upper}=\frac{g_2}{2}\zeta^2+g_1a\zeta+\frac{g_1a^2}{3},\quad
g_1=-\frac{s\,b}{2aH},\quad g_2=\frac{s(2H+b)}{2bH},
```

```math
\frac{\partial\eta}{\partial x}=\frac{\nu g_2}{g},\qquad \frac{\partial\zeta}{\partial x}=\frac{\nu(g_1-g_2)}{g'}.
```

Since $g_1<0$ the lower layer flows upwind at the interface and downwind along the bed (the counter-rotating cell); the
velocity does not depend on the density difference, only the interface tilt does.

Setup: as above with two layers (999 over 1001 kg/m³, interface at mid-depth, initially tilted with the exact slope as
a staircase of whole cells), $n_z=40$, 2400 s (at 600 s the internal seiche excited by the staircase is not yet damped),
averages over the last 600 s, both modes.

Validation section: profile error below 2 % of the exact surface velocity (observed 0.16-0.17 %), downwind bed flow,
upwind flow below the interface, surface slope within 2 % (observed 0.09 %) and interface slope within 3 % (observed
0.3 % hydrostatic, 0.5 % non-hydrostatic). The experiment `plot_wind_shear_analytic.jl` runs both cases from rest.

### test_verify_nonhydrostatic_branch_produces_pressure_correction

Purpose: ensure non-hydrostatic branch computes non-trivial pressure correction.

Reference equation: non-zero $p'$ expected when predictor divergence is non-zero.

Validation section: `nonhydro_pcorr_max > 0` and finite.

### test_verify_nonhydrostatic_solution_differs_from_hydrostatic

Purpose: regression guard that hydrostatic and non-hydrostatic trajectories diverge under same forcing.

Reference physics: non-hydrostatic pressure provides additional vertical acceleration dynamics.

Validation section: non-zero differences in velocity and free-surface states.

### Non-hydrostatic free surface (section 14)

### test_verify_nonhydrostatic_wall_return_closes_surface_budget

Issue #5 regression: initialize a discrete streamfunction circulation that rises at the right wall and turns along the
surface. For both pressure solvers, with and without the explicit flat-bed mask, check that the predictor wall velocities
stay zero, water volume is conserved, and the final velocities close continuity and the surface kinematic condition.
Imposing wall velocities only after the barotropic solve and pressure projection fails these checks: the solves must
see the same boundary transport as the final velocity.

### test_verify_nonhydrostatic_free_surface_responds_to_internal_forcing

Purpose: regression guard for the rigid-lid coupling fix (section 14). A lock exchange run in non-hydrostatic mode (purely
baroclinic forcing, $\eta$ starts at 0) must reach $\max|\eta|$ well above round-off, while the kinematic closure and interior
divergence stay at round-off (the reordering must not degrade existing continuity guarantees).

### test_verify_nonhydrostatic_surface_wave_period_matches_shallow_water_not_full_dispersion

Purpose: documents a known LIMITATION OF THE DEFAULT MODE (not a `test_verify_*`/`test_validate_*` physics validation): with the
legacy Neumann top condition, the non-hydrostatic branch does not produce dispersive surface-gravity waves (section 14). Pins the
numerically observed period of a small-amplitude, uniform-density standing surface wave (domain chosen so the shallow-water and
full dispersion relations differ by about 31%) to the shallow-water relation $T=2L/\sqrt{gH}$, confirming it does NOT match the full
relation $\omega^2=gk\tanh(kH)$ in this (default) mode. Fixed by the opt-in `nonhydro_surface_dirichlet=true` (section 15); see
`test_validate_nonhydrostatic_dirichlet_surface_wave_matches_full_dispersion` below.

### Genuine non-hydrostatic surface-wave dispersion (section 15)

### test_verify_poisson_matrix_top_dirichlet_matches_manufactured_solution

Purpose: verify the Dirichlet-top elliptic operator (`build_poisson_matrix(...; top_dirichlet=true)`) against an analytical
solution, both directly (bare matrix) and through the full `PressureCorrectionSolver`/`solve!` API.

Analytical solution: $p(z)=\tfrac{C}{2}(z^2-H^2)$ solves $p''=C$ on $z\in(0,H)$ with $p'(0)=0$ (Neumann bottom), $p(H)=0$ (Dirichlet
top, at the domain edge, matching the ghost-cell closure). Checked at two grid resolutions ($n_z=10,40$): the error must shrink by
more than 8x for a 4x refinement (2nd order predicts 16x). Also checks: the matrix is non-singular (Dirichlet removes the
pure-Neumann null space); `method="spectral"` with `top_dirichlet=true` raises `ArgumentError`; the legacy (`top_dirichlet=false`)
matrix is unaffected (row-1 gauge pin still present).

### test_verify_nonhydro_dirichlet_reference_pressure_removes_uniform_density_offset

Purpose: verify the `grad_p_z_reference` mechanism that removes `solve_hydrostatic_pressure!`'s full-density (not anomaly)
convention from what reaches the Dirichlet-top projection (section 15) — the second defect found while validating this feature.

Checks: `grad_p_z_reference` is all zero unless `nonhydro_surface_dirichlet` is set; for $\rho\equiv\rho_0$ it equals the exact
hydrostatic gradient $-\rho_0 g$ at every interior face; for a two-layer (internal-seiche) reference profile it is $x$-independent
(same in every column) but genuinely $z$-dependent; and, as a full-model regression guard against reintroducing the found defect, a
$10^{-4}\,\mathrm{m}$-amplitude surface wave's corrected non-hydrostatic pressure correction stays below 5 Pa (the uncorrected defect
gave $\sim3.7\times10^4\,\mathrm{Pa}$ for the same case).

### test_validate_nonhydrostatic_dirichlet_surface_wave_matches_full_dispersion

Purpose: physics validation that `nonhydro_surface_dirichlet=true` produces genuine surface-wave dispersion (section 15), as the
positive counterpart to `test_verify_nonhydrostatic_surface_wave_period_matches_shallow_water_not_full_dispersion` above (same domain,
$kH\approx1.57$, same $31\%$ shallow-water/exact gap).

Analytical solution: full linear dispersion relation $\omega^2=gk\tanh(kH)$, period from the same zero-crossing technique as the
section 13 seiche test. Validation: the numerical period is finite at two grid levels and its relative error against the exact
relation shrinks under refinement (not just small at one lucky grid); the finer grid's error is below 2% (observed $\sim0.4\%$); and
the finer grid's period sits closer to the exact relation than to the shallow-water one -- a decisive dispersion check, not merely
"less wrong than shallow water by some margin".

### test_verify_nonhydro_fixed_grid_uniform_density_stays_uniform

Purpose: the non-hydrostatic mode always uses the fixed grid. With a moving free surface (cosine perturbation, clamp bounds
widened) a uniform density stays uniform for `upwind1` and `tvd2` (top-row advective form, section 4). Contrast: the plain flux
form on the same velocities violates constancy by more than $10^{-6}\,\mathrm{kg/m^3}$.

### test_validate_rayleigh_taylor_linear_growth_rate

Analytical solution: a small interface displacement $a_0\cos kx$ between a heavy top layer (thickness $h_1$) and a light bottom
layer ($h_2$), released from rest, grows as $a(t)=a_0\cosh\sigma t$ with the inviscid growth rate

```math
\sigma^2=\frac{g\,k\,(\rho_{top}-\rho_{bottom})}{\rho_{top}\coth(kh_1)+\rho_{bottom}\coth(kh_2)}
```

(two layers between rigid horizontal boundaries; Chandrasekhar 1961, ch. X; the free surface acts as a rigid lid for this mode).
Setup: 8 m × 4 m, 200 × 80 cells, $\rho_{top}=1002.5$, $\rho_{bottom}=998.5\,\mathrm{kg/m^3}$, interface at $0.82H$,
$a_0=0.02$ m, $m=3$ ($k=2.36$ m⁻¹), `tvd2` momentum and transport. The interface height is taken from the heavy-fluid thickness
of each column and projected onto $\cos kx$; the numerical rate is $\mathrm{acosh}(a/a_0)/t$ at $\sigma t=2$ ($ka\approx0.2$,
still linear). Required: $0.8\lt\sigma_{num}/\sigma\lt1.05$. Measured: 0.84 (0.87 on 400 × 160 cells; the smeared numerical
interface lowers the rate and converges slowly); without the top-row advective form 0.61, so the test detects that defect.

### Variable-thickness top layer (section 13)

All tests below use `test_verify_*` (code verification against analytical identities or solutions); tolerances are named at the top
of the block (`TOL_IDENTITY`, `TOL_MODEL_IDENTITY`, ...).

### test_verify_hydrostatic_pressure_top_half_layer

Purpose: `top_half_layer=true` gives the exact hydrostatic pressure at layer centres.

Analytical solution (uniform density): $p_k=\rho_0 g\,\Delta z\,(n_z-k+\tfrac12)$; it equals the legacy field plus $\rho_0 g\Delta z/2$.

### test_verify_constituent_mass_with_variable_top_layer

Purpose: constituent mass with the top-layer thickness $\Delta z+\eta$.

Analytical solution (uniform density): $M=\rho_0\Delta x\sum_i(H+\eta_i)$.

### test_verify_variable_thickness_barotropic_reduces_to_constant_depth

Purpose: `solve_variable_thickness!` with the thickness of a flat surface reproduces the constant-depth `solve!`.

### test_verify_variable_thickness_barotropic_closes_volume_budget

Purpose: verify the derivation of the variable-thickness free-surface system.

Identity: with the final velocities $u^{n+1}=\hat u-g\Delta t\,\partial_x\eta^{n+1}$,

```math
\eta^{n+1}_i-\eta^n_i+\Delta t\,\frac{U^{n+1}_{i+1}-U^{n+1}_i}{\Delta x}=0,\qquad U_i=\Delta z\sum_{k\lt n_z}u_{k,i}+h^n_iu_{n_z,i},
```

for random $\hat u$ and a surface within $\pm30\%$ of the top layer; residual at round-off level.

### test_verify_top_layer_density_transport_conserves_mass_and_constancy

Purpose: kernel-level check of `advect_density_top_layer!` with $(u,w,\eta^n,\eta^{n+1})$ taken from a model step.

Identities: $\sum\rho h\Delta x$ is conserved for a random density; a uniform density stays uniform (residual $\sim10^{-13}$). The
constancy check caught a real defect during development (the horizontal flux of the layers below the top layer had been scaled by
$\Delta z$); mass conservation alone cannot detect it, because any scaling of a flux-form divergence stays conservative.

### test_verify_variable_top_layer_uniform_density_stays_uniform

Purpose: full-model version of the constancy property with a moving free surface (clamp bounds widened so that they cannot hide a
deviation). Both the variable top layer and the fixed grid (top-row advective form, section 4) keep a uniform density uniform.
Contrast: the plain flux form applied to the fixed-grid velocities of the same run violates it by more than
$10^{-6}\,\mathrm{kg/m^3}$ (the defect that motivated section 13).

### test_verify_variable_top_layer_closes_kinematic_and_volume_budgets

Purpose: lock-exchange run with a surface perturbation: top-layer kinematic residual at round-off level, water volume and
constituent mass conserved, and $w_{top}=(\eta^{n+1}-\eta^n)/\Delta t$.

### test_verify_top_row_advective_form_zero_for_uniform_u

Purpose: top-row advective-form correction. For a uniform $u$ and an arbitrary (divergent) $w$ the top-row tendency vanishes for
`upwind1` and `tvd2`, while the legacy flux form does not (contrast).

### test_verify_variable_top_layer_config_and_thickness_guards

Purpose: argument validation. `variable_top_layer=true` with `use_hydrostatic=false` and an invalid `top_layer_min_fraction` raise
`ArgumentError`; a surface that leaves the top layer stops the run with an error; `copy_config` carries the options.

### test_verify_variable_top_layer_seiche_period_and_convergence

Purpose: analytical solution and spatial convergence of the variable-thickness free-surface solve.

Analytical solution: linear standing wave in a closed basin of uniform density, $\eta=A\cos(\pi x/L)\cos\omega t$,
$\omega=\pi\sqrt{gH}/L$, $T=2L/\sqrt{gH}$. The period is taken from the zero crossing of $\eta$ in the first cell
($\omega t=\pi/2$). The discrete relative period error (period too long) follows $+(\pi\Delta x/2L)^2/6\approx$ 4.1e-3, 1.0e-3,
2.6e-4 for $n_x=10,20,40$; the test requires an observed order in $[1.6,2.4]$ on both grid pairs and an error below 2e-3 on the finest grid. The
amplitude is tiny ($A=10^{-4}$ m): a finite-amplitude wave carries a crossing-time shift of about $0.15\,A/H$ that would mask the
second-order error.

### Density transport schemes (section 4)

`test_verify_top_layer_density_transport_conserves_mass_and_constancy` (above) runs for both `upwind1` and `tvd2`.

### test_verify_transport_tvd2_convergence_order_smooth_advection

Analytical solution: with a uniform velocity $v$ the profile is translated, $\rho(x,T)=\rho_0(x-vT)$. A Gaussian ($\sigma=0.05$ m,
1 m closed box, far from the walls) is advected along $x$ and, separately, along $z$ at a fixed Courant number of 0.4 on
$n=100,200,400$ cells, so the L1 error measures the full space-time order. Required: `tvd2` order in $(1.8,2.6)$ on both grid
pairs (measured 2.4, 2.2); `upwind1` order in $(0,1.1)$ (measured 0.64, 0.77, still pre-asymptotic); `upwind1` error more
than 20 times the `tvd2` error on the finest grid (measured about 75).

### test_verify_transport_tvd2_conservative_bounded_and_constancy

A divergence-free cellular flow (from a streamfunction, zero normal velocity at the walls) at $C_x+C_z=0.45$ transports a
lock-exchange step for 200 steps WITHOUT clamping. The constituent mass is conserved to round-off, no new extrema appear (local
maximum principle), and the front is not smeared out (contrast check). A uniform density stays uniform, and the semi-discrete
tendency (`dt = 0`) of a uniform density vanishes.

### test_verify_model_runs_with_tvd2_transport_and_config_guards

Full-model runs with `transport_advection_scheme="tvd2"`, with and without the variable top layer: constituent mass conserved
and density within the initial bounds; the density differs from an `upwind1` run (contrast). An unknown scheme name raises
`ArgumentError`, `copy_config` carries the option, and the default is `upwind1`. The output file tag (`output_case_tag`,
used by `run!` and by `case_tag` in `experiments/common/helpers.jl`) names both schemes, e.g.
`hydro_mom-tvd2_trans-upwind1_dt0p0025_L30p0_H4p0_dx0p300_dz0p200`.

### Variable bed (issue #3)

`ModelConfig.bed_level` (m above the grid bottom, one value per column) defines a staircase bed on the z-level grid
(`BedGeometry`, `src/geometry.jl`). Each value must be a whole number of cells and leave at least two wet cells, so the
bed stays below the top layer; dry columns and `periodic_x` are not supported. The masks: wet cells; open u-faces
(both neighbouring cells wet, u may be non-zero) and active u-faces (at least one wet neighbour, walls with u = 0 that
take part in the stencils like the outer domain faces); open w-faces (above the bed) and active w-faces (plus the bed
face, w = 0). An explicit `bed_level`, also all zeros, uses the masked code paths; `nothing` keeps the legacy paths.

### test_verify_bed_geometry_masks_and_checks

Purpose: masks of an obstacle three cells high in two columns (lowest wet cell, wet-cell count, face bed levels, wall and
interior faces of the obstacle, bed face of w); a zero bed has no solid cells and all faces open; invalid input (not a
whole number of cells, fewer than two wet cells, negative, wrong length, periodic) raises `ArgumentError`;
`copy_config` copies the bed level; the plotting helper leaves solid cells blank (NaN) and passes flat-bed fields through;
a Float32 model builds the same masks (the alignment check uses the Float64 cell height, since 1.0/0.2f0 is not a whole
number in Float32).

### test_verify_bed_mass_and_surface_pressure_accounting

Purpose: with solid cells the constituent mass counts the wet cells only (an obstacle of 2 x 3 cells in a 12 m x 4 m
domain leaves 46.8 m² of water, mass 46,800 kg/m for 1000 kg/m³), unaffected by the unused density stored in solid
cells; with the variable top layer the surface elevation adds $\rho\,\eta\,\Delta x$ per column (48,000 kg/m for
$\eta=0.1$ m). The obstacle-force diagnostic of the experiment (`leo_obstacle_force`) includes the free-surface pressure
$\rho_0 g(\eta_{left}-\eta_{right})$, which the model applies separately from `state.p`: with $p=0$ and an elevation
difference of 0.2 m over a 0.6 m high obstacle the force is $\rho_0 g\cdot0.2\cdot0.6$.

### test_verify_invalid_bed_velocities_are_masked_before_advection

Purpose: arbitrary velocities on closed u-faces and on or below the bed (here 10 m/s, e.g. from an initializer that
does not know the bed) are removed at the start of the step, so one step gives the same state as from a clean start.

### test_verify_flat_bed_reproduces_legacy_results_hydrostatic

Purpose: an explicit zero bed runs through every masked code path (advection, diffusion and transport masks, velocity
masking, per-face barotropic depth, w reconstruction from the bed, mass-correction mask). After 200 lock-exchange steps
u, w, rho and eta must equal the legacy flat-bed run bit for bit, with and without the variable top layer.

### test_verify_raised_flat_bed_equals_shallower_domain_hydrostatic

Purpose: a uniform bed of 1 m (5 cells) under a 4 m domain must give the same lock exchange as a 3 m domain with 15
layers: velocities within 1e-10 m/s, densities within 1e-9 kg/m³, eta within 1e-11 m after 200 steps (round-off from
the summation order of the mass correction); u and w are exactly zero in and on the bed.

### test_verify_rest_state_over_obstacle_hydrostatic

Purpose: horizontally stratified water at rest over an obstacle (6 cells high) stays exactly at rest for 400 steps,
both top-layer modes. On the z-level grid the hydrostatic pressure at a level depends only on the water above, so the
obstacle walls produce no pressure-gradient error.

### test_verify_lock_exchange_over_obstacle_conserves_hydrostatic

Purpose: lock exchange over the obstacle for 2000 steps: water volume and constituent mass conserved, density within
its initial bounds in the wet cells, u exactly zero on all closed faces, and the current is running (contrast).

### test_validate_seiche_over_raised_bed

Analytical solution: linear standing wave over a uniformly raised bed of height d, period $T=2L/\sqrt{g(H-d)}$
($L=20$ m, $H=4$ m, $d=1$ m). Required: relative period error below 5e-4 on $n_x=40$ (measured 2.7e-4) and an
observed order in $(1.6, 2.4)$ between $n_x=20$ and 40 (measured 1.96). Same errors as the flat-bed seiche test, so the
per-face depth of the barotropic solve is consistent.

Non-hydrostatic projection over a bed: with solid cells the Poisson problem for $p'$ is solved on the wet cells only
(sparse method). A wet cell couples only to wet neighbours (zero-flux walls at the bed and the obstacle, as at the outer
walls), solid cells get $p'=0$, the gauge pin moves to the first wet cell and the mean removal runs over the wet cells.
The spectral solver needs a rectangular domain and is refused when solid cells exist. A zero bed keeps the rectangular
problem, so both solvers are unchanged there.

### test_verify_flat_bed_reproduces_legacy_results_nonhydrostatic

Purpose: zero bed in the non-hydrostatic mode reproduces the legacy results bit for bit, with the spectral and with the
sparse solver.

### test_verify_raised_flat_bed_equals_shallower_domain_nonhydrostatic

Purpose: as the hydrostatic version, through the masked Poisson problem (sparse solver): a 1 m bed under 4 m equals a
3 m domain to round-off.

### test_verify_rest_state_and_projection_over_obstacle_nonhydrostatic

Purpose: (1) stratified water at rest over the obstacle stays at rest to the round-off of the projection (below 1e-9 m/s
and m after 400 steps; measured about 4e-11). (2) Lock exchange over the obstacle for 2000 steps: the projected
velocities $(\hat u,\hat w)$ are divergence-free in every wet cell (below 1e-10 1/s against a velocity scale
$u/\Delta x\approx0.1$ 1/s; measured 8e-13), volume and mass are conserved, u is zero on all closed faces. (3) The
spectral solver raises `ArgumentError` with solid cells.

### test_verify_obstacle_energy_uses_wet_cells

Check `leo_energy` against uniform analytical kinetic energy and the exact integral of a constant density anomaly's
potential energy above the staircase bed. An arbitrary solid-cell density must not contribute. Velocities are averaged
to cell centres, matching the issue #5 diagnostic, not a staggered-face energy norm.

### test_verify_nonhydrostatic_lock_exchange_wall_splash_energy

Run the obstacle experiment and the same tank without the obstacle to t* = 12 on a 4 mm grid, with TVD2 momentum and
density transport, physical viscosity, no Rayleigh damping and no Froude limiter. Require kinetic energy below 1.05 times
released anomaly potential energy at every sample, top speed below sqrt(2 g' 2a), conserved volume and constituent mass,
bounded density, and round-off continuity and kinematic residuals. This covers the previously undocumented wall-return
phase; the 2 mm experiment sweep supplies the finer-grid check.

The wall-predictor enforcement added in PR #7 is essential: reversing just that ordering reproduces the reported
surface jet and artificial energy growth at 2 mm. The fixed top-cell geometry was not the cause of that defect.

### test_verify_masked_poisson_manufactured_solution

Manufactured solution on the wet cells above a raised flat bed of height $d$:
$p=\cos(\pi x/L)\cos(\pi(z-d)/(H-d))$ has zero normal derivative at the side walls, the bed and the top, and
$\nabla^2p=-\pi^2(1/L^2+1/(H-d)^2)\,p$ ($L=2$ m, $H=1$ m, $d=0.25$ m). The masked matrix of `build_poisson_matrix`
(gauge pin at the first wet cell, mean removal over the wet cells) reproduces it with maximum errors 5.1e-3, 1.3e-3 and
3.2e-4 on $32\times16$, $64\times32$ and $128\times64$ cells: observed orders 1.99 and 2.00 (required: 1.8-2.2).


### Vegetation tests (`test_vegetation.jl`, section 16)

- `test_verify_rigid_patch_drag_per_unit_volume`: kernel check, $-\tfrac12C_Da\,u|u|$ per unit volume with the
  canopy-top cell fraction and the patch-edge weights, random velocities of both signs, to round-off.
- `test_verify_patch_stiffness_must_be_positive`: zero, negative and NaN bending stiffness of a patch are rejected.
- `test_verify_emergent_drag_up_to_top_of_momentum_layers`: emergent rigid stem and patch with a non-flat surface, fixed
  and variable top layer: stem force $\tfrac12\rho_0C_Db\,u^2$ times the top of the momentum layers ($n_z\Delta z$, or
  $(n_z-1)\Delta z+h_{top}$ interpolated to the base), volume-integrated sink with the layer thicknesses equal to the stem
  force, and $\tfrac12C_Da\,u|u|$ per unit volume in the top row of a patch, all to round-off.
- `test_verify_stem_drag_equals_fluid_momentum_sink`: three flexible stems in a sheared profile; the momentum removed
  from the fluid ($\rho_0\Delta x\Delta z\Delta y\sum a$) equals the sum of the stem drags to round-off; the stems bend
  downstream and their drag stays below the rigid value.
- `test_verify_cantilever_small_deflection_second_order`: uniform load $q=\tfrac12\rho C_Db U^2$, $Ca=0.005$; tip
  deflection against $ql^4/(8EI)$, observed order 2.0 (errors $1.0\cdot10^{-2}$, $2.5\cdot10^{-3}$, $6.3\cdot10^{-4}$).
- `test_validate_flexible_stem_reconfiguration`: $R=F/F_{rigid}\to1$ for $Ca=0.1$; slope of $\log R$ against
  $\log Ca$ between $Ca=300$ and $3000$ within 0.02 of $-1/3$ (observed $-0.333$); buoyancy number $B=100$ keeps the
  stem nearly rigid at $Ca=10$ ($R>0.95$), without buoyancy $R<0.6$.
- `test_verify_elastica_against_independent_reference`: single stem in a uniform current against the continuous elastica
  integrated independently from the tip (RK4, 20000 steps, shooting on the tip angle; `elastica_reference_uniform`).
  In a uniform current the loads depend on $\theta$ only, so the tip angle is the only unknown; the physical root is the
  first sign change of $\theta(0)$ below $\theta_{tip}=\pi/2$. Cases $(Ca,B)=(1,0),(100,0),(1000,0),(300,30)$: order
  above 1.8 in the number of segments for $R$ and the tip position, $|R/R_{ref}-1|<3\cdot10^{-3}$ and tip error below
  $10^{-3}l$ at 80 segments (observed $5\cdot10^{-6}$ to $1.9\cdot10^{-3}$). Over $Ca=0.1\ldots10^4$ the model with 80
  segments agrees within 0.9 % ($B\le10$) and within 0.4 % for $B=100$ up to $Ca=3162$; beyond that the shooting
  reference is badly conditioned. The reference gives $R\,Ca^{1/3}=0.890$ at large $Ca$.
- `test_validate_flexible_canopy_against_coupled_reference`: flexible submerged canopy in a periodic channel
  ($h_v=H/2$, $a=4$ 1/m, $EI=2\cdot10^{-5}$ N m², constant $\nu$) against the independent coupled 1D reference
  `flexible_canopy_reference`: the continuous elastica in the velocity along the stem, iterated with
  $\nu u''=\tfrac12C_Da\cos^2\theta(z)\phi(z)u|u|-G$ until posture and profile are steady (converged: doubling its
  resolution changes $u$ by $3\cdot10^{-8}$ m/s). The model starts from the reference state and runs 1000 s (from rest it
  needs about 6000 s; at 3000 s the error was still $2\cdot10^{-4}$ m/s). Velocity error below $10^{-4}$ m/s and tip error
  below 1.5 mm at $n_z=20$, and both at least halved at $n_z=40$ (observed $5.4\cdot10^{-5}$ and $1.6\cdot10^{-5}$ m/s,
  tip (0.67, 0.60) mm and (0.16, 0.13) mm; surface velocity 0.098 m/s, deflected height 0.283 m).
- `test_verify_luhar_nepf_2011_published_postures`: the published values of Luhar & Nepf (2011) for their (and our) blade
  model: $l_e/l=0.30$, $h/l=0.61$ at Ca = 32, B = 0 (observed 0.2995, 0.6079); the zero-stiffness solution of their
  Eq. 12, a straight blade with $\sin\theta=(B^{-1}Ca)\cos^2\theta$ ($\theta$ = 58.9°, $l_e/l=0.138$, $h/l=0.517$ for
  $B^{-1}Ca=3.2$; published 59°, 0.14, 0.52), approached by our model at fixed $B^{-1}Ca$ as B grows: errors fall by more
  than 2.5 per decade of B (about $B^{-1/2}$, the stiffness boundary layer at the base), below 0.015 at $B=10^4$.
- `test_validate_luhar_nepf_2011_blade_drag`: laboratory data. Drag of flexible model blades in uniform flow (Luhar &
  Nepf 2011; HDPE stiffness-dominated, silicone foam buoyancy-dominated; their blade properties, C_D = 1.95, no
  fitting); forces read from their Fig. 4. Forces of at least 5 mN of the 25 cm HDPE and the 5 cm and 25 cm foam
  blades: median deviation below 10 % (observed 7 %), largest below 45 % (observed 39 %). The 5 cm HDPE blade, which
  the authors' identical model also overpredicts (short-plate drag coefficient), is left out. Also checks that the
  reconstructed parameters give Ca = 320 and B = 0.62 of their Table 1.
- `test_validate_emergent_rigid_canopy_uniform_velocity`: periodic channel, body force $G$; $U=\sqrt{2G/(C_Da)}$ exact,
  observed $4\cdot10^{-12}$, both modes.
- `test_validate_submerged_rigid_canopy_profile`: $h_v=H/2$, constant $\nu$; against the 1D reference
  `canopy_reference_profile` ($\nu u''=\tfrac12C_Da\phi u|u|-G$, 4000 cells, Newton); orders above 1.8 (observed 1.94,
  1.98), both modes.
- `test_verify_individual_stems_match_patch`: 25 rigid stems at random positions with the patch-equivalent lateral
  spacing give the patch velocity and balance the body force (observed $1.4\cdot10^{-6}$).
- `test_verify_flexible_patch_momentum_balance`: flexible submerged patch; steady total drag $=\rho_0GHL$ within
  $10^{-4}$, the stems bend, and the velocity exceeds the rigid case.
- `test_validate_patch_head_loss`: inflow/outflow channel, emergent patch; $g\Delta\eta=\tfrac12C_DaLU^2$ within
  0.3 % (observed $-0.03$ %), both modes.
- `test_verify_explicit_drag_stability_check`: $\Delta t\,K=2$ is rejected.
- `test_validate_wave_height_decay_over_rigid_canopy`: regular linear waves ($H_0/h=0.005$, $kh=0.2$) through an
  emergent canopy ($a=10$ 1/m) over 60 m, wavemaker at the left boundary, sponge at the right end. The wave height
  decays as $H_0/(1+\beta x)$ of Dalrymple, Kirby & Hwang (1984),
  $\beta=\tfrac{4}{9\pi}C_DaH_0k\,(\sinh^3k\alpha h+3\sinh k\alpha h)/((\sinh2kh+2kh)\sinh kh)$ (derived from the wave
  energy balance with the time-averaged drag dissipation). Compared: decay with canopy divided by decay without canopy,
  which removes the numerical damping of the fully implicit free surface ($\approx2\pi^2/N$ per period with $N$ steps per
  period). Within 1.5 % (observed 0.25-0.70 %). At larger amplitude ($H_0/h=0.05$, Ursell number about 50) the first
  harmonic loses energy to higher harmonics by nonlinear steepening, which the linear theory does not describe.
- `test_validate_canopy_lock_exchange_drag_regime`: lock exchange through a dense emergent canopy ($a=80$ 1/m,
  $H=0.2$ m, $g'=0.02$ m/s²): the front exponent decreases from the inertial value towards $x_f\propto t^{2/3}$ of the
  quadratic-drag regime (triangular current, pressure gradient $g'H/x_f$ balancing $\tfrac12C_Dau^2$); late exponent
  within 0.04 of 2/3 (observed 0.685).
- `test_validate_vegetation_damps_free_surface_wave_motion`: a rigid patch around a velocity antinode of a standing wave
  reduces the wave energy after 10 s to below 0.8 of the case without vegetation (observed 0.69).
- `test_validate_vegetation_slows_lock_exchange_front`: a rigid patch in the right half delays the dense front by more than
  1 m after 60 s.

### Turbulence tests (`test_turbulence.jl`, section 17)

- `test_verify_k_epsilon_decaying_homogeneous_turbulence`: no shear and no gradients; against the exact solution
  $k=k_0(1+(C_2-1)\varepsilon_0t/k_0)^{-1/(C_2-1)}$; order above 0.9 in $\Delta t$ (observed 0.93-0.98).
- `test_verify_implicit_vertical_diffusion_cosine_mode`: $\partial_t\phi=K\partial_z^2\phi$ with zero flux; the
  cosine mode decays as $e^{-K(\pi/H)^2t}$; first order in $\Delta t$; column integral conserved to round-off.
- `test_validate_k_epsilon_open_channel`: uniform flow over a rough bed (periodic channel, both modes): bed shear
  $=gSH$, total stress at mid-depth $=gS(H-z)$ and wall-function $k_1=u_*^2/\sqrt{C_\mu}$ within $10^{-4}$ (observed
  $1.2\cdot10^{-5}$ after 600 s), depth-mean velocity within 8 % of the log law $U/u_*=(\ln(H/z_0)-1)/\kappa$
  (observed +6.0 %, consistent with the wake of channel flows); $\nu_t$ peaks at mid-depth ($z/H=0.53$).
- `test_verify_k_epsilon_vegetation_balance`: homogeneous emergent canopy; $\varepsilon/P_v\to C_1C_{f\varepsilon}/C_2
  =0.9975$ within $10^{-3}$ below the two surface rows (observed 0.9975).
- `test_validate_k_epsilon_canopies_against_lopez_garcia`: runs 1, 9, 13, 17 of Dunn et al. (1996) (measured $C_{DB}$;
  flexible runs as rigid canopies of the deflected height) against the computed profiles of Lopez & Garcia (1997,
  Figs. 6.1, 6.2, 6.4, 6.5; digitized from the scan: frames and ticks located in the image, the thick k-epsilon dots
  separated from data symbols by size and fill). Velocity within 10 % at all heights (observed 0.92-1.03), stress peak
  within 8 % (0.97-1.05), stress below $0.8h_p$ within 40 % (0.68-1.35), depth mean 1.0-1.3 times the measured
  (1.05-1.20).
- `test_validate_k_epsilon_rigid_canopies_against_dunn_profiles`: laboratory data. The 12 rigid runs against the
  measured horizontally averaged velocity profiles (Appendix C of the report). Compared is the shape in the canopy,
  $u(z)/u(h_p)$, because the magnitudes and the stress above the canopy are affected by secondary currents: every point
  within 25 % (observed 21 %), median below 7 % (4.8 %); depth mean 1.05-1.35 times the measured (1.09-1.29). nz = 20
  (profiles within 3 % of nz = 40).
- `test_validate_k_epsilon_plant_density_series_against_lopez_garcia`: their numerical experiment (Figs. 6.16, 6.18;
  S = 0.0036, $h_p$ = 0.10 m, q = 1.12 m²/s, smooth bed), 1.6-160 stems/m², at their computed depth: bed shear ratio
  $u_{*b}/\sqrt{gHS}$ within 8 % (observed 0.98-1.00), Manning n within 30 % (observed 0.95-1.25, growing with density;
  see section 17), n increasing with density. The figures are labelled q = 1.21 m²/s; the text and the digitized H and n
  give 1.12-1.13 m²/s.
- `test_verify_turbulence_closure_selection`: default and legacy flag, state creation with the fixed top layer as the
  automatic default, invalid choices (unknown closure, bed geometry, explicit variable top layer, nz = 1) and a change
  of the closure across the k-epsilon boundary after construction rejected.
- `test_verify_clear_vegetation_resets_k_epsilon_production`: the vegetation production $P_v$ is zero after
  `clear_vegetation!`.

### Wetting and drying tests (`test_wetting_drying.jl`, section 18)

Exactness, conservation and positivity tests run for the four variants hydrostatic/non-hydrostatic x `upwind1`/`tvd2`
(suffix `_hydro_upwind1` etc.), each in 1D (nz = 1) and 2DV.

- `test_verify_wd_layer_thickness_kernel`: `layer_thicknesses!` gives non-negative layers summing to
  $\max(\zeta-z_b,0)$, reduces to the variable top layer for a flat bed, keeps the top layer open above the grid top and
  cuts the layers at bed and surface correctly.
- `test_verify_wd_configuration_checks`: unknown momentum scheme, `periodic_x` and a negative bed are rejected; one layer,
  a bed that is not a whole number of cells, the non-hydrostatic mode and `tvd2` are accepted.
- `test_verify_wd_lake_at_rest_{1d,2dv,2dv_stratified}_*`: still water over a beach rising out of the water, an island and
  a submerged bump (no level a whole number of cells), 500 steps: largest velocity and largest change of the level below
  $10^{-13}$ (observed exactly zero), dry cells stay dry; the stratified variant has a two-layer density in z.
- `test_verify_wd_volume_and_constituent_conservation_{1d,2dv}_*`: sloshing basin with beaches on both sides and two
  water masses, 1000 steps: water volume and constituent mass to $10^{-12}$ relative (observed $\le 9\cdot10^{-16}$),
  depth $\ge0$, density within its initial bounds, at least 5 columns change between wet and dry (depth above 1 mm).
- `test_verify_wd_depth_positivity_large_time_step_{1d,2dv}_*`: dam break onto a dry bed and beach at advective Courant
  number 6, where columns fall dry within the free-surface solve (more than one Newton iteration): depth $\ge0$ in every
  step, clamped round-off volume below $10^{-14}$ of the total, volume and mass conserved (`docs/depth_positivity.md`).
- `test_verify_wd_advection_depth_uniform_mms_{upwind1,tvd2}`: operator-level manufactured solution of section 18
  (depth-uniform $u$ under a surface crossing six interfaces): the maximum error over all layers converges with an
  observed order of at least 0.8 per grid halving and stays within twice the 1D `upwind1` error (observed: 2DV 1.2e-2 ...
  1.3e-3, 1D 7.8e-3 ... 9.7e-4 for $\Delta x=1\ldots0.125$ m).
- `test_verify_wd_negative_depth_is_rejected`: a level written below the bed stops the step with a `DomainError`.
- `test_verify_ritter_reference_solution`: the Ritter solution conserves the released volume (quadrature, $10^{-5}$), is
  $h_0$ behind the rarefaction and 0 ahead of the front, $h=4h_0/9$ and $u=2c_0/3$ at the dam.
- `test_verify_carrier_greenspan_reference_solution`: the hodograph solution satisfies the nonlinear shallow-water
  equations over the sloping bed (central differences, residuals below $10^{-6}$ of the term scales at nine points), the
  linear limit gives the shoreline amplitude $A/4$, points beyond the shoreline are dry.
- `test_validate_wd_ritter_dry_dam_break_1d`: $h_0=1$ m, $t=5$ s, $\Delta x=0.25$ m: relative L1 depth error $\le3$ %
  (observed 0.91 %) and converging (0.8 per halving; 1.42 % at 0.5 m); front ($h=1$ cm) within 12 % of the travelled
  distance (observed 11.3 %) and converging. WHY 12 %: the thin tip $h\propto(x_f-x)^2$ lags by first-order numerical
  diffusion and converges slowly (12.7, 11.3, 7.7, 5.0 % for $\Delta x=0.5\ldots0.0625$ m), while the depth converges at first
  order.
- `test_validate_wd_ritter_dry_dam_break_2dv`: 12 layers: the same bounds, depth within 1 % of the 1D run (observed equal
  to the printed digits), spread of the layer velocities $\le0.01$ m/s (observed 0.0013 m/s; hydrostatic theory: 0).
- `test_validate_wd_ritter_tvd2_improves_on_upwind1`: `tvd2` has a smaller L1 error and front lag than `upwind1`, in 1D
  (0.51 % / 5.6 % against 0.91 % / 11.3 %) and 2DV (0.54 % / 6.6 % against 0.91 % / 11.3 %); 2DV layer spread $\le0.01$
  m/s (0.0028 m/s).
- `test_validate_wd_ritter_dry_dam_break_2dv_nonhydro_{upwind1,tvd2}`: the non-hydrostatic mode within the same bounds.
- `test_validate_wd_carrier_greenspan_runup_1d`: $A=0.4$, slope 1:20, $L=20$ m, one period, forced by the analytical
  velocity at the offshore boundary: shoreline elevation within 10 % of the runup amplitude $A/4\,sL=0.1$ m at
  $\Delta x=0.05$ m (observed 9.2 %), converging (14.6 % at 0.1 m), surface elevation of the wet cells within 5 % (1.8 %).
- `test_validate_wd_carrier_greenspan_runup_2dv`: 15 layers, same bounds (shoreline 9.2 %, surface 2.1 %).
- `test_validate_wd_carrier_greenspan_runup_tvd2`: `tvd2` gives a smaller shoreline error than `upwind1` (14.3 against
  14.6 mm at $\Delta x=0.1$ m) and a surface error within 5 %; both are dominated by the wet/dry front, where `tvd2` falls back
  to first order.
- `test_validate_wd_carrier_greenspan_runup_2dv_nonhydro`: long wave ($kH\ll1$): the non-hydrostatic mode within the
  hydrostatic bounds.
- `test_validate_wd_nonhydro_standing_wave_full_dispersion_{upwind1,tvd2}`: closed basin, $kH=\pi/2$ (the shallow-water
  and the full period differ by more than 25 %): period error below 2 % on 80 x 40 cells (observed 0.89 %), smaller than
  on 40 x 20 (1.75 %), closer to $\omega^2=gk\tanh(kH)$ than to the shallow-water relation.
- `test_validate_wd_nonhydro_standing_wave_deep_water`: $kH=3$ (shallow-water period 42 % too short): period error
  below 3 % on 40 x 40 cells (observed 1.9 %). Over $kH=0.25\ldots3$ the phase speed is within 0.0-1.9 % of the exact one
  (`experiments/wetting_drying`, mode `dispersion`).
- `test_validate_wd_solitary_wave_runup`: non-breaking solitary wave, $H/d=0.019$, beach 1:19.85, against the maximum
  runup $R/d=2.831\sqrt{\cot\beta}\,(H/d)^{5/4}$ of Synolakis (1987): 1D within 10 % at $\Delta x=0.05$ m (observed 6.2 %) and
  smaller than at 0.1 m (11.2 %); 2DV non-hydrostatic (12 layers, `tvd2`) within 15 % at 0.1 m (observed 12.9 %). The
  runup converges at first order to the law (1.00 at 0.0125 m); it is set by the wet/dry front.

## Additional validation scripts and literature basis

The repository includes standalone validation scripts under `experiments/validation`:

- `validate_surface_dam_break_front_speed.jl`: compares gravity-wave front speed with $c=\sqrt{gH}$ before reflection.
- `validate_internal_seiche_period.jl`: compares dominant period with two-layer long-wave estimate.
- `validate_lock_exchange_front_speed.jl`: checks fitted lock-exchange front-speed coefficient against literature-informed range.
- `validate_wind_shear_transport_mixing.jl`: checks stress-velocity scaling and density-variance reduction.

These scripts extend unit/regression testing with case-level physics checks tied to reference solutions.
