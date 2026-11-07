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
| `test_dam_break.jl` | Stoker (wet bed) and Ritter (dry bed) dam breaks in layers, passive constituent (section 33) |
| `test_internal_seiche.jl` | internal seiche period against the two-layer relation, density extrapolation of the baroclinic pressure (section 34) |
| `test_kelvin_helmholtz.jl` | Kelvin-Helmholtz growth rate of a resolved tanh shear layer against the Taylor-Goldstein mode (section 35) |
| `test_plane_jet.jl` | turbulent plane jet with k-epsilon against the thin-shear-layer reference, free-jet spreading rate (section 36) |
| `test_nonhydro_multigrid.jl` | column-line multigrid for the non-hydrostatic wetting-drying pressure against the direct solver (section 37) |

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

For each advected component $\phi\in\lbrace u,w\rbrace$, the code computes conservative flux divergence:

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

Density is advanced by one forward-Euler step of conservative face fluxes $F_f=v_f\thinspace\rho_f$ on the C-grid velocities,
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
  $C=1$ in 1D, since MC lies in Sweby's TVD region $0\le\psi(r)\le\min(2r,2)$. The unsplit 2D update satisfies a local maximum
  principle for $C_x+C_z\le1/2$ (each face coefficient is at most $C(2-C)\le2C$). It has no corner-transport
  (cross-derivative) term, so for flow oblique to the grid the time error is $O(\Delta t\thinspace|uw|\thinspace\partial_{xz}\rho)$: first
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
The fixed-grid mass $\sum\rho\thinspace\Delta x\Delta z$ changes only by terms proportional to the top-row density anomaly (the uniform
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

The former kernel $\nu\thinspace\nabla^2u$ skipped the top and bottom rows: the top row had no viscous coupling to the row below,
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

and one-sided edge coefficients at boundaries. With the $\theta$ scheme (section 19) $\alpha$ carries the factor
$\theta^2$ and $\hat U$ is the transport of $\theta\hat u + (1-\theta)u^n$.

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
$-g\thinspace\partial_x\eta$ and the top-face velocity $w_{top}=\eta_t$ (section 9). Density and momentum advection set the flux through
the top face to zero. In flux form that is consistent only if $w_{top}=0$ (rigid lid). With a moving surface a zero top flux is
a source of $\rho\thinspace\eta_t\thinspace\Delta t/\Delta z$ per step in the top row, and $\rho\approx 10^3\thinspace\mathrm{kg/m^3}$ makes it large; this
was the original motivation for section 13 (see the wind-shear test note below). The surface moves in both modes: since the
reordering of section 14 the non-hydrostatic $\eta$ is driven by the momentum predictor, so the earlier assumption that the
non-hydrostatic projection keeps $\eta$ at round-off no longer holds. In the Rayleigh-Taylor experiment the source reached
1 kg/m³ per step (26 % of the density contrast) and lowered the linear growth rate to about 60 % of theory.
The fixed-grid density transport therefore advances the top row in advective form (`_density_top_row_advective_form!`, section 4),
which keeps a uniform density uniform; it is the $h\to\Delta z$ limit of the variable-top-layer budget below. The fixed-grid
momentum advection is unchanged.

`variable_top_layer=true` (hydrostatic mode; the default) lets the top layer follow the surface: layers $k\lt n_z$ keep $\Delta z$,
the top layer has $h=\Delta z+\eta$, and the free surface is a material boundary (zero flux through it). Per step:

1. Face thickness (old level): $h_{i+1/2}^{n}=\Delta z+\tfrac12(\eta^n_i+\eta^n_{i+1})$ (`top_layer_face_thickness!`).
2. Semi-implicit free surface (`solve_variable_thickness!`), same thicknesses in the right-hand side and the matrix:

```math
\eta_i^{n+1}-a_{i+1}\left(\eta_{i+1}^{n+1}-\eta_i^{n+1}\right)+a_i\left(\eta_i^{n+1}-\eta_{i-1}^{n+1}\right)=\eta_i^n-\Delta t\,\frac{\hat U_{i+1}-\hat U_i}{\Delta x},
```

```math
\hat U_i=\Delta z\sum_{k\lt n_z}\hat u_{k,i}+h^n_i\,\hat u_{n_z,i},\qquad
a_i=\frac{g\Delta t^2}{\Delta x^2}\left((n_z-1)\Delta z+h^n_i\right),
```

with $a=0$ at the boundary faces; the boundary-face values of $\hat u$ are set to the boundary condition first, so the solve sees the
transport the final velocities have. Then $u^{n+1}=\hat u-g\Delta t\thinspace\partial_x\eta^{n+1}$ in every layer, which gives
$\eta^{n+1}-\eta^n=-\Delta t\sum_k\partial_x(h_k u_k^{n+1})$ exactly.
3. Continuity: layers below the top layer as before, $w_{k+1}=w_k-\Delta z\thinspace(\partial_x u)_k$; the top layer closes its own budget

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

which preserves a uniform density exactly and conserves $\sum\rho h\thinspace\Delta x$ to round-off. The constituent mass, the mass
correction (section 11) and the diagnostics use the actual cell volumes.
5. Momentum: the top row uses the advective form, $\mathcal{A}(u)\leftarrow\partial_x(v u)+\partial_z(w u)-u\thinspace\nabla\negthinspace\cdot v$, with
$\nabla\negthinspace\cdot v$ built from the same face velocities and zero-flux boundary conventions as the flux form. It supplies the missing
$u\thinspace\eta_t/h$ term and makes a uniform $u$ advection-free. Below the top row the validated flux form is unchanged.
6. Hydrostatic pressure: the reference at the top-cell centre is the weight of the upper half layer, $p=\rho_{top}g\Delta z/2$
(`top_half_layer=true`), instead of 0, so the top row sees the baroclinic gradient $(g\Delta z/2)\thinspace\partial_x\rho_{top}$.

Limitations: the surface must stay inside the top layer, $\Delta z+\eta\ge$ `top_layer_min_fraction`$\cdot\Delta z$ (the run stops
with a clear error otherwise; there is no layer removal or wetting/drying). Cases with a large surface excursion relative to
$\Delta z$ must set `variable_top_layer=false` explicitly: the surface dam break (`initialize_surface_dam_break!`, $|\eta|$ set
directly by `eta_jump`) and any case forcing a large-amplitude free-surface wave (e.g. the vegetation wave-damping experiment,
amplitude of order 1 m against $\Delta z$ of order 0.1 m); see the `variable_top_layer=false` comments
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

Before this fix, the non-hydrostatic branch's free surface $\eta$ stayed pinned at round-off ($\sim10^{-13}\thinspace\mathrm{m}$)
regardless of real internal (baroclinic) forcing, effectively a rigid lid, even though the SAME barotropic solve (section 8) and
surface-slope correction (section 9) as the hydrostatic branch were being computed every step. Cause: in `step_forward!`, the
barotropic solve used to run AFTER the non-hydrostatic projection, on the projection's OUTPUT $\hat u$. The projection makes
$(\hat u,\hat w)$ exactly divergence-free cell-by-cell (Neumann-Neumann Poisson solve, section 7); summed over depth this forces
the transport divergence the barotropic solve sees, $\sum_k\partial_x\hat u_k$, to equal $-\hat w_{top}$ — but $\hat w_{top}$ is
never touched by the projection (it only corrects interior faces $2{:}n_z$) and still carries the value copied from the OLD
$w_{top}$ (the previous step's $\eta_t$) at the top of the function. The recursion $\eta^{n+1}\approx\eta^n+\Delta t\thinspace\eta_t^n$ is
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
$\max|\eta|\sim7\times10^{-4}\thinspace\mathrm{m}$ after 400 steps ($\Delta t=5\times10^{-4}\thinspace\mathrm{s}$), several orders of magnitude
above round-off, while the kinematic closure and interior divergence stay at round-off (the reordering does not degrade the
existing continuity guarantees). No other test in the suite changed outcome from this fix (full-suite regression check).

**Limitation of the DEFAULT (legacy Neumann) mode: non-hydrostatic surface waves are non-dispersive.** The fix above makes $\eta$
respond to real (baroclinic) internal dynamics, but a pure free-surface gravity wave — uniform density, so zero baroclinic
pressure gradient and zero buoyancy forcing — is governed only by the barotropic surface-slope force $-g\thinspace\partial_x\eta$, applied
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
$-g\thinspace\partial_x\eta$ (a REFERENCE-density, $\rho_0$-based, depth-UNIFORM approximation of the surface's dynamic pressure) applied
separately. For a linear surface wave, potential-flow theory gives the exact dynamic pressure
$p_{dyn}(z)=\rho_0 g\thinspace\eta\thinspace\cosh(k(z+H))/\cosh(kH)$: equal to the barotropic approximation $\rho_0 g\eta$ exactly AT the surface
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
harmless with the Dirichlet condition, where the absolute magnitude of $p'$ is now physically meaningful (no mean removal): empirically,
for a $\rho\equiv\rho_0$ test case the reference contribution alone gave $|p'|\sim3.7\times10^4\thinspace\mathrm{Pa}$, dwarfing the genuine
dynamic-pressure signal ($\rho_0 g\eta\sim1\thinspace\mathrm{Pa}$ for a $10^{-4}\thinspace\mathrm{m}$ wave) by four orders of magnitude — besides the
conceptual problem of applying the surface condition $p'=0$ to a quantity dominated by something that is not $p'$ at all.
Fix: `Model.grad_p_z_reference` caches $\partial_z p_{ref}$, where $p_{ref}$ is the hydrostatic pressure of the reference profile
(the initial state's column 1, matching `rho_0_w`'s reference, repeated across $x$ so it is exactly $x$-independent — its own
$x$-gradient is therefore already exactly 0 without correction); it is subtracted from `ws.grad_p_z` before the $\hat w$ predictor,
only when `nonhydro_surface_dirichlet` is active. After the fix, the same test case's $|p'|$ dropped to $\approx0.44\thinspace\mathrm{Pa}$,
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
$F/(\rho_0\Delta x\thinspace h\thinspace\Delta y)$, split linearly between the two neighbouring u-faces. Here $h$ is the layer thickness
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
from the vertical. Per unit vertical extent the horizontal drag is $\tfrac12\rho_0C_Db\thinspace u|u|\cos^2\theta$. Discretisation:
$N$ straight segments, angles at the segment midpoints, curvature at the nodes ($\theta_1=\kappa_0\Delta s/2$,
$\theta_m=\theta_{m-1}+\kappa_{m-1}\Delta s$, second order). The segments are split at the cell boundaries; a piece in row
$k$ uses the velocity of that row, and the same pieces load the stem and, with the opposite sign, the fluid (exact
momentum exchange; a rigid patch gives exactly $\tfrac12C_Da\thinspace u|u|$ per unit volume, $a=nb$). The posture is solved
with Newton's method (finite-difference Jacobian, backtracking), warm-started every step. If Newton fails (e.g. from an
upright stem at $Ca\gtrsim3000$), the load is applied in stages (velocity times $\sqrt{\lambda}$, $\lambda=1/n\ldots1$, with
$n=8, 32, 128$) from the start posture; postures that still do not converge are counted in
`VegetationModel.posture_failures`. Stem parts above the free
surface carry no load; the vertical drag component is not applied to $w$.

The drag enters the u predictor explicitly, from the velocities of time level n. It is monotone for
$\Delta t\thinspace K\le1$ with the linearised drag rate $K$ (rigid patch: $\tfrac12C_Da|u|$); `step_forward!` checks this.
WHY explicit: an implicit factor applied to the predictor before the free-surface solve would make the steady state
depend on $\Delta t$, because the surface-slope correction is not divided by it.

`ModelConfig.body_force_x` (m/s²) adds a uniform body force (channel slope) for steady periodic-channel flow.
`sponge_length` (m) and `sponge_rate` (1/s) add an absorbing layer at the right end: over the last `sponge_length` the
u predictor is multiplied by $1/(1+\Delta t\thinspace r\thinspace s^2)$, $s$ the relative position in the layer (quadratic ramp, implicit,
stable for any rate); used for the wave cases. The
former ad hoc vegetation viscosity (`vegetation_turbulence_strength`) and the old stem model (force relaxation 0.35,
response gain 8, `source_strength`, string-like bending) were removed; in the old tests that model changed the wave
amplitude by about $10^{-15}$ and the lock-exchange front by less than one cell.

### 17) k-epsilon turbulence closure (`src/turbulence.jl`, issue #8)

Selected with `turbulence_closure = "k_epsilon"`, on the fixed grid with a flat bed, with the variable top layer
(`variable_top_layer=true` explicitly; the automatic default stays the fixed top layer) and with wetting and drying,
which covers any bed and the cut cells at the surface (section 27, issue #30). The closure cannot be switched on or off
after the model is built. Standard high-Reynolds-number k-epsilon (Launder & Spalding 1974; Rodi 1993) on the cell
centres:

```math
\frac{Dk}{Dt}=\partial_z\!\left(\Big(\nu+\frac{\nu_t}{\sigma_k}\Big)\partial_z k\right)+P+G+C_{fk}P_v-\varepsilon,\qquad
\frac{D\varepsilon}{Dt}=\partial_z\!\left(\Big(\nu+\frac{\nu_t}{\sigma_\varepsilon}\Big)\partial_z\varepsilon\right)
+\frac{1}{T}\left(C_1(P+C_{f\varepsilon}P_v)+C_3\max(G,0)-C_2\varepsilon\right),\qquad \nu_t=C_\mu kT,\qquad
T=\min\Big(\frac{k}{\varepsilon},\ \frac{\alpha}{\sqrt6\,C_\mu|S|}\Big),
```

with $P=\nu_t[(\partial_zu+\partial_xw)^2+2(\partial_xu)^2+2(\partial_zw)^2]$, the buoyancy term
$G=-(\nu_t/\sigma_\rho)N^2$, $N^2=-(g/\rho_0)\partial_z\rho$ ($C_3=0$ stable, 1 unstable; Rodi 1987), and the vegetation
production $P_v$, the work of the drag per unit mass (from the u-face accelerations of `vegetation_source!` times $u$,
averaged to the cells; $\tfrac12C_Da|u|^3$ for a rigid canopy). Constants: $C_\mu=0.09$, $C_1=1.44$, $C_2=1.92$,
$\sigma_k=1.0$, $\sigma_\varepsilon=1.3$, $\sigma_\rho=0.7$; vegetation weights $C_{fk}=1.0$, $C_{f\varepsilon}=1.33$
(Lopez & Garcia 1997, WES Technical Report WRP-CP-10, Eqs. 26-27 and p. 58; adjustable per `KEpsilonState`).

**Realizability bound (issue #30).** The time scale $T$ is $k/\varepsilon$, bounded by $\alpha/(\sqrt6C_\mu|S|)$ with
$|S|^2=S_{ij}S_{ij}$ and $\alpha=0.6$ (Durbin 1996, Int. J. Heat Fluid Flow 17, 89-90; Medic & Durbin 2002). In
equilibrium shear flow ($P=\varepsilon$) $k|S|/\varepsilon=1/\sqrt{2C_\mu}=2.36$, below the bound 2.72, so the channel and
canopy results are unchanged (the open channel to all printed digits). Where the strain rises faster than
$\varepsilon$ can follow, the explicit production lets k grow by orders of magnitude in one step while $\varepsilon$
lags: in a wetting-drying dam break $k^2/\varepsilon$ gave $\nu_t=41$ m²/s at the front in the second step, against
$2.6\cdot10^{-3}$ m²/s with the bound (the k-epsilon stagnation-point anomaly, also expected in front of gates and
obstacles).

Numerics after each flow and density update: explicit first-order upwind advection (conservative, closed walls,
periodic or zero-gradient inflow), then per column a backward-Euler vertical diffusion with the sinks linearised
implicitly (Patankar: $\varepsilon/k^\ast\thinspace k$, $C_2\varepsilon^\ast/k^\ast\thinspace\varepsilon$, negative buoyancy); positive for
any $\Delta t$, first order in time. Bed: with `bed_roughness_length` $z_0>0$ a log-law quadratic friction
$c_b|u_1|u_1$ on the lowest row, $c_b=(\kappa/\ln(z_1/z_0))^2$, $z_1=\Delta z/2$, and the wall functions
$k_1=u_{\ast}^2/\sqrt{C_\mu}$, $\varepsilon_1=u_{\ast}^3/(\kappa z_1)$ (Rodi 1993); otherwise free slip and zero flux. Surface:
zero flux for $k$; for $\varepsilon$ the value $\varepsilon_s=C_\mu^{3/4}k_s^{3/2}/(\kappa\thinspace 0.07H)$ (Rodi 1993; default, as
in Lopez & Garcia 1997) or zero flux.
The momentum equation uses $\nu+\nu_t$ (u-faces: mean of the neighbouring cells; the mixing-length cap does not apply).
With `implicit_vertical_terms=true` the vertical part is implicit (section 19): the steady channel is the same to five
digits for $\Delta t=0.1$ and 2 s, the latter about 50 times the explicit limit $0.2\Delta z^2/\nu_t$ (test).
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
   $p'(z)=g\int_z^{\zeta}(\rho-\rho_0)\thinspace dz'$ at the *nominal* layer centres $z_k+\Delta z/2$, layers above the surface
   extrapolated with the surface-layer density. WHY nominal centres: a horizontally uniform stratification at rest then
   gives bit-identical $p'$ in neighbouring columns whatever their bed (exact lake at rest, also stratified).
2. Momentum predictor on the wet face layers: advection (below), baroclinic pressure, body force, Rayleigh damping, sponge,
   implicit vertical viscosity with zero stress at bed and surface.
3. Free surface in the volume formulation of Casulli (2009): $V(\eta)+T\eta=b$ with $V_i=\max(0,H+\eta_i-z_{b,i})$,
   $\sum_jT_{ij}\eta_j=a_{i+1/2}(\eta_i-\eta_{i+1})+a_{i-1/2}(\eta_i-\eta_{i-1})$, $a_f=g\Delta t^2H_f/\Delta x^2$,
   $b_i=V_i(\eta^n)-\frac{\Delta t}{\Delta x}(\hat U_{i+1/2}-\hat U_{i-1/2})$; Newton's method of Brugnano & Casulli (2008)
   from $\eta^n$. Every depth stays $\ge0$ for any $\Delta t$; the iteration ends after at most $n_x+1$ steps; a lake at rest
   has a zero residual and is not iterated (proof and conditions: `docs/depth_positivity.md`). In the validation cases
   the solve takes one iteration per step (the linear semi-implicit solve of Casulli & Cheng 1992).
4. $u=\hat u-g\Delta t\thinspace\partial_x\eta^{n+1}$ on the wet face layers; non-hydrostatic correction (below).
5. $\eta^{n+1}$ from the flux divergence (volume conserved to round-off); a depth below zero by more than the bound of
   `docs/depth_positivity.md`, section 5, stops the run (`DomainError`), as does a negative depth at the start of a step.
6. Vertical fluxes through the fixed interfaces from the layer continuity with the new thicknesses, integrated upwards
   from the bed.
7. Conservative transport of $\rho$ with the same discharges and cell volumes, in increment form
   $\rho^{n+1}=\rho^n+D/h^{n+1}$, $D=(h\rho)^{n+1}-h^{n+1}\rho^n$. WHY the increment: $(h\rho)/h$ changes $\rho$ by round-off even
   without flow, which drove an internal motion of $5\cdot10^{-9}$ m/s in a stratified lake at rest. Face values:
   `upwind1`, or `tvd2` = the flux-limited Lax-Wendroff value of section 4 with the Courant number of the cut cell
   $|q|\Delta t/(\Delta x\thinspace h_{up})$. Empty cells ($h<10^{-12}$ m) pass their increment to the thickest cell of the column.
   The scheme is monotone only if no cell loses more than its old volume in a step, which thin cut cells can violate;
   values outside the initial bounds are clipped and the clipped mass is redistributed over the cells with room for it
   (slack-weighted), so the mass stays exact. There is no global mass correction (section 11) in this mode.

**Momentum advection.** Momentum-conservative upwind form of Stelling & Duinmeijer (2003), per layer with the
discharges $q_{k,f}=h_{k,f}u_{k,f}$ and the interface fluxes $W$ of time $n$:

```math
\frac{du_{k,f}}{dt}=\frac{a_L(u^*_L-u)-a_R(u^*_R-u)}{\Delta x}+\frac{\bar W_{bot}(u^*_{bot}-u)-\bar W_{top}(u^*_{top}-u)}{\bar h_{k}},
```

with $u^{\ast}$ the value upwinded by the sign of each flux: the upwind velocity (`upwind1`; on an outflow side $u^{\ast}=u$, so
the form is a sum over the inflow sides with non-negative coefficients) or the flux-limited Lax-Wendroff value
$u^{\ast}=u_{up}+\tfrac12(1-c)\thinspace s_{up}$ with the MC slope $s$ and the Courant number $c$ of the flux
($|a|\Delta t/\Delta x$, $|\bar W|\Delta t/\bar h_k$) (`tvd2`, the same scheme as the momentum `tvd2` of section 3; the outflow
sides then contribute $(1-c)s_f/2$). The update is capped: with $C=\Delta t\sum$(inflow coefficients) the tendency is scaled
by $1/C$ where $C>1$, so for `upwind1` the update stays a convex combination (thin layers at the front are stable).

*Time correction (issue #66, found with the stratified mixing layer of issue #42).* Until issue #66 the `tvd2` value of this step was the plain MUSCL
extrapolation $u_{up}+s_{up}/2$ with forward Euler. Where the solution is smooth the MC slope is Fromm's,
$s_i=(u_{i+1}-u_{i-1})/2$, and the update of linear advection $u_t+au_x=0$ with $C=a\Delta t/\Delta x$ has the amplification
factor

```math
G(\theta)=1-C\left[(1-e^{-i\theta})+\frac{1-c}{4}\left(e^{i\theta}-1-e^{-i\theta}+e^{-2i\theta}\right)\right],
```

with $c=0$ for the plain extrapolation and $c=C$ with the time correction (Fromm's scheme). Without it $|G|>1$ for every
$C>0$: the semi-discrete operator is only weakly (third-order) dissipative, and forward Euler adds $\tfrac12(C\theta)^2$.
With it $|G|\le1$ for $C\le1$:

| $C$ | $\max_\theta\lvert G\rvert$, $c=0$ | at wavelength | e-folding | $\max_\theta\lvert G\rvert$, $c=C$ |
|---|---|---|---|---|
| 0.2 | 1.0048 | 8.9 $\Delta x$ | 207 steps | 1 |
| 0.4 | 1.044 | 5.8 $\Delta x$ | 23 steps | 1 |
| 0.6 | 1.153 | 4.6 $\Delta x$ | 7 steps | 1 |
| 0.8 | 1.342 | 4.0 $\Delta x$ | 3 steps | 1 |

The limiter stops the growth only at extrema, so in pure advection the plain extrapolation stays bounded. In the splitter
plate (two streams of 1016 and 1031 kg/m³, 0.3 and 0.5 m/s, $C=0.6$) a 4 $\Delta x$ wave in the weakly stratified upper
layer grew until the run failed after 75 s, hydrostatic and non-hydrostatic alike; at $C=0.4$, 0.3 and 0.2 it still grew,
more slowly. `upwind1` and the time correction remove it. The fixed-grid and variable-top-layer momentum `tvd2` had the
correction since issue #3 (section 3). Effect on the validation cases of this section: Ritter unchanged within 2 % (L1
0.51 % against 0.50 %), Carrier-Greenspan shoreline 14.05 against 13.88 mm (`upwind1` 14.04 mm).

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
   $\bar q_{k,L}/\bar h_k\approx u\thinspace r_k$ with $r_k-1\approx-\Delta x\thinspace\partial_x h_k/h_k$. For full layers $\partial_xh_k=0$; for
   the layer that the surface cuts, $\partial_x h_k=\partial_x\zeta$, so the relative error is $\Delta x|\zeta_x|/h_k$, which is
   O(1) wherever the cut layer is thinner than the surface drop over one cell, i.e. at every interface crossing, on every
   grid. At the worst face of the harness ($\Delta x=0.25$ m) the cut layer was 1.3 mm thick, $r=5.3$ and the tendency 5
   times the exact one. The present weights are the layer velocity times the column factor of the 1D scheme,

   $$a_C=\frac{\bar q_{k,C}}{\tilde h_{k,C}}\thickspace\frac{\tilde H_C}{\bar H_f},\qquad \tilde h_{k,C}=\tfrac12(h_{k,f'}+h_{k,f''}),\quad \tilde H_C=\sum_k\tilde h_{k,C},\quad \bar H_f=\sum_k\bar h_k,$$

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
cells, $w^{\ast}=w^n-\Delta t\thinspace\mathrm{adv}(w)$ (first-order upwind, advective form, capped); (ii) a Poisson problem for $q$ on
the wet cells below the surface cell of each column ($q=0$ in the surface cell, the dynamic surface condition, and in
empty cells): every such cell satisfies its continuity with $u=u_{fs}-\frac{\Delta t}{\rho_0}\partial_xq$ and
$w=w^{\ast}-\frac{\Delta t}{\rho_0}\partial_zq$ ($\partial_z$ over $\tfrac12(h_{k-1}+h_k)$, $w=0$ at the bed), which gives a symmetric
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
with wetting and drying: `periodic_x`, vegetation, `wind_stress_x`, the mixing-length model; `bed_roughness_length`,
Chezy friction and k-epsilon (section 27) need `implicit_vertical_terms=true`. Not applied: the Froude limiter and the
horizontal viscosity.

**Reference solutions** (`src/reference_solutions.jl`).
- Ritter (1892) dam break on a dry bed: $h=(2c_0-\xi)^2/(9g)$, $u=\tfrac23(\xi+c_0)$ for $-c_0<\xi<2c_0$,
  $\xi=(x-x_0)/t$, $c_0=\sqrt{gh_0}$.
- Carrier & Greenspan (1958) standing wave on a planar beach. Dimensionless with length $L$, elevation $sL$, velocity
  $\sqrt{gsL}$ and time $T=\sqrt{L/(gs)}$ ($s$ the slope; $x$ offshore, bed $-x$, $u$ offshore): from the Riemann
  invariants $u\pm2c-t$ ($c^2=x+\eta$), with $\sigma=4c$, $\lambda=2(t-u)$,
  $u=\phi_\sigma/\sigma$, $\eta=\phi_\lambda/4-u^2/2$, $x=\sigma^2/16-\eta$, $t=\lambda/2+u$, $\partial_\sigma(\sigma\phi_\sigma)=\sigma\phi_{\lambda\lambda}$;
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

### 19) Theta time integration and its stability range (issue #34)

The free surface is integrated with the $\theta$ scheme of Casulli & Cattani (1994), `ModelConfig(theta=0.7)` by
default: the surface level in the pressure gradient and the velocity in the continuity flux are taken at $n+\theta$
(layer thicknesses of level $n$),

```math
u^{n+1} = \hat u - g\Delta t\,\partial_x\bigl(\theta\eta^{n+1} + (1-\theta)\eta^n\bigr),
\qquad
\eta^{n+1} = \eta^n - \Delta t\,\partial_x \sum_k \Delta Z_k^n\bigl(\theta u_k^{n+1} + (1-\theta)u_k^n\bigr),
```

so the coefficient $\alpha$ of section 8 becomes $\theta^2 gH\Delta t^2/\Delta x^2$. The non-hydrostatic pressure is at
$n+\theta$ as well (incremental form), and the transport uses $u^{n+\theta}$ and the matching $w^{n+\theta}$. With
$\theta<1$ the stored $w^{n+1}$ no longer equals $\eta_t$ at the surface; the surface moves with $w^{n+\theta}$
(`ws.w_theta`). $\theta=1$ is the former scheme. Derivations, tests and decisions are in
`docs/plan_issue_34_theta_scheme.md`; the analysis script is `experiments/theta_scheme/stability_analysis.jl`.

**Allowed range.** There are two conditions, both from von Neumann analysis of the linearised 1D equations.

Gravity waves without mean flow: the amplification factor

```math
\lvert G\rvert^2 = \frac{1+\mu^2(1-\theta)^2}{1+\mu^2\theta^2},
\qquad
\mu = \frac{2\sqrt{gH}\,\Delta t}{\Delta x}\sin\frac{k\Delta x}{2},
```

is at most 1 for every time step if and only if $\theta\ge\frac12$. `ModelConfig` rejects $\theta$ outside
$[\frac12, 1]$ (`THETA_MIN`, `THETA_MAX`). The damping per wave period of $N$ steps is about $4\pi^2(\theta-\frac12)/N$;
$\theta=\frac12$ is second order and undamped.

Mean flow with the Froude number $\mathrm{Fr}=\lvert U\rvert/\sqrt{gH}$ and explicit momentum advection: stable for

```math
\mathrm{Fr} \le \mathrm{Fr}_{\max}(\theta) = \frac{2\theta-1}{2(1-\theta)}
\qquad\Longleftrightarrow\qquad
\theta \ge \frac{1+2\,\mathrm{Fr}}{2(1+\mathrm{Fr})},
```

returned by `theta_froude_max(theta)`. The explicit advection shifts the phase of a long wave by $Uk\Delta t$ without
damping, and the explicit share $1-\theta$ of the gravity term turns this shift into growth that the implicit share has
to absorb. The bound is sharp for Lax-Wendroff advection (the smooth limit of `tvd2`) with an advected surface, and
sufficient for `upwind1` and for fixed layer thicknesses. Above it, gravity waves on the current grow slowly (factor
1.04 per step at $\theta=0.55$, Fr = 0.45).

| $\theta$ | 0.5 | 0.55 | 0.6 | 0.65 | **0.7** | 0.75 | 0.8 | 0.9 | 1 |
|---|---|---|---|---|---|---|---|---|---|
| largest stable Fr | 0 | 0.11 | 0.25 | 0.43 | **0.67** | 1 | 1.5 | 4 | any |

`test_verify_theta_froude_bound_matches_linear_stability` recomputes the spectral radius for $\theta=0.55\ldots0.8$
and $C=0.1\ldots0.9$: it is at most 1 at $0.9~\mathrm{Fr_{max}}$ and above 1 at $1.1~\mathrm{Fr_{max}}$.

The default $\theta=0.7$ covers subcritical flows up to Fr = 2/3 and damps waves 2.5 times less than $\theta=1$. The
first default, 0.55, allows only Fr ≤ 0.11: a hydrostatic periodic channel at Fr = 0.45 (fixed layer thicknesses, `tvd2`,
$C=0.2$) grows by a factor of 36 in 50 s at $\theta=0.55$ and decays from $\theta=0.59$ on, the value of the analysis for
fixed layer thicknesses (measured with the momentum fluxes of issue #37, section 20; before, the advection defect
inflated the growth to about $10^7$). With fixed layer thicknesses `theta_froude_max` is conservative (the channel
still decays at 1.4 times the bound); it is sharp for an advected surface (variable top layer).
Physical damping (bed friction, turbulent viscosity) can hide the weak growth. Flows with larger Froude numbers need a
larger $\theta$ (for example $\theta=1$).

**Froude-bound warning (issue #39).** After every step `step_forward!` evaluates `surface_froude_number(model)`, the
largest $|U|/\sqrt{gh}$ over the u-faces. Here $U$ is the depth-averaged velocity and $h$ the actual water depth of the
face: bed and surface level included, or the cut face layers with wetting and drying. Faces shallower than
`SURFACE_FROUDE_MIN_DEPTH` = 1 cm (a shoreline film) are left out. The largest value of the run is kept in
`model.max_surface_froude` (diagnostics `"surface_froude"` and `"surface_froude_max"`). When it exceeds
`theta_froude_max(theta)` by more than `SURFACE_FROUDE_WARNING_MARGIN` = 0.01, the model warns once and names the
$\theta$ needed for that Froude number, `theta_for_froude(Fr)` $=(1+2\mathrm{Fr})/(2(1+\mathrm{Fr}))$.

The margin avoids two false alarms:
- at $\theta=1/2$ the bound is 0, so the orbital velocities of any small wave (Fr ≈ 1e-5) would warn;
- runs exactly at the bound would warn on round-off.

An excess of 0.01 needs $\theta$ at most 0.005 higher. In the test suite the warning comes from 48 runs with transient
fast flow:
- 35 wetting-drying tests (dam breaks and runup, Fr 0.68 to 1.2 at $\theta=0.7$);
- the $\theta=0.5$ and $0.55$ cases of the θ tests;
- a k-epsilon channel at Fr = 0.68.

They stay stable because the excursion is short and the upwind and limited advection is diffusive. The internal Froude number `froude_number` (with $g'$) is a different
quantity.

Tests:
- `test_verify_theta_for_froude_inverts_bound`: `theta_for_froude` inverts `theta_froude_max`.
- `test_verify_surface_froude_number_of_uniform_flow`: $U/\sqrt{gH}$ exactly in a periodic channel, the depth $H+\eta$
  with the variable top layer, the shallowest faces over a staircase bed, and the shoreline film left out with
  wetting and drying.
- `test_verify_froude_bound_warning`: Fr = 0.8 at $\theta=0.7$ warns once and names $\theta=0.72$. There is no warning
  below the bound, at $\theta=0.75$ or $\theta=1$, within the margin, or for Fr = 1e-5 at $\theta=1/2$.

**Non-hydrostatic $w^n$ with wetting and drying (issue #38).** The $w$ predictor of the wet/dry projection starts from
$w^n$. It used to be recovered from the interface fluxes of the step before,
$w^{n+1}=(w^{n+\theta}-(1-\theta)w^n)/\theta$. An error of this recursion is multiplied by $-(1-\theta)/\theta$ every
step: $-0.82$ at $\theta=0.55$, $-1$ at $\theta=1/2$. On the Carrier-Greenspan beach (15 layers, `tvd2`) a sawtooth in
time grew offshore, 10 m from the shoreline in 0.7 m depth. At $\theta=0.55$ the surface error reached 0.37 of the runup
amplitude $A$; at $\theta=1/2$ the run blew up. The hydrostatic runs were not affected.

Now $w^n$ has no history:
- at an interface between two cells that are wet at time $n$ it is the layer continuity of the discharges of time $n$,
  $w^n_{k+1/2}=w^n_{k-1/2}-\partial_x(h_f u^n)_k$, as on the fixed grid (there $w^{n+1}$ is the continuity of $u^{n+1}$).
  All cells below such an interface are full at $n$ and $n+1$ or cut by the fixed bed, so their thickness does not
  change and the integral is exact;
- at the other interfaces, which the surface rises through in this step, it is the interface flux of the last step, as
  at $\theta=1$ ($\theta=1$ is unchanged, bit for bit).

The continuity of $u^n$ at every interface was tried first. Where a bore lifts the surface through several layers in one
step, the cells below fill up, and the integral without their thickness change gave $w$ of metres per second: the
non-hydrostatic Stoker bore of section 33 blew up with 3 to 10 layers. With the restriction it is the same as before
(L1 and bore errors within 0.03 percentage points).

Surface error / $A$ after one period (480 x 15 cells, $\Delta t$ = 0.01 s):

| $\theta$ | 0.5 | 0.55 | 0.6 | 0.7 | 1 |
|---|---|---|---|---|---|
| hydrostatic, `tvd2` | 0.023 | 0.022 | 0.021 | 0.021 | 0.021 |
| non-hydrostatic, `tvd2`, before | blow-up | 0.37 | 0.021 | 0.021 | 0.021 |
| non-hydrostatic, `tvd2` | 0.021-0.024 | 0.021 | 0.021 | 0.021 | 0.021 |

Convergence at $\theta=0.55$ (15 layers, $\Delta t$ = 0.2 s/m $\cdot\,\Delta x$, $\Delta x$ = 0.2 to 0.025 m): before the
change the coarse grids were right and the error grew with refinement (0.040, 0.37 of $A$, blow-up at 0.025 m); now the
surface error converges at first order like the hydrostatic one (orders 0.92, 0.91; hydrostatic 0.93, 0.83, 0.84), the
shoreline error at order 0.6-0.7 in both modes. On the finest grid the non-hydrostatic surface error is 0.017 $A$
(hydrostatic 0.012): at $\theta=0.55$ short non-hydrostatic waves from the start are damped less (largest error
offshore at $t$ = 0.27 period; 0.015 $A$ at the shoreline with $\theta$ = 0.7 and 1), and they decay from period to
period (max $|w|$ 0.088, 0.069, 0.066 m/s over three periods).

$\theta=1/2$, non-hydrostatic, remains marginal: on 480 x 15 cells the surface error is 0.021 or 0.024 $A$ depending on
round-off ($\Delta t$ = 0.01 s or 0.010000000000000002 s), and on the finest grid max $|w|$ grows from 0.12 to
0.30 m/s between the first and the second period and then stays (0.31 m/s), while the hydrostatic run at $\theta=1/2$
stays at 0.045 m/s. The cause is not determined (a candidate is the explicit advection of $w$ in the non-hydrostatic
predictor, which the hydrostatic mode does not have); $\theta\ge0.55$ is the usable range of the non-hydrostatic wet/dry
step, the default 0.7 is unaffected. Figures: `experiments/wetting_drying/`, mode `theta`.

**Baroclinic coupling (issue #23).** With the density transported by $u^{n+\theta}$ and the baroclinic pressure taken
at time $n$, internal waves grow as $|\lambda|^2=1+(1-\theta)(\omega\Delta t)^2$ for every $\theta<1$ (hydrostatic
internal seiche on 160 x 160 cells). The predictor now extrapolates the density to $n+2(1-\theta)$
(`baroclinic_extrapolation_weight`); analysis, stability limit and results in section 34.

The issue also asked whether the shoreline needs local damping, since $|U|/\sqrt{gh}$ exceeds
$\mathrm{Fr}_{\max}(\theta)$ there for every $\theta<1$. The shoreline errors are the same for all $\theta$ (8.3-9.4 % of
$A$, hydrostatic and non-hydrostatic), so the thin film there needs none.

The explicit momentum advection had Courant-number limits of its own, independent of $\theta$ (`upwind1` from
$C=U\Delta t/\Delta x\approx0.3$ with 8 layers, `tvd2` already at $C=0.1$); they came from inconsistent discrete momentum
fluxes and were removed in issue #37 (section 20).

$\theta$ is fixed when the `Model` is built (the free-surface matrix contains $\theta^2$); `step_forward!` rejects a
changed `config.theta`. The check compares in the precision of the state, so a `Float32` model (which stores
`Float32(0.7)`) is not rejected spuriously (`test_verify_theta_range_is_checked`).

With `implicit_vertical_terms=true` the vertical viscosity, the upwind vertical momentum advection and the bed friction
form one tridiagonal matrix $A$ per u-face column (backward Euler, Casulli & Cheng 1992), and the free surface sees the
effective depth $\sum_k\Delta Z_k r_k$ with the column response $r=A^{-1}\Delta Z$ (`src/implicit_vertical.jl`). $A$ is in
flux form, so its columns sum to $\Delta Z$ and, without friction, the effective depth equals the water depth exactly.
With the variable top layer the top row of the momentum advection carries the advective-form correction
$-u_{\mathrm{top}}\nabla\cdot v$ (section 13). Its vertical part $u_{\mathrm{top}}W/h_{\mathrm{top}}$ stays in the
explicit predictor and is not put on the top diagonal of $A$: a diagonal term would change the effective depth by
$O(\Delta t W/\Delta z)$ (−3 % at $\Delta t W/\Delta z=0.1$ in a single column, up to 0.01 m in the test below). It is
divided by the top-layer thickness $h_{\mathrm{top}}$, as the implicit column divides its vertical flux. Until issue #14
it was divided by $\Delta z$, which gave a uniform flow with $W\ne0$ a spurious tendency (section 21). The explicit
term is stable for $\Delta t\lvert W\rvert/h_{\mathrm{top}}<2$, i.e. as long as the surface moves less than two top-layer
thicknesses per step, far beyond the accuracy range of the variable top layer.
`test_verify_implicit_variable_top_effective_depth_equals_water_depth` checks $\sum_k\Delta Z_k r_k=(n_z-1)\Delta z+h_{\mathrm{top}}$
on every interior face to $10^{-11}H$ in a standing wave with $\max\Delta t\lvert W\rvert/\Delta z\ge0.01$.

References: Casulli & Cattani (1994), Comput. Math. Appl. 27(4), 99-112; Casulli & Cheng (1992), Int. J. Numer. Methods
Fluids 15, 629-648; Casulli (1999), Int. J. Numer. Methods Fluids 30, 425-440; van Kan (1986), SIAM J. Sci. Stat.
Comput. 7, 870-891 (incremental pressure correction). Equation numbers were not checked.

### 20) Momentum advection with a mean flow (issue #37)

The momentum advection is in flux form, $\nabla\cdot(\mathbf v u)$ and $\nabla\cdot(\mathbf v w)$, on the control volumes
of the staggered points. It equals the advective form $\mathbf v\cdot\nabla u$ only if the face velocities of each control
volume satisfy the discrete continuity. Linearised around a uniform current $U$, the horizontal flux of $u^2$ carries
$2Uu'$; the vertical flux $Uw'$ must remove the extra $Uu'_x$. Two inconsistencies broke this and made the explicit
scheme unstable far below $C=1$:

1. On the fixed grid (`variable_top_layer=false`: every periodic and non-hydrostatic run) the grid top carries
   $w=\partial_t\eta$ but no momentum flux. The top row was advected at $2U$ by itself and a surface-wave perturbation at
   about $(n_z+1)U$. Now the top row uses the advective form $\nabla\cdot(\mathbf v u)-u\nabla\cdot\mathbf v$
   (`_top_row_advective_form!`) on both grids, as the density transport (section 4).
2. The vertical velocity at the u control volumes was interpolated via the layer centres,
   $(w_{k-1}+2w_k+w_{k+1})/4$. Now it is the C-grid average $(w_{k,i-1}+w_{k,i})/2$ (`_w_at_u_interfaces!`), and the
   horizontal velocity at the w control volumes is $(u_{k-1,i}+u_{k,i})/2$ (`_u_at_w_faces!`, non-hydrostatic). With
   the face velocities $(u_{i}+u_{i+1})/2$ of the cell centres the divergence of the u control volume is
   ```math
   \frac{(u_{i+1}+u_i)-(u_i+u_{i-1})}{2\Delta x}+\frac{1}{2}\Bigl(\frac{w_{k+1,i-1}-w_{k,i-1}}{\Delta z}+\frac{w_{k+1,i}-w_{k,i}}{\Delta z}\Bigr)=0,
   ```
   the mean of the continuity of the two adjacent cells (Arakawa & Lamb 1977).

Spectral radius of forward Euler for the linearised `upwind1` operator (periodic, fixed grid, no gravity waves,
`experiments/advection_stability/operator_spectrum.jl`):

| | C = 0.5 | C = 0.75 | C = 1 |
|---|---|---|---|
| before (any nz) | 1.154 | 1.521 | 2.000 |
| top-row form only, nz = 2 / 16 | 1 / 1.150 | 1.024 / 1.513 | 1.25 / 1.99 |
| both fixes (any nz) | 1 | 1 | 1 |

In the full model (periodic channel, `experiments/advection_stability/channel_scan.jl`) every case with nz = 2, 8, 16,
Fr = 0.2 and 0.5, `upwind1` and `tvd2`, hydrostatic and non-hydrostatic, $\theta=0.7$ and 1 now decays for
$C=0.1\ldots0.9$; before, `tvd2` grew from $C=0.1$ (up to blow-up) and `upwind1` with 8 layers from $C=0.3$.

The kinetic-energy excess of the non-hydrostatic lock exchange (section on
`test_verify_nonhydrostatic_lock_exchange_wall_splash_energy`) is not affected (1.026 at $\theta=1$, 1.061 at
$\theta=0.7$, as before): it is a start-up effect, see that test.

References: Arakawa & Lamb (1977), Methods Comput. Phys. 17, 173-265; Casulli & Walters (2000), Int. J. Numer. Methods
Fluids 32, 331-348. Equation numbers were not checked.

### 21) Sloped channel: Chezy friction, discharge and water-level boundaries (issue #14)

Channel of the issue: $L=1000$ m, $H=4$ m, $i_b=10^{-4}$, $q=4$ m²/s, $C=50$ m$^{1/2}$/s. The normal (uniform) flow has
$U=C\sqrt{Hi_b}=1$ m/s, so $q=UH=4$ m²/s: the data of the issue are consistent ($\mathrm{Fr}=0.16$).

**Chezy friction** (`ModelConfig(chezy_coefficient=C)`). The bed stress acts on the depth-averaged velocity of each
u-face,
```math
\frac{\tau_b}{\rho_0}=c_f\lvert U\rvert U,\qquad c_f=\frac{g}{C^2},\qquad U=\frac{1}{H}\sum_k\Delta Z_k u_k,
```
as a stress on the lowest open layer. Depth-integrated, the momentum balance contains the Chezy law exactly, whatever the
vertical viscosity, so uniform flow has $U=C\sqrt{Hi_b}$ and starts from rest as $U(t)=U_{eq}\tanh(t/T)$,
$T=C\sqrt{H/i_b}/g=1019$ s (`chezy_spinup_velocity`). With a constant viscosity $\nu$ the steady profile is the parabola
$u(z)=U+(gi_b/\nu)(Hz-z^2/2-H^2/3)$ (`chezy_viscous_profile`). Its bed value $U-gi_bH^2/(3\nu)$ is negative for
$\nu<gi_bH^2/(3U)=0.005$ m²/s here: the depth-averaged law needs a turbulent viscosity of the order $\kappa u_*H/6$
(0.017 m²/s here; the tests use 0.02 m²/s). On the layer grid the discrete steady state is that parabola at the layer
centres plus a constant, because central differences of a quadratic are exact.

- Explicit path (`implicit_vertical_terms=false`): forward Euler with $U^n$, stable for $\Delta t\,c_f|U|/H<1$.
- Implicit path (`implicit_vertical_terms=true`, also wetting and drying): $c_f|U^n|\,U^{n+1}$ on the lowest layer.
  This is a rank-one row $A'=A+e_1v^\top$ with $v=\Delta t\,c_f|U^n|\,\Delta Z/H$, solved by the Sherman-Morrison
  formula with one extra tridiagonal sweep $z=A^{-1}e_1$ (`implicit_vertical_column!`, keyword `gamma_mean`). $A'$ is no
  M-matrix, but its effective depth is $H/(1+\Delta t\,c_f|U^n|/H)>0$ (flux form, $\mathbf 1^\top A=\Delta Z^\top$),
  which is all that the free-surface matrix and the positivity of the wetting-drying solve need.
- Order in time of the spin-up: explicit 1, implicit 2. The product $|U^n|U^{n+1}$ is the geometric mean of the two
  levels, a midpoint-like approximation of $U^2$.
- `chezy_roughness_length(C, H)` $=H\exp(-(1+\kappa C/\sqrt g))$ (2.1 mm here) is the log-law roughness whose depth
  average gives the same $C$, for `bed_roughness_length` with k-epsilon, whose wall functions need $z_0$.
  `chezy_coefficient` excludes `bed_roughness_length` and k-epsilon.

**Discharge boundary** (`set_x_inflow_outflow!(model, shape; left_discharge=q)`). The profile is a shape, scaled every
step to $\sum_k\Delta Z_ku_k=q$ with the face layer thicknesses of time $n$: the variable top layer $\Delta z+\eta_1$, or
the cut face layers in the wetting-drying step. A velocity profile alone carries a transport that changes with the inlet
level; at the backwater level of the tests that is 10 % of $q$.

In the wetting-drying step $q$ can be a function of the model time, `left_discharge=t -> q(t)` (issue #61).

When `set_x_inflow_outflow!` is called again (e.g. to ramp $q$ on the fixed grid), the inlet velocity it sets is $u^n$ of the next step
and carries $(1-\theta)$ of that step's inflow. It is therefore scaled with the face layers that this step uses. In the
wetting-drying case these are the cut layers at the current inlet level. Scaled with the full layers of the flat grid,
a ramped inflow over the raised bed of the sloped channel settled 0.6 % too low ($U=0.995$ m/s).

**Water-level boundary** (`right_level=eta_R`; hydrostatic, `implicit_vertical_terms=true`, not periodic). The level
$\eta_R$ lies on the boundary face $x=L$, half a cell from the last centre. The velocity of the boundary face has its own
momentum equation:
- surface slope $(\eta_R-\eta_{n_x})/(\Delta x/2)$, implicit in the free-surface solve like an interior face, with
  $a_{n_x+1}=2\theta^2\Delta t^2gH_{n_x+1}/\Delta x^2$ on the diagonal and $a_{n_x+1}\eta_R$ on the right-hand side;
- its own implicit vertical column (viscosity, vertical advection, friction);
- baroclinic pressure gradient and horizontal diffusion of face $n_x$ (zero gradient);
- advection in an open control volume. Its outer face has the velocity $\tfrac32u_b-\tfrac12u_{n_x}$, that of a ghost
  cell with the $w$ of the last cell, and carries the zero-gradient flux $u_b$. This keeps the boundary control volume
  divergence-free, like the interior ones (section 20).

  With a closed outer face, or with the advection tendency of face $n_x$ copied to the boundary face, the normal flow
  was unstable at the boundary face. A perturbation of the top layer grew by about 0.012 1/s whatever the time step:
  the top-row correction of face $n_x$ depends on the $w$ of the last cell and so on the lower layers of the boundary
  face.

The top layer of the boundary face has the level $\eta_R$, both with the variable top layer and in the wetting-drying
step. In the wetting-drying Newton solve the Dirichlet coupling only adds diagonal weight, so $(P+T)$ stays an M-matrix,
and the flux divergence still reproduces $V(\eta)\ge0$ (docs/depth_positivity.md).

**Top-row vertical advection with implicit vertical terms** (fixed in this issue). With `implicit_vertical_terms=true`
and the variable top layer, two terms of the top row disagreed:
- the implicit column advects the top row vertically with $-Wu_{up}/h_{top}$;
- the explicit top-row correction added $+u_{top}W/\Delta z$.

A uniform flow with $W\ne0$ below the surface layer, i.e. any gradually varied flow, therefore had the tendency
$uW(1/\Delta z-1/h_{top})$. The backwater curve came out 0.1-0.2 % of the rise too high, growing with $n_z$
(3.3e-4 m with 8 layers, 5.9e-4 m with 16). The correction now divides by $h_{top}$ (`_top_row_advective_form!`, keyword
`h_top_vertical`), and the difference to the reference drops to 3e-5 m; 3e-6 m with a uniform velocity profile. The
remainder is the momentum-flux coefficient of the sheared profile, which the reference ($\beta=1$) neglects.

Limits:
- The variable top layer is accurate for $|\eta|\lesssim\Delta z$ (section 13). A 0.5 m level step at the outlet of the
  normal flow (nz = 10) sends a bore that reflects at the discharge inlet (a discharge boundary reflects waves) and
  reaches 0.82 m. The top layer is then three times $\Delta z$, and the run failed; the fixed grid and the
  wetting-drying step run the same case. The backwater tests therefore start from the reference profile with
  $|\eta_R|=0.3$ m $<\Delta z=0.5$ m.
- The density is transported through the open boundaries as before: the outer faces carry no density flux, and a
  uniform density stays uniform through the bounds clipping.

Results (`experiments/sloped_channel/run_cases.jl`, tables in its README):
- Normal flow is a steady state to round-off of the variable top layer, the fixed grid and the wetting-drying step;
  from 0.8 times the normal flow it returns to $10^{-13}$.
- Backwater (M1) and drawdown (M2) curves agree with the gradually-varied-flow equation (`chezy_backwater_depth`) within
  1e-5 to 8e-5 m (nx = 25 to 100), 0.003-0.03 % of the level change.
- With a real bed slope (wetting-drying cut cells; the bed falls 0.1 m, a quarter of a layer), the flow is uniform:
  with nx = 50 the depth is within 1.3e-5 m and the velocity within 2.2e-6 m/s of the normal flow, and the differences
  halve with each refinement.
- The log-law friction with `chezy_roughness_length` and k-epsilon gives a flow 4-5 % faster than the Chezy
  equilibrium (nz = 10 to 40). The conversion is an equivalent roughness within a few percent, not an identity.

References: Chow (1959), Open-Channel Hydraulics, ch. 5 and 9-10; Golub & Van Loan (2013), Matrix Computations, 4th ed.,
section 2.1.4 (Sherman-Morrison); Casulli & Cheng (1992), Int. J. Numer. Methods Fluids 15, 629-648. Equation numbers
were not checked.

### 22) Open boundaries for transcritical flow (issue #46)

The number of boundary conditions follows the characteristics $U\pm\sqrt{gh}$ (Fr = $|U|/\sqrt{gh}$):

| boundary | subcritical (Fr < 1) | supercritical (Fr > 1) |
|---|---|---|
| inflow | discharge (`left_discharge`) | discharge and level (`left_discharge`, `left_level`) |
| outflow | level (`right_level`) or free (`right_free`) | free (`right_free`) |

**Supercritical inflow** (`left_level`). The level $\eta_L$ of the inlet face replaces the level of the first cell in the
face thicknesses: the top layer is $\Delta z+\eta_L$ with the variable top layer, or the cut face layers of the
wetting-drying step. The discharge scaling then gives the inflow its depth and velocity. On the fixed grid every face
has the depth $H$, so `left_level` would do nothing there and is rejected.

**Free outflow** (`right_free=true`). The boundary face keeps the momentum equation of the water-level boundary
(section 21), with a level $\eta_b$ that comes from the interior (`_update_free_outflow_level!`):
- Supercritical outflow: linear extrapolation, $\eta_b=\tfrac32\eta_{n_x}-\tfrac12\eta_{n_x-1}$.
- Subcritical outflow: radiation condition for the outgoing long wave, with speed $c=U+\sqrt{gh}$, applied to the
  deviation from a surface parallel to the bed:
  ```math
  \frac{d\eta_b}{dt}+c\Bigl(\frac{\eta_b-\eta_{n_x}}{\Delta x/2}-s_{bed}\Bigr)=0,\qquad s_{bed}=\frac{z_{b,n_x}-z_{b,n_x-1}}{\Delta x}.
  ```
  It is backward Euler and implicit in the free-surface solve: with $\mu=c\Delta t/(\Delta x/2)$ the slope of the
  boundary face at $n+1$ is $w(\eta_{eff}-\eta^{n+1}_{n_x})/(\Delta x/2)$, where $w=1/(1+\mu)$ and
  $\eta_{eff}=\eta_b^n+\mu s_{bed}\Delta x/2$. That is the Dirichlet entry of the water-level boundary with the weight
  $w$, so the M-matrix property of the free-surface solve and the positivity of the wetting-drying solve stay as they
  are.

Two designs that failed, kept here because they look natural:
1. **The slope of face $n_x$ at the boundary face** (zero gradient of the slope, or $\eta_b$ relaxed to the level
   extrapolated to the face). The boundary face then moves exactly like face $n_x$, so the last cell keeps its level and
   an outgoing wave reflects as at a fixed level: 51 % left after the run, the same as `right_level`.
2. **The radiation condition with $\eta_{n_x}$ of time $n$.** The lag of one step is a whole cell at the gravity-wave
   Courant number 1, so 41 % was reflected at every resolution. Implicit, the reflected amplitude is 0.7 % ($\Delta x$ =
   10 m) and 0.55 % (5 m) of the incoming one (`test_validate_free_outflow_transmits_long_wave`).

Face bed at a boundary with a known level (water level, free outflow, `left_level`; wetting-drying step). The level of
such a face is the level at the end of the domain, so its depth is measured from the bed extrapolated linearly to
$x=0$ or $L$ (`_wd_face_bed`). The other faces keep the higher neighbouring bed. With the bed of the last column, which
is $i_b\Delta x/2=1$ mm higher on the #14 slope, the free outflow raised the whole channel by 1.7 cm: nothing pins the
level, and the outlet face was too shallow. The prescribed level of #14 hid this, and its sloped-bed case is now exact
as well ($U=1.00000$ m/s).

Results (`test_open_boundaries.jl`):
- **Supercritical channel** (body force, Fr = 1.6, $\theta=0.9\ge$ `theta_for_froude(1.6)` = 0.81): normal flow is
  steady to round-off with the variable top layer and in the wetting-drying step, and returns from 0.8 times the
  velocity.
- **Subcritical #14 channel with a free outflow:** no level is prescribed anywhere; discharge and friction set the
  normal depth, which is reached from 0.8 times the normal flow within $10^{-6}$. The volume budget with the boundary
  fluxes and the kinematic condition hold to round-off.
- **Real #14 slope with a free outflow:** depth within $3.4\cdot10^{-4}$ m of the normal depth.

Limits:
- **Neutral level on the fixed grid.** Every face there has depth $H$, so with a free outflow and no prescribed level
  the mean level is a neutral mode: velocities converge, a uniform level offset stays. Use the variable top layer or
  the wetting-drying step.
- **Steep real slopes.** At $i_b=0.01$ (Fr = 1.6) the supercritical flow over the cut cells deepens from the inlet state
  by up to 1 % along the channel. The flow is supercritical, so the outlet cannot cause this. The cause has not been
  identified; one candidate is the vertical advection in advective form of the wetting-drying step, through which flow
  parallel to the bed crosses the fixed layer interfaces with $W=Ui_b$. It matters for #47 (weirs).
- **A prescribed level at a supercritical outflow, or at a subcritical inflow, over-specifies the flow.** The model
  warns once per run when the boundary Froude number is beyond 1 ± `BOUNDARY_FROUDE_MARGIN` = 0.05.

References: Orlanski (1976), J. Comput. Phys. 21, 251-269 (radiation condition); Chow (1959), Open-Channel Hydraulics,
ch. 3 (critical flow). Equation numbers were not checked.

### 23) Flow over obstacles (issue #47)

Steady frictionless flow over a bed obstacle keeps the energy head $E=q^2/(2gh^2)+h+z_b$ except across a hydraulic
jump, which keeps the momentum flux. `src/reference_solutions.jl` provides:
- `critical_depth` $h_c=(q^2/g)^{1/3}$, the minimum $\tfrac32h_c$ of the specific energy;
- `bernoulli_depth`, the sub- or supercritical root by bisection;
- `conjugate_depth`, the Bélanger relation $h_2/h_1=\tfrac12(\sqrt{1+8\mathrm{Fr}_1^2}-1)$;
- `hydraulic_jump_position`, the first point downstream of the crest where the conjugate of the supercritical depth
  equals the subcritical depth of the outlet energy;
- `bump_flow_depth`, the depth profile for the three regimes.

The SWASHES jump lies at $x=11.6656$ m (Delestre et al. 2013).

Cases, all in the wetting-drying step with the obstacle as a real bed (cut cells), $\theta=1$ for the transcritical
cases and 0.75 for the subcritical case (`theta_for_froude` of the crest flow):

| case | 1D error, nx = 125 / 250 / 500 | 2DV (nz = 8), nx = 125 |
|---|---|---|
| bump, subcritical ($q=4.42$ m²/s) | 9.0 / 4.5 / 2.2 mm | the same as 1D |
| bump, transcritical ($q=1.53$ m²/s) | 22 / 12 / 6.1 mm | the same as 1D |
| bump, jump ($q=0.18$ m²/s), 0.5 m away from it | 10 / 5.4 / 2.9 mm | 18 mm ($\nu=10^{-3}$ m²/s) |
| jump position | 52 / 17 / 12 mm | about 0.15 m upstream |

The convergence is first order, as expected for the depth at a critical point and with the first-order wet/dry
geometry. In 2DV the hydrostatic jump forms a roller with backflow at the bed. With $\nu=10^{-4}$ m²/s it does not
become steady (the discharge varies by 3 %), because the roller is resolved by only one or two layers of the 8 cm
supercritical depth. With $\nu=10^{-3}$ m²/s it is steady, and the jump sits upstream of the depth-averaged one.

**Broad-crested weir** (trapezoid, $p=0.5$ m, $q=1$ m²/s). In free flow the model head reproduces the weir law
$q=(2/3)^{3/2}\sqrt g H^{3/2}$ within 1.25 % ($\Delta x=0.4$ m) and 0.45 % (0.2 m). The upstream energy is the same for
every tailwater below the modular limit; above it, it equals the tailwater energy, as frictionless theory says
($E_{up}=\max(E_t,\,p+\tfrac32h_c)$; figure in `experiments/obstacles`).

**Start of a frictionless transcritical run.** Without friction and without an outlet level, a lake carrying $q$ is
also a steady state. The free outflow keeps a lake as it is, because the radiation condition only lets waves out. The
crest-controlled solution therefore needs water only upstream of the crest at the start.

**Tracer** (`set_x_inflow_outflow!(...; left_rho)`, wetting-drying step). The face density of an inflow at the left
boundary is the prescribed $\rho_{in}$; otherwise the density of the first cell is used. $\rho_{in}$ widens the bounds
of the monotone transport (`_include_inflow_density_in_bounds!`), so a new extreme is not clipped away. Through all
three bump cases, 1D and 2DV, the content $\sum h\rho\,\Delta x$ changes every step exactly by
$\Delta t\,(q_{in}\rho_{in}-q_{out}\rho_{nx})$, and the values stay within $[\rho_0,\rho_{in}]$. The transport of the
other modes has no flux through the outer faces, so `left_rho` is rejected there.

Not covered: the sharp-crested (thin-plate) weir. Its curved nappe needs the non-hydrostatic pressure, and open
boundaries in the non-hydrostatic mode came with issue #49 (section 24); the weir case is still to be set up.

References: Delestre et al. (2013), SWASHES, Int. J. Numer. Methods Fluids 72, 269-300, section 3.1.1; Chow (1959),
Open-Channel Hydraulics, ch. 3, 14 and 15; Henderson (1966), Open Channel Flow, ch. 6 (weirs). Equation numbers were
not checked.

### 24) Open boundaries in the non-hydrostatic mode (issue #49)

The water-level outlet and the free outflow (sections 21 and 22) now also work in the non-hydrostatic mode. The boundary
face keeps its own momentum equation; the open boundary needs a condition for the non-hydrostatic pressure $q$. The
discharge inlet stays a prescribed velocity, with no correction ($\partial q/\partial x=0$).

**$\partial q/\partial x=0$ on the open face, fixed grid and wetting-drying step.** The boundary face gets no $q$
correction, so the pressure problem keeps the boundary conditions of a closed basin. On the fixed grid, both pressure
solvers (sparse and spectral) work, with the Neumann or the Dirichlet top (section 15).

**Why not $q=0$ on the face.** That was the first design: a Dirichlet row in the Poisson matrix, with the face
corrected by $(0-q_{n_x})/(\Delta x/2)$. It reflects much more of a dispersive wave. Measured with a Gaussian wave of
width 1 m in 1 m depth leaving through a free outflow, as a fraction of its amplitude:

| condition on the open face | wetting-drying step | fixed grid, Dirichlet top | fixed grid, Neumann top |
|---|---|---|---|
| $q=0$ on the face | 10-29 % (widths 1-5 m) | 4.4 % | 0.25 % |
| $q=0$ one cell beyond | 8-21 % | | |
| $\partial q/\partial x=0$ | 1.15 % (2-5 % for widths 1-5 m) | 1.34 % | 0.23 % |

In the wetting-drying step, $q$ is zero in the surface cell rather than at the surface. On the fixed grid with the
Neumann top, surface waves do not disperse (section 14); there the wave reaches the outlet as a long wave, and both
conditions transmit it equally well. The first version of issue #49 was tested only in this mode, which hid the
difference. The test now checks that the reference wave has its dispersive trough.

The remaining reflection of a dispersive wave (1.2-1.3 %) comes mainly from the radiation condition of the free outflow.
It uses the long-wave speed $\sqrt{gH}$, while the dispersive components are slower; a component with phase speed $c$
is reflected by about $(\sqrt{gH}-c)/(\sqrt{gH}+c)$.

Results:
- **Uniform flow** in the #14 channel with a water-level outlet stays steady to $10^{-9}$ for 300 steps (fixed grid
  with both solvers and with the Dirichlet top, wetting-drying step). With the Neumann projection, $q$ holds the
  $x$-independent hydrostatic part of the reference density (about $1.7\cdot10^4$ Pa here); it exerts no force,
  also on the open face.
- **A dispersive wave** leaves through the free outflow with 1.34 % (fixed grid, Dirichlet top) and 1.15 %
  (wetting-drying step) of its amplitude reflected; a prescribed level reflects it completely.
- The volume budget with the boundary transports closes to round-off. On the fixed grid with the Neumann top, the
  kinematic condition holds in every column, which includes the divergence of the boundary column.
- Figures: `experiments/open_boundaries/`.

### 25) Relaxed free outflow (issue #54)

The prescribed level (section 21) reflects every outgoing wave completely, with inverted sign. The free outflow
(section 22) lets waves out, but no level is prescribed; on the fixed grid the mean level is then a neutral mode. The
relaxed free outflow lies in between: outgoing waves leave, and the level returns to a target $\eta_R$ on the time
scale $\tau$ (radiation plus nudging; Chapman 1985; Marchesiello, McWilliams & Shchepetkin 2001; Blayo & Debreu 2005).

`set_x_inflow_outflow!(model, P; right_free=true, right_level=η_R, right_relaxation_time=τ)`, with $\tau$ in s and
$\tau>0$. `right_free` together with `right_level` is still an error without $\tau$.

**Formulation.** The subcritical radiation condition of section 22 gains a relaxation term:

$$\partial_t\eta_b + c\left[\frac{\eta_b-\eta_{n_x}}{\Delta x/2} - s_b\right] = -\frac{\eta_b-\eta_R}{\tau}, \qquad c = U+\sqrt{gh}.$$

Backward Euler with $\mu = c\,\Delta t/(\Delta x/2)$ and $r=\Delta t/\tau$:

$$\eta_b^{n+1}\,(1+\mu+r) = \eta_b^n + \mu\left(\eta_{n_x}^{n+1} + s_b\,\Delta x/2\right) + r\,\eta_R.$$

This is the weighted Dirichlet condition of the free outflow, $\eta_b = w\,\eta_\text{eff} + (1-w)\,\eta_{n_x}^{n+1}$, with
$w=(1+r)/(1+\mu+r)$ and $\eta_\text{eff} = (\eta_b^n + \mu s_b\Delta x/2 + r\eta_R)/(1+r)$. Only these two numbers
change (`_update_free_outflow_level!`), so the free-surface solve, the wetting-drying Newton solve and the
non-hydrostatic mode carry over. The limits are:
- $\tau=\infty$ ($r=0$) gives the free outflow, bit for bit;
- $\tau\to0$ gives the prescribed level.

Supercritical outflow ignores the relaxation, because no information enters the domain there.

**Reflection.** For a wave $e^{i(kx-\omega t)}$ with $c=\omega/k$, the continuous condition gives

$$R = \frac{-1}{1-2i\omega\tau}, \qquad |R| = \frac{1}{\sqrt{1+4\omega^2\tau^2}}.$$

The condition is a low-pass filter. Periods much longer than $\tau$ are imposed by the target level, and shorter ones are
radiated. For a reflection below $\varepsilon$ at frequency $\omega$, choose $\tau \gtrsim 1/(2\omega\varepsilon)$.
For example, a tidal period (12.4 h) cannot be both prescribed and radiated. A wind wave or a seiche of a few minutes
leaves with less than 1 % reflection if $\tau$ is an hour (period 5 min: $\omega\tau = 75$, $|R| = 0.7$ %).

**Return of the level.** In a basin of length $L$ closed at $x=0$, the lowest mode follows from the boundary condition
with $u_b = -L\,\eta_t/h$ and $\eta_x(L) = L\,\eta_{tt}/(gh)$:

$$\eta_t + \frac{L}{c}\,\eta_{tt} = -\frac{\eta}{\tau}.$$

A level offset therefore decays at the rate $\lambda = \bigl(1-\sqrt{1-4(L/c)/\tau}\bigr)/(2L/c)$, which is $\approx1/\tau$ for
$\tau \gg L/c$.

**Results** (`test_open_boundaries.jl`, figure `experiments/open_boundaries/figures/open_boundaries_relaxed.png`):
- **Limits.** $\tau=\infty$ reproduces the free outflow bit for bit. With $\Delta t/\tau=10^8$ the result deviates from
  the prescribed level by $2\cdot10^{-9}\,A$.
- **Reflection.** A Gaussian long wave of width 50 m in 4 m depth, with $\tau$ = 2 to 50 s ($\omega\tau$ = 0.25 to 6 at the
  pulse frequency). The reflected record is compared with the prescribed-level record (R = −1, same grid propagation)
  filtered with $R(\omega)/(-1)$; the filter is causal, so a finite record is compared exactly. The difference is
  0.4–1.4 % of the amplitude, on the fixed grid and in the wetting-drying step, for reflected waves of 15–85 %. It does
  not decrease with resolution beyond the 0.7 % radiation error of the free outflow itself. In a 4 km channel (complete
  records), the spectral ratio $|R(\omega)|$ follows the theory over $\omega\tau$ from 0.005 to 10.
- **Return of the level.** With $\tau = 20\,L/c$, the fitted decay rate agrees with $\lambda$ within 0.2 % (fixed grid
  and wetting-drying step); within 0.5 % at $\tau=10\,L/c$. The free outflow alone keeps the offset.
- **Uniform flow** in the #14 channel with the target equal to the normal level ($\tau$ = 1000 s): steady to round-off.
  From 0.8 times the normal flow it returns within $10^{-9}$, including the level, also on the fixed grid without the
  variable top layer.
- **Non-hydrostatic.** The dispersive wave of section 24 with $\tau$ = 50 s reflects 1.60 % (fixed grid, Dirichlet top)
  and 1.40 % (wetting-drying step), against 1.36 % and 1.15 % for the free outflow. The relaxation adds a low-frequency
  tail of order $A\sigma/(c\tau)$.

### 26) Sluice gate (issues #48, #57)

`set_sluice_gate!(model, x; lower_edge, contraction=1.0)` puts a thin vertical gate on the interior u-face at $x$ (m).
It closes the water column above its lower edge $z_g$ (m above the grid bottom). The gate works only in the
wetting-drying step, where face layers can be partly open, and it can be moved between steps. `clear_sluice_gates!`
removes all gates.

**Geometry.** The face layer thicknesses (`_wd_face_thicknesses!`) gain an upper bound:

$$h_{f,k} = \max\bigl(0,\ \min(z_{k+1},\ \zeta_f,\ z_b + C_c(z_g-z_b)) - \max(z_k,\ z_b)\bigr),$$

with $z_b$ the face bed, $\zeta_f$ the upwind level and $C_c$ the contraction coefficient (`contraction`). Layers
above the edge are closed, and the layer cut by the edge is partly open, as a layer cut by the surface is. While the
upstream level lies below the edge, the gate changes nothing (bit for bit). An edge at the bed closes the face. Through
the face thicknesses the edge is also a solid boundary for the free-surface solve, the non-hydrostatic pressure problem
and the transport.

**Contraction.** The jet leaving a sharp-edged gate contracts to the depth $C_ca$ (vena contracta, $a=z_g-z_b$),
with $C_c\approx0.61$ for a vertical gate (Henderson 1966, ch. 6). The contraction comes from curved streamlines and a
non-hydrostatic pressure, so the hydrostatic step cannot form it: with $C_c=1$ its jet leaves with the full opening
(measured: $C_c=1.00$ from the upstream head). In the hydrostatic step `contraction` therefore narrows the flow area of
the face to $C_ca$. The non-hydrostatic step resolves the contraction itself and requires `contraction=1`.

**Advection at the gate face.** Stelling & Duinmeijer (2003) conserve momentum where the flow expands and the energy
head where it contracts. The momentum-conservative advection of section 18 uses the advective velocity $q/\bar h$ of
the neighbouring cell. Under a gate the face is much shallower than both cells (0.06 m under 0.35 m in the tests), so
this velocity is far too small. The face then gains too little kinetic energy, and the upstream head came out 28 % too
low in 1D (6 % in 2DV, where the layer of the opening is about as thick as the face). On a face that the gate cuts in
this step (`gate_active`), the flow accelerating into the face therefore uses the energy-head form, first-order upwind:

$$\frac{\partial u_f}{\partial t} + \frac{u_f+u_{f-1}}{2}\,\frac{u_f-u_{f-1}}{\Delta x} = -g\frac{\eta_R-\eta_L}{\Delta x},$$

which in steady flow is Bernoulli between the faces, $\tfrac12(u_f^2-u_{f-1}^2)=g(\eta_L-\eta_R)$ (likewise for a flow
from the right). Behind the gate, the momentum-conservative advection of the next faces gives the momentum balance of
the expanding jet (Borda-Carnot loss). Elsewhere nothing changes; at a smooth contraction (the bump of section 23) the
two forms differ by $O(\Delta x)$ only.

**Theory** (`src/reference_solutions.jl`, frictionless, specific discharge $q$, $y_c=C_ca$):
- free flow: $E_1 = y_c + q^2/(2gy_c^2)$ (`sluice_gate_free_head`), i.e. $q=C_ca\sqrt{2g(E_1-C_ca)}$ with the upstream
  energy head $E_1$ (the formula of the issue with the approach velocity included);
- modular limit: free flow as long as the tailwater depth $y_t$ is at most the conjugate depth of $y_c$
  (`sluice_gate_modular_limit`);
- submerged flow (`sluice_gate_submerged_head`): the depth $y_s$ at the vena contracta (jet below dead water) from the
  momentum balance $y_s^2/2+q^2/(gy_c)=y_t^2/2+q^2/(gy_t)$, and $E_1=y_s+q^2/(2gy_c^2)$.

**Results** (`test_sluice_gate.jl`; 20 m channel, $q=0.15$ m²/s, $a=0.1$ m, $\Delta x=0.2$ m, $\theta=1$; figures in
`experiments/sluice_gate`):

| case | upstream energy head against theory |
|---|---|
| free flow, $C_c=0.61$, 1D and 2DV ($n_z=8$) | +0.04 % |
| free flow, $C_c=1$, $q=0.08$ m²/s, $a=0.05$ m, 1D and 2DV | +0.03 % |
| submerged, tailwater 0.28 and 0.34 m, 1D | -0.012 % and -0.001 % |
| submerged, 2DV ($n_z=8$, $\nu=3\cdot10^{-3}$ m²/s) | -1.6 % and -4.8 % |

The discharge is the same at all faces within 0.13 %, and the volume budget closes to round-off every step. A closed
gate holds the water: the inflow fills the reach upstream, and the tailwater stays at rest.

The hydrostatic 2DV submerged flow is a wall jet under a recirculating roller, the counterpart of the roller under the
jump of section 23. With $\nu=10^{-4}$ m²/s it does not become steady. With $\nu=3\cdot10^{-3}$ m²/s it is steady for
the 0.28 m tailwater, while the discharge of the 0.34 m tailwater still varies by 8-17 % along the channel. Its upstream head lies
below the momentum theory, which assumes a uniform tailwater section. Recirculation behind obstacles is issue #50.

**Non-hydrostatic contraction.** With `contraction=1`, a 6 m channel and $\Delta x=\Delta z$, the coefficient
$C_c$ computed from the upstream head falls with resolution towards the classical value (time step as in the test, and 20 % smaller):

| grid ($n_x \times n_z$, $a/\Delta z$) | $C_c$ from the upstream head | the same, 20 % smaller $\Delta t$ | smallest depth behind the gate / $a$ (smaller $\Delta t$) |
|---|---|---|---|
| 120 x 12 (2) | 0.82 | 0.79 | 0.86 |
| 240 x 24 (4) | 0.76 | 0.71 | 0.75 |
| 480 x 48 (8) | 0.64 | 0.64 | 0.65 |

The coefficient from the upstream head is an effective discharge coefficient: it includes the energy lost at the gate.
On the coarse grids the jet dips only briefly behind the gate and then runs deeper than $C_ca$ (0.09 m at 240 x 24,
where $C_ca$ from the head would be 0.076 m), so part of the head is lost rather than turned into a thinner jet. On the
finest grid both measures agree (0.64 and 0.65). The steps (0.06-0.08 and 0.07-0.12) do not yet show a clean order,
and on the finest grid the discharge still varies by 4 % after 40 s; the coefficient is close to the classical 0.61,
but not converged.

The hydrostatic step on the coarsest grid gives 1.00.

**Hydraulic jump behind the gate (issue #57).** Without friction, a jump behind the gate has no stable position: the
jet keeps the depth $C_ca$, so its conjugate depth is the same everywhere. A lower tailwater pushes the jump out of the
reach, a higher one against the gate (submerged flow). With Chezy friction ($C$, m$^{1/2}$/s) the jet deepens downstream
and its conjugate depth falls, while the subcritical flow deepens upstream from the outlet. The jump stands where the
two meet, and that position is stable. Theory (`src/reference_solutions.jl`, horizontal bed or mild slope $i_b$):
- the jet, from $C_ca$ at the gate (H3 or M3 curve): the gradually-varied-flow equation written for $x(h)$,
  $dx/dh=(1-Fr^2)/(i_b-i_f)$ with $i_f=q^2/(C^2h^3)$, which stays regular up to $h_c$ (where $dh/dx$ is singular),
  integrated by Simpson's rule (`chezy_supercritical_distance`, inverted by `chezy_supercritical_depth`). On the
  horizontal bed $x=C^2/q^2\,\bigl(h_c^3(h-h_0)-(h^4-h_0^4)/4\bigr)$;
- the subcritical flow, from the outlet depth upstream (H2 or M2 curve, `chezy_backwater_depth`), and upstream of
  the gate from the free-flow depth $h_1$ of $E_1$ (Bernoulli to the vena contracta, frictionless);
- the jump where the conjugate depth of the jet equals the subcritical depth (`sluice_gate_jump_position`, bisection
  on the jet depth). On the horizontal bed both curves scale with $C^2$: without friction the jump moves away
  infinitely far.

**No bed friction on an active gate face.** The control volume of the gate face straddles the vena contracta, and the
friction there acts with the jet velocity on the face opening. That is a head loss $i_f\Delta x$, first order in
$\Delta x$, which the frictionless gate relation does not have: with $C=50$ the upstream depth came out 2.5 % high
($\Delta x=0.2$ m) and 1.2 % high ($\Delta x=0.1$ m). On an active gate face the Chezy and the log-law friction are
therefore switched off (`_wd_implicit_vertical!`); the jet feels the friction from the next face on.

**Results** ($C=50$ m$^{1/2}$/s, horizontal bed, 20 m channel, gate at 8 m, $q=0.15$ m²/s, $a=0.1$ m, $C_c=0.61$,
400 s; jump position in the model: where the depth rises through the mean of the jet depth and its conjugate):

| outlet depth (m) | jump, theory (m) | 1D, $\Delta x=0.2$ m | 1D, $\Delta x=0.1$ m | 2DV ($n_z=8$, $\Delta x=0.2$ m) |
|---|---|---|---|---|
| 0.155 | 15.01 | | | +0.09 |
| 0.16 | 14.64 | +0.14 | +0.07 | +0.11 |
| 0.17 | 13.82 | | | +0.16 |
| 0.18 | 12.93 | +0.08 | +0.04 | +0.25 |
| 0.19 | 12.02 | | | +0.36 |
| 0.20 | 11.10 | +0.07 | +0.03 | +0.05 |
| 0.21 | 10.21 | | | +0.43 |

In 1D the jump lies within 0.7 cells downstream of the theory, and the offset halves with $\Delta x$ (first order: the
sharp shallow-water jump is spread over a few cells, a real jump is about $6h_2$ long). The depths in 1D agree with the
gradually varied flow within 0.15 % in the jet and behind the jump, and 0.17 % upstream (0.08 % with $\Delta x=0.1$ m).
Since the jet depths include the friction, the implicit friction is accurate at this time step. In 2DV ($\nu=10^{-4}$
m²/s) the subcritical depths lie up to 0.7 % above the theory: the friction acts on the depth average of a sheared
velocity profile, whose momentum and energy coefficients exceed 1. With $\nu=3\cdot10^{-3}$ m²/s the depths agree
better (0.3 %), but the jump lies 0.3-0.6 m downstream. The 2DV jump lies 0.05-0.43 m (up to 2.2 cells) downstream of
the theory. The offset grows with the outlet depth, except for a dip at 0.20 m, whose cause was not investigated (all
runs are steady; the jump forms a roller over few layers, the recirculation of issue #50). The flow is steady in 1D and
2DV (discharge spread below $2\cdot10^{-3}$).

**Non-hydrostatic jump.** With `contraction=1` the jet contracts by itself, but on practical grids not to $0.61a$
(see above). The jump position depends strongly on the jet depth (with $C=35$ and a 0.15 m outlet it moves by about
1.4 m between $C_c=0.7$ and 0.8). The theory therefore starts at the vena contracta of the model (its smallest depth
within 1 m behind the gate): the comparison checks friction and momentum, not the contraction. Case: 8 m channel, gate
at 2.4 m, $C=40$ m$^{1/2}$/s, outlet depth 0.15 m, $\Delta x=\Delta z$, $\nu=10^{-4}$ m²/s; steady after about 100 s
(discharge spread $10^{-6}$ at 140 s):

| grid ($n_x\times n_z$) | vena contracta | jump, theory | jump (50 % of its height) | jet | 1.5 m behind the jump | midway to the outlet |
|---|---|---|---|---|---|---|
| 160 x 12 | 0.0996 m at 2.43 m (not contracted) | none: no supercritical reach | at the gate (2.7 m) | | | |
| 320 x 24 | 0.0884 m at 2.61 m | 3.12 m | 3.26 m | -0.9 % | +3.4 % | +3.4 % |

On 320 x 24 the jump (half its height, interpolated) stands 0.8 $h_2$ ($h_2=0.18$ m) downstream of the theory. It is
short, rising within 0.1-0.2 m, against about $6h_2$ for a real jump, whose turbulent roller the laminar non-hydrostatic
model does not resolve. Behind the jump the depths lie 3.4 % above the gradually varied flow, as in 2DV hydrostatic but
more: the velocity profile behind the jump is strongly sheared (wall jet under the roller), while the theory assumes a
uniform velocity. The contraction is the one of the free flow at this resolution: the vena contracta lies at 0.88 $a$,
while the upstream head gives an effective $C_c$ of 0.75, the difference being the energy lost at the gate. On 160 x 12
the opening spans two layers and the jet does not contract (1.00 $a$; effective $C_c$ 0.79 from the head); its conjugate
depth is then below the subcritical depth at the gate and the jump stands at the gate. (Before the time correction of the
`tvd2` momentum advection, section 18, the jet contracted to 0.88 $a$ on this grid as well, and the jump stood 1.4 $h_2$
behind the theory: an effect of the anti-diffusive time error.)

References: Henderson (1966), Open Channel Flow, ch. 6 (sluice gates: contraction, free and submerged flow); Chow (1959),
Open-Channel Hydraulics, ch. 17; von Mises (1917), Berechnung von Ausfluss- und Ueberfallzahlen, Z. VDI 61 (the
contraction coefficient 0.611 of a slot); Stelling & Duinmeijer (2003), Int. J. Numer. Methods Fluids 43, 1329-1354
(energy head in contractions, momentum in expansions); Chow (1959), ch. 9-10 (gradually varied flow) and 15 (hydraulic
jump). Equation numbers were not checked.

### 27) k-epsilon with wetting and drying and the variable top layer (issue #30)

**Wetting and drying.** The closure of section 17 runs on the cut cells of the new geometry $h^{n+1}$
(`_wd_k_epsilon_step!`, after the constituent transport). Wet cells are thicker than `WD_EMPTY_CELL_THICKNESS`; in a
column they run without gaps from the lowest wet cell (cut by the bed) to the top wet cell (cut by the surface).
- Production from the velocities of time $n+1$: cell-centred $u$ from the open faces of the step, $\partial_zu$ between
  the cell centres, and $2(\partial_xu)^2+2(\partial_zw)^2=4(\partial_xu)^2$. The interface fluxes $w$ of a cut cell
  include its volume change, so $(w_{k+1}-w_k)/h$ is no strain rate in a thin cut cell (it grows without bound as
  $h\to0$).
- Advection: conservative first-order upwind with the layer discharges and interface fluxes of the step, in the
  increment form of the density transport (a uniform field stays uniform while cells fill and empty).
- Diffusion, sources and sinks: per column backward Euler in finite volumes with the cell thicknesses and centre
  distances $(h_k+h_{k+1})/2$, Patankar sinks; positive for any $\Delta t$.
- Bed: the wall functions in the lowest wet cell at $z_1=\max(h/2,\,10z_0)$, the height of the log-law friction of the
  implicit column (section 18). Surface: $\varepsilon_s$ in the top wet cell with the water depth of the column.
- Momentum: $\nu+\nu_t$ at the interfaces of the implicit vertical column of each face (`implicit_vertical_terms=true`
  required), $\nu_t$ of a layer at the face from its wet cells. Density: implicit vertical diffusion with $\nu_t/\sigma_\rho$
  on the wet cells, conservative. Empty cells keep their k and $\varepsilon$ and have $\nu_t=0$.

**Variable top layer.** The top row has the thickness $h_t=\Delta z+\eta$ of its column. It enters the centre distances
($\partial_zu$, $N^2$, diffusion), the finite-volume diffusion of the top row, its vertical advection, the depth of the
surface condition and the density diffusion; $\partial_zw=-\partial_xu$ in the top row, as in the cut cells.

**Results** ($H=0.5$ m, $S=10^{-3}$, $z_0=1$ mm, 20 layers; reference: the periodic fixed-grid channel of section 17):

| case | result |
|---|---|
| wetting-drying open channel, 80 m (inlet discharge of the periodic equilibrium, outlet at the normal depth), at 60 m | velocity profile, $k_1$, max $\nu_t$ within $10^{-4}$ of the periodic channel (6 % in $\nu_t$ at 30 m of a 40 m reach: still developing; equal to four digits at 120 m of 160 m) |
| the same with $\Delta t=0.5$ s instead of 0.05 s (40 m) | within $10^{-3}$ |
| bed raised by 0.2 m (whole layers) | the flat channel shifted, to round-off |
| bed raised by 0.21 m (lowest wet cell 0.015 m) | depth-mean velocity $10^{-5}$, bed stress $6\cdot10^{-4}$ |
| variable top layer, grid 1 cm below the water level (top row 0.0345 m, $\Delta z=0.0245$ m), 80 m | depth-mean velocity $6\cdot10^{-5}$, $k_1$ $1.3\cdot10^{-3}$, max $\nu_t$ $3\cdot10^{-3}$ |
| dam break of a stratified column onto a dry slope, hydrostatic and non-hydrostatic, 40 s | volume and density mass to round-off; k, $\varepsilon$ positive; max $\nu_t$ $2.6\cdot10^{-3}$ m²/s |

In the dam break (`experiments/k_epsilon_wetting_drying`) the stratification is mixed almost completely by the overturning
flow: a density range of 0.8 kg/m³ of the initial 10 kg/m³ is left after 40 s with k-epsilon and 2.3 kg/m³ without a
closure, on a coarse grid ($\Delta x=0.1$ m, $\Delta z=0.05$ m) that does not separate turbulent from numerical mixing. The
bound removes the peak of $\nu_t$ in the first second and changes the end state little (0.7 kg/m³ without it). A seiche
in a closed basin is damped the same with and without k-epsilon (bed friction dominates), with the fixed and with the
variable top layer.

`KEpsilonState.realizability = false` switches the bound off for such comparisons. A cell of 10-50 µm that has just got
wet can briefly hold a density far from its neighbours (within the initial bounds, mass exact), with and without a
closure.

Limitations: k and $\varepsilon$ are advected first-order upwind (as on the fixed grid); first order in time. The
horizontal eddy diffusion of the wetting-drying step is in section 28.

### 28) Recirculation behind obstacles (issue #50)

**Horizontal eddy diffusion.** The wetting-drying step has the term $\partial_x(\nu_t\,\partial_x u)$ in the momentum
equation and $\partial_x(\nu_t/\sigma\,\partial_x\phi)$ for $\phi=k,\varepsilon$. It is second order in $\Delta x$ on the
uniform grid and explicit, so the coefficients are limited: the sum of the weights of $u$ to
`WD_HORIZONTAL_DIFFUSION_MAX` $=0.5$ per step (the limit of the explicit scheme), the flux of $\phi$ at a face to a
quarter of the smaller neighbouring cell. A closed neighbouring face has no flux. The turbulence update is conservative.
`KEpsilonState.horizontal_diffusion = false` switches it off. It changes neither the swings below nor the results of
section 27.

**Reattachment length.** `reattachment_length(u, h, xc, x_start)` is the position behind `x_start` where the dividing
streamline meets the bed: the last column in which the lowest value of $\psi=\int_0^z u\,dz'$ is negative, with the zero
of the bed velocity interpolated.

**Case.** Backward-facing step ST1 of Nakagawa & Nezu (1987), as simulated by Luo, Dolcetti, Stoesser & Tait (2023, JFM
966, A18): $S=2$ cm, depth 6 cm downstream, $q=0.00876$ m²/s, $Re=4380$, $Fr=0.19$, smooth bed. The LES gives
$x_R=6.15\,S$ and agrees with the measurements according to the paper; the measured table of 1987 was not available to
this work (it is not public), so 6.15 is the reference, not the measurement. The channel is 70 S long with the step at 40 S.

Discharge inlet, relaxed outflow ($\tau=20$ s), means over the last 30 of 90 s; std is that of the instantaneous
length. Equilibrium wall function of section 29 (the default since issue #62; with the local wall function of section
27 the lengths were 6.16, 6.99, 7.41 and 6.94). With the time correction of the `tvd2` momentum advection (issue #66,
section 18); before it the lengths were 6.07, 7.22, 7.73 and 7.18:

| grid ($\Delta x$, $\Delta z$) | non-hydrostatic $x_R/S$ | std | largest $w$ (m/s) |
|---|---|---|---|
| S/5, S/5 | 6.06 | 0.07 | 0.022 |
| S/10, S/10 | 7.21 | 0.05 | 0.017 |
| S/10, S/20 | 7.70 | 0.07 | 0.014 |
| S/20, S/10 | 7.17 | 0.03 | 0.028 |

The length moves with $\Delta z$ and $\Delta x$ matters less. The trend is upwards and away from the LES. The test grid
S/5 agrees with the LES within 2 %, but that is a cancellation of errors, not convergence. What holds is $x_R$ within
20 % of the LES on these grids except the finest $\Delta z$ (25 %); the test uses S/5 with this tolerance.

**Sensitivity to the bed cell** (local wall function of section 27; the treatment that removes it is in section 29).
In the table above the bed and the step top lie on layer interfaces. Shifting the bed against the layers by a fraction
of $\Delta z$ shaves the lowest wet cell to $(1-f)\Delta z$ at the same geometry and the same $\Delta z$:

| shift $f$ | step S/5: $x_R/S$ (std) | step S/10 | weir S/10 |
|---|---|---|---|
| 0 | 6.18 (0.00) | 6.99 (0.10) | 5.29 (0.08) |
| 0.25 | 6.06 (0.08) | 7.02 (0.04) | 5.32 (0.08) |
| 0.5 | 6.00 (0.00) | 7.11 (0.07) | 4.15 (0.05) |
| 0.75 | 6.29 (0.11) | 7.37 (0.33) | |
| 0.9 | 8.52 (1.15) | 12.47 (2.39) | |

Moderate shaving moves the step by about 5 %, a thin cell breaks it: 40-80 % longer and no longer steady. The weir,
whose sloped face has shaved cells at every shift, moves by 22 % at $f=0.5$. The wall functions are the suspect. The
lowest wet cell takes $k=u_*^2/\sqrt{C_\mu}$ and $\varepsilon=u_*^3/(\kappa z_1)$ as Dirichlet values with $u_*$ from the
local velocity and $z_1=h/2$:

- $u_*\to 0$ where the bed velocity changes sign, so $k$ of the bed cell is forced to about zero at reattachment. It is
  10 to several hundred times below the $k$ of the cell above, also without shaving.
- $\varepsilon$ of the cell follows $z_1$: at $x=5\,S$ on S/5 it is $3\cdot10^{-6}$ m²/s³ at full thickness and
  $2\cdot10^{-3}$ m²/s³ at $f=0.9$.
- The bed cell is below the log layer everywhere in the recirculation, on all grids: $z_1^+=2$ to 8 on S/5 at full
  thickness, below 1 at $f=0.9$. The drift of $x_R$ with $\Delta z$ can therefore not be assigned to one grid.

Section 29 separates the causes: the thin cell, not $u_*$ of the local velocity, breaks the step, and the
equilibrium wall function on a wall cell of at least $0.3\,\Delta z$ removes the sensitivity.

**The start-up mode.** Fixed discharge and fixed level make a quarter-wave mode of the reach, of period $4L/c=7.3$ s. An
impulsive start excites it; with the water level at the outlet, $x_R$ swings between 3.0 and 10.4 S and the swing decays
over about 40 s. The relaxed outflow removes it (`recirculation_boundary.png`): 6.05 to 7.25 S. Both give 7.0 S over
the 10-90 s window. Means over a shorter run are not of the steady state.

**Hydrostatic blow-up with a fixed level.** The hydrostatic run with a fixed level crashes although the non-hydrostatic
one does not. k-epsilon is not the cause (the run fails without closure too, and earlier). The cause has two parts:

1. *The discharge at the step swings.* Linear long waves on a mean flow $U$ ($c=\sqrt{gh}$): with $a=(c+U)\eta_+$ and
   $b=(c-U)\eta_-$ the amplitudes of the waves going down- and upstream, $q'=a-b$ and the head perturbation is
   $(g/c)(a+b)$, so a lossless step only scatters $a$ and $b$. The ends set the size per reflection:

   | end | condition | factor |
   |---|---|---|
   | inlet, velocity | $u'=0$ | $a/b=(c+U)/(c-U)=2.07$ ($Fr=0.35$) |
   | inlet, discharge | $q'=0$ | $a/b=1$ |
   | outlet, level | $\eta'=0$ | $b/a=-(c-U)/(c+U)=-0.68$ ($Fr=0.19$) |

   Per period the mode changes by a factor 0.68 with a discharge inlet and 1.41 with a velocity inlet: a prescribed
   velocity feeds it, because a high level at the inlet then carries more water in. This reach has the faster flow on
   the inlet side, so the gain wins. Measured with a velocity inlet: 1.3 to 1.6 per period, the same hydrostatic and
   non-hydrostatic. Friction and the head loss at the step are left out of the estimate.
2. *Explicit advection has a limit.* At the time step of the runs, $u=0.625$ m/s is Courant number 1 (section 20). Once
   the swing lifts the jet there, the run breaks within about 0.5 s ($w$ from 0.4 to 4 m/s and $u$ to 1.5 m/s, then the
   free-surface solve fails). The peak $u$ itself depends on $\Delta t$ near the limit: over the first 6 s of the
   impulsive start it is 0.49 m/s at $\Delta t/2$, about 0.52 at 0.8 $\Delta t$ and about 0.70 at 0.9 $\Delta t$, where
   $C_x$ stays near 1, and at the full step the run breaks. The combined number $|u|\Delta t/\Delta x+|w|\Delta t/\Delta z$
   is not the limit (it reaches 2.8 at 0.9 $\Delta t$ without a crash; the vertical terms are implicit), and neither are
   thin top cells (outflow Courant number 6 in the stable non-hydrostatic run).

Runs at S/10 with k-epsilon (Courant number from the largest $u$):

| pressure | inlet | start | outlet | $\Delta t$ | result |
|---|---|---|---|---|---|
| hydrostatic | discharge | impulsive | level | full | blows up at 3.6 s, u 0.63 m/s; the drawdown reflected at the outlet passes the step |
| hydrostatic | discharge | impulsive | level | 1/2 | stable, peak u 0.49 m/s; swing of $q$ from 1.68 to 0.01 q in 60 s |
| hydrostatic | discharge | ramp 10 s | level | full | stable, Courant 0.50; swing to 0.01 q |
| hydrostatic | velocity | ramp 10 s | level | full | swing grows, blows up at 26.7 s |
| hydrostatic | velocity | ramp 10 s | level | 1/2 | swing grows (0.53, 0.87, 1.10, 1.73 q), Courant 0.46 at 40 s |
| non-hydrostatic | velocity | ramp 10 s | level | full | swing grows (0.55, 0.90, 1.14 q), blows up at 44.5 s |
| non-hydrostatic | discharge | impulsive | level | full | stable over 8 s, Courant 0.76 |
| hydrostatic | discharge | impulsive | relaxed | full | stable over 90 s, Courant 0.49 |

Practice: against a fixed level, prescribe the discharge, not a velocity, and start from a ramp
(`left_discharge=t -> q(t)`, wetting-drying step), or use the relaxed outflow.

`step_forward!` warns once per model when the advective Courant number passes `ModelConfig(advective_courant_warning)`,
default `ADVECTIVE_COURANT_WARNING` $=0.8$; `Inf` switches it off (the largest value of a smooth run above was 0.76, the
runs break from about 0.85; the record is `advective_courant_max` of `compute_diagnostics`), and once when a velocity inlet stands against a fixed level with
`reach_mode_gain` above `REACH_MODE_GAIN_WARNING` $=1.1$. That is the factor of the table with
$U_2=U_1h_1/h_2$, the outlet velocity of the steady flow: the velocity at the outlet lags in a transient, and a uniform
channel would seem to grow. The factor does not include friction or losses at obstacles, so the warning says the mode
can grow, not that it does.

**Hydrostatic recirculation.** Stable is not the same as steady. With the relaxed outflow the hydrostatic step has a
recirculation of about the right length, but it never settles:

| grid | hydrostatic $x_R/S$ | std | largest $w$ (m/s) |
|---|---|---|---|
| S/5 | 5.64 | 1.00 | 0.091 |
| S/10 (half $\Delta t$) | 5.65 | 1.12 | 0.165 |
| S/20 (ramped start, 70 s) | 4.88 | 1.26 | 0.860 |

(S/5 and S/20 with the local wall function of section 27 and before issue #66; S/10 with the equilibrium one and the
time correction of issue #66: 5.69 S, std 1.20 and 0.179 m/s before the correction, 5.67 S, std 1.30 and 0.205 m/s with
the local wall function.)

$w$ follows from continuity alone and nothing limits it behind the step: the downdraft has $w\approx u$ with
$\Delta x\approx\Delta z$. Its peak grows 2-4 times per halving of the grid, against a non-hydrostatic value that stays
near 0.02 m/s, so these grids show no grid limit of $w$. The hydrostatic mean surface also has a dip behind the step
that the non-hydrostatic one does not have (`recirculation_step.png`).

**Obstacle with a sloped face.** A trapezoidal weir of height S (ramp 1:3, crest 4 S, downstream face 1:1, no reference):
non-hydrostatic $x_R=5.25\,S$ (std 0.05; 5.23 before issue #66, 5.29 with the local wall function) from the toe; hydrostatic: no reverse flow at
all. The hydrostatic model can
therefore miss a recirculation altogether, although it keeps the one behind the vertical step.

Limitations: $x_R$ drifts with $\Delta z$ (above); the bed cell is treated in section 29. One validated case, against a
simulation; the measured table of 1987 is not public.

### 29) Bed cell of k-epsilon on the z-level grid (issue #62)

**Problem.** On the z-level grid the bed cuts the lowest wet cell to any thickness $h_1\in(0,\Delta z]$. With the local
wall function of section 27 the cell takes $k=u_*^2/\sqrt{C_\mu}$ and $\varepsilon=u_*^3/(\kappa z_1)$ as Dirichlet values,
$u_*$ from its own velocity and $z_1=h_1/2$. Both depend on the cut: $u_*\to0$ where the bed velocity changes sign, so
$k$ is forced to zero at reattachment, and $\varepsilon\propto1/h_1$ grows without bound in a thin cell. Section 28
shows the effect: shifting the bed of the step against the layers by $0.9\,\Delta z$ makes $x_R$ 40-80 % longer and
unsteady.

**Equilibrium wall function** (`KEpsilonState.wall_function = :equilibrium`, the default; Launder & Spalding 1974;
Versteeg & Malalasekera 2007, sec. 9.11). The velocity scale is $u_k=C_\mu^{1/4}k_w^{1/2}$ of the wall cell, not the
local $u_*$:

- $k$ of the wall cell is solved (zero flux at the bed) with the production $\tau^2/(\kappa u_kz_1)$ of the shear stress
  $\tau=\kappa u_k|U_w|/\ln(z_1/z_0)$ and the dissipation $u_k^3/(\kappa z_1)$ as a sink;
- $\varepsilon$ of the wall cell is the Dirichlet value $u_k^3/(\kappa z_1)$ with the new $k$;
- in the log layer $P=\varepsilon$ gives $u_k^2=c_bU_w^2$ with $c_b=(\kappa/\ln(z_1/z_0))^2$, the local wall function;
  $u_k$ stays finite where $U_w$ changes sign.

**Wall cell.** The wall cell is the lowest wet cell at least $\theta\Delta z$ thick, $\theta=$
`KEpsilonState.wall_cell_min_fraction`, default `KE_WALL_CELL_MIN_FRACTION` $=0.3$. The shaved cells below it take its
$k$ and $\varepsilon$, $z_1$ is the height of its centre above the bed (shaved cells included), and the log-law bed
friction of the implicit column (section 18) acts on its layer at that $z_1$ (`implicit_vertical_column!(...;
friction_row)`); the shaved layers below follow through $\nu+\nu_t$. Without the move of the friction the uniform channel
with $f=0.9$ is 0.7 % too shallow and its $\nu_t$ 9 % too low, because the friction of a thin layer at $z_1=0.05\,\Delta z$
and the wall function one layer up disagree (`test_verify_implicit_vertical_friction_row`: the column solve with the
friction on layer $k_f$ equals the assembled matrix). `wall_function = :local` keeps the formulation of section 27.

**Shift study** (`experiments/obstacles/plot_bed_cell.jl`; ST1 and the weir of section 28, non-hydrostatic, 90 s, means
over the last 30 s). The bed is lifted by $f\Delta z$ against the layers (one layer more, the same $\Delta z$ and
geometry), so the lowest wet cell of the step is $(1-f)\Delta z$ thick; the sloped weir face has cut cells at every $f$.
$x_R/S$ (std of the instantaneous length in brackets where above 0.2). The runs predate the time correction of the `tvd2`
momentum advection (issue #66, section 18); rerun with it, the step S/5 with the local wall function gives 6.17, 6.04,
5.98, 6.27 and 8.46 (1.08), at most 0.06 S (0.7 %) from the row below, as the step of section 28 (6.07 to 6.06 S at
S/5, 7.22 to 7.21 S at S/10):

| case | wall function | $f=0$ | 0.25 | 0.5 | 0.75 | 0.9 |
|---|---|---|---|---|---|---|
| step S/5 | local | 6.18 | 6.06 | 6.00 | 6.29 | 8.52 (1.15) |
| | equilibrium, $\theta=0$ | 6.08 | 6.12 | 6.22 | 6.65 | 9.02 (1.06) |
| | equilibrium, $\theta=0.3$ | 6.08 | 6.12 | 6.22 | 5.33 | 5.72 |
| | equilibrium, $\theta=0.5$ | 6.08 | 6.12 | 6.22 | 5.33 | 5.72 |
| step S/10 | local | 6.99 | 7.02 | 7.11 | 7.37 (0.33) | 12.47 (2.39) |
| | equilibrium, $\theta=0$ | 7.22 | 7.31 | 7.44 | 7.18 | 12.51 (2.00) |
| | equilibrium, $\theta=0.3$ | 7.22 | 7.31 | 7.44 | 7.18 | 7.30 |
| | equilibrium, $\theta=0.5$ | 7.22 | 7.31 | 6.59 | 7.18 | 7.30 |
| weir S/10 | local | 5.29 | 5.32 | 4.15 | | |
| | equilibrium, $\theta=0$ | 5.44 | 5.64 | 4.04 | | |
| | equilibrium, $\theta=0.3$ | 5.23 | 5.30 | 4.04 | | |
| | equilibrium, $\theta=0.5$ | 4.51 | 5.30 | 1.80 | | |

- The equilibrium formulation alone ($\theta=0$) does not help: a wall cell of $0.1\,\Delta z$ breaks the step as the
  local one does. The thin cell, not $u_*$ of the local velocity, is the main cause.
- With the wall cell above the thin cell ($\theta=0.3$) the step at $f=0.9$ is steady and within 6 % (S/5) and 1 % (S/10)
  of the bed on an interface. The largest deviation is $f=0.75$ on S/5 (12 %): there the wall cell is a full layer with
  $z_1=1.25\,\Delta z$, the coarsest wall resolution of the study.
- $\theta=0.5$ puts the threshold on the half-shaved cells of $f=0.5$, where rounding decides column by column (S/10:
  6.59), and the jumps of $z_1$ between neighbouring cut cells along the weir slope shorten its recirculation (4.51 at
  $f=0$, 1.80 at $f=0.5$). $\theta=0.3$ follows the local wall function on the weir within 3 %.
- The weir at $f=0.5$ is 22 % shorter than at $f=0$ with every wall function. Its sloped face is cut differently at
  each shift, so this is the resolution of the geometry, not of the bed cell.

At $f=0.9$ the profiles behind the step with the equilibrium wall function lie on those at $f=0$; the local one
collapses $k$, $\varepsilon$ and $\nu_t$ over the whole recirculation (`bed_cell_profiles.png`, `bed_cell_fields.png`,
`bed_cell_step.gif`). The local wall function at $f=0$ also has $k$ and $\varepsilon$ of the bed cell two to four
decades below the cell above at $x=5$-7 S; the equilibrium one has no such drop.

**Uniform channel** (section 27, 20 layers, $f$ against the layers, `bed_cell_channel.png`):

| $f$ | wall function | depth (m) | max $\nu_t$ (m²/s) |
|---|---|---|---|
| 0 | local | 0.5000 | $2.941\cdot10^{-3}$ |
| 0 | equilibrium | 0.5000 | $2.942\cdot10^{-3}$ |
| 0.5 | local | 0.5001 | $3.011\cdot10^{-3}$ |
| 0.5 | equilibrium | 0.5001 | $3.014\cdot10^{-3}$ |
| 0.9 | local | 0.5001 | $3.087\cdot10^{-3}$ |
| 0.9 | equilibrium | 0.5001 | $2.966\cdot10^{-3}$ |

In uniform flow both wall functions agree on the interface (P = ε); at $f=0.9$ the equilibrium one is within 1 % in
$\nu_t$, the local one 5 % above.

Limitations: the wall cell is at least $\theta\Delta z$ thick, so the wall resolution varies between
$z_1=0.15\,\Delta z$ and $1.3\,\Delta z$ with the position of the bed; a scalable or low-Reynolds treatment of the
viscous sublayer is not included ($z_1^+$ of 2-8 behind the step, section 28). $\theta$ comes from one step and one weir.

### 30) Stratified mixing layer behind a splitter plate (issue #42)

**Case.** Two streams meet at the end of a horizontal plate of zero thickness, after the splitter plate test of the
Delft3D-FLOW Validation Document (2008): below the plate 0.306 m of water of 43.3 ppt at 0.30 m/s, above it 0.268 m of
22.6 ppt at 0.50 m/s (densities 1031.2 and 1015.3 kg/m³, Eckart 1958 at 20 °C); 40 m reach, flat bed with Chézy
65 m$^{1/2}$/s ($z_0=4.3\cdot10^{-5}$ m for the wall function), level 0.574 m at the plate, relaxed outflow to 0.570 m
($\tau=20$ s); k-epsilon with buoyancy terms, 6 min, means over the last 2 min. `experiments/splitter_plate/` (README).

**Inflow per layer.** `set_x_inflow_outflow!` takes `left_rho` as one value per layer and `left_k`, `left_eps` (one value
per layer, both or neither; wetting and drying with k-epsilon): the upwind flux through the inlet face carries the
prescribed value of its layer instead of that of the first cell; the density bounds of the monotone transport include
the inflow values. The plate is not in the domain (option B of the issue): the inlet gets $u$, $k$ and $\varepsilon$ at
the end of two 4 m precursor channels (lower stream: rough bed under a free-slip plate; upper stream: free-slip bed, the
frictionless plate of the issue, log profile from 0.3 m/s). Each precursor inlet carries the log profile with $k$ and
$\varepsilon$ in equilibrium with it (stress $u_*^2(1-z/h)$, mixing length $\kappa z$; Rodi 1993):
$k=u_*^2(1-z/h)/\sqrt{C_\mu}$, $\varepsilon=u_*^3(1-z/h)/(\kappa z)$.

**Plate height from the measurements.** The measured profiles (extracted visually from the figures of the validation
document; `data/*.csv`, about one pixel accuracy) put the density interface (relative density 0.5) at 0.307 and 0.305 m
at 2 and 5 m, and the measured velocity integrates to 0.094/0.092 m²/s below it (mean 0.31/0.30 m/s) and 0.133/0.131
m²/s above it (0.50/0.49 m/s). The issue had estimated 0.250 m below and 0.324 m above the plate; with those streams
the model interface stayed at 0.25 m.

**Grid.** 41 layers of 13.9 mm, 22 below the plate (the plate on a layer interface); the top layer is open above and holds
the surface over the whole reach. $\Delta x$ = 5 and 10 cm, advective Courant number 0.6.

**Results** (means; measured values extracted from the figures):

| station | interface (rel. density 0.5), model / measured | mean $u$ below 0.2 m, model / measured | $Ri_g$ at the interface | $k$ at the interface (m²/s²) |
|---|---|---|---|---|
| 2 m | 0.305 / 0.307 m | 0.278 / 0.282 m/s | 0.16 | $6.5\cdot10^{-5}$ |
| 5 m | 0.306 / 0.305 m | 0.274 / 0.277 m/s | 0.21 | $7\cdot10^{-6}$ |
| 10 m | 0.309 / 0.305 m | 0.269 / 0.286 m/s | 0.25 | $2\cdot10^{-6}$ |
| 40 m | 0.327 / 0.287 m | 0.253 / 0.287 m/s | 0.17 | $3.9\cdot10^{-5}$ |

$\Delta x$ = 5 and 10 cm agree within $10^{-3}$ m/s and 0.011 in relative density, hydrostatic and non-hydrostatic within
$10^{-4}$ m/s and $6\cdot10^{-4}$ (the mean flow has no vertical dynamics). Outflow and inflow agree within $10^{-4}$ over
the last 2 min; the salt budget closes with the boundary fluxes to $10^{-11}$ of the salt inflow; the density stays
within the two inflow densities. The density interface (relative density 0.1 to 0.9) is 11 mm thick at the inlet and
34-38 mm from 2 m on.

**Analysis.** Up to 10 m the model follows the measured velocity, density and the lower-stream $k$. Two differences:
the measured $k$ has a maximum of $2$-$5\cdot10^{-4}$ m²/s² in the mixing layer, where the model $k$ collapses; and the
model's lower stream slows down along the reach (0.278 to 0.253 m/s), while the measured one keeps 0.28-0.29 m/s, so
that it thickens and the interface rises to 0.327 m at 40 m. In homogeneous stratified shear $k$ is stationary when
$P(C_2-C_1)=B(C_2-C_3)$, i.e. at the gradient Richardson number

$$Ri_{st}=\sigma_\rho\,\frac{C_2-C_1}{C_2-C_3}$$

(Burchard & Baumert 1995), 0.175 for $C_3=0$ and $\sigma_\rho=0.7$. The interface has $Ri_g$ = 0.16-0.36 (layer by
layer), mostly above, so the turbulence decays there ($\nu_t\approx2\cdot10^{-7}$ m²/s at 10 m). To keep the lower stream
moving against its bed stress (about $2\cdot10^{-4}$ m²/s²), the interface would have to pass that stress at a shear of
about 2 1/s, i.e. $\nu_t\approx10^{-4}$ m²/s. The measurements show this momentum exchange while the density stays
stratified; at high $Ri$ momentum is exchanged more efficiently than mass (internal waves), which one constant turbulent
Schmidt number cannot represent.

**Sensitivity** ($\Delta x$ = 10 cm, non-hydrostatic; at 40 m):

| run | interface | mean $u$ below 0.2 m | interface thickness |
|---|---|---|---|
| measured | 0.287 m | 0.287 m/s | |
| as the issue (frictionless plate, $C_3=0$, `tvd2` density transport) | 0.327 m | 0.253 m/s | 38 mm |
| rough plate ($z_0$ on both sides) | 0.309 m | 0.253 m/s | 79 mm |
| $C_3$ = 0.58 in stable stratification ($Ri_{st}=0.25$) | 0.330 m | 0.253 m/s | 71 mm |
| `upwind1` density transport | 0.324 m | 0.253 m/s | 49 mm |

- Rough plate: the lower stream is then a duct between two rough walls; unstratified, it is symmetric about its
  mid-height (a plane without shear), so its precursor is an open channel of half the depth, mirrored. The upper stream
  gets the measured level of $k$ (up to $5\cdot10^{-4}$ m²/s² at 5 m), but the plate wake (zero shear at the plate, where
  the densities meet, $Ri_g>1$) is too deep at 2 m.
- $C_3$: `KEpsilonState(c3_stable=...)` (default 0, Rodi 1987); `ke_c3_for_richardson(Ri)` gives
  $C_3=C_2-\sigma_\rho(C_2-C_1)/Ri$, 0.576 for 0.25. For $C_3>0$ the stable term $C_3G<0$ is an implicit (Patankar) sink
  of $\varepsilon$. The value $C_3=-0.4$ of Burchard & Baumert (1995) and Umlauf & Burchard (2003) belongs to closures
  with Richardson-dependent stability functions; with a constant $\sigma_\rho$ it gives $Ri_{st}=0.145$.
- `upwind1`: the numerical mixing of the density transport thickens the interface but does not move the lower stream.

None of the three changes the lower stream: the missing process is the momentum flux through the stratified interface.
Candidates are a background (internal-wave) viscosity and diffusivity with a Prandtl number of about 10 (Large et al.
1994) or Richardson-dependent stability functions (Schumann & Gerz 1995; Canuto et al. 2001); both change the closure for
every stratified case and are left to a separate issue.

Limitations: option B (nothing downstream acts on the streams above the end of the plate); level drop 2.2 mm against the
4 mm of the test (bed friction alone gives about 2.6 mm), and the relaxed outflow keeps the outflow level 2.9 mm above
its target in the steady state; the measurements are extracted from figures, not the original data.

### 31) Suspended cohesive sediment and a jet pulse eroding the bed (issue #64)

**Equations** (`src/sediment.jl`, wetting-drying step). One fraction of suspended mud, concentration $c$ (kg/m³):

$$\partial_t(hc)+\partial_x(qc)+\Delta_z(Wc)=\partial_z\left(K\,\partial_zc\right)+\partial_z(w_sc),\qquad K=\nu_t/\sigma_c,\ \sigma_c=0.7,$$

with the bed exchange in the lowest wet cell: erosion $E=M(\tau_b/\tau_{ce}-1)$ for $\tau_b>\tau_{ce}$ (Partheniades 1965)
and deposition $D=w_sc_b$ (continuous deposition; Winterwerp & van Kesteren 2004, sec. 7.4), $M$ the erosion parameter
(kg/m²/s), $\tau_{ce}$ the critical stress (Pa), $w_s$ the settling velocity. The erodible bed mass per column limits $E$
(default unlimited). The bed shear stress is that of the momentum bed friction, $\tau_b=\rho_0c_fU_w^2$ with
$c_f=(\kappa/\ln(z_1/z_0))^2$ in the wall cell (the equilibrium wall cell of section 29 with k-epsilon).

**Numerics.**
- Advection: the conservative transport of the density (section 18) as a scalar kernel (`_wd_transport_scalar!`, the
  density path unchanged): increment form, `upwind1` or flux-limited Lax-Wendroff `tvd2`, bounds $[0,\max c]$; the inflow
  at the left face carries `inflow_concentration`, water entering through the right face `outside_concentration`
  (default 0: clean water from outside; zero gradient there would carry the plume back in at a flow reversal).
- Settling, turbulent diffusion and the bed exchange: per column backward Euler in finite volumes on the wet cells of
  $h^{n+1}$, the settling flux upwinded from above (out through the lower face of each cell, out of the lowest one into
  the bed), $E$ as a source in the lowest cell. The matrix is an M-matrix (positive diagonal, non-positive off-diagonal),
  so $c\ge0$ for any $\Delta t$ without clipping, and the column sums give
  $\sum_kh_kc_k^{n+1}=\sum_kh_kc_k^*+\Delta t\,(E-w_sc_b^{n+1})$: the suspended mass changes exactly by the bed exchange.
- Density coupling (default): the mixture density $\rho+c(1-\rho/\rho_s)$ replaces $\rho$ in the baroclinic pressure and
  in the buoyancy term of k-epsilon; the transported $\rho$ is the water density.

**Verification** (`test_sediment.jl`):

| test | measured |
|---|---|
| settling column, uniform $c_0$, no erosion: suspended mass against $c_0(H-w_st)$ until the clear-water front is near the bed ($t=0.2H/w_s$) | $1.4\cdot10^{-6}$ (implicit upwind smearing of the front) |
| suspended + deposited mass | constant to $10^{-15}$ |
| erosion flux against $M(\tau_b/\tau_{ce}-1)$ with the $\tau_b$ of the step; limited by a small bed mass | to round-off |
| channel with erosion, clean inflow, outflow: suspended mass against inflow - outflow + erosion - deposition | $2\cdot10^{-14}$ |
| reversed flow at the open boundary: the entering water carries `outside_concentration` | to round-off |
| lock exchange driven by sediment against the equivalent salt density $\rho+c(1-\rho/\rho_s)$ | same flow to $10^{-9}$ |
| equilibrium profile in a uniform k-epsilon channel against the Rouse profile $[(H-z)/z\cdot z_r/(H-z_r)]^P$, $P=w_s\sigma_c/(\kappa u_*)=0.52$ | $-5.4$ to $+4.8$ % in $0.15$-$0.75\,H$ |

(Rouse 1937: the profile of the parabolic $\nu_t=\kappa u_*z(1-z/H)$; the k-epsilon $\nu_t$ is not exactly parabolic.)

**Jet pulse** (`experiments/jet_sediment/`, README). 200 m basin, 28 m deep, mud bed ($z_0=2\cdot10^{-4}$ m; $w_s=0.5$
mm/s, $\tau_{ce}=0.2$ Pa, $M=10^{-4}$ kg/m²/s, assumed). A pipe of 3 m diameter in the left wall, lower edge 5 m above the
bed: in 2DV a slot from 5 to 8 m at 3 m/s (9 m²/s per metre: the 120 m³/s of the issue over a basin width of 13 m),
3 min with 5 s ramps; relaxed outflow at the right end; k-epsilon, non-hydrostatic, $\Delta x$ = 1 m,
$\Delta z$ = 0.5 m; 15 min.

The offset jet attaches to the bed within about 20 m (Coanda effect) and runs along it as a wall jet to 70-80 m, where it
lifts the mud (up to 259 mg/l); the bed shear stress exceeds 10 Pa over the first 68 m (peak 29.7 Pa at $x$ = 26 m). Of
the 82 kg/m eroded in 15 min, 44 kg/m is eroded during the pulse and 37 kg/m after it, under the near-bed branch of the
two recirculation cells the jet leaves in the first 100 m; 63 kg/m of the net loss comes from the first 50 m, 16 kg/m from
50-150 m, 0.3 kg/m beyond. At 900 s 79 kg/m is in suspension (about 30 mg/l over the whole depth in the first 110 m),
90 % of it within 100 m of the pipe; settling returns 1.9 kg/m, the outflow takes 0.3 kg/m. The budget closes to
$3\cdot10^{-12}$ kg/m.

Non-hydrostatic only: the attachment of the offset jet is due to the low pressure between jet and bed, which the
hydrostatic model does not have. In a run with a 6 m pipe at 2 m/s for 3 min the hydrostatic jet did not attach and
eroded a tenth of the non-hydrostatic one; there $\Delta x$ = 1 and 0.5 m agreed within 10 %.

Limitations: 2DV (the jet does not spread sideways, so the wall jet and the recirculation cells reach further than in the
real basin; during the pulse the discharge has to leave through the basin, a 0.32 m/s through-flow that holds the bed
above $\tau_{ce}$ away from the jet and raises the level by about $q/\sqrt{gH}=0.54$ m); one fraction without flocculation
and consolidation; unlimited erodible bed; assumed mud parameters.

### 32) Performance: type-stable kernels, reuse of the pressure factorization, performance overview (issue #10)

**Measurement.** Every model records its steps and their wall-clock time and allocations (`model.performance`, set in
`step_forward!`); with the section timers (TimerOutputs: `step/*` of the fixed grid, `wd/*` of the wetting-drying step,
`pressure/*` of the fixed-grid pressure solver) they are collected per session. `print_performance_report()`
(`src/performance.jl`) prints wall-clock, CPU (all threads), compilation and GC time, the allocations, one row per
model with the cost per cell and step and the merged section timers; `reset_performance!()` starts a new overview.
Every script in `experiments/` includes `experiments/common/performance.jl`, which prints the overview at exit.

**Causes found** (allocation profiler, `Profile.Allocs`, on the jet of section 31 and the lock exchange):
- untyped fields of `Model` (`turbulence`, `wetdry`, `x_boundary_eta_left`, `x_boundary_eta_right`,
  `x_boundary_rho_left`, all `Any`): every value computed from them in a kernel was dynamically typed and boxed, 16 bytes
  per cell and operation. Affected: the k-epsilon terms of the wetting-drying predictor, the k-epsilon and sediment
  steps, the free-surface solve (level of the open boundary), the face thicknesses and the surface Froude check of every
  face (also on the fixed grid: about 1 MiB per step on 300 x 40 cells). Now typed (`Union{Nothing,KEpsilonState{T}}`,
  `Union{Nothing,WetDryState{T}}`, `Union{Nothing,Float64}`, `Union{Nothing,Float64,Vector{Float64}}`); `WetDryState` is
  defined before `Model` (`src/wetting_drying_state.jl`);
- the mixture density of the sediment coupling, read through the untyped `wd.sediment` (now asserted `Matrix{T}`), and
  a closure allocated per cell in the sediment column (now a function);
- the constituent mass of the fixed grid with a bed: `sum(rho[wet_cell])` copied the wet cells every step (now gathered
  into a work array: the same pairwise sum, bit for bit);
- the non-hydrostatic pressure of the wetting-drying step: assembled with `sparse` and factorized with a new symbolic
  analysis in every step (12.2 ms of a 44 ms step on 200 x 56 cells, 11000 unknowns).

**Reuse of the pressure factorization** (`_wd_pressure_matrix!`, `_wd_pressure_factor!`). The symbolic analysis
(fill-reducing ordering, elimination tree) depends only on the pattern of the matrix, which changes only when a cell
becomes a pressure unknown or stops being one (the surface cell of a column moves to another layer, a column falls dry or
wets). While the triplet pattern equals that of the last step, the values are written into the stored matrix through the
position of each triplet (no sort) and only the numeric factorization is repeated (`cholesky!`). The same pattern gives
the same ordering and the same operations: the result is bit-identical to a new factorization. In the jet one analysis
was needed in 305 steps; assembly 0.83 -> 0.22 ms, factorization 12.2 -> 7.0 ms.

**Tried and not adopted** (the same matrices, captured from the jet):
- CG with the tridiagonal blocks of the columns as preconditioner (line Jacobi), warm start from the last step: 230-280
  iterations, 60-80 ms (the horizontal coupling over 200 columns needs a multigrid preconditioner);
- CG preconditioned with the factor of an earlier step (refactorization only when the iterations grow): 5-7 iterations
  of 1.3 ms, no faster than the numeric factorization plus one solve (7.9 ms);
- the number of BLAS threads used by CHOLMOD (1, 2, 4, 6): within the scatter of the measurement.

**Results** (warm steps, Windows, 6 BLAS threads, one Julia thread; the same set-ups with `main` before and with
issue #10; the wetting-drying cases are the jet basin of section 31 at $\Delta x$ = 1 m with the 3 m/s inflow):

| set-up | before (ms/step) | after (ms/step) | speed-up | after (us per cell and step) | allocated per step, before -> after |
|---|---|---|---|---|---|
| fixed grid 300 x 40, non-hydrostatic, spectral (lock exchange) | 2.82 | 1.25 | 2.3 | 0.104 | 1015 KiB -> 12 KiB |
| fixed grid 300 x 40, hydrostatic, variable top layer | 2.17 | 0.84 | 2.6 | 0.070 | 1007 KiB -> 0.2 KiB |
| fixed grid 300 x 40, non-hydrostatic, k-epsilon | 5.34 | 2.22 | 2.4 | 0.185 | 1015 KiB -> 13 KiB |
| fixed grid 300 x 40, non-hydrostatic, bed step (sparse solver) | 3.68 | 2.26 | 1.6 | 0.19 | 1252 KiB -> 0.1 KiB |
| wetting-drying 200 x 56, hydrostatic | 10.2 | 1.32 | 7.7 | 0.118 | 8027 KiB -> 1.4 KiB |
| wetting-drying 200 x 56, hydrostatic, k-epsilon | 20.9 | 3.02 | 6.9 | 0.270 | 16572 KiB -> 1.4 KiB |
| wetting-drying 200 x 56, non-hydrostatic | 25.3 | 11.4 | 2.2 | 1.015 | 18695 KiB -> 2950 KiB |
| wetting-drying 200 x 56, non-hydrostatic, k-epsilon, sediment | 42.3 | 14.2 | 3.0 | 1.27 | 30547 KiB -> 2950 KiB |

Line coverage of `src/` (julia --code-coverage, full suite): 96.8 % (4174 of 4314 lines; `visualization.jl` 89 %, the
non-Windows branch of `process_cpu_time` is not reached on Windows).
The machine load changed the absolute times by up to 40 % between runs; the ratios were stable. The full test suite
went from 34.5 to 11.0 min. The results are unchanged bit for bit (against `main`: the jet, hydrostatic and
non-hydrostatic with k-epsilon and sediment, 150 steps, and a non-hydrostatic dam break over a slope, 300 steps).

**Comparison with the lock exchange.** The fixed grid costs 0.07-0.25 us per cell and step in every mode, and the
hydrostatic wetting-drying step is now as cheap (0.12, with k-epsilon 0.27). The non-hydrostatic wetting-drying step
remains about ten times dearer than the spectral lock exchange: the cut cells and the moving surface exclude the
spectral solver, and the numeric Cholesky factorization of every step is half of its cost (7 of 14 ms on 11000
unknowns; CHOLMOD also allocates 1.9 MiB for its copy of the matrix). The time step of these runs is set by the
advective Courant number ($U\Delta t/\Delta x$ = 0.5 in the jet), not by the solver.

**Ideas for further issues.**
- Multigrid-preconditioned CG for the non-hydrostatic wetting-drying pressure: done in issue #71 (section 37), a
  column-line multigrid, from 5000 unknowns on the default.
- Multi-threading of the column kernels: the implicit vertical terms, k-epsilon, the sediment column solve and the face
  loops of the predictor are independent per column or face (`Threads.@threads` over columns, work arrays per thread).
- Parallel parameter studies: the experiments that scan resolutions or set-ups (bed cell, obstacles, splitter plate)
  run independent models one after the other; one thread or process per case would use all cores.
- GPU (KernelAbstractions) for the fixed grid: the explicit stencils and the spectral solver (cuFFT) map directly; the
  wetting-drying step (Newton solve, sparse factorization) much less so.

### 33) Dam breaks: Stoker on a wet bed and Ritter on a dry bed, in layers (issues #11, #12)

**Reference** (`stoker_dam_break`, `stoker_middle_state`, `src/reference_solutions.jl`). Stoker (1957, ch. 10; SWASHES
Sec. 4.1.1): depth $h_0$ behind the dam at $x_0$, $h_1$ in front of it, flat frictionless bed. With $c_0=\sqrt{gh_0}$
the middle state $c_m=\sqrt{gh_m}$ solves

$$-8gh_1c_m^2(c_0-c_m)^2+(c_m^2-gh_1)^2(c_m^2+gh_1)=0,\qquad \sqrt{gh_1}<c_m<c_0,$$

(bisection to machine precision), $u_m=2(c_0-c_m)$ (Riemann invariant), and the bore runs at
$s=2c_m^2(c_0-c_m)/(c_m^2-gh_1)$. The solution is still water for $\xi=(x-x_0)/t\le-c_0$, the rarefaction
$h=(2c_0-\xi)^2/(9g)$, $u=\tfrac23(\xi+c_0)$ for $-c_0<\xi\le u_m-c_m$, the middle state up to $\xi=s$ and still water
$h_1$ beyond; $h_1=0$ gives Ritter.

**Set-up** (`test/test_dam_break.jl`, `experiments/wetting_drying/`, mode `stoker`). The Ritter channel of section 18
(100 m, dam at 40 m, $h_0$ = 1 m, grid depth 1.2 m, $\Delta t$ = 0.01 s/m $\cdot\,\Delta x$, comparison at 5 s), Stoker
with $h_1$ = 0.2 m ($h_m$ = 0.508 m, $u_m$ = 1.80 m/s, $s$ = 2.97 m/s, bore Froude number 2.1), upwind1. The bore position
is where the depth crosses $(h_m+h_1)/2$, interpolated linearly between the cells. The constituent of issues #11 and #12
is a passive tracer: `attach_sediment!` with $w_s=0$, no erosion and no density coupling (now allowed on a frictionless
bed, since the bed shear stress only drives the erosion), 1 kg/m³ behind the dam.

**Results** at $\Delta x$ = 0.25 m (relative L1 depth error; bore or front position error relative to the distance
travelled):

| case | layers | hydrostatic L1 | bore / front | non-hydrostatic L1 | bore / front |
|---|---|---|---|---|---|
| Stoker | 1 | 0.42 % | -0.34 % | | |
| Stoker | 2 | 0.42 % | -0.39 % | 0.54 % | -0.51 % |
| Stoker | 3 | 0.43 % | -0.83 % | 0.81 % | -2.2 % |
| Stoker | 5 | 0.51 % | -1.5 % | 0.96 % | -3.4 % |
| Stoker | 10 | 0.52 % | -1.7 % | 1.01 % | -3.6 % |
| Ritter | 1-10 | 0.90 % | 10.3-11.3 % | 1.00-1.04 % | 10.3-11.3 % |

- Stoker 1D converges at first order (L1 1.2 %, 0.74 %, 0.42 %, 0.25 %, 0.15 % from $\Delta x$ = 1 to 1/16 m; bore
  0.40 % to 0.12 %); tvd2: L1 0.28 %, bore -0.10 %.
- Hydrostatic layers: the depth stays within 0.23 % of the 1D depth, but the bore lags once it lifts the surface through
  a layer interface (from 3 layers), by 1.5-2 % of the distance travelled, converging slowly (10 layers: 2.0 % at
  $\Delta x$ = 1 m, 1.5 % at 0.125 m). Behind the bore the layer velocities differ by up to 0.08 m/s (4 % of $u_m$):
  across a shock the split of the momentum between the layers depends on the scheme (the layers below $h_1$ start at
  rest, those above are filled through the bore), and without viscosity nothing evens it out. Initialising a newly opened
  face layer with the velocity of the same layer upstream instead of the layer below moved the bore by 0.1 % and raised
  the spread to 0.44 m/s; the source of the lag is open.
- Non-hydrostatic: the bore front has the width of the depth and lags the shallow-water bore by about 0.5 m at
  $\Delta x$ = 0.25 m and 0.125 m alike (3.6 %, 3.7 % with 10 layers); the middle state is 1.6 % deeper. The shallow-water
  solution is not the limit of the non-hydrostatic equations at a bore; the tests bound the difference.
- Ritter is unchanged with 2, 3, 5 or 10 layers in both modes (the falling surface opens no layers).
- Water volume and constituent mass are conserved to $10^{-15}$ (relative) in all runs; the tracer stays within its
  initial bounds and does not change the flow (bit for bit).

### 34) Internal seiche: period against the two-layer relation, and the density extrapolation (issue #23)

**Reference** (`two_layer_internal_wave`, `src/reference_solutions.jl`). Two homogeneous layers with a free surface,
linear, Boussinesq, $g'=g\Delta\rho/\rho_0$. Kinematic conditions at the surface $\eta$ and the interface $\zeta$,
$\phi_t+g\eta=0$ at the surface and $[\phi_t]=-g'\zeta$ across the interface give, with $C_{u,l}=\coth(kh_{u,l})$,

$$(C_uC_l+1)\,\omega^4-\left(gk(C_u+C_l)+g'kC_u\right)\omega^2+gg'k^2=0,\qquad
\frac{\eta}{\zeta}=-\frac{\omega^2}{gk\sinh(kh_u)-\omega^2\cosh(kh_u)}$$

(smaller root: the internal mode), and hydrostatic $c^4-(gH+g'h_l)c^2+gg'h_uh_l=0$, $\eta/\zeta=(c^2-g'h_l)/(gh_l)$. The
rigid-lid limits are $\omega^2=g'k/(C_u+C_l)$ (Gill 1982, ch. 6) and $c^2=g'h_uh_l/H$; the free surface lengthens the
periods by $O(g'/g)$.

**Case** (`test/test_internal_seiche.jl`, `experiments/internal_seiche/`). The 8 m x 4 m closed basin of the issue,
998.5 over 1001.5 kg/m³, interface at mid-depth displaced by 0.12 m as $\cos(2\pi x/L)$ (one wavelength, $kh=1.57$),
`tvd2`, no damping or limiter, $\theta=0.7$, $\Delta t$ = 0.05 s on all grids ($\omega\Delta t\le0.007$). The initial
interface is cell-averaged (`initialize_internal_seiche!(...; cell_average=true)`): the step profile would put it on the
cell faces and round the 0.12 m displacement to whole cells. The interface height of a column is the depth of the heavy
fluid, $\sum\Delta z\,(\rho-\rho_u)/(\rho_l-\rho_u)$, projected on $\cos kx$; the period follows from its zero crossings
over three periods. References: $T$ = 61.03 s non-hydrostatic, 46.65 s hydrostatic (rigid lid 61.0 s and 46.6 s, the
values of the issue).

**Results** (period error; amplitude kept per period):

| grid ($\Delta x=\Delta z$) | hydrostatic | non-hydrostatic |
|---|---|---|
| 40 x 40 (0.2 m) | +3.06 %; 0.984 | +5.09 %; 0.981 |
| 80 x 80 (0.1 m) | +1.74 %; 0.989 | +3.06 %; 0.988 |
| 160 x 160 (0.05 m) | +1.00 %; 0.991 | +1.86 %; 0.991 |
| observed order | 0.82, 0.80 | 0.73, 0.72 |

- The periods converge at about first order and from above: the interface spreads over about one cell, a continuous
  stratification that lowers the frequency. The non-hydrostatic period is the full relation, 31 % longer than the
  long-wave one.
- The surface moves in counter-phase. Hydrostatic: $\eta/\zeta=-0.00150$ as the relation. Non-hydrostatic with the
  default Neumann top condition also $-0.00150$, the hydrostatic response (docs section 15: that condition leaves the
  surface part hydrostatic); with `nonhydro_surface_dirichlet=true` $-0.00070$ and $-0.00065$ on 40 x 40 and 80 x 80,
  converging to the relation's $-0.00060$. The period is the same with both conditions.
- Water volume and salt mass are conserved to round-off (the salt mass is enforced by `_correct_constituent_mass!`).

**Hydrostatic internal waves were unstable for $\theta<1$.** On 160 x 160 cells the hydrostatic run at $\theta=0.7$ grew
from about 45 s and stopped at 58 s (max $|u|$ 0.06 m/s, the seiche itself 0.01 m/s); 160 x 80 grew as well, 80 x 160
did not, `upwind1` grew too, half the time step slowed the growth, $\theta=1$ and the non-hydrostatic mode were stable.
Cause: the momentum predictor takes the density of time $n$ (explicit baroclinic pressure), the transport moves it with
$u^{n+\theta}=\theta u^{n+1}+(1-\theta)u^n$ (section 19). For an internal wave of frequency $\omega$, $s=\omega\Delta t$,
the oscillator model $(\lambda-1)u=-s\,r$, $(\lambda-1)r=s(\theta\lambda+1-\theta)u$ gives
$\lambda^2-(2-\theta s^2)\lambda+1+(1-\theta)s^2=0$, so $|\lambda|^2=1+(1-\theta)s^2>1$ for every $\theta<1$; $\theta=1$ is
the neutral forward-backward scheme. The growth is fastest for the short hydrostatic internal waves (their frequency
grows with $k$; the non-hydrostatic pressure bounds it), and shows once $\Delta x$ is fine enough that numerical
diffusion no longer damps them.

**Fix** (`baroclinic_extrapolation_weight`, `_baroclinic_density`). The baroclinic pressure and the non-hydrostatic
buoyancy of the predictor take the extrapolated density $\rho^n+\beta(\rho^n-\rho^{n-1})$ with $\beta=2(1-\theta)$; the
characteristic equation becomes $\lambda(\lambda-1)^2+s^2((1+\beta)\lambda-\beta)(\theta\lambda+1-\theta)=0$ with the
growth term $(1-\theta)-\beta$ at order $s^2$. $\beta=1-\theta$ leaves a growth of order $s^4$; $\beta=2(1-\theta)$ damps
every wave up to $s_{\max}$ = 0.82 ($\theta=1/2$), 0.86 (0.55), 1.13 (0.7), 1.89 (0.9), an internal Courant number up to
about $s_{\max}/\pi$. A resolved wave loses about $\pi(1-\theta)\omega\Delta t$ of its amplitude per period (0.6 % in the
seiche above); $\theta=1$ is unchanged, bit for bit. The first step after a (re)initialisation does not extrapolate;
with wetting and drying the buoyant (mixture) density is extrapolated, except in cells that were dry at the last step.
The full suite is unchanged within its bounds; the Rayleigh-Taylor growth rate (0.8409 of the theory) and a lock-exchange
front (4.838 m at 20 s, hydrostatic and non-hydrostatic) are identical to four digits, since the extrapolation changes a
resolved wave by $O((1-\theta)s^2)$ per step.

### 35) Kelvin-Helmholtz instability: growth rate of a resolved shear layer (issue #22)

**Reference** (`kelvin_helmholtz_tanh_mode`, `src/reference_solutions.jl`). Shear layer $\bar u=U\tanh((z-z_0)/\delta)$
with $N^2=J(U/\delta)^2\,\mathrm{sech}^2((z-z_0)/\delta)$ (a tanh density profile of the same thickness), walls at the bed
and the top. Taylor-Goldstein equation

$$\phi''=\left(k^2+\frac{\bar u''}{\bar u-c}-\frac{N^2}{(\bar u-c)^2}\right)\phi,\qquad \phi(0)=\phi(H)=0,$$

solved by shooting (RK4 from the bed, secant iteration on the complex phase speed $c$); the growth rate is
$\sigma=k\,c_i$. A result whose eigenfunction does not meet the top wall ($|\phi(H)|>10^{-6}\max|\phi|$) is rejected:
beyond the neutral wavenumber the secant can stall at a non-root. Checks: unbounded and unstratified, the largest rate
is 0.1897 $U/\delta$ at $k\delta=0.4446$ (Michalke 1964; computed 0.18970 with walls 20 $\delta$ away); $c_r=0$; the rate
falls with $J$ and goes to zero towards $J=1/4$ (Miles 1961, Howard 1961).

**Case** (`test/test_kelvin_helmholtz.jl`, `experiments/kelvin_helmholtz/plot_growth_rate.jl`). $\delta$ = 0.1 m,
$U$ = 0.05 m/s, depth 0.8 m (8 $\delta$), $J$ = 0.05 ($\Delta\rho$ = 0.255 kg/m³) and $J$ = 0 as a control; periodic box
of one wavelength $2\pi/k$ with $k\delta=0.4446$ (1.413 m); non-hydrostatic, `tvd2`, no viscosity, damping or limiter;
$U\Delta t/\Delta x$ = 0.2 on every grid. `initialize_shear_layer_mode!` sets the mean flow and density plus the
Taylor-Goldstein eigenfunction (velocity from a stream function at the cell corners, so the discrete divergence is zero;
density from the linearised density equation) with an interface displacement of 1 mm: the mode grows at its rate
from the first step. The rate is the slope of $\log|\hat w|$ (the $e^{ikx}$ component of $w$ at $z_0$) over 5-35 s;
saturation begins after about 45 s. Theory: $\sigma$ = 0.07382 1/s ($J$ = 0.05), 0.08808 1/s ($J$ = 0).

**Results** (relative error of the growth rate):

| cells per $\delta$ ($\Delta x=\Delta z$) | $J$ = 0.05 | $J$ = 0 |
|---|---|---|
| 8 (12.5 mm, 113 x 64) | -1.02 % | -0.87 % |
| 16 (6.25 mm, 226 x 128) | -0.52 % | -0.46 % |
| 32 (3.125 mm, 452 x 256) | -0.28 % | -0.27 % |

- First-order convergence from below (orders 0.98, 0.89 and 0.90, 0.76): the numerical diffusion widens the shear layer
  and lowers the rate.
- Wavelength: in a box of four wavelengths (16 cells per $\delta$) seeded with divergence-free noise in the modes 1-12,
  the rates of the modes 2-7 ($k\delta$ = 0.22-0.78) follow the Taylor-Goldstein curve within 3 % (fit 20-50 s). Mode 1
  (the longest) has not left the transient of the non-eigenfunction start, and the modes 8, 11, 12 are fed by the
  harmonics of the dominant modes (mode 8 = 2 x mode 4, modes 11, 12 above the neutral wavenumber $k\delta\approx0.95$);
  9 and 10 decay. The dominant mode of $w$ moves from 7 to 5 during the transient, is 5 and 4 (the fastest) up to
  100 s and 3 after the billows merge. The wavelength is therefore set by the shear layer, not by the grid, unlike the
  showcase run of `plot_kelvin_helmholtz.jl` (shear layer two cells thick).
- Roll-up and mixing: the billows have the same shape on the three grids; the mixed layer (heavy fraction 0.05-0.95)
  grows from 0.30 m to 0.445, 0.414, 0.384 m at 150 s and the density variance falls to 0.67, 0.75, 0.83 of its initial
  value (8, 16, 32 cells per $\delta$): without viscosity the mixing is numerical and decreases with refinement.
- Periodic seam: the same run shifted by half a wavelength (the billow forms across the seam) agrees with the interior
  run after shifting back to $5\cdot10^{-8}$, $1.4\cdot10^{-6}$, $1.4\cdot10^{-5}$, $4.8\cdot10^{-5}\,\Delta\rho$ at 40, 70, 100, 130 s (round-off
  amplified by the roll-up).
- Salt mass and water volume are conserved to round-off.

### 36) Turbulent plane jet with k-epsilon (issue #25)

**Why a half-jet in a co-flow.** The issue asks for a plane jet from a slot at mid-depth into still water. In the 2DV
channel (bed and free surface, no lateral boundaries) the jet has nowhere to entrain from except the open outflow:
- a symmetric jet at mid-depth attached to the bed within a few seconds (Coanda effect);
- a half-jet on the frictionless bed (the symmetry plane of the jet, z = 0) with zero inflow outside the slot drives a
  recirculation cell the size of the channel (the confined jet of Craya and Curtet): the whole lower half moves forward,
  the upper half returns through the outflow, and the excess momentum flux grew by a factor 3.7 along the jet;
- a weak co-flow (5 % of $U_0$) needs a channel 48 $b_0$ deep and five flow-throughs to lose the start-up circulation,
  and the finer grid still recirculated.

The case is therefore a half-jet in a co-flow strong enough to supply the entrainment on every grid (15 % of $U_0$), and
the reference is the same jet in the same channel: the thin-shear-layer k-epsilon solution. Its free-jet limit (weak
co-flow, deep channel) carries the comparison with the measured spreading rate of the issue.

**Reference** (`plane_jet_thin_shear_layer`, `src/reference_solutions.jl`). The boundary-layer equations of the half-jet
with the k-epsilon model of `KEpsilonState` (the same constants, the Durbin (1996) time-scale bound, the $\varepsilon$
sources $C_1P/T$ and $C_2\varepsilon/T$), marched downstream (implicit in z, first order in x). A uniform pressure
gradient keeps the volume flux of the channel: it decelerates the free stream $U_e$ as the jet entrains (from 0.073 to
0.061 m/s here), and the excess momentum flux $J=2\int u(u-U_e)\,dz$ then follows the momentum integral
$dJ/dx=-(dU_e/dx)\int(u-U_e)\,dz$ (it rises by 2.6 % between 10 and 30 $b_0$). The march stops with an error if the free
stream comes to rest.

Free-jet limit (co-flow 4 % of $U_0$, depth 160 $b_0$, momentum length $J/U_a^2$ = 31 m): standard k-epsilon spreads at
$S=dy_{1/2}/dx$ = 0.0973 (15-40 $b_0$; 0.1009 over 10-30 $b_0$) with the centreline excess decaying as
$(x-x_0)^{-0.509}$: the measured 0.10-0.11 and $-1/2$ of plane jets (Pope 2000, ch. 5). With the time-scale bound, the
model default, $S$ = 0.0922 and $-0.513$: in a plane jet $(k/\varepsilon)|\partial u/\partial z|\approx3.3$ exceeds the
bound 2.72, which lowers $\nu_t$ and delays the development of the jet by about 10 $b_0$.

**Case** (`test/test_plane_jet.jl`, `experiments/jet/plane_jet.jl`). Slot $b_0$ = 0.05 m (half slot 0.025 m on the bed),
$U_0$ = 0.5 m/s ($U_0b_0/\nu$ = 25000), co-flow $U_a$ = 0.075 m/s, depth 1.2 m (24 $b_0$), length 2.25 m (45 $b_0$);
inflow turbulence intensity 5 %, length scale 0.1 $b_0$ (per-layer inlet k and $\varepsilon$, issue #42, so the
wetting-drying step); relaxed free outflow; tvd2; $U_0\Delta t/\Delta x$ = 0.5, $\Delta x=2\Delta z$; 150 s, the mean over
the last 30 s (the flow is steady from about 40 s). The free stream of a model profile is the velocity at the edge of the
jet, $z=3y_{1/2}$ (iterated), and $J$ is integrated up to that edge. WHY: the non-hydrostatic outer flow is not uniform
(the entrainment sink at the bed curves it: at 30 $b_0$, 0.069 m/s at 0.3 m against 0.056 m/s at the surface); with the
surface velocity as free stream the non-hydrostatic $J$ came out 15 % too high. The hydrostatic outer flow is uniform.

**Results** (relative to the reference at 10, 20, 30 $b_0$; $S$ and the decay exponent fitted over 10-30 $b_0$; reference
$S$ = 0.0649, exponent $-0.525$):

| run | $U_c$ | $y_{1/2}$ | $J$ | $S$ | exponent |
|---|---|---|---|---|---|
| hydrostatic, 3 cells across the half slot | -4.0, -2.5, -0.9 % | +7.5, +4.7, +1.8 % | +0.3 % | 0.0633 | -0.522 |
| non-hydrostatic, 3 cells | -4.1, -2.7, -1.1 % | +7.6, +4.8, +1.9 % | -0.1 to +0.1 % | 0.0633 | -0.522 |
| hydrostatic, 6 cells | -1.5, -1.2, -0.4 % | +3.6, +2.4, +1.0 % | +0.3 % | 0.0641 | -0.526 |
| non-hydrostatic, 6 cells | -1.6, -1.4, -0.6 % | +3.6, +2.5, +1.0 % | -0.1 to +0.1 % | 0.0641 | -0.527 |

- The model converges to the reference (the errors halve with the grid); hydrostatic and non-hydrostatic agree within
  0.2 % (a slender jet).
- The profiles collapse in similarity coordinates close to $\exp(-\ln2\,\eta^2)$.
- The co-flow lowers the spreading rate (0.065 against 0.092 in the free limit, both with the time-scale bound); the
  decay exponent stays close to $-1/2$, as it does for a strong and for a weak jet in a co-flow.
- Cost: the non-hydrostatic runs take 4.5 times the hydrostatic ones (3 cells 296 s, 6 cells 47 min), 70 % of it in the
  non-hydrostatic correction (a new Cholesky factorization every step; issue #71).

### 37) Column-line multigrid for the non-hydrostatic pressure of the wetting-drying step (issue #71)

**Why.** The non-hydrostatic pressure of the wetting-drying step was solved with a sparse Cholesky factorization in every
step (section 32): the pattern changes whenever the surface or a shoreline moves, and the numeric factorization grows
faster than the number of unknowns. In the plane jet of section 36 it was 70 % of a non-hydrostatic step (47 min for the
fine run).

**Tried first** (systems captured from the plane jet at 8.6e3, 7.7e4 and 7e5 unknowns, with AlgebraicMultigrid.jl and
Krylov.jl in a separate environment):
- algebraic multigrid (smoothed aggregation) preconditioned CG: 12-17 iterations independent of the size, but a V-cycle
  costs 14 matrix-vector products and the setup a third of the solve: about 1.5 times the direct time; with the
  hierarchy reused for 40 steps (it stays as good) and rtol 1e-6, 0.7 and about 0.5 of the direct time, but only with a
  fixed numbering of the unknowns (the jet's unknowns already change on the fine grid);
- CG preconditioned with the Cholesky factor of an earlier step: 7-12 iterations of a full solve, no faster (as in #10).

**Method** (`src/multigrid.jl`). Conjugate gradients with a V-cycle built for the z-level grid: line relaxation along the
strong and semi-coarsening along the weak direction (Trottenberg, Oosterlee & Schueller 2001, ch. 5; Dendy 1982):
- the vertical coupling of the cells ($1/\Delta z$) is much stronger than the horizontal one ($h_f/\Delta x^2$);
- smoother: block Gauss-Seidel over the columns, each column solved exactly (tridiagonal); a forward sweep before and a
  backward sweep after the coarse-grid correction keep the V-cycle symmetric;
- coarse levels: neighbouring columns merged in pairs, the layers kept, down to a single column (solved exactly);
- coarse operators: Galerkin products with piecewise-constant prolongation, the horizontal part halved. Merging two
  columns doubles the spacing, and the plain Galerkin product keeps the fine face coupling where the doubled spacing has
  half of it: with it the iterations grew with the grid (20, 34, 54 on 8.6e3, 7.7e4, 7e5 unknowns; over-correction made
  it worse), with the factor 1/2 they stay at 12-13;
- every level is a 5-point stencil on its (layers x columns) grid, stored as arrays; inactive cells (surface cell, dry
  and empty cells) have the diagonal 1 and no couplings and keep 0. The fine level takes its coefficients from the same
  loop that assembles the sparse matrix. The setup is a few sums per cell (8 ms on 7.7e4 unknowns), so the hierarchy is
  rebuilt every step and a changing pattern costs nothing.

**Options** (`ModelConfig`): `nonhydro_solver` = "auto" (default; the multigrid from `NH_MULTIGRID_MIN_UNKNOWNS` = 5000
unknowns on, below it the direct solver, which is faster there), "direct" or "multigrid"; `nonhydro_solver_tolerance`
= 1e-8 (relative residual of CG; the pressure then differs from the direct solution by about 1e-9 relative); and
`nonhydro_solver_max_iterations` = 200 (normally 7-17 are needed). CG starts from zero: a step does not depend on the
pressure of the last one (a re-initialised model takes the same step as a fresh one). If CG does not converge, the step
is solved directly (warned once per model; `wd.nh_fallbacks`). The residual of the pressure system is a divergence error
of the cells below the surface of the same relative size, far below the splitting error of the step (the repeated
free-surface solve changes the fluxes by more); volume conservation is not affected (the free-surface solve keeps it).

**Results** (`experiments/benchmarks/bench_nonhydro_solver.jl`; time per whole step, warm, Windows, one Julia thread):

| case | unknowns | direct (ms/step) | multigrid (ms/step) | ratio | CG iterations |
|---|---|---|---|---|---|
| jet basin 200 x 56 (section 31, the benchmark of #10) | 11000 | 20.2 | 18.0 | 0.89 | 13 |
| plane jet, 2 cells across the half slot | 8550 | 14.9 | 13.1 | 0.88 | 13 |
| plane jet, 6 cells | 77490 | 200 | 133 | 0.67 | 15 |
| plane jet, 18 cells | 699024 | 3237 | 1091 | 0.34 | 15 |

The solver alone (captured systems, rtol 1e-8): 0.63, 0.54 and 0.38 of the direct time at 8.6e3, 7.7e4 and 7e5
unknowns; below about 5e3 unknowns the direct solve takes well under a millisecond and the multigrid 3-7 times as long.

**Accuracy.** Per solve the multigrid solution is the direct one within 1e-9 (relative). In the plane jet the two
solvers differ by 1 % of the velocity after 50 steps: the start-up of the jet amplifies any perturbation (a perturbation
of 1e-9 of the initial velocity grows the same way with the direct solver), so whole runs are compared on linear or
steady flows: the standing-wave period (section 18) and the reattachment length behind the backward step (section 29)
agree within 3e-11 and 4e-11.

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

Reference condition (theta scheme, section 19; $\theta=1$ gives $w^{n+1}=\eta_t$ at the surface):

```math
w_{top}^{n+\theta} = \theta w_{top}^{n+1} + (1-\theta)w_{top}^n = \frac{\eta^{n+1}-\eta^n}{\Delta t}.
```

Validation section: in the default configuration (variable top layer) the surface value of $w^{n+\theta}$, integrated
from the layer continuity of $u^{n+\theta}$, matches $\eta_t$ in every step (`model.last_kinematic_mismatch`). With a
fixed top layer the stored $w$ of two consecutive steps satisfies the relation above as well. With the variable top
layer the stored $w^n$ was integrated with the top thicknesses of level $n-1$, so that second check is for the fixed
top layer only.

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

Validation section: $|\sum\eta\thinspace\Delta x|<10^{-12}$ m² and the kinematic mismatch $|w_{top}^{rec}-\eta_t|$ below
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

### test_verify_momentum_control_volumes_divergence_free

Purpose: the face velocities of the momentum control volumes satisfy the layer continuity (section 20).

Setup: random $u$ with closed walls, $w$ from the fixed-grid continuity (top value $\partial_t\eta\neq0$).

Validation section: the discrete divergence of every u control volume (faces `_face_velocity_x!`,
`_w_at_u_interfaces!`) and every interior w control volume (`_u_at_w_faces!`, `_face_velocity_z!`) is below
$10^{-11}$; contrast: the former interpolation via the layer centres gives a divergence above $10^{-5}$.

### test_verify_momentum_advection_stable_with_mean_flow_up_to_courant_one

Purpose: von Neumann analysis of the discrete 2DV `upwind1` operator with a mean flow (section 20).

Setup: periodic channel, fixed grid, uniform $U=0.5$ m/s, nx = 8, nz = 1, 2, 4; the linear part $L$ of the tendency
from central differences of `advection_term_u!` (exact, the tendency is quadratic in the perturbation).

Validation section: $\max\lvert\mathrm{eig}(I-\Delta tL)\rvert\le1+10^{-10}$ at $C=0.5$, 0.75 and 1. The former scheme
gave 1.15, 1.52 and 2 (8 of the 9 checks fail on it).

### test_verify_mean_flow_channel_perturbation_decays

Purpose: full-model regression test of issue #37.

Setup: periodic channel ($L=1$ m, $H=0.5$ m, nx = 16, nz = 8), `tvd2`, $\theta=0.7$, Fr = 0.5 (below
`theta_froude_max(0.7)` = 0.67), $C=0.3$, surface perturbation $10^{-8}$ m, 20 s, hydrostatic and non-hydrostatic.

Validation section: the surface perturbation and the velocity deviation from $U$ end below $10^{-8}$ (decay); the
former scheme blew up.

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
mean u over $15\le x\le25$ m averaged over the last 200 s, $n_z=5,10,20$, both modes.

Validation section: observed order of the maximum profile error above 1.8 (observed 1.98 and 1.88; errors
$1.3\cdot10^{-4}$, $3.4\cdot10^{-5}$, $9.2\cdot10^{-6}$ m/s, which equal the midpoint-rule offset $s\thinspace\Delta z^2/(24H)$ of
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

### test_verify_fixed_grid_reinitialization_clears_pressure_history

Purpose: an `initialize_*!` call on a model that has already run must give the same first step as a fresh model.

Reference: in the incremental pressure correction (van Kan 1986, section 19) the predictor carries the full
non-hydrostatic pressure gradient of the previous step ($q^n$ in `ws.p_corr`), so a stale $q^n$ would act as a spurious
force after a re-initialisation. Every `initialize_*!` therefore resets this history (`_reset_nonhydrostatic_history!`,
also `q_nh` of the wetting-drying mode; its `w_nh` is recomputed from $u^n$ every step, issue #38).

Validation section: after 10 non-hydrostatic lock-exchange steps `p_corr` is non-zero; re-initialised to uniform density,
one step matches a fresh model in $u$, $w$ and $\eta$ to $10^{-11}$.

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
$10^{-4}\thinspace\mathrm{m}$-amplitude surface wave's corrected non-hydrostatic pressure correction stays below 5 Pa (the uncorrected defect
gave $\sim3.7\times10^4\thinspace\mathrm{Pa}$ for the same case).

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
form on the same velocities violates constancy by more than $10^{-6}\thinspace\mathrm{kg/m^3}$.

### test_validate_rayleigh_taylor_linear_growth_rate

Analytical solution: a small interface displacement $a_0\cos kx$ between a heavy top layer (thickness $h_1$) and a light bottom
layer ($h_2$), released from rest, grows as $a(t)=a_0\cosh\sigma t$ with the inviscid growth rate

```math
\sigma^2=\frac{g\,k\,(\rho_{top}-\rho_{bottom})}{\rho_{top}\coth(kh_1)+\rho_{bottom}\coth(kh_2)}
```

(two layers between rigid horizontal boundaries; Chandrasekhar 1961, ch. X; the free surface acts as a rigid lid for this mode).
Setup: 8 m × 4 m, 200 × 80 cells, $\rho_{top}=1002.5$, $\rho_{bottom}=998.5\thinspace\mathrm{kg/m^3}$, interface at $0.82H$,
$a_0=0.02$ m, $m=3$ ($k=2.36$ m⁻¹), `tvd2` momentum and transport. The interface height is taken from the heavy-fluid thickness
of each column and projected onto $\cos kx$; the numerical rate is $\mathrm{acosh}(a/a_0)/t$ at $\sigma t=2$ ($ka\approx0.2$,
still linear). Required: $0.8\lt\sigma_{num}/\sigma\lt1.05$. Measured: 0.84 (0.87 on 400 × 160 cells; the smeared numerical
interface lowers the rate and converges slowly); without the top-row advective form 0.61, so the test detects that defect.

### Variable-thickness top layer (section 13)

All tests below use `test_verify_*` (code verification against analytical identities or solutions); tolerances are named at the top
of the block (`TOL_IDENTITY`, `TOL_MODEL_IDENTITY`, ...).

### test_verify_hydrostatic_pressure_top_half_layer

Purpose: `top_half_layer=true` gives the exact hydrostatic pressure at layer centres.

Analytical solution (uniform density): $p_k=\rho_0 g\thinspace\Delta z\thinspace(n_z-k+\tfrac12)$; it equals the legacy field plus $\rho_0 g\Delta z/2$.

### test_verify_constituent_mass_with_variable_top_layer

Purpose: constituent mass with the top-layer thickness $\Delta z+\eta$.

Analytical solution (uniform density): $M=\rho_0\Delta x\sum_i(H+\eta_i)$.

### test_verify_variable_thickness_barotropic_reduces_to_constant_depth

Purpose: `solve_variable_thickness!` with the thickness of a flat surface reproduces the constant-depth `solve!`.

### test_verify_variable_thickness_barotropic_closes_volume_budget

Purpose: verify the derivation of the variable-thickness free-surface system.

Identity: with the final velocities $u^{n+1}=\hat u-g\Delta t\thinspace\partial_x\eta^{n+1}$,

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
$10^{-6}\thinspace\mathrm{kg/m^3}$ (the defect that motivated section 13).

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
amplitude is tiny ($A=10^{-4}$ m): a finite-amplitude wave carries a crossing-time shift of about $0.15\thinspace A/H$ that would mask the
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
cells; with the variable top layer the surface elevation adds $\rho\thinspace\eta\thinspace\Delta x$ per column (48,000 kg/m for
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
density transport, physical viscosity, no Rayleigh damping and no Froude limiter. Require kinetic energy below 1.08 times
released anomaly potential energy at every sample (every 100 steps), top speed below sqrt(2 g' 2a), conserved volume and constituent mass,
bounded density, and round-off continuity and kinematic residuals. This covers the previously undocumented wall-return
phase; the 2 mm experiment sweep supplies the finer-grid check.

The wall-predictor enforcement added in PR #7 is essential: reversing just that ordering reproduces the reported
surface jet and artificial energy growth at 2 mm. The fixed top-cell geometry was not the cause of that defect.

Why 1.08 and not 1.05 (issues #34, #37): the largest ratio occurs at the first sample ($t^*=0.25$, kinetic energy below
2 % of its later maximum), not after the wall impact; it is 1.026 at $\theta=1$ and 1.061 at $\theta=0.7$, with and
without the issue #37 momentum-flux fix. At the start both energies are tiny and the ratio is not a converging
quantity: sampled every 25 steps it reaches 2.4 ($\theta=1$) to 2.7 ($\theta=0.7$) before $t^*=0.1$ and grows further
when $\Delta t$ is halved (2.65, 2.84), so it is not a time-integration error of the advection. The surface potential
energy is not part of the released energy. Issue #5 gave about 1.5 after the wall impact; the bound still detects that.

### test_verify_masked_poisson_manufactured_solution

Manufactured solution on the wet cells above a raised flat bed of height $d$:
$p=\cos(\pi x/L)\cos(\pi(z-d)/(H-d))$ has zero normal derivative at the side walls, the bed and the top, and
$\nabla^2p=-\pi^2(1/L^2+1/(H-d)^2)\thinspace p$ ($L=2$ m, $H=1$ m, $d=0.25$ m). The masked matrix of `build_poisson_matrix`
(gauge pin at the first wet cell, mean removal over the wet cells) reproduces it with maximum errors 5.1e-3, 1.3e-3 and
3.2e-4 on $32\times16$, $64\times32$ and $128\times64$ cells: observed orders 1.99 and 2.00 (required: 1.8-2.2).


### Vegetation tests (`test_vegetation.jl`, section 16)

- `test_verify_rigid_patch_drag_per_unit_volume`: kernel check, $-\tfrac12C_Da\thinspace u|u|$ per unit volume with the
  canopy-top cell fraction and the patch-edge weights, random velocities of both signs, to round-off.
- `test_verify_patch_stiffness_must_be_positive`: zero, negative and NaN bending stiffness of a patch are rejected.
- `test_verify_emergent_drag_up_to_top_of_momentum_layers`: emergent rigid stem and patch with a non-flat surface, fixed
  and variable top layer: stem force $\tfrac12\rho_0C_Db\thinspace u^2$ times the top of the momentum layers ($n_z\Delta z$, or
  $(n_z-1)\Delta z+h_{top}$ interpolated to the base), volume-integrated sink with the layer thicknesses equal to the stem
  force, and $\tfrac12C_Da\thinspace u|u|$ per unit volume in the top row of a patch, all to round-off.
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
  reference is badly conditioned. The reference gives $R\thinspace Ca^{1/3}=0.890$ at large $Ca$.
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
- `test_verify_explicit_drag_stability_check`: $\Delta t\thinspace K=2$ is rejected.
- `test_validate_wave_height_decay_over_rigid_canopy`: regular linear waves ($H_0/h=0.005$, $kh=0.2$) through an
  emergent canopy ($a=10$ 1/m) over 60 m, wavemaker at the left boundary, sponge at the right end. The wave height
  decays as $H_0/(1+\beta x)$ of Dalrymple, Kirby & Hwang (1984),
  $\beta=\tfrac{4}{9\pi}C_DaH_0k\thinspace(\sinh^3k\alpha h+3\sinh k\alpha h)/((\sinh2kh+2kh)\sinh kh)$ (derived from the wave
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
  $=gSH$, total stress at mid-depth $=gS(H-z)$ and wall-function $k_1=u_{\ast}^2/\sqrt{C_\mu}$ within $10^{-4}$ (observed
  $1.2\cdot10^{-5}$ after 600 s), depth-mean velocity within 8 % of the log law $U/u_{\ast}=(\ln(H/z_0)-1)/\kappa$
  (observed +6.0 %, consistent with the wake of channel flows); $\nu_t$ peaks at mid-depth ($z/H=0.53$).
- `test_verify_k_epsilon_vegetation_balance`: homogeneous emergent canopy;
  $\varepsilon/P_v\to C_1C_{f\varepsilon}/C_2=0.9975$ within $10^{-3}$ below the two surface rows (observed 0.9975).
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
  automatic default, the explicit variable top layer and wetting and drying (with implicit vertical terms) accepted,
  invalid choices (unknown closure, bed geometry without wetting and drying, wetting and drying without implicit
  vertical terms or with the mixing length, nz = 1) and a change of the closure across the k-epsilon boundary after
  construction rejected.
- `test_verify_k_epsilon_time_step_independence` (section 27): the steady channel with implicit vertical terms is the
  same for $\Delta t=0.1$ and 2 s ($10^{-4}$; observed five digits).
- `test_validate_k_epsilon_wetting_drying_open_channel`: the developed wetting-drying channel equals the periodic one
  ($2\cdot10^{-3}$: depth, velocity profile, max $\nu_t$, $k_1$, $u_\ast^2=gSH$).
- `test_validate_k_epsilon_variable_top_layer_open_channel`: open channel with the variable top layer and a top row
  1.4 times $\Delta z$: depth-mean velocity, $u_\ast^2$ and max $\nu_t$ within 1 % of the periodic channel.
- `test_verify_k_epsilon_wetting_drying_raised_bed`: a bed raised by whole layers gives the flat channel to round-off; a
  bed that cuts a layer the same depth-mean velocity and bed stress ($2\cdot10^{-3}$).
- `test_verify_k_epsilon_wetting_drying_moving_shoreline`: stratified dam break onto a dry slope, both modes: volume and
  density mass to round-off, k and $\varepsilon$ positive and finite, max $\nu_t<10^{-2}$ m²/s.
- `test_verify_k_epsilon_realizability_bound`: the time scale is $k/\varepsilon$ without strain and in equilibrium shear,
  the bound above; on by default.
- `test_verify_k_epsilon_realizability_switch`: without the bound the dam-break front exceeds $\nu_t=1$ m²/s within ten
  steps, with it $\nu_t$ stays below $10^{-2}$ m²/s.
- `test_verify_clear_vegetation_resets_k_epsilon_production`: the vegetation production $P_v$ is zero after
  `clear_vegetation!`.
- `test_verify_k_epsilon_equilibrium_wall_function_channel` (section 29): the wetting-drying channel of section 27 with
  the equilibrium wall function: on an interface the same as the local one and $k_1\sqrt{C_\mu}=gSH$ ($5\cdot10^{-3}$); with
  the bed $0.9\,\Delta z$ above an interface the depth within $10^{-3}$ and max $\nu_t$ within 2 % of the unshifted
  channel, the shaved cell has the $k$ and $\varepsilon$ of the wall cell; an unknown wall function is rejected.
- `test_verify_k_epsilon_stationary_richardson_number` (section 30): homogeneous stratified shear, 0D with the source and
  sink of the closure ($T=k/\varepsilon$): after 100 shear times $k$ grows at $Ri=0.9\,Ri_{st}$ and decays at
  $1.1\,Ri_{st}$, for $C_3=0$ ($Ri_{st}=0.175$) and $C_3=0.576$ ($Ri_{st}=0.25$); `ke_c3_for_richardson` inverts the
  relation; $C_3\ge C_2$ is rejected; the default stays 0.

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
- `test_verify_wd_tvd2_momentum_von_neumann`: the `tvd2` momentum value (`_wd_upwind_value`) applied to linear advection
  of a small sine on a linear ramp (so that the MC limiter stays in its smooth branch), $C=0.2,0.6,0.9$, wavelengths 4 to
  20 $\Delta x$: the measured growth per step equals the analytical $|G|$ of section 18 within $2\cdot10^{-3}$, with the time
  correction (at most 1) and without it (the unstable forward-Euler factor, e.g. 1.15 at $C=0.6$ and 4.6 $\Delta x$); and
  $\max|G|\le1$ with the correction for $C\le1$.
- `test_verify_wd_advection_depth_uniform_mms_{upwind1,tvd2}`: operator-level manufactured solution of section 18
  (depth-uniform $u$ under a surface crossing six interfaces): the maximum error over all layers converges with an
  observed order of at least 0.8 per grid halving and stays within twice the 1D `upwind1` error (observed: 2DV 1.2e-2 ...
  1.3e-3, 1D 7.8e-3 ... 9.7e-4 for $\Delta x=1\ldots0.125$ m).
- `test_verify_wd_negative_depth_is_rejected`: a level written below the bed stops the step with a `DomainError`.
- `test_verify_wd_cut_cell_output_masking`: the plot masking blanks the cells above the water level and dry columns (cut
  cells, fractional bed), and keeps the wet cells.
- `test_verify_wd_boundary_profiles_and_wet_flux`: a constant inflow profile replaces an earlier time-dependent
  callback; the `boundary_flux` diagnostic uses the wet face discharge ($-0.5$ m²/s for 1 m/s over 0.5 m of water in a
  2 m grid); callback and right-profile lengths other than nz raise an `ArgumentError`.
- `test_verify_wd_float32_dry_columns`: with `T=Float32` the dry levels are representable (depth $\ge0$ after
  initialisation and five steps); the Newton and round-off tolerances use the precision of the state type.
- `test_verify_wd_pressure_components_include_nonhydrostatic_correction`: in the non-hydrostatic mode `state.p` is the
  cut-cell hydrostatic pressure plus the solved non-hydrostatic pressure, and `pressure_buoyancy_components` returns
  these two components.
- `test_verify_wd_reinitialization_clears_nonhydrostatic_history`: `initialize_water_level!` on a model that has run
  five non-hydrostatic steps (non-zero `q_nh`, `w_nh`) gives the same next step as a fresh model ($u$, $w$, $\eta$ to
  $10^{-11}$; see `test_verify_fixed_grid_reinitialization_clears_pressure_history`).
- `test_verify_wd_nz1_pressure_diagnostics_output`: `run!` with one layer writes the pressure/buoyancy diagnostics (no
  interior interface).
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
  velocity at the offshore boundary: shoreline elevation within 10 % of the runup amplitude $A/4\thinspace sL=0.1$ m at
  $\Delta x=0.05$ m (observed 9.2 %), converging (14.6 % at 0.1 m), surface elevation of the wet cells within 5 % (1.8 %).
- `test_validate_wd_carrier_greenspan_runup_2dv`: 15 layers, same bounds (shoreline 9.2 %, surface 2.1 %).
- `test_validate_wd_carrier_greenspan_runup_tvd2`: the `tvd2` shoreline error at most 1 % above the `upwind1` one (14.05
  against 14.04 mm at $\Delta x=0.1$ m) and a surface error within 5 % (3.51 mm; `upwind1` 3.54 mm); both are dominated by
  the wet/dry front, where `tvd2` falls back to first order. Without the time correction of section 18 the shoreline was
  1 % better (13.88 mm), an effect of the anti-diffusive time error.
- `test_validate_wd_carrier_greenspan_runup_2dv_nonhydro`: long wave ($kH\ll1$): the non-hydrostatic mode within the
  hydrostatic bounds.
- `test_validate_wd_carrier_greenspan_runup_2dv_nonhydro_theta`: the same run at $\theta$ = 1/2, 0.55, 0.6 and 1: within
  the same bounds, and the surface errors / $A$ differ by at most 0.005 (observed 0.0211-0.0241, the value at 1/2 depends on round-off; 0.37 at $\theta=0.55$
  and a blow-up at 1/2 before issue #38, section 19).
- `test_validate_wd_nonhydro_standing_wave_full_dispersion_{upwind1,tvd2}`: closed basin, $kH=\pi/2$ (the shallow-water
  and the full period differ by more than 25 %): period error below 2 % on 80 x 40 cells (observed 0.89 %), smaller than
  on 40 x 20 (1.75 %), closer to $\omega^2=gk\tanh(kH)$ than to the shallow-water relation.
- `test_validate_wd_nonhydro_standing_wave_deep_water`: $kH=3$ (shallow-water period 42 % too short): period error
  below 3 % on 40 x 40 cells (observed 1.9 %). Over $kH=0.25\ldots3$ the phase speed is within 0.0-1.9 % of the exact one
  (`experiments/wetting_drying`, mode `dispersion`).
- `test_validate_wd_solitary_wave_runup`: non-breaking solitary wave, $H/d=0.019$, beach 1:19.85, against the maximum
  runup $R/d=2.831\sqrt{\cot\beta}\thinspace(H/d)^{5/4}$ of Synolakis (1987): 1D within 10 % at $\Delta x=0.05$ m (observed 6.2 %) and
  smaller than at 0.1 m (11.2 %); 2DV non-hydrostatic (12 layers, `tvd2`) within 15 % at 0.1 m (observed 12.9 %). The
  runup converges at first order to the law (1.00 at 0.0125 m); it is set by the wet/dry front.

### Sloped channel tests (`test_sloped_channel.jl`, section 21)

Channel of issue #14: C = 50 m$^{1/2}$/s, H = 4 m, $i_b=10^{-4}$, $\nu=0.02$ m²/s. The slope is a body force unless
stated otherwise.

- `test_verify_chezy_reference_values_of_issue_14`: $U=1$ m/s and $q=4$ m²/s. $U(T)=\tanh(1)U$. The depth average of
  `chezy_viscous_profile` is $U$ (Simpson, exact for the parabola). `chezy_roughness_length` inverts
  $C=(\sqrt g/\kappa)(\ln(H/z_0)-1)$.
- `test_verify_chezy_column_depth_integrated_balance`: random columns, flux form with the Chezy row:
  $\Delta Z^\top u=\Delta Z^\top\hat u-\Delta t\gamma\bar u$ and effective depth $H/(1+\Delta t\gamma/H)$ to round-off. The
  Sherman-Morrison solution equals a dense solve of $A+e_1v^\top$. The missing work vector is rejected.
- `test_verify_chezy_channel_profile_is_steady_state`: periodic channel, explicit and implicit. The discrete parabola
  with the Chezy mean stays to 1e-9 (200 steps); a perturbed profile returns to it within 1e-8 after 10 friction times.
- `test_verify_chezy_spinup_order_in_time`: spin-up from rest against $U_{eq}\tanh(t/T)$ at $t=1000$ s. Observed order
  1 (explicit) and 2 (implicit) within 0.05.
- `test_verify_top_row_uniform_flow_with_implicit_vertical_advection`: variable top layer with $h_{top}\ne\Delta z$ and $W\ne0$
  below it. The explicit top-row correction together with the implicit column keeps a uniform top row unchanged to
  round-off (before the fix it changed by $\Delta t\,uW(1/\Delta z-1/h_{top})$).
- `test_verify_level_boundary_keeps_open_channel_equilibrium`: discharge inlet, $\eta_R=0$; variable top layer, fixed
  grid and wetting-drying step, `upwind1` and `tvd2`. The normal flow stays to 1e-9 for 500 steps (with a closed
  boundary control volume it grew from round-off to O(1) within 3000 s).
- `test_verify_level_boundary_converges_to_normal_flow`: from 0.8 times the normal flow, within 1e-8 after 20000 s. Every
  step closes the volume budget $\Delta V=\Delta t(Q_{in}-Q_{out})$ with the boundary transports, and the kinematic
  surface condition holds in every column.
- `test_verify_level_boundary_fills_lake_to_prescribed_level`: no slope, closed inlet. A lake at the boundary level
  stays at rest to round-off; a lake 0.1 m below it fills to within 1e-4 m in 30000 s.
- `test_verify_discharge_boundary_transport`: the inflow face carries $q$ to round-off with the thicknesses of time $n$
  while the inlet level moves. With the discharge set again before every step, the θ flux that moves the level is $q$
  as well; in the wetting-drying step it was too small by up to 0.6 % before the fix.
- `test_validate_backwater_curve_matches_gradually_varied_flow`: $\eta_R=\pm0.3$ m (M1, M2), variable top layer and
  wetting-drying step, nx = 50, nz = 8. The steady depth is within 1e-4 m of `chezy_backwater_depth` (observed
  2-4e-5 m; with the old top-row vertical advection 3.3e-4 and 5.9e-4 m).
- `test_validate_sloped_bed_channel_normal_flow`: real bed slope in the wetting-drying step (no body force), discharge
  inlet with the parabolic shape, normal depth at the outlet. Depth and $U(L/2)$ within 1e-4 (relative) of the normal
  flow.
- `test_verify_open_boundary_config_checks` and `test_verify_chezy_config_checks`: invalid combinations are rejected.
  These are `right_level` together with `right_profile`, a non-finite level, explicit vertical terms,
  `periodic_x`, negative discharge shapes, negative $C$, Chezy with `bed_roughness_length` or
  k-epsilon, and Chezy on a staircase bed or with wetting and drying without implicit vertical terms.

### Open boundary tests (`test_open_boundaries.jl`, section 22)

- `test_verify_supercritical_channel_with_inlet_level_and_free_outflow`: Fr = 1.6 with inlet discharge and level and a
  free outflow. Normal flow stays to 1e-9, and returns from 0.8 times the velocity within 1e-6 (variable top layer,
  wetting-drying step). On the fixed grid, without an inlet level, the velocity returns within 1e-4 (neutral level).
- `test_verify_free_outflow_subcritical_converges_to_normal_flow`: #14 channel with a discharge inlet and a free
  outflow. Steady to round-off; from 0.8 times the normal flow within 1e-6 after 40000 s; volume budget and kinematic
  condition to round-off.
- `test_validate_free_outflow_transmits_long_wave`: a right-going long wave leaves with less than 2 % reflected
  (measured 0.7 %); with a prescribed level more than 40 % remains (full reflection, $\theta$-damped).
- `test_validate_free_outflow_over_real_bed_slope`: real #14 slope with cut cells and a free outflow; depth within
  1e-3 m of normal flow (measured 3.4e-4 m).
- `test_verify_boundary_froude_warnings`: a level at a supercritical outflow and an inlet level with subcritical inflow
  warn once; a supercritical inlet with a free outflow does not warn.
- `test_verify_open_boundary_46_config_checks`: rejected combinations are
  - `left_level` without discharge, non-finite, or on the fixed grid;
  - `right_free` together with `right_level` or `right_profile`;
  - `right_free` with explicit vertical terms.
- Non-hydrostatic mode (section 24):
  - `test_verify_nonhydro_open_channel_uniform_flow_is_steady`: fixed grid (sparse, spectral, Dirichlet top) and
    wetting-drying step; $u$ and $\eta$ stay to 1e-9 for 300 steps.
  - `test_validate_nonhydro_free_outflow_transmits_dispersive_wave`: the reference (a twice as long channel) shows
    the dispersive trough (> 10 % of $A$; fixed grid with the Dirichlet top, wetting-drying step). Reflection is below
    3 % for the dispersive wave (measured 1.34 % on the fixed grid with the Dirichlet top, 1.15 % in the wetting-drying
    step) and below 1 % for the long wave of the Neumann top (measured 0.23 %). A prescribed level reflects more than
    15 %. Also checks the volume budget, and with the Neumann top the kinematic condition, every step.
  - `test_verify_nonhydro_open_boundary_any_pressure_solver`: the spectral and the sparse solver are accepted.
- Relaxed free outflow (section 25):
  - `test_verify_relaxed_outflow_limits`: $\tau=\infty$ equals the free outflow bit for bit; $\Delta t/\tau=10^8$ equals
    the prescribed level within $10^{-6}A$ (measured $2\cdot10^{-9}$); fixed grid and wetting-drying step.
  - `test_validate_relaxed_outflow_reflection_matches_theory`: reflected record against the prescribed-level record
    filtered with $1/(1+2i\Omega\tau)$, $\tau$ = 2 to 50 s; difference below 2 % of $A$ (measured 0.4-1.4 %).
  - `test_validate_relaxed_outflow_mean_level_returns`: decay rate of a level offset in a closed basin against the
    lowest mode, within 2 % (measured 0.2 %).
  - `test_verify_relaxed_outflow_uniform_flow`: #14 channel, steady and returning from 0.8 times the normal flow to
    1e-9, fixed grid, variable top layer and wetting-drying step.
  - `test_validate_relaxed_outflow_nonhydro_dispersive_wave`: the dispersive wave of section 24 with $\tau$ = 50 s
    reflects below 3 % (measured 1.6 % and 1.4 %); volume budget every step.
  - `test_verify_relaxed_outflow_config_checks`: `right_free` with `right_level` needs $\tau$; $\tau$ needs both;
    $\tau \le 0$ or NaN is rejected; `clear_x_inflow_outflow!` resets $\tau$.

### Obstacle tests (`test_obstacles.jl`, section 23)

- `test_verify_bump_reference_solutions`: properties of the reference functions.
  - Critical depth has Fr = 1; both Bernoulli roots have the same specific energy.
  - `conjugate_depth` is an involution, and a head below $\tfrac32h_c$ is rejected.
  - The crest depth of the transcritical case is $h_c$, to $10^{-7}$: at the energy minimum round-off only fixes $h$
    to $\sqrt\varepsilon$.
  - The SWASHES jump position is 11.6656 m, with conjugate depths on both sides.
- `test_validate_swashes_bump_subcritical`: 1D and 2DV within 1.5 cm (nx = 125); the discharge is steady; the observed
  order from nx = 125 and 250 is above 0.8.
- `test_validate_swashes_bump_transcritical`: 1D and 2DV within 3 cm, supercritical downstream of the bump.
- `test_validate_swashes_bump_shock`: 0.5 m away from the jump within 2 cm (1D) or 3 cm (2DV, $\nu=10^{-3}$ m²/s);
  jump position within 0.1 m (1D); steady discharge.
- `test_verify_tracer_through_bump_monotone_and_conserved`: all three cases, 1D and 2DV. The tracer budget closes to
  round-off every step, the values stay within the inflow and initial densities, and the front passes the crest.
- `test_validate_broad_crested_weir_free_and_submerged`: 1D and 2DV. In free flow the weir law gives $q$ from the
  upstream head within 2 % (measured 1.25 %), the same for two tailwaters; submerged, the upstream energy is the
  tailwater energy within 3 mm.
- `test_verify_inflow_density_config_checks`: `left_rho` is rejected without wetting and drying and when not finite,
  and it widens the transport bounds.

### Sluice gate tests (`test_sluice_gate.jl`, section 26)

- `test_verify_sluice_gate_reference_solutions`: the free head satisfies $q=C_ca\sqrt{2g(E_1-C_ca)}$; the modular limit
  is the conjugate depth of $C_ca$; the submerged head is continuous there, larger above it and the free head below;
  $C_c$ is recovered from the head.
- `test_verify_sluice_gate_inactive_is_bitwise_identity`: a gate above the water surface, and a gate removed by
  `clear_sluice_gates!`, give the same $u$, $\eta$ and $w$ as no gate, bit for bit (200 steps, 4 layers).
- `test_verify_sluice_gate_closed_holds_the_water`: a gate down to the bed keeps its face closed; the volume upstream
  grows exactly by the inflow, and the tailwater stays at rest (1D and 4 layers, $10^{-9}$).
- `test_validate_sluice_gate_free_flow`: upstream energy head within 0.2 % of $C_ca+q^2/(2g(C_ca)^2)$ for
  $C_c=0.61$ and $C_c=1$, 1D and 2DV (measured 0.03-0.04 %); steady discharge within 0.5 %; volume budget to round-off.
- `test_validate_sluice_gate_submerged_flow`: tailwater 0.28 and 0.34 m (above the modular limit 0.245 m), 1D:
  upstream head within 0.2 % of the momentum theory (measured 0.012 % and 0.001 %).
- `test_validate_sluice_gate_nonhydro_contraction`: non-hydrostatic with `contraction=1`, 12 layers: $0.61<C_c<0.85$
  (measured 0.82); the hydrostatic step on the same grid gives $C_c>0.98$ (measured 1.00).
- `test_verify_sluice_gate_jump_reference_solutions`: the jet profile equals the closed-form H3 curve on the horizontal
  bed and satisfies the gradually-varied-flow equation on a mild slope (central difference, $10^{-6}$); at the jump the
  conjugate depth of the jet equals the subcritical depth; a higher outlet moves the jump towards the gate; the jump
  position scales with $C^2$ (frictionless limit); submerged and washed-out cases throw.
- `test_validate_sluice_gate_hydraulic_jump`: Chezy $C=50$, outlet depths 0.16, 0.18 and 0.20 m, 1D and 2DV
  (8 layers): jump position within one cell (1D) and three cells (2DV) of the theory, in the theoretical order; depths in the jet
  within 0.5 %, upstream and behind the jump within 0.5 % (1D) and 1 % (2DV); steady discharge within 0.5 %; the
  position error with $\Delta x/2$ is below 0.6 times that with $\Delta x$ (measured 0.49).
- `test_validate_sluice_gate_nonhydro_jump`: non-hydrostatic, `contraction=1`, $C=40$, outlet 0.15 m, 320 x 24 cells
  (four layers in the opening; with two the jet does not contract, see above), 120 s: steady discharge within 0.5 %; with
  the theory from the vena contracta of the model, the jump (half its height) within 2.5 $h_2$ (measured 0.8 $h_2$), the
  jet depth within 3 % (measured 0.9 %) and the depths behind the jump within 4 % (measured 3.4 %).
- `test_verify_sluice_gate_config_checks`: positions off a u-face or on the boundary, an edge below the bed, a
  contraction outside (0, 1], a contraction below 1 in the non-hydrostatic mode and the fixed-grid step are rejected.

### Recirculation tests (`test_recirculation.jl`, section 28)

- `test_verify_wd_horizontal_eddy_diffusion_second_order`: MMS $u=\sin(2\pi x/L)$, constant $\nu_t$ and $h$: the term is
  $-\nu_t(2\pi/L)^2u$, order above 1.9 on 40, 80, 160 cells.
- `test_verify_wd_horizontal_turbulence_diffusion_second_order_and_conservative`: without flow, $\phi=2+\cos(\pi x/L)$:
  the update is $\Delta t\,\nu_t\partial_x^2\phi$, order above 1.9; the sum of $\phi$ is unchanged with closed walls.
- `test_verify_wd_horizontal_diffusion_limits`: with $\nu_t=10^6$ m²/s the update of $u$ is limited to
  `WD_HORIZONTAL_DIFFUSION_MAX`, that of $\phi$ stays bounded, non-negative and conservative; at a closed neighbouring
  face only the open side contributes.
- `test_verify_reattachment_length`: a bed velocity with a known zero is recovered; attached flow gives 0.
- `test_validate_backward_step_reattachment_nonhydrostatic`: ST1 on S/5, 90 s, mean of the last 30 s: $x_R$ within 20 %
  of the LES value 6.15 S (measured 6.07 S).
- `test_validate_backward_step_hydrostatic_vertical_velocity`: same case: the largest $w$ of the hydrostatic run is above
  3 times the non-hydrostatic one (measured 3.6; 4.3 before the time correction of issue #66).
- `test_validate_backward_step_reattachment_shifted_bed` (section 29): the same case with the bed and the step
  $0.9\,\Delta z$ above the layer interfaces (lowest wet cell $0.1\,\Delta z$): $x_R$ within 10 % of the run with the bed
  on an interface (measured 6.0 %: 5.70 against 6.07 S; the local wall function gives 38 % and no steady state).

Time-dependent discharge and the mode of the reach (`test_open_boundaries.jl`, issue #61):

- `test_verify_left_discharge_function`: a ramp reaches the inlet face at every step; the volume follows its sum to
  rounding (closed end, $\theta=1$); a constant function equals the constant discharge exactly; non-finite values and the
  fixed grid are rejected.
- `test_verify_advective_courant_warning`: silent below the threshold, once above it; the largest value is recorded;
  `advective_courant_warning = Inf` switches it off, a value <= 0 is rejected. Tests that run at a large step on purpose
  (positivity, conservation, order in time, uniform flow) switch it off with a `# WHY:` comment.
- `test_verify_reach_mode_gain`: the value of ST1 (1.41), 1 for a uniform channel and without mean flow, below 1 with the
  fast end at the outlet, NaN for outflow and supercritical flow.
- `test_validate_reach_mode_grows_with_velocity_inlet`: reach of ST1, one layer, no closure and no friction: the swing of
  the outlet discharge grows per period within 0.2 of 1.41 (measured 1.48), and the warning is given.
- `test_validate_reach_mode_decays_with_discharge_inlet`: same reach with a ramped discharge: the swing decays by more
  than 0.68 per period (measured 0.46; the head loss at the step adds to the estimate), no warning. No warning with the
  relaxed outflow.

### Splitter plate tests (`test_splitter_plate.jl`, section 30)

Reduced splitter plate: 4 m channel, 4 + 4 layers of 5 cm, 0.3 and 0.5 m/s, 1031 and 1016 kg/m³, k-epsilon, relaxed
outflow.

- `test_verify_inflow_density_per_layer_enters_the_first_cell`: one transport step with and without `left_rho` per layer
  from the same state: the first cell differs by $\Delta t/\Delta x\,q_k(\rho_{in,k}-\rho_{1,k})/h$ to round-off, no other
  cell differs. (A steady run is no test of this: the density difference drives a baroclinic flow at the interface.)
- `test_verify_inflow_k_eps_per_layer_enter_the_first_cell`: the same for the advection of $k$ and $\varepsilon$ with
  `left_k`, `left_eps`.
- `test_verify_two_streams_density_bounds_and_salt_budget`: 60 s: the density stays within the two inflow densities in
  every step, and the mass of the density excess over the upper stream changes by the boundary fluxes within
  $10^{-9}$ of the inflow (measured to round-off).
- `test_validate_two_streams_steady_discharge`: mean outflow over the last 20 s within 1 % of the inflow.
- `test_verify_inflow_per_layer_argument_checks`: wrong lengths, `left_k` without `left_eps`, non-positive values and
  `left_k` without k-epsilon are rejected; `clear_x_inflow_outflow!` clears them.

### Sediment tests (`test_sediment.jl`, section 31)

- `test_verify_sediment_settling_column`: 2 m column at rest, no turbulence, $c_0=0.5$ kg/m³, $w_s=1$ mm/s: at
  $t=0.2H/w_s$ the suspended mass equals $c_0(H-w_st)$ within $10^{-4}$ (measured $1.4\cdot10^{-6}$), suspended plus
  deposited mass is conserved to round-off, $c\ge0$, the deposit is uniform.
- `test_verify_sediment_erosion_partheniades`: one step of the channel: $E=M(\tau_b/\tau_{ce}-1)$ to round-off; a bed
  mass of $0.1\,M\Delta t$ limits $E$ to $0.1\,M$ and keeps the bed non-negative.
- `test_verify_sediment_mass_budget`: channel with erosion and clean inflow, 200 steps: suspended mass against inflow -
  outflow + erosion - deposition within $10^{-9}$ (measured $2\cdot10^{-14}$); the bed change matches deposition minus
  erosion.
- `test_verify_sediment_clean_water_enters_at_the_open_boundary`: reversed flow at the right face: the last cell differs
  from the zero-gradient case by $\Delta t/\Delta x\,q(c_{out}-c_{nx})/h$, no other cell differs, no sediment enters.
- `test_verify_sediment_density_coupling_equals_salt`: non-hydrostatic lock exchange driven by 2 kg/m³ of sediment
  ($w_s=0$) against the same driven by the salt density $\rho+c(1-\rho/\rho_s)$: the velocities and densities agree
  within $10^{-9}$.
- `test_validate_sediment_rouse_profile`: uniform channel (1 m, 1 m/s, $z_0=1$ mm, $w_s=2$ cm/s, $P=0.52$), 300 s:
  $c(z)/c(H/2)$ within 10 % of the Rouse profile at $z/H$ = 0.15, 0.3, 0.6, 0.75 (measured $-5.4$ to $+4.8$ %); erosion
  and deposition balance within 5 %.
- `test_verify_sediment_argument_checks`: negative settling velocity, zero critical stress, negative concentration, wrong
  inflow length, particles lighter than the water, no wetting and drying, no bed roughness are rejected.

### Performance tests (`test_performance.jl`, section 32)

- `test_verify_performance_report`: a model built after `reset_performance!` is recorded; after 5 steps its record
  counts 5 steps with a positive time and its label names grid and mode; the report lists wall-clock, CPU, compilation
  and GC time, allocations, the model and the merged sections; a model without steps is not listed, models beyond the
  table rows are summed in one row; `reset_performance!` empties the list.
- `test_verify_performance_format_bytes`: bytes, KiB, MiB, GiB.
- `test_verify_wd_pressure_factor_reuse`: non-hydrostatic dam break over a slope, 150 steps, with the reuse of the
  factorization and with a new analysis in every step: u, eta and q identical bit for bit; the front changed the
  pattern (more than one analysis), but not in every step.
- `test_verify_wd_step_allocations`: hydrostatic wetting-drying step with k-epsilon, sediment and an inflow: below
  16 KiB per step (measured 0.4 KiB; 300 KiB before issue #10, when the untyped fields boxed every cell).

### Visualization tests (`test_visualization.jl`, section 32)

- `test_verify_plots_fixed_grid`, `test_verify_plots_bed_with_solid_cells`, `test_verify_plots_wetting_drying_with_vegetation`:
  the final-state plot, the animation (3 frames) and the pressure-buoyancy diagnostics are written for a small
  non-hydrostatic fixed grid, a bed step with solid cells and a wetting-drying slope with a stem; every file exists and
  is not empty. The other tests switch the plots off; these keep the plotting code covered.

### Dam-break tests (`test_dam_break.jl`, section 33)

- `test_verify_stoker_reference_solution`: jump conditions of mass and momentum at the bore and the Riemann invariant
  through the rarefaction to round-off; still water ahead of the rarefaction and beyond the bore, the rarefaction joins
  the middle state continuously; the released volume fills the channel (quadrature, $10^{-5}$); $h_1=0$ gives Ritter;
  $h_1\ge h_0$, $h_1=0$ in `stoker_middle_state` and $t\le0$ are rejected.
- `test_validate_stoker_wet_dam_break_1d`: at $\Delta x$ = 0.25 m L1 below 1 % and at most 0.8 times the error at
  0.5 m, bore within 1 %; tvd2 better than upwind1, bore within 1 %.
- `test_validate_stoker_wet_dam_break_2dv_hydrostatic_nz{2,3,5,10}`: L1 below 1 %, bore within 2.5 %, depth within 1 %
  of the 1D depth, layer velocities within 0.1 m/s.
- `test_validate_stoker_wet_dam_break_2dv_nonhydrostatic_nz{2,3,5,10}`: L1 below 1.5 %, bore within 5 %, depth within
  1.5 % of the 1D (hydrostatic) depth.
- `test_validate_ritter_dry_dam_break_2dv_{hydrostatic,nonhydrostatic}_nz{2,3,5,10}`: L1 below 3 %, front within 12 %
  (the bounds of test_wetting_drying.jl), hydrostatic layer velocities within 0.01 m/s.
- `test_verify_dam_break_constituent_mass_{stoker,ritter}_{hydrostatic,nonhydrostatic}`: 5 layers, passive tracer:
  mass conserved to $10^{-12}$, depth and velocity identical to the run without tracer, $0\le c\le1$ kg/m³.
- Every run checks the water volume ($10^{-12}$) and finite velocities.

### Internal-seiche tests (`test_internal_seiche.jl`, section 34)

- `test_verify_two_layer_internal_wave_relation`: the root satisfies the linear two-layer system (interface kinematics
  and pressure continuity, residuals below $10^{-10}$) for three geometries, lies within $2g'/g$ of the rigid-lid relation,
  is negative (counter-phase); the hydrostatic root satisfies its quartic and the upper-layer mass balance, is the
  long-wave limit of the non-hydrostatic one ($kH=10^{-3}$, $10^{-6}$); the rigid-lid formulas give the 61.0 s and 46.6 s
  of the issue, the free surface changes them by less than $g'/g$; invalid arguments are rejected.
- `test_verify_baroclinic_extrapolation_oscillator`: without the extrapolation the oscillator model grows for
  $\theta$ = 0.5, 0.55, 0.7, 0.9 at $s$ = 0.05 to 0.8; with $\beta=2(1-\theta)$ it does not; $\theta=1$ needs none.
- `test_verify_baroclinic_extrapolation_history`: at $\theta=1$ the predictor uses $\rho^n$ itself; a re-initialised
  model takes the same next step as a fresh one.
- `test_validate_internal_seiche_period_{hydrostatic,nonhydrostatic}`: three grids: period error on the finest grid
  $\le$ 1.5 % / 2.5 % (observed 1.00 % / 1.86 %), dropping by at least 0.7 per refinement (0.57-0.61), amplitude kept per
  period $\ge0.98$, $|u|\le$ 0.02 m/s (stable on 160 x 160), volume to $10^{-12}$ m², salt mass to $10^{-12}$; on the
  finest grid the surface/interface ratio of the hydrostatic relation within 1 % (both modes, Neumann top condition).
- `test_validate_internal_seiche_surface_ratio_nonhydrostatic_dirichlet`: with the Dirichlet top condition the
  non-hydrostatic ratio within 15 % of the relation on 80 x 80 (observed 8 %).

### Kelvin-Helmholtz tests (`test_kelvin_helmholtz.jl`, section 35)

- `test_verify_kelvin_helmholtz_tanh_mode`: Michalke's 0.1897 $U/\delta$ at $k\delta=0.4446$ (walls at 20 $\delta$), a
  maximum there, $c_r=0$, step convergence ($10^{-8}$), the $U/\delta$ scaling, the rate falling with $J$ and below
  0.03 of the unstratified one at $J=0.249$, $k\delta=1/2$; the eigenfunction meets the walls; no unstable mode beyond
  the neutral wavenumber; invalid arguments are rejected.
- `test_verify_shear_layer_mode_initial_state`: the seeded velocity is discretely divergence-free ($10^{-14}$ 1/s), the
  periodic seam face equal to the first face, $w=0$ at the bed and the top, the density within its layer values; the
  initializer returns the rate of the reference and rejects a non-periodic model.
- `test_validate_kelvin_helmholtz_growth_rate`: $J$ = 0.05 on 8 and 16 cells per $\delta$: rate error $\le$ 1.5 % and
  0.8 % (observed 1.02 %, 0.52 %), dropping by at least 0.7 (0.51); salt mass and volume to $10^{-12}$; $J$ = 0 on 16
  cells: within 0.8 % (0.46 %) and faster than the stratified run.

### Plane-jet tests (`test_plane_jet.jl`, section 36)

- `test_verify_plane_jet_thin_shear_layer`: the change of $J$ over 10-30 $b_0$ follows the momentum integral within 5 %
  (2.9 %; the change is 2.6 % of $J$), the volume flux of the channel is kept to round-off, half the cells change the
  centreline excess and the half-width by less than 2 % (1.3 %), a co-flow that cannot feed the jet and invalid arguments
  are rejected.
- `test_validate_plane_jet_spreading_rate`: free-jet limit of the reference: standard k-epsilon $S$ in 0.095-0.115
  (0.0973), with the time-scale bound below it (0.0922), decay exponent within 0.03 of $-1/2$ ($-0.509$, $-0.513$).
- `test_validate_plane_jet_against_thin_shear_layer_hydrostatic`: 3 cells across the half slot: centreline excess within
  5 %, half-width within 10 %, $J$ within 3 % of the reference at 10, 20, 30 $b_0$, no backflow, exponent within 0.03 of
  $-1/2$.
- `test_validate_plane_jet_against_thin_shear_layer_nonhydrostatic`: 2 cells (3 cells take 5 min): 8 %, 15 %, 3 %
  (observed $-6.8$, $-4.8$, $-3.8$ %; $+11.1$, $+7.7$, $+2.4$ %; within 2 %), exponent within 0.05 ($-0.535$).
- `test_validate_plane_jet_hydrostatic_equals_nonhydrostatic`: 2 cells: centreline excess and $J$ within 3 % (1.8 %,
  2.4 %).
- The file takes about 3.5 min (three model runs of 150 s).

### Multigrid tests (`test_nonhydro_multigrid.jl`, section 37)

- `test_verify_multigrid_operator_and_solution_{jet,stoker,beach}`: after 20 steps with the multigrid (a plane jet, a
  Stoker dam break with its moving surface, a wave on a beach with columns falling dry): the fine-level operator is the
  assembled matrix to round-off, the solution the direct one within 1e-7 (observed 1e-9), the true residual within the
  tolerance, at most 20 iterations, no fallback.
- `test_verify_multigrid_iterations_independent_of_grid`: the jet on two grids: the finer one needs at most 4 iterations
  more (observed 13 and 14).
- `test_verify_multigrid_lake_at_rest`: a lake over a wavy bed with the multigrid: q = 0 and u = 0 exactly.
- `test_verify_multigrid_fallback_to_direct`: with 2 iterations allowed the direct solver takes over: the step equals
  the direct one bit for bit.
- `test_verify_nonhydro_solver_options`: invalid options are rejected; "auto" uses the direct solver below 5000 unknowns
  and the multigrid above.
- With the multigrid forced: `test_validate_wd_nonhydro_standing_wave_full_dispersion_*` (period within 1e-6 of the
  direct run) and `test_validate_backward_step_reattachment_nonhydrostatic_multigrid` (reattachment length within 1e-6
  of the direct run and within the bound of the LES).

## Additional validation scripts and literature basis

The repository includes standalone validation scripts under `experiments/validation`:

- `validate_surface_dam_break_front_speed.jl`: compares gravity-wave front speed with $c=\sqrt{gH}$ before reflection.
- `validate_lock_exchange_front_speed.jl`: checks fitted lock-exchange front-speed coefficient against literature-informed range.
- `validate_wind_shear_transport_mixing.jl`: checks stress-velocity scaling and density-variance reduction.

These scripts extend unit/regression testing with case-level physics checks tied to reference solutions.
