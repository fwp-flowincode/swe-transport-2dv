# 2DV SWE Test Technical Documentation

This document provides technical background, governing equations, and literature context for all test blocks of the test suite.
`test/runtests.jl` includes one file per functional area. Shared packages and tolerances live in `test/test_common.jl`, which
every file includes, so each file also runs on its own (`julia --project=. test/test_advection.jl`):

| File | Area |
|---|---|
| `test_operators_and_diagnostics.jl` | state layout, grad/div operators, viscous diffusion (section 5), buoyancy, diagnostics |
| `test_advection.jl` | density (transport) and momentum advection, `upwind1` and `tvd2` (sections 3, 4) |
| `test_pressure.jl` | hydrostatic integration, Poisson matrix, non-hydrostatic pressure solvers (sections 6, 7) |
| `test_free_surface.jl` | barotropic solve, kinematic surface condition, continuity closure, wind forcing budgets (sections 8-10) |
| `test_model.jl` | full-model runs: conservation, stabilisation, configuration, initial conditions, dense front, wind-driven double gyre |
| `test_vegetation.jl` | flexible-vegetation stem model |
| `test_nonhydrostatic.jl` | non-hydrostatic branch and free surface (sections 14, 15) |
| `test_variable_top_layer.jl` | variable-thickness top layer (section 13) |

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
mixing-length eddy viscosity $\nu_t=l^2|\partial u/\partial z|$, capped at `max_turbulent_viscosity`, and the vegetation
contribution) in flux form with face viscosities $\nu_f=\tfrac12(\nu_a+\nu_b)$:

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
grid that the variable top layer replaces with a different one (cross-referenced above): `test_hydrostatic_continuity_and_surface_closure_residuals`
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
`test_nonhydrostatic_surface_wave_period_matches_shallow_water_not_full_dispersion` (a domain with $k H\approx1.57$, chosen so the
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

## Test-by-test documentation

### test_state_shapes

Purpose: verify staggered-array storage layout for $u$, $w$, $p$, $\rho$, and $\eta$.

Numerical basis: C-grid style staggering requires:

```math
u \in \mathbb{R}^{n_z \times (n_x+1)},\quad
w \in \mathbb{R}^{(n_z+1) \times n_x},\quad
p,\rho \in \mathbb{R}^{n_z \times n_x},\quad
\eta \in \mathbb{R}^{n_x}.
```

Validation section: shape consistency ensures discrete divergence/gradient operators are conformable.

### test_grad_and_div_shapes_and_consistency

Purpose: verify centered gradient operators and combined divergence structure.

Reference equations:

```math
\nabla_h p = \left(\frac{\partial p}{\partial x}, \frac{\partial p}{\partial z}\right),
\qquad
\nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x} + \frac{\partial w}{\partial z}.
```

Validation section: for a linear $p(x)$, discrete $\partial p/\partial x$ is constant and $\partial p/\partial z=0$.

### test_density_advection_zero_for_uniform_density

Purpose: confirm density advection operator vanishes for spatially uniform tracer.

Reference equation:

```math
\nabla\cdot(\rho\mathbf{u}) = \rho\,\nabla\cdot\mathbf{u} + \mathbf{u}\cdot\nabla\rho.
```

For constant $\rho$ and divergence-free velocity, RHS is zero.

Validation section: exact discrete near-zero tendency threshold is used.

### test_density_advection_conservative_closed_box

Purpose: verify closed-box conservative transport of constituent mass.

Reference invariant:

```math
M_\rho = \int_\Omega \rho\, d\Omega
```

must remain constant with impermeable boundaries.

Validation section: compares $M_\rho^{n+1}$ vs $M_\rho^n$ at tight tolerance.

### test_hydrostatic_pressure_monotone_with_depth

Purpose: validate hydrostatic pressure solver monotonicity and free-surface reference condition.

Reference equation:

```math
\frac{\partial p_h}{\partial z} = -\rho g,
```

with pressure fixed at the top cell in the implemented convention.

Validation section: checks top boundary and monotone increase with depth.

### test_barotropic_solver_zero_flow_preserves_eta

Purpose: ensure barotropic free-surface solve preserves constant elevation when depth-integrated transport is zero.

Reference equation:

```math
\frac{\partial \eta}{\partial t} + \frac{\partial U}{\partial x}=0,
\quad U=\int u\,dz.
```

Validation section: if $U=0$, then $\eta$ remains unchanged.

### test_model_step_conservation_and_bounds_short_run

Purpose: short-run system conservation and boundedness checks.

Checked invariants:

```math
\Delta V \approx 0, \quad \Delta M_\rho \approx 0, \quad \Phi_{\partial\Omega} \approx 0.
```

Validation section: includes lower/upper bounds against initial density extrema.

### test_froude_diagnostic_behaviour

Purpose: verify finite positive internal Froude diagnostic.

Reference scaling:

```math
Fr = \frac{U}{c_i}, \quad c_i \sim \sqrt{g' H}.
```

Validation section: positivity and finiteness in stratified flow state.

### test_diagnostics_boundary_flux_closed_streamfunction_field

Purpose: check zero net normal flux for streamfunction-generated closed flow.

Reference identity:

```math
\mathbf{u} = \left(\frac{\partial \psi}{\partial z}, -\frac{\partial \psi}{\partial x}\right)
\Rightarrow
\nabla\cdot\mathbf{u}=0.
```

Validation section: integrated boundary-normal flux is near zero.

### test_poisson_matrix_and_nonhydrostatic_zero_rhs

Purpose: verify Poisson operator structure and zero-RHS pressure-correction behavior.

Reference equation:

```math
\nabla^2 p' = \frac{\rho_0}{\Delta t}\nabla\cdot\mathbf{u}^{\ast}.
```

Validation section: matrix diagonals for boundary/topology and zero solution for divergence-free predictor field.

### test_pressure_solver_spectral_matches_sparse_reference

Purpose: cross-validate spectral and sparse solvers.

Reference requirement: both methods should produce equivalent pressure gradients for same RHS.

Validation section: compares $\nabla p'$ components with tolerance.

### test_hydrostatic_pressure_vectorized_matches_reference_loop

Purpose: ensure vectorized hydrostatic integration matches loop baseline exactly.

Reference equation:

```math
p_k = p_{k+1} + \frac{1}{2}(\rho_{k+1}+\rho_k)g\Delta z.
```

Validation section: elementwise equality at tight absolute/relative tolerances.

### test_buoyancy_sign_heavy_down_light_up

Purpose: confirm buoyancy sign convention.

Reference equation:

```math
b = -g\frac{\rho-\rho_{ref}}{\rho_{ref}}.
```

Validation section: heavy parcels produce negative buoyancy, light positive.

### test_pressure_component_decomposition_matches_total_pressure

Purpose: ensure pressure decomposition consistency.

Reference relation:

```math
p = p_h + p'.
```

Validation section: reconstructed total pressure equals model stored pressure.

### test_model_froude_limiter_short_run

Purpose: verify limiter constrains Froude to configured maximum.

Reference policy:

```math
Fr_{new} \le Fr_{max}.
```

Validation section: tracks maximum diagnostic Froude through run.

### test_requested_domain_grid_dimensions

Purpose: enforce requested geometric discretization.

Validation section:

```math
\Delta x = L/n_x,\quad \Delta z = H/n_z.
```

### test_requested_salinity_based_initial_densities

Purpose: enforce lock-exchange initial density partition from salinity-derived target values.

Validation section: left/right halves match configured $\rho_{left}$ and $\rho_{right}$.

### test_initialize_rayleigh_taylor_layering_and_reset

Purpose: verify RT initializer layering and state reset, and switch-back lock-exchange initializer.

Reference setup: heavy-over-light with optional perturbation (Rayleigh-Taylor canonical).

Validation section: layer values and zero initial velocities/free-surface checks.

### test_initialize_rayleigh_taylor_argument_validation

Purpose: input-domain safety checks for RT initializer parameters.

Validation section: invalid parameter ranges must raise `ArgumentError`.

### test_free_surface_kinematic_bc_consistency

Purpose: enforce kinematic free-surface closure.

Reference condition:

```math
w_{top} = \frac{\partial \eta}{\partial t}.
```

Validation section: compares reconstructed top vertical velocity to finite-difference $\eta_t$ each step.

### test_hydrostatic_continuity_and_surface_closure_residuals

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

### test_long_run_conservation_and_froude_control

Purpose: long-run integration stability and conservation.

Validation section: volume/mass conservation and limiter compliance over extended steps.

### test_momentum_advection_zero_for_uniform_velocity_fields

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

### test_momentum_advection_conservative_in_closed_box

Purpose: ensure net momentum-advection tendency integrates to near zero in closed domain.

Validation section: integral of advection tendencies for $u$ and $w$ near zero for both schemes.

### test_model_runs_with_tvd2_momentum_advection

Purpose: smoke/stability check for TVD2 branch with full step coupling.

Validation section: finite diagnostic metrics after integration.

### test_baroclinic_pressure_gradient_accelerates_dense_to_right

Purpose: ensure pressure-gradient sign from baroclinic density contrast is physically consistent.

Reference relation:

```math
a_x = -\frac{1}{\rho_0}\frac{\partial p}{\partial x}.
```

Validation section: gradient sign and acceleration sign in expected direction.

### test_short_time_dense_front_moves_right_with_low_damping

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

### test_nonhydrostatic_branch_produces_pressure_correction

Purpose: ensure non-hydrostatic branch computes non-trivial pressure correction.

Reference equation: non-zero $p'$ expected when predictor divergence is non-zero.

Validation section: `nonhydro_pcorr_max > 0` and finite.

### test_nonhydrostatic_solution_differs_from_hydrostatic

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

### test_nonhydrostatic_surface_wave_period_matches_shallow_water_not_full_dispersion

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
positive counterpart to `test_nonhydrostatic_surface_wave_period_matches_shallow_water_not_full_dispersion` above (same domain,
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

Purpose: kernel-level check of `advect_density_upwind_top_layer!` with $(u,w,\eta^n,\eta^{n+1})$ taken from a model step.

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

## Additional validation scripts and literature basis

The repository includes standalone validation scripts under `experiments/validation`:

- `validate_surface_dam_break_front_speed.jl`: compares gravity-wave front speed with $c=\sqrt{gH}$ before reflection.
- `validate_internal_seiche_period.jl`: compares dominant period with two-layer long-wave estimate.
- `validate_lock_exchange_front_speed.jl`: checks fitted lock-exchange front-speed coefficient against literature-informed range.
- `validate_wind_shear_transport_mixing.jl`: checks stress-velocity scaling and density-variance reduction.

These scripts extend unit/regression testing with case-level physics checks tied to reference solutions.
