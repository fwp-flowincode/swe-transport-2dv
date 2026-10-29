Momentum advection with a mean flow (issue #37)
================================================

With a uniform current the explicit momentum advection (`upwind1`, `tvd2`) became unstable far below the advective
Courant number $C=U\Delta t/\Delta x=1$, also with $\theta=1$. Two inconsistencies of the discrete momentum fluxes were
the cause; both are fixed in `src/physics.jl`.

Causes
------

1. **No momentum flux through the top of the fixed grid.** On the fixed grid (`variable_top_layer=false`: every periodic
   and every non-hydrostatic run) the grid top carries the vertical velocity $w_s=\partial_t\eta$, but the momentum flux
   through it was zero. Linearised around $U$, the horizontal flux of $u^2$ carries $2Uu'$; in the layers below, the
   vertical flux $Uw'$ removes the extra $Uu'_x$ (continuity), in the top row the missing $Uw_s$ did not. The top row
   was advected at $2U$ by itself and, for a depth-uniform surface-wave perturbation, at about $(n_z+1)U$. Fix: the
   advective form in the top row, $\partial_x(uu)+\partial_z(wu)-u\,\nabla\cdot v$ (`_top_row_advective_form!`), now also on
   the fixed grid, as for the variable top layer and the density transport.
2. **Smoothed vertical velocity at the u control volumes.** $w$ was interpolated to the layer centres, to the u-points
   and back to the interfaces, $(w_{k-1}+2w_k+w_{k+1})/4$. Then the u control volume is not divergence-free: a
   zero-transport mode that alternates over the layers is advected at about $2U$ with a central part, which forward Euler
   amplifies at any time step (`tvd2` removes the upwind damping that hid this for `upwind1`). Fix: the plain horizontal
   average $(w_{k,i-1}+w_{k,i})/2$ at the interface (`_w_at_u_interfaces!`, Arakawa C-grid), which makes
   $\nabla\cdot v=0$ exact for the u control volume. The same holds for the w momentum (non-hydrostatic):
   $(u_{k-1,i}+u_{k,i})/2$ at the u-points (`_u_at_w_faces!`) instead of a twice horizontally smoothed $u$. The implicit
   vertical terms already used the horizontal average.

Linear analysis: `operator_spectrum.jl`
---------------------------------------

Spectral radius of forward Euler, $\max\lvert\mathrm{eig}(I-\Delta t L)\rvert$, of the linearised `upwind1` advection operator
$L$ around $U$ (periodic, fixed grid, $w$ from continuity, no gravity waves), built with the model's own kernels:

| variant | nz | C = 0.25 | C = 0.5 | C = 0.75 | C = 1 |
|---|---|---|---|---|---|
| legacy (before) | 1 ... 16 | 1 | 1.154 | 1.521 | 2.000 |
| top-row form only | 2 / 4 / 16 | 1 | 1 / 1.086 / 1.150 | 1.024 / 1.382 / 1.513 | 1.25 / 1.81 / 1.99 |
| consistent (now) | 1 ... 16 | 1 | 1 | 1 | 1 |

With both fixes the 2DV operator is stable up to $C=1$ for every nz, like 1D upwind. With only one of them it is not:
the legacy top row runs at $2U$ ($\lvert G\rvert=2$ at $C=1$), and the smoothed $w$ makes the high vertical modes grow
more with more layers.

Full model: `channel_scan.jl`
-----------------------------

Periodic channel, $L=1$ m, $H=0.5$ m, nx = 16, surface perturbation $10^{-8}$ m, final amplitude over the initial one.
Before the fix ($\theta=1$, hydrostatic, 200 s):

| scheme, Fr, nz | C = 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|
| upwind1, 0.2, 8 | decays | decays | 7.4 | blow-up | blow-up |
| tvd2, 0.2, 2 | decays | 1.1e4 | 1.0e5 | 1.4e5 | 4.8e5 |
| tvd2, 0.2, 8 | 2.2e4 | 1.5e5 | blow-up | blow-up | blow-up |
| tvd2, 0.5, 2 | 1.8e4 | 2.5e5 | 5.8e5 | 6.9e5 | blow-up |
| tvd2, 0.5, 8 | 1.8e5 | blow-up | blow-up | blow-up | blow-up |

After the fix every case decays for $C=0.1\ldots0.9$, nz = 2, 8, 16, both schemes, Fr = 0.2 and 0.5, hydrostatic and
non-hydrostatic, at $\theta=1$ and $\theta=0.7$ (within the Froude bound of the theta scheme, `theta_froude_max`,
[../theta_scheme/README.md](../theta_scheme/README.md)).

Run: `julia --project experiments/advection_stability/operator_spectrum.jl` (about a minute) and
`julia --project experiments/advection_stability/channel_scan.jl [theta]` (a few minutes).

References: Arakawa & Lamb (1977), Methods Comput. Phys. 17, 173-265 (C-grid momentum fluxes); Casulli & Walters (2000),
Int. J. Numer. Methods Fluids 32, 331-348 (z-layer free surface). Equation numbers were not checked.
