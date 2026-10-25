@inline function _minmod_scalar(a::Real, b::Real)
    if a * b > 0.0
        return abs(a) < abs(b) ? a : b
    end
    return 0.0
end

@inline function _monotonized_central_scalar(a::Real, b::Real)
    return _minmod_scalar(_minmod_scalar(2.0 * a, 2.0 * b), 0.5 * (a + b))
end

mutable struct AdvectionWorkspace{T<:Real}
    vx_face::Matrix{T}
    vz_face::Matrix{T}
    flux_x::Matrix{T}
    flux_z::Matrix{T}
    slope_x::Matrix{T}
    slope_z::Matrix{T}
end

function AdvectionWorkspace(nz::Int, nx::Int; T::Type{<:Real}=Float64)
    return AdvectionWorkspace(
        zeros(T, nz, nx + 1),
        zeros(T, nz + 1, nx),
        zeros(T, nz, nx + 1),
        zeros(T, nz + 1, nx),
        zeros(T, nz, nx),
        zeros(T, nz, nx),
    )
end

mutable struct InterpolationWorkspace{T<:Real}
    w_center::Matrix{T}
    w_u::Matrix{T}
    u_center::Matrix{T}
    u_w::Matrix{T}
end

function InterpolationWorkspace(nz::Int, nx::Int; T::Type{<:Real}=Float64)
    return InterpolationWorkspace(
        zeros(T, nz, nx),
        zeros(T, nz, nx + 1),
        zeros(T, nz, nx),
        zeros(T, nz + 1, nx),
    )
end

mutable struct DensityWorkspace{T<:Real}
    flux_x::Matrix{T}
    flux_z::Matrix{T}
    adv::Matrix{T}
    slope_x::Matrix{T}   # limited cell slopes (tvd2 only), same unit as the transported field
    slope_z::Matrix{T}
end

function DensityWorkspace(nz::Int, nx::Int; T::Type{<:Real}=Float64)
    return DensityWorkspace(
        zeros(T, nz, nx + 1),
        zeros(T, nz + 1, nx),
        zeros(T, nz, nx),
        zeros(T, nz, nx),
        zeros(T, nz, nx),
    )
end

function grad_x_c!(res::AbstractMatrix{<:Real}, f::AbstractMatrix{<:Real}, dx::Real; periodic_x::Bool=false)
    nz, nx = size(f)
    fill!(res, 0.0)
    if periodic_x
        @inbounds for k in 1:nz
            seam_gradient = (f[k, 1] - f[k, nx]) / dx
            res[k, 1] = seam_gradient
            res[k, nx + 1] = seam_gradient
        end
    end
    @inbounds for i in 2:nx
        for k in 1:nz
            res[k, i] = (f[k, i] - f[k, i - 1]) / dx
        end
    end
    return res
end

function grad_x_c(f::AbstractMatrix{<:Real}, dx::Real; periodic_x::Bool=false)
    res = zeros(eltype(f), size(f, 1), size(f, 2) + 1)
    return grad_x_c!(res, f, dx; periodic_x=periodic_x)
end

function grad_z_c!(res::AbstractMatrix{<:Real}, f::AbstractMatrix{<:Real}, dz::Real)
    nz, nx = size(f)
    fill!(res, 0.0)
    @inbounds for i in 1:nx
        for k in 2:nz
            res[k, i] = (f[k, i] - f[k - 1, i]) / dz
        end
    end
    return res
end

function grad_z_c(f::AbstractMatrix{<:Real}, dz::Real)
    res = zeros(eltype(f), size(f, 1) + 1, size(f, 2))
    return grad_z_c!(res, f, dz)
end

function div_x_u!(res::AbstractMatrix{<:Real}, u::AbstractMatrix{<:Real}, dx::Real)
    nz, nxp1 = size(u)
    nx = nxp1 - 1
    @inbounds for i in 1:nx
        for k in 1:nz
            res[k, i] = (u[k, i + 1] - u[k, i]) / dx
        end
    end
    return res
end

function div_x_u(u::AbstractMatrix{<:Real}, dx::Real)
    res = zeros(eltype(u), size(u, 1), size(u, 2) - 1)
    return div_x_u!(res, u, dx)
end

function div_z_w!(res::AbstractMatrix{<:Real}, w::AbstractMatrix{<:Real}, dz::Real)
    nzp1, nx = size(w)
    nz = nzp1 - 1
    @inbounds for i in 1:nx
        for k in 1:nz
            res[k, i] = (w[k + 1, i] - w[k, i]) / dz
        end
    end
    return res
end

function div_z_w(w::AbstractMatrix{<:Real}, dz::Real)
    res = zeros(eltype(w), size(w, 1) - 1, size(w, 2))
    return div_z_w!(res, w, dz)
end

function _interpolate_w_to_u!(w_u::AbstractMatrix{<:Real}, w::AbstractMatrix{<:Real}, w_center::AbstractMatrix{<:Real}; periodic_x::Bool=false)
    nz, nx = size(w_center)

    @inbounds for i in 1:nx
        for k in 1:nz
            w_center[k, i] = 0.5 * (w[k, i] + w[k + 1, i])
        end
    end

    @inbounds for k in 1:nz
        w_u[k, 1] = periodic_x ? 0.5 * (w_center[k, nx] + w_center[k, 1]) : w_center[k, 1]
        for i in 2:nx
            w_u[k, i] = 0.5 * (w_center[k, i - 1] + w_center[k, i])
        end
        w_u[k, nx + 1] = periodic_x ? w_u[k, 1] : w_center[k, nx]
    end

    return w_u
end

function _interpolate_u_to_w!(u_w::AbstractMatrix{<:Real}, u::AbstractMatrix{<:Real}, u_center::AbstractMatrix{<:Real})
    nz, nx = size(u_center)

    @inbounds for i in 1:nx
        for k in 1:nz
            u_center[k, i] = 0.5 * (u[k, i] + u[k, i + 1])
        end
    end

    @inbounds for i in 1:nx
        u_w[1, i] = u_center[1, i]
        for k in 2:nz
            u_w[k, i] = 0.5 * (u_center[k - 1, i] + u_center[k, i])
        end
        u_w[nz + 1, i] = u_center[nz, i]
    end

    return u_w
end

function _face_velocity_x!(vx_face::AbstractMatrix{<:Real}, vx::AbstractMatrix{<:Real}; periodic_x::Bool=false)
    nz, nx = size(vx)
    @inbounds for k in 1:nz
        vx_face[k, 1] = periodic_x ? 0.5 * (vx[k, nx] + vx[k, 1]) : vx[k, 1]
        for i in 2:nx
            vx_face[k, i] = 0.5 * (vx[k, i - 1] + vx[k, i])
        end
        vx_face[k, nx + 1] = periodic_x ? vx_face[k, 1] : vx[k, nx]
    end
    return vx_face
end

function _face_velocity_z!(vz_face::AbstractMatrix{<:Real}, vz::AbstractMatrix{<:Real})
    nz, nx = size(vz)
    @inbounds for i in 1:nx
        vz_face[1, i] = vz[1, i]
        for k in 2:nz
            vz_face[k, i] = 0.5 * (vz[k - 1, i] + vz[k, i])
        end
        vz_face[nz + 1, i] = vz[nz, i]
    end
    return vz_face
end

"""
    _advective_flux_divergence!(out, phi, vx, vz, dx, dz, scheme, ws, periodic_x=false; active=nothing, dt=0.0)

Conservative advective tendency `out = div(v*phi)` of a momentum component `phi` (m/s; units of phi times 1/s) on its own
grid points, with the point velocities `vx`, `vz` (m/s) averaged to the faces. `scheme`: `"upwind1"` (donor cell) or
`"tvd2"`: MC-limited piecewise-linear reconstruction with the Lax-Wendroff time correction, face value
`phi_up + 0.5 (1 - c) s_up` with the face Courant number c = |v| dt/dx (resp. dz) -- the same scheme as the density
transport (`_density_face_fluxes!`). `dt = 0` (s) gives the semi-discrete MUSCL flux. `active`: points next to solid
walls (variable bed), see `BedGeometry`.

WHY the time correction (model passes dt = cfg.dt): the momentum equations are advanced with one forward-Euler step,
which adds kinetic energy of order (dt^2/2)|A|^2 per step; without the correction the MUSCL/forward-Euler update is
first order in time and only weakly dissipative. In the lock exchange with obstacle (issue #3) the uncorrected scheme
created kinetic energy after the gravity current hit the side wall (kinetic energy up to 1.5 times the released
potential energy) and blew up; with the correction the energy budget stays consistent much longer.
Sources: Sweby (1984), SIAM J. Numer. Anal. 21, 995-1011; LeVeque (2002), Finite Volume Methods for Hyperbolic
Problems, ch. 6 (equation numbers not checked).
"""
function _advective_flux_divergence!(
    out::AbstractMatrix{<:Real},
    phi::AbstractMatrix{<:Real},
    vx::AbstractMatrix{<:Real},
    vz::AbstractMatrix{<:Real},
    dx::Real,
    dz::Real,
    scheme::AbstractString,
    ws::AdvectionWorkspace,
    periodic_x::Bool=false;
    active::Union{Nothing,AbstractMatrix{Bool}}=nothing,
    dt::Real=0.0,
)
    nz, nx = size(phi)

    if scheme != "upwind1" && scheme != "tvd2"
        throw(ArgumentError("Unsupported advection scheme '$(scheme)'"))
    end

    _face_velocity_x!(ws.vx_face, vx; periodic_x=periodic_x)
    _face_velocity_z!(ws.vz_face, vz)

    if scheme == "upwind1"
        @inbounds for k in 1:nz
            if periodic_x
                v = ws.vx_face[k, 1]
                ws.flux_x[k, 1] = v >= 0.0 ? v * phi[k, nx] : v * phi[k, 1]
                ws.flux_x[k, nx + 1] = ws.flux_x[k, 1]
            else
                ws.flux_x[k, 1] = 0.0
                ws.flux_x[k, nx + 1] = 0.0
            end
            for i in 2:nx
                v = ws.vx_face[k, i]
                if v >= 0.0
                    ws.flux_x[k, i] = v * phi[k, i - 1]
                else
                    ws.flux_x[k, i] = v * phi[k, i]
                end
                if active !== nothing && !(active[k, i - 1] && active[k, i])
                    ws.flux_x[k, i] = 0.0   # solid wall between the two points
                end
            end
        end

        @inbounds for i in 1:nx
            ws.flux_z[1, i] = 0.0
            for k in 2:nz
                v = ws.vz_face[k, i]
                if v >= 0.0
                    ws.flux_z[k, i] = v * phi[k - 1, i]
                else
                    ws.flux_z[k, i] = v * phi[k, i]
                end
                if active !== nothing && !(active[k - 1, i] && active[k, i])
                    ws.flux_z[k, i] = 0.0
                end
            end
            ws.flux_z[nz + 1, i] = 0.0
        end
    else
        fill!(ws.slope_x, 0.0)
        fill!(ws.slope_z, 0.0)

        if periodic_x && nx >= 3
            @inbounds for i in 1:nx
                for k in 1:nz
                    ws.slope_x[k, i] = _monotonized_central_scalar(
                        phi[k, i] - phi[k, mod1(i - 1, nx)],
                        phi[k, mod1(i + 1, nx)] - phi[k, i],
                    )
                end
            end
        elseif nx >= 3
            @inbounds for i in 2:(nx - 1)
                for k in 1:nz
                    a = phi[k, i] - phi[k, i - 1]
                    b = phi[k, i + 1] - phi[k, i]
                    ws.slope_x[k, i] = _monotonized_central_scalar(a, b)
                    # WHY: next to a solid point the slope is zero, as in the outermost points of the domain
                    if active !== nothing && !(active[k, i - 1] && active[k, i] && active[k, i + 1])
                        ws.slope_x[k, i] = 0.0
                    end
                end
            end
        end

        if nz >= 3
            @inbounds for i in 1:nx
                for k in 2:(nz - 1)
                    a = phi[k, i] - phi[k - 1, i]
                    b = phi[k + 1, i] - phi[k, i]
                    ws.slope_z[k, i] = _monotonized_central_scalar(a, b)
                    if active !== nothing && !(active[k - 1, i] && active[k, i] && active[k + 1, i])
                        ws.slope_z[k, i] = 0.0
                    end
                end
            end
        end

        @inbounds for k in 1:nz
            if periodic_x
                v = ws.vx_face[k, 1]
                time_factor = 0.5 * (1.0 - abs(v) * dt / dx)   # Lax-Wendroff correction; 0.5 for dt = 0
                left_x = phi[k, nx] + time_factor * ws.slope_x[k, nx]
                right_x = phi[k, 1] - time_factor * ws.slope_x[k, 1]
                ws.flux_x[k, 1] = v >= 0.0 ? v * left_x : v * right_x
                ws.flux_x[k, nx + 1] = ws.flux_x[k, 1]
            else
                ws.flux_x[k, 1] = 0.0
                ws.flux_x[k, nx + 1] = 0.0
            end
            for i in 2:nx
                v = ws.vx_face[k, i]
                time_factor = 0.5 * (1.0 - abs(v) * dt / dx)
                left_x = phi[k, i - 1] + time_factor * ws.slope_x[k, i - 1]
                right_x = phi[k, i] - time_factor * ws.slope_x[k, i]
                ws.flux_x[k, i] = v >= 0.0 ? v * left_x : v * right_x
                if active !== nothing && !(active[k, i - 1] && active[k, i])
                    ws.flux_x[k, i] = 0.0
                end
            end
        end

        @inbounds for i in 1:nx
            ws.flux_z[1, i] = 0.0
            for k in 2:nz
                v = ws.vz_face[k, i]
                time_factor = 0.5 * (1.0 - abs(v) * dt / dz)
                lower_z = phi[k - 1, i] + time_factor * ws.slope_z[k - 1, i]
                upper_z = phi[k, i] - time_factor * ws.slope_z[k, i]
                ws.flux_z[k, i] = v >= 0.0 ? v * lower_z : v * upper_z
                if active !== nothing && !(active[k - 1, i] && active[k, i])
                    ws.flux_z[k, i] = 0.0
                end
            end
            ws.flux_z[nz + 1, i] = 0.0
        end
    end

    @inbounds for i in 1:nx
        for k in 1:nz
            out[k, i] = (ws.flux_x[k, i + 1] - ws.flux_x[k, i]) / dx + (ws.flux_z[k + 1, i] - ws.flux_z[k, i]) / dz
        end
    end

    return out
end

# Top-row correction to the advective-form momentum tendency for a variable-thickness top layer.
#
# WHY: the flux-form tendency div(v*phi) equals v.grad(phi) only if div(v) = 0 for the same discrete fluxes. With a
# moving free surface the top-layer control volume has div(v) = -(d eta/dt)/h (its volume changes), and the top face
# carries no flux (material surface). Subtracting phi*div(v), built from the very same face velocities and the same
# zero-flux conventions as _advective_flux_divergence!, restores the advective form in the top row, so a uniform phi
# has zero tendency. It is applied to the top row only: below it div(v) vanishes up to interpolation error and the
# validated legacy scheme is left untouched.
# Source: v.grad(phi) = div(v*phi) - phi*div(v); z-layer free-surface treatment as in Casulli & Walters (2000),
# Int. J. Numer. Methods Fluids 32, 331-348 (equation numbers not checked).
function _top_row_advective_form!(
    out::AbstractMatrix{<:Real},
    phi::AbstractMatrix{<:Real},
    dx::Real,
    dz::Real,
    ws::AdvectionWorkspace;
    periodic_x::Bool=false,
)
    nz, nx = size(phi)
    if nz < 2
        return out
    end

    @inbounds for i in 1:nx
        volume_flux_in = i >= 2 ? ws.vx_face[nz, i] : (periodic_x ? ws.vx_face[nz, 1] : 0.0)
        volume_flux_out = i + 1 <= nx ? ws.vx_face[nz, i + 1] : (periodic_x ? ws.vx_face[nz, nx + 1] : 0.0)
        div_v = (volume_flux_out - volume_flux_in) / dx - ws.vz_face[nz, i] / dz   # zero volume flux through the top face
        out[nz, i] -= phi[nz, i] * div_v
    end

    return out
end

function advection_term_u!(
    out::AbstractMatrix{<:Real},
    u::AbstractMatrix{<:Real},
    w::AbstractMatrix{<:Real},
    dx::Real,
    dz::Real,
    scheme::AbstractString,
    ws_adv::AdvectionWorkspace,
    ws_interp::InterpolationWorkspace;
    top_row_advective_form::Bool=false,
    periodic_x::Bool=false,
    active::Union{Nothing,AbstractMatrix{Bool}}=nothing,
    dt::Real=0.0,
)
    _interpolate_w_to_u!(ws_interp.w_u, w, ws_interp.w_center; periodic_x=periodic_x)
    if periodic_x
        nx = size(u, 2) - 1
        _advective_flux_divergence!(@view(out[:, 1:nx]), @view(u[:, 1:nx]), @view(u[:, 1:nx]), @view(ws_interp.w_u[:, 1:nx]), dx, dz, scheme, ws_adv, true; dt=dt)
        @views out[:, nx + 1] .= out[:, 1]
    else
        _advective_flux_divergence!(out, u, u, ws_interp.w_u, dx, dz, scheme, ws_adv; active=active, dt=dt)
    end
    if top_row_advective_form
        if periodic_x
            _top_row_advective_form!(@view(out[:, 1:(size(u, 2) - 1)]), @view(u[:, 1:(size(u, 2) - 1)]), dx, dz, ws_adv; periodic_x=true)
        else
            _top_row_advective_form!(out, u, dx, dz, ws_adv)
        end
    end
    return out
end

function advection_term_u(
    u::AbstractMatrix{<:Real},
    w::AbstractMatrix{<:Real},
    dx::Real,
    dz::Real;
    scheme::AbstractString="upwind1",
    periodic_x::Bool=false,
)
    out = zeros(eltype(u), size(u))
    nz, nxp1 = size(u)
    nx = nxp1 - 1
    ws_adv = AdvectionWorkspace(nz, nxp1; T=eltype(u))
    ws_interp = InterpolationWorkspace(nz, nx; T=eltype(u))
    advection_term_u!(out, u, w, dx, dz, scheme, ws_adv, ws_interp; periodic_x=periodic_x)
    return out
end

function advection_term_w!(
    out::AbstractMatrix{<:Real},
    u::AbstractMatrix{<:Real},
    w::AbstractMatrix{<:Real},
    dx::Real,
    dz::Real,
    scheme::AbstractString,
    ws_adv::AdvectionWorkspace,
    ws_interp::InterpolationWorkspace,
    periodic_x::Bool=false;
    active::Union{Nothing,AbstractMatrix{Bool}}=nothing,
    dt::Real=0.0,
)
    _interpolate_u_to_w!(ws_interp.u_w, u, ws_interp.u_center)
    _advective_flux_divergence!(out, w, ws_interp.u_w, w, dx, dz, scheme, ws_adv, periodic_x; active=active, dt=dt)
    return out
end

function advection_term_w(
    u::AbstractMatrix{<:Real},
    w::AbstractMatrix{<:Real},
    dx::Real,
    dz::Real;
    scheme::AbstractString="upwind1",
    periodic_x::Bool=false,
)
    out = zeros(eltype(w), size(w))
    nzp1, nx = size(w)
    nz = nzp1 - 1
    ws_adv = AdvectionWorkspace(nzp1, nx; T=eltype(w))
    ws_interp = InterpolationWorkspace(nz, nx; T=eltype(u))
    advection_term_w!(out, u, w, dx, dz, scheme, ws_adv, ws_interp, periodic_x)
    return out
end

_nu_at(nu::Real, k::Int, i::Int) = nu
_nu_at(nu::AbstractMatrix{<:Real}, k::Int, i::Int) = nu[k, i]

"""
    diffusion_term!(res, phi, nu, dx, dz; periodic_x=false, active=nothing)

Viscous term `d/dx(nu dphi/dx) + d/dz(nu dphi/dz)` (units of `phi` per s) of the face velocity `phi` (m/s, nz x nx) with
the viscosity `nu` (m^2/s, >= 0; a number, or an nz x nx matrix at the points of `phi`), in flux form with the face
viscosity `nu_f = 0.5*(nu_a + nu_b)`:

    res[k,i] = (F[k+1/2] - F[k-1/2]) / dz + (G[i+1/2] - G[i-1/2]) / dx,
    F[k+1/2] = nu_f (phi[k+1] - phi[k]) / dz,   G[i+1/2] = nu_f (phi[i+1] - phi[i]) / dx.

Boundaries: zero flux through the bed and the free surface (`F[1/2] = F[nz+1/2] = 0`: free slip at the bed, no stress at
the top; a wind stress enters the top row as a body force in `step_forward!`, which is the same stress condition). The
outermost points in x (the wall faces) get `res = 0`; their value (`phi = 0` at a wall) enters the neighbouring flux.
With `active` (nz x nx; for u: a face next to at least one wet cell), inactive points get `res = 0` and the vertical flux
to an inactive neighbour (the face below the bed) is zero, i.e. the same free-slip bed as at k = 1; obstacle walls
(active, `u = 0`) act like the outer walls.

Second order in space for smooth `nu`; sum(phi) dz per column is conserved by the vertical part. Explicit
forward-Euler stability: `2 nu dt (1/dx^2 + 1/dz^2) <= 1`. For constant `nu` this is the 5-point Laplacian in the
interior.
WHY flux form over the top and bottom rows: the former kernel `nu * Laplacian` skipped rows 1 and nz, so the top row had
no viscous coupling to the row below, and a wind stress stayed in the top cell (issue #4).
"""
function diffusion_term!(res::AbstractMatrix{<:Real}, phi::AbstractMatrix{<:Real}, nu::Union{Real,AbstractMatrix{<:Real}}, dx::Real, dz::Real; periodic_x::Bool=false, active::Union{Nothing,AbstractMatrix{Bool}}=nothing)
    nz, nx = size(phi)
    fill!(res, 0.0)
    nphysical = periodic_x ? nx - 1 : nx
    ilo, ihi = periodic_x ? (1, nphysical) : (2, nx - 1)
    @inbounds for i in ilo:ihi
        il = periodic_x ? mod1(i - 1, nphysical) : i - 1
        ir = periodic_x ? mod1(i + 1, nphysical) : i + 1
        for k in 1:nz
            if active !== nothing && !active[k, i]
                continue   # res stays 0
            end
            nu_c = _nu_at(nu, k, i)
            g_right = 0.5 * (nu_c + _nu_at(nu, k, ir)) * (phi[k, ir] - phi[k, i]) / dx
            g_left = 0.5 * (nu_c + _nu_at(nu, k, il)) * (phi[k, i] - phi[k, il]) / dx
            has_top = k < nz && (active === nothing || active[k + 1, i])
            has_bottom = k > 1 && (active === nothing || active[k - 1, i])
            f_top = has_top ? 0.5 * (nu_c + _nu_at(nu, k + 1, i)) * (phi[k + 1, i] - phi[k, i]) / dz : 0.0
            f_bottom = has_bottom ? 0.5 * (nu_c + _nu_at(nu, k - 1, i)) * (phi[k, i] - phi[k - 1, i]) / dz : 0.0
            res[k, i] = (g_right - g_left) / dx + (f_top - f_bottom) / dz
        end
    end
    if periodic_x
        @views res[:, end] .= res[:, 1]
    end
    return res
end

function diffusion_term(phi::AbstractMatrix{<:Real}, nu::Real, dx::Real, dz::Real)
    nz, nx = size(phi)
    res = zeros(eltype(phi), nz, nx)
    diffusion_term!(res, phi, nu, dx, dz)
    return res
end

"""
    _density_face_fluxes!(ws, rho, u, w, dt, dx, dz, scheme; h_top_face=nothing)

Fill `ws.flux_x` (nz x nx+1) and `ws.flux_z` (nz+1 x nx) with the advective face fluxes `v_f * rho_f` of the cell-centred
field `rho` (kg/m^3; fluxes in kg/m^2/s) on the C-grid velocities `u` (m/s, x-faces) and `w` (m/s, z-faces). All outer
faces carry zero flux. With `h_top_face` (m, length nx+1) the x-fluxes of the top row are multiplied by the face thickness
(thickness-integrated top-layer budget, see `advect_density_top_layer!`).

Schemes for the face value `rho_f`:

- `"upwind1"`: donor cell, `rho_f = rho_upwind` (first order, monotone for `dt*(|u|/dx + |w|/dz) <= 1`).
- `"tvd2"`: limited piecewise-linear reconstruction with the same MC slope as the momentum `tvd2` scheme,
  plus the Lax-Wendroff time correction (Courant number `c = |v_f|*dt/dx` resp. `|v_f|*dt/dz`):

      rho_f = rho_up + 0.5 * (1 - c) * sign_f * s_up,     s = MC(Delta^- rho, Delta^+ rho),   s = 0 in boundary cells,

  where `sign_f = +1` if the upwind cell lies on the low-index side of the face, else `-1`. Second order in space and
  time where the solution is smooth, first order at extrema. In 1D the scheme is TVD for `c <= 1` (MC lies in Sweby's
  TVD region `0 <= psi(r) <= min(2r, 2)`); for the unsplit 2D update a local maximum principle holds for
  `dt*(|u|/dx + |w|/dz) <= 1/2` per cell (each face coefficient is at most `c*(2 - c) <= 2c`).
  `dt = 0` gives the semi-discrete MUSCL flux of the momentum `tvd2` scheme.
  The update is unsplit and has no corner-transport (cross-derivative) term, so for flow oblique to the grid the time
  error is O(dt * |u*w| * d2rho/dxdz): first order in time, second order in space. Measured L1 orders (Gaussian, CFL 0.4):
  about 2.2 for grid-aligned flow, 1.0 for diagonal flow (error still ~4x below upwind1); 1.8 for diagonal flow at CFL 0.02.

WHY the (1 - c) factor, which the momentum `tvd2` scheme does not have: density is advanced with a single forward-Euler
step. Without the time correction the MUSCL/forward-Euler update is first order in time and TVD only for `c <= 1/2` in
1D; with it the update is the flux-limited Lax-Wendroff scheme, second order in time and TVD up to `c = 1`.
Sources: van Leer (1977), J. Comput. Phys. 23, 276-299 (MC limiter); Sweby (1984), SIAM J. Numer. Anal. 21, 995-1011
(flux-limited Lax-Wendroff, TVD region); LeVeque (2002), Finite Volume Methods for Hyperbolic Problems, ch. 6
(slope-limiter form of the same scheme; equation numbers not checked).
"""
function _density_face_fluxes!(
    ws::DensityWorkspace,
    rho::AbstractMatrix{<:Real},
    u::AbstractMatrix{<:Real},
    w::AbstractMatrix{<:Real},
    dt::Real,
    dx::Real,
    dz::Real,
    scheme::AbstractString;
    h_top_face::Union{Nothing,AbstractVector{<:Real}}=nothing,
    periodic_x::Bool=false,
    active::Union{Nothing,AbstractMatrix{Bool}}=nothing,
)
    nz, nx = size(rho)

    if scheme == "upwind1"
        @inbounds for k in 1:nz
            scale_row = h_top_face !== nothing && k == nz
            if periodic_x
                v = u[k, 1]
                scale = scale_row ? h_top_face[1] : 1.0
                ws.flux_x[k, 1] = scale * (v > 0.0 ? v * rho[k, nx] : (v < 0.0 ? v * rho[k, 1] : 0.0))
                ws.flux_x[k, nx + 1] = ws.flux_x[k, 1]
            else
                ws.flux_x[k, 1] = 0.0
                ws.flux_x[k, nx + 1] = 0.0
            end
            for i in 2:nx
                v = scale_row ? h_top_face[i] * u[k, i] : u[k, i]
                ws.flux_x[k, i] = v > 0.0 ? v * rho[k, i - 1] : (v < 0.0 ? v * rho[k, i] : 0.0)
            end
        end

        @inbounds for i in 1:nx
            ws.flux_z[1, i] = 0.0
            for k in 2:nz
                v = w[k, i]
                ws.flux_z[k, i] = v > 0.0 ? v * rho[k - 1, i] : (v < 0.0 ? v * rho[k, i] : 0.0)
            end
            ws.flux_z[nz + 1, i] = 0.0
        end
    elseif scheme == "tvd2"
        fill!(ws.slope_x, 0.0)
        fill!(ws.slope_z, 0.0)

        if periodic_x && nx >= 3
            @inbounds for i in 1:nx
                for k in 1:nz
                    ws.slope_x[k, i] = _monotonized_central_scalar(
                        rho[k, i] - rho[k, mod1(i - 1, nx)],
                        rho[k, mod1(i + 1, nx)] - rho[k, i],
                    )
                end
            end
        elseif nx >= 3
            @inbounds for i in 2:(nx - 1)
                for k in 1:nz
                    ws.slope_x[k, i] = _monotonized_central_scalar(rho[k, i] - rho[k, i - 1], rho[k, i + 1] - rho[k, i])
                    # WHY: next to a solid cell the slope is zero, as in the outermost cells of the domain
                    if active !== nothing && !(active[k, i - 1] && active[k, i] && active[k, i + 1])
                        ws.slope_x[k, i] = 0.0
                    end
                end
            end
        end

        if nz >= 3
            @inbounds for i in 1:nx
                for k in 2:(nz - 1)
                    ws.slope_z[k, i] = _monotonized_central_scalar(rho[k, i] - rho[k - 1, i], rho[k + 1, i] - rho[k, i])
                    if active !== nothing && !(active[k - 1, i] && active[k, i] && active[k + 1, i])
                        ws.slope_z[k, i] = 0.0
                    end
                end
            end
        end

        @inbounds for k in 1:nz
            scale_row = h_top_face !== nothing && k == nz
            if periodic_x
                v = u[k, 1]
                time_factor = 0.5 * (1.0 - abs(v) * dt / dx)
                rho_face = v >= 0.0 ? rho[k, nx] + time_factor * ws.slope_x[k, nx] : rho[k, 1] - time_factor * ws.slope_x[k, 1]
                scale = scale_row ? h_top_face[1] : 1.0
                ws.flux_x[k, 1] = scale * v * rho_face
                ws.flux_x[k, nx + 1] = ws.flux_x[k, 1]
            else
                ws.flux_x[k, 1] = 0.0
                ws.flux_x[k, nx + 1] = 0.0
            end
            for i in 2:nx
                v = u[k, i]
                time_factor = 0.5 * (1.0 - abs(v) * dt / dx)   # 0.5 * (1 - Courant number)
                rho_face = v >= 0.0 ? rho[k, i - 1] + time_factor * ws.slope_x[k, i - 1] : rho[k, i] - time_factor * ws.slope_x[k, i]
                ws.flux_x[k, i] = (scale_row ? h_top_face[i] * v : v) * rho_face
            end
        end

        @inbounds for i in 1:nx
            ws.flux_z[1, i] = 0.0
            for k in 2:nz
                v = w[k, i]
                time_factor = 0.5 * (1.0 - abs(v) * dt / dz)
                rho_face = v >= 0.0 ? rho[k - 1, i] + time_factor * ws.slope_z[k - 1, i] : rho[k, i] - time_factor * ws.slope_z[k, i]
                ws.flux_z[k, i] = v * rho_face
            end
            ws.flux_z[nz + 1, i] = 0.0
        end
    else
        throw(ArgumentError("Unsupported transport advection scheme '$(scheme)'"))
    end

    return ws
end

"""
    advection_term_rho!(adv, u, w, rho, dx, dz, ws; scheme="upwind1", dt=0.0)

Conservative advective tendency `adv = div(v*rho)` (kg/m^3/s) of `rho` (kg/m^3) on the C-grid velocities `u`, `w` (m/s),
with closed outer faces. `scheme` is `"upwind1"` or `"tvd2"` (see `_density_face_fluxes!`); `dt` (s) only enters the
Lax-Wendroff correction of `"tvd2"`, `dt = 0` gives its semi-discrete form.
"""
function advection_term_rho!(
    adv::AbstractMatrix{<:Real},
    u::AbstractMatrix{<:Real},
    w::AbstractMatrix{<:Real},
    rho::AbstractMatrix{<:Real},
    dx::Real,
    dz::Real,
    ws::DensityWorkspace;
    scheme::AbstractString="upwind1",
    dt::Real=0.0,
    periodic_x::Bool=false,
    active::Union{Nothing,AbstractMatrix{Bool}}=nothing,
)
    nz, nx = size(rho)
    _density_face_fluxes!(ws, rho, u, w, dt, dx, dz, scheme; periodic_x=periodic_x, active=active)

    @inbounds for i in 1:nx
        for k in 1:nz
            adv[k, i] = (ws.flux_x[k, i + 1] - ws.flux_x[k, i]) / dx + (ws.flux_z[k + 1, i] - ws.flux_z[k, i]) / dz
        end
    end

    return adv
end

function advection_term_rho(
    u::AbstractMatrix{<:Real},
    w::AbstractMatrix{<:Real},
    rho::AbstractMatrix{<:Real},
    dx::Real,
    dz::Real;
    scheme::AbstractString="upwind1",
    dt::Real=0.0,
    periodic_x::Bool=false,
)
    nz, nx = size(rho)
    ws = DensityWorkspace(nz, nx; T=eltype(rho))
    adv = zeros(eltype(rho), nz, nx)
    advection_term_rho!(adv, u, w, rho, dx, dz, ws; scheme=scheme, dt=dt, periodic_x=periodic_x)
    return adv
end

"""
    _density_top_row_advective_form!(adv, rho, u, w, dx, dz)

Fixed-grid free-surface correction of the density tendency `adv` (kg/m^3/s) in the top row: subtracts `rho * div(v)`,
built from the same face velocities and the same zero-flux conventions as the face fluxes (closed outer x-faces, no flux
through the top face), so that the top row is advanced in advective form

    adv_top = div(v*rho) - rho*div(v) = v . grad(rho).

`rho` in kg/m^3, `u`, `w` in m/s, `dx`, `dz` in m. Rows below the top row are unchanged (there div(v) vanishes to
round-off because `w` is reconstructed from continuity).

WHY: on the fixed grid the top face carries no density flux, although the discrete velocity there is d(eta)/dt, so the
top-row control volume has div(v) = -(d eta/dt)/dz. In flux form that acts as a spurious source rho*eta_t*dt/dz per step
in the top row, with the full density rho ~ 1000 kg/m^3 (not the anomaly): e.g. up to 1 kg/m^3 per step in the
Rayleigh-Taylor experiment, more than the density contrast allows. The advective form keeps a uniform density uniform.
It is the h -> dz limit of the variable-top-layer budget (see `advect_density_top_layer!`), which writes
d(h rho)/dt with dh/dt = w_bot - div(h u). The fixed-grid mass sum(rho)*dx*dz changes only by
dt*sum_i (rho_top - mean) * div(v)*dx*dz, because sum_i eta_t = 0 in a closed basin; the model's global mass correction
removes this small residual.
Source: v.grad(phi) = div(v*phi) - phi*div(v), as `_top_row_advective_form!` for momentum; z-layer free surface as in
Casulli & Walters (2000), Int. J. Numer. Methods Fluids 32, 331-348 (equation numbers not checked).
"""
function _density_top_row_advective_form!(
    adv::AbstractMatrix{<:Real},
    rho::AbstractMatrix{<:Real},
    u::AbstractMatrix{<:Real},
    w::AbstractMatrix{<:Real},
    dx::Real,
    dz::Real;
    periodic_x::Bool=false,
)
    nz, nx = size(rho)
    if nz < 2
        return adv
    end

    @inbounds for i in 1:nx
        volume_flux_in = i >= 2 ? u[nz, i] : (periodic_x ? u[nz, 1] : 0.0)
        volume_flux_out = i + 1 <= nx ? u[nz, i + 1] : (periodic_x ? u[nz, nx + 1] : 0.0)
        div_v = (volume_flux_out - volume_flux_in) / dx - w[nz, i] / dz   # zero volume flux through the top face
        adv[nz, i] -= rho[nz, i] * div_v
    end

    return adv
end

"""
    advect_density!(rho_new, rho, u, w, dt, dx, dz, ws; scheme="upwind1", top_row_advective_form=false, rho_min=nothing, rho_max=nothing)

One forward-Euler transport step `rho_new = rho - dt * div(v*rho)` of `rho` (kg/m^3) on fixed layers of thickness `dz`
(m) with closed outer faces, followed by optional clamping to `[rho_min, rho_max]`. `dt` in s, `dx`, `dz` in m, `u`, `w`
in m/s. `scheme`: `"upwind1"` (first-order donor cell, monotone for `dt*(|u|/dx + |w|/dz) <= 1`) or `"tvd2"`
(flux-limited Lax-Wendroff with MC limiter, monotone for `dt*(|u|/dx + |w|/dz) <= 1/2`); see `_density_face_fluxes!`.
`top_row_advective_form=true` applies the fixed-grid free-surface correction of `_density_top_row_advective_form!`
(used by the model whenever the variable top layer is off).
"""
function advect_density!(
    rho_new::AbstractMatrix{<:Real},
    rho::AbstractMatrix{<:Real},
    u::AbstractMatrix{<:Real},
    w::AbstractMatrix{<:Real},
    dt::Real,
    dx::Real,
    dz::Real,
    ws::DensityWorkspace;
    scheme::AbstractString="upwind1",
    top_row_advective_form::Bool=false,
    periodic_x::Bool=false,
    active::Union{Nothing,AbstractMatrix{Bool}}=nothing,
    rho_min=nothing,
    rho_max=nothing,
)
    advection_term_rho!(ws.adv, u, w, rho, dx, dz, ws; scheme=scheme, dt=dt, periodic_x=periodic_x, active=active)
    if top_row_advective_form
        _density_top_row_advective_form!(ws.adv, rho, u, w, dx, dz; periodic_x=periodic_x)
    end
    nz, nx = size(rho)

    @inbounds for i in 1:nx
        for k in 1:nz
            rho_new[k, i] = rho[k, i] - dt * ws.adv[k, i]
        end
    end

    if !isnothing(rho_min) || !isnothing(rho_max)
        lo = isnothing(rho_min) ? minimum(rho) : rho_min
        hi = isnothing(rho_max) ? maximum(rho) : rho_max
        @inbounds for i in 1:nx
            for k in 1:nz
                rho_new[k, i] = clamp(rho_new[k, i], lo, hi)
            end
        end
    end

    return rho_new
end

function advect_density(
    rho::AbstractMatrix{<:Real},
    u::AbstractMatrix{<:Real},
    w::AbstractMatrix{<:Real},
    dt::Real,
    dx::Real,
    dz::Real;
    scheme::AbstractString="upwind1",
    top_row_advective_form::Bool=false,
    periodic_x::Bool=false,
    rho_min=nothing,
    rho_max=nothing,
)
    rho_new = similar(rho)
    ws = DensityWorkspace(size(rho, 1), size(rho, 2); T=eltype(rho))
    advect_density!(rho_new, rho, u, w, dt, dx, dz, ws; scheme=scheme, top_row_advective_form=top_row_advective_form, periodic_x=periodic_x, rho_min=rho_min, rho_max=rho_max)
    return rho_new
end

"""
    advect_density_upwind!(rho_new, rho, u, w, dt, dx, dz, ws; rho_min, rho_max)

First-order upwind transport step; equivalent to `advect_density!(...; scheme="upwind1")`.
"""
function advect_density_upwind!(
    rho_new::AbstractMatrix{<:Real},
    rho::AbstractMatrix{<:Real},
    u::AbstractMatrix{<:Real},
    w::AbstractMatrix{<:Real},
    dt::Real,
    dx::Real,
    dz::Real,
    ws::DensityWorkspace;
    rho_min=nothing,
    rho_max=nothing,
)
    return advect_density!(rho_new, rho, u, w, dt, dx, dz, ws; scheme="upwind1", rho_min=rho_min, rho_max=rho_max)
end

function advect_density_upwind(
    rho::AbstractMatrix{<:Real},
    u::AbstractMatrix{<:Real},
    w::AbstractMatrix{<:Real},
    dt::Real,
    dx::Real,
    dz::Real;
    rho_min=nothing,
    rho_max=nothing,
)
    return advect_density(rho, u, w, dt, dx, dz; scheme="upwind1", rho_min=rho_min, rho_max=rho_max)
end

"""
    top_layer_face_thickness!(h_face, eta, dz)

Fill `h_face` (length `nx + 1`, m) with the top-layer thickness at the u-faces, `dz + eta_face`, where `eta_face` is
the mean of the two neighbouring cell values (`eta` in m, length `nx`; `dz` in m). The two outer faces take the value
of the adjacent cell. Requires `dz + eta > 0`.
"""
function top_layer_face_thickness!(h_face::AbstractVector{<:Real}, eta::AbstractVector{<:Real}, dz::Real)
    nx = length(eta)
    h_face[1] = dz + eta[1]
    @inbounds for i in 2:nx
        h_face[i] = dz + 0.5 * (eta[i - 1] + eta[i])
    end
    h_face[nx + 1] = dz + eta[nx]
    return h_face
end

"""
    advect_density_top_layer!(rho_new, rho, u, w, dt, dx, dz, ws, h_top_face, eta_old, eta_new; scheme="upwind1", rho_min, rho_max)

Transport of `rho` (kg/m^3) with the face-value scheme `scheme` (`"upwind1"` or `"tvd2"`, see `_density_face_fluxes!`;
`rho_bot_face` and the top-layer face values below are those of the scheme) on fixed z-layers whose top layer has the variable thickness
`h = dz + eta` (m). Layers `k < nz` have the thickness `dz`. The top layer obeys the budget

    d(h*rho)/dt + d(h_face*u*rho)/dx = w_bot * rho_bot_face          (zero flux through the free surface),

with `h_face = h_top_face` (evaluated with the OLD eta, the same thickness the barotropic solve and the continuity
step use), `w_bot = w[nz, :]` the vertical velocity at the lower face of the top layer (m/s), and the update

    rho_top^{n+1} = (h_old * rho_top^n - dt * (div(h_face*u*rho) - w_bot*rho_bot_face)) / h_new,   h_new = dz + eta_new.

WHY: if `h_new - h_old = dt * (w_bot - div(h_face*u))` holds exactly (it does when `eta_new` comes from the
barotropic solve and `w` from the top-layer continuity), a uniform density stays uniform and the total mass
`sum(rho*h)*dx` is conserved to round-off. The legacy kernel `advect_density_upwind!` sets the top-face flux to zero
although the discrete velocity there is `d eta/dt`; with rho ~ 1000 kg/m^3 that acts as a spurious source of
rho*eta_t*dt/dz per step in the top row.
Source: free-surface z-layer budget as in Casulli & Walters (2000), Int. J. Numer. Methods Fluids 32, 331-348, and the
semi-implicit free surface of Casulli & Cheng (1992), Int. J. Numer. Methods Fluids 15, 629-648 (equation numbers not checked).
Stability: unsplit first-order upwind is monotone if dt*(outflow velocities/dx + outflow velocities/dz) <= 1 per cell,
`"tvd2"` if dt*(|u|/dx + |w|/dz) <= 1/2 per cell.
"""
function advect_density_top_layer!(
    rho_new::AbstractMatrix{<:Real},
    rho::AbstractMatrix{<:Real},
    u::AbstractMatrix{<:Real},
    w::AbstractMatrix{<:Real},
    dt::Real,
    dx::Real,
    dz::Real,
    ws::DensityWorkspace,
    h_top_face::AbstractVector{<:Real},
    eta_old::AbstractVector{<:Real},
    eta_new::AbstractVector{<:Real};
    scheme::AbstractString="upwind1",
    active::Union{Nothing,AbstractMatrix{Bool}}=nothing,
    rho_min=nothing,
    rho_max=nothing,
)
    nz, nx = size(rho)

    # WHY: only the top-layer equation is written for h*rho (thickness-integrated), so only the top-row x-fluxes carry
    # h_top_face; the layers below use the plain density flux, exactly as advect_density!. The face flux at the free
    # surface (flux_z[nz + 1, :]) is zero: material boundary, no flux through the top of the top layer.
    _density_face_fluxes!(ws, rho, u, w, dt, dx, dz, scheme; h_top_face=h_top_face, active=active)

    @inbounds for i in 1:nx
        for k in 1:(nz - 1)
            rho_new[k, i] = rho[k, i] - dt * ((ws.flux_x[k, i + 1] - ws.flux_x[k, i]) / dx + (ws.flux_z[k + 1, i] - ws.flux_z[k, i]) / dz)
        end
        h_old = dz + eta_old[i]   # m
        h_new = dz + eta_new[i]   # m
        rho_new[nz, i] = (h_old * rho[nz, i] - dt * ((ws.flux_x[nz, i + 1] - ws.flux_x[nz, i]) / dx - ws.flux_z[nz, i])) / h_new
    end

    if !isnothing(rho_min) || !isnothing(rho_max)
        lo = isnothing(rho_min) ? minimum(rho) : rho_min
        hi = isnothing(rho_max) ? maximum(rho) : rho_max
        @inbounds for i in 1:nx
            for k in 1:nz
                rho_new[k, i] = clamp(rho_new[k, i], lo, hi)
            end
        end
    end

    return rho_new
end

"""
    advect_density_upwind_top_layer!(rho_new, rho, u, w, dt, dx, dz, ws, h_top_face, eta_old, eta_new; rho_min, rho_max)

First-order upwind variable-top-layer transport; equivalent to `advect_density_top_layer!(...; scheme="upwind1")`.
"""
function advect_density_upwind_top_layer!(
    rho_new::AbstractMatrix{<:Real},
    rho::AbstractMatrix{<:Real},
    u::AbstractMatrix{<:Real},
    w::AbstractMatrix{<:Real},
    dt::Real,
    dx::Real,
    dz::Real,
    ws::DensityWorkspace,
    h_top_face::AbstractVector{<:Real},
    eta_old::AbstractVector{<:Real},
    eta_new::AbstractVector{<:Real};
    rho_min=nothing,
    rho_max=nothing,
)
    return advect_density_top_layer!(rho_new, rho, u, w, dt, dx, dz, ws, h_top_face, eta_old, eta_new; scheme="upwind1", rho_min=rho_min, rho_max=rho_max)
end

function buoyancy!(out::AbstractArray{<:Real}, rho::AbstractArray{<:Real}, rho_0::AbstractArray{<:Real})
    @. out = -9.81 * (rho - rho_0) / rho_0
    return out
end

function buoyancy(rho::AbstractArray{<:Real}, rho_0::AbstractArray{<:Real})
    out = similar(rho)
    buoyancy!(out, rho, rho_0)
    return out
end
