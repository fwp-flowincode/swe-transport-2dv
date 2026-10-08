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
end

function DensityWorkspace(nz::Int, nx::Int; T::Type{<:Real}=Float64)
    return DensityWorkspace(
        zeros(T, nz, nx + 1),
        zeros(T, nz + 1, nx),
        zeros(T, nz, nx),
    )
end

function grad_x_c!(res::AbstractMatrix{<:Real}, f::AbstractMatrix{<:Real}, dx::Real)
    nz, nx = size(f)
    fill!(res, 0.0)
    @inbounds for i in 2:nx
        for k in 1:nz
            res[k, i] = (f[k, i] - f[k, i - 1]) / dx
        end
    end
    return res
end

function grad_x_c(f::AbstractMatrix{<:Real}, dx::Real)
    res = zeros(eltype(f), size(f, 1), size(f, 2) + 1)
    return grad_x_c!(res, f, dx)
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

function _interpolate_w_to_u!(w_u::AbstractMatrix{<:Real}, w::AbstractMatrix{<:Real}, w_center::AbstractMatrix{<:Real})
    nz, nx = size(w_center)

    @inbounds for i in 1:nx
        for k in 1:nz
            w_center[k, i] = 0.5 * (w[k, i] + w[k + 1, i])
        end
    end

    @inbounds for k in 1:nz
        w_u[k, 1] = w_center[k, 1]
        for i in 2:nx
            w_u[k, i] = 0.5 * (w_center[k, i - 1] + w_center[k, i])
        end
        w_u[k, nx + 1] = w_center[k, nx]
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

function _face_velocity_x!(vx_face::AbstractMatrix{<:Real}, vx::AbstractMatrix{<:Real})
    nz, nx = size(vx)
    @inbounds for k in 1:nz
        vx_face[k, 1] = vx[k, 1]
        for i in 2:nx
            vx_face[k, i] = 0.5 * (vx[k, i - 1] + vx[k, i])
        end
        vx_face[k, nx + 1] = vx[k, nx]
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

function _advective_flux_divergence!(
    out::AbstractMatrix{<:Real},
    phi::AbstractMatrix{<:Real},
    vx::AbstractMatrix{<:Real},
    vz::AbstractMatrix{<:Real},
    dx::Real,
    dz::Real,
    scheme::AbstractString,
    ws::AdvectionWorkspace,
)
    nz, nx = size(phi)

    if scheme != "upwind1" && scheme != "tvd2"
        throw(ArgumentError("Unsupported advection scheme '$(scheme)'"))
    end

    _face_velocity_x!(ws.vx_face, vx)
    _face_velocity_z!(ws.vz_face, vz)

    if scheme == "upwind1"
        @inbounds for k in 1:nz
            ws.flux_x[k, 1] = 0.0
            for i in 2:nx
                v = ws.vx_face[k, i]
                if v >= 0.0
                    ws.flux_x[k, i] = v * phi[k, i - 1]
                else
                    ws.flux_x[k, i] = v * phi[k, i]
                end
            end
            ws.flux_x[k, nx + 1] = 0.0
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
            end
            ws.flux_z[nz + 1, i] = 0.0
        end
    else
        fill!(ws.slope_x, 0.0)
        fill!(ws.slope_z, 0.0)

        if nx >= 3
            @inbounds for i in 2:(nx - 1)
                for k in 1:nz
                    a = phi[k, i] - phi[k, i - 1]
                    b = phi[k, i + 1] - phi[k, i]
                    ws.slope_x[k, i] = _monotonized_central_scalar(a, b)
                end
            end
        end

        if nz >= 3
            @inbounds for i in 1:nx
                for k in 2:(nz - 1)
                    a = phi[k, i] - phi[k - 1, i]
                    b = phi[k + 1, i] - phi[k, i]
                    ws.slope_z[k, i] = _monotonized_central_scalar(a, b)
                end
            end
        end

        @inbounds for k in 1:nz
            ws.flux_x[k, 1] = 0.0
            for i in 2:nx
                left_x = phi[k, i - 1] + 0.5 * ws.slope_x[k, i - 1]
                right_x = phi[k, i] - 0.5 * ws.slope_x[k, i]
                v = ws.vx_face[k, i]
                ws.flux_x[k, i] = v >= 0.0 ? v * left_x : v * right_x
            end
            ws.flux_x[k, nx + 1] = 0.0
        end

        @inbounds for i in 1:nx
            ws.flux_z[1, i] = 0.0
            for k in 2:nz
                lower_z = phi[k - 1, i] + 0.5 * ws.slope_z[k - 1, i]
                upper_z = phi[k, i] - 0.5 * ws.slope_z[k, i]
                v = ws.vz_face[k, i]
                ws.flux_z[k, i] = v >= 0.0 ? v * lower_z : v * upper_z
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
    ws::AdvectionWorkspace,
)
    nz, nx = size(phi)
    if nz < 2
        return out
    end

    @inbounds for i in 1:nx
        volume_flux_in = i >= 2 ? ws.vx_face[nz, i] : 0.0           # x-face volume flux, zero at the outer boundary
        volume_flux_out = i + 1 <= nx ? ws.vx_face[nz, i + 1] : 0.0
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
)
    _interpolate_w_to_u!(ws_interp.w_u, w, ws_interp.w_center)
    _advective_flux_divergence!(out, u, u, ws_interp.w_u, dx, dz, scheme, ws_adv)
    if top_row_advective_form
        _top_row_advective_form!(out, u, dx, dz, ws_adv)
    end
    return out
end

function advection_term_u(
    u::AbstractMatrix{<:Real},
    w::AbstractMatrix{<:Real},
    dx::Real,
    dz::Real;
    scheme::AbstractString="upwind1",
)
    out = zeros(eltype(u), size(u))
    nz, nxp1 = size(u)
    nx = nxp1 - 1
    ws_adv = AdvectionWorkspace(nz, nxp1; T=eltype(u))
    ws_interp = InterpolationWorkspace(nz, nx; T=eltype(u))
    advection_term_u!(out, u, w, dx, dz, scheme, ws_adv, ws_interp)
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
)
    _interpolate_u_to_w!(ws_interp.u_w, u, ws_interp.u_center)
    _advective_flux_divergence!(out, w, ws_interp.u_w, w, dx, dz, scheme, ws_adv)
    return out
end

function advection_term_w(
    u::AbstractMatrix{<:Real},
    w::AbstractMatrix{<:Real},
    dx::Real,
    dz::Real;
    scheme::AbstractString="upwind1",
)
    out = zeros(eltype(w), size(w))
    nzp1, nx = size(w)
    nz = nzp1 - 1
    ws_adv = AdvectionWorkspace(nzp1, nx; T=eltype(w))
    ws_interp = InterpolationWorkspace(nz, nx; T=eltype(u))
    advection_term_w!(out, u, w, dx, dz, scheme, ws_adv, ws_interp)
    return out
end

function diffusion_term!(res::AbstractMatrix{<:Real}, phi::AbstractMatrix{<:Real}, nu::Real, dx::Real, dz::Real)
    nz, nx = size(phi)
    fill!(res, 0.0)
    @inbounds for i in 2:(nx - 1)
        for k in 2:(nz - 1)
            lap_x = (phi[k, i + 1] - 2.0 * phi[k, i] + phi[k, i - 1]) / (dx * dx)
            lap_z = (phi[k + 1, i] - 2.0 * phi[k, i] + phi[k - 1, i]) / (dz * dz)
            res[k, i] = nu * (lap_x + lap_z)
        end
    end
    return res
end

function diffusion_term!(res::AbstractMatrix{<:Real}, phi::AbstractMatrix{<:Real}, nu::AbstractMatrix{<:Real}, dx::Real, dz::Real)
    nz, nx = size(phi)
    fill!(res, 0.0)
    @inbounds for i in 2:(nx - 1)
        for k in 2:(nz - 1)
            lap_x = (phi[k, i + 1] - 2.0 * phi[k, i] + phi[k, i - 1]) / (dx * dx)
            lap_z = (phi[k + 1, i] - 2.0 * phi[k, i] + phi[k - 1, i]) / (dz * dz)
            res[k, i] = nu[k, i] * (lap_x + lap_z)
        end
    end
    return res
end

function diffusion_term(phi::AbstractMatrix{<:Real}, nu::Real, dx::Real, dz::Real)
    nz, nx = size(phi)
    res = zeros(eltype(phi), nz, nx)
    diffusion_term!(res, phi, nu, dx, dz)
    return res
end

function advection_term_rho!(
    adv::AbstractMatrix{<:Real},
    u::AbstractMatrix{<:Real},
    w::AbstractMatrix{<:Real},
    rho::AbstractMatrix{<:Real},
    dx::Real,
    dz::Real,
    ws::DensityWorkspace,
)
    nz, nx = size(rho)

    @inbounds for k in 1:nz
        ws.flux_x[k, 1] = 0.0
        for i in 2:nx
            v = u[k, i]
            if v > 0.0
                ws.flux_x[k, i] = v * rho[k, i - 1]
            elseif v < 0.0
                ws.flux_x[k, i] = v * rho[k, i]
            else
                ws.flux_x[k, i] = 0.0
            end
        end
        ws.flux_x[k, nx + 1] = 0.0
    end

    @inbounds for i in 1:nx
        ws.flux_z[1, i] = 0.0
        for k in 2:nz
            v = w[k, i]
            if v > 0.0
                ws.flux_z[k, i] = v * rho[k - 1, i]
            elseif v < 0.0
                ws.flux_z[k, i] = v * rho[k, i]
            else
                ws.flux_z[k, i] = 0.0
            end
        end
        ws.flux_z[nz + 1, i] = 0.0
    end

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
    dz::Real,
)
    nz, nx = size(rho)
    ws = DensityWorkspace(nz, nx; T=eltype(rho))
    adv = zeros(eltype(rho), nz, nx)
    advection_term_rho!(adv, u, w, rho, dx, dz, ws)
    return adv
end

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
    advection_term_rho!(ws.adv, u, w, rho, dx, dz, ws)
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
    rho_new = similar(rho)
    ws = DensityWorkspace(size(rho, 1), size(rho, 2); T=eltype(rho))
    advect_density_upwind!(rho_new, rho, u, w, dt, dx, dz, ws; rho_min=rho_min, rho_max=rho_max)
    return rho_new
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
    advect_density_upwind_top_layer!(rho_new, rho, u, w, dt, dx, dz, ws, h_top_face, eta_old, eta_new; rho_min, rho_max)

First-order upwind transport of `rho` (kg/m^3) on fixed z-layers whose top layer has the variable thickness
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
Stability: unsplit first-order upwind is monotone if dt*(outflow velocities/dx + outflow velocities/dz) <= 1 per cell.
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
    nz, nx = size(rho)

    @inbounds for k in 1:nz
        ws.flux_x[k, 1] = 0.0
        for i in 2:nx
            v = u[k, i]
            rho_face = v > 0.0 ? rho[k, i - 1] : (v < 0.0 ? rho[k, i] : 0.0)
            # WHY: only the top-layer equation is written for h*rho (thickness-integrated); the layers below use the
            # plain density flux, exactly as advect_density_upwind!.
            ws.flux_x[k, i] = k == nz ? h_top_face[i] * v * rho_face : v * rho_face
        end
        ws.flux_x[k, nx + 1] = 0.0
    end

    @inbounds for i in 1:nx
        ws.flux_z[1, i] = 0.0
        for k in 2:nz
            v = w[k, i]
            ws.flux_z[k, i] = v > 0.0 ? v * rho[k - 1, i] : (v < 0.0 ? v * rho[k, i] : 0.0)
        end
        ws.flux_z[nz + 1, i] = 0.0   # material free surface: no flux through the top of the top layer
    end

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

function buoyancy!(out::AbstractArray{<:Real}, rho::AbstractArray{<:Real}, rho_0::AbstractArray{<:Real})
    @. out = -9.81 * (rho - rho_0) / rho_0
    return out
end

function buoyancy(rho::AbstractArray{<:Real}, rho_0::AbstractArray{<:Real})
    out = similar(rho)
    buoyancy!(out, rho, rho_0)
    return out
end
