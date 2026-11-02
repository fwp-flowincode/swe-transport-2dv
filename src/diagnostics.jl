function water_volume(depth::Real, eta::AbstractVector{<:Real}, dx::Real)
    return sum(depth .+ eta) * dx
end

# Constituent mass per unit width (kg/m). With `eta` (m, length nx) given, the top layer has the thickness dz + eta
# (variable top layer); without it all cells have the volume dx*dz (legacy fixed grid).
function constituent_mass(rho::AbstractMatrix{<:Real}, dx::Real, dz::Real; eta::Union{Nothing,AbstractVector{<:Real}}=nothing)
    mass = sum(rho) * dx * dz
    if eta !== nothing
        nz = size(rho, 1)
        @inbounds for i in eachindex(eta)
            mass += rho[nz, i] * eta[i] * dx
        end
    end
    return mass
end

"""
    reattachment_length(u, h, xc, x_start) -> m

Length of a recirculation zone behind `x_start` (m), from a time-mean flow: the distance to the last column with
reverse transport near the bed, i.e. where the dividing streamline psi(z) = int_0^z u dz' = 0 lies above the bed (Luo et
al. 2023, JFM 966 A18). `u` (m/s) and `h` (m) are cell values (nz, nx), `xc` (m) the cell centres. Linear interpolation
of the lowest psi between the columns; 0 without reverse transport.
"""
function reattachment_length(u::AbstractMatrix{<:Real}, h::AbstractMatrix{<:Real}, xc::AbstractVector{<:Real}, x_start::Real)
    lowest = zeros(length(xc))   # m^2/s, lowest value of psi in each column (<= 0)
    @inbounds for i in eachindex(xc), k in axes(u, 1)
        psi = 0.0
        for kk in 1:k
            psi += u[kk, i] * h[kk, i]
        end
        lowest[i] = min(lowest[i], psi)
    end
    last = findlast(i -> xc[i] > x_start && lowest[i] < 0.0, eachindex(xc))
    last === nothing && return 0.0
    last == length(xc) && return xc[last] - x_start
    # WHY the lowest psi is 0 from the reattachment point on: the interpolation between the last column with reverse
    # transport and the next takes the bed velocity, which changes sign there
    ub, ua = u[findfirst(>(0.0), view(h, :, last)), last], u[findfirst(>(0.0), view(h, :, last + 1)), last + 1]
    t = ua > ub ? clamp(-ub / (ua - ub), 0.0, 1.0) : 0.0
    return xc[last] + t * (xc[last + 1] - xc[last]) - x_start
end

function boundary_normal_flux(u::AbstractMatrix{<:Real}, w::AbstractMatrix{<:Real}, dx::Real, dz::Real)
    flux_left = -sum(@view u[:, 1]) * dz
    flux_right = sum(@view u[:, end]) * dz
    flux_bottom = -sum(@view w[1, :]) * dx
    flux_top = sum(@view w[end, :]) * dx
    return flux_left + flux_right + flux_bottom + flux_top
end

function cfl_numbers(u::AbstractMatrix{<:Real}, w::AbstractMatrix{<:Real}, dt::Real, dx::Real, dz::Real)
    max_u = maximum(abs, u)
    max_w = maximum(abs, w)
    cfl_x = max_u * dt / dx
    cfl_z = max_w * dt / dz
    return cfl_x, cfl_z, cfl_x + cfl_z
end

function reduced_gravity(rho::AbstractMatrix{<:Real}, rho0::Real, g::Real)
    delta_rho = maximum(rho) - minimum(rho)
    return g * delta_rho / max(abs(rho0), 1e-12)
end

function internal_wave_speed(rho::AbstractMatrix{<:Real}, rho0::Real, g::Real, depth::Real)
    g_prime = reduced_gravity(rho, rho0, g)
    return sqrt(max(g_prime * depth, 0.0))
end

function froude_number(u::AbstractMatrix{<:Real}, rho::AbstractMatrix{<:Real}, rho0::Real, g::Real, depth::Real)
    c_int = internal_wave_speed(rho, rho0, g, depth)
    umax = maximum(abs, u)
    if c_int <= 1e-12
        return umax <= 1e-12 ? 0.0 : Inf
    end
    return umax / c_int
end
