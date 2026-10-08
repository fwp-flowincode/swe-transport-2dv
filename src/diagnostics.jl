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
