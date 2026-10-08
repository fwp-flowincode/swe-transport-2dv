Base.@kwdef mutable struct VegetationStemConfig{T<:Real}
    base_x::T
    height::T
    base_z::T = zero(T)
    diameter::T = T(0.02)
    drag_coefficient::T = T(1.2)
    stiffness::T = T(2.0)
    damping::T = T(0.35)
    mass_per_length::T = T(1.0)
    sample_points::Int = 8
    source_strength::T = T(1.0)
end

const VEGETATION_RESPONSE_GAIN = 8.0
const VEGETATION_MAX_DEFLECTION_SPEED = 1.5
const VEGETATION_SUBSTEP_DT_MAX = 0.0025
const VEGETATION_FEEDBACK_RELAXATION = 0.35
const VEGETATION_MAX_NODE_STEP_FRACTION = 0.10
const VEGETATION_MAX_TANGENT_X = 0.98

mutable struct VegetationStem{T<:Real}
    config::VegetationStemConfig{T}
    deflection::Vector{T}
    deflection_velocity::Vector{T}
    angle::T
    angular_velocity::T
end

function VegetationStem(; angle::Real=0.0, angular_velocity::Real=0.0, kwargs...)
    cfg = VegetationStemConfig{Float64}(; kwargs...)
    npts = max(3, cfg.sample_points)
    return VegetationStem{Float64}(cfg, zeros(Float64, npts), zeros(Float64, npts), Float64(angle), Float64(angular_velocity))
end

mutable struct VegetationModel{T<:Real}
    stems::Vector{VegetationStem{T}}
end

function VegetationModel(stems::VegetationStem...)
    stem_list = collect(stems)
    if isempty(stem_list)
        return VegetationModel(VegetationStem{Float64}[])
    end
    return VegetationModel(stem_list)
end

function _grid_position_x(x::Real, dx::Real, nxp1::Int)
    return clamp((x / dx) + 1.0, 1.0, nxp1 * 1.0)
end

function _grid_position_z(z::Real, dz::Real, nz::Int)
    return clamp((z / dz) + 0.5, 1.0, nz * 1.0)
end

function _sample_centered_field(field::AbstractMatrix{<:Real}, x::Real, z::Real, dx::Real, dz::Real)
    nz, nx = size(field)
    px = clamp((x / dx) + 0.5, 1.0, nx * 1.0)
    pz = clamp((z / dz) + 0.5, 1.0, nz * 1.0)

    i0 = clamp(floor(Int, px), 1, nx)
    i1 = min(i0 + 1, nx)
    k0 = clamp(floor(Int, pz), 1, nz)
    k1 = min(k0 + 1, nz)

    tx = i1 == i0 ? 0.0 : px - i0
    tz = k1 == k0 ? 0.0 : pz - k0

    f00 = field[k0, i0]
    f10 = field[k0, i1]
    f01 = field[k1, i0]
    f11 = field[k1, i1]

    return (1.0 - tx) * (1.0 - tz) * f00 + tx * (1.0 - tz) * f10 + (1.0 - tx) * tz * f01 + tx * tz * f11
end

function _sample_u(u::AbstractMatrix{<:Real}, x::Real, z::Real, dx::Real, dz::Real)
    nz, nxp1 = size(u)
    px = _grid_position_x(x, dx, nxp1)
    pz = _grid_position_z(z, dz, nz)

    i0 = clamp(floor(Int, px), 1, nxp1)
    i1 = min(i0 + 1, nxp1)
    k0 = clamp(floor(Int, pz), 1, nz)
    k1 = min(k0 + 1, nz)

    tx = i1 == i0 ? 0.0 : px - i0
    tz = k1 == k0 ? 0.0 : pz - k0

    u00 = u[k0, i0]
    u10 = u[k0, i1]
    u01 = u[k1, i0]
    u11 = u[k1, i1]

    return (1.0 - tx) * (1.0 - tz) * u00 + tx * (1.0 - tz) * u10 + (1.0 - tx) * tz * u01 + tx * tz * u11
end

function _deposit_u_source!(source_u::AbstractMatrix{<:Real}, x::Real, z::Real, value::Real, dx::Real, dz::Real)
    nz, nxp1 = size(source_u)
    px = _grid_position_x(x, dx, nxp1)
    pz = _grid_position_z(z, dz, nz)

    i0 = clamp(floor(Int, px), 1, nxp1)
    i1 = min(i0 + 1, nxp1)
    k0 = clamp(floor(Int, pz), 1, nz)
    k1 = min(k0 + 1, nz)

    tx = i1 == i0 ? 0.0 : px - i0
    tz = k1 == k0 ? 0.0 : pz - k0

    w00 = (1.0 - tx) * (1.0 - tz)
    w10 = tx * (1.0 - tz)
    w01 = (1.0 - tx) * tz
    w11 = tx * tz

    source_u[k0, i0] += value * w00
    source_u[k0, i1] += value * w10
    source_u[k1, i0] += value * w01
    source_u[k1, i1] += value * w11

    return nothing
end

function _smooth_stem_load!(load::AbstractVector{<:Real})
    npts = length(load)
    if npts < 3
        return load
    end

    smoothed = similar(load)
    smoothed[1] = load[1]
    smoothed[end] = load[end]
    @inbounds for j in 2:(npts - 1)
        smoothed[j] = 0.25 * load[j - 1] + 0.5 * load[j] + 0.25 * load[j + 1]
    end
    copyto!(load, smoothed)
    return load
end

function _stem_node_positions(stem::VegetationStem{T}) where {T<:Real}
    cfg = stem.config
    npts = length(stem.deflection)
    ds = cfg.height / max(npts - 1, 1)

    x = zeros(T, npts)
    z = zeros(T, npts)
    x[1] = cfg.base_x
    z[1] = cfg.base_z

    if npts == 1
        return x, z
    end

    @inbounds for j in 2:npts
        tangent_x = (stem.deflection[j] - stem.deflection[j - 1]) / max(ds, eps(T))
        tangent_x = clamp(tangent_x, -VEGETATION_MAX_TANGENT_X, VEGETATION_MAX_TANGENT_X)
        tangent_z = sqrt(max(1.0 - tangent_x * tangent_x, eps(T)))

        x[j] = x[j - 1] + tangent_x * ds
        z[j] = z[j - 1] + tangent_z * ds
    end

    return x, z
end

function _update_stem_and_source!(
    source_u::AbstractMatrix{<:Real},
    stem::VegetationStem{T},
    u::AbstractMatrix{<:Real},
    p::Union{Nothing,AbstractMatrix{<:Real}},
    dt::Real,
    dx::Real,
    dz::Real,
    rho0::Real,
) where {T<:Real}
    cfg = stem.config
    npts = length(stem.deflection)
    ds = cfg.height / max(npts - 1, 1)
    load = zeros(T, npts)
    slope_scale = max(ds, eps(T))
    max_node_step = VEGETATION_MAX_NODE_STEP_FRACTION * ds

    # Guard against occasional non-finite values during strong coupling transients.
    @inbounds for j in 1:npts
        if !isfinite(stem.deflection[j])
            stem.deflection[j] = 0.0
        end
        if !isfinite(stem.deflection_velocity[j])
            stem.deflection_velocity[j] = 0.0
        end
    end

    x_nodes, z_nodes = _stem_node_positions(stem)

    @inbounds for j in 1:npts
        arc = (j - 1) * ds
        x = x_nodes[j]
        z = z_nodes[j]
        local_slope = 0.0
        if j > 1
            local_slope += stem.deflection[j] - stem.deflection[j - 1]
        end
        if j < npts
            local_slope += stem.deflection[j + 1] - stem.deflection[j]
        end
        projected_area_factor = max(0.15, 1.0 / sqrt(1.0 + (local_slope / slope_scale)^2))

        if !isfinite(x) || !isfinite(z)
            x = cfg.base_x
            z = cfg.base_z + arc
        end

        u_local = _sample_u(u, x, z, dx, dz)
        if !isfinite(u_local)
            u_local = 0.0
        end
        stem_velocity_x = stem.deflection_velocity[j]
        u_rel = u_local - stem_velocity_x
        drag_force = 0.5 * rho0 * cfg.drag_coefficient * cfg.diameter * ds * projected_area_factor * u_rel * abs(u_rel)
        pressure_force = 0.0
        if p !== nothing
            p_left = _sample_centered_field(p, x - 0.5 * dx, z, dx, dz)
            p_right = _sample_centered_field(p, x + 0.5 * dx, z, dx, dz)
            pressure_force = -(p_right - p_left) * cfg.diameter * ds / max(dx, eps(T))
        end
        load[j] = drag_force + pressure_force
    end

    _smooth_stem_load!(load)

    @inbounds for j in 1:npts
        x = x_nodes[j]
        z = z_nodes[j]
        _deposit_u_source!(source_u, x, z, -cfg.source_strength * load[j] / (rho0 * dx * dz), dx, dz)
    end

    if npts >= 1
        stem.deflection[1] = 0.0
        stem.deflection_velocity[1] = 0.0
    end
    if npts >= 2
        stem.deflection[2] = 0.0
        stem.deflection_velocity[2] = 0.0
    end

    mass_per_node = max(cfg.mass_per_length * ds, eps(T))
    for j in 3:(npts - 1)
        bend = cfg.stiffness * (stem.deflection[j - 1] - 2.0 * stem.deflection[j] + stem.deflection[j + 1]) / (ds * ds)
        forcing = VEGETATION_RESPONSE_GAIN * (load[j] + bend) / mass_per_node
        damping_fac = max(1.0 + dt * VEGETATION_RESPONSE_GAIN * cfg.damping / mass_per_node, 1.0)
        v_new = (stem.deflection_velocity[j] + dt * forcing) / damping_fac
        v_new = clamp(v_new, -VEGETATION_MAX_DEFLECTION_SPEED, VEGETATION_MAX_DEFLECTION_SPEED)
        step = clamp(dt * v_new, -max_node_step, max_node_step)
        stem.deflection_velocity[j] = v_new
        stem.deflection[j] += step
    end

    if npts > 2
        top_bend = cfg.stiffness * (stem.deflection[npts - 1] - stem.deflection[npts]) / (ds * ds)
        forcing_top = VEGETATION_RESPONSE_GAIN * (load[npts] + top_bend) / mass_per_node
        damping_fac_top = max(1.0 + dt * VEGETATION_RESPONSE_GAIN * cfg.damping / mass_per_node, 1.0)
        v_top = (stem.deflection_velocity[npts] + dt * forcing_top) / damping_fac_top
        v_top = clamp(v_top, -VEGETATION_MAX_DEFLECTION_SPEED, VEGETATION_MAX_DEFLECTION_SPEED)
        step_top = clamp(dt * v_top, -max_node_step, max_node_step)
        stem.deflection_velocity[npts] = v_top
        stem.deflection[npts] += step_top
    end

    stem.deflection[1] = 0.0
    stem.deflection_velocity[1] = 0.0
    stem.deflection[2] = 0.0
    stem.deflection_velocity[2] = 0.0

    max_deflection = 0.95 * cfg.height
    @inbounds for j in 3:npts
        stem.deflection[j] = clamp(stem.deflection[j], -max_deflection, max_deflection)
    end

    if npts > 1
        stem.angle = atan((stem.deflection[end] - stem.deflection[end - 1]) / ds)
        stem.angular_velocity = (stem.deflection_velocity[end] - stem.deflection_velocity[end - 1]) / ds
    else
        stem.angle = 0.0
        stem.angular_velocity = 0.0
    end

    return nothing
end

function vegetation_path(stem::VegetationStem{T}; npts::Int=24) where {T<:Real}
    npts = max(2, min(npts, length(stem.deflection)))
    x_all, z_all = _stem_node_positions(stem)
    x = x_all[1:npts]
    z = z_all[1:npts]
    return x, z
end

function copy_vegetation(stem::VegetationStem{T}) where {T<:Real}
    cfg = stem.config
    copied_cfg = VegetationStemConfig{T}(
        base_x=cfg.base_x,
        height=cfg.height,
        base_z=cfg.base_z,
        diameter=cfg.diameter,
        drag_coefficient=cfg.drag_coefficient,
        stiffness=cfg.stiffness,
        damping=cfg.damping,
        mass_per_length=cfg.mass_per_length,
        sample_points=cfg.sample_points,
        source_strength=cfg.source_strength,
    )
    return VegetationStem{T}(copied_cfg, copy(stem.deflection), copy(stem.deflection_velocity), stem.angle, stem.angular_velocity)
end

function copy_vegetation(vegetation::VegetationModel{T}) where {T<:Real}
    return VegetationModel{T}([copy_vegetation(stem) for stem in vegetation.stems])
end

function vegetation_source!(
    source_u::AbstractMatrix{<:Real},
    vegetation::VegetationModel{T},
    u::AbstractMatrix{<:Real},
    dt::Real,
    dx::Real,
    dz::Real,
    rho0::Real,
) where {T<:Real}
    return vegetation_source!(source_u, vegetation, u, nothing, dt, dx, dz, rho0)
end

function vegetation_source!(
    source_u::AbstractMatrix{<:Real},
    vegetation::VegetationModel{T},
    u::AbstractMatrix{<:Real},
    p::AbstractMatrix{<:Real},
    dt::Real,
    dx::Real,
    dz::Real,
    rho0::Real,
) where {T<:Real}
    fill!(source_u, 0.0)
    nsub = max(1, ceil(Int, dt / VEGETATION_SUBSTEP_DT_MAX))
    dt_sub = dt / nsub
    source_sub = similar(source_u)

    for _ in 1:nsub
        fill!(source_sub, 0.0)
        for stem in vegetation.stems
            _update_stem_and_source!(source_sub, stem, u, p, dt_sub, dx, dz, rho0)
        end
        @. source_u = source_u + source_sub / nsub
    end

    @. source_u = VEGETATION_FEEDBACK_RELAXATION * source_u
    return source_u
end

function attach_vegetation!(model, vegetation::VegetationModel)
    model.vegetation = vegetation
    return model
end

function clear_vegetation!(model)
    model.vegetation = nothing
    return model
end