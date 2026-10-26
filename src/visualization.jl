const QUIVER_STRIDE = 6               # -, minimum stride (cells) between velocity arrows
const QUIVER_TARGET_COUNT_X = 60      # -, at most about this many arrows across the domain
const QUIVER_TARGET_COUNT_Z = 12      # -, at most about this many arrows over the depth
const QUIVER_TARGET_LENGTH_FACTOR = 0.09
const VECTOR_LINE_COLOR = :white
const VECTOR_LINE_WIDTH = 1.4
const VECTOR_MIN_VISIBLE_FACTOR = 0.0
const VECTOR_MIN_DRAW_FACTOR = 0.02
const VECTOR_HEAD_LENGTH_FACTOR = 0.12
const VECTOR_HEAD_ANGLE_RAD = 0.35
const PLOT_RANGE_MIN_SPAN = 1e-6
const VEGETATION_DISPLAY_EXAGGERATION = 1.0
const VEGETATION_ENFORCE_INEXTENSIBLE_DISPLAY = true

const _PLOTS_BACKEND_CONFIGURED = Ref(false)

function _ensure_batch_plot_backend!()
    if _PLOTS_BACKEND_CONFIGURED[]
        return nothing
    end

    # Prefer a non-interactive GR workstation in batch runs if the user did not override it.
    if !haskey(ENV, "GKSwstype") && !haskey(ENV, "GKS_WSTYPE")
        ENV["GKSwstype"] = "100"
        ENV["GKS_WSTYPE"] = "100"
    end

    try
        gr()
    catch
    end

    default(show=false)
    _PLOTS_BACKEND_CONFIGURED[] = true
    return nothing
end

"""
    _blank_solid_cells(field, config) -> field or copy

Cell-centred `field` (nz x nx) with NaN in the cells below `config.bed_level` (variable bed), so that heatmaps leave the
bed blank and `_finite_clims` ignores it. Returns `field` itself for a flat bed.
"""
function _blank_solid_cells(field::AbstractMatrix{<:Real}, config)
    if config.bed_level === nothing
        return field
    end
    nz, nx = size(field)
    geo = BedGeometry(config.bed_level, nz, nx, config.depth / nz)
    if !geo.any_solid
        return field
    end
    blanked = Float64.(field)
    blanked[.!geo.wet_cell] .= NaN
    return blanked
end

function _finite_clims(data; center::Bool=false, min_span::Real=PLOT_RANGE_MIN_SPAN)
    lo = Inf
    hi = -Inf
    found = false

    @inbounds for x in data
        if isfinite(x)
            xv = Float64(x)
            lo = min(lo, xv)
            hi = max(hi, xv)
            found = true
        end
    end

    if !found
        return center ? (-1.0, 1.0) : (0.0, 1.0)
    end

    if center
        m = max(abs(lo), abs(hi), min_span)
        return (-m, m)
    end

    if (hi - lo) < min_span
        mid = 0.5 * (hi + lo)
        half = max(abs(mid) * 0.05, min_span / 2.0)
        return (mid - half, mid + half)
    end

    return (lo, hi)
end

function _save_plot!(fig, filename::AbstractString)
    try
        savefig(fig, filename)
    catch err
        # GR may throw intermittently on heavy batches; retry once after a GC cycle.
        @warn "savefig failed, retrying once" file=filename exception=(err, catch_backtrace())
        GC.gc()
        savefig(fig, filename)
    end
    return nothing
end

function _cell_center_velocity!(u_c::AbstractMatrix{<:Real}, w_c::AbstractMatrix{<:Real}, u::AbstractMatrix{<:Real}, w::AbstractMatrix{<:Real})
    nz, nx = size(u_c)
    @inbounds for i in 1:nx
        for k in 1:nz
            u_c[k, i] = 0.5 * (u[k, i] + u[k, i + 1])
            w_c[k, i] = 0.5 * (w[k, i] + w[k + 1, i])
        end
    end
    return u_c, w_c
end

function _cell_center_velocity(u::AbstractMatrix{<:Real}, w::AbstractMatrix{<:Real})
    nz, nxp1 = size(u)
    nx = nxp1 - 1
    u_c = zeros(eltype(u), nz, nx)
    w_c = zeros(eltype(w), nz, nx)
    return _cell_center_velocity!(u_c, w_c, u, w)
end

# WHY: a fixed stride gives arrow counts that grow with the grid (1500 x 200: ~8500 arrows, 3000 x 400: ~33500) and
# arrow fields too dense to read; the stride grows so that at most about QUIVER_TARGET_COUNT_X x QUIVER_TARGET_COUNT_Z
# arrows are drawn, with QUIVER_STRIDE as the minimum (small grids are unchanged).
function _quiver_sampling(config)
    sx = max(QUIVER_STRIDE, cld(config.nx, QUIVER_TARGET_COUNT_X))
    sz = max(QUIVER_STRIDE, cld(config.nz, QUIVER_TARGET_COUNT_Z))
    i_idx = collect(1:sz:config.nz)
    j_idx = collect(1:sx:config.nx)
    return i_idx, j_idx
end

function _quiver_scale(u_c::AbstractMatrix{<:Real}, w_c::AbstractMatrix{<:Real}, config)
    speed = Float64[]
    @inbounds for idx in eachindex(u_c)
        ui = u_c[idx]
        wi = w_c[idx]
        if !isfinite(ui) || !isfinite(wi)
            continue
        end
        push!(speed, hypot(ui, wi))
    end

    speed95 = isempty(speed) ? 1e-12 : quantile(speed, 0.95)
    speed95 = max(speed95, 1e-12)
    target_len = max(QUIVER_TARGET_LENGTH_FACTOR * config.depth, 1e-12)
    return speed95 / target_len
end

function _quiver_vectors(
    u_c::AbstractMatrix{<:Real},
    w_c::AbstractMatrix{<:Real},
    x_centers,
    z_centers,
    i_idx::AbstractVector{Int},
    j_idx::AbstractVector{Int},
    q_scale::Real,
    min_visible_len::Real=0.0,
)
    ni = length(i_idx)
    nj = length(j_idx)
    npts = ni * nj

    xq = Vector{Float64}(undef, npts)
    zq = Vector{Float64}(undef, npts)
    uq = Vector{Float64}(undef, npts)
    wq = Vector{Float64}(undef, npts)

    idx = 1
    @inbounds for j in j_idx
        xj = x_centers[j]
        for i in i_idx
            xq[idx] = xj
            zq[idx] = z_centers[i]
            u_loc = u_c[i, j] / q_scale
            w_loc = w_c[i, j] / q_scale
            if min_visible_len > 0.0
                mag = hypot(u_loc, w_loc)
                if mag > 0.0 && mag < min_visible_len
                    s = min_visible_len / mag
                    u_loc *= s
                    w_loc *= s
                end
            end
            uq[idx] = u_loc
            wq[idx] = w_loc
            idx += 1
        end
    end

    return xq, zq, uq, wq
end

function _plot_velocity_vectors!(
    p,
    xq::AbstractVector{<:Real},
    zq::AbstractVector{<:Real},
    uq::AbstractVector{<:Real},
    wq::AbstractVector{<:Real};
    color=VECTOR_LINE_COLOR,
    linewidth::Real=VECTOR_LINE_WIDTH,
    min_draw_mag::Real=0.0,
)
    n = length(xq)
    xk = Vector{Float64}(undef, n)
    zk = Vector{Float64}(undef, n)
    uk = Vector{Float64}(undef, n)
    wk = Vector{Float64}(undef, n)

    kept = 0
    @inbounds for idx in 1:n
        xi = xq[idx]
        zi = zq[idx]
        ui = uq[idx]
        wi = wq[idx]
        if !isfinite(xi) || !isfinite(zi) || !isfinite(ui) || !isfinite(wi)
            continue
        end
        if hypot(ui, wi) <= min_draw_mag
            continue
        end
        kept += 1
        xk[kept] = xi
        zk[kept] = zi
        uk[kept] = ui
        wk[kept] = wi
    end

    if kept == 0
        return p
    end

    # WHY: all shafts and heads go into ONE line series, separated by NaN (Plots breaks a line at NaN). The former
    # three plot! calls per arrow created ~25000 series per frame on 1500 x 200 and made the built-in animation take
    # hours; the drawn segments are the same.
    # PERF: 9 points per arrow (3 segments x (2 points + NaN break)), preallocated once per call.
    xs = fill(NaN, 9 * kept)
    zs = fill(NaN, 9 * kept)
    c = cos(VECTOR_HEAD_ANGLE_RAD)
    s = sin(VECTOR_HEAD_ANGLE_RAD)
    @inbounds for idx in 1:kept
        x0 = xk[idx]
        z0 = zk[idx]
        u0 = uk[idx]
        w0 = wk[idx]
        x1 = x0 + u0
        z1 = z0 + w0
        o = 9 * (idx - 1)

        xs[o + 1], zs[o + 1] = x0, z0       # shaft
        xs[o + 2], zs[o + 2] = x1, z1

        mag = hypot(u0, w0)
        if mag <= eps(Float64)
            continue                         # no head; its points stay NaN (not drawn)
        end

        head_len = VECTOR_HEAD_LENGTH_FACTOR * mag
        ux = u0 / mag
        wz = w0 / mag
        hx1 = -c * ux + s * wz
        hz1 = -s * ux - c * wz
        hx2 = -c * ux - s * wz
        hz2 = s * ux - c * wz

        xs[o + 4], zs[o + 4] = x1, z1       # first head line
        xs[o + 5], zs[o + 5] = x1 + head_len * hx1, z1 + head_len * hz1
        xs[o + 7], zs[o + 7] = x1, z1       # second head line
        xs[o + 8], zs[o + 8] = x1 + head_len * hx2, z1 + head_len * hz2
    end
    plot!(p, xs, zs; color=color, linewidth=linewidth, legend=false)
    return p
end

function _plot_vegetation!(p, vegetation; color=:forestgreen, linewidth::Real=3.0)
    if vegetation === nothing
        return p
    end

    for stem in vegetation.stems
        x, z = vegetation_path(stem)
        x0 = first(x)
        x = x0 .+ VEGETATION_DISPLAY_EXAGGERATION .* (x .- x0)

        if VEGETATION_ENFORCE_INEXTENSIBLE_DISPLAY && length(x) > 1
            ds = stem.config.height / max(length(x) - 1, 1)
            for j in 2:length(x)
                dx = x[j] - x[j - 1]
                dz = z[j] - z[j - 1]
                seg = hypot(dx, dz)
                if seg <= eps(Float64)
                    dx = 0.0
                    dz = 1.0
                    seg = 1.0
                end
                x[j] = x[j - 1] + ds * dx / seg
                z[j] = z[j - 1] + ds * dz / seg
            end
        end

        plot!(p, x, z; color=color, linewidth=linewidth, legend=false)
        scatter!(p, [first(x)], [first(z)]; color=color, markershape=:circle, markersize=5, legend=false)
        scatter!(p, [last(x)], [last(z)]; color=:goldenrod, markershape=:diamond, markersize=4, legend=false)
    end

    # Patches: outline up to the canopy height (flexible: mean deflected height of the representative stems).
    for (ip, patch) in enumerate(vegetation.patches)
        height = patch.stem_height
        theta = vegetation.patch_theta[ip]
        columns = [c for c in axes(theta, 2) if any(!iszero, @view theta[:, c])]
        if !isempty(columns)
            ds = patch.stem_height / size(theta, 1)
            height = sum(ds * sum(cos, @view theta[:, c]) for c in columns) / length(columns)
        end
        plot!(p, [patch.x_start, patch.x_start, patch.x_end, patch.x_end], [0.0, height, height, 0.0];
            color=color, linewidth=linewidth, linestyle=:dash, legend=false)
    end

    return p
end

function _physical_z_top(depth::Real, eta_amp::Real)
    pad = max(0.05 * depth, 0.05 * eta_amp, 1e-3)
    return depth + eta_amp + pad
end

function plot_state(state, config; title::String="", filename::String="", eta_prev=nothing, dt=nothing, vegetation=nothing)
    _ensure_batch_plot_backend!()

    plot_title = isempty(title) ? "" : title

    u = state.u
    w = state.w
    p = state.p
    rho = state.rho
    eta = state.eta
    eta_max = maximum(abs.(eta))
    z_top = _physical_z_top(config.depth, eta_max)

    x_edges = range(0.0, config.length, length=config.nx + 1)
    z_edges = range(0.0, config.depth, length=config.nz + 1)
    x_centers = 0.5 .* (x_edges[1:end-1] .+ x_edges[2:end])
    z_centers = 0.5 .* (z_edges[1:end-1] .+ z_edges[2:end])
    cell_len = min(config.length / config.nx, config.depth / config.nz)
    min_visible_len = VECTOR_MIN_VISIBLE_FACTOR * cell_len
    min_draw_mag = VECTOR_MIN_DRAW_FACTOR * cell_len

    u_c, w_c = _cell_center_velocity(u, w)
    i_idx, j_idx = _quiver_sampling(config)
    q_scale = _quiver_scale(u_c, w_c, config)
    # the arrows use the unblanked velocities (zero in the bed, not drawn); the heatmaps leave the bed blank
    u_plot, w_plot, p_plot, rho_plot = (_blank_solid_cells(f, config) for f in (u_c, w_c, p, rho))

    p1 = heatmap(x_centers, z_centers, u_plot; aspect_ratio=:auto, title="u-velocity (cell-centered)", xlabel="x (m)", ylabel="z (m, physical)", color=:viridis, clims=_finite_clims(u_plot), ylims=(0.0, z_top))
    p2 = heatmap(x_centers, z_centers, w_plot; aspect_ratio=:auto, title="w-velocity (cell-centered)", xlabel="x (m)", ylabel="z (m, physical)", color=:viridis, clims=_finite_clims(w_plot), ylims=(0.0, z_top))
    p3 = heatmap(x_centers, z_centers, p_plot; aspect_ratio=:auto, title="Internal Pressure", xlabel="x (m)", ylabel="z (m, physical)", color=:viridis, clims=_finite_clims(p_plot), ylims=(0.0, z_top))
    p4 = heatmap(x_centers, z_centers, rho_plot; aspect_ratio=:auto, title="Density", xlabel="x (m)", ylabel="z (m, physical)", color=:viridis, clims=_finite_clims(rho_plot), ylims=(0.0, z_top))

    xq, zq, uq, wq = _quiver_vectors(u_c, w_c, x_centers, z_centers, i_idx, j_idx, q_scale, min_visible_len)
    _plot_velocity_vectors!(p4, xq, zq, uq, wq; min_draw_mag=min_draw_mag)
    _plot_vegetation!(p1, vegetation)
    _plot_vegetation!(p4, vegetation)
    plot!(p1, x_centers, config.depth .+ eta; color=:white, linewidth=2, legend=false)
    plot!(p4, x_centers, config.depth .+ eta; color=:white, linewidth=2, legend=false)

    eta_ylim = max(0.1, 1.1 * eta_max)
    p5 = plot(x_centers, eta; title="Free Surface Elevation (eta)", xlabel="x (m)", ylabel="Elevation (m)", legend=false, ylim=(-eta_ylim, eta_ylim))

    div = div_x_u(u, config.length / config.nx) .+ div_z_w(w, config.depth / config.nz)
    div = _blank_solid_cells(div, config)
    p6 = heatmap(x_centers, z_centers, div; aspect_ratio=:auto, title="Divergence (du/dx + dw/dz)", xlabel="x (m)", ylabel="z (m)", color=:balance, clims=_finite_clims(div; center=true))

    if eta_prev !== nothing && dt !== nothing && dt > 0.0
        eta_t = (eta .- eta_prev) ./ dt
        mismatch = (@view w[end, :]) .- eta_t
        max_abs = max(PLOT_RANGE_MIN_SPAN, maximum(abs, mismatch))
        p7 = plot(x_centers, mismatch; title=@sprintf("Free-Surface Kinematic Closure (max=%.2e)", maximum(abs, mismatch)), xlabel="x (m)", ylabel="m/s", ylim=(-1.1 * max_abs, 1.1 * max_abs), legend=false)
        hline!(p7, [0.0]; linestyle=:dash, color=:red)
    else
        p7 = plot(x_centers, zeros(length(x_centers)); title="Free-Surface Kinematic Closure", xlabel="x (m)", ylabel="m/s", ylim=(-1.0, 1.0), legend=false)
    end

    fig = plot(p1, p2, p3, p4, p5, p6, p7; layout=(7, 1), size=(1200, 2000), plot_title=plot_title)
    if isempty(filename)
        display(fig)
    else
        _save_plot!(fig, filename)
    end
end

function create_animation(history, config; filename::String="animation.gif", save_interval::Int=1, eta_prev_history=nothing, frame_times=nothing, vegetation_history=nothing)
    _ensure_batch_plot_backend!()

    if isempty(history)
        @warn "History is empty, cannot create animation."
        return
    end

    x_edges = range(0.0, config.length, length=config.nx + 1)
    z_edges = range(0.0, config.depth, length=config.nz + 1)
    x_centers = 0.5 .* (x_edges[1:end-1] .+ x_edges[2:end])
    z_centers = 0.5 .* (z_edges[1:end-1] .+ z_edges[2:end])
    cell_len = min(config.length / config.nx, config.depth / config.nz)
    min_visible_len = VECTOR_MIN_VISIBLE_FACTOR * cell_len
    min_draw_mag = VECTOR_MIN_DRAW_FACTOR * cell_len

    vmin = minimum([minimum(st.rho) for st in history])
    vmax = maximum([maximum(st.rho) for st in history])
    if !isfinite(vmin) || !isfinite(vmax)
        vmin, vmax = (0.0, 1.0)
    elseif (vmax - vmin) < PLOT_RANGE_MIN_SPAN
        mid = 0.5 * (vmax + vmin)
        half = max(abs(mid) * 0.05, PLOT_RANGE_MIN_SPAN / 2.0)
        vmin = mid - half
        vmax = mid + half
    end

    i_idx, j_idx = _quiver_sampling(config)
    eta_ref = max(PLOT_RANGE_MIN_SPAN, maximum(maximum(abs.(st.eta)) for st in history))
    eta_ylim = 1.1 * eta_ref
    z_top_fixed = _physical_z_top(config.depth, eta_ref)

    speed95_ref = 0.0
    for st in history
        u_c, w_c = _cell_center_velocity(st.u, st.w)
        speed95_ref = max(speed95_ref, _quiver_scale(u_c, w_c, config) * max(QUIVER_TARGET_LENGTH_FACTOR * config.depth, 1e-12))
    end
    speed95_ref = max(speed95_ref, 1e-12)
    target_len = max(QUIVER_TARGET_LENGTH_FACTOR * config.depth, 1e-12)
    q_scale = speed95_ref / target_len

    frame_dt = config.dt * save_interval

    anim = @animate for i in eachindex(history)
        st = history[i]
        vegetation = vegetation_history === nothing ? nothing : vegetation_history[i]
        u_c, w_c = _cell_center_velocity(st.u, st.w)
        t = frame_times !== nothing && i <= length(frame_times) ? frame_times[i] : (i - 1) * frame_dt
        density_title = @sprintf("Density at t=%.2fs", t)

        p_top = heatmap(x_centers, z_centers, _blank_solid_cells(st.rho, config); aspect_ratio=:auto, color=:viridis, clims=(vmin, vmax), title=density_title, xlabel="x (m)", ylabel="z (m, physical)", ylims=(0.0, z_top_fixed))
        xq, zq, uq, wq = _quiver_vectors(u_c, w_c, x_centers, z_centers, i_idx, j_idx, q_scale, min_visible_len)
        _plot_velocity_vectors!(p_top, xq, zq, uq, wq; min_draw_mag=min_draw_mag)
        _plot_vegetation!(p_top, vegetation)
        plot!(p_top, x_centers, config.depth .+ st.eta; color=:white, linewidth=2, legend=false)

        if eta_prev_history !== nothing && i <= length(eta_prev_history) && config.dt > 0.0
            eta_t = (history[i].eta .- eta_prev_history[i]) ./ config.dt
            mismatch = (@view history[i].w[end, :]) .- eta_t
        elseif i == 1
            mismatch = zeros(length(st.eta))
        else
            eta_t = (history[i].eta .- history[i - 1].eta) ./ frame_dt
            mismatch = (@view history[i].w[end, :]) .- eta_t
        end

        max_abs = max(PLOT_RANGE_MIN_SPAN, maximum(abs, mismatch))
        p_bot = plot(x_centers, mismatch; xlabel="x (m)", ylabel="m/s", ylim=(-1.1 * max_abs, 1.1 * max_abs), legend=false, title=@sprintf("w_top - d eta/dt (max=%.2e)", max_abs))
        hline!(p_bot, [0.0]; linestyle=:dash, color=:red)

        p_eta = plot(x_centers, st.eta; xlabel="x (m)", ylabel="eta (m)", ylim=(-eta_ylim, eta_ylim), legend=false, title="Free-surface elevation")
        hline!(p_eta, [0.0]; linestyle=:dash, color=:red)

        plot(p_top, p_eta, p_bot; layout=(3, 1), size=(1100, 1100))
    end

    withenv("GKSwstype" => get(ENV, "GKSwstype", "100"), "GKS_WSTYPE" => get(ENV, "GKS_WSTYPE", "100")) do
        gif(anim, filename; fps=10)
    end
    println("Animation saved to $(filename)")
end

function plot_pressure_buoyancy_diagnostics(state, config, p_hydro::AbstractMatrix{<:Real}, p_corr::AbstractMatrix{<:Real}, buoy_w::AbstractMatrix{<:Real}; filename::String="")
    _ensure_batch_plot_backend!()

    x_edges = range(0.0, config.length, length=config.nx + 1)
    z_edges = range(0.0, config.depth, length=config.nz + 1)
    x_centers = 0.5 .* (x_edges[1:end-1] .+ x_edges[2:end])
    z_centers = 0.5 .* (z_edges[1:end-1] .+ z_edges[2:end])

    p_total = state.p

    buoy_c = zeros(eltype(state.p), size(state.p))
    if size(buoy_c, 1) > 2
        @views buoy_c[2:end-1, :] .= 0.5 .* (buoy_w[1:end-1, :] .+ buoy_w[2:end, :])
    end
    @views buoy_c[1, :] .= buoy_w[1, :]
    @views buoy_c[end, :] .= buoy_w[end, :]

    pc_lim = max(PLOT_RANGE_MIN_SPAN, maximum(abs, p_corr))
    b_lim = max(PLOT_RANGE_MIN_SPAN, maximum(abs, buoy_c))

    p_hydro, p_corr, p_total, buoy_c = (_blank_solid_cells(f, config) for f in (p_hydro, p_corr, p_total, buoy_c))
    p1 = heatmap(x_centers, z_centers, p_hydro; aspect_ratio=:auto, title="Hydrostatic Pressure", xlabel="x (m)", ylabel="z (m)", color=:viridis, clims=_finite_clims(p_hydro))
    p2 = heatmap(x_centers, z_centers, p_corr; aspect_ratio=:auto, title="Non-hydrostatic Pressure Correction", xlabel="x (m)", ylabel="z (m)", color=:balance, clims=_finite_clims(p_corr; center=true))
    p3 = heatmap(x_centers, z_centers, p_total; aspect_ratio=:auto, title="Total Internal Pressure", xlabel="x (m)", ylabel="z (m)", color=:viridis, clims=_finite_clims(p_total))
    p4 = heatmap(x_centers, z_centers, buoy_c; aspect_ratio=:auto, title="Buoyancy Acceleration", xlabel="x (m)", ylabel="z (m)", color=:balance, clims=_finite_clims(buoy_c; center=true))

    fig = plot(p1, p2, p3, p4; layout=(4, 1), size=(1200, 1400), plot_title=@sprintf("Pressure/Buoyancy Diagnostics | p_corr max=%.2e, buoyancy range=[%.2e, %.2e]", pc_lim, minimum(buoy_c), maximum(buoy_c)))

    if isempty(filename)
        display(fig)
    else
        _save_plot!(fig, filename)
    end
end
