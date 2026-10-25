# Wind-driven double gyre (issue #4): runs the case in cases.jl in the hydrostatic and the non-hydrostatic mode and
# writes animations and figures.
#
# The script writes to output_barotropic/wind_shear_figures/:
# - wind_shear_<mode>.gif             density with streamlines (psi contours), one frame per ANIMATION_INTERVAL_S
# - wind_shear_<mode>_snapshots.png   density and streamlines at SNAPSHOT_TIMES_S
# - wind_shear_<mode>.npz             gyre-strength series, time-averaged u and final density (restyle without rerun)
# and, once both modes have run:
# - wind_shear_gyres.png              upper- and lower-cell strength against time, both modes
# - wind_shear_profile.png            time-averaged mid-basin u profile, both modes
# - wind_shear_mean_streamlines.png   time-averaged streamfunction over the basin, both modes
# - wind_shear_profiles.gif           u(z) at PROFILE_STATIONS_M against time, both modes
# - wind_shear_profiles_mean.png      time-averaged u(z) at PROFILE_STATIONS_M, both modes
# - wind_shear_hovmoller.png          mid-basin u(z) against time (time-depth diagram), both modes
# <mode> is `hydro` or `nonhydro`. The figures in figures/ are copies of these files (the GIFs reduced in size).
#
# Usage (any working directory):
#   julia --project=<path to the repository> experiments/wind_shear/plot_wind_shear.jl [hydro|nonhydro|both|figures]
# `figures` only redraws the comparison figures from the .npz files. Run time: about 8 min (hydrostatic) and
# 11 min (non-hydrostatic) for 90,000 steps on 400 x 160 cells.

# WHY: batch run without a display; same GR workstation choice as src/visualization.jl.
get!(ENV, "GKSwstype", "100")

using SweTransport2DV
using Plots
using NPZ
using Printf
using Statistics

include(joinpath(@__DIR__, "cases.jl"))

const T_END_S = 1800.0                     # s, simulated time
const AVERAGE_START_S = 900.0              # s, start of the averaging window of the mean profile and streamlines
const ANIMATION_INTERVAL_S = 10.0          # s, simulated time between animation frames
const SAMPLE_INTERVAL_S = 2.0              # s, sampling interval of the gyre strengths and the time average
const SNAPSHOT_TIMES_S = (30.0, 120.0, 300.0, 600.0, 1200.0, 1800.0)   # s
const PROGRESS_INTERVAL_S = 100.0          # s, simulated time between progress lines
const TVD2_CFL_LIMIT = 0.5                 # -, monotonicity limit of tvd2 transport (CFL_x + CFL_z)
const PSI_LEVELS_UPPER = -0.1:0.005:-0.005 # m^2/s, streamline levels of the clockwise (upper) cell
const PSI_LEVELS_LOWER = 0.0025:0.0025:0.05   # m^2/s, streamline levels of the counter-clockwise (lower) cell
const ANIMATION_FPS = 12                   # frames/s of the GIF
const FRAME_SIZE_PX = (900, 300)           # px, one animation frame
const SNAPSHOT_SIZE_PX = (1400, 640)       # px, 3 x 2 snapshot panel
const SERIES_SIZE_PX = (800, 400)          # px, gyre-strength figure
const PROFILE_SIZE_PX = (520, 480)         # px, mean profile
const MEAN_STREAMLINES_SIZE_PX = (1000, 620)   # px, two mean-streamline panels
const PROFILE_STATIONS_M = (2.5, 5.0, 10.0, 15.0, 17.5)   # m, x positions of the u(z) profiles (on u-faces)
const PROFILE_U_LIMITS = (-0.04, 0.08)     # m/s, fixed u axis of the profile GIF and the mean profiles
const PROFILE_FRAME_INTERVAL_S = 10.0      # s, simulated time between profile-GIF frames (multiple of SAMPLE_INTERVAL_S)
const PROFILES_SIZE_PX = (1100, 380)       # px, one profile-GIF frame / mean-profile figure (1 x 5 panels)
const HOVMOLLER_SIZE_PX = (1000, 620)      # px, two time-depth panels
const HOVMOLLER_COLORMAP = :balance        # diverging: blue upwind, red downwind, white at rest
const FIGURE_MARGIN_MM = 4Plots.mm         # margin so that labels are not clipped
const COLORMAP = :viridis                  # same colormap as plot_state in src/visualization.jl
const DENSITY_LABEL = "Density (kg/m³)"
const MODES = Dict("hydro" => true, "nonhydro" => false)   # mode name => use_hydrostatic
const MODE_TITLE = Dict("hydro" => "hydrostatic", "nonhydro" => "non-hydrostatic")
const MODE_COLOR = Dict("hydro" => :steelblue, "nonhydro" => :crimson)   # line colours of the profile figures
const PROFILE_U_TICKS = -0.04:0.04:0.08     # m/s, u ticks of the station profiles

# WHY: anchored at this file, not the cwd, so outputs land in the project's output_barotropic/.
const OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "output_barotropic", "wind_shear_figures")

"""
    density_streamlines(rho, u, cfg; title="", kwargs...)

xz heatmap of the cell-centred density `rho` (kg/m³, nz x nx) with the streamlines of the face velocities `u`
(m/s): white contours of the clockwise cell (psi < 0), orange contours of the counter-clockwise cell (psi > 0).
"""
function density_streamlines(rho::AbstractMatrix{<:Real}, u::AbstractMatrix{<:Real}, cfg::ModelConfig; title::AbstractString="",
    colorbar_title::AbstractString=DENSITY_LABEL, kwargs...)
    dx = cfg.length / cfg.nx    # m
    dz = cfg.depth / cfg.nz     # m
    p = heatmap(
        range(0.5 * dx, cfg.length - 0.5 * dx, length=cfg.nx),   # m, cell centres
        range(0.5 * dz, cfg.depth - 0.5 * dz, length=cfg.nz),    # m
        rho;
        color=COLORMAP,
        clims=(WS_RHO_TOP, WS_RHO_BOTTOM),
        xlabel="x (m)",
        ylabel="z (m)",
        colorbar_title=colorbar_title,
        title=title,
        xlims=(0.0, cfg.length),
        ylims=(0.0, cfg.depth),
        aspect_ratio=:equal,
        kwargs...,
    )
    add_streamlines!(p, streamfunction(u, dz), cfg)
    return p
end

"""
    add_streamlines!(p, psi, cfg)

Contours of the streamfunction `psi` (m^2/s, (nz+1) x (nx+1), on the u-faces and the w-levels) on the plot `p`.
"""
function add_streamlines!(p, psi::AbstractMatrix{<:Real}, cfg::ModelConfig)
    x_faces = range(0.0, cfg.length, length=cfg.nx + 1)    # m
    z_faces = range(0.0, cfg.depth, length=cfg.nz + 1)     # m
    # WHY the guards: GR warns about contour levels outside the data range.
    lo, hi = extrema(psi)
    lev_up = [l for l in PSI_LEVELS_UPPER if l > lo]
    lev_lo = [l for l in PSI_LEVELS_LOWER if l < hi]
    # WHY the linear map of psi into the density colour range: GR shares the colour limits of a subplot between its
    # series and does not draw contour levels outside the heatmap's clims (kg/m^3). The map leaves the lines unchanged;
    # their colours are explicit.
    psi_lo, psi_hi = first(PSI_LEVELS_UPPER), last(PSI_LEVELS_LOWER)
    to_density(v) = WS_RHO_TOP + (v - psi_lo) / (psi_hi - psi_lo) * (WS_RHO_BOTTOM - WS_RHO_TOP)
    mapped = to_density.(psi)
    isempty(lev_up) || contour!(p, x_faces, z_faces, mapped; levels=to_density.(lev_up), color=:white, linewidth=1, colorbar_entry=false)
    isempty(lev_lo) || contour!(p, x_faces, z_faces, mapped; levels=to_density.(lev_lo), color=:orange, linewidth=1, colorbar_entry=false)
    return p
end

"""
    run_mode(mode) -> String

Runs the case for `mode` ("hydro" or "nonhydro"), writes its GIF, snapshot panel and .npz; returns the .npz path.
"""
function run_mode(mode::AbstractString)
    cfg = wind_shear_config(; use_hydrostatic=MODES[mode], t_end=T_END_S, write_plots=false, write_animation=false,
        write_pressure_buoyancy_diagnostics=false)
    model = Model(cfg)
    initialize_wind_shear!(model)
    nz_interface = round(Int, WS_INTERFACE_HEIGHT_FRACTION * cfg.nz)
    @printf("Wind shear (%s) | %d x %d cells, dt=%.3f s, t_end=%.0f s, tau=%.2f Pa after %.0f s\n",
        MODE_TITLE[mode], cfg.nx, cfg.nz, cfg.dt, T_END_S, WS_WIND_STRESS_X, WS_RAMP_S)

    frame_stride = round(Int, ANIMATION_INTERVAL_S / cfg.dt)
    sample_stride = round(Int, SAMPLE_INTERVAL_S / cfg.dt)
    progress_stride = round(Int, PROGRESS_INTERVAL_S / cfg.dt)
    average_start_step = round(Int, AVERAGE_START_S / cfg.dt)
    snapshot_steps = Dict(round(Int, t / cfg.dt) => t for t in SNAPSHOT_TIMES_S)

    times = [0.0]; psi_upper = [0.0]; psi_lower = [0.0]   # s, m^2/s, m^2/s
    station_faces = [round(Int, x / model.dx) + 1 for x in PROFILE_STATIONS_M]   # u-face indices of the stations
    station_profiles = [model.state.u[:, station_faces]]  # m/s, nz x n_stations per sample
    mid_profiles = [mid_basin_profile(model.state.u)]     # m/s, nz per sample
    u_mean = zeros(size(model.state.u))                    # m/s, time average over [AVERAGE_START_S, T_END_S]
    n_mean = 0
    snapshots = Dict{Float64,Tuple{Matrix{Float64},Matrix{Float64}}}()
    anim = Animation()
    frame(anim, density_streamlines(model.state.rho, model.state.u, cfg; title=@sprintf("%s, t = 0 s", MODE_TITLE[mode]), size=FRAME_SIZE_PX))
    max_cfl = 0.0   # -, running max of CFL_x + CFL_z

    t0 = time()
    for n in 1:cfg.nt
        # WHY t at the end of the step: the stress acts over the step (n-1)dt..n dt; the difference is one step of the ramp.
        cfg.wind_stress_x = wind_stress_at(n * cfg.dt)
        step_forward!(model)
        t = n * cfg.dt
        # PERF: cfl_numbers scans u and w once, small against one step.
        max_cfl = max(max_cfl, cfl_numbers(model.state.u, model.state.w, cfg.dt, model.dx, model.dz)[3])
        if n % sample_stride == 0
            up, lo = gyre_strengths(model.state.u, model.dz, nz_interface)
            push!(times, t); push!(psi_upper, up); push!(psi_lower, lo)
            push!(station_profiles, model.state.u[:, station_faces])
            push!(mid_profiles, mid_basin_profile(model.state.u))
            if n > average_start_step
                u_mean .+= model.state.u
                n_mean += 1
            end
        end
        if n % frame_stride == 0
            frame(anim, density_streamlines(model.state.rho, model.state.u, cfg; title=@sprintf("%s, t = %.0f s", MODE_TITLE[mode], t), size=FRAME_SIZE_PX))
        end
        if haskey(snapshot_steps, n)
            snapshots[snapshot_steps[n]] = (copy(model.state.rho), copy(model.state.u))
        end
        if n % progress_stride == 0
            @printf("t=%6.0f s | psi upper %+.4f lower %+.4f m^2/s | max|u|=%.3f m/s | max|w|=%.4f m/s | max CFL so far=%.3f | wall=%.0f s\n",
                t, psi_upper[end], psi_lower[end], maximum(abs, model.state.u), maximum(abs, model.state.w), max_cfl, time() - t0)
        end
    end
    u_mean ./= n_mean
    @printf("CASE=wind_shear_%s FULL_WALL_SECONDS=%.2f | max CFL=%.3f (tvd2 limit %.1f) | density range [%.4f, %.4f] kg/m^3 | sum(eta) dx = %.2e m^2\n",
        mode, time() - t0, max_cfl, TVD2_CFL_LIMIT, minimum(model.state.rho), maximum(model.state.rho), sum(model.state.eta) * model.dx)
    if max_cfl >= TVD2_CFL_LIMIT
        @warn "CFL number reached the tvd2 monotonicity limit; reduce dt" max_cfl
    end

    stem = joinpath(OUTPUT_DIR, "wind_shear_" * mode)
    # Order: raw data first, so that a failing figure cannot cost the run.
    npzwrite(stem * ".npz", Dict("t_s" => times, "psi_upper_m2_s" => psi_upper, "psi_lower_m2_s" => psi_lower,
        "u_mean_m_s" => u_mean, "rho_final" => copy(model.state.rho), "dz_m" => [model.dz],
        "average_start_s" => [AVERAGE_START_S],
        # nz x n_stations x n_samples and nz x n_samples, sampled at t_s
        "u_stations_m_s" => cat(station_profiles...; dims=3), "station_x_m" => collect((station_faces .- 1) .* model.dx),
        "u_mid_m_s" => reduce(hcat, mid_profiles)))
    gif(anim, stem * ".gif"; fps=ANIMATION_FPS)
    panels = [density_streamlines(snapshots[t]..., cfg; title=@sprintf("t = %.0f s", t), colorbar=false) for t in SNAPSHOT_TIMES_S]
    savefig(plot(panels...; layout=(3, 2), size=SNAPSHOT_SIZE_PX, left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM,
        plot_title=MODE_TITLE[mode]), stem * "_snapshots.png")
    println("Saved: $(abspath(stem)).{npz,gif}, _snapshots.png")
    return stem * ".npz"
end

"""
    comparison_figures()

Gyre strengths against time, mean mid-basin u profile and mean streamlines of both modes from their .npz files.
"""
function comparison_figures()
    files = Dict(m => joinpath(OUTPUT_DIR, "wind_shear_$(m).npz") for m in keys(MODES))
    if !all(isfile, values(files))
        println("Comparison figures skipped: run both modes first.")
        return nothing
    end
    data = Dict(m => npzread(f) for (m, f) in files)
    cfg = wind_shear_config(; use_hydrostatic=true, t_end=T_END_S)
    style = Dict("hydro" => :solid, "nonhydro" => :dash)
    order = ("hydro", "nonhydro")

    p = plot(xlabel="t (s)", ylabel="ψ (m²/s)", legend=:right, size=SERIES_SIZE_PX, left_margin=FIGURE_MARGIN_MM,
        bottom_margin=FIGURE_MARGIN_MM, title="Strength of the two cells (mean over the middle half of the basin)")
    for m in order
        d = data[m]
        plot!(p, d["t_s"], d["psi_upper_m2_s"]; color=:steelblue, linestyle=style[m], linewidth=2, label="upper cell, $(MODE_TITLE[m])")
        plot!(p, d["t_s"], d["psi_lower_m2_s"]; color=:darkorange, linestyle=style[m], linewidth=2, label="lower cell, $(MODE_TITLE[m])")
    end
    hline!(p, [0.0]; color=:gray, label="")
    vspan!(p, [0.0, WS_RAMP_S]; color=:gray, alpha=0.15, label="wind ramp")
    savefig(p, joinpath(OUTPUT_DIR, "wind_shear_gyres.png"))

    nxp1 = cfg.nx + 1
    mid = (nxp1 ÷ 4 + 1):(3 * nxp1 ÷ 4)
    z = range(0.5 * cfg.depth / cfg.nz, cfg.depth - 0.5 * cfg.depth / cfg.nz, length=cfg.nz)   # m
    p = plot(xlabel="mean u (m/s)", ylabel="z (m)", legend=:bottomright, size=PROFILE_SIZE_PX, left_margin=FIGURE_MARGIN_MM,
        bottom_margin=FIGURE_MARGIN_MM, title=@sprintf("Mean u, middle half, t = %.0f–%.0f s", AVERAGE_START_S, T_END_S))
    for m in order
        u = data[m]["u_mean_m_s"]
        plot!(p, vec(mean(u[:, mid]; dims=2)), z; linestyle=style[m], color=MODE_COLOR[m], linewidth=2, label=MODE_TITLE[m])
    end
    vline!(p, [0.0]; color=:gray, label="")
    hline!(p, [WS_INTERFACE_HEIGHT_FRACTION * cfg.depth]; color=:gray, linestyle=:dot, label="initial interface")
    savefig(p, joinpath(OUTPUT_DIR, "wind_shear_profile.png"))

    panels = [density_streamlines(data[m]["rho_final"], data[m]["u_mean_m_s"], cfg;
        title=@sprintf("%s: mean streamlines t = %.0f–%.0f s, final density (kg/m³)", MODE_TITLE[m], AVERAGE_START_S, T_END_S),
        colorbar_title="") for m in order]
    # WHY no colorbar title here: at this panel height GR draws it over the tick labels; the unit is in the panel title.
    savefig(plot(panels...; layout=(2, 1), size=MEAN_STREAMLINES_SIZE_PX, left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM),
        joinpath(OUTPUT_DIR, "wind_shear_mean_streamlines.png"))
    println("Saved: wind_shear_gyres.png, wind_shear_profile.png, wind_shear_mean_streamlines.png in $(abspath(OUTPUT_DIR))")

    if !all(haskey(data[m], "u_stations_m_s") for m in order)
        println("Profile figures skipped: the .npz files predate the profile sampling; rerun both modes.")
        return nothing
    end
    profile_figures(data, z, style, order)
    return nothing
end

"""
    station_panels(profiles, x_stations, z, style, order; title_prefix="")

1 x n_stations panels of u(z) (m/s): `profiles[m]` is the nz x n_stations matrix of mode `m`.
"""
function station_panels(profiles::Dict, x_stations::AbstractVector, z, style::Dict, order; title_prefix::AbstractString="")
    panels = map(enumerate(x_stations)) do (j, x)
        p = plot(xlabel="u (m/s)", ylabel=j == 1 ? "z (m)" : "", xlims=PROFILE_U_LIMITS, xticks=PROFILE_U_TICKS, ylims=(0.0, WS_DEPTH),
            title=@sprintf("%sx = %.1f m", title_prefix, x), titlefontsize=10, legend=j == length(x_stations) ? :bottomright : false)
        vline!(p, [0.0]; color=:gray, label="")
        hline!(p, [WS_INTERFACE_HEIGHT_FRACTION * WS_DEPTH]; color=:gray, linestyle=:dot, label="")
        for m in order
            plot!(p, profiles[m][:, j], z; linestyle=style[m], color=MODE_COLOR[m], linewidth=2, label=MODE_TITLE[m])
        end
        p
    end
    return panels
end

"""
    profile_figures(data, z, style, order)

Profile GIF and mean profiles at PROFILE_STATIONS_M and the mid-basin time-depth diagram of both modes.
"""
function profile_figures(data::Dict, z, style::Dict, order)
    t = data[first(order)]["t_s"]                     # s, sample times (same in both modes)
    x_stations = data[first(order)]["station_x_m"]    # m
    margins = (left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM)

    stride = round(Int, PROFILE_FRAME_INTERVAL_S / SAMPLE_INTERVAL_S)
    anim = Animation()
    for s in 1:stride:length(t)
        profiles = Dict(m => data[m]["u_stations_m_s"][:, :, s] for m in order)
        panels = station_panels(profiles, x_stations, z, style, order)
        frame(anim, plot(panels...; layout=(1, length(panels)), size=PROFILES_SIZE_PX, margins...,
            plot_title=@sprintf("u profiles, t = %.0f s (wind 0.2 Pa from the left, ramp %.0f s)", t[s], WS_RAMP_S),
            plot_titlefontsize=11))
    end
    gif(anim, joinpath(OUTPUT_DIR, "wind_shear_profiles.gif"); fps=ANIMATION_FPS)

    window = t .> AVERAGE_START_S
    means = Dict(m => dropdims(mean(data[m]["u_stations_m_s"][:, :, window]; dims=3); dims=3) for m in order)
    panels = station_panels(means, x_stations, z, style, order)
    savefig(plot(panels...; layout=(1, length(panels)), size=PROFILES_SIZE_PX, margins...,
        plot_title=@sprintf("Mean u profiles, t = %.0f–%.0f s", AVERAGE_START_S, T_END_S), plot_titlefontsize=11),
        joinpath(OUTPUT_DIR, "wind_shear_profiles_mean.png"))

    u_max = maximum(maximum(abs, data[m]["u_mid_m_s"]) for m in order)   # m/s, symmetric colour range
    panels = [heatmap(t, z, data[m]["u_mid_m_s"]; color=HOVMOLLER_COLORMAP, clims=(-u_max, u_max), xlabel="t (s)",
        ylabel="z (m)", colorbar_title="u (m/s)", title="$(MODE_TITLE[m]): mean u over the middle half of the basin")
        for m in order]
    for p in panels
        hline!(p, [WS_INTERFACE_HEIGHT_FRACTION * WS_DEPTH]; color=:black, linestyle=:dot, label="")
        vline!(p, [WS_RAMP_S]; color=:black, linestyle=:dash, label="")
    end
    savefig(plot(panels...; layout=(2, 1), size=HOVMOLLER_SIZE_PX, margins..., right_margin=FIGURE_MARGIN_MM),
        joinpath(OUTPUT_DIR, "wind_shear_hovmoller.png"))
    println("Saved: wind_shear_profiles.gif, wind_shear_profiles_mean.png, wind_shear_hovmoller.png in $(abspath(OUTPUT_DIR))")
    return nothing
end

mkpath(OUTPUT_DIR)
what = isempty(ARGS) ? "both" : ARGS[1]
if what in ("hydro", "nonhydro")
    run_mode(what)
elseif what == "both"
    run_mode("hydro")
    run_mode("nonhydro")
elseif what != "figures"
    error("unknown argument $(what); use hydro, nonhydro, both or figures")
end
comparison_figures()
