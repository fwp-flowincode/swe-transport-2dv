# Wetting and drying (issue #31): figures and animations for the README.
#
# Usage (any working directory):
#   julia --project=<repository> experiments/wetting_drying/plot_wetting_drying.jl [mode ...]
# Modes: lake, ritter, runup, sloshing, mms (manufactured solution of the momentum advection), schemes (upwind1 against
# tvd2), dispersion (non-hydrostatic standing waves), solitary (solitary wave runup, Synolakis 1987); default: all.
# Each takes a few minutes or less.
# Output: output_barotropic/wetting_drying_figures/; the PNG and GIF files are copied to figures/.

# WHY: batch run without a display; same GR workstation choice as src/visualization.jl.
get!(ENV, "GKSwstype", "100")

using SweTransport2DV
using Plots
using Printf

include(joinpath(@__DIR__, "cases.jl"))

const MODES = isempty(ARGS) ? ["lake", "ritter", "runup", "sloshing", "mms", "schemes", "dispersion", "solitary"] : ARGS
const OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "output_barotropic", "wetting_drying_figures")
const FIGURE_DIR = joinpath(@__DIR__, "figures")

const PLOT_MIN_THICKNESS = 1e-4     # m, cells thinner than this are drawn empty in the sections
const BED_COLOR = :gray55
const ANIMATION_FPS = 12            # frames/s
const SECTION_SIZE_PX = (900, 300)  # px, one x-z section
const LINE_SIZE_PX = (820, 420)     # px
const FIGURE_MARGIN_MM = 4Plots.mm

# Lake at rest
const LAKE_STEPS = 500              # -, 25 s
# Ritter
const RITTER_PROFILE_TIMES = (1.25, 2.5, 5.0)            # s
const RITTER_PROFILE_NX = 400                            # -, dx = 0.25 m
const RITTER_NX_1D = (100, 200, 400, 800, 1600)          # -, convergence study (dx = 1 ... 0.0625 m)
const RITTER_NX_2DV = (100, 200, 400, 800)               # -
const RITTER_NZ_2DV = 12                                 # -, layers of 0.1 m
const RITTER_T_END = 5.0                                 # s, comparison time
const RITTER_ANIMATION_END = 8.0                         # s, before the front reaches the right wall
const RITTER_FRAME_INTERVAL = 0.2                        # s
# Carrier-Greenspan
const CG_RUNS = ((label="1D, dx = 0.1 m", nx=240, nz=1, dt=0.02), (label="1D, dx = 0.05 m", nx=480, nz=1, dt=0.01),
    (label="2DV, dx = 0.05 m, nz = 15", nx=480, nz=15, dt=0.01))
const CG_PHASES = 8                                      # -, profiles per period
const CG_FRAME_INTERVAL = 0.2                            # s
const CG_XLIMS = (10.0, CG_LENGTH + CG_ONSHORE)          # m, zoom on the beach
# Sloshing
const SLOSH_T_END = 40.0                                 # s
const SLOSH_NZ_2DV = 10                                  # -, layers of 0.15 m
const SLOSH_FRAME_INTERVAL = 0.4                         # s
const SLOSH_SNAPSHOT_TIMES = (0.0, 4.0, 8.0, 12.0, 20.0, 40.0)   # s
const SLOSH_CLIMS = (WD_RHO0, SLOSH_RHO_DENSE + SLOSH_RHO_BOTTOM)  # kg/m^3
const SLOSH_WET_DEPTH = 1e-3                             # m, the shoreline is the last column deeper than this (drained beaches
                                                         # keep a film of about dry_depth_threshold)

"""
    section(model, field; clims, title, color, colorbar_title, xlims) -> Plot

x-z section of a cell field (nz x nx) on the wet part of the cut cells (cells thinner than PLOT_MIN_THICKNESS are empty),
with the bed (grey) and the water surface (black line). The part of a cut cell above the surface is masked (white), so
the colours end at the water surface.
"""
function section(model::Model, field::AbstractMatrix; clims, title="", color=:viridis, colorbar_title="",
        xlims=(0.0, model.config.length), ylims=(0.0, model.config.depth), kwargs...)
    cfg = model.config
    x = wd_x_centres(cfg)
    z = [(k - 0.5) * model.dz for k in 1:cfg.nz]
    bed = model.wetdry.bed
    zeta = wd_water_level(model)
    h = zeros(cfg.nz)
    F = fill(NaN, cfg.nz, cfg.nx)
    for i in 1:cfg.nx
        layer_thicknesses!(h, zeta[i], bed[i], model.dz)
        for k in 1:cfg.nz
            h[k] > PLOT_MIN_THICKNESS && (F[k, i] = field[k, i])
        end
    end
    surface = [zeta[i] - bed[i] > PLOT_MIN_THICKNESS ? zeta[i] : NaN for i in 1:cfg.nx]
    p = heatmap(x, z, F; clims=clims, color=color, title=title, xlabel="x (m)", ylabel="z (m)", colorbar_title=colorbar_title,
        xlims=xlims, ylims=ylims, kwargs...)
    # WHY the mask: the heatmap colours whole cells; above the surface the cut top cells hold no water
    plot!(p, x, max.(zeta, bed); fillrange=cfg.depth, color=:white, linewidth=0, label="")
    plot!(p, x, bed; fillrange=0.0, color=BED_COLOR, linewidth=0, label="")
    plot!(p, x, surface; color=:black, linewidth=1.5, label="")
    return p
end

"Cell-centre average of the face velocity u (m/s, nz x nx)."
cell_velocity(model::Model) = 0.5 .* (model.state.u[:, 1:(end - 1)] .+ model.state.u[:, 2:end])

function save_figure(fig, name)
    path = joinpath(OUTPUT_DIR, name)
    savefig(fig, path)
    cp(path, joinpath(FIGURE_DIR, name); force=true)
    println("Saved: $(abspath(path))")
end

function save_gif(anim, name)
    path = joinpath(OUTPUT_DIR, name)
    gif(anim, path; fps=ANIMATION_FPS)
    cp(path, joinpath(FIGURE_DIR, name); force=true)
    println("Saved: $(abspath(path))")
end

# ---- Lake at rest --------------------------------------------------------------------------------------------------
function plot_lake()
    println("\n=== Lake at rest: $(LAKE_STEPS) steps ===")
    panels = []
    for (label, nz, stratified) in (("1D", 1, false), ("2DV", 10, false), ("2DV stratified", 10, true))
        model = lake_model(nz; stratified=stratified)
        wet = water_depth(model) .> 0.0
        max_u, max_dlevel = 0.0, 0.0   # m/s, m
        for _ in 1:LAKE_STEPS
            step_forward!(model)
            max_u = max(max_u, maximum(abs, model.state.u))
            max_dlevel = max(max_dlevel, maximum(abs, wd_water_level(model)[wet] .- LAKE_LEVEL))
        end
        dry_ok = all(water_depth(model)[.!wet] .== 0.0)
        @printf("%-15s | max|u| = %.3e m/s | max|zeta - level| = %.3e m | dry cells stay dry: %s\n", label, max_u, max_dlevel, dry_ok)
        if stratified
            push!(panels, section(model, model.state.rho; clims=(WD_RHO0, LAKE_RHO_LOWER), colorbar_title="ρ (kg/m³)",
                title=@sprintf("2DV stratified lake at rest after %d steps: max|u| = %.1g m/s", LAKE_STEPS, max_u),
                size=SECTION_SIZE_PX, left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM))
        end
    end
    save_figure(panels[1], "lake_at_rest.png")
end

# ---- Ritter dam break ------------------------------------------------------------------------------------------------
function run_to!(model::Model, t_end::Real; on_step=nothing)
    nsteps = round(Int, t_end / model.config.dt)
    for n in 1:nsteps
        step_forward!(model)
        on_step === nothing || on_step(model, n * model.config.dt)
    end
    return model
end

function plot_ritter()
    println("\n=== Ritter dam break on a dry bed ===")
    # profiles, 1D and 2DV
    fig_h = plot(xlabel="x (m)", ylabel="depth h (m)", legend=:topright, size=LINE_SIZE_PX, xlims=(10.0, 85.0),
        left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM,
        title=@sprintf("Ritter dam break, dx = %.2f m", RITTER_LENGTH / RITTER_PROFILE_NX))
    fig_u = plot(xlabel="x (m)", ylabel="depth-mean velocity (m/s)", legend=:topleft, size=LINE_SIZE_PX, xlims=(10.0, 85.0),
        left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM)
    m1d, m2dv = ritter_model(RITTER_PROFILE_NX, 1), ritter_model(RITTER_PROFILE_NX, RITTER_NZ_2DV)
    x = wd_x_centres(m1d.config)
    xs = range(0.0, RITTER_LENGTH, length=4001)
    t_prev = 0.0
    for (j, t) in enumerate(RITTER_PROFILE_TIMES)
        run_to!(m1d, t - t_prev)
        run_to!(m2dv, t - t_prev)
        t_prev = t
        exact = [ritter_dam_break(xi, t; h0=RITTER_H0, x0=RITTER_X0) for xi in xs]
        plot!(fig_h, xs, first.(exact); color=:black, linestyle=:dash, label=j == 1 ? "Ritter (1892)" : "")
        plot!(fig_h, x, water_depth(m1d); color=j, linewidth=2, label=@sprintf("1D, t = %.2f s", t))
        plot!(fig_h, x, water_depth(m2dv); color=j, linewidth=1, linestyle=:dot, label=@sprintf("2DV, t = %.2f s", t))
        plot!(fig_u, xs, [h > 0 ? u : NaN for (h, u) in exact]; color=:black, linestyle=:dash, label=j == 1 ? "Ritter (1892)" : "")
        for (m, style, lab) in ((m1d, :solid, "1D"), (m2dv, :dot, "2DV"))
            hd = water_depth(m)
            q = [sum(m.wetdry.h[:, i] .* 0.5 .* (m.state.u[:, i] .+ m.state.u[:, i + 1])) for i in 1:m.config.nx]
            plot!(fig_u, x, [hd[i] > RITTER_FRONT_DEPTH ? q[i] / hd[i] : NaN for i in eachindex(hd)]; color=j,
                linewidth=style == :solid ? 2 : 1, linestyle=style, label=@sprintf("%s, t = %.2f s", lab, t))
        end
    end
    save_figure(plot(fig_h, fig_u; layout=(2, 1), size=(820, 760)), "ritter_profiles.png")

    # convergence
    fig_c = plot(xscale=:log10, yscale=:log10, xlabel="dx (m)", ylabel="relative error (-)", legend=:bottomright,
        size=LINE_SIZE_PX, left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM,
        title=@sprintf("Ritter dam break at t = %.0f s: convergence", RITTER_T_END))
    for (label, nz, nxs, style) in (("1D", 1, RITTER_NX_1D, :solid), ("2DV", RITTER_NZ_2DV, RITTER_NX_2DV, :dash))
        dxs, l1s, fronts = Float64[], Float64[], Float64[]
        for nx in nxs
            m = run_to!(ritter_model(nx, nz), RITTER_T_END)
            t = round(Int, RITTER_T_END / m.config.dt) * m.config.dt
            h, ha = water_depth(m), ritter_depth(m, t)
            push!(dxs, RITTER_LENGTH / nx)
            push!(l1s, sum(abs, h .- ha) / sum(ha))
            push!(fronts, abs(ritter_front(wd_x_centres(m.config), h) - ritter_front_exact(t)) / (ritter_front_exact(t) - RITTER_X0))
            @printf("%-4s dx = %.4f m | L1 depth error %.4f | front error %.4f\n", label, dxs[end], l1s[end], fronts[end])
        end
        plot!(fig_c, dxs, l1s; marker=:circle, linestyle=style, color=1, label="$(label): L1 depth error")
        plot!(fig_c, dxs, fronts; marker=:square, linestyle=style, color=2, label="$(label): front position (h = 1 cm)")
    end
    dx_ref = [RITTER_LENGTH / RITTER_NX_1D[1], RITTER_LENGTH / RITTER_NX_1D[end]]
    plot!(fig_c, dx_ref, 0.02 .* dx_ref ./ dx_ref[1]; color=:gray, linestyle=:dot, label="first order")
    save_figure(fig_c, "ritter_convergence.png")

    # 2DV animation of the velocity
    m = ritter_model(RITTER_PROFILE_NX, RITTER_NZ_2DV)
    stride = round(Int, RITTER_FRAME_INTERVAL / m.config.dt)
    umax = 2sqrt(m.config.g * RITTER_H0)   # m/s, front speed 2 c0
    anim = Animation()
    frame_at(t) = frame(anim, section(m, cell_velocity(m); clims=(0.0, umax), color=:thermal, colorbar_title="u (m/s)",
        title=@sprintf("Ritter dam break, 2DV (%d x %d), t = %.1f s", m.config.nx, m.config.nz, t), size=SECTION_SIZE_PX,
        left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM))
    frame_at(0.0)
    run_to!(m, RITTER_ANIMATION_END; on_step=(mm, t) -> (round(Int, t / mm.config.dt) % stride == 0 && frame_at(t)))
    save_gif(anim, "ritter_2dv.gif")
end

# ---- Carrier-Greenspan runup -----------------------------------------------------------------------------------------
function plot_runup()
    println("\n=== Carrier-Greenspan standing wave on a planar beach ===")
    T = carrier_greenspan_time_scale(CG_WAVE)
    period = pi * T   # s
    amplitude = cg_runup_amplitude()
    fig_s = plot(xlabel="t / period (-)", ylabel="shoreline elevation (m)", legend=:topright, size=LINE_SIZE_PX,
        left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM,
        title=@sprintf("Carrier-Greenspan runup, A = %.1f, slope 1:%d", CG_AMPLITUDE, round(Int, 1 / CG_SLOPE)))
    ts = range(0.0, period, length=801)
    t0 = carrier_greenspan_rest_time(CG_WAVE)
    plot!(fig_s, ts ./ period, [carrier_greenspan_shoreline(CG_WAVE, t0 + t)[2] for t in ts]; color=:black, linewidth=2,
        linestyle=:dash, label="Carrier & Greenspan (1958)")
    profile_panels = []
    anim = Animation()
    for (j, run) in enumerate(CG_RUNS)
        model, t0 = cg_model(run.nx, run.nz, run.dt)
        nsteps = round(Int, period / run.dt)
        phase_steps = Set(round(Int, p * nsteps / CG_PHASES) for p in 0:(CG_PHASES - 1))
        frame_stride = round(Int, CG_FRAME_INTERVAL / run.dt)
        last_run = j == length(CG_RUNS)
        x = wd_x_centres(model.config)
        bed = model.wetdry.bed .- CG_STILL_LEVEL
        tt, shore = Float64[0.0], Float64[cg_shoreline(model)[1]]
        max_error = 0.0
        for n in 0:nsteps
            if n > 0
                step_forward!(model)
                push!(tt, n * run.dt)
                push!(shore, cg_shoreline(model)[1])
                max_error = max(max_error, abs(shore[end] - carrier_greenspan_shoreline(CG_WAVE, t0 + tt[end])[2]))
            end
            last_run || continue
            t = n * run.dt
            surface = [h > PLOT_MIN_THICKNESS ? z - CG_STILL_LEVEL : NaN for (h, z) in zip(water_depth(model), wd_water_level(model))]
            if n in phase_steps
                p = plot(x, bed; fillrange=-0.4, color=BED_COLOR, linewidth=0, label="", xlims=CG_XLIMS, ylims=(-0.25, 0.25),
                    title=@sprintf("t / period = %.3f", t / period), titlefontsize=9, legend=false)
                plot!(p, x, surface; color=1, linewidth=3)
                plot!(p, x, cg_surface_exact(x, t0 + t); color=:black, linestyle=:dash, linewidth=1.5)
                push!(profile_panels, p)
            end
            if n % frame_stride == 0
                p = section(model, cell_velocity(model); clims=(-0.6, 0.6), color=:balance, colorbar_title="u (m/s)",
                    xlims=CG_XLIMS, ylims=(0.6, CG_GRID_DEPTH), title=@sprintf("Carrier-Greenspan runup, %s, t / period = %.2f", run.label, t / period),
                    size=SECTION_SIZE_PX, left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM)
                plot!(p, x, cg_surface_exact(x, t0 + t) .+ CG_STILL_LEVEL; color=:white, linestyle=:dash, linewidth=1.5, label="")
                frame(anim, p)
            end
        end
        @printf("%-26s | max shoreline error %.4f m = %.1f %% of the runup amplitude %.3f m\n", run.label, max_error,
            100max_error / amplitude, amplitude)
        plot!(fig_s, tt ./ period, shore; color=j, linewidth=j == length(CG_RUNS) ? 1 : 2, label=run.label)
    end
    save_figure(fig_s, "runup_shoreline.png")
    save_figure(plot(profile_panels...; layout=(2, 4), size=(1300, 560), plot_title="Surface elevation above the still level (m): model (blue), analytical (dashed)",
        plot_titlefontsize=11, left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM), "runup_profiles.png")
    save_gif(anim, "runup_2dv.gif")
end

# ---- Sloshing with flooding and drying, two water masses -------------------------------------------------------------
function plot_sloshing()
    println("\n=== Sloshing over two beaches, two water masses ===")
    fig_cons = plot(xlabel="t (s)", ylabel="relative change (-)", yscale=:log10, legend=:bottomright, size=LINE_SIZE_PX,
        left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM, ylims=(1e-18, 1e-10),
        title="Conservation with flooding and drying (running maximum)")
    fig_wet = plot(xlabel="t (s)", ylabel="shoreline position x (m)", legend=:right, size=LINE_SIZE_PX,
        left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM)
    anim = Animation()
    panels = Dict{Float64,Any}()
    # WHY the floor: the change is often exactly zero; a log axis needs a positive value (shown at the axis bottom)
    floor_value = 1e-18
    for (label, nz, style) in (("1D", 1, :solid), ("2DV", SLOSH_NZ_2DV, :dash))
        model = sloshing_model(nz)
        cfg = model.config
        d0 = compute_diagnostics(model)
        rho_lo, rho_hi = extrema(model.state.rho)
        nsteps = round(Int, SLOSH_T_END / cfg.dt)
        frame_stride = round(Int, SLOSH_FRAME_INTERVAL / cfg.dt)
        snapshot_steps = Dict(round(Int, t / cfg.dt) => t for t in SLOSH_SNAPSHOT_TIMES)
        tt, dvol, dmass, shore_l, shore_r = Float64[], Float64[], Float64[], Float64[], Float64[]
        x = wd_x_centres(cfg)
        min_depth, overshoot = Inf, 0.0
        is2dv = nz > 1
        draw(t) = section(model, model.state.rho; clims=SLOSH_CLIMS, colorbar_title="ρ (kg/m³)",
            title=@sprintf("Sloshing, 2DV (%d x %d), t = %.1f s", cfg.nx, cfg.nz, t), left_margin=FIGURE_MARGIN_MM,
            bottom_margin=FIGURE_MARGIN_MM)
        for n in 0:nsteps
            n > 0 && step_forward!(model)
            t = n * cfg.dt
            d = compute_diagnostics(model)
            push!(tt, t)
            push!(dvol, max(abs(d["water_volume"] / d0["water_volume"] - 1.0), floor_value))
            push!(dmass, max(abs(d["constituent_mass"] / d0["constituent_mass"] - 1.0), floor_value))
            wet = findall(>(SLOSH_WET_DEPTH), water_depth(model))
            push!(shore_l, x[first(wet)])
            push!(shore_r, x[last(wet)])
            min_depth = min(min_depth, minimum(water_depth(model)))
            overshoot = max(overshoot, maximum(model.state.rho) - rho_hi, rho_lo - minimum(model.state.rho))
            is2dv || continue
            n % frame_stride == 0 && frame(anim, plot(draw(t); size=SECTION_SIZE_PX))
            haskey(snapshot_steps, n) && (panels[snapshot_steps[n]] = draw(t))
        end
        @printf("%-4s | max rel. volume change %.2e | max rel. mass change %.2e | min depth %.2e m | density overshoot %.2e kg/m^3 | shorelines x = %.2f ... %.2f m and %.2f ... %.2f m\n",
            label, maximum(dvol), maximum(dmass), min_depth, overshoot, extrema(shore_l)..., extrema(shore_r)...)
        plot!(fig_cons, tt, accumulate(max, dvol); color=1, linestyle=style, label="$(label): water volume")
        plot!(fig_cons, tt, accumulate(max, dmass); color=2, linestyle=style, label="$(label): constituent mass")
        plot!(fig_wet, tt, shore_l; color=1, linestyle=style, label="$(label): left shoreline")
        plot!(fig_wet, tt, shore_r; color=2, linestyle=style, label="$(label): right shoreline")
    end
    save_figure(plot(fig_cons, fig_wet; layout=(2, 1), size=(820, 700)), "sloshing_conservation.png")
    save_figure(plot([panels[t] for t in SLOSH_SNAPSHOT_TIMES]...; layout=(3, 2), size=(1300, 820)), "sloshing_snapshots.png")
    save_gif(anim, "sloshing_2dv.gif")
end


# ---- Manufactured solution for the momentum advection ----------------------------------------------------------------
const MMS_NX = (100, 200, 400, 800, 1600)   # -, grids of the convergence plot (dx = 1 ... 1/16 m)
const MMS_NZ = 12                           # -, layers of 0.1 m
const MMS_PROFILE_NX = 400                  # -, grid of the tendency profiles

function plot_mms()
    println("\n=== Manufactured solution: momentum advection of a depth-uniform velocity ===")
    maxerr(xf, tend, exact) = maximum(abs(tend[k, f] - exact[f]) for f in axes(tend, 2), k in axes(tend, 1) if !isnan(tend[k, f]))
    fig = plot(xscale=:log10, yscale=:log10, xlabel="dx (m)", ylabel="max |tendency error| (m/s²)", legend=:outerright,
        size=(900, 420), left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM,
        title="Advection of a depth-uniform u, maximum error over all layers", titlefontsize=11)
    dxs = [MMS_LENGTH / nx for nx in MMS_NX]
    for (label, nz, method, color, style) in (("1D, upwind1", 1, "upwind1", 1, :solid), ("1D, tvd2", 1, "tvd2", 2, :solid),
            ("2DV, upwind1", MMS_NZ, "upwind1", 1, :dash), ("2DV, tvd2", MMS_NZ, "tvd2", 2, :dash),
            ("2DV, layer-wise weights (first version)", MMS_NZ, "layerwise", 3, :dashdot))
        errs = [maxerr(mms_advection_tendencies(nx, nz; method=method)...) for nx in MMS_NX]
        @printf("%-42s max error %s\n", label, join([@sprintf("%.2e", e) for e in errs], " "))
        plot!(fig, dxs, errs; marker=:circle, color=color, linestyle=style, label=label)
    end
    plot!(fig, dxs, 0.008 .* dxs; color=:gray, linestyle=:dot, label="first order")
    save_figure(fig, "mms_advection_convergence.png")

    # tendency of every layer along x (2DV): first-version weights against the present ones
    panels = []
    for (method, title) in (("layerwise", "layer-wise weights (first version)"), ("upwind1", "column-consistent weights (upwind1)"))
        xf, tend, exact = mms_advection_tendencies(MMS_PROFILE_NX, MMS_NZ; method=method)
        p = plot(xf, exact; color=:black, linewidth=3, label="exact -u du/dx", xlabel="x (m)", ylabel="du/dt (m/s²)",
            title="2DV, dx = $(MMS_LENGTH / MMS_PROFILE_NX) m: $(title)", ylims=(-1.2, 0.4), legend=:bottomleft,
            titlefontsize=10)
        for k in 1:MMS_NZ
            plot!(p, xf, tend[k, :]; color=k, linewidth=1, label=k == 1 ? "layers 1 ... $(MMS_NZ)" : "", seriestype=:scatter,
                markersize=1.5, markerstrokewidth=0)
        end
        push!(panels, p)
    end
    save_figure(plot(panels...; layout=(2, 1), size=(900, 700), left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM),
        "mms_advection_layers.png")
end

# ---- upwind1 against tvd2 ----------------------------------------------------------------------------------------------
const SCHEMES = ("upwind1", "tvd2")
const SCHEME_COLORS = Dict("upwind1" => 1, "tvd2" => 2)
const SCHEME_SNAPSHOT_TIMES = (8.0, 20.0, 40.0)   # s, sloshing comparison

layer_spread(m) = maximum((begin
    us = [m.state.u[k, f] for k in 1:m.config.nz if m.wetdry.hf[k, f] > 1e-3]
    length(us) > 1 ? maximum(us) - minimum(us) : 0.0
end for f in 2:m.config.nx); init=0.0)

function plot_schemes()
    println("\n=== upwind1 against tvd2 ===")
    # Ritter: profiles near the front, 1D and 2DV
    fig_h = plot(xlabel="x (m)", ylabel="depth h (m)", xlims=(40.0, 75.0), ylims=(0.0, 0.45), legend=:topright,
        title=@sprintf("Ritter dam break, t = %.0f s, dx = %.2f m", RITTER_T_END, RITTER_LENGTH / RITTER_PROFILE_NX))
    xs = range(40.0, 75.0, length=2001)
    plot!(fig_h, xs, [ritter_dam_break(x, RITTER_T_END; h0=RITTER_H0, x0=RITTER_X0)[1] for x in xs]; color=:black, linewidth=2,
        linestyle=:dash, label="Ritter (1892)")
    layer_panels = []
    for scheme in SCHEMES, (nz, style, lab) in ((1, :solid, "1D"), (RITTER_NZ_2DV, :dot, "2DV"))
        m = run_to!(ritter_model(RITTER_PROFILE_NX, nz; scheme=scheme), RITTER_T_END)
        plot!(fig_h, wd_x_centres(m.config), water_depth(m); color=SCHEME_COLORS[scheme], linestyle=style, linewidth=2,
            label="$(lab), $(scheme)")
        if nz > 1
            xf = [(f - 1) * m.dx for f in 1:(m.config.nx + 1)]
            p = plot(xlabel="x (m)", ylabel="u (m/s)", xlims=(10.0, 75.0), legend=:topleft, titlefontsize=10,
                title=@sprintf("2DV %s: velocity of every layer (spread %.3f m/s)", scheme, layer_spread(m)))
            xa = range(10.0, 75.0, length=2001)
            plot!(p, xa, [ritter_dam_break(x, RITTER_T_END; h0=RITTER_H0, x0=RITTER_X0)[1] > 0 ?
                ritter_dam_break(x, RITTER_T_END; h0=RITTER_H0, x0=RITTER_X0)[2] : NaN for x in xa]; color=:black, linewidth=2,
                linestyle=:dash, label="Ritter (1892)")
            for k in 1:nz
                plot!(p, xf, [m.wetdry.hf[k, f] > 1e-3 ? m.state.u[k, f] : NaN for f in eachindex(xf)]; color=k, linewidth=1,
                    label=k == 1 ? "layers 1 ... $(nz)" : "")
            end
            push!(layer_panels, p)
        end
    end
    save_figure(plot(fig_h; size=(820, 420), left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM), "schemes_ritter_front.png")
    save_figure(plot(layer_panels...; layout=(2, 1), size=(820, 700), left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM),
        "schemes_ritter_layers.png")

    # Ritter: convergence of both schemes
    fig_c = plot(xscale=:log10, yscale=:log10, xlabel="dx (m)", ylabel="relative error (-)", legend=:outerright, size=(950, 440),
        left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM, title="Ritter dam break: upwind1 against tvd2")
    for scheme in SCHEMES, (label, nz, nxs, style) in (("1D", 1, RITTER_NX_1D, :solid), ("2DV", RITTER_NZ_2DV, RITTER_NX_2DV, :dash))
        dxs, l1s, fronts = Float64[], Float64[], Float64[]
        for nx in nxs
            m = run_to!(ritter_model(nx, nz; scheme=scheme), RITTER_T_END)
            t = round(Int, RITTER_T_END / m.config.dt) * m.config.dt
            h, ha = water_depth(m), ritter_depth(m, t)
            push!(dxs, RITTER_LENGTH / nx)
            push!(l1s, sum(abs, h .- ha) / sum(ha))
            push!(fronts, abs(ritter_front(wd_x_centres(m.config), h) - ritter_front_exact(t)) / (ritter_front_exact(t) - RITTER_X0))
            @printf("%-7s %-4s dx = %.4f m | L1 %.4f | front %.4f | layer spread %.4f m/s\n", scheme, label, dxs[end], l1s[end],
                fronts[end], layer_spread(m))
        end
        plot!(fig_c, dxs, l1s; marker=:circle, linestyle=style, color=SCHEME_COLORS[scheme], label="$(label) $(scheme): L1 depth")
        plot!(fig_c, dxs, fronts; marker=:square, linestyle=style, color=SCHEME_COLORS[scheme] + 2,
            label="$(label) $(scheme): front (h = 1 cm)")
    end
    dx_ref = [RITTER_LENGTH / RITTER_NX_1D[1], RITTER_LENGTH / RITTER_NX_1D[end]]
    plot!(fig_c, dx_ref, 0.02 .* dx_ref ./ dx_ref[1]; color=:gray, linestyle=:dot, label="first order")
    save_figure(fig_c, "schemes_ritter_convergence.png")

    # Carrier-Greenspan: shoreline error over one period
    T = carrier_greenspan_time_scale(CG_WAVE)
    period = pi * T
    fig_s = plot(xlabel="t / period (-)", ylabel="shoreline elevation error (m)", legend=:outerright, size=(950, 420),
        left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM, title="Carrier-Greenspan runup: upwind1 against tvd2")
    for scheme in SCHEMES, (nx, nz, dt, lab, style) in ((480, 1, 0.01, "1D", :solid), (480, 15, 0.01, "2DV", :dash))
        model, t0 = cg_model(nx, nz, dt; scheme=scheme)
        n = round(Int, period / dt)
        err = Float64[]
        for s in 1:n
            step_forward!(model)
            push!(err, cg_shoreline(model)[1] - carrier_greenspan_shoreline(CG_WAVE, t0 + s * dt)[2])
        end
        @printf("%-7s %-3s dx = %.2f m | max shoreline error %.4f m\n", scheme, lab, (CG_LENGTH + CG_ONSHORE) / nx, maximum(abs, err))
        plot!(fig_s, (1:n) .* dt ./ period, err; color=SCHEME_COLORS[scheme], linestyle=style, label="$(lab), $(scheme), dx = 0.05 m")
    end
    hline!(fig_s, [0.0]; color=:black, label="")
    save_figure(fig_s, "schemes_runup_shoreline.png")

    # Sloshing: density in 2DV, snapshots and side-by-side animation
    models = Dict(scheme => sloshing_model(SLOSH_NZ_2DV; scheme=scheme) for scheme in SCHEMES)
    cfg = models["upwind1"].config
    nsteps = round(Int, SLOSH_T_END / cfg.dt)
    frame_stride = round(Int, SLOSH_FRAME_INTERVAL / cfg.dt)
    snapshot_steps = Dict(round(Int, t / cfg.dt) => t for t in SCHEME_SNAPSHOT_TIMES)
    draw(scheme, t; colorbar=true) = section(models[scheme], models[scheme].state.rho; clims=SLOSH_CLIMS,
        colorbar_title="ρ (kg/m³)", title=@sprintf("%s, t = %.1f s", scheme, t), titlefontsize=10, colorbar=colorbar)
    anim = Animation()
    panels = Dict{Tuple{String,Float64},Any}()
    for n in 0:nsteps
        if n > 0
            foreach(step_forward!, values(models))
        end
        t = n * cfg.dt
        n % frame_stride == 0 && frame(anim, plot(draw("upwind1", t), draw("tvd2", t); layout=(2, 1), size=(900, 560),
            left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM))
        if haskey(snapshot_steps, n)
            for scheme in SCHEMES
                panels[(scheme, snapshot_steps[n])] = draw(scheme, t; colorbar=snapshot_steps[n] == last(SCHEME_SNAPSHOT_TIMES))
            end
        end
    end
    for scheme in SCHEMES
        rho = models[scheme].state.rho
        @printf("sloshing %-7s t = %.0f s | density range [%.3f, %.3f] kg/m^3\n", scheme, SLOSH_T_END, extrema(rho)...)
    end
    save_figure(plot([panels[(scheme, t)] for scheme in SCHEMES for t in SCHEME_SNAPSHOT_TIMES]...; layout=grid(2, 3, widths=[0.3, 0.3, 0.4]),
        size=(1500, 620), left_margin=6Plots.mm, bottom_margin=6Plots.mm), "schemes_sloshing.png")
    save_gif(anim, "schemes_sloshing.gif")
end

# ---- Non-hydrostatic: dispersion of a standing wave ------------------------------------------------------------------
const DISP_KH = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)   # -, relative depths of the dispersion curve
const DISP_GRIDS = ((20, 10), (40, 20), (80, 40), (160, 80))   # (cells per half wavelength, layers) at k H = pi/2
const DISP_TIMESERIES_PERIODS = 2.0                 # -, length of the surface time series

function plot_dispersion()
    println("\n=== Non-hydrostatic: dispersion of a standing wave ===")
    g = 9.81
    H = NH_WAVE_DEPTH
    kh_curve = range(0.05, 3.2, length=300)
    fig = plot(kh_curve, sqrt.(tanh.(kh_curve) ./ kh_curve); color=:black, linewidth=2, label="exact: ω² = g k tanh(k H)",
        xlabel="k H (-)", ylabel="phase speed c / √(g H) (-)", legend=:bottomleft, size=LINE_SIZE_PX,
        left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM, title="Standing wave, 40 cells per half wavelength, 40 layers")
    hline!(fig, [1.0]; color=:gray, linestyle=:dash, label="shallow water")
    for (hydro, label, marker) in ((true, "hydrostatic", :square), (false, "non-hydrostatic", :circle))
        speeds = Float64[]
        for kH in DISP_KH
            model, period = standing_wave_model(kH; use_hydrostatic=hydro, momentum_advection_scheme="tvd2", transport_advection_scheme="tvd2")
            # WHY a longer search window for the hydrostatic mode: its period is the shallow-water one
            measured = standing_wave_measured_period(model, hydro ? 2pi / (pi / model.config.length * sqrt(g * H)) : period)
            k = pi / model.config.length
            push!(speeds, 2pi / (measured * k) / sqrt(g * H))
            @printf("%-15s kH = %.2f | period %.4f s, exact %.4f s, error %+.2f %%\n", label, kH, measured, period, 100(measured - period) / period)
        end
        scatter!(fig, collect(DISP_KH), speeds; marker=marker, markersize=6, label=label)
    end
    save_figure(fig, "nonhydro_dispersion.png")

    # convergence at k H = pi/2 and surface time series at the wall
    kH = pi / 2
    errs, dxs = Float64[], Float64[]
    for (nxh, nz) in DISP_GRIDS
        model, period = standing_wave_model(kH; nx_per_half_wave=nxh, nz=nz, use_hydrostatic=false)
        measured = standing_wave_measured_period(model, period)
        push!(errs, abs(measured - period) / period)
        push!(dxs, model.dx)
        @printf("k H = pi/2, %3d x %2d cells | period error %.3f %%\n", nxh, nz, 100errs[end])
    end
    fig_c = plot(dxs, errs; xscale=:log10, yscale=:log10, marker=:circle, label="non-hydrostatic (upwind1)", xlabel="dx (m)",
        ylabel="relative period error (-)", legend=:bottomright, title="k H = π/2: convergence (dz = dx / 2)")
    plot!(fig_c, dxs, errs[1] .* dxs ./ dxs[1]; color=:gray, linestyle=:dot, label="first order")
    fig_t = plot(xlabel="t / T (-)", ylabel="η at the wall / a (-)", legend=:bottomleft, title="Surface at the left wall, k H = π/2")
    for (hydro, label) in ((true, "hydrostatic"), (false, "non-hydrostatic"))
        model, period = standing_wave_model(kH; use_hydrostatic=hydro)
        n = round(Int, DISP_TIMESERIES_PERIODS * period / model.config.dt)
        series = [model.state.eta[1]]
        for _ in 1:n
            step_forward!(model)
            push!(series, model.state.eta[1])
        end
        plot!(fig_t, (0:n) .* model.config.dt ./ period, series ./ NH_WAVE_AMPLITUDE; linewidth=2, label=label)
    end
    ts = range(0.0, DISP_TIMESERIES_PERIODS, length=500)
    plot!(fig_t, ts, cos.(2pi .* ts); color=:black, linestyle=:dash, label="exact (linear)")
    save_figure(plot(fig_t, fig_c; layout=(1, 2), size=(1200, 420), left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM),
        "nonhydro_standing_wave.png")
end

# ---- Solitary wave runup on a plane beach ------------------------------------------------------------------------------
const SOL_T_END = 22.0                       # s, past the maximum runup (about 18.5 s)
const SOL_DX_1D = (0.2, 0.1, 0.05, 0.025, 0.0125)   # m
const SOL_DX_2DV = (0.2, 0.1, 0.05)          # m
const SOL_NZ = 12                            # -, layers of 0.1 m
const SOL_DT_DX = 0.1                        # s/m, time step per grid spacing
const SOL_DOC_DX = 0.05                      # m, grid of the time series, profiles and animation
const SOL_PROFILE_TIMES = (6.0, 12.0, 16.0, 18.5)   # s
const SOL_FRAME_INTERVAL = 0.2               # s
const SOL_XLIMS = (20.0, SOL_OFFSHORE + SOL_ONSHORE)   # m

"Runs the solitary-wave case; returns times (s), shoreline elevation (m above the still level), the maximum runup and the model."
function run_solitary(dx, nz; hydro=true, scheme="upwind1", on_step=nothing)
    model = solitary_runup_model(dx, nz; dt=SOL_DT_DX * dx, use_hydrostatic=hydro, momentum_advection_scheme=scheme,
        transport_advection_scheme=scheme)
    n = round(Int, SOL_T_END / model.config.dt)
    ts, shore = Float64[], Float64[]
    for s in 1:n
        step_forward!(model)
        i = findlast(>(SOL_SHORE_DEPTH), water_depth(model))
        push!(ts, s * model.config.dt)
        push!(shore, wd_water_level(model)[i] - SOL_STILL_LEVEL)
        on_step === nothing || on_step(model, s * model.config.dt)
    end
    return ts, shore, maximum(shore), model
end

function plot_solitary()
    println("\n=== Solitary wave runup on a plane beach (Synolakis 1987) ===")
    R = solitary_runup_law()
    @printf("runup law: R = %.4f m (H/d = %.3f, cot beta = %.2f)\n", R, SOL_HEIGHT, SOL_COT_BETA)
    fig_c = plot(xscale=:log10, xlabel="dx (m)", ylabel="maximum runup / law (-)", legend=:bottomleft, size=LINE_SIZE_PX,
        left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM, ylims=(0.75, 1.05),
        title=@sprintf("Solitary wave runup, H/d = %.3f, slope 1:%.2f", SOL_HEIGHT, SOL_COT_BETA))
    hline!(fig_c, [1.0]; color=:black, linestyle=:dash, label="Synolakis (1987)")
    for (label, nz, hydro, scheme, dxs, color, style) in (("1D upwind1", 1, true, "upwind1", SOL_DX_1D, 1, :solid),
            ("1D tvd2", 1, true, "tvd2", SOL_DX_1D, 2, :solid),
            ("2DV hydrostatic, tvd2", SOL_NZ, true, "tvd2", SOL_DX_2DV, 3, :dash),
            ("2DV non-hydrostatic, tvd2", SOL_NZ, false, "tvd2", SOL_DX_2DV, 4, :dash))
        ratios = Float64[]
        for dx in dxs
            _, _, rmax, _ = run_solitary(dx, nz; hydro=hydro, scheme=scheme)
            push!(ratios, rmax / R)
            @printf("%-26s dx = %.4f m | max runup %.4f m = %.3f of the law\n", label, dx, rmax, rmax / R)
        end
        plot!(fig_c, collect(dxs), ratios; marker=:circle, color=color, linestyle=style, label=label)
    end
    save_figure(fig_c, "solitary_runup_convergence.png")

    # time series, profiles and animation at SOL_DOC_DX: 2DV hydrostatic against non-hydrostatic
    fig_s = plot(xlabel="t (s)", ylabel="shoreline elevation (m)", legend=:topleft, size=LINE_SIZE_PX, left_margin=FIGURE_MARGIN_MM,
        bottom_margin=FIGURE_MARGIN_MM, title=@sprintf("Shoreline, dx = %.2f m, %d layers", SOL_DOC_DX, SOL_NZ))
    hline!(fig_s, [R]; color=:black, linestyle=:dash, label="maximum runup, Synolakis (1987)")
    profiles = Dict{Tuple{Bool,Float64},Vector{Float64}}()
    anim = Animation()
    x = Float64[]
    bed = Float64[]
    for (hydro, label) in ((true, "hydrostatic"), (false, "non-hydrostatic"))
        profile_steps = Dict(round(Int, t / (SOL_DT_DX * SOL_DOC_DX)) => t for t in SOL_PROFILE_TIMES)
        frame_stride = round(Int, SOL_FRAME_INTERVAL / (SOL_DT_DX * SOL_DOC_DX))
        function record(m, t)
            s = round(Int, t / m.config.dt)
            surface = [h > PLOT_MIN_THICKNESS ? z - SOL_STILL_LEVEL : NaN for (h, z) in zip(water_depth(m), wd_water_level(m))]
            haskey(profile_steps, s) && (profiles[(hydro, profile_steps[s])] = surface)
            if !hydro && s % frame_stride == 0
                frame(anim, plot(section(m, cell_velocity(m); clims=(-0.2, 0.2), color=:balance, colorbar_title="u (m/s)",
                    xlims=SOL_XLIMS, ylims=(0.0, SOL_GRID_DEPTH), title=@sprintf("Solitary wave runup, non-hydrostatic, t = %.1f s", t));
                    size=SECTION_SIZE_PX, left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM))
            end
        end
        ts, shore, rmax, m = run_solitary(SOL_DOC_DX, SOL_NZ; hydro=hydro, scheme="tvd2", on_step=record)
        plot!(fig_s, ts, shore; linewidth=2, label=@sprintf("%s (max %.1f %% of the law)", label, 100rmax / R))
        x = wd_x_centres(m.config)
        bed = m.wetdry.bed .- SOL_STILL_LEVEL
    end
    save_figure(fig_s, "solitary_shoreline.png")
    panels = []
    for t in SOL_PROFILE_TIMES
        p = plot(x, bed; fillrange=-1.0, color=BED_COLOR, linewidth=0, label="", xlims=SOL_XLIMS, ylims=(-0.05, 0.12),
            title=@sprintf("t = %.1f s", t), titlefontsize=10, xlabel="x (m)", ylabel="η (m)", legend=t == first(SOL_PROFILE_TIMES) ? :topleft : false)
        plot!(p, x, profiles[(true, t)]; color=1, linewidth=2, label="hydrostatic")
        plot!(p, x, profiles[(false, t)]; color=2, linewidth=2, linestyle=:dash, label="non-hydrostatic")
        push!(panels, p)
    end
    save_figure(plot(panels...; layout=(2, 2), size=(1100, 600), left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM),
        "solitary_profiles.png")
    save_gif(anim, "solitary_nonhydro.gif")
end

mkpath(OUTPUT_DIR)
mkpath(FIGURE_DIR)
for mode in MODES
    mode == "lake" ? plot_lake() :
    mode == "ritter" ? plot_ritter() :
    mode == "runup" ? plot_runup() :
    mode == "sloshing" ? plot_sloshing() :
    mode == "mms" ? plot_mms() :
    mode == "schemes" ? plot_schemes() :
    mode == "dispersion" ? plot_dispersion() :
    mode == "solitary" ? plot_solitary() :
    error("unknown mode '$(mode)'; use lake, ritter, runup, sloshing, mms, schemes, dispersion or solitary")
end
