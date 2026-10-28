# Wetting and drying (issue #31): figures and animations for the README.
#
# Usage (any working directory):
#   julia --project=<repository> experiments/wetting_drying/plot_wetting_drying.jl [mode ...]
# Modes: lake, ritter, runup, sloshing (default: all four); each takes well under a minute.
# Output: output_barotropic/wetting_drying_figures/; the PNG and GIF files are copied to figures/.

# WHY: batch run without a display; same GR workstation choice as src/visualization.jl.
get!(ENV, "GKSwstype", "100")

using SweTransport2DV
using Plots
using Printf

include(joinpath(@__DIR__, "cases.jl"))

const MODES = isempty(ARGS) ? ["lake", "ritter", "runup", "sloshing"] : ARGS
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

mkpath(OUTPUT_DIR)
mkpath(FIGURE_DIR)
for mode in MODES
    mode == "lake" ? plot_lake() :
    mode == "ritter" ? plot_ritter() :
    mode == "runup" ? plot_runup() :
    mode == "sloshing" ? plot_sloshing() : error("unknown mode '$(mode)'; use lake, ritter, runup or sloshing")
end
