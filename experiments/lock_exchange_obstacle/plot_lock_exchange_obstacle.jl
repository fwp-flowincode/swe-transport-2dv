# Lock exchange with obstacle (issue #3): figures for the README.
#
# Usage (any working directory):
#   julia --project=<repository> experiments/lock_exchange_obstacle/plot_lock_exchange_obstacle.jl sweep
#       drho = 1, 10, 50 kg/m^3 on 292 x 292 cells (2 mm), front position and obstacle force against t* for the
#       three density differences (Boussinesq self-similarity check); about 2 min per run.
#   julia --project=<repository> experiments/lock_exchange_obstacle/plot_lock_exchange_obstacle.jl documented
#       drho = 10 kg/m^3 on 584 x 584 cells (1 mm): density GIF, snapshot panel and final density; about 20 min.
# Output: output_barotropic/lock_exchange_obstacle_figures/; the figures in figures/ are copies.

# WHY: batch run without a display; same GR workstation choice as src/visualization.jl.
get!(ENV, "GKSwstype", "100")

using SweTransport2DV
using Plots
using NPZ
using Printf

include(joinpath(@__DIR__, "cases.jl"))

# WHY 7: from t* ~ 7.5 the splash at the right wall drives a spurious thin jet along the free surface (top cell of the
# fixed grid in the non-hydrostatic mode; see README, Limitations). All runs are clean up to t* = 7.
const T_END_STAR = 7.0                      # -, simulated time in units of sqrt(a/g')
const SAMPLE_INTERVAL_STAR = 0.05           # -, sampling interval of front position and force
const ANIMATION_INTERVAL_STAR = 0.1         # -, time between animation frames
const SNAPSHOT_TIMES_STAR = (0.0, 1.0, 2.0, 3.0, 5.0, 7.0)    # -
const SWEEP_DRHO = (1.0, 10.0, 50.0)        # kg/m^3
const SWEEP_N = 292                         # -, cells per direction (2 mm)
const DOCUMENTED_DRHO = 10.0                # kg/m^3
const DOCUMENTED_N = 584                    # -, cells per direction (1 mm)
const TVD2_CFL_LIMIT = 0.5                  # -, monotonicity limit of tvd2 transport
const PROGRESS_INTERVAL_STAR = 1.0          # -
const ANIMATION_FPS = 10                    # frames/s
const FRAME_SIZE_PX = (520, 440)            # px, one animation frame (square tank plus colour bar)
const SNAPSHOT_SIZE_PX = (1200, 760)        # px, 2 x 3 panel
const LINE_SIZE_PX = (720, 380)             # px, front and force figures
const FIGURE_MARGIN_MM = 4Plots.mm
const COLORMAP = :viridis
const BED_COLOR = :gray55                   # colour of the bed / obstacle in the density plots

const MODE = isempty(ARGS) ? "sweep" : ARGS[1]
const OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "output_barotropic", "lock_exchange_obstacle_figures")

"""
    density_heatmap(model, drho; title="", kwargs...)

Density (kg/m³) on the square tank, colour scale fixed to [rho0, rho0 + drho]; solid cells are drawn in BED_COLOR.
"""
function density_heatmap(model::Model, drho::Real; title::AbstractString="", kwargs...)
    cfg = model.config
    rho = Float64.(model.state.rho)
    rho[.!model.geometry.wet_cell] .= NaN
    x = range(0.5 * model.dx, cfg.length - 0.5 * model.dx, length=cfg.nx)   # m
    z = range(0.5 * model.dz, cfg.depth - 0.5 * model.dz, length=cfg.nz)    # m
    p = heatmap(x, z, rho; color=COLORMAP, clims=(LEO_RHO_LIGHT, LEO_RHO_LIGHT + drho), xlabel="x (m)", ylabel="z (m)",
        colorbar_title="Density (kg/m³)", title=title, xlims=(0.0, cfg.length), ylims=(0.0, cfg.depth), aspect_ratio=:equal,
        background_color_inside=BED_COLOR, kwargs...)
    # WHY: the NaN (solid) cells show the plot background, so the inside background colours the bed; the tank is
    # full of water, so the bed is the only NaN region.
    return p
end

"""
    run_case(n, drho; frames=false, snapshots=false) -> NamedTuple

Run drho (kg/m^3) on n x n cells up to T_END_STAR; returns the series of t*, front position (m) and obstacle force
(N/m), the max CFL number, and optionally the animation and snapshot heatmaps.
"""
function run_case(n::Int, drho::Real; frames::Bool=false, snapshots::Bool=false)
    cfg = lock_exchange_obstacle_config(; n=n, drho=drho, t_end_star=T_END_STAR, write_plots=false, write_animation=false,
        write_pressure_buoyancy_diagnostics=false)
    model = Model(cfg)
    initialize_lock_exchange_obstacle!(model, drho)
    tscale = leo_time_scale(drho)   # s
    @printf("\n=== drho = %g kg/m^3, %d x %d cells, dt = %.5f s, t_end = %.2f s (t* = %.1f), time scale %.3f s ===\n",
        drho, n, n, cfg.dt, cfg.nt * cfg.dt, T_END_STAR, tscale)

    stride(interval_star) = max(1, round(Int, interval_star * tscale / cfg.dt))
    sample_stride, frame_stride, progress_stride = stride(SAMPLE_INTERVAL_STAR), stride(ANIMATION_INTERVAL_STAR), stride(PROGRESS_INTERVAL_STAR)
    snapshot_steps = Dict(round(Int, ts * tscale / cfg.dt) => ts for ts in SNAPSHOT_TIMES_STAR)

    t_star = [0.0]
    front = [leo_front_position(model, drho)]       # m
    force = [leo_obstacle_force(model)]             # N/m
    anim = frames ? Animation() : nothing
    frames && frame(anim, density_heatmap(model, drho; title="t* = 0.0", size=FRAME_SIZE_PX))
    panels = Dict{Float64,Any}()
    snapshots && (panels[0.0] = density_heatmap(model, drho; title="t* = 0", colorbar=false))
    max_cfl = 0.0   # -
    t0 = time()
    for step in 1:cfg.nt
        step_forward!(model)
        # WHY interior w faces only: the surface face carries the barotropic d(eta)/dt, which does not advect density.
        # PERF: two scans of u and w against a sparse Poisson solve per step.
        max_cfl = max(max_cfl, maximum(abs, model.state.u) * cfg.dt / model.dx + maximum(abs, @view(model.state.w[2:end-1, :])) * cfg.dt / model.dz)
        ts = step * cfg.dt / tscale
        if step % sample_stride == 0
            push!(t_star, ts)
            push!(front, leo_front_position(model, drho))
            push!(force, leo_obstacle_force(model))
        end
        if frames && step % frame_stride == 0
            frame(anim, density_heatmap(model, drho; title=@sprintf("t* = %.1f", ts), size=FRAME_SIZE_PX))
        end
        if snapshots && haskey(snapshot_steps, step)
            panels[snapshot_steps[step]] = density_heatmap(model, drho; title=@sprintf("t* = %g", snapshot_steps[step]), colorbar=false)
        end
        if step % progress_stride == 0
            @printf("t* = %5.1f | front x/a = %.2f | max|u| = %.3f m/s | max CFL so far = %.3f | wall %.0f s\n",
                ts, front[end] / LEO_A, maximum(abs, model.state.u), max_cfl, time() - t0)
        end
    end
    wet_rho = model.state.rho[model.geometry.wet_cell]
    @printf("CASE=lock_exchange_obstacle_drho%g_n%d FULL_WALL_SECONDS=%.1f | max CFL %.3f (tvd2 limit %.1f) | density [%.4f, %.4f]\n",
        drho, n, time() - t0, max_cfl, TVD2_CFL_LIMIT, minimum(wet_rho), maximum(wet_rho))
    max_cfl >= TVD2_CFL_LIMIT && @warn "CFL number reached the tvd2 monotonicity limit; reduce dt" max_cfl
    return (t_star=t_star, front=front, force=force, max_cfl=max_cfl, anim=anim, panels=panels, model=model)
end

mkpath(OUTPUT_DIR)

if MODE == "sweep"
    results = Dict{Float64,Any}()
    for drho in SWEEP_DRHO
        local r = run_case(SWEEP_N, drho)
        results[drho] = r
        npzwrite(joinpath(OUTPUT_DIR, @sprintf("sweep_drho%g_n%d.npz", drho, SWEEP_N)),
            Dict("t_star" => r.t_star, "front_m" => r.front, "force_N_per_m" => r.force, "rho_final" => copy(r.model.state.rho)))
    end
    # Boussinesq scaling: front x/a and force F / (rho0 g' a^2) against t* should collapse for all drho
    fig_front = plot(xlabel="t* = t sqrt(g'/a) (-)", ylabel="front position x/a (-)", legend=:bottomright, size=LINE_SIZE_PX,
        left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM)
    fig_force = plot(xlabel="t* = t sqrt(g'/a) (-)", ylabel="F / (ρ₀ g' a²) (-)", legend=:topright, size=LINE_SIZE_PX,
        left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM)
    for drho in SWEEP_DRHO
        local r = results[drho]
        label = @sprintf("Δρ = %g kg/m³", drho)
        plot!(fig_front, r.t_star, r.front ./ LEO_A; label=label, linewidth=2)
        plot!(fig_force, r.t_star, r.force ./ (LEO_RHO_LIGHT * leo_reduced_gravity(drho) * LEO_A^2); label=label, linewidth=2)
    end
    hline!(fig_front, [2.0, 2.0 + LEO_D / LEO_A]; color=:gray, linestyle=:dash, label="obstacle")
    savefig(fig_front, joinpath(OUTPUT_DIR, "lock_exchange_obstacle_front.png"))
    savefig(fig_force, joinpath(OUTPUT_DIR, "lock_exchange_obstacle_force.png"))
    println("Saved: front and force figures in $(abspath(OUTPUT_DIR))")
elseif MODE == "documented"
    r = run_case(DOCUMENTED_N, DOCUMENTED_DRHO; frames=true, snapshots=true)
    stem = joinpath(OUTPUT_DIR, @sprintf("lock_exchange_obstacle_drho%g_n%d", DOCUMENTED_DRHO, DOCUMENTED_N))
    npzwrite(stem * ".npz", Dict("t_star" => r.t_star, "front_m" => r.front, "force_N_per_m" => r.force, "rho_final" => copy(r.model.state.rho)))
    gif(r.anim, stem * ".gif"; fps=ANIMATION_FPS)
    savefig(plot([r.panels[ts] for ts in SNAPSHOT_TIMES_STAR]...; layout=(2, 3), size=SNAPSHOT_SIZE_PX,
        left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM), stem * "_snapshots.png")
    savefig(density_heatmap(r.model, DOCUMENTED_DRHO; title=@sprintf("t* = %g", T_END_STAR), size=(620, 520)), stem * "_final.png")
    println("Saved: $(abspath(stem)).{npz,gif}, _snapshots.png, _final.png")
else
    error("unknown mode '$(MODE)'; use sweep or documented")
end
