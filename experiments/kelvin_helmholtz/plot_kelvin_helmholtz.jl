# Kelvin-Helmholtz shear-layer instability: density animation, final density, snapshot panel and mixed-layer
# thickness for the case in cases.jl (800 x 400 cells, 120 s).
#
# The script writes to output_barotropic/kelvin_helmholtz_figures/:
# - kelvin_helmholtz.gif            density animation, one frame per ANIMATION_INTERVAL_S (fixed colour scale)
# - kelvin_helmholtz_final.png      density at t = T_END_S
# - kelvin_helmholtz_snapshots.png  density at SNAPSHOT_TIMES_S
# - kelvin_helmholtz_mixing.png     mixed-layer thickness against time
# - kelvin_helmholtz.npz            mixed-layer series and final density (restyle without rerunning)
# The figures in figures/ are copies of these files (the GIF reduced in size).
#
# Usage (any working directory):
#   julia --project=<path to the repository> experiments/kelvin_helmholtz/plot_kelvin_helmholtz.jl
# Run time: about 1 h (48,000 steps on 800 x 400 cells).

# WHY: batch run without a display; same GR workstation choice as src/visualization.jl.
get!(ENV, "GKSwstype", "100")

using SweTransport2DV
using Plots
using NPZ
using Printf

include(joinpath(@__DIR__, "cases.jl"))

const T_END_S = 120.0                      # s, simulated time (4x the former 30 s preview)
const ANIMATION_INTERVAL_S = 1.0           # s, simulated time between animation frames
const SNAPSHOT_TIMES_S = (0.0, 10.0, 20.0, 40.0, 80.0, 120.0)   # s
const MIXING_INTERVAL_S = 0.5              # s, sampling interval of the mixed-layer thickness
const PROGRESS_INTERVAL_S = 10.0           # s, simulated time between progress lines
const MIXED_FRACTION_RANGE = (0.05, 0.95)  # -, heavy-water fraction that counts as mixed
const TVD2_CFL_LIMIT = 0.5                 # -, monotonicity limit of tvd2 transport (CFL_x + CFL_z)
const ANIMATION_FPS = 10                   # frames/s of the GIF
const FRAME_SIZE_PX = (640, 340)           # px, one animation frame
const FINAL_SIZE_PX = (900, 480)           # px, final density
const SNAPSHOT_SIZE_PX = (1200, 560)       # px, 2 x 3 snapshot panel
const MIXING_SIZE_PX = (720, 380)          # px, mixed-layer figure
const FIGURE_MARGIN_MM = 4Plots.mm         # margin so that labels are not clipped
const COLORMAP = :viridis                  # same colormap as plot_state in src/visualization.jl
const DENSITY_LABEL = "Density (kg/m³)"

# WHY: anchored at this file, not the cwd, so outputs land in the project's output_barotropic/.
const OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "output_barotropic", "kelvin_helmholtz_figures")

"""
    density_heatmap(rho, cfg; title="", kwargs...)

xz heatmap of the cell-centred density `rho` (kg/m³, nz x nx), colour scale fixed to the two layer densities.
"""
function density_heatmap(rho::AbstractMatrix{<:Real}, cfg::ModelConfig; title::AbstractString="", kwargs...)
    dx = cfg.length / cfg.nx    # m
    dz = cfg.depth / cfg.nz     # m
    return heatmap(
        range(0.5 * dx, cfg.length - 0.5 * dx, length=cfg.nx),   # m, cell centres
        range(0.5 * dz, cfg.depth - 0.5 * dz, length=cfg.nz),    # m
        rho;
        color=COLORMAP,
        clims=(KH_RHO_TOP, KH_RHO_BOTTOM),
        xlabel="x (m)",
        ylabel="z (m)",
        colorbar_title=DENSITY_LABEL,
        title=title,
        xlims=(0.0, cfg.length),
        ylims=(0.0, cfg.depth),
        aspect_ratio=:equal,
        kwargs...,
    )
end

"""
    mixed_layer_thickness(rho, dx, dz) -> Float64

Thickness (m) of the mixed layer: area of the cells whose heavy-water fraction
c = (rho - rho_top)/(rho_bottom - rho_top) lies in MIXED_FRACTION_RANGE, divided by the domain length.
`rho` in kg/m³, `dx`, `dz` in m. Zero for the sharp initial interface (one cell row at most).
"""
function mixed_layer_thickness(rho::AbstractMatrix{<:Real}, dx::Real, dz::Real)
    lo, hi = MIXED_FRACTION_RANGE
    n = 0
    @inbounds for idx in eachindex(rho)
        c = (rho[idx] - KH_RHO_TOP) / (KH_RHO_BOTTOM - KH_RHO_TOP)
        n += lo < c < hi
    end
    return n * dx * dz / (size(rho, 2) * dx)
end

mkpath(OUTPUT_DIR)
cfg = kelvin_helmholtz_config(; t_end=T_END_S, write_plots=false, write_animation=false, write_pressure_buoyancy_diagnostics=false)
model = Model(cfg)
initialize_kelvin_helmholtz_case!(model)
@printf("Kelvin-Helmholtz | %d x %d cells, dt=%.4f s, t_end=%.0f s, periodic x\n", cfg.nx, cfg.nz, cfg.dt, T_END_S)

frame_stride = round(Int, ANIMATION_INTERVAL_S / cfg.dt)
mixing_stride = round(Int, MIXING_INTERVAL_S / cfg.dt)
progress_stride = round(Int, PROGRESS_INTERVAL_S / cfg.dt)
snapshot_steps = Dict(round(Int, t / cfg.dt) => t for t in SNAPSHOT_TIMES_S)

times = [0.0]                                                            # s
mixing = [mixed_layer_thickness(model.state.rho, model.dx, model.dz)]    # m
snapshots = Dict{Float64,Matrix{Float64}}(0.0 => copy(model.state.rho))
anim = Animation()
frame(anim, density_heatmap(model.state.rho, cfg; title="t = 0 s", size=FRAME_SIZE_PX))
max_cfl = 0.0   # -, running max of CFL_x + CFL_z

t0 = time()
for n in 1:cfg.nt
    step_forward!(model)
    t = n * cfg.dt
    # PERF: cfl_numbers scans u and w once (~1 ms) against ~70 ms per step.
    global max_cfl = max(max_cfl, cfl_numbers(model.state.u, model.state.w, cfg.dt, model.dx, model.dz)[3])
    if n % mixing_stride == 0
        push!(times, t)
        push!(mixing, mixed_layer_thickness(model.state.rho, model.dx, model.dz))
    end
    if n % frame_stride == 0
        frame(anim, density_heatmap(model.state.rho, cfg; title=@sprintf("t = %.0f s", t), size=FRAME_SIZE_PX))
    end
    if haskey(snapshot_steps, n)
        snapshots[snapshot_steps[n]] = copy(model.state.rho)
    end
    if n % progress_stride == 0
        @printf("t=%6.1f s | mixed layer %.3f m | max|u|=%.3f m/s | max|w|=%.3f m/s | max CFL so far=%.3f | wall=%.0f s\n",
            t, mixing[end], maximum(abs, model.state.u), maximum(abs, model.state.w), max_cfl, time() - t0)
    end
end
@printf("CASE=kelvin_helmholtz FULL_WALL_SECONDS=%.2f | max CFL=%.3f (tvd2 limit %.1f) | density range [%.4f, %.4f] kg/m^3\n",
    time() - t0, max_cfl, TVD2_CFL_LIMIT, minimum(model.state.rho), maximum(model.state.rho))
if max_cfl >= TVD2_CFL_LIMIT
    @warn "CFL number reached the tvd2 monotonicity limit; reduce dt" max_cfl
end

stem = joinpath(OUTPUT_DIR, "kelvin_helmholtz")
# Order: raw data first, so that a failing figure cannot cost the run.
npzwrite(stem * ".npz", Dict("t_s" => times, "mixed_layer_m" => mixing, "rho_final" => copy(model.state.rho)))
gif(anim, stem * ".gif"; fps=ANIMATION_FPS)
savefig(density_heatmap(model.state.rho, cfg; title=@sprintf("t = %.0f s", T_END_S), size=FINAL_SIZE_PX,
    left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM), stem * "_final.png")
panels = [density_heatmap(snapshots[t], cfg; title=@sprintf("t = %.0f s", t), colorbar=false) for t in SNAPSHOT_TIMES_S]
savefig(plot(panels...; layout=(2, 3), size=SNAPSHOT_SIZE_PX, left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM),
    stem * "_snapshots.png")
savefig(plot(times, mixing; xlabel="t (s)", ylabel="mixed-layer thickness (m)", legend=false, linewidth=2,
    size=MIXING_SIZE_PX, left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM), stem * "_mixing.png")
println("Saved: $(abspath(stem)).{npz,gif}, _final.png, _snapshots.png, _mixing.png")
