# Rayleigh-Taylor instability: density animations, snapshot panels and linear-growth check for the cases in cases.jl.
#
# For every case the script writes to output_barotropic/rayleigh_taylor_figures/:
# - rayleigh_taylor_<case>.gif            density animation (fixed colour scale, elapsed time as title)
# - rayleigh_taylor_<case>_snapshots.png  density at SNAPSHOT_TIMES_S
# - rayleigh_taylor_<case>.npz            amplitude time series and final density (restyle without rerunning)
# and, after all cases, rayleigh_taylor_growth.png: interface-mode amplitude a(t)/a0 against sigma*t with the linear
# theory a0*cosh(sigma t) (see linear_growth_rate in cases.jl).
# The figures in figures/ are copies of these files.
#
# Usage (any working directory):
#   julia --project=<path to the repository> experiments/rayleigh_taylor/plot_rayleigh_taylor.jl [case ...]
# Without case names both cases run ("reference", "fast"). Run time: ~3 min (reference) and ~2 min (fast).

# WHY: batch run without a display; same GR workstation choice as src/visualization.jl.
get!(ENV, "GKSwstype", "100")

using SweTransport2DV
using Plots
using NPZ
using Printf

include(joinpath(@__DIR__, "cases.jl"))

# Per case: simulated duration (s), time between animation frames (s), times of the snapshot panels (s).
# WHY 60 s for the reference case (production script: 25 s): its e-folding time is ~15 s, so the instability needs
# ~50 s to become nonlinear; the fast case overturns after ~12 s and is mixed by 25 s.
const PLOT_SETTINGS = Dict(
    "reference" => (t_end=60.0, frame_interval=1.0, snapshot_times=(0.0, 30.0, 45.0, 60.0)),
    "fast" => (t_end=25.0, frame_interval=0.5, snapshot_times=(0.0, 8.0, 14.0, 25.0)),
)
const AMPLITUDE_INTERVAL_S = 0.1          # s, sampling interval of the interface-mode amplitude
const PROGRESS_INTERVAL_S = 5.0           # s, simulated time between progress lines
const ANIMATION_FPS = 8                   # frames/s of the GIFs
const FRAME_SIZE_PX = (640, 340)          # px, width x height of one animation frame
const SNAPSHOT_SIZE_PX = (1000, 560)      # px, 2 x 2 snapshot panel
const GROWTH_SIZE_PX = (720, 440)         # px, growth figure
const FIGURE_MARGIN_MM = 4Plots.mm        # margin so that labels are not clipped
const COLORMAP = :viridis                 # same colormap as plot_state in src/visualization.jl
const DENSITY_LABEL = "Density (kg/m³)"

const SELECTED_CASES = isempty(ARGS) ? ["reference", "fast"] : ARGS

# WHY: anchored at this file, not the cwd, so outputs land in the project's output_barotropic/.
const OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "output_barotropic", "rayleigh_taylor_figures")

"""
    density_heatmap(rho, case_name, cfg; title="", kwargs...)

xz heatmap of the cell-centred density `rho` (kg/m³, nz x nx) with the colour scale fixed to the two initial layer
densities of the case, so that all frames and panels are comparable.
"""
function density_heatmap(rho::AbstractMatrix{<:Real}, case_name::AbstractString, cfg::ModelConfig; title::AbstractString="", kwargs...)
    case = RT_CASES[case_name]
    dx = cfg.length / cfg.nx    # m
    dz = cfg.depth / cfg.nz     # m
    return heatmap(
        range(0.5 * dx, cfg.length - 0.5 * dx, length=cfg.nx),   # m, cell centres
        range(0.5 * dz, cfg.depth - 0.5 * dz, length=cfg.nz),    # m
        rho;
        color=COLORMAP,
        clims=(case.rho_bottom, case.rho_top),
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

mkpath(OUTPUT_DIR)
growth_series = Dict{String,Any}()

for case_name in SELECTED_CASES
    haskey(RT_CASES, case_name) || error("unknown case '$(case_name)'; available: $(join(keys(RT_CASES), ", "))")
    case = RT_CASES[case_name]
    settings = PLOT_SETTINGS[case_name]
    nt = round(Int, settings.t_end / case.dt)
    cfg = rayleigh_taylor_config(case_name; nt=nt, write_plots=false, write_animation=false, write_pressure_buoyancy_diagnostics=false)
    model = Model(cfg)
    initialize_rayleigh_taylor_case!(model, case_name)

    sigma = linear_growth_rate(case_name; g=cfg.g)   # 1/s
    atwood = (case.rho_top - case.rho_bottom) / (case.rho_top + case.rho_bottom)
    println("\n=== Rayleigh-Taylor, case '$(case_name)', dt=$(cfg.dt) s, t_end=$(settings.t_end) s ===")
    @printf("Atwood=%.3e | k=%.3f 1/m | linear growth rate sigma=%.4f 1/s (e-folding time %.1f s)\n",
        atwood, 2pi * case.perturbation_wavenumber / cfg.length, sigma, 1.0 / sigma)

    frame_stride = round(Int, settings.frame_interval / cfg.dt)
    amplitude_stride = round(Int, AMPLITUDE_INTERVAL_S / cfg.dt)
    progress_stride = round(Int, PROGRESS_INTERVAL_S / cfg.dt)
    snapshot_steps = Dict(round(Int, t / cfg.dt) => t for t in settings.snapshot_times)

    times = [0.0]                                                                   # s
    amplitudes = [interface_mode_amplitude(model.state.rho, case_name, model.dx, model.dz)]   # m
    snapshots = Dict{Float64,Matrix{Float64}}(0.0 => copy(model.state.rho))
    anim = Animation()
    frame(anim, density_heatmap(model.state.rho, case_name, cfg; title="t = 0 s", size=FRAME_SIZE_PX))
    max_cfl = 0.0   # -, running max of CFL_x + CFL_z
    max_fr = 0.0    # -, running max of the Froude number (limiter threshold 0.9, limiter off)

    t0 = time()
    for n in 1:nt
        step_forward!(model)
        t = n * cfg.dt
        # PERF: cfl_numbers / froude_number scan the velocity once (~1 ms) against ~12 ms per step.
        max_cfl = max(max_cfl, cfl_numbers(model.state.u, model.state.w, cfg.dt, model.dx, model.dz)[3])
        max_fr = max(max_fr, froude_number(model.state.u, model.state.rho, cfg.rho0, cfg.g, cfg.depth))
        if n % amplitude_stride == 0
            push!(times, t)
            push!(amplitudes, interface_mode_amplitude(model.state.rho, case_name, model.dx, model.dz))
        end
        if n % frame_stride == 0
            frame(anim, density_heatmap(model.state.rho, case_name, cfg; title=@sprintf("t = %.0f s", t), size=FRAME_SIZE_PX))
        end
        if haskey(snapshot_steps, n)
            snapshots[snapshot_steps[n]] = copy(model.state.rho)
        end
        if n % progress_stride == 0
            @printf("t=%5.1f s | a=%.4f m (theory %.4f m) | max CFL=%.3f | max Fr=%.2f | wall=%.0f s\n",
                t, amplitudes[end], amplitudes[1] * cosh(sigma * t), max_cfl, max_fr, time() - t0)
        end
    end
    @printf("CASE=rayleigh_taylor_%s FULL_WALL_SECONDS=%.2f | max CFL=%.3f | max Fr=%.2f | density range [%.4f, %.4f] kg/m^3\n",
        case_name, time() - t0, max_cfl, max_fr, minimum(model.state.rho), maximum(model.state.rho))

    stem = joinpath(OUTPUT_DIR, "rayleigh_taylor_$(case_name)")
    npzwrite(stem * ".npz", Dict("t_s" => times, "amplitude_m" => amplitudes, "sigma_per_s" => sigma, "rho_final" => copy(model.state.rho)))
    gif(anim, stem * ".gif"; fps=ANIMATION_FPS)

    panels = [density_heatmap(snapshots[t], case_name, cfg; title=@sprintf("t = %.0f s", t), colorbar=false) for t in settings.snapshot_times]
    savefig(plot(panels...; layout=(2, 2), size=SNAPSHOT_SIZE_PX, left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM), stem * "_snapshots.png")
    println("Saved: $(abspath(stem)).{gif,npz} and _snapshots.png")

    growth_series[case_name] = (t=times, a=amplitudes, sigma=sigma)
end

# Growth figure: normalised amplitude against sigma*t collapses both cases onto the linear solution cosh(sigma t).
if !isempty(growth_series)
    s_max = maximum(g.sigma * g.t[end] for g in values(growth_series))
    s_line = range(0.0, s_max, length=200)
    fig = plot(s_line, cosh.(s_line); label="linear theory, cosh(σt)", color=:black, linestyle=:dash, yscale=:log10,
        xlabel="σt (-)", ylabel="a / a₀ (-)", legend=:topleft, size=GROWTH_SIZE_PX, left_margin=FIGURE_MARGIN_MM,
        bottom_margin=FIGURE_MARGIN_MM, ylims=(0.5, 30.0))
    for (case_name, g) in sort(collect(growth_series); by=first, rev=true)
        plot!(fig, g.sigma .* g.t, max.(g.a ./ g.a[1], 1e-3); label=@sprintf("model, %s (σ = %.3f 1/s)", case_name, g.sigma), linewidth=2)
    end
    growth_path = joinpath(OUTPUT_DIR, "rayleigh_taylor_growth.png")
    savefig(fig, growth_path)
    println("Saved: $(abspath(growth_path))")
end
