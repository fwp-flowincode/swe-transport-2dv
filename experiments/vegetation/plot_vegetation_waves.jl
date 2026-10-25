# Regular waves through rigid canopies against the wave-height decay H0 / (1 + beta x) of Dalrymple, Kirby & Hwang (1984):
# 1. shallow water (kh = 0.2), emergent canopy, hydrostatic and non-hydrostatic;
# 2. intermediate depth (kh = 1), emergent canopy, non-hydrostatic with the dispersive surface condition (section 15);
# 3. intermediate depth (kh = 1), submerged canopy (stem height h/2), same mode.
# Each case also runs without canopy; the ratio of the two removes the small numerical damping of the implicit free
# surface.
#
# The script writes to output_barotropic/vegetation_figures/:
# - vegetation_waves.png   wave height with / without canopy along the canopy, Dalrymple's decay, and the bare run
# - vegetation_waves.npz   wave heights
# The figure in figures/ is a copy.
#
# Usage: julia --project=<path to the repository> experiments/vegetation/plot_vegetation_waves.jl
# Run time: about 20 min.

# WHY: batch run without a display; same GR workstation choice as src/visualization.jl.
get!(ENV, "GKSwstype", "100")

using SweTransport2DV
using Plots
using NPZ
using Printf

include(joinpath(@__DIR__, "cases.jl"))

const DEPTH = 0.5               # m
const CASES = [
    # title, kh, wave height (m), frontal area (1/m), alpha, modes (use_hydrostatic, dispersive), canopy (m), periods
    ("shallow, emergent: kh = 0.2, a = 10 1/m", 0.2, 0.0025, 10.0, 1.0, [(true, false), (false, false)], (10.0, 70.0), 20),
    ("intermediate, emergent: kh = 1, a = 1 1/m", 1.0, 0.025, 1.0, 1.0, [(false, true)], (5.0, 20.0), 60),
    ("intermediate, submerged α = 0.5: kh = 1, a = 2 1/m", 1.0, 0.025, 2.0, 0.5, [(false, true)], (5.0, 20.0), 60),
]
const STEPS_PER_PERIOD = Dict(0.2 => 4000, 1.0 => 3000)   # implicit free-surface damping 2 pi^2/N per period
const FIGURE_SIZE_PX = (1300, 420)
const FIGURE_MARGIN_MM = 5Plots.mm
const MODE_LABEL = Dict((true, false) => "hydrostatic", (false, false) => "non-hydrostatic", (false, true) => "non-hydrostatic, dispersive")
const MODE_COLOR = Dict((true, false) => :steelblue, (false, false) => :crimson, (false, true) => :crimson)
const OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "output_barotropic", "vegetation_figures")

mkpath(OUTPUT_DIR)
panels = []
out = Dict{String,Any}()
for (ic, (title, kh, height, area, alpha, modes, canopy, periods)) in enumerate(CASES)
    p = plot(xlabel="x − x₀ (m) into the canopy", ylabel="H / H₀", title=title, titlefontsize=10, legend=:bottomleft, ylims=(0.6, 1.05))
    for mode in modes
        setup = (; depth=DEPTH, kh=kh, wave_height=height, alpha=alpha, use_hydrostatic=mode[1], dispersive=mode[2], canopy=canopy,
            steps_per_period=STEPS_PER_PERIOD[kh], periods=periods)
        t0 = time()
        bare = run_wave_flume(; frontal_area=0.0, setup...)
        vegetated = run_wave_flume(; frontal_area=area, setup...)
        i0 = findfirst(>=(canopy[1]), vegetated.x)
        i1 = findlast(<=(canopy[2]), vegetated.x)
        x = vegetated.x[i0:i1] .- vegetated.x[i0]
        h_veg = vegetated.H[i0:i1] ./ vegetated.H[i0]
        h_bare = bare.H[i0:i1] ./ bare.H[i0]
        beta = dalrymple_decay_coefficient(; cd=1.0, a=area, H0=vegetated.H[i0], k=vegetated.k, h=DEPTH, alpha=alpha)
        reference = 1 ./ (1 .+ beta .* x)
        @printf("case %d, %s: H/H0 at the canopy end %.4f (without canopy %.4f, ratio %.4f), Dalrymple %.4f; max deviation of the ratio %.2f %% (%.0f s)\n",
            ic, MODE_LABEL[mode], h_veg[end], h_bare[end], h_veg[end] / h_bare[end], reference[end],
            100 * maximum(abs, (h_veg ./ h_bare) ./ reference .- 1), time() - t0)
        plot!(p, x, h_veg ./ h_bare; color=MODE_COLOR[mode], linewidth=2, linestyle=mode[1] ? :solid : :dash,
            label="with / without canopy, $(MODE_LABEL[mode])")
        plot!(p, x, h_bare; color=MODE_COLOR[mode], linewidth=1, linestyle=:dot, alpha=0.6, label="without canopy (numerical damping)")
        mode == first(modes) && plot!(p, x, reference; color=:black, linewidth=3, alpha=0.4, label="Dalrymple et al. (1984)")
        out["case$(ic)_$(MODE_LABEL[mode])_x"] = x
        out["case$(ic)_$(MODE_LABEL[mode])_H_veg"] = h_veg
        out["case$(ic)_$(MODE_LABEL[mode])_H_bare"] = h_bare
    end
    push!(panels, p)
end
savefig(plot(panels...; layout=(1, 3), size=FIGURE_SIZE_PX, left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM),
    joinpath(OUTPUT_DIR, "vegetation_waves.png"))
npzwrite(joinpath(OUTPUT_DIR, "vegetation_waves.npz"), out)
println("Saved: ", abspath(joinpath(OUTPUT_DIR, "vegetation_waves.png")))
