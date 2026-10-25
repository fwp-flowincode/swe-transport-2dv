# Reconfiguration of a single flexible stem in a uniform current (quasi-static elastica of src/vegetation.jl): the
# reconfiguration number R = F / F_rigid (= effective length l_e/l, Luhar & Nepf 2011) against the Cauchy number
# Ca = rho C_D b U² l³ / (2 EI) for three buoyancy numbers B = B_l l³ / EI, and the stem shapes.
#
# The script writes to output_barotropic/vegetation_figures/:
# - vegetation_reconfiguration.png   R(Ca) with the rigid limit and the Ca^(-1/3) law; stem shapes at Ca = 1 ... 10^4
# - vegetation_reconfiguration.npz   the curves
# The figure in figures/ is a copy.
#
# Usage: julia --project=<path to the repository> experiments/vegetation/plot_vegetation_reconfiguration.jl
# Run time: a few seconds.

# WHY: batch run without a display; same GR workstation choice as src/visualization.jl.
get!(ENV, "GKSwstype", "100")

using SweTransport2DV
using Plots
using NPZ
using Printf

include(joinpath(@__DIR__, "cases.jl"))

const CAUCHY = 10.0 .^ range(-1, 4, length=51)     # -
const BUOYANCY_NUMBERS = (0.0, 10.0, 100.0)        # -
const SHAPE_CAUCHY = (1.0, 10.0, 100.0, 1000.0, 10000.0)
const N_SEGMENTS = 80                              # -
const FIGURE_SIZE_PX = (1100, 460)
const FIGURE_MARGIN_MM = 5Plots.mm
const OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "output_barotropic", "vegetation_figures")

mkpath(OUTPUT_DIR)
curves = Dict{Float64,Vector{Float64}}()
for bn in BUOYANCY_NUMBERS
    theta = zeros(N_SEGMENTS)                      # continuation in Ca: warm start from the previous posture
    curves[bn] = [first(reconfiguration_number(ca; n_segments=N_SEGMENTS, buoyancy_number=bn, theta=theta)) for ca in CAUCHY]
end
r1000 = first(reconfiguration_number(1000.0; n_segments=N_SEGMENTS))
slope = log(first(reconfiguration_number(3000.0; n_segments=N_SEGMENTS)) / first(reconfiguration_number(300.0; n_segments=N_SEGMENTS))) / log(10.0)
@printf("R(Ca=0.1) = %.5f | R(Ca=1000) = %.5f, R Ca^(1/3) = %.3f | slope between Ca = 300 and 3000: %.4f\n",
    curves[0.0][1], r1000, r1000 * 1000^(1 / 3), slope)

p = plot(xscale=:log10, yscale=:log10, xlabel="Cauchy number Ca = ρ C_D b U² l³ / (2 EI)", ylabel="R = F / F_rigid = l_e / l",
    legend=:bottomleft, title="Drag reduction by reconfiguration", ylims=(0.02, 1.5))
for (bn, color) in zip(BUOYANCY_NUMBERS, (:black, :steelblue, :darkorange))
    plot!(p, CAUCHY, curves[bn]; color=color, linewidth=2, label="buoyancy number B = $(Int(bn))")
end
plot!(p, CAUCHY, r1000 .* (CAUCHY ./ 1000) .^ (-1 / 3); color=:gray, linestyle=:dash, label="∝ Ca^(-1/3) (drag ∝ U^(4/3))")
hline!(p, [1.0]; color=:gray, linestyle=:dot, label="rigid")

q = plot(xlabel="x / l", ylabel="z / l", aspect_ratio=:equal, xlims=(-0.05, 1.05), ylims=(0, 1.05), legend=:topright,
    title="Stem shapes (B = 0, flow from the left)")
for (ca, color) in zip(SHAPE_CAUCHY, cgrad(:viridis, length(SHAPE_CAUCHY); categorical=true))
    _, theta = reconfiguration_number(ca; n_segments=N_SEGMENTS)
    x = [0.0; cumsum(sin.(theta))] ./ N_SEGMENTS
    z = [0.0; cumsum(cos.(theta))] ./ N_SEGMENTS
    plot!(q, x, z; color=color, linewidth=2, label=@sprintf("Ca = %g", ca))
end
savefig(plot(p, q; layout=(1, 2), size=FIGURE_SIZE_PX, left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM),
    joinpath(OUTPUT_DIR, "vegetation_reconfiguration.png"))
npzwrite(joinpath(OUTPUT_DIR, "vegetation_reconfiguration.npz"),
    merge(Dict("cauchy" => collect(CAUCHY)), Dict("R_B$(Int(bn))" => curves[bn] for bn in BUOYANCY_NUMBERS)))
println("Saved: ", abspath(joinpath(OUTPUT_DIR, "vegetation_reconfiguration.png")))
