# Flexible model blades in uniform flow against the laboratory data of Luhar & Nepf (2011), Limnol. Oceanogr. 56(6),
# 2003-2017: our stem model (quasi-static elastica with drag and buoyancy, src/vegetation.jl) for their HDPE
# (stiffness-dominated) and silicone-foam (buoyancy-dominated) blades. Measured forces of the 5 cm and 25 cm blades are
# read from their Fig. 4 (LN2011_FORCES in cases.jl).
#
# The script writes to output_barotropic/vegetation_figures/:
# - vegetation_luhar_nepf.png   drag against velocity (model and data), effective length against the Cauchy number for
#                               all 80 cases of their Table 1, computed blade shapes of the 25 cm blades
# - vegetation_luhar_nepf.npz   model forces and effective lengths
# The figure in figures/ is a copy.
#
# Usage: julia --project=<path to the repository> experiments/vegetation/plot_vegetation_luhar_nepf.jl
# Run time: a few seconds.

# WHY: batch run without a display; same GR workstation choice as src/visualization.jl.
get!(ENV, "GKSwstype", "100")

using SweTransport2DV
using Plots
using NPZ
using Printf

include(joinpath(@__DIR__, "cases.jl"))

const U_CURVE = range(0.01, 0.35, length=69)        # m/s, model curves
const LOAD_CELL_ACCURACY = 0.10                      # -, relative, from the paper
const SHAPE_VELOCITIES = (0.16, 0.32)                # m/s, as in their Fig. 4c-f
const COLORS = Dict(0.05 => :black, 0.10 => :steelblue, 0.15 => :darkgreen, 0.20 => :darkorange, 0.25 => :gray45)
const MARKERS = Dict(0.05 => :rect, 0.25 => :circle)
const FIGURE_SIZE_PX = (1400, 860)
const FIGURE_MARGIN_MM = 5Plots.mm
const OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "output_barotropic", "vegetation_figures")

mkpath(OUTPUT_DIR)
out = Dict{String,Any}()
force_panels = []
for material in ("HDPE", "foam")
    p = plot(xlabel="U (m/s)", ylabel="F_x (N)", title="$material blades: drag against velocity", legend=:topleft, xlims=(0, 0.35))
    for len in (0.05, 0.25)
        curve = [luhar_nepf_blade(material, len, u).force for u in U_CURVE]
        plot!(p, U_CURVE, curve; color=COLORS[len], linewidth=2, label="model, l = $(Int(100len)) cm")
        measured = LN2011_FORCES[(material, len)]
        keep = .!isnan.(measured)
        scatter!(p, collect(LN2011_VELOCITIES)[keep], measured[keep]; yerror=LOAD_CELL_ACCURACY .* measured[keep], color=COLORS[len],
            markershape=MARKERS[len], markersize=5, label="measured, l = $(Int(100len)) cm")
        model_at = [luhar_nepf_blade(material, len, u).force for u in LN2011_VELOCITIES]
        @printf("%s l = %2d cm: model / measured = %s\n", material, Int(100len), join([isnan(m) ? "  -  " : @sprintf("%.2f", f / m) for (f, m) in zip(model_at, measured)], " "))
        out["force_$(material)_$(Int(100len))cm"] = curve
    end
    push!(force_panels, p)
end

length_panels = []
for material in ("HDPE", "foam")
    p = plot(xscale=:log10, yscale=:log10, xlabel="Cauchy number Ca", ylabel="l_e / l", ylims=(0.03, 1.5), legend=:bottomleft,
        title="$material: effective length, all lengths of their Table 1")
    for len in LN2011_LENGTHS
        res = [luhar_nepf_blade(material, len, u) for u in range(0.02, 0.32, length=40)]
        bn = res[1].buoyancy
        plot!(p, [r.cauchy for r in res], [r.effective_length for r in res]; color=COLORS[len], linewidth=2,
            label=@sprintf("model, l = %d cm (B = %.3g)", Int(100len), bn))
        if haskey(LN2011_FORCES, (material, len))
            measured = LN2011_FORCES[(material, len)]
            keep = .!isnan.(measured)
            ca = [luhar_nepf_blade(material, len, u).cauchy for u in LN2011_VELOCITIES]
            le = measured ./ (0.5 .* LN2011_RHO .* LN2011_CD .* LN2011_WIDTH .* len .* collect(LN2011_VELOCITIES) .^ 2)
            scatter!(p, ca[keep], le[keep]; color=COLORS[len], markershape=MARKERS[len], markersize=5, label="measured, l = $(Int(100len)) cm")
        end
    end
    push!(length_panels, p)
end

shape = plot(xlabel="x (cm)", ylabel="z (cm)", aspect_ratio=:equal, title="Computed shapes of the 25 cm blades (flow from the left)",
    legend=:topleft, xlims=(-1, 26), ylims=(0, 26))
for material in ("HDPE", "foam"), u in SHAPE_VELOCITIES
    r = luhar_nepf_blade(material, 0.25, u)
    ds = 25.0 / length(r.theta)
    plot!(shape, [0.0; cumsum(ds .* sin.(r.theta))], [0.0; cumsum(ds .* cos.(r.theta))]; linewidth=2,
        color=material == "HDPE" ? :black : :darkorange, linestyle=u < 0.2 ? :dash : :solid,
        label=@sprintf("%s, U = %.2f m/s (h/l = %.2f)", material, u, sum(cos, r.theta) / length(r.theta)))
end

savefig(plot(force_panels..., length_panels..., shape, plot(; framestyle=:none, grid=false, showaxis=false, ticks=false);
    layout=(3, 2), size=(1400, 1250), left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM),
    joinpath(OUTPUT_DIR, "vegetation_luhar_nepf.png"))
npzwrite(joinpath(OUTPUT_DIR, "vegetation_luhar_nepf.npz"), merge(out, Dict("U_curve" => collect(U_CURVE))))
println("Saved: ", abspath(joinpath(OUTPUT_DIR, "vegetation_luhar_nepf.png")))
