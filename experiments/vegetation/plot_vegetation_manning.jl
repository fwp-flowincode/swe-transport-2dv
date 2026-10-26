# Flow resistance of a submerged rigid canopy against plant density: the numerical experiment of Lopez & Garcia (1997),
# WES Technical Report WRP-CP-10, Figs. 6.16-6.18 (S = 0.0036, h_p = 0.10 m, D = 6.4 mm, q = 1.12 m²/s, smooth bed,
# 0.16-1581 stems/m²), repeated with our k-epsilon closure (same vegetation terms) at their computed depths
# (lg1997_manning_case in cases.jl). Compared: the momentum transfer to the bed u*_bed/sqrt(g H S) (their Fig. 6.16)
# and the Manning coefficient n = H^(5/3) S^(1/2) / q at their depth (their Fig. 6.18), with the smooth-bed log law
# U/u* = (ln(E H u*/nu) - 1)/kappa (E = 9) for the unvegetated limit.
#
# The script writes to output_barotropic/vegetation_figures/:
# - vegetation_manning.png   bed shear ratio against H a, Manning n and discharge ratio against stem density
# - vegetation_manning.npz   model values
# and prints the table. The figure in figures/ is a copy.
#
# Usage: julia --project=<path to the repository> experiments/vegetation/plot_vegetation_manning.jl
# Run time: about 8 min.

# WHY: batch run without a display; same GR workstation choice as src/visualization.jl.
get!(ENV, "GKSwstype", "100")

using SweTransport2DV
using Plots
using NPZ
using Printf

include(joinpath(@__DIR__, "cases.jl"))

const DZ = 0.01                      # m (10 cells in the canopy)
const KAPPA_LG = 0.40                # -, von Karman constant of Lopez & Garcia (Appendix A)
const FIGURE_SIZE_PX = (1500, 480)
const FIGURE_MARGIN_MM = 6Plots.mm
const OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "output_barotropic", "vegetation_figures")

mkpath(OUTPUT_DIR)
ncase = length(LG1997_MANNING_H)
res = [lg1997_manning_case(i; dz=DZ) for i in 1:ncase]
lambda = LG1997_MANNING_LAMBDA
println(" stems/m²   H (m)    Ha     q (m²/s)  n model  n L&G   ratio   u*b/sqrt(gHS) model  L&G")
for (i, r) in enumerate(res)
    lg = i <= length(LG1997_BED_SHEAR) ? @sprintf("%.3f", LG1997_BED_SHEAR[i]) : "  -  "
    @printf("%9.3g  %.3f  %7.3g  %.3f    %.4f   %.4f  %.2f    %.3f                %s\n", lambda[i], r.H, r.Ha, r.q, r.n, LG1997_MANNING_N[i],
        r.n / LG1997_MANNING_N[i], r.bed_shear, lg)
end
# smooth-bed log law at their depths (no vegetation)
loglaw_n = map(LG1997_MANNING_H) do h
    us = sqrt(9.81 * h * LG1997_MANNING_S)
    u = us / KAPPA_LG * (log(LG1997_WALL_E * h * us / 1e-6) - 1)
    h^(2 / 3) * sqrt(LG1997_MANNING_S) / u
end

p1 = plot(xscale=:log10, xlabel="H a (-)", ylabel="u*_bed / sqrt(g H S)", title="Momentum transfer to the bed", legend=:bottomleft, ylims=(0, 1.05))
plot!(p1, LG1997_BED_SHEAR_HA, LG1997_BED_SHEAR; color=:darkorange, linestyle=:dash, linewidth=2, marker=:cross, label="Lopez & Garcia (1997), Fig. 6.16")
plot!(p1, [r.Ha for r in res], [r.bed_shear for r in res]; color=:black, linewidth=2, marker=:circle, label="this model")
p2 = plot(xscale=:log10, yscale=:log10, xlabel="stems per m²", ylabel="Manning n (s/m^(1/3))", title="Manning coefficient at their depth",
    legend=:topleft)
plot!(p2, lambda, LG1997_MANNING_N; color=:darkorange, linestyle=:dash, linewidth=2, marker=:cross, label="Lopez & Garcia (1997), Fig. 6.18")
plot!(p2, lambda, [r.n for r in res]; color=:black, linewidth=2, marker=:circle, label="this model")
hline!(p2, [loglaw_n[1]]; color=:gray60, linestyle=:dot, label="smooth-bed log law, no plants")
p3 = plot(xscale=:log10, xlabel="stems per m²", ylabel="q model / 1.12 m²/s", title="Discharge at their computed depth", legend=false)
plot!(p3, lambda, [r.q / LG1997_MANNING_Q for r in res]; color=:black, linewidth=2, marker=:circle)
hline!(p3, [1.0]; color=:gray60, linestyle=:dot)
savefig(plot(p1, p2, p3; layout=(1, 3), size=FIGURE_SIZE_PX, margin=FIGURE_MARGIN_MM,
    plot_title="Plant-density series of Lopez & Garcia (1997): S = 0.0036, h_p = 0.10 m, q = 1.12 m²/s", plot_titlefontsize=11),
    joinpath(OUTPUT_DIR, "vegetation_manning.png"))
npzwrite(joinpath(OUTPUT_DIR, "vegetation_manning.npz"), Dict("stems_per_m2" => lambda, "H" => [r.H for r in res],
    "q" => [r.q for r in res], "n" => [r.n for r in res], "bed_shear" => [r.bed_shear for r in res], "loglaw_n" => loglaw_n))
println("figure written to ", OUTPUT_DIR)
