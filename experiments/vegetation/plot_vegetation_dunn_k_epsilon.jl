# Submerged canopies of Dunn, Lopez & Garcia (1996) with the k-epsilon closure (src/turbulence.jl, vegetation terms of
# Lopez & Garcia 1997): rigid runs 1-12 and flexible runs 13-18 (computed as rigid canopies of the deflected height, as
# Lopez & Garcia did), periodic channel per run, measured bulk drag coefficient C_DB of each run, smooth flume bed
# (z0 = 5e-6 m). Compared with
# - the measured horizontally averaged profiles of velocity and Reynolds stress (Appendix C of the report,
#   DUNN1996_PROFILES in cases.jl) and the measured mean velocity Q/(b H);
# - the computed k-epsilon profiles of Lopez & Garcia (1997), Figs. 6.1, 6.2, 6.4, 6.5 (runs 1, 9, 13, 17;
#   LG1997_PROFILES in cases.jl).
#
# The script writes to output_barotropic/vegetation_figures/:
# - vegetation_dunn_velocity.png   velocity profiles of all 18 runs
# - vegetation_dunn_stress.png     Reynolds-stress profiles of all 18 runs, with the 2D momentum balance g S (H - z)
# - vegetation_dunn_k_epsilon.npz  model profiles and depth means
# and prints the depth-mean velocities for C_D = C_DB and C_D = 1 and the profile ratios. The figures in figures/ are
# copies.
#
# Usage: julia --project=<path to the repository> experiments/vegetation/plot_vegetation_dunn_k_epsilon.jl
# Run time: about 6 min.

# WHY: batch run without a display; same GR workstation choice as src/visualization.jl.
get!(ENV, "GKSwstype", "100")

using SweTransport2DV
using Plots
using NPZ
using Printf
using Statistics

include(joinpath(@__DIR__, "cases.jl"))

const RUNS = 1:18
const FIGURE_SIZE_PX = (1800, 1500)
const FIGURE_MARGIN_MM = 3Plots.mm
const OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "output_barotropic", "vegetation_figures")

mkpath(OUTPUT_DIR)
out = Dict{String,Any}()
velocity_panels = []
stress_panels = []
ratios = Dict(:measured => Float64[], 1.0 => Float64[])
for run in RUNS
    r = dunn1996_k_epsilon(run)
    d = r.run
    p = DUNN1996_PROFILES[run]
    kind = run <= 12 ? "rigid" : "flexible"
    push!(ratios[:measured], mean(r.u) / d.U)
    if run <= 12
        push!(ratios[1.0], mean(dunn1996_k_epsilon(run; drag_coefficient=1.0).u) / d.U)
    end
    # WHY: explicit x limits; hline! otherwise widens every panel to the same range
    pu = plot(xlabel="u (m/s)", ylabel="z (m)", title="run $run ($kind)", legend=run == 1 ? :topleft : false, ylims=(0, d.H),
        xlims=(0, 1.1 * max(maximum(r.u), maximum(p.u), haskey(LG1997_PROFILES, run) ? maximum(LG1997_PROFILES[run].u) : 0.0)),
        titlefontsize=9, guidefontsize=8, tickfontsize=7, legendfontsize=6)
    hline!(pu, [d.hp]; color=:green, linestyle=:dash, label="canopy top")
    plot!(pu, r.u, r.z; color=:black, linewidth=2, label="k-epsilon (this model)")
    scatter!(pu, p.u, p.z; color=:steelblue, markersize=3, label="measured (Dunn et al. 1996)")
    vline!(pu, [d.U]; color=:steelblue, linestyle=:dot, label="measured Q/(b H)")
    pt = plot(xlabel="-u'w' (m²/s²)", ylabel="z (m)", title="run $run ($kind)", legend=run == 1 ? :topright : false, ylims=(0, d.H),
        xlims=(0, 1.15 * max(maximum(r.tau), maximum(p.uw))),
        titlefontsize=9, guidefontsize=8, tickfontsize=7, legendfontsize=6)
    hline!(pt, [d.hp]; color=:green, linestyle=:dash, label="canopy top")
    zl = range(d.hp, d.H, length=10)
    plot!(pt, 9.81 * d.S .* (d.H .- zl), zl; color=:gray60, label="g S (H - z)")
    plot!(pt, r.tau, r.zf; color=:black, linewidth=2, label="k-epsilon (this model)")
    scatter!(pt, p.uw, p.z; color=:steelblue, markersize=3, label="measured")
    if haskey(LG1997_PROFILES, run)
        g = LG1997_PROFILES[run]
        plot!(pu, g.u, g.uz; color=:darkorange, linestyle=:dash, linewidth=2, label="k-epsilon, Lopez & Garcia (1997)")
        plot!(pt, g.tau, g.tz; color=:darkorange, linestyle=:dash, linewidth=2, label="k-epsilon, Lopez & Garcia (1997)")
        ru = linear_interpolation.(g.uz, Ref(r.z), Ref(r.u)) ./ g.u
        rt = linear_interpolation.(g.tz, Ref(r.zf), Ref(r.tau)) ./ g.tau
        @printf("run %2d vs Lopez & Garcia: velocity ratio %.2f-%.2f, stress ratio %.2f-%.2f (inside the canopy), peak %.2f\n", run,
            extrema(ru)..., extrema(rt[g.tz .< d.hp])..., maximum(r.tau) / maximum(g.tau))
    end
    inside = p.z .< d.hp
    _, itop = findmin(abs.(p.z .- d.hp))
    um = linear_interpolation.(p.z, Ref(r.z), Ref(r.u))
    shape = (um ./ um[itop]) ./ (p.u ./ p.u[itop])
    @printf("run %2d: U model/measured %.3f, velocity model/measured %.2f-%.2f, shape u/u(h_p) model/measured %.2f-%.2f (canopy)\n",
        run, mean(r.u) / d.U, extrema(um ./ p.u)..., extrema(shape[inside])...)
    push!(velocity_panels, pu)
    push!(stress_panels, pt)
    out["run$(run)_z"] = r.z; out["run$(run)_u"] = r.u; out["run$(run)_zf"] = r.zf; out["run$(run)_tau"] = r.tau
end
for (cd, v) in ratios
    @printf("C_D = %s: U model/measured mean %.3f, range %.3f-%.3f (%d runs)\n", string(cd), mean(v), extrema(v)..., length(v))
end
out["U_ratio_measured_cd"] = ratios[:measured]
out["U_ratio_cd1"] = ratios[1.0]

savefig(plot(velocity_panels...; layout=(3, 6), size=FIGURE_SIZE_PX, margin=FIGURE_MARGIN_MM,
    plot_title="Dunn et al. (1996): velocity, k-epsilon with measured C_DB (flexible runs as rigid canopies of the deflected height)",
    plot_titlefontsize=11), joinpath(OUTPUT_DIR, "vegetation_dunn_velocity.png"))
savefig(plot(stress_panels...; layout=(3, 6), size=FIGURE_SIZE_PX, margin=FIGURE_MARGIN_MM,
    plot_title="Dunn et al. (1996): Reynolds stress (the measured stress above the canopy is reduced by secondary currents)",
    plot_titlefontsize=11), joinpath(OUTPUT_DIR, "vegetation_dunn_stress.png"))
npzwrite(joinpath(OUTPUT_DIR, "vegetation_dunn_k_epsilon.npz"), out)
println("figures written to ", OUTPUT_DIR)
