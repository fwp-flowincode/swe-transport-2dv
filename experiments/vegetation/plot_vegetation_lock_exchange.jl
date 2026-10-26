# Lock exchange through emergent rigid canopies of increasing density (a = 0, 20, 80 1/m; H = 0.2 m, g' = 0.02 m/s²,
# stem diameter 6.4 mm, non-hydrostatic): front position against time (inertial x_f ~ t, drag-dominated x_f ~ t^(2/3)),
# density snapshots (block profile without canopy, triangular profile in the canopy; Tanino, Nepf & Kulis 2005) and an
# animation of the case without canopy against the dense canopy.
#
# The script writes to output_barotropic/vegetation_figures/:
# - vegetation_lock_exchange_front.png       front position (log-log) with the power laws
# - vegetation_lock_exchange_snapshots.png   density at SNAPSHOT_TIMES_S for the three canopies
# - vegetation_lock_exchange.gif             density, no canopy (top) against a = 80 1/m (bottom)
# - vegetation_lock_exchange.npz             front positions
# The figures in figures/ are copies (the GIF reduced in size).
#
# Usage: julia --project=<path to the repository> experiments/vegetation/plot_vegetation_lock_exchange.jl
# Run time: about 10 min.

# WHY: batch run without a display; same GR workstation choice as src/visualization.jl.
get!(ENV, "GKSwstype", "100")

using SweTransport2DV
using Plots
using NPZ
using Printf

include(joinpath(@__DIR__, "cases.jl"))

const FRONTAL_AREAS = (0.0, 20.0, 80.0)        # 1/m
const LENGTH, DEPTH = 8.0, 0.2                 # m
const REDUCED_GRAVITY = 0.02                   # m/s^2
const NX, NZ, DT = 800, 20, 0.01               # -, -, s
const T_END = 900.0                            # s
const SNAPSHOT_TIMES_S = (100.0, 300.0)        # s
const FRAME_INTERVAL_S = 10.0                  # s
const EXPONENT_WINDOW_S = (150.0, 400.0)       # s, before the dense fronts reach the wall
const COLORS = Dict(0.0 => :gray, 20.0 => :steelblue, 80.0 => :darkgreen)
const FIGURE_MARGIN_MM = 5Plots.mm
const OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "output_barotropic", "vegetation_figures")

mkpath(OUTPUT_DIR)
frame_times = Tuple(FRAME_INTERVAL_S:FRAME_INTERVAL_S:T_END)
runs = Dict{Float64,Any}()
for a in FRONTAL_AREAS
    t0 = time()
    runs[a] = run_canopy_lock_exchange(; frontal_area=a, use_hydrostatic=false, length=LENGTH, depth=DEPTH, reduced_gravity=REDUCED_GRAVITY,
        nx=NX, nz=NZ, dt=DT, t_end=T_END, snapshot_times=(SNAPSHOT_TIMES_S..., frame_times...))
    r = runs[a]
    @printf("a = %4.0f 1/m: front at 100/300 s = %.2f / %.2f m, exponents 10-50 s %.3f, 50-150 s %.3f, %.0f-%.0f s %.3f (%.0f s wall)\n",
        a, r.front[100], r.front[300], power_law_exponent(r.t, r.front, 10.0, 50.0), power_law_exponent(r.t, r.front, 50.0, 150.0),
        EXPONENT_WINDOW_S..., power_law_exponent(r.t, r.front, EXPONENT_WINDOW_S...), time() - t0)
end

# front position
p = plot(xscale=:log10, yscale=:log10, xlabel="t (s)", ylabel="front position from the gate (m)", legend=:topleft,
    title="Front position (H = 0.2 m, g' = 0.02 m/s²)", size=(760, 460),
    left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM, xlims=(5, T_END), ylims=(0.05, 5))
for a in FRONTAL_AREAS
    r = runs[a]
    keep = (r.front .> 0.02) .& (r.front .< LENGTH / 2 - 0.05)
    plot!(p, r.t[keep], r.front[keep]; color=COLORS[a], linewidth=2, label=a == 0 ? "no canopy" : "a = $(Int(a)) 1/m")
end
t_ref = [20.0, 400.0]
plot!(p, t_ref, 0.03 .* t_ref; color=:black, linestyle=:dash, label="∝ t (inertial)")
r80 = runs[80.0]
x_anchor = r80.front[findfirst(>=(200.0), r80.t)]
plot!(p, [60.0, 600.0], x_anchor .* ([60.0, 600.0] ./ 200.0) .^ (2 / 3); color=:black, linestyle=:dot, label="∝ t^(2/3) (quadratic drag)")
savefig(p, joinpath(OUTPUT_DIR, "vegetation_lock_exchange_front.png"))

# snapshots
x = range(0.5 * LENGTH / NX, LENGTH - 0.5 * LENGTH / NX, length=NX)
z = range(0.5 * DEPTH / NZ, DEPTH - 0.5 * DEPTH / NZ, length=NZ)
rho_range = (VEG_RHO0, VEG_RHO0 + REDUCED_GRAVITY / 9.81 * VEG_RHO0)
density_panel(rho, title; kw...) = heatmap(x, z, rho; color=:viridis, clims=rho_range, xlabel="x (m)", ylabel="z (m)", title=title,
    titlefontsize=9, colorbar=false, xlims=(0, LENGTH), ylims=(0, DEPTH), kw...)
panels = [density_panel(runs[a].snapshots[t], (a == 0 ? "no canopy" : "a = $(Int(a)) 1/m") * @sprintf(", t = %.0f s", t))
          for a in FRONTAL_AREAS for t in SNAPSHOT_TIMES_S]
savefig(plot(panels...; layout=(3, 2), size=(1300, 520), left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM),
    joinpath(OUTPUT_DIR, "vegetation_lock_exchange_snapshots.png"))

# animation: no canopy against the dense canopy
anim = Animation()
for t in frame_times
    top = density_panel(runs[0.0].snapshots[t], @sprintf("no canopy, t = %.0f s", t))
    bottom = density_panel(runs[80.0].snapshots[t], @sprintf("emergent canopy a = 80 1/m, t = %.0f s", t))
    frame(anim, plot(top, bottom; layout=(2, 1), size=(900, 360), left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM))
end
gif(anim, joinpath(OUTPUT_DIR, "vegetation_lock_exchange.gif"); fps=10)

npzwrite(joinpath(OUTPUT_DIR, "vegetation_lock_exchange.npz"), merge(Dict("t" => runs[0.0].t),
    Dict("front_a$(Int(a))" => runs[a].front for a in FRONTAL_AREAS)))
println("Saved: vegetation_lock_exchange_{front,snapshots}.png, vegetation_lock_exchange.gif in ", abspath(OUTPUT_DIR))
