# Lock exchange running into vegetation in the right half of the tank, for the different representations of the same
# canopy (frontal area a = 20 1/m, stem diameter 6.4 mm, stem length = depth): no vegetation, rigid patch, individual
# rigid stems, individual flexible stems and a flexible patch. The vegetation is drawn into every figure: patches as a
# shaded region with their representative stems, individual stems as lines (their computed shapes).
#
# The script writes to output_barotropic/vegetation_figures/:
# - vegetation_lock_canopies.gif             density and vegetation, all cases stacked
# - vegetation_lock_canopies_snapshots.png   density and vegetation at SNAPSHOT_TIMES_S
# - vegetation_lock_canopies_front.png       front position against time
# - vegetation_lock_canopies.npz             front positions
# The figures in figures/ are copies (the GIF reduced in size).
#
# Usage: julia --project=<path to the repository> experiments/vegetation/plot_vegetation_lock_exchange_canopies.jl
# Run time: about 35 min (the flexible patch alone about 22 min: one posture solve per u-face and time step).

# WHY: batch run without a display; same GR workstation choice as src/visualization.jl.
get!(ENV, "GKSwstype", "100")

using SweTransport2DV
using Plots
using NPZ
using Printf

include(joinpath(@__DIR__, "cases.jl"))

const LENGTH, DEPTH = 8.0, 0.2              # m
const REDUCED_GRAVITY = 0.02                # m/s^2
const NX, NZ, DT = 800, 20, 0.01            # -, -, s
const T_END = 300.0                         # s
const CANOPY = (4.0, 8.0)                   # m, vegetated part of the tank (right half)
const FRONTAL_AREA = 20.0                   # 1/m
const STEM_DIAMETER = 0.0064                # m
const STEM_ROW_SPACING = 0.1                # m, distance between the rows of individual stems along x
const BENDING_STIFFNESS = 1e-7              # N m^2, flexible stems (Ca = rho C_D b U^2 l^3/(2 EI) about 25 at U = 0.01 m/s,
                                            # the current speed in the canopy; 1e-6 bent the stems by only about 1 cm)
const FRAME_INTERVAL_S = 5.0                # s
const SNAPSHOT_TIMES_S = (60.0, 150.0, 300.0)
const PATCH_STEMS_DRAWN_EVERY = 0.2         # m, representative patch stems drawn at this spacing
const CASES = [
    ("no vegetation", :none),
    ("rigid patch (averaged quantities)", :rigid_patch),
    ("individual rigid stems (rows every 10 cm)", :rigid_stems),
    ("individual flexible stems (EI = 1e-7 N m²)", :flexible_stems),
    ("flexible patch (EI = 1e-7 N m²)", :flexible_patch),
]
const COLORS = Dict(:none => :gray, :rigid_patch => :darkgreen, :rigid_stems => :steelblue, :flexible_stems => :darkorange,
    :flexible_patch => :purple)
const FIGURE_MARGIN_MM = 4Plots.mm
const OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "output_barotropic", "vegetation_figures")

"""
    canopy_vegetation(kind) -> Union{Nothing,VegetationModel}

The canopy of `kind` in CANOPY, all with the frontal area FRONTAL_AREA: a patch with n = a / b stems/m², or rows of
individual stems every STEM_ROW_SPACING with the lateral spacing 1 / (n * STEM_ROW_SPACING) (same stem density).
"""
function canopy_vegetation(kind::Symbol)
    density = FRONTAL_AREA / STEM_DIAMETER                       # stems/m^2
    kind == :none && return nothing
    if kind in (:rigid_patch, :flexible_patch)
        return VegetationModel(; patches=[VegetationPatch(x_start=CANOPY[1], x_end=CANOPY[2], stem_height=DEPTH, stem_diameter=STEM_DIAMETER,
            stem_density=density, drag_coefficient=1.0, bending_stiffness=kind == :flexible_patch ? BENDING_STIFFNESS : Inf, n_segments=20)])
    end
    spacing = 1 / (density * STEM_ROW_SPACING)                   # m, lateral spacing of each row
    positions = (CANOPY[1] + STEM_ROW_SPACING / 2):STEM_ROW_SPACING:CANOPY[2]
    return VegetationModel([VegetationStem(base_x=x, height=DEPTH, diameter=STEM_DIAMETER, drag_coefficient=1.0, lateral_spacing=spacing,
        bending_stiffness=kind == :flexible_stems ? BENDING_STIFFNESS : Inf, n_segments=20) for x in positions]...)
end

"""
    draw_vegetation!(p, vegetation, dx)

Draws the vegetation on the plot `p`: patches as a shaded region up to the canopy height with their representative
stems every PATCH_STEMS_DRAWN_EVERY, individual stems as lines.
"""
function draw_vegetation!(p, vegetation, dx)
    vegetation === nothing && return p
    for (ip, patch) in enumerate(vegetation.patches)
        theta = vegetation.patch_theta[ip]
        ds = patch.stem_height / patch.n_segments
        plot!(p, [patch.x_start, patch.x_end, patch.x_end, patch.x_start], [0, 0, DEPTH, DEPTH]; seriestype=:shape,
            fillcolor=:white, fillalpha=0.12, linecolor=:white, linealpha=0.4, label="")
        for x in (patch.x_start + PATCH_STEMS_DRAWN_EVERY / 2):PATCH_STEMS_DRAWN_EVERY:patch.x_end
            i = round(Int, x / dx) + 1
            th = size(theta, 2) >= i ? theta[:, i] : zeros(patch.n_segments)
            plot!(p, x .+ [0.0; cumsum(ds .* sin.(th))], [0.0; cumsum(ds .* cos.(th))]; color=:white, linewidth=0.8, linealpha=0.7, label="")
        end
    end
    for stem in vegetation.stems
        xs, zs = vegetation_path(stem)
        plot!(p, xs, zs; color=:white, linewidth=0.8, linealpha=0.7, label="")
    end
    return p
end

mkpath(OUTPUT_DIR)
drho = REDUCED_GRAVITY / 9.81 * VEG_RHO0
frame_steps = round(Int, FRAME_INTERVAL_S / DT)
x = range(0.5 * LENGTH / NX, LENGTH - 0.5 * LENGTH / NX, length=NX)
z = range(0.5 * DEPTH / NZ, DEPTH - 0.5 * DEPTH / NZ, length=NZ)
results = Dict{Symbol,Any}()
for (title, kind) in CASES
    cfg = ModelConfig(length=LENGTH, depth=DEPTH, nx=NX, nz=NZ, dt=DT, nt=round(Int, T_END / DT), use_hydrostatic=false, rho0=VEG_RHO0,
        rho_left=VEG_RHO0 + drho, rho_right=VEG_RHO0, rayleigh_damping=0.0, apply_froude_limiter=false, momentum_advection_scheme="tvd2",
        transport_advection_scheme="tvd2", nu=1e-6, write_plots=false, write_animation=false, write_pressure_buoyancy_diagnostics=false)
    model = Model(cfg)
    initialize_lock_exchange!(model)
    vegetation = canopy_vegetation(kind)
    vegetation === nothing || attach_vegetation!(model, vegetation)
    times = Float64[]; fronts = Float64[]; frames = Any[]
    threshold = VEG_RHO0 + 0.5 * drho
    t0 = time()
    for n in 1:cfg.nt
        step_forward!(model)
        if n % frame_steps == 0
            heavy = findall(>(threshold), @view model.state.rho[1, :])
            push!(times, n * DT); push!(fronts, (maximum(heavy) - 0.5) * model.dx - LENGTH / 2)
            push!(frames, (copy(model.state.rho), vegetation === nothing ? nothing : copy_vegetation(model.vegetation)))
        end
    end
    tip = kind == :flexible_stems ? @sprintf(", mean tip deflection %.3f m", sum(stem_tip_deflection(s) for s in model.vegetation.stems) / length(model.vegetation.stems)) : ""
    @printf("%-45s front at 60/150/300 s = %.2f / %.2f / %.2f m%s (%.0f s wall)\n", title, fronts[12], fronts[30], fronts[end], tip, time() - t0)
    results[kind] = (title=title, t=times, front=fronts, frames=frames, dx=model.dx)
end

rho_range = (VEG_RHO0, VEG_RHO0 + drho)
function panel(kind, iframe; bottom=false, kw...)
    r = results[kind]
    rho, veg = r.frames[iframe]
    p = heatmap(x, z, rho; color=:viridis, clims=rho_range, colorbar=false, xlims=(0, LENGTH), ylims=(0, DEPTH), title=r.title,
        titlefontsize=9, xlabel=bottom ? "x (m)" : "", ylabel="z (m)", kw...)
    return draw_vegetation!(p, veg, r.dx)
end

anim = Animation()
for iframe in eachindex(results[:none].t)
    panels = [panel(kind, iframe; bottom=kind == last(CASES)[2]) for (_, kind) in CASES]
    frame(anim, plot(panels...; layout=(length(CASES), 1), size=(900, 180 * length(CASES)), left_margin=FIGURE_MARGIN_MM,
        bottom_margin=1Plots.mm, plot_title=@sprintf("Lock exchange into vegetation (right half), t = %.0f s", results[:none].t[iframe]),
        plot_titlefontsize=11))
end
gif(anim, joinpath(OUTPUT_DIR, "vegetation_lock_canopies.gif"); fps=8)

snapshot_panels = []
for (_, kind) in CASES, t in SNAPSHOT_TIMES_S
    iframe = findfirst(>=(t - 1e-9), results[kind].t)
    push!(snapshot_panels, panel(kind, iframe; bottom=kind == last(CASES)[2], title=@sprintf("%s, t = %.0f s", results[kind].title, t), titlefontsize=8))
end
savefig(plot(snapshot_panels...; layout=(length(CASES), length(SNAPSHOT_TIMES_S)), size=(1500, 190 * length(CASES)),
    left_margin=FIGURE_MARGIN_MM, bottom_margin=1Plots.mm), joinpath(OUTPUT_DIR, "vegetation_lock_canopies_snapshots.png"))

pf = plot(xlabel="t (s)", ylabel="front position from the gate (m)", legend=:topleft, size=(760, 440), left_margin=FIGURE_MARGIN_MM,
    bottom_margin=FIGURE_MARGIN_MM, title="Dense front along the bed (vegetation from x − gate = 0 m)")
for (_, kind) in CASES
    plot!(pf, results[kind].t, results[kind].front; color=COLORS[kind], linewidth=2, label=results[kind].title)
end
savefig(pf, joinpath(OUTPUT_DIR, "vegetation_lock_canopies_front.png"))
npzwrite(joinpath(OUTPUT_DIR, "vegetation_lock_canopies.npz"), merge(Dict("t" => results[:none].t),
    Dict("front_$(kind)" => results[kind].front for (_, kind) in CASES)))
println("Saved: vegetation_lock_canopies.gif, _snapshots.png, _front.png in ", abspath(OUTPUT_DIR))
