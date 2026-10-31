# k-epsilon with wetting and drying and the variable top layer (issue #30).
#
# The script writes to output_barotropic/k_epsilon_wetting_drying_figures/ and copies the files to figures/:
# - k_epsilon_dambreak.gif       stratified dam break onto a dry slope: density with k-epsilon and without a closure, and
#                                the eddy viscosity (cells cut at the bed and the free surface)
# - k_epsilon_dambreak.png       the same at three times
# - k_epsilon_bound.png          largest eddy viscosity and density range over time: k-epsilon with and without the
#                                realizability time-scale bound (Durbin 1996), and no closure
# - k_epsilon_channels.png       open-channel profiles: periodic fixed grid, wetting-drying channel (flat bed, bed cutting
#                                a layer) and the variable top layer
#
# Usage: julia --project=<path to the repository> experiments/k_epsilon_wetting_drying/plot_k_epsilon_wetting_drying.jl
# [product ...] (products: dambreak bound channels; default all). Run time: about 5 min for all.

# WHY: batch run without a display; same GR workstation choice as src/visualization.jl.
get!(ENV, "GKSwstype", "100")

using SweTransport2DV
using Plots
using Printf
using Statistics

const RHO_LIGHT = 1000.0        # kg/m^3, upper layer
const RHO_DENSE = 1010.0        # kg/m^3, lower layer
# Dam break: 10 m basin, 0.4 m column of water left of 3 m, dry bed rising at 6 % from 4 m
const DB_LENGTH = 10.0          # m
const DB_GRID_DEPTH = 0.5       # m
const DB_NX = 100               # -, dx = 0.1 m
const DB_NZ = 10                # -, dz = 0.05 m
const DB_DENSE_LAYERS = 4       # -, the lowest 0.2 m are dense
const DB_LEVEL = 0.4            # m
const DB_GATE_X = 3.0           # m, initial edge of the column
const DB_SLOPE_START = 4.0      # m
const DB_SLOPE = 0.06           # -
const DB_Z0 = 1e-3              # m, bed roughness length
const DB_DT = 0.005             # s
const DB_TIME = 40.0            # s
const DB_FRAME_TIME = 1.0       # s
const DB_DRAWN_LAYER_MIN = 1e-3 # m, thinner layers are not drawn or counted in the density range: a cell of 10-50 um
                                # that has just got wet can briefly hold a density far from its neighbours
const COLOUR_BINS = 64          # -, colour levels of the cut cells
const DB_SNAPSHOT_TIMES = (2.0, 8.0, 40.0)   # s
const NU_T_LOG_RANGE = (-6.0, -2.0)          # log10 of nu_t (m^2/s), colour range
# Open channel (section 27 of the docs): H = 0.5 m, slope 1e-3, z0 = 1 mm, 20 layers
const CH_DEPTH = 0.5            # m
const CH_SLOPE = 1e-3           # -
const CH_Z0 = 1e-3              # m
const CH_NZ = 20                # -
const CH_LENGTH = 80.0          # m
const CH_NX = 80                # -
const CH_COLUMN = 60            # -, column at 3/4 of the reach
const CH_DT = 0.25              # s
const CH_TIME = 900.0           # s
const CH_CUT_BED = 0.21         # m, raised bed that cuts a layer (grid depth CH_DEPTH + 0.2, 28 layers)
const CH_TOP_GAP = 0.01         # m, grid this far below the water level: thicker top row of the variable top layer
const ANIMATION_FPS = 8
const FRAME_SIZE_PX = (760, 560)
const FIGURE_MARGIN_MM = 5Plots.mm
const OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "output_barotropic", "k_epsilon_wetting_drying_figures")
const FIGURES_DIR = joinpath(@__DIR__, "figures")

function save_figure(fig, name)
    path = joinpath(OUTPUT_DIR, name)
    savefig(fig, path)
    cp(path, joinpath(FIGURES_DIR, name); force=true)
    @printf("wrote %s (%.0f kB)\n", path, filesize(path) / 1024)
end

# --- dam break -----------------------------------------------------------------------------------------------------

# closure: :k_epsilon, :unbounded (k-epsilon without the time-scale bound) or :none
function dambreak_model(closure; nonhydro=false)
    xc = ((1:DB_NX) .- 0.5) .* (DB_LENGTH / DB_NX)
    cfg = ModelConfig(; length=DB_LENGTH, depth=DB_GRID_DEPTH, nx=DB_NX, nz=DB_NZ, dt=DB_DT, use_hydrostatic=!nonhydro,
        nu=1e-6, turbulence_closure=closure == :none ? "none" : "k_epsilon", bed_roughness_length=DB_Z0,
        rayleigh_damping=0.0, apply_froude_limiter=false, momentum_advection_scheme="tvd2", wetting_drying=true,
        implicit_vertical_terms=true, theta=1.0, bed_level=[max(0.0, DB_SLOPE * (x - DB_SLOPE_START)) for x in xc],
        write_plots=false)
    model = Model(cfg)
    closure == :unbounded && (model.turbulence.realizability = false)
    for i in 1:DB_NX, k in 1:DB_NZ
        model.state.rho[k, i] = k <= DB_DENSE_LAYERS ? RHO_DENSE : RHO_LIGHT
    end
    refresh_density_metadata!(model)
    initialize_water_level!(model, [x < DB_GATE_X ? DB_LEVEL : 0.0 for x in xc])
    return model, xc
end

# Wet part of every layer cell as a polygon: from max(z_k, bed) to the free surface, drawn piecewise linear between the
# cell centres (a dry neighbour contributes its bed level, so the shoreline slopes), capped at the layer top.
function shaved_cells!(p, model, xc, values; clims, colormap, label)
    dz, dx, nz = model.dz, model.dx, model.config.nz
    wd = model.wetdry
    level = model.config.depth .+ model.state.eta   # m, water level (the bed in a dry column)
    nx = length(xc)
    face(f, i) = (f == 1 || f == nx + 1) ? level[i] : 0.5 * (level[f - 1] + level[f])
    shapes, colours = Shape[], Float64[]
    for i in 1:nx
        px = (xc[i] - 0.5dx, xc[i], xc[i] + 0.5dx)
        pz = (face(i, i), level[i], face(i + 1, i))
        for k in 1:nz
            wd.h[k, i] > DB_DRAWN_LAYER_MIN || continue
            zb, zt = max((k - 1) * dz, wd.bed[i]), k * dz
            xs = collect(px)
            for j in 1:2, zl in (zb, zt)
                t = (zl - pz[j]) / (pz[j + 1] - pz[j])
                0 < t < 1 && push!(xs, px[j] + t * (px[j + 1] - px[j]))
            end
            sort!(xs)
            top = [clamp(x <= px[2] ? pz[1] + (pz[2] - pz[1]) * (x - px[1]) / (px[2] - px[1]) :
                                      pz[2] + (pz[3] - pz[2]) * (x - px[2]) / (px[3] - px[2]), zb, zt) for x in xs]
            push!(shapes, Shape(vcat(xs, reverse(xs)), vcat(fill(zb, length(xs)), reverse(top))))
            push!(colours, values[k, i])
        end
    end
    # WHY colour bins with an explicit fill and line colour: fill_z left single cells unfilled and hairlines between cells
    # in the GR output, although their values are regular
    palette = cgrad(colormap)
    bins = [Shape[] for _ in 1:COLOUR_BINS]
    for (shape, value) in zip(shapes, colours)
        b = clamp(round(Int, (value - clims[1]) / (clims[2] - clims[1]) * (COLOUR_BINS - 1)), 0, COLOUR_BINS - 1)
        push!(bins[b + 1], shape)
    end
    for (b, group) in enumerate(bins)
        isempty(group) && continue
        c = palette[(b - 1) / (COLOUR_BINS - 1)]
        plot!(p, group; fillcolor=c, linecolor=c, linewidth=0.5, label="")
    end
    # colour bar of the range, from an invisible point
    scatter!(p, [-1.0], [-1.0]; marker_z=[clims[1]], color=colormap, clims=clims, colorbar=true, colorbar_title=label,
        markersize=0, markerstrokewidth=0, label="")
    # bed (staircase of the cell beds)
    bx = vcat([[xc[i] - 0.5dx, xc[i] + 0.5dx] for i in 1:nx]...)
    bz = vcat([[wd.bed[i], wd.bed[i]] for i in 1:nx]...)
    plot!(p, bx, bz; color=:black, lw=1.5, label="")
    return p
end

function dambreak_panels(models, t)
    (mk, xc), (mn, _) = models
    common = (xlims=(0, DB_LENGTH), ylims=(0, DB_LEVEL + 0.05), xlabel="x (m)", ylabel="z (m)",
        background_color_inside=:gray92, legend=false)
    p1 = plot(; title=@sprintf("density, k-epsilon, t = %.1f s", t), common...)
    shaved_cells!(p1, mk, xc, mk.state.rho; clims=(RHO_LIGHT, RHO_DENSE), colormap=:thermal, label="ρ (kg/m³)")
    p2 = plot(; title=@sprintf("density, no closure, t = %.1f s", t), common...)
    shaved_cells!(p2, mn, xc, mn.state.rho; clims=(RHO_LIGHT, RHO_DENSE), colormap=:thermal, label="ρ (kg/m³)")
    p3 = plot(; title="eddy viscosity, k-epsilon", common...)
    lognut = log10.(max.(mk.turbulence.nu_t, 10.0^NU_T_LOG_RANGE[1]))
    shaved_cells!(p3, mk, xc, lognut; clims=NU_T_LOG_RANGE, colormap=:viridis, label="log10 ν_t (m²/s)")
    return p1, p2, p3
end

function dambreak_products()
    models = (dambreak_model(:k_epsilon), dambreak_model(:none))
    anim = Animation()
    snapshots = []
    stride = round(Int, DB_FRAME_TIME / DB_DT)
    for n in 0:round(Int, DB_TIME / DB_DT)
        if n > 0
            step_forward!(models[1][1])
            step_forward!(models[2][1])
        end
        t = n * DB_DT
        if n % stride == 0
            frame(anim, plot(dambreak_panels(models, t)...; layout=(3, 1), size=FRAME_SIZE_PX, margin=FIGURE_MARGIN_MM))
        end
        any(ts -> abs(t - ts) < 0.5DB_DT, DB_SNAPSHOT_TIMES) && push!(snapshots, dambreak_panels(models, t))
    end
    path = joinpath(OUTPUT_DIR, "k_epsilon_dambreak.gif")
    gif(anim, path; fps=ANIMATION_FPS)
    cp(path, joinpath(FIGURES_DIR, "k_epsilon_dambreak.gif"); force=true)
    @printf("wrote %s (%.0f kB)\n", path, filesize(path) / 1024)
    # snapshots: one column per time
    panels = [snapshots[j][r] for r in 1:3 for j in eachindex(snapshots)]
    save_figure(plot(panels...; layout=(3, length(snapshots)), size=(1500, 760), margin=FIGURE_MARGIN_MM), "k_epsilon_dambreak.png")
end

# Largest eddy viscosity and density range of the wet cells over time, for k-epsilon with and without the bound and
# without a closure
function bound_figure()
    runs = [(:k_epsilon, "k-epsilon (bounded time scale)", :steelblue), (:unbounded, "k-epsilon, T = k/ε", :crimson),
        (:none, "no closure", :gray40)]
    p1 = plot(xlabel="t (s)", ylabel="largest ν_t (m²/s)", yscale=:log10, title="Eddy viscosity in the dam break",
        legend=:right)
    p2 = plot(xlabel="t (s)", ylabel="density range (kg/m³)", title="Mixing of the stratification", legend=:topright)
    for (closure, label, color) in runs
        model, _ = dambreak_model(closure)
        ts, nut, spread = Float64[], Float64[], Float64[]
        peak = 0.0   # m^2/s, largest nu_t since the last sample: the peak without the bound lasts a few steps
        for n in 1:round(Int, DB_TIME / DB_DT)
            step_forward!(model)
            closure == :none || (peak = max(peak, maximum(model.turbulence.nu_t)))
            n % 10 == 0 || continue
            wet = model.wetdry.h .> DB_DRAWN_LAYER_MIN
            push!(ts, n * DB_DT)
            push!(nut, closure == :none ? NaN : max(peak, 1e-12))
            push!(spread, maximum(model.state.rho[wet]) - minimum(model.state.rho[wet]))
            peak = 0.0
        end
        closure == :none || plot!(p1, ts, nut; color=color, lw=2, label=label)
        plot!(p2, ts, spread; color=color, lw=2, label=label)
        @printf("%-34s max nu_t %.3g m^2/s, final density range %.2f kg/m^3\n", label, maximum(filter(isfinite, vcat(nut, 0.0))), spread[end])
    end
    save_figure(plot(p1, p2; layout=(1, 2), size=(1250, 440), margin=FIGURE_MARGIN_MM), "k_epsilon_bound.png")
end

# --- open channels ------------------------------------------------------------------------------------------------

function channel_reference()
    cfg = ModelConfig(; length=1.0, depth=CH_DEPTH, nx=4, nz=CH_NZ, dt=0.1, use_hydrostatic=true, periodic_x=true,
        variable_top_layer=false, pressure_solver_method="sparse", nu=1e-6, body_force_x=9.81 * CH_SLOPE,
        turbulence_closure="k_epsilon", bed_roughness_length=CH_Z0, rayleigh_damping=0.0, apply_froude_limiter=false,
        momentum_advection_scheme="tvd2", implicit_vertical_terms=true, write_plots=false, write_animation=false,
        write_pressure_buoyancy_diagnostics=false)
    model = Model(cfg)
    fill!(model.state.rho, RHO_LIGHT)
    refresh_density_metadata!(model)
    for _ in 1:round(Int, CH_TIME / cfg.dt)
        step_forward!(model)
    end
    z = ((1:CH_NZ) .- 0.5) .* model.dz
    u = vec(mean(model.state.u[:, 1:end-1]; dims=2))
    return z, u, vec(mean(model.turbulence.nu_t; dims=2)), vec(mean(model.turbulence.k; dims=2))
end

# Open channel of length CH_LENGTH: inlet discharge q, outlet at the normal depth; wetting and drying (flat or raised
# bed) or the fixed grid with the variable top layer. Profiles of column CH_COLUMN against the height above the bed.
function channel_profiles(q; wetting_drying, bed=0.0, grid_depth=CH_DEPTH, nz=CH_NZ)
    cfg = ModelConfig(; length=CH_LENGTH, depth=grid_depth, nx=CH_NX, nz=nz, dt=CH_DT, use_hydrostatic=true, nu=1e-6,
        body_force_x=9.81 * CH_SLOPE, turbulence_closure="k_epsilon", bed_roughness_length=CH_Z0, rayleigh_damping=0.0,
        apply_froude_limiter=false, momentum_advection_scheme="tvd2", implicit_vertical_terms=true, theta=1.0,
        wetting_drying=wetting_drying, variable_top_layer=wetting_drying ? nothing : true,
        bed_level=bed == 0.0 ? nothing : fill(bed, CH_NX), write_plots=false, write_animation=false,
        write_pressure_buoyancy_diagnostics=false)
    model = Model(cfg)
    fill!(model.state.rho, RHO_LIGHT)
    refresh_density_metadata!(model)
    offset = bed + CH_DEPTH - grid_depth   # m, outlet level relative to the grid top
    if wetting_drying
        initialize_water_level!(model, fill(bed + CH_DEPTH, CH_NX))
    else
        model.state.eta .= offset
    end
    set_x_inflow_outflow!(model, ones(nz); left_discharge=q, right_level=offset)
    for _ in 1:round(Int, CH_TIME / CH_DT)
        step_forward!(model)
    end
    i, dz = CH_COLUMN, model.dz
    if wetting_drying
        h = model.wetdry.h[:, i]
        wet = findall(>(1e-12), h)
        z = [max((k - 1) * dz, bed) + 0.5 * h[k] - bed for k in wet]
    else
        wet = collect(1:nz)
        h_top = dz + model.state.eta[i]
        z = [k < nz ? (k - 0.5) * dz : (nz - 1) * dz + 0.5 * h_top for k in wet]
    end
    u = [0.5 * (model.state.u[k, i] + model.state.u[k, i + 1]) for k in wet]
    return z, u, model.turbulence.nu_t[wet, i], model.turbulence.k[wet, i]
end

function channels_figure()
    zr, ur, nr, kr = channel_reference()
    q = mean(ur) * CH_DEPTH
    cases = [
        ("wetting-drying, flat bed", :crimson, :circle, channel_profiles(q; wetting_drying=true)),
        (@sprintf("wetting-drying, bed raised %.2f m (cut layer)", CH_CUT_BED), :darkorange, :utriangle,
            channel_profiles(q; wetting_drying=true, bed=CH_CUT_BED, grid_depth=CH_DEPTH + 0.2, nz=28)),
        (@sprintf("variable top layer, top row %.0f %% thicker", 100 * CH_TOP_GAP / (CH_DEPTH / CH_NZ)), :green4, :diamond,
            channel_profiles(q; wetting_drying=false, grid_depth=CH_DEPTH - CH_TOP_GAP)),
    ]
    ustar = sqrt(9.81 * CH_SLOPE * CH_DEPTH)
    labels = ("u (m/s)", "ν_t (m²/s)", "k (m²/s²)")
    panels = map(1:3) do j
        p = plot(xlabel=labels[j], ylabel=j == 1 ? "height above the bed (m)" : "", legend=j == 1 ? :topleft : false,
            ylims=(0, CH_DEPTH))
        plot!(p, (ur, nr, kr)[j], zr; color=:black, lw=2, label="periodic fixed grid (reference)")
        j == 1 && plot!(p, ustar / 0.41 .* log.(zr ./ CH_Z0), zr; color=:gray, ls=:dash, label="log law u*/κ ln(z/z0)")
        for (label, color, marker, prof) in cases
            scatter!(p, prof[j + 1], prof[1]; color=color, marker=marker, ms=4, label=label)
        end
        p
    end
    save_figure(plot(panels...; layout=(1, 3), size=(1400, 480), margin=FIGURE_MARGIN_MM,
        plot_title=@sprintf("k-epsilon open channel, H = %.1f m, S = %.0e, z0 = %.0f mm, at %.0f m of %.0f m", CH_DEPTH,
            CH_SLOPE, 1000CH_Z0, CH_COLUMN - 0.5, CH_LENGTH)), "k_epsilon_channels.png")
end

const PRODUCTS = Dict("dambreak" => dambreak_products, "bound" => bound_figure, "channels" => channels_figure)

mkpath(OUTPUT_DIR)
mkpath(FIGURES_DIR)
for name in (isempty(ARGS) ? ["dambreak", "bound", "channels"] : ARGS)
    haskey(PRODUCTS, name) || error("unknown product $(name); choose from $(join(sort(collect(keys(PRODUCTS))), ", "))")
    PRODUCTS[name]()
end
