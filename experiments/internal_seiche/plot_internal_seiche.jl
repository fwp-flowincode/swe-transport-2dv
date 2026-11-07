# Internal seiche (issue #23): period against the two-layer relation on three grids, hydrostatic and non-hydrostatic, the
# interface amplitude over time, and the stability of hydrostatic internal waves with the density extrapolation of the
# baroclinic pressure (baroclinic_extrapolation_weight). The case is the one of test/test_internal_seiche.jl.
#
# Usage (any working directory):
#   julia --project=<repository> experiments/internal_seiche/plot_internal_seiche.jl
# About 1.5 min; also writes the animation seiche_2dv.gif (hydrostatic and non-hydrostatic side by side). Output: output_barotropic/internal_seiche_figures/ (figures and CSV files); the PNG files are copied to
# figures/.
# The "before" curve of the stability figure is the same script run against the package before issue #23 (it only
# writes its CSV file there):
#   git worktree add ../swe-before d696c11
#   WD_SEICHE_TAG=before julia --project=../swe-before experiments/internal_seiche/plot_internal_seiche.jl
# WHY the same script for both: identical set-up and diagnostics; only the package differs.

include(joinpath(@__DIR__, "..", "common", "performance.jl"))   # performance overview at exit (issue #10)
using SweTransport2DV
using Plots
using Printf

# WHY: batch run without a display; same GR workstation choice as src/visualization.jl.
get!(ENV, "GKSwstype", "100")

const OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "output_barotropic", "internal_seiche_figures")
const FIGURE_DIR = joinpath(@__DIR__, "figures")
const TAG = get(ENV, "WD_SEICHE_TAG", "now")

# ---- Case (test/test_internal_seiche.jl) ------------------------------------------------------------------------------
const LENGTH = 8.0              # m, one wavelength (mode 2 of the closed basin)
const DEPTH = 4.0               # m
const RHO_TOP = 998.5           # kg/m^3
const RHO_BOTTOM = 1001.5       # kg/m^3
const RHO0 = 1000.0             # kg/m^3, ModelConfig default
const G = 9.81                  # m/s^2, ModelConfig default
const INTERFACE_FRACTION = 0.5  # -, interface at mid-depth
const AMPLITUDE = 0.12          # m, interface displacement
const DT = 0.05                 # s
const PERIODS = 3               # -, simulated periods
const GRIDS = ((40, 40), (80, 80), (160, 160))   # (nx, nz)
const STABILITY_GRID = (160, 160)                # hydrostatic grid of the stability figure (grew before issue #23)
const STABILITY_T_END = 150.0                    # s
const SECTION_SIZE_PX = (900, 420)               # px
const LINE_SIZE_PX = (1150, 430)                 # px
const FIGURE_MARGIN_MM = 4Plots.mm
# WHY 80 x 80, 2.5 s and 15 colour levels: the file must stay well below 1 MB (160 x 160 every 1.25 s with a continuous
# colour map gave 9.7 MB)
const ANIMATION_GRID = (80, 80)                  # (nx, nz) of the animation
const ANIMATION_T_END = 125.0                    # s, about two hydrostatic and two non-hydrostatic periods
const ANIMATION_FRAME_INTERVAL = 2.5             # s between frames
const ANIMATION_FPS = 8                          # frames/s
const ANIMATION_SIZE_PX = (900, 310)             # px
const ANIMATION_U_LIMIT = 0.013                  # m/s, colour range of u (the largest |u| is about 0.011 m/s)
const ANIMATION_COLOR_LEVELS = 15                # -, discrete colours of u

wavenumber() = 2pi / LENGTH
layers() = (DEPTH * (1 - INTERFACE_FRACTION), DEPTH * INTERFACE_FRACTION)   # m, (upper, lower)
reference(hydrostatic) = two_layer_internal_wave(; k=wavenumber(), h_upper=layers()[1], h_lower=layers()[2],
    rho_upper=RHO_TOP, rho_lower=RHO_BOTTOM, rho0=RHO0, g=G, hydrostatic=hydrostatic)

function seiche_model(nx, nz; hydrostatic, kwargs...)
    cfg = ModelConfig(; length=LENGTH, depth=DEPTH, nx=nx, nz=nz, dt=DT, use_hydrostatic=hydrostatic,
        momentum_advection_scheme="tvd2", transport_advection_scheme="tvd2", rayleigh_damping=0.0,
        apply_froude_limiter=false, nu=0.0, write_plots=false, write_animation=false,
        write_pressure_buoyancy_diagnostics=false, kwargs...)
    model = Model(cfg)
    init(; kw...) = initialize_internal_seiche!(model; rho_top=RHO_TOP, rho_bottom=RHO_BOTTOM,
        interface_height_fraction=INTERFACE_FRACTION, interface_tilt_amplitude_fraction=AMPLITUDE / DEPTH, kw...)
    try
        init(; cell_average=true)
    catch e
        # WHY: the package before issue #23 (the "before" run) has no cell_average; the same cell-averaged field by hand
        e isa MethodError || rethrow()
        init()
        for i in 1:nx, k in 1:nz
            z_interface = DEPTH * INTERFACE_FRACTION + AMPLITUDE * cos(wavenumber() * (i - 0.5) * model.dx)
            below = clamp((z_interface - (k - 1) * model.dz) / model.dz, 0.0, 1.0)
            model.state.rho[k, i] = below * RHO_BOTTOM + (1.0 - below) * RHO_TOP
        end
        refresh_density_metadata!(model)
    end
    return model
end

"Amplitude (m) of the cos(k x) mode of the interface height (depth of the heavy fluid, m) of `model`."
function interface_mode(model)
    nx, dx, dz = model.config.nx, model.dx, model.dz
    zeta = vec(sum((model.state.rho .- RHO_TOP) ./ (RHO_BOTTOM - RHO_TOP); dims=1)) .* dz .- DEPTH * INTERFACE_FRACTION
    return 2 / LENGTH * dx * sum(zeta[i] * cos(wavenumber() * (i - 0.5) * dx) for i in 1:nx)
end

"Period (s) from the zero crossings of the series `a` sampled every DT, and the amplitude kept per period (-)."
function period_and_damping(a)
    crossings = Float64[]
    for j in 2:length(a)
        a[j - 1] * a[j] < 0 && push!(crossings, (j - 2 + a[j - 1] / (a[j - 1] - a[j])) * DT)
    end
    period = 2 * (crossings[end] - crossings[1]) / (length(crossings) - 1)
    peaks = [maximum(abs, a[(floor(Int, crossings[j] / DT) + 1):(ceil(Int, crossings[j + 1] / DT) + 1)])
             for j in 1:(length(crossings) - 1)]
    return period, (peaks[end] / peaks[1])^(2 / (length(peaks) - 1))
end

function save_figure(fig, name)
    path = joinpath(OUTPUT_DIR, name)
    savefig(fig, path)
    cp(path, joinpath(FIGURE_DIR, name); force=true)
    println("Saved: $(abspath(path))")
end

function write_csv(path, header, rows)
    open(path, "w") do io
        println(io, join(header, ","))
        foreach(r -> println(io, join(r, ",")), rows)
    end
    println("Saved: $(abspath(path))")
end

function read_csv(path)
    isfile(path) || return nothing
    lines = readlines(path)
    header = Symbol.(split(lines[1], ","))
    return [NamedTuple{Tuple(header)}(Tuple(parse.(Float64, split(l, ",")))) for l in lines[2:end]]
end

# max |u| (m/s) of the hydrostatic seiche on STABILITY_GRID every second, until STABILITY_T_END or a failure
function stability_series(; kwargs...)
    model = seiche_model(STABILITY_GRID...; hydrostatic=true, kwargs...)
    rows = Tuple{Float64,Float64}[]
    every = round(Int, 1.0 / DT)
    for n in 1:round(Int, STABILITY_T_END / DT)
        ok = try
            step_forward!(model)
            all(isfinite, model.state.u)
        catch
            false
        end
        ok || break
        n % every == 0 && push!(rows, (n * DT, maximum(abs, model.state.u)))
    end
    return rows
end

mkpath(OUTPUT_DIR)
mkpath(FIGURE_DIR)

println("\n=== Hydrostatic stability on $(STABILITY_GRID[1]) x $(STABILITY_GRID[2]) cells ($(TAG)) ===")
for (label, kw) in (("theta07", NamedTuple()), ("theta1", (theta=1.0,)))
    rows = stability_series(; kw...)
    @printf("%-8s run until t = %.0f s, largest |u| %.4f m/s\n", label, rows[end][1], maximum(last, rows))
    write_csv(joinpath(OUTPUT_DIR, "stability_$(label)_$(TAG).csv"), ("t", "umax"), rows)
end
TAG == "now" || exit()

println("\n=== Period against the grid ===")
results = Dict{Tuple{Bool,Int},Any}()
for hydrostatic in (true, false), (nx, nz) in GRIDS
    model = seiche_model(nx, nz; hydrostatic=hydrostatic)
    a, eta, times = [interface_mode(model)], Float64[], [0.0]
    T_ref = 2pi / reference(hydrostatic)[1]
    for n in 1:round(Int, PERIODS * T_ref / DT)
        step_forward!(model)
        push!(a, interface_mode(model))
        push!(times, n * DT)
    end
    period, kept = period_and_damping(a)
    results[(hydrostatic, nx)] = (; period, kept, a, times, model)
    @printf("%-15s %3d x %3d | period %.2f s (reference %.2f s, %+.2f %%) | amplitude kept per period %.4f\n",
        hydrostatic ? "hydrostatic" : "non-hydrostatic", nx, nz, period, T_ref, 100 * (period - T_ref) / T_ref, kept)
end

# 1. period error against the grid spacing
p = plot(xlabel="dx = dz (m)", ylabel="period error (%)", xscale=:log10, yscale=:log10, legend=:bottomright,
    xticks=([LENGTH / nx for (nx, _) in GRIDS], [@sprintf("%g", LENGTH / nx) for (nx, _) in GRIDS]),
    yticks=([1, 2, 3, 5, 8], ["1", "2", "3", "5", "8"]),
    title="Internal seiche, 8 m x 4 m, period against the two-layer relation", titlefontsize=11, size=(700, 450),
    left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM)
for (hydrostatic, color, name) in ((true, 1, "hydrostatic"), (false, 2, "non-hydrostatic"))
    T_ref = 2pi / reference(hydrostatic)[1]
    dx = [LENGTH / nx for (nx, _) in GRIDS]
    err = [100 * abs(results[(hydrostatic, nx)].period - T_ref) / T_ref for (nx, _) in GRIDS]
    orders = join([@sprintf("%.2f", log2(err[j] / err[j + 1])) for j in 1:(length(err) - 1)], ", ")
    plot!(p, dx, err; color=color, marker=:circle, linewidth=2,
        label=@sprintf("%s (T = %.2f s; orders %s)", name, T_ref, orders))
    plot!(p, dx, err[end] .* dx ./ dx[end]; color=color, linestyle=:dot, label="")
end
save_figure(p, "seiche_period_convergence.png")

# 2. interface amplitude over time on the finest grid, against the reference cos(omega t)
p = plot(xlabel="t (s)", ylabel="interface amplitude (m)", legend=:bottomleft, size=LINE_SIZE_PX,
    title="Interface mode amplitude, $(GRIDS[end][1]) x $(GRIDS[end][2]) cells", titlefontsize=11,
    left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM)
for (hydrostatic, color, name) in ((true, 1, "hydrostatic"), (false, 2, "non-hydrostatic"))
    r = results[(hydrostatic, GRIDS[end][1])]
    om = reference(hydrostatic)[1]
    plot!(p, r.times, r.a; color=color, linewidth=2, label="$(name), model")
    plot!(p, r.times, AMPLITUDE .* cos.(om .* r.times); color=color, linestyle=:dash, linewidth=1.5,
        label=@sprintf("%s, two-layer relation (T = %.2f s)", name, 2pi / om))
end
save_figure(p, "seiche_interface.png")

# 3. density and velocity at a quarter period, non-hydrostatic, finest grid
let r = results[(false, GRIDS[end][1])], m = seiche_model(GRIDS[end]...; hydrostatic=false)
    T = 2pi / reference(false)[1]
    for _ in 1:round(Int, T / 4 / DT)
        step_forward!(m)
    end
    nx, nz = m.config.nx, m.config.nz
    x = [(i - 0.5) * m.dx for i in 1:nx]
    z = [(k - 0.5) * m.dz for k in 1:nz]
    uc = 0.5 .* (m.state.u[:, 1:(end - 1)] .+ m.state.u[:, 2:end])
    p1 = heatmap(x, z, m.state.rho; color=:viridis, clims=(RHO_TOP, RHO_BOTTOM), colorbar_title="ρ (kg/m³)",
        xlabel="x (m)", ylabel="z (m)", title=@sprintf("density, t = T/4 = %.1f s", round(Int, T / 4 / DT) * DT), titlefontsize=10)
    umax = maximum(abs, uc)
    p2 = heatmap(x, z, uc; color=:balance, clims=(-umax, umax), colorbar_title="u (m/s)", xlabel="x (m)", ylabel="z (m)",
        title="horizontal velocity", titlefontsize=10)
    save_figure(plot(p1, p2; layout=(1, 2), size=(1300, 430), left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM,
        plot_title="Internal seiche, non-hydrostatic, $(nx) x $(nz) cells", plot_titlefontsize=11), "seiche_section.png")
end

# 4. stability of the hydrostatic seiche on the finest grid, before and after issue #23
p = plot(xlabel="t (s)", ylabel="max |u| (m/s)", yscale=:log10, legend=:topleft, size=LINE_SIZE_PX,
    title="Hydrostatic seiche on $(STABILITY_GRID[1]) x $(STABILITY_GRID[2]) cells, dt = $(DT) s", titlefontsize=11,
    left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM)
for (file, label, style) in (("theta07_before", "θ = 0.7, before issue #23 (run stops)", (color=2, linestyle=:dash)),
        ("theta07_now", "θ = 0.7, density extrapolation (issue #23)", (color=2, linestyle=:solid)),
        ("theta1_now", "θ = 1", (color=1, linestyle=:dot)))
    rows = read_csv(joinpath(OUTPUT_DIR, "stability_$(file).csv"))
    rows === nothing && (println("No file stability_$(file).csv: run the \"before\" case first (see the header)."); continue)
    plot!(p, getfield.(rows, :t), getfield.(rows, :umax); label=label, linewidth=2, style...)
end
save_figure(p, "seiche_stability.png")

# 5. animation: horizontal velocity and interface, hydrostatic and non-hydrostatic side by side on the same clock
let models = [seiche_model(ANIMATION_GRID...; hydrostatic=h) for h in (true, false)]
    nx, nz = ANIMATION_GRID
    x = [(i - 0.5) * LENGTH / nx for i in 1:nx]
    z = [(k - 0.5) * DEPTH / nz for k in 1:nz]
    heavy_depth(m) = vec(sum((m.state.rho .- RHO_TOP) ./ (RHO_BOTTOM - RHO_TOP); dims=1)) .* m.dz   # m
    stride = round(Int, ANIMATION_FRAME_INTERVAL / DT)
    anim = Animation()
    for n in 0:round(Int, ANIMATION_T_END / DT)
        n > 0 && foreach(step_forward!, models)
        n % stride == 0 || continue
        panels = map(zip(models, ("hydrostatic", "non-hydrostatic"))) do (m, name)
            T = 2pi / reference(m.config.use_hydrostatic)[1]
            uc = 0.5 .* (m.state.u[:, 1:(end - 1)] .+ m.state.u[:, 2:end])
            p = heatmap(x, z, uc; color=cgrad(:balance, ANIMATION_COLOR_LEVELS; categorical=true), clims=(-ANIMATION_U_LIMIT, ANIMATION_U_LIMIT), colorbar_title="u (m/s)",
                xlabel="x (m)", ylabel="z (m)", titlefontsize=10,
                title=@sprintf("%s, T = %.1f s, t / T = %.2f", name, T, n * DT / T))
            plot!(p, x, heavy_depth(m); color=:black, linewidth=2, label="")
            p
        end
        frame(anim, plot(panels...; layout=(1, 2), size=ANIMATION_SIZE_PX, left_margin=FIGURE_MARGIN_MM,
            bottom_margin=FIGURE_MARGIN_MM, top_margin=FIGURE_MARGIN_MM, plot_title=@sprintf("Internal seiche, %d x %d cells, t = %.1f s (black: interface)",
            nx, nz, n * DT), plot_titlefontsize=11))
    end
    path = joinpath(OUTPUT_DIR, "seiche_2dv.gif")
    gif(anim, path; fps=ANIMATION_FPS)
    cp(path, joinpath(FIGURE_DIR, "seiche_2dv.gif"); force=true)
    println("Saved: $(abspath(path))")
end
