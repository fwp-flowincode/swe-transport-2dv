# Recirculation behind obstacles, hydrostatic against non-hydrostatic, with k-epsilon (issue #50).
#
# The script writes to output_barotropic/obstacles_figures/ and copies the files to figures/:
# - recirculation_step.png       backward-facing step (case ST1): mean velocity and dividing streamline, both modes
# - recirculation_step_grids.png reattachment length against the grid, with the LES of Luo et al. (2023)
# - recirculation_boundary.png   length over time and spectrum: level boundary against relaxed outflow
# - recirculation_weir.png       weir with a steep downstream face: both modes (no reference)
# - recirculation_step.gif       the step, both modes
#
# Usage: julia --project=<path to the repository> experiments/obstacles/plot_recirculation.jl [product ...]
# (products: step grids boundary weir animation; default all). Run time: about 5 h for all, one after the other; the
# products can run in parallel.

# WHY: batch run without a display; same GR workstation choice as src/visualization.jl.
get!(ENV, "GKSwstype", "100")

include(joinpath(@__DIR__, "..", "common", "performance.jl"))   # performance overview at exit (issue #10)
using SweTransport2DV
using Plots
using Printf
using Statistics

# Case ST1 of Nakagawa & Nezu (1987), as in Luo et al. (2023, JFM 966 A18): expansion ratio 1.5, Re = 4380, Fr = 0.19
const S = 0.02                  # m, step height
const DEPTH = 0.06              # m, depth downstream of the step
const Q = 0.00876               # m^2/s, Re nu / S * (DEPTH - S)
const NU = 1e-6                 # m^2/s
const Z0 = 1e-5                 # m, smooth bed, nu / (9 u*)
const U_REF = 0.25              # m/s, largest velocity, for the time step
const COURANT = 0.4             # -
const X_STEP = 40S              # m, 40 step heights of development
const LENGTH = 70S              # m
const TAU = 20.0                # s, relaxed outflow: reflects < 3 % of the 8 s mode of the reach
const XR_LES = 6.15             # step heights, Luo et al. (2023)
const RES = 10                  # -, cells per step height
# WHY: margin only. With the relaxed outflow the hydrostatic run is stable at the full time step as well (largest
# advective Courant number 0.49 in 90 s); its blow-up with a fixed level is explained in docs section 28.
const HYDRO_DT_FACTOR = 0.5     # -
# WHY long runs: the start excites a mode of the reach that decays over about 40 s; means over shorter runs are not of
# the steady state
const RUN_TIME = 90.0           # s
const AVERAGE_TIME = 30.0       # s
const SERIES_AVERAGE_TIME = 80.0   # s, the boundary figure shows the decay
const GRIDS = ((5, 5), (10, 10), (10, 20), (20, 10))   # cells per step height, in x and z
const SAMPLE_TIME = 0.05        # s
const VIEW_UPSTREAM = 5S        # m
const VIEW_DOWNSTREAM = 16S     # m
# Weir: trapezoid of height S, ramp 1:3 upstream, crest 4 S, downstream face 1:1
const WEIR_START = 25S          # m
const WEIR_CREST = 4S           # m
const WEIR_DEPTH = 0.06         # m
const ANIMATION_FPS = 8
const FRAME_TIME = 1.0          # s
const ANIMATION_TIME = 30.0     # s
const FRAME_SIZE_PX = (640, 440)
const FIGURE_MARGIN_MM = 5Plots.mm
const OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "output_barotropic", "obstacles_figures")
const FIGURES_DIR = joinpath(@__DIR__, "figures")

function save_figure(fig, name)
    path = joinpath(OUTPUT_DIR, name)
    savefig(fig, path)
    cp(path, joinpath(FIGURES_DIR, name); force=true)
    @printf("wrote %s (%.0f kB)\n", path, filesize(path) / 1024)
end

const U_CLIMS = (-0.1, 0.3)     # m/s, colour range of the velocity fields
const GRADIENT_SAMPLES = 32     # -, colours per half of the diverging map

# Diverging colour map with its white middle at u = 0 inside the asymmetric range `clims` (m/s).
# WHY: the plain map puts white at the middle of clims (0.1 m/s); stretching each half keeps the reverse flow, a few
# cm/s, in full colour while white marks the dividing line between forward and reverse flow.
function zero_centred_gradient(clims; name=:balance, n=GRADIENT_SAMPLES)
    z = -clims[1] / (clims[2] - clims[1])
    g = cgrad(name)
    t = vcat(range(0, 0.5; length=n), range(0.5, 1; length=n)[2:end])
    pos = vcat(range(0, z; length=n), range(z, 1; length=n)[2:end])
    return cgrad([g[s] for s in t], pos)
end

step_bed(x) = x < X_STEP ? S : 0.0

function weir_bed(x)
    x0 = WEIR_START
    x1 = x0 + 3S
    x2 = x1 + WEIR_CREST
    x3 = x2 + S
    return x <= x0 ? 0.0 : x <= x1 ? (x - x0) / 3 : x <= x2 ? S : x <= x3 ? x3 - x : 0.0
end

function case_model(; hydro, bed, resx=RES, resz=RES, dt_factor=1.0, tau=TAU, depth=DEPTH)
    dx, dz = S / resx, S / resz
    nx, nz = round(Int, LENGTH / dx), round(Int, depth / dz)
    dt = dt_factor * COURANT * min(dx, S / RES) / U_REF
    xc = ((1:nx) .- 0.5) .* dx
    cfg = ModelConfig(; length=LENGTH, depth=depth, nx=nx, nz=nz, dt=dt, use_hydrostatic=hydro, nu=NU,
        turbulence_closure="k_epsilon", bed_roughness_length=Z0, rayleigh_damping=0.0, apply_froude_limiter=false,
        momentum_advection_scheme="tvd2", wetting_drying=true, implicit_vertical_terms=true, theta=1.0,
        bed_level=bed.(xc), write_plots=false)
    model = Model(cfg)
    fill!(model.state.rho, 1000.0)
    refresh_density_metadata!(model)
    initialize_water_level!(model, fill(depth, nx))
    if tau === nothing
        set_x_inflow_outflow!(model, ones(nz); left_discharge=Q, right_level=0.0)
    else
        set_x_inflow_outflow!(model, ones(nz); left_discharge=Q, right_level=0.0, right_free=true, right_relaxation_time=tau)
    end
    return model, xc
end

cell_velocity(model) = 0.5 .* (model.state.u[:, 1:(end - 1)] .+ model.state.u[:, 2:end])

# Instantaneous length from the bed velocity, step heights
function snapshot_length(model, xc, x_start)
    u = cell_velocity(model)
    wd = model.wetdry
    last = findlast(i -> xc[i] > x_start && u[findfirst(>(0.0), view(wd.h, :, i)), i] < 0.0, eachindex(xc))
    return last === nothing ? 0.0 : (xc[last] - x_start) / S
end

# Runs to run_time; time means over the last average_time: velocity, thickness, surface; series of the length
function run_mean!(model, xc, x_start, run_time, average_time; frame=nothing)
    dt = model.config.dt
    steps = round(Int, run_time / dt)
    start = steps - round(Int, average_time / dt)
    sample = max(round(Int, SAMPLE_TIME / dt), 1)
    ubar, hbar, wmax = zero(model.wetdry.h), zero(model.wetdry.h), 0.0
    etabar = zero(model.state.eta)
    ts, xrs, etas = Float64[], Float64[], Float64[]
    for n in 1:steps
        step_forward!(model)
        frame === nothing || frame(n)
        n > start || continue
        ubar .+= cell_velocity(model)
        hbar .+= model.wetdry.h
        etabar .+= model.state.eta
        wmax = max(wmax, maximum(abs, model.state.w))
        if n % sample == 0
            push!(ts, n * dt)
            push!(xrs, snapshot_length(model, xc, x_start))
            push!(etas, model.state.eta[1])
        end
    end
    m = steps - start
    return (; u=ubar ./ m, h=hbar ./ m, eta=etabar ./ m, wmax, ts, xrs, etas)
end

function speed_panel(model, xc, field, x_start; title, clims=U_CLIMS)
    dz = model.dz
    zc = ((1:model.config.nz) .- 0.5) .* dz
    drawn = [model.wetdry.bed[i] < k * dz - 1e-9 ? field[k, i] : NaN for k in axes(field, 1), i in axes(field, 2)]
    p = heatmap((xc .- x_start) ./ S, zc ./ S, drawn; color=zero_centred_gradient(clims), clims=clims, xlabel="(x - x_step) / S", ylabel="z / S",
        title=title, xlims=(-VIEW_UPSTREAM / S, VIEW_DOWNSTREAM / S), ylims=(0, DEPTH / S), colorbar_title="u (m/s)",
        titlefontsize=10)
    return p
end

function streamline!(p, model, xc, mean_flow, x_start)
    dz = model.dz
    zc = ((1:model.config.nz) .- 0.5) .* dz
    psi = cumsum(mean_flow.u .* mean_flow.h; dims=1)
    contour!(p, (xc .- x_start) ./ S, zc ./ S, psi; levels=[0.0], color=:black, linewidth=1.5, colorbar_entry=false)
    return p
end

function run_step(hydro)
    model, xc = case_model(; hydro=hydro, bed=step_bed, dt_factor=hydro ? HYDRO_DT_FACTOR : 1.0)
    return model, xc, run_mean!(model, xc, X_STEP, RUN_TIME, AVERAGE_TIME)
end

function report(name, mean_flow, xc, x_start)
    xr = reattachment_length(mean_flow.u, mean_flow.h, xc, x_start) / S
    @printf("%s: mean xR/S = %.2f, snapshots %.2f, std %.2f, range %.2f-%.2f, wmax %.3f m/s\n", name, xr,
        mean(mean_flow.xrs), std(mean_flow.xrs), minimum(mean_flow.xrs), maximum(mean_flow.xrs), mean_flow.wmax)
    flush(stdout)
    return xr
end

function step_figure()
    panels = Any[]
    surfaces = Any[]
    for (hydro, name) in ((false, "non-hydrostatic"), (true, "hydrostatic"))
        model, xc, m = run_step(hydro)
        xr = report(name, m, xc, X_STEP)
        p = speed_panel(model, xc, m.u, X_STEP; title=@sprintf("%s: mean u, dividing streamline, xR = %.2f S", name, xr))
        streamline!(p, model, xc, m, X_STEP)
        vline!(p, [XR_LES]; color=:black, linestyle=:dash, label="LES 6.15 S")
        push!(panels, p)
        push!(surfaces, (name, (xc .- X_STEP) ./ S, 1000 .* (m.eta .- m.eta[end])))
    end
    q = plot(; xlabel="(x - x_step) / S", ylabel="surface (mm, against the outlet)", xlims=(-VIEW_UPSTREAM / S, VIEW_DOWNSTREAM / S),
        title="mean water surface", titlefontsize=10, legend=:topright)
    for (name, x, e) in surfaces
        plot!(q, x, e; label=name, linewidth=1.5)
    end
    save_figure(plot(panels..., q; layout=(3, 1), size=(1000, 900), margin=FIGURE_MARGIN_MM), "recirculation_step.png")
end

function grids_figure()
    fig = plot(; xlabel="cells per step height", ylabel="xR / S", title="Reattachment length, non-hydrostatic",
        legend=:bottomright, xscale=:log2, titlefontsize=10)
    hline!(fig, [XR_LES]; color=:black, linestyle=:dash, label="LES, Luo et al. (2023)")
    for (resx, resz) in GRIDS
        model, xc = case_model(; hydro=false, bed=step_bed, resx=resx, resz=resz)
        m = run_mean!(model, xc, X_STEP, RUN_TIME, AVERAGE_TIME)
        xr = report("S/dx = $resx, S/dz = $resz", m, xc, X_STEP)
        scatter!(fig, [resz], [xr]; yerror=[std(m.xrs)], label="S/dx = $resx, S/dz = $resz", markersize=5)
    end
    save_figure(plot(fig; size=(800, 480), margin=FIGURE_MARGIN_MM), "recirculation_step_grids.png")
end

function spectrum(x, t)
    d = t[2] - t[1]
    y = x .- mean(x)
    n = length(y)
    freq = (1:(n ÷ 2)) ./ (n * d)
    amp = [2 / n * abs(sum(y .* cis.(-2pi * j .* (0:(n - 1)) ./ n))) for j in 1:(n ÷ 2)]
    return 1 ./ freq, amp
end

function boundary_figure()
    series = plot(; xlabel="t (s)", ylabel="xR / S", title="Length from the bed velocity", titlefontsize=10, legend=:topright)
    spec = plot(; xlabel="period (s)", ylabel="amplitude of xR / S", title="Spectrum", titlefontsize=10, xscale=:log10,
        xlims=(1, 40), legend=:topright)
    for (tau, name) in ((nothing, "water level"), (TAU, "relaxed outflow"))
        model, xc = case_model(; hydro=false, bed=step_bed, tau=tau)
        m = run_mean!(model, xc, X_STEP, RUN_TIME, SERIES_AVERAGE_TIME)
        report(name, m, xc, X_STEP)
        plot!(series, m.ts, m.xrs; label=name, linewidth=1)
        period, amp = spectrum(m.xrs, m.ts)
        plot!(spec, period, amp; label=name, linewidth=1.5)
    end
    vline!(spec, [4 * LENGTH / sqrt(9.81 * DEPTH)]; color=:black, linestyle=:dash, label="4 L / c")
    save_figure(plot(series, spec; layout=(1, 2), size=(1200, 420), margin=FIGURE_MARGIN_MM), "recirculation_boundary.png")
end

function weir_figure()
    x_start = WEIR_START + 3S + WEIR_CREST + S   # m, toe of the downstream face
    panels = Any[]
    for (hydro, name) in ((false, "non-hydrostatic"), (true, "hydrostatic"))
        model, xc = case_model(; hydro=hydro, bed=weir_bed, dt_factor=hydro ? HYDRO_DT_FACTOR : 1.0)
        m = run_mean!(model, xc, x_start, RUN_TIME, AVERAGE_TIME)
        xr = report("weir, $name", m, xc, x_start)
        p = speed_panel(model, xc, m.u, x_start; title=@sprintf("%s: mean u, xR = %.2f S from the toe", name, xr))
        streamline!(p, model, xc, m, x_start)
        push!(panels, p)
    end
    save_figure(plot(panels...; layout=(2, 1), size=(1000, 620), margin=FIGURE_MARGIN_MM), "recirculation_weir.png")
end

function step_animation()
    # WHY the two runs in lockstep: one frame shows both modes at the same time
    nh, xc = case_model(; hydro=false, bed=step_bed, dt_factor=HYDRO_DT_FACTOR)
    hy, _ = case_model(; hydro=true, bed=step_bed, dt_factor=HYDRO_DT_FACTOR)
    every = round(Int, FRAME_TIME / nh.config.dt)
    anim = Animation()
    for n in 1:round(Int, ANIMATION_TIME / nh.config.dt)
        step_forward!(nh)
        step_forward!(hy)
        n % every == 0 || continue
        t = n * nh.config.dt
        frame(anim, plot(speed_panel(nh, xc, cell_velocity(nh), X_STEP; title=@sprintf("non-hydrostatic, t = %.1f s", t)),
            speed_panel(hy, xc, cell_velocity(hy), X_STEP; title="hydrostatic"); layout=(2, 1), size=FRAME_SIZE_PX,
            margin=FIGURE_MARGIN_MM))
    end
    path = joinpath(OUTPUT_DIR, "recirculation_step.gif")
    gif(anim, path; fps=ANIMATION_FPS)
    cp(path, joinpath(FIGURES_DIR, "recirculation_step.gif"); force=true)
    @printf("wrote %s (%.0f kB)\n", path, filesize(path) / 1024)
end

const PRODUCTS = Dict("step" => step_figure, "grids" => grids_figure, "boundary" => boundary_figure,
    "weir" => weir_figure, "animation" => step_animation)

function main(args)
    mkpath(OUTPUT_DIR)
    for name in (isempty(args) ? ["step", "grids", "boundary", "weir", "animation"] : args)
        haskey(PRODUCTS, name) || error("unknown product $name; available: $(join(keys(PRODUCTS), ", "))")
        PRODUCTS[name]()
    end
end

main(ARGS)
