# Sediment plume eroded by a short jet pulse close to the bed (issue #64).
#
# A pipe of 3 m diameter in the left wall of a 200 m basin, 28 m deep, discharges for 3 min at 3 m/s, its lower edge 5 m
# above a mud bed. The jet erodes the mud where it reaches the bed (Partheniades erosion, src/sediment.jl), and the
# suspended sediment, heavier than the water, spreads as a plume. 2DV: the pipe is a slot over the layers from 5 m to 8 m
# above the bed, 9 m^2/s per metre of width (the 120 m^3/s of the issue spread over 13 m of basin width). In 2DV the
# discharge has to leave through the basin: a through-flow of 0.32 m/s over the whole depth during the pulse (README).
#
# Usage: julia --project=<path to the repository> experiments/jet_sediment/jet_sediment.jl <command>
#   run <dx_m> <mode>   one run (mode: nonhydrostatic or hydrostatic); saves snapshots and time series
#   runs                the run of the figures (dx 1 m, non-hydrostatic)
#   figures             figures from the saved runs
#   animation           the plume (dx 0.5 m, non-hydrostatic) from the saved snapshots
# Output in output_barotropic/jet_sediment/ (runs) and output_barotropic/jet_sediment_figures/, copies in figures/:
# - jet_sediment_snapshots.png  concentration and velocity at six times
# - jet_sediment_budget.png     eroded, suspended, deposited and outflowing sediment; largest bed shear stress
# - jet_sediment_bed.png        bed shear stress during the pulse and the bed change at the end against x
# - jet_sediment_runs.png       suspended mass, its spread from the pipe and its centre of mass over time
# - jet_sediment.gif            the plume over the run

# WHY: batch run without a display; same GR workstation choice as src/visualization.jl.
get!(ENV, "GKSwstype", "100")

include(joinpath(@__DIR__, "..", "common", "performance.jl"))   # performance overview at exit (issue #10)
using SweTransport2DV
using Plots
using Printf
using Serialization
using Statistics

# Basin and jet (issue #64)
const LENGTH = 200.0            # m
const DEPTH = 28.0              # m
const JET_DISCHARGE = 120.0     # m^3/s, pipe (issue #64)
const JET_VELOCITY = 3.0        # m/s, jet velocity
const JET_BOTTOM = 5.0          # m above the bed, lower edge of the pipe
const JET_HEIGHT = 3.0          # m, slot height = pipe diameter (6 layers of 0.5 m): 6 m^2/s per metre of width
const EQUIVALENT_WIDTH = JET_DISCHARGE / (JET_VELOCITY * JET_HEIGHT)   # m, 13.3: basin width the slot stands for
const PULSE_RAMP = 5.0          # s, ramps up and down
const PULSE_FULL = 170.0        # s, full discharge (3 min with the ramps)
const JET_INTENSITY = 0.05      # -, turbulence intensity of the pipe flow: k = 3/2 (I U)^2
const JET_LENGTH_SCALE = 0.07   # -, mixing length of the pipe flow over its size: eps = C_mu^(3/4) k^(3/2) / (0.07 D)
# Mud (issue #64 gives no values: typical of a soft mud bed, Winterwerp & van Kesteren 2004, ch. 7-8)
const SETTLING_VELOCITY = 5e-4  # m/s
const CRITICAL_STRESS = 0.2     # Pa
const EROSION_PARAMETER = 1e-4  # kg/m^2/s
const BED_Z0 = 2e-4             # m, roughness length of the mud bed
# Run
const RUN_TIME = 900.0          # s; settling (0.5 mm/s) moves the plume 0.45 m in this time
const SNAPSHOT_TIME = 10.0      # s
const SERIES_TIME = 1.0         # s
const ADVECTIVE_COURANT = 0.5   # -, JET_VELOCITY dt / dx (the jet accelerates a little: margin to 1)
const TAU = 20.0                # s, relaxed outflow
const NU = 1e-6                 # m^2/s
const RHO_WATER = 1000.0        # kg/m^3
const SNAPSHOT_TIMES = (60.0, 120.0, 180.0, 300.0, 600.0, 900.0)   # s, figure
const MASS_FRACTIONS = (0.5, 0.9)   # -, distances from the pipe within which these shares of the suspended mass lie
# WHY mass fractions, not a concentration threshold: the through-flow of the pulse lifts traces from the whole bed
const MG_PER_L = 1e3            # mg/l per kg/m^3
const C_LOG_RANGE = (-1.0, 2.5) # log10 mg/l, colour range of c (the plume reaches about 200 mg/l)
const ANIMATION_FPS = 8
const ANIMATION_STRIDE = 2      # snapshots per frame (20 s); GIF below 1 MB
const FIGURE_MARGIN_MM = 5Plots.mm
const RUN_DIR = joinpath(@__DIR__, "..", "..", "output_barotropic", "jet_sediment")
const OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "output_barotropic", "jet_sediment_figures")
const FIGURES_DIR = joinpath(@__DIR__, "figures")
const MODES = ("nonhydrostatic", "hydrostatic")
# WHY non-hydrostatic only: the offset jet attaches to the bed by the low pressure between jet and bed, which the
# hydrostatic model does not have (with the hydrostatic mode the erosion was a tenth; README)
# dx = 1 m: it gave the same erosion as 0.5 m within 10 % (6 m pipe, 2 m/s, 3 min; README)
const RUNS = ((1.0, "nonhydrostatic"),)
const DZ = 0.5                  # m

const PULSE_END = 2PULSE_RAMP + PULSE_FULL   # s

# Pipe velocity (m/s) at time t (s): linear ramps, full velocity in between
pulse(t) = t <= 0 ? 0.0 : t < PULSE_RAMP ? JET_VELOCITY * t / PULSE_RAMP :
           t <= PULSE_RAMP + PULSE_FULL ? JET_VELOCITY :
           t < PULSE_END ? JET_VELOCITY * (PULSE_END - t) / PULSE_RAMP : 0.0

run_tag(dx, mode) = @sprintf("dx%03d_%s", round(Int, 100dx), mode)
run_path(dx, mode) = joinpath(RUN_DIR, run_tag(dx, mode) * ".jls")
load_run(dx, mode) = isfile(run_path(dx, mode)) ? deserialize(run_path(dx, mode)) : nothing
cell_velocity(model) = 0.5 .* (model.state.u[:, 1:(end - 1)] .+ model.state.u[:, 2:end])

function save_figure(fig, name)
    mkpath(OUTPUT_DIR)
    mkpath(FIGURES_DIR)
    path = joinpath(OUTPUT_DIR, name)
    savefig(fig, path)
    cp(path, joinpath(FIGURES_DIR, name); force=true)
    @printf("wrote %s (%.0f kB)\n", path, filesize(path) / 1024)
end

function basin_model(dx, mode)
    mode in MODES || error("mode: $(join(MODES, " or "))")
    nx = round(Int, LENGTH / dx)
    nz = round(Int, DEPTH / DZ)
    cfg = ModelConfig(; length=LENGTH, depth=nz * DZ, nx=nx, nz=nz, dt=ADVECTIVE_COURANT * dx / JET_VELOCITY,
        use_hydrostatic=mode == "hydrostatic", nu=NU, rho0=RHO_WATER, turbulence_closure="k_epsilon",
        bed_roughness_length=BED_Z0, rayleigh_damping=0.0, apply_froude_limiter=false, momentum_advection_scheme="tvd2",
        transport_advection_scheme="tvd2", wetting_drying=true, implicit_vertical_terms=true, theta=1.0, write_plots=false)
    model = Model(cfg)
    initialize_water_level!(model, DEPTH; rho=RHO_WATER)
    zc = ((1:nz) .- 0.5) .* DZ
    jet = JET_BOTTOM .< zc .< JET_BOTTOM + JET_HEIGHT
    # pipe turbulence: k = 3/2 (I U)^2, eps = C_mu^(3/4) k^(3/2) / l with l = 0.07 D, D the slot height
    k_jet = 1.5 * (JET_INTENSITY * JET_VELOCITY)^2
    eps_jet = KE_C_MU_VALUE^0.75 * k_jet^1.5 / (JET_LENGTH_SCALE * JET_HEIGHT)
    k_in = [jet[k] ? k_jet : model.turbulence.k[k, 1] for k in 1:nz]
    eps_in = [jet[k] ? eps_jet : model.turbulence.eps[k, 1] for k in 1:nz]
    set_x_inflow_outflow!(model, t -> pulse(t) .* jet; right_level=0.0, right_free=true, right_relaxation_time=TAU,
        left_rho=RHO_WATER, left_k=k_in, left_eps=eps_in)
    sed = attach_sediment!(model; settling_velocity=SETTLING_VELOCITY, critical_erosion_stress=CRITICAL_STRESS,
        erosion_parameter=EROSION_PARAMETER, inflow_concentration=0.0)
    xc = ((1:nx) .- 0.5) .* dx
    return model, sed, xc, zc
end

function run_case(dx, mode)
    model, sed, xc, zc = basin_model(dx, mode)
    dt = model.config.dt
    steps = round(Int, RUN_TIME / dt)
    snap_every = round(Int, SNAPSHOT_TIME / dt)
    series_every = max(round(Int, SERIES_TIME / dt), 1)
    snaps = (; t=Float64[], c=Matrix{Float32}[], u=Matrix{Float32}[], h=Matrix{Float32}[])
    series = (; t=Float64[], suspended=Float64[], eroded=Float64[], deposited=Float64[], outflow=Float64[],
        tau_max=Float64[], x_tau_max=Float64[])
    tau_peak = zeros(length(xc))   # Pa, largest bed shear stress per column
    elapsed = @elapsed for n in 1:steps
        step_forward!(model)
        tau_peak .= max.(tau_peak, sed.tau_b)
        if n % series_every == 0
            push!(series.t, n * dt)
            push!(series.suspended, suspended_sediment_mass(model))
            push!(series.eroded, sed.eroded)
            push!(series.deposited, sed.deposited)
            push!(series.outflow, sed.outflow)
            j = argmax(sed.tau_b)
            push!(series.tau_max, sed.tau_b[j])
            push!(series.x_tau_max, xc[j])
        end
        if n % snap_every == 0
            push!(snaps.t, n * dt)
            push!(snaps.c, Float32.(sed.c))
            push!(snaps.u, Float32.(cell_velocity(model)))
            push!(snaps.h, Float32.(model.wetdry.h))
        end
    end
    result = (; dx, mode, xc, zc, dt, elapsed, snaps, series, tau_peak, bed_change=copy(sed.bed_change))
    mkpath(RUN_DIR)
    serialize(run_path(dx, mode), result)
    budget = series.suspended[end] - (series.eroded[end] - series.deposited[end] - series.outflow[end])
    @printf("%s: %.0f s wall clock; eroded %.1f kg/m, suspended %.1f, deposited %.2f, outflow %.2f; budget %.1e kg/m; peak tau_b %.1f Pa\n",
        run_tag(dx, mode), elapsed, series.eroded[end], series.suspended[end], series.deposited[end], series.outflow[end],
        budget, maximum(tau_peak))
    return result
end

# Plume diagnostics of a snapshot: mass (kg/m), the distances (m) within which the MASS_FRACTIONS of it lie, centre of
# mass (x, z) (m)
function plume_metrics(r, n)
    c, h = r.snaps.c[n], r.snaps.h[n]
    dx = r.xc[2] - r.xc[1]
    column = vec(sum(c .* h; dims=1)) .* dx   # kg/m per column
    m = sum(column)
    cum = cumsum(column) ./ max(m, eps())
    within = [r.xc[min(searchsortedfirst(cum, f), length(cum))] for f in MASS_FRACTIONS]
    xm = sum(column .* r.xc) / max(m, eps())
    zm = sum(c .* h .* r.zc) * dx / max(m, eps())
    return m, within, xm, zm
end

function field_pair(r, n; title_prefix="")
    c, u, h = r.snaps.c[n], r.snaps.u[n], r.snaps.h[n]
    wet = h .> 0
    logc = [wet[k, i] && c[k, i] > 0 ? log10(MG_PER_L * c[k, i]) : NaN for k in axes(c, 1), i in axes(c, 2)]
    t = r.snaps.t[n]
    pc = heatmap(r.xc, r.zc, logc; color=:viridis, clims=C_LOG_RANGE, colorbar_title="log10 c (mg/l)",
        title=@sprintf("%st = %.0f s: suspended sediment", title_prefix, t), titlefontsize=9, ylabel="z (m)", xlabel="x (m)")
    pu = heatmap(r.xc, r.zc, [wet[k, i] ? u[k, i] : NaN for k in axes(u, 1), i in axes(u, 2)]; color=:balance,
        clims=(-1.5, 1.5), colorbar_title="u (m/s)", title=@sprintf("t = %.0f s: u", t), titlefontsize=9, xlabel="x (m)")
    return pc, pu
end

function snapshots_figure()
    r = load_run(first(RUNS)...)
    r === nothing && return
    panels = Any[]
    for ts in SNAPSHOT_TIMES
        n = argmin(abs.(r.snaps.t .- ts))
        append!(panels, field_pair(r, n))
    end
    save_figure(plot(panels...; layout=(length(SNAPSHOT_TIMES), 2), size=(1400, 230 * length(SNAPSHOT_TIMES)),
        margin=FIGURE_MARGIN_MM), "jet_sediment_snapshots.png")
end

function budget_figure()
    r = load_run(first(RUNS)...)
    r === nothing && return
    s = r.series
    pm = plot(s.t, s.eroded; label="eroded", lw=2, xlabel="t (s)", ylabel="sediment (kg per m width)",
        title=@sprintf("sediment budget (dx = %.1f m, %s)", first(RUNS)...), titlefontsize=10, legend=:right)
    plot!(pm, s.t, s.suspended; label="suspended", lw=2)
    plot!(pm, s.t, s.deposited; label="deposited", lw=2)
    plot!(pm, s.t, s.outflow; label="through the outflow", lw=2)
    vspan!(pm, [0, PULSE_END]; color=:gray, alpha=0.15, label="pulse")
    pt = plot(s.t, s.tau_max; lw=2, xlabel="t (s)", ylabel="Pa", label="largest bed shear stress", yscale=:log10,
        title="bed shear stress", titlefontsize=10)
    hline!(pt, [CRITICAL_STRESS]; ls=:dash, color=:black, label="critical stress for erosion")
    save_figure(plot(pm, pt; layout=(1, 2), size=(1300, 450), margin=FIGURE_MARGIN_MM), "jet_sediment_budget.png")
end

function bed_figure()
    pt = plot(; xlabel="x (m)", ylabel="Pa", yscale=:log10, title="largest bed shear stress during the run", titlefontsize=10,
        legend=:topright)
    pb = plot(; xlabel="x (m)", ylabel="kg/m²", title="bed change at the end (deposited - eroded)", titlefontsize=10,
        legend=:bottomright)
    for (n, key) in enumerate(RUNS)
        r = load_run(key...)
        r === nothing && continue
        label = @sprintf("dx = %.1f m, %s", key...)
        plot!(pt, r.xc, max.(r.tau_peak, 1e-4); lw=1.5, color=n, label=label)
        plot!(pb, r.xc, r.bed_change; lw=1.5, color=n, label=label)
    end
    hline!(pt, [CRITICAL_STRESS]; ls=:dash, color=:black, label="critical stress")
    save_figure(plot(pt, pb; layout=(1, 2), size=(1300, 450), margin=FIGURE_MARGIN_MM), "jet_sediment_bed.png")
end

function runs_figure()
    pm = plot(; xlabel="t (s)", ylabel="kg/m", title="suspended sediment", titlefontsize=10, legend=:bottomright)
    pe = plot(; xlabel="t (s)", ylabel="m", title="50 % (dashed) and 90 % (solid) of the mass within x", titlefontsize=10,
        legend=false)
    pz = plot(; xlabel="t (s)", ylabel="m", title="centre of mass: x (solid), z (dashed)", titlefontsize=10, legend=false)
    for (n, key) in enumerate(RUNS)
        r = load_run(key...)
        r === nothing && continue
        metrics = [plume_metrics(r, j) for j in eachindex(r.snaps.t)]
        label = @sprintf("dx = %.1f m, %s", key...)
        plot!(pm, r.snaps.t, getindex.(metrics, 1); lw=1.5, color=n, label=label)
        plot!(pe, r.snaps.t, [m[2][2] for m in metrics]; lw=1.5, color=n)
        plot!(pe, r.snaps.t, [m[2][1] for m in metrics]; lw=1.5, color=n, ls=:dash)
        plot!(pz, r.snaps.t, getindex.(metrics, 3); lw=1.5, color=n)
        plot!(pz, r.snaps.t, getindex.(metrics, 4); lw=1.5, color=n, ls=:dash)
    end
    save_figure(plot(pm, pe, pz; layout=(1, 3), size=(1500, 420), margin=FIGURE_MARGIN_MM), "jet_sediment_runs.png")
end

function figures()
    snapshots_figure()
    budget_figure()
    bed_figure()
    runs_figure()
end

function animation()
    r = load_run(first(RUNS)...)
    r === nothing && error("run $(first(RUNS)) first")
    anim = Animation()
    for n in 1:ANIMATION_STRIDE:length(r.snaps.t)
        pc, pu = field_pair(r, n)
        frame(anim, plot(pc, pu; layout=(2, 1), size=(720, 450), margin=FIGURE_MARGIN_MM))
    end
    mkpath(OUTPUT_DIR)
    mkpath(FIGURES_DIR)
    path = joinpath(OUTPUT_DIR, "jet_sediment.gif")
    gif(anim, path; fps=ANIMATION_FPS)
    cp(path, joinpath(FIGURES_DIR, "jet_sediment.gif"); force=true)
    @printf("wrote %s (%.0f kB)\n", path, filesize(path) / 1024)
end

function main(args)
    isempty(args) && error("command: run <dx_m> <mode> | runs | figures | animation")
    cmd = args[1]
    if cmd == "run"
        run_case(parse(Float64, args[2]), args[3])
    elseif cmd == "runs"
        for key in RUNS
            run_case(key...)
        end
    elseif cmd == "figures"
        figures()
    elseif cmd == "animation"
        animation()
    else
        error("unknown command $cmd")
    end
end

abspath(PROGRAM_FILE) == @__FILE__() && main(ARGS)
