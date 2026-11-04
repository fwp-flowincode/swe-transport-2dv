# k-epsilon bed cell on the z-level grid: sensitivity to the position of the bed against the layers (issue #62).
#
# Same geometry and dz, bed (and surface) lifted by a fraction f of dz, so that the lowest wet cell has (1 - f) dz.
# Cases: backward-facing step ST1 (as plot_recirculation.jl) at S/5 and S/10, the weir with a 1:1 downstream face at
# S/10, and the uniform open channel; wall functions :local and :equilibrium (KEpsilonState.wall_function). A wall
# function is given as `local`, `equilibrium` (default wall-cell threshold) or `equilibrium@<threshold>`
# (KEpsilonState.wall_cell_min_fraction).
#
# Usage: julia --project=<path to the repository> experiments/obstacles/plot_bed_cell.jl <command>
#   run <case> <res> <f> <wall_function>   one case (step, weir: 90 s, non-hydrostatic); saves its means
#   runs                                   all cases of the figures, one after the other (about 10 h; run them in
#                                          parallel with `run` instead)
#   figures                                figures from the saved means (missing cases are left out)
#   channel                                uniform channel profiles (runs its own cases, about 5 min)
#   animation                              the step at f = 0.9, both wall functions (about 1 h)
# Output in output_barotropic/bed_cell/ (means) and output_barotropic/obstacles_figures/, copies in figures/:
# - bed_cell_shift.png     xR against f
# - bed_cell_fields.png    mean u and dividing streamline at f = 0 and 0.9, both wall functions (step S/10)
# - bed_cell_profiles.png  u, k, eps, nu_t near the bed behind the step (S/10)
# - bed_cell_channel.png   uniform channel: u, k, nu_t against z at three shifts
# - bed_cell_step.gif      the step at f = 0.9, both wall functions

# WHY: batch run without a display; same GR workstation choice as src/visualization.jl.
get!(ENV, "GKSwstype", "100")

using SweTransport2DV
using Plots
using Printf
using Serialization
using Statistics

# Case ST1 (plot_recirculation.jl, docs section 28)
const S = 0.02                  # m, step height
const DEPTH = 0.06              # m
const Q = 0.00876               # m^2/s
const NU = 1e-6                 # m^2/s
const Z0 = 1e-5                 # m, smooth bed
const U_REF = 0.25              # m/s, for the time step
const COURANT = 0.4             # -
const X_STEP = 40S              # m
const LENGTH = 70S              # m
const TAU = 20.0                # s, relaxed outflow
const XR_LES = 6.15             # step heights, Luo et al. (2023)
const RUN_TIME = 90.0           # s
const AVERAGE_TIME = 30.0       # s
const SAMPLE_TIME = 0.05        # s
const WEIR_TOE = 33S            # m, toe of the 1:1 face (plot_recirculation.jl)
const SHIFTS = (0.0, 0.25, 0.5, 0.75, 0.9)
const WEIR_SHIFTS = (0.0, 0.25, 0.5)
const WALL_FUNCTIONS = ("local", "equilibrium@0", "equilibrium@0.5", "equilibrium")
const WF_STYLE = Dict("local" => (:solid, :circle, 1), "equilibrium@0" => (:dash, :diamond, 2),
    "equilibrium@0.5" => (:dashdot, :utriangle, 3), "equilibrium" => (:dot, :square, 4))
const PROFILE_STATIONS = (1, 3, 5, 7)   # step heights behind the step
const FIELD_WALL_FUNCTIONS = ("equilibrium",)   # beside local in the fields figure and the animation

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
# (f, wall function, colour) of the profile figure
const PROFILE_CASES = ((0.0, "local", 1), (0.9, "local", 2), (0.0, "equilibrium", 3), (0.9, "equilibrium", 4))
# Uniform channel (test_validate_k_epsilon_wetting_drying_open_channel)
const CH_DEPTH, CH_SLOPE, CH_Z0, CH_NZ = 0.5, 1e-3, 1e-3, 20   # m, -, m, -
const CH_SHIFTS = (0.0, 0.5, 0.9)
const CH_TIME = 900.0           # s
const ANIMATION_SHIFT = 0.9
const ANIMATION_TIME = 30.0     # s, 40 s gave 1.2 MB
const FRAME_TIME = 1.0          # s
const ANIMATION_FPS = 6
const FRAME_SIZE_PX = (640, 480)
const FIGURE_MARGIN_MM = 5Plots.mm
const RUN_DIR = joinpath(@__DIR__, "..", "..", "output_barotropic", "bed_cell")
const OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "output_barotropic", "obstacles_figures")
const FIGURES_DIR = joinpath(@__DIR__, "figures")

function save_figure(fig, name)
    path = joinpath(OUTPUT_DIR, name)
    savefig(fig, path)
    cp(path, joinpath(FIGURES_DIR, name); force=true)
    @printf("wrote %s (%.0f kB)\n", path, filesize(path) / 1024)
end

step_bed(x) = x < X_STEP ? S : 0.0

function weir_bed(x)
    x0 = 25S
    x1 = x0 + 3S
    x2 = x1 + 4S
    x3 = x2 + S
    return x <= x0 ? 0.0 : x <= x1 ? (x - x0) / 3 : x <= x2 ? S : x <= x3 ? x3 - x : 0.0
end

x_start(case) = case == "step" ? X_STEP : WEIR_TOE
tag(case, res, f, wf) = @sprintf("%s_S%d_f%03d_%s", case, res, round(Int, 100f), replace(wf, "@" => "_t", "." => "p"))

# wall function and wall-cell threshold of a spec "local", "equilibrium" or "equilibrium@0.3"
function set_wall_function!(ke, wf)
    parts = split(wf, "@")
    ke.wall_function = Symbol(parts[1])
    length(parts) > 1 && (ke.wall_cell_min_fraction = parse(Float64, parts[2]))
    return ke
end
wf_label(wf) = wf == "equilibrium" ? @sprintf("equilibrium@%g (default)", SweTransport2DV.KE_WALL_CELL_MIN_FRACTION) : wf

# The case with the bed lifted by f dz: one layer more, so that the depth and the water above the bed are the same
function shifted_model(case, res, f, wf)
    dx = dz = S / res
    nz0 = round(Int, DEPTH / dz)
    nz = f == 0 ? nz0 : nz0 + 1
    lift = f * dz                   # m
    surface = DEPTH + lift          # m, above the bottom of the grid
    nx = round(Int, LENGTH / dx)
    xc = ((1:nx) .- 0.5) .* dx
    bed = case == "step" ? step_bed : weir_bed
    cfg = ModelConfig(; length=LENGTH, depth=nz * dz, nx=nx, nz=nz, dt=COURANT * dx / U_REF, use_hydrostatic=false, nu=NU,
        turbulence_closure="k_epsilon", bed_roughness_length=Z0, rayleigh_damping=0.0, apply_froude_limiter=false,
        momentum_advection_scheme="tvd2", wetting_drying=true, implicit_vertical_terms=true, theta=1.0,
        bed_level=bed.(xc) .+ lift, write_plots=false)
    model = Model(cfg)
    set_wall_function!(model.turbulence, wf)
    fill!(model.state.rho, 1000.0)
    refresh_density_metadata!(model)
    initialize_water_level!(model, fill(surface, nx))
    set_x_inflow_outflow!(model, ones(nz); left_discharge=Q, right_level=surface - nz * dz, right_free=true,
        right_relaxation_time=TAU)
    return model, xc, lift
end

cell_velocity(model) = 0.5 .* (model.state.u[:, 1:(end - 1)] .+ model.state.u[:, 2:end])

function snapshot_length(model, xc, x0)
    u = cell_velocity(model)
    h = model.wetdry.h
    last = findlast(i -> xc[i] > x0 && u[findfirst(>(0.0), view(h, :, i)), i] < 0.0, eachindex(xc))
    return last === nothing ? 0.0 : (xc[last] - x0) / S
end

function run_case(case, res, f, wf)
    model, xc, lift = shifted_model(case, res, f, wf)
    x0 = x_start(case)
    dt = model.config.dt
    steps = round(Int, RUN_TIME / dt)
    start = steps - round(Int, AVERAGE_TIME / dt)
    sample = max(round(Int, SAMPLE_TIME / dt), 1)
    ke = model.turbulence
    sums = Dict(name => zero(model.wetdry.h) for name in (:u, :h, :k, :eps, :nu_t))
    xrs = Float64[]
    for n in 1:steps
        step_forward!(model)
        n > start || continue
        sums[:u] .+= cell_velocity(model)
        sums[:h] .+= model.wetdry.h
        sums[:k] .+= ke.k
        sums[:eps] .+= ke.eps
        sums[:nu_t] .+= ke.nu_t
        n % sample == 0 && push!(xrs, snapshot_length(model, xc, x0))
    end
    m = steps - start
    means = Dict(name => v ./ m for (name, v) in sums)
    xr = reattachment_length(means[:u], means[:h], xc, x0) / S
    result = (; case, res, f, wf, xc, lift, dz=model.dz, bed=copy(model.wetdry.bed), xr, xrs, means...)
    mkpath(RUN_DIR)
    serialize(joinpath(RUN_DIR, tag(case, res, f, wf) * ".jls"), result)
    @printf("%s: xR/S = %.2f, snapshots %.2f, std %.2f\n", tag(case, res, f, wf), xr, mean(xrs), std(xrs))
    return result
end

function load_case(case, res, f, wf)
    path = joinpath(RUN_DIR, tag(case, res, f, wf) * ".jls")
    return isfile(path) ? deserialize(path) : nothing
end

function shift_figure()
    panels = Any[]
    for (case, res, shifts) in (("step", 5, SHIFTS), ("step", 10, SHIFTS), ("weir", 10, WEIR_SHIFTS))
        p = plot(; xlabel="bed shift f (lowest cell (1 - f) dz)", ylabel="xR / S", title="$case, dz = S/$res",
            titlefontsize=10, legend=(case == "weir" ? :bottomright : :topleft), xlims=(-0.05, 0.95),
            ylims=(case == "weir" ? (1.5, 6.5) : (:auto, :auto)))
        case == "step" && hline!(p, [XR_LES]; color=:black, linestyle=:dot, label="LES 6.15 S")
        for wf in WALL_FUNCTIONS
            runs = filter(!isnothing, [load_case(case, res, f, wf) for f in shifts])
            isempty(runs) && continue
            style, marker, color = WF_STYLE[wf]
            plot!(p, [r.f for r in runs], [r.xr for r in runs]; yerror=[std(r.xrs) for r in runs], linestyle=style,
                marker=marker, color=color, markercolor=color, markerstrokecolor=color, label=wf_label(wf), linewidth=1.5)
        end
        push!(panels, p)
    end
    save_figure(plot(panels...; layout=(1, 3), size=(1300, 420), margin=FIGURE_MARGIN_MM), "bed_cell_shift.png")
end

# z above the bed (m) of the wet-cell centres of column i, and the wet cells
function column(r, i)
    wet = findall(>(1e-9), view(r.h, :, i))
    zb = r.bed[i]
    z = Float64[]
    bottom = zb
    for k in wet
        push!(z, bottom + 0.5 * r.h[k, i] - zb)
        bottom += r.h[k, i]
    end
    return wet, z
end

function field_panel(r; title)
    x0 = x_start(r.case)
    nz = size(r.u, 1)
    zc = ((1:nz) .- 0.5) .* r.dz .- r.lift
    drawn = [r.h[k, i] > 1e-9 ? r.u[k, i] : NaN for k in 1:nz, i in eachindex(r.xc)]
    p = heatmap((r.xc .- x0) ./ S, zc ./ S, drawn; color=zero_centred_gradient(U_CLIMS), clims=U_CLIMS, xlims=(-2, 14), ylims=(0, 3),
        xlabel="(x - x_step) / S", ylabel="z / S", title=title, titlefontsize=9, colorbar_title="u (m/s)")
    psi = cumsum(r.u .* r.h; dims=1)
    contour!(p, (r.xc .- x0) ./ S, zc ./ S, psi; levels=[0.0], color=:black, linewidth=1.5, colorbar_entry=false)
    vline!(p, [XR_LES]; color=:black, linestyle=:dot, label="")
    return p
end

function fields_figure()
    panels = Any[]
    for f in (0.0, 0.9), wf in ("local", FIELD_WALL_FUNCTIONS...)
        r = load_case("step", 10, f, wf)
        r === nothing && continue
        push!(panels, field_panel(r; title=@sprintf("f = %.1f, %s: xR = %.2f S (std %.2f)", f, wf_label(wf), r.xr, std(r.xrs))))
    end
    isempty(panels) && return
    save_figure(plot(panels...; layout=(cld(length(panels), 2), 2), size=(1300, 260 * cld(length(panels), 2)),
        margin=FIGURE_MARGIN_MM), "bed_cell_fields.png")
end

function profiles_figure()
    rows = Any[]
    for xs in PROFILE_STATIONS
        pu = plot(; xlabel="u (m/s)", ylabel="z above bed / S", title="x = $xs S: u", titlefontsize=9, legend=false)
        pk = plot(; xlabel="k (m²/s²)", title="k", titlefontsize=9, xscale=:log10, legend=false)
        pe = plot(; xlabel="eps (m²/s³)", title="eps", titlefontsize=9, xscale=:log10, legend=false)
        pn = plot(; xlabel="nu_t (1e-5 m²/s)", title="nu_t", titlefontsize=9,
            legend=(xs == first(PROFILE_STATIONS) ? :topright : false), legendfontsize=7)
        for (f, wf, color) in PROFILE_CASES
            r = load_case("step", 10, f, wf)
            r === nothing && continue
            i = findfirst(>=(X_STEP + xs * S), r.xc)
            wet, z = column(r, i)
            style, marker, _ = WF_STYLE[wf]
            kw = (; linestyle=style, color=color, marker=marker, markersize=3, markercolor=color, markerstrokewidth=0,
                linewidth=1.5, label=@sprintf("f = %.1f, %s", f, wf_label(wf)))
            plot!(pu, r.u[wet, i], z ./ S; kw...)
            plot!(pk, max.(r.k[wet, i], 1e-10), z ./ S; kw...)
            plot!(pe, max.(r.eps[wet, i], 1e-14), z ./ S; kw...)
            plot!(pn, 1e5 .* r.nu_t[wet, i], z ./ S; kw...)
        end
        for p in (pu, pk, pe, pn)
            ylims!(p, (0, 1.5))
        end
        push!(rows, plot(pu, pk, pe, pn; layout=(1, 4)))
    end
    save_figure(plot(rows...; layout=(length(rows), 1), size=(1400, 330 * length(rows)), margin=FIGURE_MARGIN_MM),
        "bed_cell_profiles.png")
end

function figures()
    mkpath(OUTPUT_DIR)
    shift_figure()
    fields_figure()
    profiles_figure()
end

# Uniform open channel, discharge of the periodic fixed-grid channel, outlet at the normal depth
function channel_case(f, wf, q)
    dz = CH_DEPTH / CH_NZ
    nz = f == 0 ? CH_NZ : CH_NZ + 1
    lift = f * dz
    nx, dt = 80, 0.25
    cfg = ModelConfig(; length=80.0, depth=nz * dz, nx=nx, nz=nz, dt=dt, use_hydrostatic=true, nu=NU,
        body_force_x=9.81 * CH_SLOPE, turbulence_closure="k_epsilon", bed_roughness_length=CH_Z0, rayleigh_damping=0.0,
        apply_froude_limiter=false, momentum_advection_scheme="tvd2", wetting_drying=true, implicit_vertical_terms=true,
        theta=1.0, bed_level=lift == 0.0 ? nothing : fill(lift, nx), write_plots=false)
    model = Model(cfg)
    set_wall_function!(model.turbulence, wf)
    fill!(model.state.rho, 1000.0)
    refresh_density_metadata!(model)
    initialize_water_level!(model, fill(lift + CH_DEPTH, nx))
    set_x_inflow_outflow!(model, ones(nz); left_discharge=q, right_level=lift + CH_DEPTH - nz * dz)
    for _ in 1:round(Int, CH_TIME / dt)
        step_forward!(model)
    end
    i = 60
    wet = findall(>(1e-9), view(model.wetdry.h, :, i))
    h = model.wetdry.h[wet, i]
    z = cumsum(h) .- 0.5 .* h
    u = cell_velocity(model)[wet, i]
    return (; z, u, k=model.turbulence.k[wet, i], nu_t=model.turbulence.nu_t[wet, i], depth=sum(h))
end

function channel_figure()
    mkpath(OUTPUT_DIR)
    ref = Model(ModelConfig(; length=1.0, depth=CH_DEPTH, nx=4, nz=CH_NZ, dt=0.1, use_hydrostatic=true, periodic_x=true,
        variable_top_layer=false, pressure_solver_method="sparse", nu=NU, body_force_x=9.81 * CH_SLOPE,
        turbulence_closure="k_epsilon", bed_roughness_length=CH_Z0, rayleigh_damping=0.0, apply_froude_limiter=false,
        momentum_advection_scheme="tvd2", implicit_vertical_terms=true, write_plots=false))
    fill!(ref.state.rho, 1000.0)
    refresh_density_metadata!(ref)
    for _ in 1:round(Int, CH_TIME / 0.1)
        step_forward!(ref)
    end
    q = mean(ref.state.u[:, 1:(end - 1)]) * CH_DEPTH
    ustar = sqrt(9.81 * CH_SLOPE * CH_DEPTH)   # m/s
    pu = plot(; xlabel="u (m/s)", ylabel="z / H", title="u", titlefontsize=10, legend=:topleft)
    pk = plot(; xlabel="k sqrt(C_mu) / u*²", title="k", titlefontsize=10, legend=false)
    pn = plot(; xlabel="nu_t / (kappa u* H)", title="nu_t", titlefontsize=10, legend=false)
    zl = range(0.01, 1.0; length=100)
    plot!(pu, ustar / 0.41 .* log.(zl .* CH_DEPTH ./ CH_Z0), zl; color=:black, linestyle=:dot, label="log law")
    plot!(pn, zl .* (1 .- zl), zl; color=:black, linestyle=:dot, label="parabola")
    for f in CH_SHIFTS, wf in ("local", FIELD_WALL_FUNCTIONS...)
        c = channel_case(f, wf, q)
        style, marker, _ = WF_STYLE[wf]
        color = f == 0 ? 1 : f == 0.5 ? 2 : 3
        kw = (; linestyle=style, color=color, marker=marker, markersize=3, markercolor=color, markerstrokewidth=0,
            label=@sprintf("f = %.1f, %s", f, wf_label(wf)))
        plot!(pu, c.u, c.z ./ CH_DEPTH; kw...)
        plot!(pk, c.k .* sqrt(0.09) ./ ustar^2, c.z ./ CH_DEPTH; kw...)
        plot!(pn, c.nu_t ./ (0.41 * ustar * CH_DEPTH), c.z ./ CH_DEPTH; kw...)
        @printf("channel f = %.1f %-16s depth %.4f m, max nu_t %.3e\n", f, wf, c.depth, maximum(c.nu_t))
    end
    for p in (pu, pk, pn)
        ylims!(p, (0, 1))
    end
    save_figure(plot(pu, pk, pn; layout=(1, 3), size=(1300, 450), margin=FIGURE_MARGIN_MM), "bed_cell_channel.png")
end

function animation()
    mkpath(OUTPUT_DIR)
    # WHY in lockstep: one frame shows both wall functions at the same time
    a, xc, lift = shifted_model("step", 10, ANIMATION_SHIFT, "local")
    b, _, _ = shifted_model("step", 10, ANIMATION_SHIFT, first(FIELD_WALL_FUNCTIONS))
    dt = a.config.dt
    every = round(Int, FRAME_TIME / dt)
    anim = Animation()
    panel(m, title) = field_panel((; case="step", xc, lift, dz=m.dz, u=cell_velocity(m), h=m.wetdry.h); title)
    for n in 1:round(Int, ANIMATION_TIME / dt)
        step_forward!(a)
        step_forward!(b)
        n % every == 0 || continue
        t = n * dt
        frame(anim, plot(panel(a, @sprintf("f = %.1f, local, t = %.0f s", ANIMATION_SHIFT, t)),
            panel(b, wf_label(first(FIELD_WALL_FUNCTIONS))); layout=(2, 1), size=FRAME_SIZE_PX, margin=FIGURE_MARGIN_MM))
    end
    path = joinpath(OUTPUT_DIR, "bed_cell_step.gif")
    gif(anim, path; fps=ANIMATION_FPS)
    cp(path, joinpath(FIGURES_DIR, "bed_cell_step.gif"); force=true)
    @printf("wrote %s (%.0f kB)\n", path, filesize(path) / 1024)
end

function all_runs()
    for wf in ("local", FIELD_WALL_FUNCTIONS...)
        for res in (5, 10), f in SHIFTS
            run_case("step", res, f, wf)
        end
        for f in WEIR_SHIFTS
            run_case("weir", 10, f, wf)
        end
    end
end

function main(args)
    isempty(args) && error("command: run <case> <res> <f> <wall_function> | runs | figures | channel | animation")
    cmd = args[1]
    if cmd == "run"
        run_case(args[2], parse(Int, args[3]), parse(Float64, args[4]), args[5])
    elseif cmd == "runs"
        all_runs()
    elseif cmd == "figures"
        figures()
    elseif cmd == "channel"
        channel_figure()
    elseif cmd == "animation"
        animation()
    else
        error("unknown command $cmd")
    end
end

main(ARGS)
