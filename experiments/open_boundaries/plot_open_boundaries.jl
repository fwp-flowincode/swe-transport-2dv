# Open boundaries (issues #46 and #49): a wave leaves the channel through the free outflow (right_free) and is reflected
# by a prescribed level (right_level) and a closed wall; hydrostatic long wave and non-hydrostatic dispersive wave, fixed
# grid and wetting-drying step; uniform channel flow through discharge inlet and level outlet in the non-hydrostatic mode.
# The set-ups are those of test/test_open_boundaries.jl. The non-hydrostatic fixed grid uses the Dirichlet top
# (nonhydro_surface_dirichlet=true): with the default Neumann top its surface waves are not dispersive (docs section 14).
#
# The script writes to output_barotropic/open_boundaries_figures/ and copies the files to figures/:
# - open_boundaries_wave.gif         dispersive wave (non-hydrostatic) against the right boundary, fixed grid and
#                                    wetting-drying step: free outflow, prescribed level, wall, and the reference
# - open_boundaries_pressure.gif     non-hydrostatic pressure q(x, z) near the free outflow (dq/dx = 0 on the open face);
#   open_boundaries_pressure.png     its frame at t = 3 s
# - open_boundaries_reflection.png   gauge record near the outlet and the channel-wide deviation from the reference
# - open_boundaries_xt.png           x-t diagrams of the surface: hydrostatic long wave and dispersive wave
# - open_boundaries_uniform.png      non-hydrostatic uniform channel flow stays at round-off
#
# Usage: julia --project=<path to the repository> experiments/open_boundaries/plot_open_boundaries.jl [product ...]
# (products: wave pressure reflection xt uniform; default all). Run time: a few minutes for all.

# WHY: batch run without a display; same GR workstation choice as src/visualization.jl.
get!(ENV, "GKSwstype", "100")

using SweTransport2DV
using Plots
using Printf

const G = 9.81                  # m/s^2, ModelConfig default
const RHO0 = 1000.0             # kg/m^3

# Non-hydrostatic dispersive wave (test_validate_nonhydro_free_outflow_transmits_dispersive_wave): Gaussian hump of
# width 1 m (k H ~ 1) in 1 m depth, at 3/4 of the 40 m channel; the reference is a twice as long closed channel.
const NH_DEPTH = 1.0            # m
const NH_LENGTH = 40.0          # m
const NH_DX = 0.2               # m
const NH_NZ = 10
const NH_SIGMA = 1.0            # m
const NH_AMPLITUDE = 0.01       # m
const NH_DT = 0.02              # s
const NH_RUN_TIME = 12.0        # s, ~ L / sqrt(g H): a reflection has crossed the channel
const NH_GAUGE_X = 36.0         # m, between the start of the hump (30 m) and the outlet
const NH_PRESSURE_XMIN = 24.0   # m, left end of the pressure frames
const NH_SURFACE_EXAGGERATION = 10.0   # -, of eta in the pressure frames
const NH_PRESSURE_SNAPSHOT = 3.0       # s, time of the pressure snapshot (the wave crest at the outlet)
# Hydrostatic long wave (test_validate_free_outflow_transmits_long_wave): A = 1 cm in 4 m depth, right-going, 1000 m.
const LW_DEPTH = 4.0            # m
const LW_LENGTH = 1000.0        # m
const LW_NX = 100
const LW_NZ = 4
const LW_SIGMA = 50.0           # m
const LW_AMPLITUDE = 0.01       # m
const LW_RUN_TIME = 320.0       # s, ~ 2 L / sqrt(g H)
# Uniform channel flow of issue #14 (test_verify_nonhydro_open_channel_uniform_flow_is_steady)
const UF_DEPTH, UF_SLOPE, UF_LENGTH = 4.0, 1e-4, 1000.0   # m, -, m
const UF_CHEZY = 50.0           # m^(1/2)/s
const UF_NU = 0.02              # m^2/s
const UF_NX, UF_NZ = 50, 8
const UF_DT = 10.0              # s
const UF_STEPS = 300
const UF_FLOOR = 1e-17          # m/s and m, plotted instead of an exact zero on the log axis

const THETA = 0.7               # -, all wave runs (as in the tests)
const FRAME_STRIDE = 10         # steps between animation frames (0.2 s)
const ANIMATION_FPS = 12
const FRAME_SIZE_PX = (900, 560)
const FIGURE_MARGIN_MM = 5Plots.mm
const BOUNDARIES = ((:free, "free outflow", :crimson, :solid), (:level, "prescribed level", :darkorange, :solid),
    (:closed, "wall", :gray40, :dash))
const OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "output_barotropic", "open_boundaries_figures")
const FIGURES_DIR = joinpath(@__DIR__, "figures")

function wave_config(; nonhydro, wd, L)
    nonhydro || return ModelConfig(nx=LW_NX, nz=LW_NZ, length=L, depth=LW_DEPTH, dt=(L / LW_NX) / sqrt(G * LW_DEPTH),
        theta=THETA, implicit_vertical_terms=true, nu=1e-6, rayleigh_damping=0.0, apply_froude_limiter=false,
        momentum_advection_scheme="tvd2", wetting_drying=wd, bed_level=wd ? zeros(LW_NX) : nothing, write_plots=false)
    nx = round(Int, L / NH_DX)
    return ModelConfig(nx=nx, nz=NH_NZ, length=L, depth=NH_DEPTH, dt=NH_DT, theta=THETA, implicit_vertical_terms=true,
        nu=1e-6, rayleigh_damping=0.0, apply_froude_limiter=false, momentum_advection_scheme="tvd2", use_hydrostatic=false,
        pressure_solver_method="sparse", nonhydro_surface_dirichlet=!wd, wetting_drying=wd,
        bed_level=wd ? zeros(nx) : nothing, write_plots=false)
end

surface(model, H) = model.wetdry === nothing ? copy(model.state.eta) : water_depth(model) .- H
nonhydro_pressure(model) = model.wetdry === nothing ? copy(model.ws.p_corr) : copy(model.wetdry.q_nh)

# Runs a wave case and records the surface (m) every step and the non-hydrostatic pressure (Pa) every FRAME_STRIDE steps.
# Non-hydrostatic: hump at rest at 3/4 of NH_LENGTH (splits); hydrostatic: right-going Gaussian at 0.6 L.
function run_wave(; nonhydro, wd, boundary, L=nonhydro ? NH_LENGTH : LW_LENGTH)
    cfg = wave_config(; nonhydro=nonhydro, wd=wd, L=L)
    model = Model(cfg)
    fill!(model.state.rho, RHO0)
    refresh_density_metadata!(model)
    H, A, sigma = nonhydro ? (NH_DEPTH, NH_AMPLITUDE, NH_SIGMA) : (LW_DEPTH, LW_AMPLITUDE, LW_SIGMA)
    x0 = nonhydro ? 0.75 * NH_LENGTH : 0.6 * LW_LENGTH
    dx = L / cfg.nx
    xc = ((1:cfg.nx) .- 0.5) .* dx
    eta0 = A .* exp.(-((xc .- x0) ./ sigma) .^ 2)
    wd ? initialize_water_level!(model, H .+ eta0) : (model.state.eta .= eta0)
    if !nonhydro
        for f in 1:(cfg.nx + 1)
            model.state.u[:, f] .= A * exp(-(((f - 1) * dx - x0) / sigma)^2) * sqrt(G / H)
        end
    end
    boundary == :free && set_x_inflow_outflow!(model, zeros(cfg.nz); right_free=true)
    boundary == :level && set_x_inflow_outflow!(model, zeros(cfg.nz); open_right=false, right_level=0.0)
    nsteps = round(Int, (nonhydro ? NH_RUN_TIME : LW_RUN_TIME) / cfg.dt)
    eta = [surface(model, H)]
    q = [nonhydro_pressure(model)]
    for n in 1:nsteps
        step_forward!(model)
        push!(eta, surface(model, H))
        n % FRAME_STRIDE == 0 && push!(q, nonhydro_pressure(model))
    end
    return (x=xc, t=(0:nsteps) .* cfg.dt, eta=reduce(hcat, eta), q=q, dz=H / cfg.nz)
end

# WHY: several products use the same runs; each is computed once per script invocation.
const RUNS = Dict{Any,Any}()
cached_run(; nonhydro, wd, boundary, L=nonhydro ? NH_LENGTH : LW_LENGTH) =
    get!(() -> run_wave(; nonhydro=nonhydro, wd=wd, boundary=boundary, L=L), RUNS, (nonhydro, wd, boundary, L))

# Reference: the twice as long closed channel, cut to the short one (nothing comes back from its far end in the run time)
function nh_reference(wd)
    r = cached_run(; nonhydro=true, wd=wd, boundary=:closed, L=2 * NH_LENGTH)
    n = round(Int, NH_LENGTH / NH_DX)
    return (x=r.x[1:n], t=r.t, eta=r.eta[1:n, :])
end
deviation(r, ref) = vec(maximum(abs.(r.eta .- ref.eta); dims=1)) ./ NH_AMPLITUDE

function save_figure(fig, name)
    path = joinpath(OUTPUT_DIR, name)
    savefig(fig, path)
    cp(path, joinpath(FIGURES_DIR, name); force=true)
    println("wrote ", path)
end

function save_gif(anim, name)
    path = joinpath(OUTPUT_DIR, name)
    gif(anim, path; fps=ANIMATION_FPS)
    cp(path, joinpath(FIGURES_DIR, name); force=true)
    @printf("wrote %s (%.0f kB)\n", path, filesize(path) / 1024)
end

grid_name(wd) = wd ? "wetting-drying step" : "fixed grid (Dirichlet top)"

function wave_animation()
    runs = Dict((wd, b) => cached_run(; nonhydro=true, wd=wd, boundary=b) for wd in (false, true), (b, _, _, _) in BOUNDARIES)
    refs = Dict(wd => nh_reference(wd) for wd in (false, true))
    anim = Animation()
    for n in 1:FRAME_STRIDE:length(refs[false].t)
        panels = map((false, true)) do wd
            ref = refs[wd]
            p = plot(ref.x, 100 .* ref.eta[:, n]; color=:black, lw=3, alpha=0.35, label="reference (no boundary)",
                ylims=(-0.4, 0.7), xlims=(0, NH_LENGTH), xlabel="x (m)", ylabel="η (cm)", legend=:topleft,
                title=@sprintf("%s, t = %4.1f s", grid_name(wd), ref.t[n]))
            for (b, label, color, style) in BOUNDARIES
                plot!(p, ref.x, 100 .* runs[(wd, b)].eta[:, n]; color=color, ls=style, lw=1.5, label=label)
            end
            vline!(p, [NH_LENGTH]; color=:black, lw=2, label="")
            p
        end
        frame(anim, plot(panels...; layout=(2, 1), size=FRAME_SIZE_PX, margin=FIGURE_MARGIN_MM,
            plot_title="Dispersive wave (non-hydrostatic) at the right boundary", plot_titlefontsize=12))
    end
    save_gif(anim, "open_boundaries_wave.gif")
end

function pressure_animation()
    runs = Dict(wd => cached_run(; nonhydro=true, wd=wd, boundary=:free) for wd in (false, true))
    r0 = runs[false]
    cols = findall(>=(NH_PRESSURE_XMIN), r0.x)
    zc = ((1:NH_NZ) .- 0.5) .* r0.dz
    qmax = maximum(maximum(abs, s[:, cols]) for r in values(runs) for s in r.q)
    anim = Animation()
    for m in eachindex(r0.q)
        n = (m - 1) * FRAME_STRIDE + 1
        panels = map((false, true)) do wd
            r = runs[wd]
            p = heatmap(r.x[cols], zc, r.q[m][:, cols]; color=:balance, clims=(-qmax, qmax), xlabel="x (m)",
                ylabel="z (m)", colorbar_title="q (Pa)", xlims=(NH_PRESSURE_XMIN, NH_LENGTH),
                ylims=(0, NH_DEPTH + 1.5 * NH_SURFACE_EXAGGERATION * NH_AMPLITUDE),
                title=@sprintf("%s, t = %4.1f s", grid_name(wd), r.t[n]))
            plot!(p, r.x[cols], NH_DEPTH .+ NH_SURFACE_EXAGGERATION .* r.eta[cols, n]; color=:black, lw=1.5,
                label=@sprintf("surface (η × %.0f)", NH_SURFACE_EXAGGERATION),
                legend=:bottomleft)
            vline!(p, [NH_LENGTH]; color=:black, lw=2, label="")
            p
        end
        fig = plot(panels...; layout=(2, 1), size=FRAME_SIZE_PX, margin=FIGURE_MARGIN_MM,
            plot_title="Non-hydrostatic pressure q, free outflow (dq/dx = 0)", plot_titlefontsize=12)
        frame(anim, fig)
        abs(r0.t[n] - NH_PRESSURE_SNAPSHOT) < 0.5 * FRAME_STRIDE * NH_DT && save_figure(fig, "open_boundaries_pressure.png")
    end
    save_gif(anim, "open_boundaries_pressure.gif")
end

function reflection_figure()
    gauges = map((false, true)) do wd
        ref = nh_reference(wd)
        ig = argmin(abs.(ref.x .- NH_GAUGE_X))
        p = plot(ref.t, 100 .* ref.eta[ig, :]; color=:black, lw=3, alpha=0.35, label="reference", xlabel="t (s)",
            ylabel="η (cm)", title=@sprintf("Gauge at x = %.0f m, %s", NH_GAUGE_X, grid_name(wd)), legend=:topright)
        for (b, label, color, style) in BOUNDARIES
            plot!(p, ref.t, 100 .* cached_run(; nonhydro=true, wd=wd, boundary=b).eta[ig, :]; color=color, ls=style, lw=1.5,
                label=label)
        end
        p
    end
    deviations = map((false, true)) do wd
        ref = nh_reference(wd)
        p = plot(yscale=:log10, ylims=(1e-5, 2), xlabel="t (s)", ylabel="max_x |η - η_ref| / A",
            title="Deviation from the reference, " * grid_name(wd), legend=:bottomright)
        for (b, label, color, style) in BOUNDARIES
            e = deviation(cached_run(; nonhydro=true, wd=wd, boundary=b), ref)
            plot!(p, ref.t[2:end], max.(e[2:end], 1e-6); color=color, ls=style, lw=1.5,
                label=@sprintf("%s (at t = %.1f s: %.2g)", label, ref.t[end], e[end]))
        end
        p
    end
    save_figure(plot(gauges..., deviations...; layout=(2, 2), size=(1200, 760), margin=FIGURE_MARGIN_MM),
        "open_boundaries_reflection.png")
end

function xt_figure()
    rows = map((false, true)) do nonhydro
        A = nonhydro ? NH_AMPLITUDE : LW_AMPLITUDE
        map(BOUNDARIES) do (b, label, _, _)
            r = cached_run(; nonhydro=nonhydro, wd=false, boundary=b)
            heatmap(r.x, r.t, permutedims(r.eta) ./ A; color=:balance, clims=(-1, 1), xlabel="x (m)", ylabel="t (s)",
                colorbar_title="η / A", title=(nonhydro ? "dispersive wave, " : "long wave (hydrostatic), ") * label)
        end
    end
    save_figure(plot(rows[1]..., rows[2]...; layout=(2, 3), size=(1500, 760), margin=FIGURE_MARGIN_MM),
        "open_boundaries_xt.png")
end

# Logarithmic Chezy profile of the uniform flow (as ob_profile in test/test_open_boundaries.jl)
function uniform_profile()
    dz = UF_DEPTH / UF_NZ
    P = [chezy_viscous_profile((k - 0.5) * dz, UF_CHEZY, UF_DEPTH, UF_SLOPE, UF_NU) for k in 1:UF_NZ]
    return P .+ (chezy_equilibrium_velocity(UF_CHEZY, UF_DEPTH, UF_SLOPE) - sum(P) / UF_NZ)
end

function uniform_figure()
    P = uniform_profile()
    p = plot(yscale=:log10, xlabel="t (h)", ylabel="deviation from uniform flow", legend=:right,
        title="Uniform channel flow, non-hydrostatic, discharge inlet and level outlet")
    for (wd, sd, name, color) in ((false, false, "fixed grid (Neumann top)", :steelblue),
            (false, true, "fixed grid (Dirichlet top)", :seagreen), (true, false, "wetting-drying step", :crimson))
        cfg = ModelConfig(nx=UF_NX, nz=UF_NZ, length=UF_LENGTH, depth=UF_DEPTH, dt=UF_DT, implicit_vertical_terms=true,
            chezy_coefficient=UF_CHEZY, nu=UF_NU, body_force_x=G * UF_SLOPE, rayleigh_damping=0.0,
            apply_froude_limiter=false, momentum_advection_scheme="tvd2", use_hydrostatic=false,
            pressure_solver_method="sparse", nonhydro_surface_dirichlet=sd, wetting_drying=wd,
            bed_level=wd ? zeros(UF_NX) : nothing, write_plots=false)
        model = Model(cfg)
        fill!(model.state.rho, RHO0)
        refresh_density_metadata!(model)
        for i in axes(model.state.u, 2)
            model.state.u[:, i] .= P
        end
        set_x_inflow_outflow!(model, P; left_discharge=UF_DEPTH, right_level=0.0)
        du = Float64[]
        deta = Float64[]
        for _ in 1:UF_STEPS
            step_forward!(model)
            push!(du, maximum(abs.(model.state.u .- P)))
            push!(deta, maximum(abs, surface(model, UF_DEPTH)))
        end
        t = (1:UF_STEPS) .* UF_DT ./ 3600
        plot!(p, t, max.(du, UF_FLOOR); color=color, lw=1.5, label="max |u - u_uniform| (m/s), " * name)
        plot!(p, t, max.(deta, UF_FLOOR); color=color, lw=1.5, ls=:dash, label="max |η| (m), " * name)
    end
    save_figure(plot(p; size=(900, 450), margin=FIGURE_MARGIN_MM), "open_boundaries_uniform.png")
end

const PRODUCTS = Dict("wave" => wave_animation, "pressure" => pressure_animation, "reflection" => reflection_figure,
    "xt" => xt_figure, "uniform" => uniform_figure)

mkpath(OUTPUT_DIR)
mkpath(FIGURES_DIR)
# all products, or those named on the command line (e.g. `... plot_open_boundaries.jl reflection xt`)
for name in (isempty(ARGS) ? ["wave", "pressure", "reflection", "xt", "uniform"] : ARGS)
    PRODUCTS[name]()
end
