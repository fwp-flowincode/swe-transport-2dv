# Wind-driven flow with a constant viscosity compared with the exact steady solutions in cases.jl (uniform density and
# the two-layer double gyre), in the hydrostatic and the non-hydrostatic mode.
#
# Part `runs`: the 20 m basin of the main experiment from rest, wind ramped over WS_RAMP_S, constant viscosity
# WS_ANALYTIC_NU, 200 x 80 cells, T_END_S. Part `convergence`: the exact steady state in a 40 m basin, run on with
# nz = 5 ... 80 cells.
#
# The script writes to output_barotropic/wind_shear_analytic_figures/:
# - wind_analytic_<case>_<mode>.npz   mid-basin profiles, errors, mean surface and interface (case: uniform, two_layer)
# - wind_analytic_profiles.png        mean mid-basin u over the last AVERAGE_WINDOW_S against the exact profiles
# - wind_analytic_profiles.gif        mid-basin u against time with the exact steady profiles
# - wind_analytic_error.png           profile error against time (spin-up towards the steady state)
# - wind_analytic_slopes.png          mean surface and interface against the exact slopes
# - wind_analytic_convergence.npz     convergence errors
# - wind_analytic_convergence.png     steady profile error against dz
# The figures in figures/ are copies of these files (the GIF reduced in size).
#
# Usage (any working directory):
#   julia --project=<path to the repository> experiments/wind_shear/plot_wind_shear_analytic.jl [runs|convergence|figures|all]
# `figures` only redraws the figures from the .npz files. Run time: about 10 min (runs) and 5 min (convergence).

# WHY: batch run without a display; same GR workstation choice as src/visualization.jl.
get!(ENV, "GKSwstype", "100")

using SweTransport2DV
using Plots
using NPZ
using Printf
using Statistics

include(joinpath(@__DIR__, "cases.jl"))

const T_END_S = 3600.0                     # s, from rest; viscous time H^2/nu = 1600 s, internal seiche damped by ~2400 s
const AVERAGE_WINDOW_S = 600.0             # s, averaging window at the end (about two internal seiche periods)
const SAMPLE_INTERVAL_S = 10.0             # s, sampling interval of the profiles, errors and averages
const PROGRESS_INTERVAL_S = 600.0          # s, simulated time between progress lines
const PROFILE_FRAME_INTERVAL_S = 20.0      # s, simulated time between GIF frames (multiple of SAMPLE_INTERVAL_S)
const CONVERGENCE_GRIDS = (5, 10, 20, 40, 80)   # nz of the convergence runs
const CONVERGENCE_LENGTH_M = 40.0          # m, basin length of the convergence runs (end circulation decayed mid-basin)
const CONVERGENCE_NX = 40                  # -
const CONVERGENCE_T_END_S = Dict("uniform" => 600.0, "two_layer" => 2400.0)   # s, from the exact state
const CONVERGENCE_T_AVERAGE_S = Dict("uniform" => 200.0, "two_layer" => 600.0) # s
const CONVERGENCE_DT_S = Dict(5 => 0.1, 10 => 0.1, 20 => 0.1, 40 => 0.1, 80 => 0.05)   # s, viscous limit at nz = 80
const ANIMATION_FPS = 12                   # frames/s of the GIF
const PROFILES_SIZE_PX = (900, 460)        # px, two profile panels
const SERIES_SIZE_PX = (800, 400)          # px, error against time
const SLOPES_SIZE_PX = (900, 620)          # px, two slope panels
const CONVERGENCE_SIZE_PX = (620, 460)     # px, convergence figure
const FIGURE_MARGIN_MM = 4Plots.mm         # margin so that labels are not clipped
const CASES = ("uniform", "two_layer")
const CASE_TITLE = Dict("uniform" => "uniform density", "two_layer" => "two layers (999 / 1001 kg/m³)")
const CASE_COLOR = Dict("uniform" => :darkgreen, "two_layer" => :darkorange)
const MODES = Dict("hydro" => true, "nonhydro" => false)   # mode name => use_hydrostatic
const MODE_ORDER = ("hydro", "nonhydro")
const MODE_TITLE = Dict("hydro" => "hydrostatic", "nonhydro" => "non-hydrostatic")
const MODE_COLOR = Dict("hydro" => :steelblue, "nonhydro" => :crimson)
const MODE_STYLE = Dict("hydro" => :solid, "nonhydro" => :dash)

# WHY: anchored at this file, not the cwd, so outputs land in the project's output_barotropic/.
const OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "output_barotropic", "wind_shear_analytic_figures")

"""
    exact_solution(case) -> NamedTuple

Exact steady solution of `case` ("uniform" or "two_layer") for the wind stress, viscosity and layers of cases.jl.
"""
function exact_solution(case::AbstractString)
    if case == "uniform"
        return wind_steady_uniform(; tau=WS_WIND_STRESS_X, nu=WS_ANALYTIC_NU, depth=WS_DEPTH)
    end
    return wind_steady_two_layer(; tau=WS_WIND_STRESS_X, nu=WS_ANALYTIC_NU, h_lower=WS_INTERFACE_HEIGHT_FRACTION * WS_DEPTH,
        depth=WS_DEPTH, drho=WS_RHO_BOTTOM - WS_RHO_TOP)
end

"""
    centre_profile(u) -> Vector{Float64}

Mean of the face velocities `u` (m/s, nz x (nx+1)) over the middle quarter of the basin, per row (m/s). In the 20 m
basin this is x = 7.5-12.5 m, at least 1.9 depths from the end walls.
"""
function centre_profile(u::AbstractMatrix{<:Real})
    nz, nxp1 = size(u)
    mid = (3 * nxp1 ÷ 8 + 1):(5 * nxp1 ÷ 8 + 1)
    return [sum(@view u[k, mid]) / length(mid) for k in 1:nz]
end

"""
    run_from_rest(case, mode) -> String

Runs `case` in `mode` from rest with the wind ramp and writes its .npz (returned path).
"""
function run_from_rest(case::AbstractString, mode::AbstractString)
    solution = exact_solution(case)
    cfg = wind_analytic_config(; use_hydrostatic=MODES[mode], t_end=T_END_S, write_plots=false, write_animation=false,
        write_pressure_buoyancy_diagnostics=false)
    model = Model(cfg)
    initialize_wind_shear!(model)
    if case == "uniform"
        fill!(model.state.rho, WS_RHO_TOP)
        refresh_density_metadata!(model)
    end
    z = [(k - 0.5) * model.dz for k in 1:cfg.nz]        # m
    u_exact = solution.u.(z)                            # m/s
    u_scale = solution.u(WS_DEPTH)                      # m/s, exact surface velocity
    sample_stride = round(Int, SAMPLE_INTERVAL_S / cfg.dt)
    progress_stride = round(Int, PROGRESS_INTERVAL_S / cfg.dt)
    average_start_step = round(Int, (T_END_S - AVERAGE_WINDOW_S) / cfg.dt)
    times = [0.0]                                       # s
    profiles = [centre_profile(model.state.u)]          # m/s
    errors = [maximum(abs, profiles[1] .- u_exact) / u_scale]   # -
    eta_mean = zeros(cfg.nx)                            # m
    interface_mean = zeros(cfg.nx)                      # m
    n_mean = 0
    max_cfl = 0.0
    @printf("Constant-viscosity wind case %s (%s) | %d x %d cells, dt=%.3f s, t_end=%.0f s, nu=%.0e m^2/s\n",
        case, MODE_TITLE[mode], cfg.nx, cfg.nz, cfg.dt, T_END_S, WS_ANALYTIC_NU)
    t0 = time()
    for n in 1:cfg.nt
        cfg.wind_stress_x = wind_stress_at(n * cfg.dt)
        step_forward!(model)
        max_cfl = max(max_cfl, cfl_numbers(model.state.u, model.state.w, cfg.dt, model.dx, model.dz)[3])
        if n % sample_stride == 0
            push!(times, n * cfg.dt)
            push!(profiles, centre_profile(model.state.u))
            push!(errors, maximum(abs, profiles[end] .- u_exact) / u_scale)
            if n > average_start_step
                eta_mean .+= model.state.eta
                case == "two_layer" && (interface_mean .+= interface_heights(model.state.rho, model.dz))
                n_mean += 1
            end
        end
        if n % progress_stride == 0
            @printf("t=%6.0f s | profile error %.4f of the exact surface velocity | max CFL so far %.3f | wall %.0f s\n",
                n * cfg.dt, errors[end], max_cfl, time() - t0)
        end
    end
    eta_mean ./= n_mean
    interface_mean ./= n_mean
    window = times .> T_END_S - AVERAGE_WINDOW_S
    u_mean = vec(mean(reduce(hcat, profiles[window]); dims=2))
    @printf("CASE=wind_analytic_%s_%s FULL_WALL_SECONDS=%.1f | mean-profile error %.4f | max CFL %.3f | sum(eta) dx = %.1e m^2\n",
        case, mode, time() - t0, maximum(abs, u_mean .- u_exact) / u_scale, max_cfl, sum(model.state.eta) * model.dx)
    path = joinpath(OUTPUT_DIR, "wind_analytic_$(case)_$(mode).npz")
    npzwrite(path, Dict("t_s" => times, "u_centre_m_s" => reduce(hcat, profiles), "error_rel" => errors, "z_m" => z,
        "u_mean_m_s" => u_mean, "eta_mean_m" => eta_mean, "interface_mean_m" => interface_mean,
        "x_m" => [(i - 0.5) * model.dx for i in 1:cfg.nx]))
    return path
end

"""
    run_convergence()

Steady profile error against dz from the exact steady state (40 m basin), both cases and modes; writes the .npz.
"""
function run_convergence()
    out = Dict{String,Any}("nz" => collect(CONVERGENCE_GRIDS), "dz_m" => [WS_DEPTH / nz for nz in CONVERGENCE_GRIDS])
    for case in CASES, mode in MODE_ORDER
        solution = exact_solution(case)
        errors = Float64[]
        for nz in CONVERGENCE_GRIDS
            t_end, t_avg, dt = CONVERGENCE_T_END_S[case], CONVERGENCE_T_AVERAGE_S[case], CONVERGENCE_DT_S[nz]
            cfg = wind_analytic_config(; use_hydrostatic=MODES[mode], t_end=t_end, nx=CONVERGENCE_NX, nz=nz, dt=dt,
                length=CONVERGENCE_LENGTH_M, wind_stress_x=WS_WIND_STRESS_X, write_plots=false, write_animation=false,
                write_pressure_buoyancy_diagnostics=false)
            model = Model(cfg)
            initialize_wind_steady_state!(model, solution; two_layer=case == "two_layer")
            acc = zeros(nz)
            n_acc = 0
            for n in 1:cfg.nt
                step_forward!(model)
                if n * dt > t_end - t_avg
                    acc .+= centre_profile(model.state.u)
                    n_acc += 1
                end
            end
            z = [(k - 0.5) * model.dz for k in 1:nz]
            push!(errors, maximum(abs, acc ./ n_acc .- solution.u.(z)) / solution.u(WS_DEPTH))
            @printf("convergence %s %s nz=%d: error %.3e of the exact surface velocity\n", case, mode, nz, errors[end])
        end
        out["error_$(case)_$(mode)"] = errors
    end
    npzwrite(joinpath(OUTPUT_DIR, "wind_analytic_convergence.npz"), out)
    return nothing
end

"""
    profile_panel(case, profiles; title)

u(z) panel of `case`: the exact steady profile (black) and `profiles[mode]` (m/s at the cell centres z_m).
"""
function profile_panel(case::AbstractString, profiles::Dict, z::AbstractVector; title::AbstractString)
    solution = exact_solution(case)
    zf = range(0.0, WS_DEPTH, length=400)
    p = plot(xlabel="u (m/s)", ylabel="z (m)", title=title, titlefontsize=10, legend=:bottomright, ylims=(0.0, WS_DEPTH),
        xlims=case == "uniform" ? (-0.02, 0.03) : (-0.006, 0.014))
    vline!(p, [0.0]; color=:gray, label="")
    case == "two_layer" && hline!(p, [WS_INTERFACE_HEIGHT_FRACTION * WS_DEPTH]; color=:gray, linestyle=:dot, label="")
    plot!(p, solution.u.(zf), zf; color=:black, linewidth=3, alpha=0.5, label="exact steady")
    for m in MODE_ORDER
        haskey(profiles, m) && plot!(p, profiles[m], z; color=MODE_COLOR[m], linestyle=MODE_STYLE[m], linewidth=2, label=MODE_TITLE[m])
    end
    return p
end

"""
    figures()

All figures from the .npz files that exist.
"""
function figures()
    margins = (left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM)
    files = Dict((c, m) => joinpath(OUTPUT_DIR, "wind_analytic_$(c)_$(m).npz") for c in CASES for m in MODE_ORDER)
    if all(isfile, values(files))
        data = Dict(k => npzread(f) for (k, f) in files)
        z = data[("uniform", "hydro")]["z_m"]
        t = data[("uniform", "hydro")]["t_s"]

        panels = [profile_panel(c, Dict(m => data[(c, m)]["u_mean_m_s"] for m in MODE_ORDER), z;
            title=@sprintf("%s, mean %.0f–%.0f s", CASE_TITLE[c], T_END_S - AVERAGE_WINDOW_S, T_END_S)) for c in CASES]
        savefig(plot(panels...; layout=(1, 2), size=PROFILES_SIZE_PX, margins...,
            plot_title=@sprintf("Mid-basin u, constant ν = %.0e m²/s", WS_ANALYTIC_NU), plot_titlefontsize=11),
            joinpath(OUTPUT_DIR, "wind_analytic_profiles.png"))

        anim = Animation()
        stride = round(Int, PROFILE_FRAME_INTERVAL_S / SAMPLE_INTERVAL_S)
        for s in 1:stride:length(t)
            panels = [profile_panel(c, Dict(m => data[(c, m)]["u_centre_m_s"][:, s] for m in MODE_ORDER), z; title=CASE_TITLE[c]) for c in CASES]
            frame(anim, plot(panels...; layout=(1, 2), size=PROFILES_SIZE_PX, margins...,
                plot_title=@sprintf("Mid-basin u, t = %.0f s (constant ν, wind ramp %.0f s)", t[s], WS_RAMP_S), plot_titlefontsize=11))
        end
        gif(anim, joinpath(OUTPUT_DIR, "wind_analytic_profiles.gif"); fps=ANIMATION_FPS)

        p = plot(; xlabel="t (s)", ylabel="max |u − u_exact| / u_exact(surface)", yscale=:log10, legend=:topright,
            size=SERIES_SIZE_PX, margins..., title="Mid-basin profile error against the exact steady solution")
        for c in CASES, m in MODE_ORDER
            d = data[(c, m)]
            plot!(p, d["t_s"][2:end], max.(d["error_rel"][2:end], 1e-6); color=CASE_COLOR[c], linestyle=MODE_STYLE[m],
                linewidth=2, label="$(CASE_TITLE[c]), $(MODE_TITLE[m])")
        end
        savefig(p, joinpath(OUTPUT_DIR, "wind_analytic_error.png"))

        x = data[("uniform", "hydro")]["x_m"]
        x_mid = WS_LENGTH / 2
        p_eta = plot(xlabel="x (m)", ylabel="mean η (mm)", legend=:topleft, title="Surface (mean of the last $(Int(AVERAGE_WINDOW_S)) s)",
            titlefontsize=10)
        for c in CASES
            sol = exact_solution(c)
            plot!(p_eta, x, 1e3 .* sol.surface_slope .* (x .- x_mid); color=:black, linewidth=3, alpha=0.4,
                label=c == "uniform" ? "exact slope" : "")
            for m in MODE_ORDER
                plot!(p_eta, x, 1e3 .* data[(c, m)]["eta_mean_m"]; color=CASE_COLOR[c], linestyle=MODE_STYLE[m], linewidth=2,
                    label="$(CASE_TITLE[c]), $(MODE_TITLE[m])")
            end
        end
        sol = exact_solution("two_layer")
        h0 = WS_INTERFACE_HEIGHT_FRACTION * WS_DEPTH
        p_int = plot(xlabel="x (m)", ylabel="interface height (m)", legend=:topright, titlefontsize=10,
            title="Interface, two layers (mean of the last $(Int(AVERAGE_WINDOW_S)) s)")
        plot!(p_int, x, h0 .+ sol.interface_slope .* (x .- x_mid); color=:black, linewidth=3, alpha=0.4, label="exact slope")
        for m in MODE_ORDER
            plot!(p_int, x, data[("two_layer", m)]["interface_mean_m"]; color=MODE_COLOR[m], linestyle=MODE_STYLE[m],
                linewidth=2, label=MODE_TITLE[m])
        end
        savefig(plot(p_eta, p_int; layout=(2, 1), size=SLOPES_SIZE_PX, margins...), joinpath(OUTPUT_DIR, "wind_analytic_slopes.png"))
        println("Saved: wind_analytic_profiles.{png,gif}, wind_analytic_error.png, wind_analytic_slopes.png in $(abspath(OUTPUT_DIR))")
    else
        println("Run figures skipped: run part `runs` first.")
    end

    conv = joinpath(OUTPUT_DIR, "wind_analytic_convergence.npz")
    if isfile(conv)
        d = npzread(conv)
        dz = d["dz_m"]
        p = plot(; xlabel="dz (m)", ylabel="max |u − u_exact| / u_exact(surface)", xscale=:log10, yscale=:log10,
            legend=:bottomright, size=CONVERGENCE_SIZE_PX, margins..., title="Steady profile error (from the exact state)",
            titlefontsize=10)
        for c in CASES, m in MODE_ORDER
            plot!(p, dz, d["error_$(c)_$(m)"]; color=CASE_COLOR[c], linestyle=MODE_STYLE[m], marker=:circle, linewidth=2,
                label="$(CASE_TITLE[c]), $(MODE_TITLE[m])")
        end
        e0 = d["error_uniform_hydro"][1]
        plot!(p, dz, e0 .* (dz ./ dz[1]) .^ 2; color=:gray, linestyle=:dot, label="second order")
        savefig(p, joinpath(OUTPUT_DIR, "wind_analytic_convergence.png"))
        println("Saved: wind_analytic_convergence.png")
    end
    return nothing
end

mkpath(OUTPUT_DIR)
what = isempty(ARGS) ? "all" : ARGS[1]
if what in ("runs", "all")
    for c in CASES, m in MODE_ORDER
        run_from_rest(c, m)
    end
end
what in ("convergence", "all") && run_convergence()
what in ("runs", "convergence", "figures", "all") || error("unknown argument $(what); use runs, convergence, figures or all")
figures()
