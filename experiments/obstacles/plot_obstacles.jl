# Flow over obstacles (issue #47): SWASHES bump (subcritical, transcritical, with hydraulic jump) and a broad-crested
# weir, frictionless, hydrostatic, 1D (wetting-drying step, nz = 1) and 2DV; tracer through the bump.
#
# The script writes to output_barotropic/obstacles_figures/ and copies the files to figures/:
# - obstacles_bump_profiles.png      water level over the bump: 1D, 2DV and the exact solution, three regimes
# - obstacles_bump_convergence.png   maximum depth error against dx (shock: away from the jump) and jump position error
# - obstacles_bump_sections.png      2DV velocity sections of the transcritical and the shock case
# - obstacles_weir.png               weir: free and submerged profiles; upstream energy against tailwater energy
# - obstacles_shock.gif              the hydraulic jump forms on the downstream slope of the bump
# - obstacles_tracer.gif             a tracer front passes the bump (2DV, transcritical)
#
# Usage: julia --project=<path to the repository> experiments/obstacles/plot_obstacles.jl [product ...]
# (products: profiles convergence sections weir shock tracer; default all). Run time: about 10 min for all.

# WHY: batch run without a display; same GR workstation choice as src/visualization.jl.
get!(ENV, "GKSwstype", "100")

using SweTransport2DV
using Plots
using Printf

const G = 9.81                  # m/s^2, ModelConfig default
const BUMP_LENGTH = 25.0        # m
const BUMP_CREST = 10.0         # m
const RUN_TIME = 200.0          # s, until steady
const COURANT = 0.4             # -, (|U| + sqrt(g h)) dt / dx of the reference solution
const NX = 250                  # -, dx = 0.1 m
const NZ_2DV = 8
const CASES = (
    (name="subcritical", q=4.42, outlet_depth=2.0, regime=:subcritical, theta=0.75, grid_depth=2.5),
    (name="transcritical", q=1.53, outlet_depth=nothing, regime=:transcritical, theta=1.0, grid_depth=1.2),
    (name="shock", q=0.18, outlet_depth=0.33, regime=:shock, theta=1.0, grid_depth=0.5),
)
const CONVERGENCE_NX = (63, 125, 250, 500)
const JUMP_MARGIN = 0.5         # m, points nearer to the jump are left out of the shock-case error
const WEIR_HEIGHT = 0.5         # m
const WEIR_Q = 1.0              # m^2/s
const WEIR_LENGTH = 40.0        # m
const WEIR_GRID_DEPTH = 1.6     # m
const WEIR_NX = 200
const WEIR_RUN_TIME = 300.0     # s
const WEIR_TAILWATERS = (0.85, 0.9, 1.0, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4)   # m
const ANIMATION_FPS = 8
const FRAME_SIZE_PX = (720, 420)
const FIGURE_SIZE_PX = (1200, 400)
const FIGURE_MARGIN_MM = 5Plots.mm
const OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "output_barotropic", "obstacles_figures")
const FIGURES_DIR = joinpath(@__DIR__, "figures")

reference(c, x) = bump_flow_depth(x, c.q; bed=swashes_bump_bed, x_crest=BUMP_CREST, regime=c.regime, outlet_depth=c.outlet_depth)
jump_position(c) = hydraulic_jump_position(c.q; bed=swashes_bump_bed, x_crest=BUMP_CREST, outlet_depth=c.outlet_depth,
    x_max=BUMP_LENGTH)
weir_bed(x) = x < 15 ? 0.0 : x < 18 ? WEIR_HEIGHT * (x - 15) / 3 : x <= 22 ? WEIR_HEIGHT : x < 25 ? WEIR_HEIGHT * (25 - x) / 3 : 0.0

# Model of the bump case `c` (see test/test_obstacles.jl: lake start, or for the transcritical case the reference level
# upstream of the crest and a film downstream). Returns the model, cell centres and reference depth.
function bump_model(c; nx=NX, nz=1, left_rho=nothing)
    dx = BUMP_LENGTH / nx
    xc = ((1:nx) .- 0.5) .* dx
    bed = swashes_bump_bed.(xc)
    href = [reference(c, x) for x in xc]
    dt = COURANT * dx / (maximum(c.q ./ href) + sqrt(G * maximum(href)))
    nu = (nz > 1 && c.regime == :shock) ? 1e-3 : 1e-4   # m^2/s; the 2DV jump needs more viscosity to become steady
    cfg = ModelConfig(nx=nx, nz=nz, length=BUMP_LENGTH, depth=c.grid_depth, dt=dt, theta=c.theta,
        implicit_vertical_terms=true, nu=nu, rayleigh_damping=0.0, apply_froude_limiter=false,
        momentum_advection_scheme="tvd2", transport_advection_scheme="tvd2", wetting_drying=true, bed_level=bed,
        write_plots=false)
    model = Model(cfg)
    fill!(model.state.rho, 1000.0)
    refresh_density_metadata!(model)
    if c.outlet_depth === nothing
        initialize_water_level!(model, [x <= BUMP_CREST ? href[i] + bed[i] : bed[i] + 0.01 for (i, x) in enumerate(xc)])
        set_x_inflow_outflow!(model, ones(nz); left_discharge=c.q, left_rho=left_rho, right_free=true)
    else
        initialize_water_level!(model, max.(c.outlet_depth, bed .+ 1e-3))
        set_x_inflow_outflow!(model, ones(nz); left_discharge=c.q, left_rho=left_rho, right_level=c.outlet_depth - c.grid_depth)
    end
    return model, xc, bed, href
end

function run_steady!(model, run_time)
    for _ in 1:round(Int, run_time / model.config.dt)
        step_forward!(model)
    end
    return model
end

function save_figure(fig, name)
    path = joinpath(OUTPUT_DIR, name)
    savefig(fig, path)
    cp(path, joinpath(FIGURES_DIR, name); force=true)
    println("wrote ", path)
end

function bump_profiles_figure()
    panels = map(CASES) do c
        model1, xc, bed, href = bump_model(c)
        run_steady!(model1, RUN_TIME)
        model2, _, _, _ = bump_model(c; nz=NZ_2DV)
        run_steady!(model2, RUN_TIME)
        xf = range(0, BUMP_LENGTH, length=1000)
        p = plot(xf, swashes_bump_bed.(xf); color=:saddlebrown, lw=2, fillrange=0, fillalpha=0.3, fillcolor=:saddlebrown,
            label="bed", xlabel="x (m)", ylabel="z (m)", title=@sprintf("%s, q = %.2f m²/s", c.name, c.q),
            legend=:right)
        plot!(p, xf, [reference(c, x) + swashes_bump_bed(x) for x in xf]; color=:black, lw=2, label="exact")
        scatter!(p, xc[1:3:end], (water_depth(model1) .+ bed)[1:3:end]; color=:crimson, ms=2.5, msw=0, label="1D")
        plot!(p, xc, water_depth(model2) .+ bed; color=:steelblue, lw=1.5, ls=:dash, label="2DV, nz = $(NZ_2DV)")
        p
    end
    save_figure(plot(panels...; layout=(1, 3), size=FIGURE_SIZE_PX, margin=FIGURE_MARGIN_MM), "obstacles_bump_profiles.png")
end

function convergence_figure()
    dxs = [BUMP_LENGTH / nx for nx in CONVERGENCE_NX]
    p1 = plot(xscale=:log10, yscale=:log10, xlabel="dx (m)", ylabel="max |h - h_exact| (m)", title="Depth error (1D)",
        legend=:topleft)
    jump_errors = Float64[]
    guide = nothing
    for (c, color) in zip(CASES, (:steelblue, :darkorange, :crimson))
        errors = map(collect(CONVERGENCE_NX)) do nx
            model, xc, bed, href = bump_model(c; nx=nx)
            run_steady!(model, RUN_TIME)
            h = water_depth(model)
            mask = c.regime == :shock ? abs.(xc .- jump_position(c)) .> JUMP_MARGIN : trues(nx)
            if c.regime == :shock
                xj = jump_position(c)
                hmid = 0.5 * (reference(c, xj - 1e-9) + reference(c, xj + 1e-9))
                i = findfirst(j -> xc[j] > BUMP_CREST && h[j] > hmid, eachindex(xc))
                push!(jump_errors, abs(xc[i - 1] + (hmid - h[i - 1]) / (h[i] - h[i - 1]) * (xc[i] - xc[i - 1]) - xj))
            end
            maximum(abs.(h[mask] .- href[mask]))
        end
        plot!(p1, dxs, errors; marker=:circle, color=color, label=c.name * (c.regime == :shock ? " (away from jump)" : ""))
        guide === nothing && (guide = errors[end] .* dxs ./ dxs[end])   # first order through the finest subcritical point
    end
    plot!(p1, dxs, guide; color=:gray, ls=:dash, label="first order")
    p2 = plot(dxs, jump_errors; marker=:circle, color=:crimson, xscale=:log10, yscale=:log10, xlabel="dx (m)",
        ylabel="|x_jump - x_exact| (m)", title="Jump position (shock case)", label="1D")
    plot!(p2, dxs, dxs ./ 2; color=:gray, ls=:dash, label="dx / 2")
    save_figure(plot(p1, p2; layout=(1, 2), size=(900, 400), margin=FIGURE_MARGIN_MM), "obstacles_bump_convergence.png")
end

# Cell-centred velocity of the wet cells (NaN: dry or empty), for sections
function cell_velocity(model)
    wd = model.wetdry
    nz, nx = model.config.nz, model.config.nx
    v = fill(NaN, nz, nx)
    for i in 1:nx, k in 1:nz
        wd.h[k, i] > 1e-6 && (v[k, i] = 0.5 * (model.state.u[k, i] + model.state.u[k, i + 1]))
    end
    return v
end

function sections_figure()
    panels = map((CASES[2], CASES[3])) do c
        model, xc, bed, _ = bump_model(c; nz=16)
        run_steady!(model, RUN_TIME)
        zc = ((1:model.config.nz) .- 0.5) .* model.dz
        p = heatmap(xc, zc, cell_velocity(model); color=:viridis, xlabel="x (m)", ylabel="z (m)", colorbar_title="u (m/s)",
            title="2DV velocity, " * c.name, xlims=(6, 16))
        plot!(p, xc, bed; color=:saddlebrown, lw=2, label="bed")
        plot!(p, xc, bed .+ water_depth(model); color=:white, lw=2, label="surface", legend=:topright)
        p
    end
    save_figure(plot(panels...; layout=(1, 2), size=(1100, 380), margin=FIGURE_MARGIN_MM), "obstacles_bump_sections.png")
end

function weir_run(ht; nz=1)
    dx = WEIR_LENGTH / WEIR_NX
    xc = ((1:WEIR_NX) .- 0.5) .* dx
    bed = weir_bed.(xc)
    dt = COURANT * dx / (WEIR_Q / critical_depth(WEIR_Q) + sqrt(G * WEIR_GRID_DEPTH))
    cfg = ModelConfig(nx=WEIR_NX, nz=nz, length=WEIR_LENGTH, depth=WEIR_GRID_DEPTH, dt=dt, theta=1.0,
        implicit_vertical_terms=true, nu=nz == 1 ? 1e-4 : 1e-3, rayleigh_damping=0.0, apply_froude_limiter=false,
        momentum_advection_scheme="tvd2", wetting_drying=true, bed_level=bed, write_plots=false)
    model = Model(cfg)
    fill!(model.state.rho, 1000.0)
    refresh_density_metadata!(model)
    initialize_water_level!(model, max.(ht, bed .+ 1e-3))
    set_x_inflow_outflow!(model, ones(nz); left_discharge=WEIR_Q, right_level=ht - WEIR_GRID_DEPTH)
    run_steady!(model, WEIR_RUN_TIME)
    return xc, bed, water_depth(model)
end

function weir_figure()
    E_free = WEIR_HEIGHT + 1.5 * critical_depth(WEIR_Q)
    p1 = plot(xlabel="x (m)", ylabel="z (m)", title="Broad-crested weir, q = 1 m²/s", legend=:topright)
    xf = range(0, WEIR_LENGTH, length=800)
    plot!(p1, xf, weir_bed.(xf); color=:saddlebrown, lw=2, fillrange=0, fillalpha=0.3, fillcolor=:saddlebrown, label="bed")
    E_up = Float64[]
    E_t = Float64[]
    for ht in WEIR_TAILWATERS
        xc, bed, h = weir_run(ht)
        i5 = argmin(abs.(xc .- 5.0))
        push!(E_up, h[i5] + WEIR_Q^2 / (2G * h[i5]^2))
        push!(E_t, ht + WEIR_Q^2 / (2G * ht^2))
        if ht in (0.9, 1.0, 1.3)
            plot!(p1, xc, bed .+ h; lw=2, label=@sprintf("tailwater %.1f m%s", ht, E_t[end] > E_free ? " (submerged)" : ""))
        end
    end
    p2 = scatter(E_t, E_up; color=:crimson, label="model (1D)", xlabel="tailwater energy (m)", ylabel="upstream energy (m)",
        title="Modular limit", legend=:topleft)
    Es = range(minimum(E_t), maximum(E_t), length=100)
    plot!(p2, Es, max.(Es, E_free); color=:black, lw=2, label="frictionless theory: max(E_t, p + 3/2 h_c)")
    save_figure(plot(p1, p2; layout=(1, 2), size=(1100, 400), margin=FIGURE_MARGIN_MM), "obstacles_weir.png")
end

function shock_animation()
    c = CASES[3]
    model, xc, bed, href = bump_model(c)
    xf = range(0, BUMP_LENGTH, length=1000)
    anim = Animation()
    stride = round(Int, 2.0 / model.config.dt)
    for n in 0:round(Int, 120.0 / model.config.dt)
        n > 0 && step_forward!(model)
        n % stride == 0 || continue
        p = plot(xf, swashes_bump_bed.(xf); color=:saddlebrown, lw=2, fillrange=0, fillalpha=0.3, fillcolor=:saddlebrown,
            label="bed", ylims=(0, 0.5), xlabel="x (m)", ylabel="z (m)", size=FRAME_SIZE_PX,
            title=@sprintf("Hydraulic jump on the bump, q = 0.18 m²/s, t = %3.0f s", n * model.config.dt))
        plot!(p, xf, [reference(c, x) + swashes_bump_bed(x) for x in xf]; color=:black, ls=:dash, label="exact steady")
        plot!(p, xc, bed .+ water_depth(model); color=:steelblue, lw=2, label="model (1D)")
        frame(anim, p)
    end
    path = joinpath(OUTPUT_DIR, "obstacles_shock.gif")
    gif(anim, path; fps=ANIMATION_FPS)
    cp(path, joinpath(FIGURES_DIR, "obstacles_shock.gif"); force=true)
    @printf("wrote %s (%.0f kB)\n", path, filesize(path) / 1024)
end

function tracer_animation()
    c = CASES[2]
    rho_in = 1000.001   # kg/m^3: a tracer (buoyancy negligible)
    model, xc, bed, _ = bump_model(c; nz=16)
    run_steady!(model, RUN_TIME)
    set_x_inflow_outflow!(model, ones(model.config.nz); left_discharge=c.q, left_rho=rho_in, right_free=true)
    zc = ((1:model.config.nz) .- 0.5) .* model.dz
    anim = Animation()
    stride = round(Int, 0.5 / model.config.dt)
    for n in 0:round(Int, 20.0 / model.config.dt)
        n > 0 && step_forward!(model)
        n % stride == 0 || continue
        wd = model.wetdry
        tracer = [wd.h[k, i] > 1e-6 ? (model.state.rho[k, i] - 1000.0) / (rho_in - 1000.0) : NaN
                  for k in 1:model.config.nz, i in 1:model.config.nx]
        p = heatmap(xc, zc, tracer; color=:viridis, clims=(0, 1), xlabel="x (m)", ylabel="z (m)", colorbar_title="tracer (-)",
            size=FRAME_SIZE_PX, title=@sprintf("Tracer over the bump (2DV, transcritical), t = %4.1f s", n * model.config.dt))
        plot!(p, xc, bed; color=:saddlebrown, lw=2, label="bed")
        plot!(p, xc, bed .+ water_depth(model); color=:white, lw=2, label="surface", legend=:topright)
        frame(anim, p)
    end
    path = joinpath(OUTPUT_DIR, "obstacles_tracer.gif")
    gif(anim, path; fps=ANIMATION_FPS)
    cp(path, joinpath(FIGURES_DIR, "obstacles_tracer.gif"); force=true)
    @printf("wrote %s (%.0f kB)\n", path, filesize(path) / 1024)
end

const PRODUCTS = Dict("profiles" => bump_profiles_figure, "convergence" => convergence_figure,
    "sections" => sections_figure, "weir" => weir_figure, "shock" => shock_animation, "tracer" => tracer_animation)

mkpath(OUTPUT_DIR)
mkpath(FIGURES_DIR)
# all products, or those named on the command line (e.g. `... plot_obstacles.jl weir tracer`)
for name in (isempty(ARGS) ? ["profiles", "convergence", "sections", "weir", "shock", "tracer"] : ARGS)
    PRODUCTS[name]()
end
