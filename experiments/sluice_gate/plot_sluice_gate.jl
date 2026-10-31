# Sluice gate (issue #48): free and submerged flow under a vertical gate, hydrostatic with the contraction coefficient
# C_c = 0.61 (1D and 2DV), and non-hydrostatic, where the jet contracts by itself (contraction=1).
#
# The script writes to output_barotropic/sluice_gate_figures/ and copies the files to figures/:
# - sluice_gate_profiles.png      water level, free and submerged flow: 1D, 2DV and the classical gate relations
# - sluice_gate_modular.png       upstream energy head against the tailwater depth (modular limit)
# - sluice_gate_contraction.png   non-hydrostatic: velocity section at the gate, contraction coefficient against resolution
# - sluice_gate_nonhydro.gif      non-hydrostatic: the jet under the gate forms and contracts
#
# Usage: julia --project=<path to the repository> experiments/sluice_gate/plot_sluice_gate.jl [product ...]
# (products: profiles modular contraction animation; default all). Run time: about 30 min for all (the finest
# non-hydrostatic run takes most of it).

# WHY: batch run without a display; same GR workstation choice as src/visualization.jl.
get!(ENV, "GKSwstype", "100")

using SweTransport2DV
using Plots
using Printf

const G = 9.81                  # m/s^2, ModelConfig default
const Q = 0.15                  # m^2/s, specific discharge
const OPENING = 0.1             # m, gate opening a
const CC = 0.61                 # -, contraction coefficient of the hydrostatic runs (Henderson 1966, ch. 6)
const GRID_DEPTH = 0.6          # m
const LENGTH = 20.0             # m
const X_GATE = 8.0              # m
const X_UPSTREAM = 3.2          # m, section of the upstream energy head
const NX = 100                  # -, dx = 0.2 m
const NZ_2DV = 8
const RUN_TIME = 150.0          # s, until steady
const COURANT = 0.3             # -, (u_jet + sqrt(g H)) dt / dx
const NU_1D = 1e-4              # m^2/s
const NU_2DV_SUBMERGED = 3e-3   # m^2/s, the hydrostatic roller under the dead water becomes steady (docs section 26)
const TAILWATERS = (0.15, 0.2, 0.25, 0.28, 0.31, 0.34, 0.37, 0.4)   # m
const PROFILE_TAILWATERS = (nothing, 0.34)   # free outflow, submerged
# Non-hydrostatic runs: 6 m channel, gate at 2.4 m, refined in x and z together (dx = dz)
const NH_LENGTH = 6.0           # m
const NH_X_GATE = 2.4           # m
const NH_GRIDS = ((120, 12), (240, 24), (480, 48))
const NH_RUN_TIME = 40.0        # s
const NH_NU = 1e-4              # m^2/s
const ANIMATION_GRID = (240, 24)
const ANIMATION_TIME = 20.0     # s
const ANIMATION_FRAME_TIME = 0.5   # s
const ANIMATION_FPS = 8
const FRAME_SIZE_PX = (760, 420)
const FIGURE_MARGIN_MM = 5Plots.mm
const OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "output_barotropic", "sluice_gate_figures")
const FIGURES_DIR = joinpath(@__DIR__, "figures")

# Gate model (as test/test_sluice_gate.jl): discharge inlet, gate, free outflow (tailwater `nothing`) or outlet level.
function gate_model(; tailwater=nothing, nx=NX, nz=1, nu=NU_1D, length=LENGTH, x_gate=X_GATE, nonhydro=false,
    contraction=nonhydro ? 1.0 : CC)
    dx = length / nx
    xc = ((1:nx) .- 0.5) .* dx
    dt = COURANT * dx / (Q / (contraction * OPENING) + sqrt(G * GRID_DEPTH))
    cfg = ModelConfig(nx=nx, nz=nz, length=length, depth=GRID_DEPTH, dt=dt, theta=1.0, implicit_vertical_terms=true,
        nu=nu, rayleigh_damping=0.0, apply_froude_limiter=false, momentum_advection_scheme="tvd2", wetting_drying=true,
        use_hydrostatic=!nonhydro, write_plots=false)
    model = Model(cfg)
    fill!(model.state.rho, 1000.0)
    refresh_density_metadata!(model)
    E = sluice_gate_free_head(Q, OPENING; contraction=contraction)
    initialize_water_level!(model, [x < x_gate ? E : (tailwater === nothing ? 0.01 : tailwater) for x in xc])
    set_sluice_gate!(model, x_gate; lower_edge=OPENING, contraction=contraction)
    if tailwater === nothing
        set_x_inflow_outflow!(model, ones(nz); left_discharge=Q, right_free=true)
    else
        set_x_inflow_outflow!(model, ones(nz); left_discharge=Q, right_level=tailwater - GRID_DEPTH)
    end
    return model, xc
end

function run_steady!(model, run_time)
    for _ in 1:round(Int, run_time / model.config.dt)
        step_forward!(model)
    end
    return model
end

energy_head(h) = h + Q^2 / (2G * h^2)
upstream_head(model, xc, x) = energy_head(water_depth(model)[argmin(abs.(xc .- x))])
theory_head(tailwater) = tailwater === nothing ? sluice_gate_free_head(Q, OPENING; contraction=CC) :
    sluice_gate_submerged_head(Q, OPENING, tailwater; contraction=CC)

function save_figure(fig, name)
    path = joinpath(OUTPUT_DIR, name)
    savefig(fig, path)
    cp(path, joinpath(FIGURES_DIR, name); force=true)
    println("wrote ", path)
end

gate_shape!(p; x=X_GATE, top=GRID_DEPTH) = plot!(p, [x, x], [OPENING, top]; color=:black, lw=4, label="gate")

function profiles_figure()
    panels = map(PROFILE_TAILWATERS) do ht
        E = theory_head(ht)
        title = ht === nothing ? "Free flow (free outflow)" : @sprintf("Submerged, tailwater %.2f m", ht)
        p = plot(xlabel="x (m)", ylabel="z (m)", title=title, legend=:topright, ylims=(0, 0.6))
        hline!(p, [E]; color=:gray, ls=:dot, label="theory: upstream energy head")
        h1 = bernoulli_depth(E, Q)
        plot!(p, [0, X_GATE], [h1, h1]; color=:black, lw=2, label="theory: upstream depth")
        if ht === nothing
            plot!(p, [X_GATE, LENGTH], fill(CC * OPENING, 2); color=:black, lw=2, ls=:dash, label="theory: C_c a")
        else
            plot!(p, [X_GATE, LENGTH], [ht, ht]; color=:black, lw=2, ls=:dash, label="tailwater")
        end
        for (nz, color, style) in ((1, :crimson, :solid), (NZ_2DV, :steelblue, :dash))
            nu = (nz > 1 && ht !== nothing) ? NU_2DV_SUBMERGED : NU_1D
            model, xc = gate_model(; tailwater=ht, nz=nz, nu=nu)
            run_steady!(model, RUN_TIME)
            plot!(p, xc, water_depth(model); color=color, lw=2, ls=style, label=nz == 1 ? "model 1D" : "model 2DV, nz = $(nz)")
        end
        gate_shape!(p)
    end
    save_figure(plot(panels...; layout=(1, 2), size=(1150, 420), margin=FIGURE_MARGIN_MM), "sluice_gate_profiles.png")
end

function modular_figure()
    limit = sluice_gate_modular_limit(Q, OPENING; contraction=CC)
    p = plot(xlabel="tailwater depth (m)", ylabel="upstream energy head E₁ (m)",
        title=@sprintf("Sluice gate, q = %.2f m²/s, a = %.2f m, C_c = %.2f", Q, OPENING, CC), legend=:topleft)
    ys = range(minimum(TAILWATERS), maximum(TAILWATERS), length=200)
    plot!(p, ys, theory_head.(ys); color=:black, lw=2, label="theory (Bernoulli + momentum)")
    vline!(p, [limit]; color=:gray, ls=:dash, label=@sprintf("modular limit %.3f m", limit))
    for (nz, color, marker) in ((1, :crimson, :circle), (NZ_2DV, :steelblue, :utriangle))
        heads = map(collect(TAILWATERS)) do ht
            nu = (nz > 1 && ht > limit) ? NU_2DV_SUBMERGED : NU_1D
            model, xc = gate_model(; tailwater=ht, nz=nz, nu=nu)
            run_steady!(model, RUN_TIME)
            upstream_head(model, xc, X_UPSTREAM)
        end
        scatter!(p, collect(TAILWATERS), heads; color=color, marker=marker, ms=5, label=nz == 1 ? "model 1D" : "model 2DV, nz = $(nz)")
    end
    save_figure(plot(p; size=(700, 450), margin=FIGURE_MARGIN_MM), "sluice_gate_modular.png")
end

const WET_LAYER_MIN = 1e-6      # m, thinner layers are drawn as dry

# Cell-centred velocity magnitude of the wet cells (NaN: dry or empty)
function cell_speed(model)
    wd = model.wetdry
    nz, nx = model.config.nz, model.config.nx
    s = fill(NaN, nz, nx)
    for i in 1:nx, k in 1:nz
        wd.h[k, i] > WET_LAYER_MIN || continue
        u = 0.5 * (model.state.u[k, i] + model.state.u[k, i + 1])
        w = 0.5 * (model.state.w[k, i] + model.state.w[k + 1, i])
        s[k, i] = hypot(u, w)
    end
    return s
end

# Free surface (m) piecewise linear between the cell centres, as the plotted surface line; at a gate face and at the
# boundaries it is held at the cell value (it jumps at the gate).
function surface_line(model, xc)
    zeta, wd, nx = water_depth(model), model.wetdry, model.config.nx
    face(f, i) = (f == 1 || f == nx + 1 || isfinite(wd.gate_edge[f])) ? zeta[i] : 0.5 * (zeta[f - 1] + zeta[f])
    xs, zs = Float64[], Float64[]
    for i in 1:nx
        append!(xs, (xc[i] - 0.5 * model.dx, xc[i], xc[i] + 0.5 * model.dx))
        append!(zs, (face(i, i), zeta[i], face(i + 1, i)))
    end
    return xs, zs
end

# Wet part of every layer cell as a polygon ("shaved" cell): layer k spans (k - 1) dz .. k dz (flat bed at z = 0) and is
# cut by the free surface of `surface_line`. Colour: the cell speed. Only cells that overlap `xlims` are drawn.
function shaved_speed!(p, model, xc; xlims, clims=nothing)
    dz, speed = model.dz, cell_speed(model)
    sx, sz = surface_line(model, xc)
    shapes, values = Shape[], Float64[]
    for i in 1:model.config.nx
        px, pz = sx[3i - 2:3i], sz[3i - 2:3i]
        (px[3] < xlims[1] || px[1] > xlims[2]) && continue
        for k in 1:model.config.nz
            isnan(speed[k, i]) && continue
            zb, zt = (k - 1) * dz, k * dz
            # corners of the cut: the cell faces and centre, and where the surface crosses the layer interfaces
            xs = copy(px)
            for j in 1:2, zl in (zb, zt)
                t = (zl - pz[j]) / (pz[j + 1] - pz[j])
                0 < t < 1 && push!(xs, px[j] + t * (px[j + 1] - px[j]))
            end
            sort!(xs)
            top = [clamp(x <= px[2] ? pz[1] + (pz[2] - pz[1]) * (x - px[1]) / (px[2] - px[1]) :
                                      pz[2] + (pz[3] - pz[2]) * (x - px[2]) / (px[3] - px[2]), zb, zt) for x in xs]
            maximum(top) - zb > WET_LAYER_MIN || continue
            push!(shapes, Shape(vcat(xs, reverse(xs)), vcat(fill(zb, length(xs)), reverse(top))))
            push!(values, speed[k, i])
        end
    end
    plot!(p, shapes; fill_z=permutedims(values),color=:viridis, clims=clims === nothing ? extrema(values) : clims, linewidth=0,
        linealpha=0, label="", colorbar=true, colorbar_title="|u| (m/s)")
    plot!(p, sx, sz; color=:white, lw=2, label="surface")
    return p
end

nh_model(nx, nz) =gate_model(; nx=nx, nz=nz, nu=NH_NU, length=NH_LENGTH, x_gate=NH_X_GATE, nonhydro=true)

function contraction_figure()
    cc = Float64[]
    section = nothing
    for (nx, nz) in NH_GRIDS
        model, xc = nh_model(nx, nz)
        run_steady!(model, NH_RUN_TIME)
        E = upstream_head(model, xc, 0.4 * NH_X_GATE)
        push!(cc, bernoulli_depth(E, Q; supercritical=true) / OPENING)
        @printf("nx = %d, nz = %d: C_c = %.3f\n", nx, nz, cc[end])
        if (nx, nz) == NH_GRIDS[2]
            xl = (NH_X_GATE - 0.6, NH_X_GATE + 1.2)
            p = plot(xlabel="x (m)", ylabel="z (m)", title="Non-hydrostatic, nx = $(nx), nz = $(nz)", xlims=xl,
                ylims=(0, 0.4), background_color_inside=:gray85)
            shaved_speed!(p, model, xc; xlims=xl)
            hline!(p, [CC * OPENING]; color=:red, ls=:dash, label="C_c a = 0.61 a")
            section = gate_shape!(p; x=NH_X_GATE, top=0.4)
        end
    end
    dz = [GRID_DEPTH / nz for (_, nz) in NH_GRIDS]
    p2 = plot(OPENING ./ dz, cc; marker=:circle, color=:steelblue, lw=2, xscale=:log2, xlabel="a / dz (= a / dx)",
        ylabel="C_c from the upstream head", title="Contraction coefficient (non-hydrostatic)", label="model",
        ylims=(0.5, 1.05), legend=:bottomleft)
    hline!(p2, [CC]; color=:black, ls=:dash, label="0.61 (Henderson 1966)")
    hline!(p2, [1.0]; color=:gray, ls=:dot, label="hydrostatic (no contraction)")
    save_figure(plot(section, p2; layout=(1, 2), size=(1200, 420), margin=FIGURE_MARGIN_MM), "sluice_gate_contraction.png")
end

function nonhydro_animation()
    model, xc = nh_model(ANIMATION_GRID...)
    anim = Animation()
    stride = round(Int, ANIMATION_FRAME_TIME / model.config.dt)
    for n in 0:round(Int, ANIMATION_TIME / model.config.dt)
        n > 0 && step_forward!(model)
        n % stride == 0 || continue
        p = plot(xlabel="x (m)", ylabel="z (m)", size=FRAME_SIZE_PX, xlims=(0, NH_LENGTH), ylims=(0, 0.45),
            legend=:topright, background_color_inside=:gray85,
            title=@sprintf("Non-hydrostatic flow under a sluice gate, t = %4.1f s", n * model.config.dt))
        shaved_speed!(p, model, xc; xlims=(0, NH_LENGTH), clims=(0, 2.5))
        gate_shape!(p; x=NH_X_GATE, top=0.45)
        frame(anim, p)
    end
    path = joinpath(OUTPUT_DIR, "sluice_gate_nonhydro.gif")
    gif(anim, path; fps=ANIMATION_FPS)
    cp(path, joinpath(FIGURES_DIR, "sluice_gate_nonhydro.gif"); force=true)
    @printf("wrote %s (%.0f kB)\n", path, filesize(path) / 1024)
end

const PRODUCTS = Dict("profiles" => profiles_figure, "modular" => modular_figure, "contraction" => contraction_figure,
    "animation" => nonhydro_animation)

mkpath(OUTPUT_DIR)
mkpath(FIGURES_DIR)
for name in (isempty(ARGS) ? ["profiles", "modular", "contraction", "animation"] : ARGS)
    haskey(PRODUCTS, name) || error("unknown product $(name); choose from $(join(sort(collect(keys(PRODUCTS))), ", "))")
    PRODUCTS[name]()
end
