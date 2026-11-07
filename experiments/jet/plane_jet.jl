# Turbulent plane jet with k-epsilon (issue #25): the 2DV model on two grids, hydrostatic and non-hydrostatic, against
# the thin-shear-layer k-epsilon reference (plane_jet_thin_shear_layer), and the free-jet limit of the reference against
# the measured spreading rate. The case of test/test_plane_jet.jl.
#
# Usage (any working directory):
#   julia --project=<repository> experiments/jet/plane_jet.jl [runs] [figures]
# runs: the four model runs (about 30 min; the fine non-hydrostatic one about 20 min), saved to the output folder;
# figures: the figures from the saved runs. Default: both.
# Output: output_barotropic/plane_jet/; the figures are copied to figures/.

include(joinpath(@__DIR__, "..", "common", "performance.jl"))   # performance overview at exit (issue #10)
using SweTransport2DV
using Plots
using Printf
using Serialization
using Statistics

# WHY: batch run without a display; same GR workstation choice as src/visualization.jl.
get!(ENV, "GKSwstype", "100")

const PARTS = isempty(ARGS) ? ["runs", "figures"] : ARGS
const OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "output_barotropic", "plane_jet")
const FIGURE_DIR = joinpath(@__DIR__, "figures")

# ---- Case (test/test_plane_jet.jl) --------------------------------------------------------------------------------------
const B0 = 0.05                 # m, slot height of the full jet (the half slot b0/2 lies on the frictionless bed)
const U0 = 0.5                  # m/s, slot velocity (U0 b0 / nu = 25000)
const COFLOW = 0.075            # m/s, co-flow (15 % of U0), supplies the entrainment of the confined jet
const DEPTH = 1.2               # m, 24 b0
const LENGTH = 2.25             # m, 45 b0
const NU = 1e-6                 # m^2/s
const INTENSITY = 0.05          # -, turbulence intensity of the inflow
const MIXING_LENGTH = 0.1       # -, inflow length scale / b0
const RELAXATION = 10.0         # s, relaxation of the outflow level
const T_END = 150.0             # s, five flow-throughs of the co-flow
const T_AVERAGE = 30.0          # s
const COURANT = 0.5             # -, U0 dt / dx
const EDGE_FACTOR = 3.0         # -, jet edge at this many half-widths (the free stream is taken there)
const EDGE_ITERATIONS = 2       # -
const GRIDS = (3, 6)            # cells across the half slot (dz = b0/6, b0/12)
const STATIONS = (10, 20, 30)   # b0, profiles and comparison
const FIT = (10, 30)            # b0, fit range of the spreading rate and the decay exponent
const FREE_COFLOW = 0.02        # m/s, free-jet limit of the reference (momentum length 31 m)
const FREE_DEPTH = 8.0          # m
const FREE_X_END = 40           # b0
const MEASURED_S = (0.10, 0.11) # -, measured plane jets (Pope 2000, ch. 5)
const FIGURE_MARGIN_MM = 4Plots.mm

k_inlet(u) = 1.5 * (INTENSITY * u)^2
eps_inlet(k) = SweTransport2DV.KE_C_MU^0.75 * k^1.5 / (MIXING_LENGTH * B0)
inlet(coflow) = (k_slot=k_inlet(U0), eps_slot=eps_inlet(k_inlet(U0)), k_ambient=k_inlet(coflow),
    eps_ambient=eps_inlet(k_inlet(coflow)))
reference(; coflow=COFLOW, depth=DEPTH, x_end=40B0, kwargs...) = plane_jet_thin_shear_layer(; velocity=U0,
    half_slot=B0 / 2, coflow=coflow, depth=depth, nu=NU, inlet(coflow)..., x_end=x_end, kwargs...)
slope(x, y) = sum((x .- mean(x)) .* (y .- mean(y))) / sum((x .- mean(x)) .^ 2)
function similarity(x, uc, yh)
    sel = (FIT[1] * B0 .<= x .<= FIT[2] * B0)
    S = slope(x[sel], yh[sel])
    x0 = mean(x[sel]) - mean(yh[sel]) / S
    return S, x0, slope(log.(x[sel] .- x0), log.(uc[sel]))
end
at(x, y, xs) = y[argmin(abs.(x .- xs))]

# jet measures of a profile u at the cell centres z (test_plane_jet.jl, pj_profile)
function profile_measures(u, z)
    dz = z[2] - z[1]
    ue, half, edge = u[end], 0.0, length(u)
    for _ in 1:EDGE_ITERATIONS
        excess = u .- ue
        j = findfirst(<(excess[1] / 2), excess)
        half = z[j - 1] + (excess[j - 1] - excess[1] / 2) / (excess[j - 1] - excess[j]) * dz
        edge = min(length(u), round(Int, EDGE_FACTOR * half / dz + 0.5))
        ue = u[edge]
    end
    excess = u .- ue
    return excess[1], half, 2 * sum(u[1:edge] .* excess[1:edge]) * dz, ue
end

run_path(cells, hydrostatic) = joinpath(OUTPUT_DIR, @sprintf("jet_%dcells_%s.jls", cells, hydrostatic ? "hydrostatic" : "nonhydrostatic"))

function run_jet(cells, hydrostatic)
    dz = B0 / 2 / cells
    nz, nx = round(Int, DEPTH / dz), round(Int, LENGTH / (2dz))
    dx = LENGTH / nx
    dt = COURANT * dx / U0
    cfg = ModelConfig(; length=LENGTH, depth=DEPTH, nx=nx, nz=nz, dt=dt, use_hydrostatic=hydrostatic, nu=NU,
        turbulence_closure="k_epsilon", bed_roughness_length=0.0, rayleigh_damping=0.0, apply_froude_limiter=false,
        momentum_advection_scheme="tvd2", transport_advection_scheme="tvd2", wetting_drying=true,
        implicit_vertical_terms=true, write_plots=false, write_animation=false, write_pressure_buoyancy_diagnostics=false)
    model = Model(cfg)
    initialize_water_level!(model, fill(DEPTH, nx))
    zc = ((1:nz) .- 0.5) .* dz
    slot = zc .< B0 / 2
    inl = inlet(COFLOW)
    u_in = [s ? U0 : COFLOW for s in slot]
    ke = model.turbulence
    fill!(model.state.u, COFLOW)
    fill!(ke.k, inl.k_ambient)
    fill!(ke.eps, inl.eps_ambient)
    set_x_inflow_outflow!(model, u_in; left_discharge=sum(u_in) * dz, left_k=[s ? inl.k_slot : inl.k_ambient for s in slot],
        left_eps=[s ? inl.eps_slot : inl.eps_ambient for s in slot], right_level=0.0, right_free=true,
        right_relaxation_time=RELAXATION)
    steps = round(Int, T_END / dt)
    start = steps - round(Int, T_AVERAGE / dt)
    ubar, wbar, kbar = zeros(nz, nx + 1), zeros(nz + 1, nx), zeros(nz, nx)
    t0 = time()
    for n in 1:steps
        step_forward!(model)
        if n > start
            ubar .+= model.state.u
            wbar .+= model.state.w
            kbar .+= ke.k
        end
    end
    m = steps - start
    result = (; cells, hydrostatic, dz, dx, nx, nz, zc, ubar=ubar ./ m, wbar=wbar ./ m, kbar=kbar ./ m, wall=time() - t0)
    serialize(run_path(cells, hydrostatic), result)
    @printf("%d cells, %s: %d x %d, %d steps in %.0f s\n", cells, hydrostatic ? "hydrostatic" : "non-hydrostatic", nx, nz,
        steps, result.wall)
    return result
end

function measures(r)
    x = collect(0:r.nx) .* r.dx
    vals = [profile_measures(r.ubar[:, i], r.zc) for i in eachindex(x)]
    return (; x, uc=getindex.(vals, 1), yh=getindex.(vals, 2), J=getindex.(vals, 3), ue=getindex.(vals, 4))
end

function save_figure(fig, name)
    path = joinpath(OUTPUT_DIR, name)
    savefig(fig, path)
    cp(path, joinpath(FIGURE_DIR, name); force=true)
    println("Saved: $(abspath(path))")
end

mkpath(OUTPUT_DIR)
mkpath(FIGURE_DIR)

if "runs" in PARTS
    println("\n=== Model runs ===")
    for cells in GRIDS, hydrostatic in (true, false)
        run_jet(cells, hydrostatic)
    end
end

if "figures" in PARTS
    println("\n=== Comparison with the thin-shear-layer reference ===")
    ref = reference()
    Sref, _, nref = similarity(ref.x, ref.centreline_excess, ref.half_width)
    @printf("reference: S = %.4f, decay exponent %.3f\n", Sref, nref)
    runs = [(cells, h) => deserialize(run_path(cells, h)) for cells in GRIDS for h in (true, false) if isfile(run_path(cells, h))]
    style(cells, h) = (color=h ? 1 : 2, linestyle=cells == GRIDS[end] ? :solid : :dash, linewidth=cells == GRIDS[end] ? 2 : 1.5)
    label(cells, h) = @sprintf("%s, %d cells/half slot", h ? "hydrostatic" : "non-hydrostatic", cells)

    # 1. centreline decay and half-width
    p1 = plot(xlabel="x / b0", ylabel="(U0 / Uc)^2", legend=:topleft, title="Centreline excess velocity", titlefontsize=10)
    p2 = plot(xlabel="x / b0", ylabel="y½ / b0", legend=:topleft, title="Half-width", titlefontsize=10)
    p3 = plot(xlabel="x / b0", ylabel="J / J(slot)", legend=:bottomright, title="Excess momentum flux", titlefontsize=10)
    J0 = ref.momentum_flux[1]
    plot!(p1, ref.x ./ B0, (U0 ./ ref.centreline_excess) .^ 2; color=:black, linewidth=2, label="thin-shear-layer reference")
    plot!(p2, ref.x ./ B0, ref.half_width ./ B0; color=:black, linewidth=2, label="reference (S = $(round(Sref, digits=4)))")
    plot!(p3, ref.x ./ B0, ref.momentum_flux ./ J0; color=:black, linewidth=2, label="reference")
    for ((cells, h), r) in runs
        mm = measures(r)
        S, _, n = similarity(mm.x, mm.uc, mm.yh)
        sel = mm.x .>= 2B0
        plot!(p1, mm.x[sel] ./ B0, (U0 ./ mm.uc[sel]) .^ 2; label=label(cells, h), style(cells, h)...)
        plot!(p2, mm.x[sel] ./ B0, mm.yh[sel] ./ B0; label=@sprintf("%s (S = %.4f)", label(cells, h), S), style(cells, h)...)
        plot!(p3, mm.x[sel] ./ B0, mm.J[sel] ./ J0; label=label(cells, h), style(cells, h)...)
        @printf("%-36s | S = %.4f, exponent %.3f | ", label(cells, h), S, n)
        for xs in STATIONS
            @printf("%d b0: Uc %+.1f %%, y1/2 %+.1f %%, J %+.1f %%  ", xs,
                100 * (at(mm.x, mm.uc, xs * B0) / at(ref.x, ref.centreline_excess, xs * B0) - 1),
                100 * (at(mm.x, mm.yh, xs * B0) / at(ref.x, ref.half_width, xs * B0) - 1),
                100 * (at(mm.x, mm.J, xs * B0) / at(ref.x, ref.momentum_flux, xs * B0) - 1))
        end
        println()
    end
    vspan!(p2, collect(FIT); color=:gray85, alpha=0.4, label="fit range")
    save_figure(plot(p1, p2, p3; layout=(1, 3), size=(1500, 460), left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM,
        plot_title=@sprintf("Plane half-jet, k-ε, co-flow %d %% of U0, depth 24 b0", round(Int, 100COFLOW / U0)),
        plot_titlefontsize=11), "plane_jet_decay.png")

    # 2. profiles in similarity coordinates
    p = plot(xlabel="z / y½", ylabel="(u - Ue) / Uc", xlims=(0, 3), legend=:topright, size=(800, 480),
        title="Profiles at 10, 20, 30 b0 in similarity coordinates", titlefontsize=11,
        left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM)
    eta = range(0, 3; length=200)
    plot!(p, eta, exp.(-log(2) .* eta .^ 2); color=:gray50, linestyle=:dot, linewidth=2, label="exp(-ln 2 η²)")
    # the reference returns only the final profile of its march (40 b0)
    plot!(p, ref.z ./ ref.half_width[end], (ref.u .- ref.u[end]) ./ ref.centreline_excess[end]; color=:black, linewidth=2,
        label="reference, $(round(Int, ref.x[end] / B0)) b0")
    for ((cells, h), r) in runs
        cells == GRIDS[end] || continue
        for (j, xs) in enumerate(STATIONS)
            i = round(Int, xs * B0 / r.dx) + 1
            uc, yh, _, ue = profile_measures(r.ubar[:, i], r.zc)
            plot!(p, r.zc ./ yh, (r.ubar[:, i] .- ue) ./ uc; color=h ? 1 : 2, linestyle=(:dot, :dash, :solid)[j],
                label=@sprintf("%s, %d b0", h ? "hydrostatic" : "non-hydrostatic", xs))
        end
    end
    save_figure(p, "plane_jet_profiles.png")

    # 3. mean velocity field (fine non-hydrostatic run, or the best available)
    best = last(runs)
    r = best[2]
    x = collect(0:r.nx) .* r.dx
    hm = heatmap(x ./ B0, r.zc ./ B0, r.ubar; color=:viridis, xlabel="x / b0", ylabel="z / b0", colorbar_title="mean u (m/s)",
        ylims=(0, 10), size=(1200, 380), title=@sprintf("Mean velocity, %s", label(best[1]...)), titlefontsize=11,
        left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM)
    mm = measures(r)
    plot!(hm, mm.x ./ B0, mm.yh ./ B0; color=:white, linewidth=2, label="half-width")
    save_figure(hm, "plane_jet_field.png")

    # 4. free-jet limit of the reference: spreading with and without the time-scale bound, against the measurements
    pf = plot(xlabel="x / b0", ylabel="y½ / b0", legend=:topleft, size=(800, 460), left_margin=FIGURE_MARGIN_MM,
        bottom_margin=FIGURE_MARGIN_MM, title=@sprintf("Free-jet limit of the reference (co-flow %.0f %% of U0, depth %.0f b0)",
        100FREE_COFLOW / U0, FREE_DEPTH / B0), titlefontsize=11)
    for (real, color, name) in ((false, 1, "standard k-ε"), (true, 2, "k-ε with the time-scale bound (model default)"))
        rf = reference(; coflow=FREE_COFLOW, depth=FREE_DEPTH, x_end=FREE_X_END * B0, realizability=real)
        sel = 15B0 .<= rf.x .<= FREE_X_END * B0
        S = slope(rf.x[sel], rf.half_width[sel])
        x0 = mean(rf.x[sel]) - mean(rf.half_width[sel]) / S
        n = slope(log.(rf.x[sel] .- x0), log.(rf.centreline_excess[sel]))
        plot!(pf, rf.x ./ B0, rf.half_width ./ B0; color=color, linewidth=2, label=@sprintf("%s: S = %.4f, exponent %.3f", name, S, n))
        @printf("free-jet limit, %s: S = %.4f (15-%d b0), exponent %.3f\n", name, S, FREE_X_END, n)
        if !real
            for Sm in MEASURED_S
                plot!(pf, [15, FREE_X_END], Sm .* ([15, FREE_X_END] .- x0 / B0); color=:gray50,
                    linestyle=:dot, label=Sm == MEASURED_S[1] ? "measured S = 0.10-0.11 (Pope 2000)" : "")
            end
        end
    end
    save_figure(pf, "plane_jet_free_limit.png")
end
