# Kelvin-Helmholtz instability of a resolved tanh shear layer (issue #22): linear growth rate against the
# Taylor-Goldstein mode on three grids, the growth rate of every wavelength from noise, roll-up and mixing on the three
# grids, and the periodic seam against the interior. The case of test/test_kelvin_helmholtz.jl.
#
# Usage (any working directory):
#   julia --project=<repository> experiments/kelvin_helmholtz/plot_growth_rate.jl [part ...]
# Parts: growth (seeded mode, 3 grids), spectrum (noise in a 4-wavelength box, with the animation), mixing (roll-up on
# 3 grids), seam (billow across the seam against the interior); default: all (about 15 min).
# Output: output_barotropic/kelvin_helmholtz_growth/; the figures are copied to figures/.

include(joinpath(@__DIR__, "..", "common", "performance.jl"))   # performance overview at exit (issue #10)
using SweTransport2DV
using Plots
using Printf
using Random

# WHY: batch run without a display; same GR workstation choice as src/visualization.jl.
get!(ENV, "GKSwstype", "100")

const PARTS = isempty(ARGS) ? ["growth", "spectrum", "mixing", "seam"] : ARGS
const OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "output_barotropic", "kelvin_helmholtz_growth")
const FIGURE_DIR = joinpath(@__DIR__, "figures")

# ---- Case (test/test_kelvin_helmholtz.jl) -----------------------------------------------------------------------------
const DELTA = 0.1                 # m, shear-layer half thickness delta
const U = 0.05                    # m/s, velocity (difference 2U across the layer)
const KDELTA = 0.4446             # -, k delta of the most unstable unstratified mode (Michalke 1964)
const DEPTH = 0.8                 # m, 8 delta
const RICHARDSON = 0.05           # -, bulk Richardson number J = g drho delta / (2 rho0 U^2)
const RHO0 = 1000.0               # kg/m^3, ModelConfig default
const G = 9.81                    # m/s^2, ModelConfig default
const SEED = 1e-3                 # m, interface displacement of the seeded mode
const COURANT = 0.2               # -, U dt / dx on every grid
const CELLS = (8, 16, 32)         # cells per delta (dx = dz)
const FIT_WINDOW = (5.0, 35.0)    # s, linear phase of the seeded mode
const GROWTH_T_END = 60.0         # s, growth curves (saturation from about 45 s)

# noise run (wavelength selection)
const NOISE_WAVELENGTHS = 4       # -, box length in wavelengths of KDELTA
const NOISE_CELLS = 16            # cells per delta
const NOISE_MODES = 1:12          # -, Fourier modes of the box that are seeded and measured (k delta = 0.11 ... 1.33)
const NOISE_AMPLITUDE = 1e-5      # m^2/s, stream-function amplitude of each mode (times sech((z - z0)/delta))
const NOISE_SEED = 22             # -, random phases
const NOISE_FIT_WINDOW = (20.0, 50.0)   # s, after the transient of the non-eigenfunction start, before saturation
const NOISE_T_END = 150.0         # s, animation
const NOISE_FRAME_INTERVAL = 2.0  # s
const ANIMATION_FPS = 8           # frames/s
const ANIMATION_COLOR_LEVELS = 12 # -, discrete colours (file size well below 1 MB)
const ANIMATION_SIZE_PX = (900, 300)

# roll-up and mixing
const MIXING_T_END = 150.0        # s
const MIXING_INTERVAL = 1.0       # s, sampling of the mixing measures
const SNAPSHOT_TIME = 90.0        # s, density snapshot on the three grids (rolled up)
const MIXED_FRACTION_RANGE = (0.05, 0.95)   # -, heavy-water fraction that counts as mixed (plot_kelvin_helmholtz.jl)
const SEAM_CELLS = 16             # cells per delta of the seam comparison
const SEAM_TIMES = (40.0, 70.0, 100.0, 130.0)   # s

const FIGURE_MARGIN_MM = 4Plots.mm
const LINE_SIZE_PX = (1150, 430)

k_seed() = KDELTA / DELTA                                   # 1/m
wavelength() = 2pi / k_seed()                               # m
drho(J) = J * 2 * RHO0 * U^2 / (G * DELTA)                  # kg/m^3
# Growth rate (1/s) on the wavenumbers k delta = kds (ascending), NaN where stable. WHY the continuation (each k also
# tries the root of the previous one): towards the neutral wavenumber the default start 0.4 i U is too far from the
# small phase speed and the secant does not converge.
function theory_curve(kds, J)
    guess = 0.4im
    return map(kds) do kd
        for g in (0.4im, guess)
            try
                sigma, c = kelvin_helmholtz_tanh_mode(; k=kd / DELTA, velocity=U, thickness=DELTA, richardson=J,
                    depth=DEPTH, initial_guess=g)
                guess = c / U
                return sigma
            catch
            end
        end
        return NaN   # stable: no unstable root
    end
end

function shear_model(cells_per_delta; wavelengths=1, t_end)
    dz = DELTA / cells_per_delta
    L = wavelengths * wavelength()
    nx, nz = round(Int, L / dz), round(Int, DEPTH / dz)
    cfg = ModelConfig(length=L, depth=DEPTH, nx=nx, nz=nz, dt=COURANT * (L / nx) / U, nt=round(Int, t_end / (COURANT * (L / nx) / U)),
        use_hydrostatic=false, periodic_x=true, momentum_advection_scheme="tvd2", transport_advection_scheme="tvd2",
        rayleigh_damping=0.0, apply_froude_limiter=false, nu=0.0, rho_left=RHO0, rho_right=RHO0,
        write_plots=false, write_animation=false, write_pressure_buoyancy_diagnostics=false)
    return Model(cfg)
end

# e^{i k x} amplitude (m/s) of w at the interface z0 (the w-interface at mid-depth) for the mode m of the box
function w_mode(model, m)
    nx, nz, L = model.config.nx, model.config.nz, model.config.length
    return 2 / nx * abs(sum(model.state.w[nz ÷ 2 + 1, i] * cis(-2pi * m * (i - 0.5) * model.dx / L) for i in 1:nx))
end

slope(t, y) = (tm = sum(t) / length(t); ym = sum(y) / length(y); sum((t .- tm) .* (y .- ym)) / sum((t .- tm) .^ 2))

# heavy-water fraction, mixed-layer thickness (m) and density variance (kg^2/m^6) of a state with the layer densities
function mixing_measures(model, J)
    dr = drho(J)
    f = (model.state.rho .- (RHO0 - dr / 2)) ./ dr
    mixed = count(x -> MIXED_FRACTION_RANGE[1] < x < MIXED_FRACTION_RANGE[2], f) * model.dx * model.dz / model.config.length
    return mixed, sum(abs2, model.state.rho .- RHO0) / length(model.state.rho)
end

function save_figure(fig, name)
    path = joinpath(OUTPUT_DIR, name)
    savefig(fig, path)
    cp(path, joinpath(FIGURE_DIR, name); force=true)
    println("Saved: $(abspath(path))")
end

function density_panel(model, J; title, kwargs...)
    nx, nz = model.config.nx, model.config.nz
    x = [(i - 0.5) * model.dx for i in 1:nx]
    z = [(k - 0.5) * model.dz for k in 1:nz]
    dr = drho(J)
    return heatmap(x, z, model.state.rho; color=:viridis, clims=(RHO0 - dr / 2, RHO0 + dr / 2), title=title,
        titlefontsize=10, xlabel="x (m)", ylabel="z (m)", colorbar_title="ρ (kg/m³)", aspect_ratio=:equal,
        xlims=(0, model.config.length), ylims=(0, DEPTH), kwargs...)
end

mkpath(OUTPUT_DIR)
mkpath(FIGURE_DIR)

# ---- growth: seeded mode on three grids -------------------------------------------------------------------------------
if "growth" in PARTS
    println("\n=== Growth of the seeded mode (k delta = $(KDELTA)) ===")
    fig = plot(xlabel="t (s)", ylabel="w amplitude at z0 (m/s)", yscale=:log10, legend=:topleft, size=LINE_SIZE_PX,
        title=@sprintf("Seeded Kelvin-Helmholtz mode, k δ = %.4f, depth 8 δ", KDELTA), titlefontsize=11,
        left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM)
    conv = Dict{Float64,Vector{Float64}}()
    for (J, color) in ((RICHARDSON, 2), (0.0, 1)), (j, cells) in enumerate(CELLS)
        model = shear_model(cells; t_end=GROWTH_T_END)
        sigma, _ = initialize_shear_layer_mode!(model; velocity=U, thickness=DELTA, density_difference=drho(J), amplitude=SEED)
        dt = model.config.dt
        t, a = [0.0], [w_mode(model, 1)]
        for n in 1:round(Int, GROWTH_T_END / dt)
            step_forward!(model)
            push!(t, n * dt)
            push!(a, w_mode(model, 1))
        end
        sel = FIT_WINDOW[1] .<= t .<= FIT_WINDOW[2]
        rate = slope(t[sel], log.(a[sel]))
        push!(get!(conv, J, Float64[]), (rate - sigma) / sigma)
        @printf("J = %.2f, %2d cells per delta (%4d x %3d) | sigma %.5f 1/s, measured %.5f 1/s (%+.2f %%)\n", J, cells,
            model.config.nx, model.config.nz, sigma, rate, 100 * (rate - sigma) / sigma)
        plot!(fig, t, a; color=color, linewidth=j == length(CELLS) ? 2.5 : 1.2, linestyle=(:dot, :dash, :solid)[j],
            label=@sprintf("J = %.2f, %d cells/δ (σ %+.2f %%)", J, cells, 100 * (rate - sigma) / sigma))
        j == length(CELLS) && plot!(fig, t, a[1] .* exp.(sigma .* t); color=:black, linestyle=:dashdot, linewidth=1,
            label=j == length(CELLS) && J == RICHARDSON ? "Taylor-Goldstein rate" : "")
    end
    vline!(fig, collect(FIT_WINDOW); color=:gray60, linestyle=:dot, label="fit window")
    save_figure(fig, "kh_growth_curves.png")
    p = plot(xlabel="dx = dz (m)", ylabel="growth-rate error (%)", xscale=:log10, yscale=:log10, legend=:bottomright,
        xticks=([DELTA / c for c in CELLS], [@sprintf("%g", DELTA / c) for c in CELLS]), size=(700, 450),
        yticks=([0.2, 0.3, 0.5, 1], ["0.2", "0.3", "0.5", "1"]), title="Growth rate against the grid",
        titlefontsize=11, left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM)
    for (J, color) in ((RICHARDSON, 2), (0.0, 1))
        e = 100 .* abs.(conv[J])
        orders = join([@sprintf("%.2f", log2(e[j] / e[j + 1])) for j in 1:(length(e) - 1)], ", ")
        plot!(p, [DELTA / c for c in CELLS], e; color=color, marker=:circle, linewidth=2,
            label=@sprintf("J = %.2f (orders %s)", J, orders))
    end
    dxs = [DELTA / c for c in CELLS]
    plot!(p, dxs, 100 * abs(conv[RICHARDSON][end]) .* dxs ./ dxs[end]; color=:gray50, linestyle=:dot, label="first order")
    save_figure(p, "kh_growth_convergence.png")
end

# ---- spectrum: noise in a box of several wavelengths -----------------------------------------------------------------
if "spectrum" in PARTS
    println("\n=== Growth rate of every wavelength from noise ($(NOISE_WAVELENGTHS) wavelengths, $(NOISE_CELLS) cells per delta) ===")
    model = shear_model(NOISE_CELLS; wavelengths=NOISE_WAVELENGTHS, t_end=NOISE_T_END)
    initialize_shear_layer_mode!(model; velocity=U, thickness=DELTA, density_difference=drho(RICHARDSON), amplitude=0.0)
    # divergence-free noise: stream function sum_m a cos(k_m x + phase_m) sech((z - z0)/delta) at the cell corners
    rng = MersenneTwister(NOISE_SEED)
    phases = 2pi .* rand(rng, length(NOISE_MODES))
    L, dx, dz, nx, nz = model.config.length, model.dx, model.dz, model.config.nx, model.config.nz
    psi(x, z) = NOISE_AMPLITUDE * sech((z - DEPTH / 2) / DELTA) * sum(cos(2pi * m * x / L + ph) for (m, ph) in zip(NOISE_MODES, phases))
    for i in 1:(nx + 1), k in 1:nz
        model.state.u[k, i] += (psi((i - 1) * dx, k * dz) - psi((i - 1) * dx, (k - 1) * dz)) / dz
    end
    @views model.state.u[:, end] .= model.state.u[:, 1]
    for i in 1:nx, k in 1:(nz + 1)
        model.state.w[k, i] -= (psi(i * dx, (k - 1) * dz) - psi((i - 1) * dx, (k - 1) * dz)) / dx
    end
    dt = model.config.dt
    t = Float64[]
    amps = [Float64[] for _ in NOISE_MODES]
    anim = Animation()
    stride = round(Int, NOISE_FRAME_INTERVAL / dt)
    dominant = Tuple{Float64,Int}[]
    for n in 0:round(Int, NOISE_T_END / dt)
        n > 0 && step_forward!(model)
        tn = n * dt
        if tn <= NOISE_FIT_WINDOW[2] + 10
            push!(t, tn)
            foreach(((j, m),) -> push!(amps[j], w_mode(model, m)), enumerate(NOISE_MODES))
        end
        if n % stride == 0
            spec = [w_mode(model, m) for m in 1:(nx ÷ 4)]
            push!(dominant, (tn, argmax(spec)))
            frame(anim, density_panel(model, RICHARDSON; size=ANIMATION_SIZE_PX, aspect_ratio=:auto,
                color=cgrad(:viridis, ANIMATION_COLOR_LEVELS; categorical=true), left_margin=FIGURE_MARGIN_MM,
                bottom_margin=FIGURE_MARGIN_MM, title=@sprintf("Kelvin-Helmholtz from noise, J = %.2f, %d cells/δ, t = %.0f s",
                RICHARDSON, NOISE_CELLS, tn)))
        end
    end
    path = joinpath(OUTPUT_DIR, "kh_noise.gif")
    gif(anim, path; fps=ANIMATION_FPS)
    cp(path, joinpath(FIGURE_DIR, "kh_noise.gif"); force=true)
    sel = NOISE_FIT_WINDOW[1] .<= t .<= NOISE_FIT_WINDOW[2]
    rates = [slope(t[sel], log.(amps[j][sel])) for j in eachindex(NOISE_MODES)]
    kd = [2pi * m / L * DELTA for m in NOISE_MODES]
    kfine = collect(range(0.02, 1.2; length=119))
    sig = theory_curve(kfine, RICHARDSON)
    p = plot(kfine, sig; color=:black, linewidth=2, label="Taylor-Goldstein, J = $(RICHARDSON), depth 8 δ",
        xlabel="k δ (-)", ylabel="growth rate (1/s)", size=(800, 450), legend=:bottom,
        title=@sprintf("Growth rate of every mode of a %d-wavelength box from noise (%d cells/δ)", NOISE_WAVELENGTHS, NOISE_CELLS),
        titlefontsize=11, left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM)
    scatter!(p, kd, rates; color=2, markersize=6, label=@sprintf("model, fit %g-%g s", NOISE_FIT_WINDOW...))
    hline!(p, [0.0]; color=:gray60, label="")
    save_figure(p, "kh_spectrum.png")
    sig_modes = theory_curve(kd, RICHARDSON)
    for (j, m) in enumerate(NOISE_MODES)
        @printf("mode %2d, k delta = %.3f | measured %+.5f 1/s, theory %s\n", m, kd[j], rates[j],
            isnan(sig_modes[j]) ? "stable" : @sprintf("%.5f 1/s", sig_modes[j]))
    end
    println("dominant mode of w at z0 (time, mode): ", join([@sprintf("(%.0f s, %d)", d...) for d in dominant[1:5:end]], " "))
end

# ---- mixing: roll-up on the three grids --------------------------------------------------------------------------------
if "mixing" in PARTS
    println("\n=== Roll-up and mixing on three grids (J = $(RICHARDSON)) ===")
    pm = plot(xlabel="t (s)", ylabel="mixed-layer thickness (m)", legend=:topleft, title="Mixed layer (heavy fraction 0.05-0.95)",
        titlefontsize=10)
    pv = plot(xlabel="t (s)", ylabel="density variance / initial (-)", legend=:bottomleft, title="Density variance",
        titlefontsize=10)
    snaps = []
    for (j, cells) in enumerate(CELLS)
        model = shear_model(cells; t_end=MIXING_T_END)
        initialize_shear_layer_mode!(model; velocity=U, thickness=DELTA, density_difference=drho(RICHARDSON), amplitude=SEED)
        dt = model.config.dt
        every = round(Int, MIXING_INTERVAL / dt)
        m0, v0 = mixing_measures(model, RICHARDSON)
        t, mixed, var = [0.0], [m0], [1.0]
        n_snap = ceil(Int, SNAPSHOT_TIME / dt)   # first step at or after the snapshot time (dt is not a round number)
        for n in 1:round(Int, MIXING_T_END / dt)
            step_forward!(model)
            n == n_snap && push!(snaps, density_panel(model, RICHARDSON;
                title=@sprintf("%d cells/δ, t = %.0f s", cells, SNAPSHOT_TIME)))
            n % every == 0 || continue
            mm, vv = mixing_measures(model, RICHARDSON)
            push!(t, n * dt); push!(mixed, mm); push!(var, vv / v0)
        end
        @printf("%2d cells per delta | mixed layer %.3f m at 0 s, %.3f m at %.0f s, %.3f m at %.0f s | variance %.3f at %.0f s\n",
            cells, mixed[1], mixed[argmin(abs.(t .- SNAPSHOT_TIME))], SNAPSHOT_TIME, mixed[end], t[end], var[end], t[end])
        plot!(pm, t, mixed; linewidth=2, label="$(cells) cells/δ")
        plot!(pv, t, var; linewidth=2, label="$(cells) cells/δ")
    end
    save_figure(plot(pm, pv; layout=(1, 2), size=LINE_SIZE_PX, left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM),
        "kh_mixing.png")
    save_figure(plot(snaps...; layout=(1, length(snaps)), size=(1400, 380), left_margin=FIGURE_MARGIN_MM,
        bottom_margin=FIGURE_MARGIN_MM), "kh_rollup_grids.png")
end

# ---- seam: the billow across the seam against the billow in the interior -----------------------------------------------
if "seam" in PARTS
    println("\n=== Periodic seam against the interior ($(SEAM_CELLS) cells per delta) ===")
    a = shear_model(SEAM_CELLS; t_end=maximum(SEAM_TIMES))
    initialize_shear_layer_mode!(a; velocity=U, thickness=DELTA, density_difference=drho(RICHARDSON), amplitude=SEED)
    b = shear_model(SEAM_CELLS; t_end=maximum(SEAM_TIMES))
    initialize_shear_layer_mode!(b; velocity=U, thickness=DELTA, density_difference=drho(RICHARDSON), amplitude=SEED)
    # b: the same state shifted by half the box, so its billow forms across the seam
    nx = b.config.nx
    iseven(nx) || error("the seam comparison needs an even nx")
    h = nx ÷ 2
    b.state.rho .= circshift(a.state.rho, (0, h))
    b.state.w .= circshift(a.state.w, (0, h))
    b.state.u[:, 1:nx] .= circshift(a.state.u[:, 1:nx], (0, h))
    @views b.state.u[:, end] .= b.state.u[:, 1]
    refresh_density_metadata!(b)
    dt = a.config.dt
    for n in 1:ceil(Int, maximum(SEAM_TIMES) / dt)
        step_forward!(a)
        step_forward!(b)
        if any(n == ceil(Int, ts / dt) for ts in SEAM_TIMES)
            diff = maximum(abs, circshift(b.state.rho, (0, -h)) .- a.state.rho) / drho(RICHARDSON)
            udiff = maximum(abs, circshift(b.state.u[:, 1:nx], (0, -h)) .- a.state.u[:, 1:nx]) / U
            @printf("t = %5.1f s | largest density difference %.2e drho, velocity difference %.2e U\n", n * dt, diff, udiff)
        end
    end
end
