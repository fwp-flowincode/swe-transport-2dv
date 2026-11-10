# Radiation of dispersive waves at the open right face (issue #86): reflection |R| of regular non-hydrostatic waves
# against k h for the former free outflow (long-wave speed, dq/dx = 0 at the face) and the new options
# (radiation_theta, radiation_period, nonhydro_open_face = "radiation"), with the radiation period of the
# wave itself and with a fixed design period (k h = 1.2). Wetting-drying step, depth 0.5 m, 10 layers, 40 cells per
# wavelength, 100 steps per period, theta = 0.7. The set-up is that of test/test_open_boundary_radiation.jl.
#
# Writes figures/open_boundaries_dispersive_radiation.png and prints the table of docs/numerics_open_boundaries.md,
# section 7. Usage: julia --project=<repository> experiments/open_boundaries/dispersive_radiation.jl (about 3 min).

get!(ENV, "GKSwstype", "100")   # WHY: batch run without a display (as plot_open_boundaries.jl)

using SweTransport2DV
using Plots
using Printf
using LinearAlgebra

const G = 9.81                     # m/s^2, ModelConfig default
const DEPTH = 0.5                  # m
const AMPLITUDE = 0.002            # m, linear waves
const NZ = 10
const CELLS_PER_WAVELENGTH = 40
const STEPS_PER_PERIOD = 100
const WAVELENGTHS = 8
const FIT_WINDOW = (4.0, 0.5)      # wavelengths before the open face
const AVERAGE_PERIODS = 2
const KH_VALUES = (0.3, 0.6, 0.9, 1.2, 1.6, 2.0, 2.5, 3.0)
const DESIGN_KH = 1.2              # k h of the fixed radiation period
const FIGURE_DIR = joinpath(@__DIR__, "figures")
# Categorical slots 1-3 of the dataviz reference palette (validated, light surface); identity is also carried by marker
# shape and line style
const COLORS = ("#2a78d6", "#eb6834", "#1baf7a")

wave_period(kh) = 2pi / sqrt(G * (kh / DEPTH) * tanh(kh))

# |R| at the open face and the fitted numerical wave number over the linear-theory one (kr/k)
function reflection(kh; options=(;))
    h = DEPTH
    omega = 2pi / wave_period(kh)
    T = 2pi / omega
    k = kh / h
    lam = 2pi / k
    L = WAVELENGTHS * lam
    nx = WAVELENGTHS * CELLS_PER_WAVELENGTH
    dt = T / STEPS_PER_PERIOD
    cg = 0.5 * omega / k * (1 + 2kh / sinh(2kh))
    cfg = ModelConfig(; nx=nx, nz=NZ, length=L, depth=h, dt=dt, theta=0.7, implicit_vertical_terms=true, nu=1e-6,
        rayleigh_damping=0.0, apply_froude_limiter=false, momentum_advection_scheme="tvd2", use_hydrostatic=false,
        wetting_drying=true, bed_level=zeros(nx), write_plots=false, options...)
    model = Model(cfg)
    fill!(model.state.rho, 1000.0)
    refresh_density_metadata!(model)
    initialize_water_level!(model, fill(h, nx))
    z = [(j - 0.5) * h / NZ for j in 1:NZ]
    profile = [AMPLITUDE * omega * cosh(k * zj) / sinh(kh) for zj in z]
    set_x_inflow_outflow!(model, t -> profile .* (min(t / (2T), 1.0) * sin(omega * t)); right_free=true)
    nsteps = round(Int, ((L + FIT_WINDOW[1] * lam) / cg + 6T) / dt)
    navg = round(Int, AVERAGE_PERIODS * T / dt)
    acc = zeros(ComplexF64, nx)
    for n in 1:nsteps
        step_forward!(model)
        n > nsteps - navg && (acc .+= (water_depth(model) .- h) .* exp(im * omega * n * dt))
    end
    A = 2 .* acc ./ navg
    x = ((1:nx) .- 0.5) .* (L / nx)
    sel = findall(xi -> L - FIT_WINDOW[1] * lam <= xi <= L - FIT_WINDOW[2] * lam, x)
    function fit(kr, ki)
        kk = kr + im * ki
        M = hcat(exp.(im * kk .* (x[sel] .- L)), exp.(-im * kk .* (x[sel] .- L)))
        coef = M \ A[sel]
        return norm(M * coef - A[sel]), abs(coef[2] / coef[1])
    end
    best = (Inf, 0.0, 0.0, 0.0)
    for kr in range(0.8k, 1.2k; length=161), ki in range(-0.02k, 0.02k; length=41)
        res, R = fit(kr, ki)
        res < best[1] && (best = (res, R, kr, ki))
    end
    for _ in 1:3
        kr0, ki0 = best[3], best[4]
        for kr in range(kr0 - 0.0025k, kr0 + 0.0025k; length=41), ki in range(ki0 - 0.001k, ki0 + 0.001k; length=41)
            res, R = fit(kr, ki)
            res < best[1] && (best = (res, R, kr, ki))
        end
    end
    return best[2], best[3] / k
end

new_options(period) = (; radiation_theta=true, nonhydro_open_face="radiation", radiation_period=period)

function main()
    former, own, design, dispersion = Float64[], Float64[], Float64[], Float64[]
    @printf("| k h | former | new, own period | new, period of k h = %.1f | |c_num/c - 1|/2 |\n|---|---|---|---|---|\n", DESIGN_KH)
    for kh in KH_VALUES
        Rf, _ = reflection(kh)
        Ro, kr = reflection(kh; options=new_options(wave_period(kh)))
        Rd, _ = reflection(kh; options=new_options(wave_period(DESIGN_KH)))
        de = abs(1 / kr - 1) / 2
        push!(former, Rf); push!(own, Ro); push!(design, Rd); push!(dispersion, de)
        @printf("| %.1f | %.2f %% | %.2f %% | %.2f %% | %.2f %% |\n", kh, 100Rf, 100Ro, 100Rd, 100de)
        flush(stdout)
    end
    kh = collect(KH_VALUES)
    p = plot(; xlabel="k h", ylabel="|R|", yscale=:log10, ylims=(1e-4, 0.5), xlims=(0, 3.2), legend=:bottomright,
        title="open face, regular non-hydrostatic waves (h = 0.5 m, 10 layers, θ = 0.7)", titlefontsize=10,
        legendfontsize=8, gridalpha=0.15, framestyle=:box, size=(760, 460), margin=4Plots.mm)
    plot!(p, kh, dispersion; color=:gray55, linestyle=:dash, linewidth=1.5, label="interior dispersion error |c_num/c − 1|/2")
    plot!(p, kh, former; color=COLORS[1], linewidth=2, marker=:circle, markersize=5, markerstrokewidth=0,
        label="former: long-wave speed, dq/dx = 0")
    plot!(p, kh, design; color=COLORS[2], linewidth=2, linestyle=:dot, marker=:diamond, markersize=6,
        markerstrokewidth=0, label=@sprintf("new, radiation period of k h = %.1f", DESIGN_KH))
    plot!(p, kh, own; color=COLORS[3], linewidth=2, marker=:utriangle, markersize=6, markerstrokewidth=0,
        label="new, radiation period of the wave")
    mkpath(FIGURE_DIR)
    savefig(p, joinpath(FIGURE_DIR, "open_boundaries_dispersive_radiation.png"))
    return nothing
end

main()
