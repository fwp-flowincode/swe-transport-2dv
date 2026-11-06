# Stratified mixing layer behind a splitter plate, after the Delft3D splitter plate test (issue #42, option B).
#
# Two streams of different velocity and salinity meet at x = 0, the end of a horizontal plate of zero thickness. The
# plate is not in the domain: the inlet at x = 0 gets the profiles of u, k and eps at the end of two precursor channels
# that replace the 4 m of plate (lower stream: rough bed, free surface; upper stream: free-slip bed, free surface). The
# plate height and the stream discharges follow from the measured profiles at 2 and 5 m (see H_LOWER), extracted from
# the Delft3D-FLOW Validation Document (2008) into data/*.csv by digitize_measurements.py.
#
# Usage: julia --project=<path to the repository> experiments/splitter_plate/splitter_plate.jl <command>
#   precursors [plate]  both precursor channels at both dx (a few minutes); saves the inlet profiles; plate:
#                       frictionless (issue #42, default) or rough
#   run <dx_cm> <mode> [plate [c3]]  main run (mode: nonhydrostatic or hydrostatic); saves means and time series
#   runs                the three main runs of the figures, one after the other (about 25 min)
#   sensitivity         rough-plate precursors and the sensitivity runs at dx = 10 cm: rough plate, C3 for a
#                       stationary Richardson number 0.25, upwind1 density transport (about 10 min)
#   figures             figures from the saved runs (missing runs are left out)
# Environment: SP_TRANSPORT=upwind1 runs the main model with first-order density transport (result files _upwind1).
# Output in output_barotropic/splitter_plate/ (runs) and output_barotropic/splitter_plate_figures/, copies in figures/:
# - splitter_plate_inflow.png    inlet profiles from the precursors (u, k, eps), both dx
# - splitter_plate_profiles.png  mean u, relative density and k at 2, 5, 10 and 40 m, the three runs, with the
#                                measurements extracted from the Delft3D-FLOW Validation Document (2008), data/*.csv
# - splitter_plate_sensitivity.png  the same for the sensitivity runs (dx = 10 cm): rough plate, C3, upwind1
# - splitter_plate_fields.png    mean relative density, u and k over the reach, all runs
# - splitter_plate_interface.png thickness of the density interface and of the shear layer against x
# - splitter_plate_budget.png    inflow and outflow discharge, salt budget over time

# WHY: batch run without a display; same GR workstation choice as src/visualization.jl.
get!(ENV, "GKSwstype", "100")

include(joinpath(@__DIR__, "..", "common", "performance.jl"))   # performance overview at exit (issue #10)
using SweTransport2DV
using Plots
using Printf
using Serialization
using Statistics

# Delft3D splitter plate test (issue #42)
const LENGTH = 40.0             # m, reach behind the plate
const PLATE_LENGTH = 4.0        # m, length of the precursor channels
const LEVEL_IN = 0.574          # m, water level at the end of the plate
# WHY the plate at 0.306 m (issue #42 estimated 0.250 m below and 0.324 m above it): from the measured profiles at 2 and
# 5 m behind the plate (data/*.csv), the density interface (relative density 0.5) lies at 0.307 and 0.305 m, and the
# measured velocity integrates to 0.094 / 0.092 m^2/s below it (mean 0.31 / 0.30 m/s) and 0.133 / 0.131 m^2/s above it
# (0.50 / 0.49 m/s): the stream velocities of the issue, with the streams 0.306 and 0.268 m thick
const H_LOWER = 0.306           # m, lower stream below the plate
const H_UPPER = LEVEL_IN - H_LOWER   # m, 0.268: upper stream above the plate
const LEVEL_OUT = 0.570         # m, water level at the outflow
const U_LOWER = 0.3             # m/s, mean velocity of the lower stream
const U_UPPER = 0.5             # m/s, mean velocity of the upper stream
const SALINITY_LOWER = 43.3     # ppt
const SALINITY_UPPER = 22.6     # ppt
const TEMPERATURE = 20.0        # deg C; WHY: not given in the test; the densities move by < 1 kg/m^3 between 15 and 20 deg C
const CHEZY = 65.0              # m^(1/2)/s, bed only
const NU = 1e-6                 # m^2/s, molecular viscosity
const RUN_TIME = 360.0          # s
const AVERAGE_TIME = 120.0      # s, means over the last part of the run
const PRECURSOR_TIME = 60.0     # s, 4 m at 0.3 m/s take 13 s
const PRECURSOR_AVERAGE_TIME = 20.0   # s
const SAMPLE_TIME = 1.0         # s, time series
const ADVECTIVE_COURANT = 0.6   # -, u_max dt / dx
const U_MAX = 0.55              # m/s, largest velocity, for the time step
const TAU = 20.0                # s, relaxation time of the outflow (issue #54)
const LAYERS_LOWER = 22         # layers below the plate (even: the mirrored precursor of the rough plate)
const LAYERS_UPPER = 19         # layers above the plate; the top layer is open above and reaches the surface
const STATIONS = (2.0, 5.0, 10.0, 40.0)   # m behind the plate
const RESOLUTIONS = (5, 10)     # cm, dx
const DISCHARGE_RTOL = 0.01     # -, steady state: mean outflow within this fraction of the inflow
const INTERFACE_LEVELS = (0.1, 0.9)   # -, relative density bounds of the interface thickness
const K_UNIT = 1e-4             # m^2/s^2, unit of k in the heatmaps (short colour-bar labels)
const K_CLIMS = (0.0, 8.0)      # K_UNIT, colour range of k
const U_CLIMS = (0.15, 0.6)     # m/s, colour range of u
const FIGURE_MARGIN_MM = 5Plots.mm
const RUN_DIR = joinpath(@__DIR__, "..", "..", "output_barotropic", "splitter_plate")
const OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "output_barotropic", "splitter_plate_figures")
const FIGURES_DIR = joinpath(@__DIR__, "figures")

const KARMAN = 0.41             # -, von Karman constant (src/turbulence.jl)

const DZ = H_LOWER / LAYERS_LOWER   # m, 13.9 mm; 41 dz = 0.5703 m, the surface (0.570-0.574 m) lies in the top layer
const Z0 = chezy_roughness_length(CHEZY, LEVEL_IN)   # m, about 4e-5

"""
    eckart_density(salinity, temperature) -> Float64

Density (kg/m^3) of sea water at salinity (ppt) and temperature (deg C), at atmospheric pressure. Eckart (1958), Am. J.
Sci. 256, 225-240, eq. 8 with the coefficients of eq. 12 and 13, as used by Delft3D-FLOW (User Manual, section 9.3.1).
"""
function eckart_density(salinity, temperature)
    s, t = salinity, temperature
    p0 = 5890.0 + 38.0t - 0.375t^2 + 3.0s
    lambda = 1779.5 + 11.25t - 0.0745t^2 - (3.8 + 0.01t) * s
    alpha0 = 0.698
    return 1000.0 * p0 / (lambda + alpha0 * p0)
end

const RHO_LOWER = eckart_density(SALINITY_LOWER, TEMPERATURE)   # kg/m^3
const RHO_UPPER = eckart_density(SALINITY_UPPER, TEMPERATURE)   # kg/m^3
const RHO0 = 0.5 * (RHO_LOWER + RHO_UPPER)                      # kg/m^3, Boussinesq reference

function save_figure(fig, name)
    mkpath(OUTPUT_DIR)
    mkpath(FIGURES_DIR)
    path = joinpath(OUTPUT_DIR, name)
    savefig(fig, path)
    cp(path, joinpath(FIGURES_DIR, name); force=true)
    @printf("wrote %s (%.0f kB)\n", path, filesize(path) / 1024)
end

time_step(dx) = ADVECTIVE_COURANT * dx / U_MAX   # s
cell_velocity(model) = 0.5 .* (model.state.u[:, 1:(end - 1)] .+ model.state.u[:, 2:end])   # m/s
relative_density(rho) = (rho - RHO_UPPER) / (RHO_LOWER - RHO_UPPER)

# Log profile u = base + (u*/kappa) ln(1 + z/z0) at the layer centres of a channel of depth h (m), with the layer mean
# `mean` (m/s); `base` (m/s): the upper stream starts at the velocity of the lower one at the plate. Returns the profile
# and u* (m/s).
function log_profile(nz, h, mean, base, z0)
    zc = ((1:nz) .- 0.5) .* (h / nz)
    shape = log.(1 .+ zc ./ z0)
    a = (mean - base) / (sum(shape) / nz)   # m/s, u*/kappa
    return base .+ a .* shape, KARMAN * a
end

# k (m^2/s^2) and eps (m^2/s^3) in equilibrium with the log profile of shear velocity u_star (m/s) in a channel of depth
# h (m), at the layer centres: shear stress u*^2 (1 - z/h), mixing length kappa z, production = dissipation:
#     k = u*^2 (1 - z/h) / sqrt(C_mu),   eps = u*^3 (1 - z/h) / (kappa z)
# Rodi (1993), Turbulence models and their application in hydraulics, IAHR, section 2.3 (open-channel flow).
function equilibrium_turbulence(nz, h, u_star)
    zc = ((1:nz) .- 0.5) .* (h / nz)
    stress = 1 .- zc ./ h
    return u_star^2 .* stress ./ sqrt(KE_C_MU_VALUE), u_star^3 .* stress ./ (KARMAN .* zc)
end

const PLATES = (:frictionless, :rough)   # the plate of issue #42 has no friction; :rough gives it the bed roughness Z0

# Precursor channel of one stream: length PLATE_LENGTH, depth h; returns the means at its end (u at the outflow face,
# k and eps of the last cell), one value per layer.
# plate = :frictionless (issue #42): lower stream with a rough bed under a free-slip plate (an open channel of the stream
# depth), upper stream with a free-slip bed (the plate) whose log profile starts at U_LOWER.
# plate = :rough: the plate has the roughness Z0 on both sides. The lower stream is then a duct between two rough walls;
# WHY half a channel: unstratified, the duct is symmetric about its mid-height, a plane without shear, so an open channel
# of half the depth with a rough bed gives the lower half, mirrored for the upper half. The upper stream has a rough bed
# (the plate) and its log profile starts from rest.
function precursor(stream, dx_cm; plate=:frictionless)
    plate in PLATES || error("plate: $(join(PLATES, " or "))")
    dx = dx_cm / 100
    lower = stream == :lower
    rough = plate == :rough
    mirror = lower && rough
    nz_stream = lower ? LAYERS_LOWER : LAYERS_UPPER
    mirror && isodd(nz_stream) && error("the mirrored lower stream needs an even number of layers")
    nz = mirror ? nz_stream ÷ 2 : nz_stream
    h = nz * DZ
    mean_u = lower ? U_LOWER : U_UPPER
    nx = round(Int, PLATE_LENGTH / dx)
    dt = time_step(dx)
    cfg = ModelConfig(; length=PLATE_LENGTH, depth=h, nx=nx, nz=nz, dt=dt, use_hydrostatic=false, nu=NU, rho0=RHO0,
        turbulence_closure="k_epsilon", bed_roughness_length=(lower || rough) ? Z0 : 0.0, rayleigh_damping=0.0,
        apply_froude_limiter=false, momentum_advection_scheme="tvd2", transport_advection_scheme="tvd2",
        wetting_drying=true, implicit_vertical_terms=true, theta=1.0, write_plots=false)
    model = Model(cfg)
    fill!(model.state.rho, lower ? RHO_LOWER : RHO_UPPER)
    refresh_density_metadata!(model)
    initialize_water_level!(model, fill(h, nx))
    # WHY the plate's velocity at the bed of the frictionless upper stream: there the two streams meet without slip
    shape, u_star = log_profile(nz, h, mean_u, (lower || rough) ? 0.0 : U_LOWER, Z0)
    k_in, eps_in = equilibrium_turbulence(nz, h, u_star)
    ke = model.turbulence
    for k in 1:nz
        model.state.u[k, :] .= shape[k]
        ke.k[k, :] .= k_in[k]
        ke.eps[k, :] .= eps_in[k]
    end
    # WHY a turbulent inlet: from the zero-gradient default the turbulence grows from the bed only, over 0.18 m in 4 m
    set_x_inflow_outflow!(model, shape; left_discharge=mean_u * h, left_k=k_in, left_eps=eps_in, right_level=0.0,
        right_free=true, right_relaxation_time=TAU)
    steps = round(Int, PRECURSOR_TIME / dt)
    start = steps - round(Int, PRECURSOR_AVERAGE_TIME / dt)
    u, k, eps = zeros(nz), zeros(nz), zeros(nz)
    for n in 1:steps
        step_forward!(model)
        n > start || continue
        u .+= model.state.u[:, nx + 1]
        k .+= ke.k[:, nx]
        eps .+= ke.eps[:, nx]
    end
    m = steps - start
    full(v) = mirror ? vcat(v, reverse(v)) : v
    return (; stream, dx_cm, plate, u=full(u ./ m), k=full(k ./ m), eps=full(eps ./ m), shape=full(shape),
        k_in=full(k_in), eps_in=full(eps_in), u_star, zc=((1:nz_stream) .- 0.5) .* DZ)
end

plate_suffix(plate) = plate == :frictionless ? "" : "_" * String(plate)
inflow_path(dx_cm, plate=:frictionless) = joinpath(RUN_DIR, @sprintf("inflow_dx%d%s.jls", dx_cm, plate_suffix(plate)))

function precursors(plate=:frictionless)
    mkpath(RUN_DIR)
    for dx_cm in RESOLUTIONS
        lo = precursor(:lower, dx_cm; plate)
        up = precursor(:upper, dx_cm; plate)
        serialize(inflow_path(dx_cm, plate), (; lower=lo, upper=up))
        @printf("dx = %d cm: lower u %.3f m/s, k %.2e-%.2e; upper u %.3f m/s, k %.2e-%.2e m^2/s^2\n", dx_cm,
            mean(lo.u), extrema(lo.k)..., mean(up.u), extrema(up.k)...)
    end
end

const MODES = ("nonhydrostatic", "hydrostatic")
const C3_DEFAULT = SweTransport2DV.KE_C3_STABLE   # -, C3 of the eps equation in stable stratification
# Density transport scheme of the main runs; SP_TRANSPORT=upwind1 for the sensitivity to numerical mixing
transport_scheme() = get(ENV, "SP_TRANSPORT", "tvd2")
transport_suffix() = transport_scheme() == "tvd2" ? "" : "_" * transport_scheme()
c3_suffix(c3) = c3 == C3_DEFAULT ? "" : "_c3" * replace(@sprintf("%g", c3), "." => "p", "-" => "m")
run_tag(dx_cm, mode, plate=:frictionless, c3=C3_DEFAULT) =
    @sprintf("dx%d_%s%s%s%s", dx_cm, mode, plate_suffix(plate), c3_suffix(c3), transport_suffix())
run_path(args...) = joinpath(RUN_DIR, run_tag(args...) * ".jls")

# Main model: 40 m behind the plate, 22 + 19 layers with an interface at the height of the plate, the inflow of the
# precursors at x = 0 and the relaxed outflow at the measured level. The initial state is the inflow everywhere.
function main_model(dx_cm, mode, plate=:frictionless, c3=C3_DEFAULT)
    mode in MODES || error("mode: $(join(MODES, " or "))")
    inflow = deserialize(inflow_path(dx_cm, plate))
    dx = dx_cm / 100
    nx = round(Int, LENGTH / dx)
    nz = LAYERS_LOWER + LAYERS_UPPER
    cfg = ModelConfig(; length=LENGTH, depth=nz * DZ, nx=nx, nz=nz, dt=time_step(dx), use_hydrostatic=mode == "hydrostatic",
        nu=NU, rho0=RHO0, turbulence_closure="k_epsilon", bed_roughness_length=Z0, rayleigh_damping=0.0,
        apply_froude_limiter=false, momentum_advection_scheme="tvd2", transport_advection_scheme=transport_scheme(),
        wetting_drying=true, implicit_vertical_terms=true, theta=1.0, write_plots=false)
    model = Model(cfg)
    model.turbulence.c3_stable = c3
    xc = ((1:nx) .- 0.5) .* dx
    rho_in = vcat(fill(RHO_LOWER, LAYERS_LOWER), fill(RHO_UPPER, LAYERS_UPPER))
    u_in = vcat(inflow.lower.u, inflow.upper.u)
    k_in = vcat(inflow.lower.k, inflow.upper.k)
    eps_in = vcat(inflow.lower.eps, inflow.upper.eps)
    ke = model.turbulence
    # WHY 41 layers (41 dz = 0.5703 m, at the outlet level 0.570 m): the surface lies in the top layer over the whole
    # reach (it is open above in the wetting-drying grid), so no layer falls dry or nearly empty
    initialize_water_level!(model, LEVEL_IN .+ (LEVEL_OUT - LEVEL_IN) .* xc ./ LENGTH; rho=rho_in * ones(nx)')
    # WHY after initialize_water_level!: it sets the flow to rest
    for k in 1:nz
        model.state.u[k, :] .= u_in[k]
        ke.k[k, :] .= k_in[k]
        ke.eps[k, :] .= eps_in[k]
    end
    set_x_inflow_outflow!(model, u_in; left_discharge=U_LOWER * H_LOWER + U_UPPER * H_UPPER, left_rho=rho_in,
        left_k=k_in, left_eps=eps_in, right_level=LEVEL_OUT - nz * DZ, right_free=true, right_relaxation_time=TAU)
    return model, xc, rho_in
end

# Mass (kg/m per unit width) of the density and volume (m^2) in the reach
reach_mass(model) = sum(model.state.rho .* model.wetdry.h) * model.dx
reach_volume(model) = sum(model.wetdry.h) * model.dx

function run_case(dx_cm, mode, plate=:frictionless, c3=C3_DEFAULT)
    model, xc, rho_in = main_model(dx_cm, mode, plate, c3)
    wd, ke = model.wetdry, model.turbulence
    nz, nx = size(model.state.rho)
    dt = model.config.dt
    steps = round(Int, RUN_TIME / dt)
    start = steps - round(Int, AVERAGE_TIME / dt)
    sample = max(round(Int, SAMPLE_TIME / dt), 1)
    sums = Dict(name => zeros(nz, nx) for name in (:u, :rho, :k, :eps, :nu_t, :h))
    mass0, volume0 = reach_mass(model), reach_volume(model)
    flux_mass = flux_volume = 0.0   # kg/m and m^2, through both boundary faces, cumulated
    rho_old = similar(model.state.rho)
    series = (; t=Float64[], q_in=Float64[], q_out=Float64[], eta_out=Float64[], budget=Float64[], excess_in=Float64[])
    excess_in = 0.0                 # kg/m, cumulated inflow of rho - RHO_UPPER
    elapsed = @elapsed for n in 1:steps
        copyto!(rho_old, model.state.rho)
        step_forward!(model)
        # boundary fluxes of the step with the face densities of the transport: prescribed at the inflow, upwind else
        for k in 1:nz
            q_l, q_r = wd.q[k, 1], wd.q[k, nx + 1]
            rho_l = q_l > 0 ? rho_in[k] : rho_old[k, 1]
            flux_mass += dt * (q_l * rho_l - q_r * rho_old[k, nx])
            flux_volume += dt * (q_l - q_r)
            excess_in += dt * q_l * (rho_l - RHO_UPPER)
        end
        if n % sample == 0
            err_mass = reach_mass(model) - mass0 - flux_mass
            err_volume = reach_volume(model) - volume0 - flux_volume
            push!(series.t, n * dt)
            push!(series.q_in, sum(view(wd.q, :, 1)))
            push!(series.q_out, sum(view(wd.q, :, nx + 1)))
            push!(series.eta_out, model.state.eta[nx])
            # WHY the excess over RHO_UPPER: the budget of rho itself is that of the volume times 1000 kg/m^3
            push!(series.budget, err_mass - RHO_UPPER * err_volume)
            push!(series.excess_in, excess_in)
        end
        n > start || continue
        sums[:u] .+= cell_velocity(model)
        sums[:rho] .+= model.state.rho
        sums[:k] .+= ke.k
        sums[:eps] .+= ke.eps
        sums[:nu_t] .+= ke.nu_t
        sums[:h] .+= wd.h
    end
    m = steps - start
    means = Dict(name => v ./ m for (name, v) in sums)
    rho_range = extrema(model.state.rho[model.wetdry.h .> 0])
    result = (; dx_cm, mode, xc, dz=model.dz, dt, elapsed, series, rho_range, means...)
    mkpath(RUN_DIR)
    serialize(run_path(dx_cm, mode, plate, c3), result)
    last = series.t .> RUN_TIME - AVERAGE_TIME
    @printf("%s: %.0f s wall clock, outflow/inflow %.4f, rho %.2f-%.2f kg/m^3, budget error %.1e of the inflow excess\n",
        run_tag(dx_cm, mode, plate, c3), elapsed, mean(series.q_out[last]) / mean(series.q_in[last]), rho_range...,
        maximum(abs, series.budget) / series.excess_in[end])
    return result
end

load_run(key...) = isfile(run_path(key...)) ? deserialize(run_path(key...)) : nothing

# Runs of the issue (frictionless plate, C3 = 0, tvd2 density transport): two grids, both modes
const RUNS = ((10, "nonhydrostatic"), (5, "nonhydrostatic"), (5, "hydrostatic"))
const RUN_STYLE = Dict((10, "nonhydrostatic") => (:dash, 1), (5, "nonhydrostatic") => (:solid, 1),
    (5, "hydrostatic") => (:solid, 2))
run_label(dx_cm, mode, args...) = @sprintf("dx = %d cm, %s", dx_cm, mode)

# Sensitivity runs, dx = 10 cm, non-hydrostatic: label, run key, density transport scheme
const C3_RI025 = SweTransport2DV.KE_C3_STABLE_RI025   # -, stationary Richardson number 0.25
const SENSITIVITY = (("as the issue (frictionless plate, C3 = 0, tvd2)", (10, "nonhydrostatic"), "tvd2"),
    ("rough plate", (10, "nonhydrostatic", :rough), "tvd2"),
    (@sprintf("C3 = %.2f (stationary Ri = 0.25)", C3_RI025), (10, "nonhydrostatic", :frictionless, C3_RI025), "tvd2"),
    ("upwind1 density transport", (10, "nonhydrostatic"), "upwind1"))

function load_sensitivity_run(key, scheme)
    previous = get(ENV, "SP_TRANSPORT", nothing)
    ENV["SP_TRANSPORT"] = scheme
    try
        return load_run(key...)
    finally
        previous === nothing ? delete!(ENV, "SP_TRANSPORT") : (ENV["SP_TRANSPORT"] = previous)
    end
end
const SHEAR_ZONE = (0.1, 0.45)  # m, heights searched for the largest du/dz
# WHY a zone: the bed boundary layer has the largest shear of the column; the mixing layer lies between

# Heights (m) of the wet-cell centres of column i above the (flat) bed, from the mean layer thicknesses
function cell_heights(r, i)
    h = view(r.h, :, i)
    return cumsum(h) .- 0.5 .* h
end

column_index(r, x) = clamp(round(Int, x / (r.xc[2] - r.xc[1]) + 0.5), 1, length(r.xc))

# Height (m) where the mean relative density of column i passes `level` (-), interpolated; density decreases upward
function density_height(r, i, level)
    z = cell_heights(r, i)
    s = relative_density.(view(r.rho, :, i))
    for k in 1:(length(z) - 1)
        if (s[k] - level) * (s[k + 1] - level) <= 0 && s[k] != s[k + 1]
            return z[k] + (level - s[k]) / (s[k + 1] - s[k]) * (z[k + 1] - z[k])
        end
    end
    return NaN
end

# Thickness (m) of the density interface (between relative densities 0.9 and 0.1) and vorticity thickness (m) of the
# shear layer, (U_UPPER - U_LOWER) / max du/dz in SHEAR_ZONE, of column i
interface_thickness(r, i) = density_height(r, i, INTERFACE_LEVELS[1]) - density_height(r, i, INTERFACE_LEVELS[2])
function vorticity_thickness(r, i)
    z = cell_heights(r, i)
    u = view(r.u, :, i)
    shear = maximum(abs((u[k + 1] - u[k]) / (z[k + 1] - z[k])) for k in 1:(length(z) - 1)
                    if SHEAR_ZONE[1] <= 0.5 * (z[k] + z[k + 1]) <= SHEAR_ZONE[2])
    return (U_UPPER - U_LOWER) / shear
end

function inflow_figure()
    panels = [plot(; xlabel=x, ylabel="z (m)", title=t, titlefontsize=10, legend=:topleft)
              for (x, t) in (("u (m/s)", "velocity"), ("k (m²/s²)", "k"), ("eps (m²/s³)", "eps"))]
    xaxis!(panels[3], :log10)
    for (n, dx_cm) in enumerate(RESOLUTIONS)
        isfile(inflow_path(dx_cm)) || continue
        d = deserialize(inflow_path(dx_cm))   # the inflow of the plate of issue #42
        z = vcat(d.lower.zc, H_LOWER .+ d.upper.zc)
        for (p, field, field_in) in zip(panels, (:u, :k, :eps), (:shape, :k_in, :eps_in))
            n == 1 && plot!(p, vcat(getfield(d.lower, field_in), getfield(d.upper, field_in)), z; color=:gray,
                ls=:dot, lw=1.5, label="inlet of the precursors")
            plot!(p, vcat(getfield(d.lower, field), getfield(d.upper, field)), z; color=n, lw=1.5,
                label=@sprintf("after 4 m, dx = %d cm", dx_cm))
        end
    end
    for p in panels
        hline!(p, [H_LOWER]; color=:black, ls=:dash, label="plate")
    end
    save_figure(plot(panels...; layout=(1, 3), size=(1200, 450), margin=FIGURE_MARGIN_MM), "splitter_plate_inflow.png")
end

const DATA_DIR = joinpath(@__DIR__, "data")

# Measured profiles, extracted visually from the figures of the Delft3D-FLOW Validation Document (2008)
# (data/*.csv, digitize_measurements.py): station x (m) -> (z (m), value)
function load_measurements(name)
    path = joinpath(DATA_DIR, name * ".csv")
    isfile(path) || return Dict{Float64,Tuple{Vector{Float64},Vector{Float64}}}()
    out = Dict{Float64,Tuple{Vector{Float64},Vector{Float64}}}()
    for line in eachline(path)
        (startswith(line, "#") || startswith(line, "x_m")) && continue
        x, z, v = parse.(Float64, split(line, ","))
        zs, vs = get!(out, x, (Float64[], Float64[]))
        push!(zs, z)
        push!(vs, v)
    end
    return out
end

# Profiles of u, relative density and k at the stations against the measurements; `entries`: (label, style, colour, run)
function profile_comparison(entries, name)
    isempty(entries) && return
    measured = Dict(q => load_measurements(n) for (q, n) in ((:u, "velocity"), (:rho, "relative_density"), (:k, "tke")))
    rows = Any[]
    for xs in STATIONS
        pu = plot(; xlabel="u (m/s)", ylabel="z (m)", title=@sprintf("x = %g m: u", xs), titlefontsize=9, legend=false)
        pr = plot(; xlabel="relative density (-)", title="(rho - rho_upper) / (rho_lower - rho_upper)", titlefontsize=9,
            legend=false, xlims=(-0.05, 1.05))
        pk = plot(; xlabel="k (m²/s²)", title="k", titlefontsize=9, legend=xs == STATIONS[1] ? :topright : false)
        for (label, style, color, r) in entries
            i = column_index(r, xs)
            z = cell_heights(r, i)
            plot!(pu, r.u[:, i], z; ls=style, color=color, lw=1.5)
            plot!(pr, relative_density.(r.rho[:, i]), z; ls=style, color=color, lw=1.5)
            plot!(pk, r.k[:, i], z; ls=style, color=color, lw=1.5, label=label)
        end
        for (p, q) in ((pu, :u), (pr, :rho), (pk, :k))
            haskey(measured[q], xs) || continue
            zs, vs = measured[q][xs]
            scatter!(p, vs, zs; marker=:circle, markersize=3.5, markercolor=:white, markerstrokecolor=:red,
                label=q == :k && xs == STATIONS[1] ? "measured (extracted, Delft3D-FLOW Validation Document 2008)" : "")
        end
        append!(rows, (pu, pr, pk))
    end
    save_figure(plot(rows...; layout=(length(STATIONS), 3), size=(1200, 330 * length(STATIONS)), margin=FIGURE_MARGIN_MM),
        name)
end

function profiles_figure()
    entries = [(run_label(key...), RUN_STYLE[key]..., load_run(key...)) for key in RUNS]
    profile_comparison(filter(e -> e[4] !== nothing, entries), "splitter_plate_profiles.png")
end

function sensitivity_figure()
    entries = [(label, :solid, n, load_sensitivity_run(key, scheme)) for (n, (label, key, scheme)) in enumerate(SENSITIVITY)]
    profile_comparison(filter(e -> e[4] !== nothing, entries), "splitter_plate_sensitivity.png")
end

# Three heatmaps (relative density, u, k) of the fields rho, u, k on the cells h (all nz x nx); title prefix `label`
function field_panels(xc, h, rho, u, k, label)
    nz = size(rho, 1)
    z = cumsum(h[:, 1]) .- 0.5 .* h[:, 1]
    wet = h .> 0
    masked(f) = [wet[kk, i] ? f(kk, i) : NaN for kk in 1:nz, i in eachindex(xc)]
    common = (; xlabel="x (m)", ylabel="z (m)", titlefontsize=10, ylims=(0, LEVEL_IN))
    return [heatmap(xc, z, masked((kk, i) -> relative_density(rho[kk, i])); color=:viridis, clims=(0, 1),
                colorbar_title="relative density", title="$label: relative density", common...),
            heatmap(xc, z, masked((kk, i) -> u[kk, i]); color=:viridis, clims=U_CLIMS, colorbar_title="u (m/s)",
                title="$label: u", common...),
            heatmap(xc, z, masked((kk, i) -> k[kk, i] / K_UNIT); color=:inferno, clims=K_CLIMS, colorbar_title="k (10⁻⁴ m²/s²)",
                title="$label: k", common...)]
end

function fields_figure()
    panels = Any[]
    for key in RUNS
        r = load_run(key...)
        r === nothing && continue
        append!(panels, field_panels(r.xc, r.h, r.rho, r.u, r.k, "mean, " * run_label(key...)))
    end
    isempty(panels) && return
    save_figure(plot(panels...; layout=(length(panels), 1), size=(1200, 230 * length(panels)), margin=FIGURE_MARGIN_MM),
        "splitter_plate_fields.png")
end

function interface_figure()
    pd = plot(; xlabel="x (m)", ylabel="thickness (m)", titlefontsize=10, legend=:topleft,
        title=@sprintf("density interface (relative density %.1f to %.1f)", INTERFACE_LEVELS[2], INTERFACE_LEVELS[1]))
    pv = plot(; xlabel="x (m)", ylabel="thickness (m)", title="shear layer: (U_upper - U_lower) / max du/dz",
        titlefontsize=10, legend=false)
    for key in RUNS
        r = load_run(key...)
        r === nothing && continue
        style, color = RUN_STYLE[key]
        plot!(pd, r.xc, [interface_thickness(r, i) for i in eachindex(r.xc)]; ls=style, color=color, lw=1.5,
            label=run_label(key...))
        plot!(pv, r.xc, [vorticity_thickness(r, i) for i in eachindex(r.xc)]; ls=style, color=color, lw=1.5)
    end
    save_figure(plot(pd, pv; layout=(1, 2), size=(1200, 420), margin=FIGURE_MARGIN_MM), "splitter_plate_interface.png")
end

function budget_figure()
    pq = plot(; xlabel="t (s)", ylabel="discharge (m²/s)", title="inflow and outflow", titlefontsize=10, legend=:bottomright)
    pb = plot(; xlabel="t (s)", ylabel="-", title="salt budget error / cumulated inflow of salt excess", titlefontsize=10,
        legend=false)
    for key in RUNS
        r = load_run(key...)
        r === nothing && continue
        style, color = RUN_STYLE[key]
        s = r.series
        key == RUNS[1] && plot!(pq, s.t, s.q_in; color=:black, lw=1, label="inflow")
        plot!(pq, s.t, s.q_out; ls=style, color=color, lw=1.5, label="outflow, " * run_label(key...))
        plot!(pb, s.t, s.budget ./ max.(s.excess_in, eps()); ls=style, color=color, lw=1.5)
    end
    vspan!(pq, [RUN_TIME - AVERAGE_TIME, RUN_TIME]; color=:gray, alpha=0.15, label="averaging window")
    save_figure(plot(pq, pb; layout=(1, 2), size=(1200, 420), margin=FIGURE_MARGIN_MM), "splitter_plate_budget.png")
end

function figures()
    inflow_figure()
    profiles_figure()
    sensitivity_figure()
    fields_figure()
    interface_figure()
    budget_figure()
end

function main(args)
    isempty(args) && error("command: precursors [plate] | run <dx_cm> <mode> [plate [c3]] | runs | sensitivity | figures")
    cmd = args[1]
    if cmd == "precursors"
        precursors(length(args) > 1 ? Symbol(args[2]) : :frictionless)
    elseif cmd == "run"
        run_case(parse(Int, args[2]), args[3], length(args) > 3 ? Symbol(args[4]) : :frictionless,
            length(args) > 4 ? parse(Float64, args[5]) : C3_DEFAULT)
    elseif cmd == "runs"
        for key in RUNS
            run_case(key...)
        end
    elseif cmd == "sensitivity"
        precursors(:rough)
        run_case(10, "nonhydrostatic", :rough)
        run_case(10, "nonhydrostatic", :frictionless, C3_RI025)
        withenv(() -> run_case(10, "nonhydrostatic"), "SP_TRANSPORT" => "upwind1")
    elseif cmd == "figures"
        figures()
    else
        error("unknown command $cmd")
    end
end

abspath(PROGRAM_FILE) == @__FILE__() && main(ARGS)
