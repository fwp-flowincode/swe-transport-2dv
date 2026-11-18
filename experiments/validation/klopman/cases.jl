# Klopman's flume (Klopman 1994, Delft Hydraulics report H840.30 Part II): regular waves following and opposing a current,
# validation case 2DV-13 (issue #85, docs/plan_issue_85_klopman_flume.md). Model set-up and the diagnostics at the LDV
# station; included by the validation script and the regression test.
#
# The model covers the flat test section (x = 6.4-40.6 m of the flume) with two free open faces (issue #86,
# docs/numerics_open_boundaries.md sections 7.4-7.6) instead of the wave boards with active absorption: both faces radiate
# the design wave with the calibrated phase speed and relax the incoming characteristic variable towards the total
# discharge q and the still-water level; the wave enters as the incoming level of the face the board stood at.

using SweTransport2DV
using Printf
using Statistics

# Experiment (report sections in brackets)
const KL_G = 9.81                    # m/s^2, ModelConfig default
const KL_DEPTH = 0.5                 # m, still-water depth over the flat test section (2.2.1)
const KL_SECTION_START = 6.4         # m, flume x of the start of the flat test section (2.2.1, Fig. 2.1)
const KL_SECTION_END = 40.6          # m, flume x of its end
const KL_LENGTH = KL_SECTION_END - KL_SECTION_START   # m, model domain
const KL_STATION = 22.5              # m, flume x of the LDVs (2.3)
const KL_DISCHARGE = 0.0805          # m^2/s, total (mass-transport) discharge per metre width: 80.5 l/s in 1.0 m (3.1.1, 3.2.6)
const KL_PERIOD = 1.44               # s, regular waves (3.1.1)
const KL_AMPLITUDE = 0.060           # m, incident amplitude at the station (3.1.1, Table 4.1a)
const KL_Z0 = 4e-5                   # m, bed roughness length from the log fit of the current-only test (4.4)
const KL_NU = 1.2e-6                 # m^2/s, water at 11-16 degC (Table 3.3)
const KL_TROUGH_MARGIN = 0.035       # m, the data are valid up to this distance below the wave trough (4.1)
# Model
# WHY theta = 0.6 and N = 400: the theta-scheme damps a wave by about 4 pi^2 (theta - 1/2)/N per period; at 0.7 and
# 100 the wave lost 45 % of its amplitude between the generating face and the station, and a decaying wave drives a
# spurious mean flow (radiation-stress gradient about 30 times the bed stress of the current). With a current the
# scheme needs theta >= (1 + 2 Fr)/(2 (1 + Fr)) = 0.57 (Fr about 0.15 with the orbital velocity), so 0.6; N = 400
# then leaves about 0.005 1/m (docs/plan_issue_85_klopman_flume.md, section 3.3).
const KL_THETA = 0.6                 # -, free-surface implicitness
const KL_STEPS_PER_PERIOD = 400      # -, Delta t = T / 400
const KL_SPINUP_STEPS_PER_PERIOD = 100   # -, Delta t of the current-only spin-up (a steady flow does not depend on it)
# Top layer and face level (issue #88): merged top layer and second-order face level, which remove the amplitude-
# dependent damping of the non-hydrostatic step (an extra 0.012 1/m along the flume with the defaults)
const KL_SCHEME = (; wd_top_layer="merged", wd_face_level="limited")
const KL_RAMP_PERIODS = 3            # -, smooth start of the incoming wave
const KL_RELAXATION_TIME = 20.0      # s, relaxation of the incoming variable at both faces (about 14 periods)

const KL_CASES = (:SP, :WMN, :CMP, :CMN)   # current only, waves only, following, opposing (3.1.1)

"Current q (m^2/s) and the face the wave enters through (:none, :left, :right) of a test."
function klopman_case(case::Symbol)
    case === :SP && return (q=KL_DISCHARGE, wave=:none)
    case === :WMN && return (q=0.0, wave=:right)    # waves travel -x, from WB-2
    case === :CMP && return (q=KL_DISCHARGE, wave=:left)
    case === :CMN && return (q=KL_DISCHARGE, wave=:right)
    throw(ArgumentError("unknown Klopman test $(case); choose from $(KL_CASES)"))
end

"""
    klopman_model(case; nx=760, nz=20, amplitude=KL_AMPLITUDE, nonhydro=true, calibrate=true,
                  theta=KL_THETA, steps_per_period=KL_STEPS_PER_PERIOD, levels=(0.0, 0.0), initial=nothing,
                  options=(;))

Model of test `case` (`:SP`, `:WMN`, `:CMP`, `:CMN`) on `nx` x `nz` cells. `amplitude` (m) is the incoming level
amplitude at the generating face (tune it to the measured 0.060 m at the station). The initial state is the uniform
current q/h at rest level, or the state of the model `initial` (u, eta, k, epsilon; e.g. the steady current-only run);
`calibrate` sets `radiation_speed_factor` from `numerical_phase_speed_factor` on the current the wave travels against
or with. `levels` (m, left and right): target levels of the relaxed faces; the steady face levels of the current-only
run, so that the steady state lies on both relaxation lines with the target discharge (plan section 3.2). `options`
override the configuration (after KL_SCHEME).
"""
function klopman_model(case::Symbol; nx::Int=760, nz::Int=20, amplitude::Real=KL_AMPLITUDE, nonhydro::Bool=true,
                       calibrate::Bool=true, theta::Real=KL_THETA, steps_per_period::Int=KL_STEPS_PER_PERIOD,
                       levels=(0.0, 0.0), initial=nothing, options=(;))
    spec = klopman_case(case)
    h, T = KL_DEPTH, KL_PERIOD
    cfg = ModelConfig(; nx=nx, nz=nz, length=KL_LENGTH, depth=h, dt=T / steps_per_period, theta=theta,
        implicit_vertical_terms=true, nu=KL_NU, rayleigh_damping=0.0, apply_froude_limiter=false,
        momentum_advection_scheme="tvd2", use_hydrostatic=!nonhydro, wetting_drying=true, bed_level=zeros(nx),
        write_plots=false, write_animation=false, write_pressure_buoyancy_diagnostics=false,
        turbulence_closure="k_epsilon", bed_roughness_length=KL_Z0, radiation_theta=true,
        nonhydro_open_face=nonhydro ? "radiation" : "neumann", radiation_period=nonhydro ? T : 0.0,
        open_face_relaxation="incoming", KL_SCHEME..., options...)
    U = spec.q / h   # m/s, depth-mean current
    # WHY the current in the direction of the wave: the factor is the model's phase speed of the design wave on it
    calibrate && nonhydro && (cfg.radiation_speed_factor = numerical_phase_speed_factor(cfg;
        current=spec.wave === :right ? -U : U))
    model = Model(cfg)
    fill!(model.state.rho, cfg.rho0)
    refresh_density_metadata!(model)
    initialize_water_level!(model, fill(h, nx))
    model.state.u .= U
    if initial !== nothing
        size(initial.state.u) == size(model.state.u) || throw(ArgumentError("initial state has another grid"))
        copyto!(model.state.u, initial.state.u)
        copyto!(model.state.eta, initial.state.eta)
        if model.turbulence !== nothing && initial.turbulence !== nothing
            copyto!(model.turbulence.k, initial.turbulence.k)
            copyto!(model.turbulence.eps, initial.turbulence.eps)
            copyto!(model.turbulence.nu_t, initial.turbulence.nu_t)
        end
    end
    omega = 2pi / T
    eta_in(t) = amplitude * 0.5 * (1 - cos(pi * min(t / (KL_RAMP_PERIODS * T), 1.0))) * sin(omega * t)
    incoming = spec.wave === :left ? (; left_incoming_level=eta_in) :
               spec.wave === :right ? (; right_incoming_level=eta_in) : (;)
    tau = KL_RELAXATION_TIME
    set_x_inflow_outflow!(model, zeros(nz); left_free=true, left_level=levels[1], left_relaxation_time=tau,
        left_target_discharge=spec.q, right_free=true, right_level=levels[2], right_relaxation_time=tau,
        right_target_discharge=spec.q, incoming...)
    return model
end

const KL_TUNE_MARGIN_PERIODS = 2    # -, after the arrival of the full wave at the station, before the tuning average

"""
    arrival_periods(case) -> periods (-)

Time (in wave periods) until the full incoming wave stands at the station: the ramp (KL_RAMP_PERIODS) plus the travel
time from the generating face at the absolute group speed c_g + U_n (linear theory with the Doppler shift,
(omega - k U_n)^2 = g k tanh(k h), U_n the current in the direction of the wave), rounded up, plus
KL_TUNE_MARGIN_PERIODS. WHY: the tuning average must not see the arriving front (the opposing wave needs about 16
periods; the former fixed 20 periods before the average left little margin).
"""
function arrival_periods(case)
    spec = klopman_case(case)
    h, T, g = KL_DEPTH, KL_PERIOD, KL_G
    omega = 2pi / T
    Un = spec.wave === :left ? spec.q / h : -spec.q / h   # m/s, current in the direction of the wave
    k = omega^2 / g
    for _ in 1:50   # Newton on (omega - k U)^2 - g k tanh(k h)
        fk = (omega - k * Un)^2 - g * k * tanh(k * h)
        dfk = -2Un * (omega - k * Un) - g * tanh(k * h) - g * k * h / cosh(k * h)^2
        k -= fk / dfk
    end
    sigma = omega - k * Un
    cg = sigma / k * 0.5 * (1 + 2k * h / sinh(2k * h)) + Un   # m/s, absolute group speed
    dist = spec.wave === :left ? KL_STATION - KL_SECTION_START : KL_SECTION_END - KL_STATION   # m, face to station
    return KL_RAMP_PERIODS + ceil(Int, dist / (cg * T)) + KL_TUNE_MARGIN_PERIODS
end

"Column index of the LDV station."
klopman_station_column(model) = clamp(round(Int, (KL_STATION - KL_SECTION_START) / model.dx + 0.5), 1, model.config.nx)

"""
    StationAverage(model; column=klopman_station_column(model))

Accumulator of the period averages at one column: Eulerian mean u and the complex first harmonic of u at the cell
centres (z = (k - 1/2) dz above the bed; a sample counts only while the cell is entirely wet), mean nu_t and k, and the
first harmonic and the trough of the surface. `accumulate!(avg, model)` after every step, `result(avg)` at the end.
"""
mutable struct StationAverage
    column::Int
    omega::Float64
    n::Int
    nwet::Vector{Int}
    u::Vector{Float64}
    u1::Vector{ComplexF64}
    nut::Vector{Float64}
    k::Vector{Float64}
    eta1::ComplexF64
    eta_min::Float64
end

function StationAverage(model; column=klopman_station_column(model))
    nz = model.config.nz
    return StationAverage(column, 2pi / KL_PERIOD, 0, zeros(Int, nz), zeros(nz), zeros(ComplexF64, nz), zeros(nz),
        zeros(nz), 0.0im, Inf)
end

function accumulate!(avg::StationAverage, model)
    i, nz, dz = avg.column, model.config.nz, model.dz
    wd = model.wetdry
    t = wd.time
    ph = exp(im * avg.omega * t)
    avg.n += 1
    eta = water_depth(model)[i] - KL_DEPTH
    avg.eta1 += eta * ph
    avg.eta_min = min(avg.eta_min, eta)
    for k in 1:nz
        wd.h[k, i] >= dz * (1 - 1e-12) || continue   # WHY: only cells wet over their whole height are fixed points
        uc = 0.5 * (model.state.u[k, i] + model.state.u[k, i + 1])
        avg.nwet[k] += 1
        avg.u[k] += uc
        avg.u1[k] += uc * ph
        # WHY the guard: without a closure (turbulence_closure="none") the model carries no turbulence state
        model.turbulence === nothing && continue
        avg.nut[k] += model.turbulence.nu_t[k, i]
        avg.k[k] += model.turbulence.k[k, i]
    end
    return avg
end

"Named tuple of the averages (z in m above the bed; layers wet in every sample only)."
function result(avg::StationAverage, model)
    dz = model.dz
    keep = findall(==(avg.n), avg.nwet)
    z = [(k - 0.5) * dz for k in keep]
    return (z=z, u=avg.u[keep] ./ avg.n, u1=2 .* abs.(avg.u1[keep]) ./ avg.n, nut=avg.nut[keep] ./ avg.n,
        k=avg.k[keep] ./ avg.n, eta1=2abs(avg.eta1) / avg.n, eta_min=avg.eta_min, samples=avg.n)
end

"Read a reference CSV of experiments/validation/klopman/references (comment lines start with #): (z, value)."
function klopman_reference(name::AbstractString)
    path = joinpath(@__DIR__, "references", name)
    rows = [split(l, ',') for l in eachline(path) if !startswith(l, '#') && !startswith(l, 'z')]
    return (z=[parse(Float64, r[1]) for r in rows], value=[parse(Float64, r[2]) for r in rows])
end
