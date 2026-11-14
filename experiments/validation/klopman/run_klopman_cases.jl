# Klopman (1994) wave-current flume, validation case 2DV-13 (issue #85): runs the four tests on one vertical grid and
# writes the period-averaged station profiles to results/.
#
#   julia --project=. experiments/validation/klopman/run_klopman_cases.jl [nz] [cases] [lambda2] [bed]
#
# nz: layers (default 20); cases: comma-separated subset of WMN,CMP,CMN (default all); lambda2: k-epsilon stabilisation
# under waves (ModelConfig.ke_wave_stabilization, default 0; result files then end in _l<lambda2>); bed: "swart" for the
# wave-enhanced bed friction (ModelConfig.wave_bed_friction; files then end in _swart), default the log law.
# ENV KL_RUN_PERIODS: production periods per case, e.g. "WMN=300" (default KL_RUN_PERIODS for all).
# Caches in results/state/ (per nz and option tag): the steady #SP state and the tuned incoming amplitude of each case
# are reused by later runs with the same settings; ENV KL_FRESH=true ignores and rewrites them (after a model change).
# Procedure
# (docs/plan_issue_85_klopman_flume.md):
# 1. current only (#SP) from the uniform current with target levels 0 to a steady state (Delta t = T/100: a steady flow
#    does not depend on it); its face levels become the target levels; continued with them until steady again. Writes
#    the station profile and the discharges.
# 2. each wave case from the steady #SP state: a tuning run gives the station amplitude (averaged over
#    KL_TUNE_AVERAGE_PERIODS from the arrival of the full wave at the station, `arrival_periods`), the incoming
#    amplitude is scaled to the measured 0.060 m there, then the production run of KL_RUN_PERIODS. Averages over the
#    last KL_AVERAGE_PERIODS; the steadiness check compares the two halves of that window.

include(joinpath(@__DIR__, "cases.jl"))

const KL_SPINUP_TIME = 600.0        # s, current only from the uniform current (steady after about 500 s, plan 3.2)
const KL_SPINUP_LEVEL_TIME = 300.0  # s, continuation with the target levels from the first spin-up
const KL_SP_AVERAGE_TIME = 100.0    # s, averaging window of the current-only profile
const KL_TUNE_AVERAGE_PERIODS = 10  # -, averaging window of the tuning run
const KL_TUNE_MARGIN_PERIODS = 2    # -, after the arrival of the full wave at the station, before the tuning average
const KL_RUN_PERIODS = 150          # -, production run of a wave case
const KL_AVERAGE_PERIODS = 50       # -, averaging window at the end of the production run

results_dir = joinpath(@__DIR__, "results")
mkpath(results_dir)
nz = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20
cases = length(ARGS) >= 2 ? Symbol.(split(ARGS[2], ",")) : [:WMN, :CMP, :CMN]
lambda2 = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.0
bed = length(ARGS) >= 4 ? ARGS[4] : "none"
opts = (; ke_wave_stabilization=lambda2, wave_bed_friction=bed)
run_periods = Dict(Symbol(first(split(e, "="))) => parse(Int, last(split(e, "="))) for e in split(get(ENV, "KL_RUN_PERIODS", ""), ",") if occursin("=", e))
tag = (lambda2 > 0 ? "_l$(lambda2)" : "") * (bed == "none" ? "" : "_$(bed)")
state_dir = joinpath(results_dir, "state")
mkpath(state_dir)
fresh = get(ENV, "KL_FRESH", "false") == "true"
using Serialization

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

# m^2/s: discharge through a u-face (sum hf u)
face_discharge(model, f) = sum(model.wetdry.hf[:, f] .* model.state.u[:, f])
discharges(model) = (left=face_discharge(model, 1), station=face_discharge(model, klopman_station_column(model)),
    right=face_discharge(model, model.config.nx + 1))

function run_time!(model, duration; avg=nothing, every=nothing, label="")
    n = round(Int, duration / model.config.dt)
    t0 = time()
    for s in 1:n
        step_forward!(model)
        avg === nothing || accumulate!(avg, model)
        if every !== nothing && s % every == 0
            q = discharges(model)
            @printf("  %s t = %7.1f s: Q left/station/right %.5f %.5f %.5f m^2/s, eta 1/nx %+.5f %+.5f m (%.0f s)\n", label,
                model.wetdry.time, q.left, q.station, q.right, model.state.eta[1], model.state.eta[end], time() - t0)
            flush(stdout)
        end
    end
    return model
end

function write_profile(name, r, header)
    open(joinpath(results_dir, name), "w") do io
        for line in header
            println(io, "# ", line)
        end
        println(io, "z_m,u_mean_m_s,u1_amplitude_m_s,nu_t_m2_s,k_m2_s2")
        for j in eachindex(r.z)
            @printf(io, "%.5f,%.6f,%.6f,%.4e,%.4e\n", r.z[j], r.u[j], r.u1[j], r.nut[j], r.k[j])
        end
    end
end

t_all = time()
sp_cache = joinpath(state_dir, "SP_state_nz$(nz)$(tag).jls")
if !fresh && isfile(sp_cache)
    # the steady current-only state of an earlier run with the same settings (u, eta, k, eps, nu_t, target levels)
    saved = deserialize(sp_cache)
    levels = saved.levels
    sp2 = (state=(u=saved.u, eta=saved.eta), turbulence=saved.turbulence)
    @printf("== nz = %d: current only (#SP) loaded from %s (target levels %+.5f / %+.5f m)\n", nz, basename(sp_cache), levels...)
else
    println("== nz = $(nz): current only (#SP), spin-up at N = $(KL_SPINUP_STEPS_PER_PERIOD)")
    sp = klopman_model(:SP; nz=nz, steps_per_period=KL_SPINUP_STEPS_PER_PERIOD, options=opts)
    log_every = round(Int, 100.0 / sp.config.dt)
    run_time!(sp, KL_SPINUP_TIME; every=log_every, label="SP")
    levels = (sp.state.eta[1], sp.state.eta[end])
    @printf("target levels from the spin-up: left %+.5f m, right %+.5f m\n", levels...)
    spm = klopman_model(:SP; nz=nz, steps_per_period=KL_SPINUP_STEPS_PER_PERIOD, levels=levels, initial=sp, options=opts)
    run_time!(spm, KL_SPINUP_LEVEL_TIME - KL_SP_AVERAGE_TIME; every=log_every, label="SP levels")
    avg = StationAverage(spm)
    run_time!(spm, KL_SP_AVERAGE_TIME; avg=avg, every=log_every, label="SP average")
    r = result(avg, spm)
    q = discharges(spm)
    write_profile("SP_nz$(nz)$(tag).csv", r, ["Klopman #SP (current only), nz = $(nz), nx = $(spm.config.nx), average over the last $(KL_SP_AVERAGE_TIME) s",
        @sprintf("target levels left %+.5f m, right %+.5f m; Q left/station/right %.5f %.5f %.5f m^2/s (target %.4f)",
            levels..., q.left, q.station, q.right, KL_DISCHARGE)])
    turb = spm.turbulence === nothing ? nothing :
        (k=copy(spm.turbulence.k), eps=copy(spm.turbulence.eps), nu_t=copy(spm.turbulence.nu_t))
    serialize(sp_cache, (u=copy(spm.state.u), eta=copy(spm.state.eta), turbulence=turb, levels=levels))
    sp2 = spm
end

for case in cases
    println("== nz = $(nz): $(case)")
    spec = klopman_case(case)
    init = spec.q > 0.0 ? sp2 : nothing   # WHY: waves only has no current; it starts at rest
    lv = spec.q > 0.0 ? levels : (0.0, 0.0)
    T = KL_PERIOD
    amp_cache = joinpath(state_dir, "amplitude_$(case)_nz$(nz)$(tag).txt")
    if !fresh && isfile(amp_cache)
        a = parse(Float64, strip(read(amp_cache, String)))
        @printf("tuning: production amplitude %.4f m loaded from %s\n", a, basename(amp_cache))
    else
        a0 = KL_AMPLITUDE
        tune = klopman_model(case; nz=nz, amplitude=a0, levels=lv, initial=init, options=opts)
        nstart = arrival_periods(case)
        run_time!(tune, nstart * T)
        tavg = StationAverage(tune)
        run_time!(tune, KL_TUNE_AVERAGE_PERIODS * T; avg=tavg)
        a_station = result(tavg, tune).eta1
        a = a0 * KL_AMPLITUDE / a_station
        write(amp_cache, @sprintf("%.6f\n", a))
        @printf("tuning (%d + %d periods): station amplitude %.4f m with %.4f m at the face; production amplitude %.4f m\n",
            nstart, KL_TUNE_AVERAGE_PERIODS, a_station, a0, a)
    end
    model = klopman_model(case; nz=nz, amplitude=a, levels=lv, initial=init, options=opts)
    every = round(Int, 25 * T / model.config.dt)
    nrun = get(run_periods, case, KL_RUN_PERIODS)
    run_time!(model, (nrun - KL_AVERAGE_PERIODS) * T; every=every, label=string(case))
    half = KL_AVERAGE_PERIODS ÷ 2
    avg1 = StationAverage(model)
    run_time!(model, half * T; avg=avg1)
    avg2 = StationAverage(model)
    run_time!(model, (KL_AVERAGE_PERIODS - half) * T; avg=avg2)
    r1, r2 = result(avg1, model), result(avg2, model)
    # the whole window: mean of the halves (equal sample counts up to one step)
    keep = intersect(r1.z, r2.z)
    i1 = [findfirst(==(z), r1.z) for z in keep]
    i2 = [findfirst(==(z), r2.z) for z in keep]
    rr = (z=keep, u=0.5 .* (r1.u[i1] .+ r2.u[i2]), u1=0.5 .* (r1.u1[i1] .+ r2.u1[i2]), nut=0.5 .* (r1.nut[i1] .+ r2.nut[i2]),
        k=0.5 .* (r1.k[i1] .+ r2.k[i2]))
    drift = maximum(abs.(r2.u[i2] .- r1.u[i1]))
    local q = discharges(model)
    write_profile("$(case)_nz$(nz)$(tag).csv", rr, ["Klopman #$(case), nz = $(nz), nx = $(model.config.nx), theta = $(model.config.theta), N = $(KL_STEPS_PER_PERIOD), average over the last $(KL_AVERAGE_PERIODS) periods of $(nrun)",
        @sprintf("incoming amplitude %.4f m; station first-harmonic amplitude %.4f / %.4f m (two halves), trough %.4f m",
            a, r1.eta1, r2.eta1, min(r1.eta_min, r2.eta_min)),
        @sprintf("steadiness: max |u_mean(second half) - u_mean(first half)| = %.4f m/s", drift),
        @sprintf("Q left/station/right %.5f %.5f %.5f m^2/s (instantaneous, end of run)", q.left, q.station, q.right)])
    @printf("%s done: station amplitude %.4f m, mean-u drift between halves %.4f m/s\n", case, 0.5 * (r1.eta1 + r2.eta1), drift)
    flush(stdout)
end
@printf("all done in %.0f s\n", time() - t_all)
