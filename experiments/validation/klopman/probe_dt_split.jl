# Delta t sensitivity of the Klopman wave effect (issue #85): CMP on the coarse grid (nx = 380, nz = 20, theta = 0.55,
# lambda2 = 0.1) from the cached #SP state, at the given N, with one candidate switched off by a method override.
# ARGS: variant ("ref" | "frozen_ke"), N list (e.g. 200,400), case (CMP).
# frozen_ke: the k-epsilon step is a no-op, nu_t/k/eps stay at the current-only state.
# Needs the cached state results/state/SP_state_nz20_l0.1_nx380.jls (not in git): KL_NX=380 julia --project=. run_klopman_cases.jl 20 CMP 0.1
root = @__DIR__
include(joinpath(root, "cases.jl"))
using Serialization
variant = get(ARGS, 1, "ref")
Ns = parse.(Int, split(get(ARGS, 2, "200,400"), ","))
case = Symbol(get(ARGS, 3, "CMP"))
if variant == "frozen_ke"
    @eval SweTransport2DV _wd_k_epsilon_step!(model::Model, wd::WetDryState, ke::KEpsilonState) = nothing
end
const NX, NZ, THETA, AMP = 380, 20, 0.55, (case === :CMP ? 0.0708 : 0.0705)
const PERIODS, AVG = 150, 50
saved = deserialize(joinpath(root, "results", "state", "SP_state_nz20_l0.1_nx380.jls"))
init = (state=(u=saved.u, eta=saved.eta), turbulence=saved.turbulence)
function shear(r)
    k = findall(z -> 0.25 <= z <= 0.36, r.z); z, u = r.z[k], r.u[k]
    zm, um = sum(z) / length(z), sum(u) / length(u)
    return sum((z .- zm) .* (u .- um)) / sum(abs2, z .- zm)
end
meanu(r) = (k = findall(z -> 0.02 <= z <= 0.36, r.z); sum(r.u[k]) / length(k))
for N in Ns
    t0 = time()
    m = klopman_model(case; nx=NX, nz=NZ, amplitude=AMP, theta=THETA, steps_per_period=N, levels=saved.levels,
        initial=init, options=(; ke_wave_stabilization=0.1))
    st = klopman_station_column(m)
    nall = PERIODS * N
    avg = StationAverage(m)
    q = 0.0   # m^2/s, period-mean discharge through the station face
    for n in 1:nall
        step_forward!(m)
        if n > nall - AVG * N
            accumulate!(avg, m)
            q += sum(m.wetdry.hf[:, st] .* m.state.u[:, st])
        end
    end
    r = result(avg, m)
    @printf("%s %s N %d: mean u %.4f m/s, shear %+.4f 1/s, ratio %+.3f, mean Q station %.5f m^2/s, eta1 %.4f m (%.0f s)\n",
        variant, case, N, meanu(r), shear(r), shear(r) / meanu(r), q / (AVG * N), r.eta1, time() - t0)
    flush(stdout)
end
