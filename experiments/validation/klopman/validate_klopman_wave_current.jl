# Klopman (1994) wave-current flume, validation case 2DV-13 (issue #85): compares the station profiles written by
# run_klopman_cases.jl (results/<case>_nz<nz>.csv) with the digitised measurements (references/).
#
#   julia --project=. experiments/validation/klopman/validate_klopman_wave_current.jl [nz ...]
#
# ENV KL_TAG: suffix of the result files (e.g. "_l0.1" for the runs with the k-epsilon stabilisation); a wave case
# without a result file of that tag is taken from the untagged run, if there is one, and marked as such.
#
# Prints, per grid:
# - RMS of the mean velocity, model - measured, over KL_RMS_ZMIN <= z <= trough - KL_TROUGH_MARGIN, normalised by
#   KL_U_SCALE, against the data as published and against the data times KL_CALIBRATION (the 7 % calibration bias of the
#   preliminary profiles, report section 4.1);
# - the primary criterion of issue #85: the mean shear du/dz over the upper half (least-squares slope over
#   KL_SHEAR_ZMIN <= z <= trough - margin) decreases for following waves (#CMP) and increases for opposing waves (#CMN)
#   relative to the current only (#SP), in the model and in the data;
# - RMS of the first-harmonic amplitude of u (model scaled to the measured station amplitude 0.060 m, and unscaled).
# Writes figures/klopman_profiles_nz<nz>.png.

include(joinpath(@__DIR__, "cases.jl"))
using Plots

const KL_U_SCALE = 0.16          # m/s, normalisation of the RMS (the mean mass-transport velocity, issue #85)
const KL_RMS_ZMIN = 0.02         # m, lower end of the RMS range (issue #85)
const KL_RMS_ZMAX = 0.40         # m, upper end at most (issue #85; the data stop there)
const KL_SHEAR_ZMIN = 0.25       # m, lower end of the upper half of the water column
const KL_CALIBRATION = 1.07      # -, factor of the calibration bias of the preliminary data (report section 4.1)

results_dir = joinpath(@__DIR__, "results")
figures_dir = joinpath(@__DIR__, "figures")
mkpath(figures_dir)

"Model profile of results/<name>.csv and its header lines."
const KL_TAG = get(ENV, "KL_TAG", "")
function model_profile(case, nz)
    path = joinpath(results_dir, "$(case)_nz$(nz)$(KL_TAG).csv")
    if !isfile(path)
        path = joinpath(results_dir, "$(case)_nz$(nz).csv")
        println("  (#$(case): no result with tag \"$(KL_TAG)\", the untagged run is used)")
    end
    lines = readlines(path)
    header = [l[3:end] for l in lines if startswith(l, '#')]
    rows = [parse.(Float64, split(l, ',')) for l in lines if !startswith(l, '#') && !startswith(l, 'z')]
    m = reduce(hcat, rows)'
    trough = something(tryparse(Float64, something(match(r"trough (-?[0-9.]+) m", join(header, " ")), (captures=["NaN"],)).captures[1]), NaN)
    eta1 = match(r"station first-harmonic amplitude ([0-9.]+) / ([0-9.]+) m", join(header, " "))
    a = eta1 === nothing ? NaN : 0.5 * (parse(Float64, eta1.captures[1]) + parse(Float64, eta1.captures[2]))
    return (z=m[:, 1], u=m[:, 2], u1=m[:, 3], nut=m[:, 4], header=header, trough=trough, a=a)
end

"Linear interpolation of (zs, vs) at z (nothing outside the range)."
function interp(zs, vs, z)
    (z < zs[1] || z > zs[end]) && return nothing
    j = searchsortedlast(zs, z)
    j == length(zs) && return vs[end]
    w = (z - zs[j]) / (zs[j + 1] - zs[j])
    return (1 - w) * vs[j] + w * vs[j + 1]
end

"Upper end (m) of the valid range: trough - margin (the trough of the run, below the still level), at most KL_RMS_ZMAX."
zmax(p) = isnan(p.trough) ? KL_RMS_ZMAX : min(KL_RMS_ZMAX, KL_DEPTH + p.trough - KL_TROUGH_MARGIN)

"RMS of model - data over [zlo, zhi] at the data heights, normalised by `scale`."
function rms_diff(pz, pv, dz, dv, zlo, zhi, scale)
    d = Float64[]
    for (z, v) in zip(dz, dv)
        zlo <= z <= zhi || continue
        m = interp(pz, pv, z)
        m === nothing || push!(d, m - v)
    end
    return isempty(d) ? NaN : sqrt(sum(abs2, d) / length(d)) / scale, length(d)
end

"Least-squares slope dv/dz over [zlo, zhi]."
function slope(z, v, zlo, zhi)
    keep = findall(x -> zlo <= x <= zhi, z)
    length(keep) >= 2 || return NaN
    zz, vv = z[keep], v[keep]
    zm, vm = sum(zz) / length(zz), sum(vv) / length(vv)
    return sum((zz .- zm) .* (vv .- vm)) / sum(abs2, zz .- zm)
end

grids = isempty(ARGS) ? [20] : parse.(Int, ARGS)
for nz in grids
    println("== nz = $(nz)")
    P = Dict(c => model_profile(c, nz) for c in KL_CASES)
    D = Dict(c => klopman_reference("klopman1994_mean_u_$(c).csv") for c in KL_CASES)
    A = Dict(c => klopman_reference("klopman1994_u1_amplitude_$(c).csv") for c in (:WMN, :CMP, :CMN))
    # the valid range of the wave cases ends 35 mm below the trough of the run; #SP has no trough
    println("mean u: RMS(model - data)/0.16 over 0.02 m <= z <= z_max (data as published | data x 1.07)")
    for c in KL_CASES
        p, d = P[c], D[c]
        zhi = zmax(p)
        r1, n = rms_diff(p.z, p.u, d.z, d.value, KL_RMS_ZMIN, zhi, KL_U_SCALE)
        r2, _ = rms_diff(p.z, p.u, d.z, KL_CALIBRATION .* d.value, KL_RMS_ZMIN, zhi, KL_U_SCALE)
        @printf("  %-4s z_max %.3f m, %2d points: %.3f | %.3f\n", c, zhi, n, r1, r2)
    end
    println("primary criterion: upper-half shear du/dz (1/s), least squares over 0.25 m <= z <= z_max")
    zsp = min(zmax(P[:CMP]), zmax(P[:CMN]))   # WHY one range for all: the shear of #SP is compared on the same heights
    s = Dict(c => (model=slope(P[c].z, P[c].u, KL_SHEAR_ZMIN, zsp), data=slope(D[c].z, D[c].value, KL_SHEAR_ZMIN, zsp)) for c in KL_CASES)
    for c in KL_CASES
        @printf("  %-4s model %+.4f, data %+.4f\n", c, s[c].model, s[c].data)
    end
    following = s[:CMP].model < s[:SP].model
    opposing = s[:CMN].model > s[:SP].model
    @printf("  following waves decrease the shear: %s (model %+.4f vs %+.4f; turns negative: %s)\n", following,
        s[:CMP].model, s[:SP].model, s[:CMP].model < 0)
    @printf("  opposing waves increase the shear: %s (model %+.4f vs %+.4f)\n", opposing, s[:CMN].model, s[:SP].model)
    println("first-harmonic amplitude of u: RMS(model - data)/0.16 over 0.02 m <= z <= z_max (model scaled to a = 0.060 m | unscaled)")
    for c in (:WMN, :CMP, :CMN)
        p, d = P[c], A[c]
        zhi = zmax(p)
        r1, n = rms_diff(p.z, p.u1 .* (KL_AMPLITUDE / p.a), d.z, d.value, KL_RMS_ZMIN, zhi, KL_U_SCALE)
        r2, _ = rms_diff(p.z, p.u1, d.z, d.value, KL_RMS_ZMIN, zhi, KL_U_SCALE)
        @printf("  %-4s station amplitude %.4f m: %.3f | %.3f\n", c, p.a, r1, r2)
    end
    # figure: mean u (left), first-harmonic amplitude (right); one colour per test in the fixed order of KL_CASES
    colors = Dict(:SP => :gray35, :WMN => :darkorange, :CMP => :royalblue, :CMN => :firebrick)
    names = Dict(:SP => "current only (#SP)", :WMN => "waves only (#WMN)", :CMP => "following (#CMP)", :CMN => "opposing (#CMN)")
    p1 = plot(xlabel="mean u (m/s)", ylabel="z (m)", title="Mean velocity, nz = $(nz)$(KL_TAG)", legend=:topleft, ylims=(0, 0.5))
    for c in KL_CASES
        plot!(p1, P[c].u, P[c].z; color=colors[c], lw=2, label=names[c] * " model")
        scatter!(p1, D[c].value, D[c].z; color=colors[c], ms=3, msw=0, label=names[c] * " data")
    end
    p2 = plot(xlabel="first-harmonic amplitude of u (m/s)", ylabel="z (m)", title="Orbital velocity (scaled to a = 0.060 m)",
        legend=:topleft, ylims=(0, 0.5))
    for c in (:WMN, :CMP, :CMN)
        plot!(p2, P[c].u1 .* (KL_AMPLITUDE / P[c].a), P[c].z; color=colors[c], lw=2, label=names[c] * " model")
        scatter!(p2, A[c].value, A[c].z; color=colors[c], ms=3, msw=0, label=names[c] * " data")
    end
    fig = plot(p1, p2; layout=(1, 2), size=(1200, 600), margin=5Plots.mm)
    savefig(fig, joinpath(figures_dir, "klopman_profiles_nz$(nz)$(KL_TAG).png"))
    println("figure: figures/klopman_profiles_nz$(nz)$(KL_TAG).png")
end
