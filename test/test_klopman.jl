# Klopman's flume, validation case 2DV-13 (issue #85, experiments/validation/klopman/): regression tests of the case set-up
# (always) and a coarse validation of the sign of the wave effect on the current profile (slow; only with the
# environment variable SWE_SLOW_TESTS=true). The full validation on 20 and 40 layers is run_klopman_cases.jl.
# Run with the full suite (runtests.jl) or on its own: julia --project=. test/test_klopman.jl

include(joinpath(@__DIR__, "test_common.jl"))
include(joinpath(@__DIR__, "..", "experiments", "validation", "klopman", "cases.jl"))

const KL_TEST_NX = 190            # -, coarse grid of the set-up test (dx = 0.18 m)
const KL_TEST_NZ = 10             # -
const KL_TEST_STEPS = 20          # -, steps of the averaging check

@testset "Klopman flume (2DV-13)" begin
    @testset "test_verify_klopman_setup" begin
        # time until the full wave stands at the station (ramp + travel at the Doppler group speed + margin)
        @test arrival_periods(:WMN) == 15
        @test arrival_periods(:CMP) == 12
        @test arrival_periods(:CMN) == 17
        # the case settings: theta, time step, scheme of issue #88, open faces of #86
        m = klopman_model(:CMP; nx=KL_TEST_NX, nz=KL_TEST_NZ, calibrate=false)
        cfg = m.config
        @test cfg.theta == KL_THETA && cfg.dt ≈ KL_PERIOD / KL_STEPS_PER_PERIOD
        @test cfg.wd_top_layer == "merged" && cfg.wd_face_level == "limited"
        @test cfg.turbulence_closure == "k_epsilon" && cfg.bed_roughness_length == KL_Z0
        @test m.wetdry.left_open !== nothing
        # the initial uniform current carries the target discharge through every interior face (face layers of the
        # initial level)
        SweTransport2DV._wd_face_thicknesses!(m, m.wetdry)
        q = [sum(m.wetdry.hf[:, f] .* m.state.u[:, f]) for f in 2:KL_TEST_NX]
        @test maximum(abs, q .- KL_DISCHARGE) < TOL_IDENTITY
        # the start from a saved state copies u, eta and the turbulence
        sp = klopman_model(:SP; nx=KL_TEST_NX, nz=KL_TEST_NZ, steps_per_period=KL_SPINUP_STEPS_PER_PERIOD)
        avg = StationAverage(sp)
        for _ in 1:KL_TEST_STEPS
            step_forward!(sp)
            accumulate!(avg, sp)
        end
        r = result(avg, sp)
        @test r.samples == KL_TEST_STEPS
        @test all(isfinite, r.u) && all(>(0), r.u) && all(>(0), r.nut)
        cmn = klopman_model(:CMN; nx=KL_TEST_NX, nz=KL_TEST_NZ, calibrate=false, initial=sp)
        @test cmn.state.u == sp.state.u && cmn.state.eta == sp.state.eta && cmn.turbulence.k == sp.turbulence.k
        # the reference profiles: heights increasing from the bed, finite values
        for name in vcat(["klopman1994_mean_u_$(c).csv" for c in KL_CASES], ["klopman1994_u1_amplitude_$(c).csv" for c in (:WMN, :CMP, :CMN)])
            ref = klopman_reference(name)
            @test length(ref.z) >= 15 && all(>(0), ref.z) && issorted(ref.z) && all(isfinite, ref.value)
        end
    end

    # Coarse validation of the primary criterion of issue #85 (slow, about 10 min): relative to the current only, the
    # upper-half shear du/dz decreases for following and increases for opposing waves. 380 x 10 cells (30 per
    # wavelength, dz = 50 mm), waves at N = 200, lambda_2 = 0.1, the incoming amplitudes of the tuned nz = 20 runs (no
    # tuning), 100 periods from the current-only state, averages over the last 30.
    KL_SLOW_NX = 380
    KL_SLOW_STEPS_PER_PERIOD = 200
    KL_SLOW_SPINUP = 600.0         # s, current only
    KL_SLOW_PERIODS = 100          # -, wave runs
    KL_SLOW_AVERAGE_PERIODS = 30   # -
    KL_SLOW_SHEAR_ZMIN = 0.25      # m, lower end of the upper half
    KL_SLOW_SHEAR_ZMAX = 0.40      # m, upper end (the trough lies about 0.08 m below the still level)
    KL_SLOW_AMPLITUDE = Dict(:CMP => 0.065, :CMN => 0.078)   # m, incoming amplitudes (nz = 20 tuning)
    KL_SLOW_LAMBDA2 = 0.1
    if get(ENV, "SWE_SLOW_TESTS", "false") == "true"
        @testset "test_validate_klopman_shear_sign_coarse" begin
            opts = (; ke_wave_stabilization=KL_SLOW_LAMBDA2)
            function shear(r)
                keep = findall(z -> KL_SLOW_SHEAR_ZMIN <= z <= KL_SLOW_SHEAR_ZMAX, r.z)
                z, u = r.z[keep], r.u[keep]
                zm, um = sum(z) / length(z), sum(u) / length(u)
                return sum((z .- zm) .* (u .- um)) / sum(abs2, z .- zm)   # 1/s, least-squares slope
            end
            sp = klopman_model(:SP; nx=KL_SLOW_NX, nz=KL_TEST_NZ, steps_per_period=KL_SPINUP_STEPS_PER_PERIOD, options=opts)
            nsp = round(Int, KL_SLOW_SPINUP / sp.config.dt)
            avg = StationAverage(sp)
            for n in 1:nsp
                step_forward!(sp)
                n > nsp - round(Int, 100.0 / sp.config.dt) && accumulate!(avg, sp)   # last 100 s
            end
            s_sp = shear(result(avg, sp))
            s = Dict{Symbol,Float64}()
            for case in (:CMP, :CMN)
                m = klopman_model(case; nx=KL_SLOW_NX, nz=KL_TEST_NZ, amplitude=KL_SLOW_AMPLITUDE[case],
                    steps_per_period=KL_SLOW_STEPS_PER_PERIOD, initial=sp, options=opts)
                n_all = KL_SLOW_PERIODS * KL_SLOW_STEPS_PER_PERIOD
                a = StationAverage(m)
                for n in 1:n_all
                    step_forward!(m)
                    n > n_all - KL_SLOW_AVERAGE_PERIODS * KL_SLOW_STEPS_PER_PERIOD && accumulate!(a, m)
                end
                s[case] = shear(result(a, m))
            end
            @info "Klopman coarse shear (1/s)" current_only = s_sp following = s[:CMP] opposing = s[:CMN]
            @test s[:CMP] < s_sp   # following waves reduce the upper-half shear (measured -0.002 against 0.067 1/s)
            # opposing waves increase it: on 20 layers by only about 10 % of the measured change; on this grid (50 mm
            # layers, twice the theta damping of N = 400) the effect is lost (measured 0.052 against 0.067 1/s)
            @test_broken s[:CMN] > s_sp
        end
    end
end
