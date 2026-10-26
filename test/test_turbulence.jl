# k-epsilon turbulence closure, bed wall functions and turbulent density diffusion (src/turbulence.jl).
# Run with the full suite (runtests.jl) or on its own: julia --project=. test/test_turbulence.jl

include(joinpath(@__DIR__, "test_common.jl"))
# Dunn et al. (1996) run data and profiles, the Lopez & Garcia (1997) reference profiles and the canopy channel helper.
include(joinpath(@__DIR__, "..", "experiments", "vegetation", "cases.jl"))

const TURB_RHO0 = 1000.0        # kg/m^3

function turbulence_channel_model(; use_hydrostatic=true, depth, nz, dt, t_end, slope, z0, kwargs...)
    cfg = ModelConfig(; length=1.0, depth=depth, nx=4, nz=nz, dt=dt, nt=round(Int, t_end / dt), use_hydrostatic=use_hydrostatic,
        periodic_x=true, variable_top_layer=false, pressure_solver_method="sparse", nu=1e-6, body_force_x=9.81 * slope,
        turbulence_closure="k_epsilon", bed_roughness_length=z0, rho0=TURB_RHO0, rho_left=TURB_RHO0, rho_right=TURB_RHO0,
        rayleigh_damping=0.0, apply_froude_limiter=false, momentum_advection_scheme="tvd2", write_plots=false, write_animation=false,
        write_pressure_buoyancy_diagnostics=false, kwargs...)
    model = Model(cfg)
    fill!(model.state.rho, TURB_RHO0)
    refresh_density_metadata!(model)
    return model
end

@testset "Turbulence" begin
    @testset "test_verify_k_epsilon_decaying_homogeneous_turbulence" begin
        # No shear, no gradients: dk/dt = -eps, deps/dt = -C2 eps²/k, exact solution
        # k(t) = k0 (1 + (C2 - 1) eps0 t / k0)^(-1/(C2 - 1)). The implicit (Patankar) sinks are first order in dt.
        # Observed errors 5.8e-2, 3.1e-2, 1.6e-2, 7.9e-3 for dt = 1, 0.5, 0.25, 0.125 s.
        K0, EPS0, T_END = 1e-3, 1e-4, 50.0          # m^2/s^2, m^2/s^3, s
        TIMESTEPS = (1.0, 0.5, 0.25, 0.125)         # s
        ORDER_MIN = 0.9
        k_exact = K0 * (1 + (KE_C2_VALUE - 1) * EPS0 * T_END / K0)^(-1 / (KE_C2_VALUE - 1))
        errors = Float64[]
        for dt in TIMESTEPS
            ke = KEpsilonState(4, 3)
            fill!(ke.k, K0); fill!(ke.eps, EPS0)
            u = zeros(4, 4); w = zeros(5, 3); rho = fill(TURB_RHO0, 4, 3)
            for _ in 1:round(Int, T_END / dt)
                fill!(ke.nu_t, 0.0)     # isolates the source/sink terms (no production, no diffusion)
                k_epsilon_step!(ke, u, w, rho, dt, 1.0, 1.0, 0.0, 9.81, TURB_RHO0)
            end
            push!(errors, abs(ke.k[2, 2] / k_exact - 1))
        end
        @test all(log2.(errors[1:end-1] ./ errors[2:end]) .> ORDER_MIN)
    end

    @testset "test_verify_implicit_vertical_diffusion_cosine_mode" begin
        # d(phi)/dt = K d²phi/dz² with zero flux at both ends: phi = cos(pi z/H) decays as exp(-K (pi/H)² t). Backward Euler:
        # first order in dt (observed order 1.0); the column integral is conserved to round-off.
        H, K, T_END, NZ = 1.0, 1e-2, 5.0, 200        # m, m^2/s, s, -
        TIMESTEPS = (0.5, 0.25, 0.125)
        ORDER_MIN = 0.9
        dz = H / NZ
        z = [(k - 0.5) * dz for k in 1:NZ]
        exact = exp(-K * (pi / H)^2 * T_END) .* cos.(pi .* z ./ H)
        errors = Float64[]
        ws = KEpsilonState(NZ, 1)
        for dt in TIMESTEPS
            phi = reshape(cos.(pi .* z ./ H) .+ 2.0, NZ, 1)
            total0 = sum(phi) * dz
            for _ in 1:round(Int, T_END / dt)
                implicit_vertical_diffusion!(phi, (k, i) -> K, (k, i) -> dz, dt, dz, ws)
            end
            @test abs(sum(phi) * dz - total0) < TOL_IDENTITY
            push!(errors, maximum(abs, vec(phi) .- 2.0 .- exact))
        end
        @test all(log2.(errors[1:end-1] ./ errors[2:end]) .> ORDER_MIN)
    end

    @testset "test_validate_k_epsilon_open_channel" begin
        # Uniform open-channel flow over a rough bed (z0 = 1 mm, H = 0.5 m, S = 1e-3), periodic channel: in the steady state
        # the bed shear equals g S H (u*²) and the total stress (nu + nu_t) du/dz = g S (H - z) (momentum balance, exact);
        # the wall function sets k = u*²/sqrt(C_mu) in the lowest cell; the depth-mean velocity follows the log law
        # U/u* = (ln(H/z0) - 1)/kappa within a few per cent (observed +6.0 %, consistent with the wake of real channel
        # flows). With the free-surface dissipation condition (Rodi 1993, default) nu_t peaks at mid-depth (observed
        # z/H = 0.53, 82 % of the parabolic kappa u* H/4); with zero flux it would peak at the surface. Both modes.
        H, SLOPE, Z0 = 0.5, 1e-3, 1e-3               # m, -, m
        BALANCE_RTOL = 1e-4             # -, observed 1.2e-5 after 600 s (still approaching the steady state)
        LOG_LAW_RTOL = 0.08
        NU_T_PEAK_RANGE = (0.35, 0.7)    # z/H of the nu_t maximum
        ustar = sqrt(9.81 * SLOPE * H)
        for use_hydrostatic in (true, false)
            model = turbulence_channel_model(; use_hydrostatic=use_hydrostatic, depth=H, nz=20, dt=0.05, t_end=600.0, slope=SLOPE, z0=Z0)
            for _ in 1:model.config.nt
                step_forward!(model)
            end
            u = vec(mean(model.state.u[:, 1:end-1]; dims=2))
            nut = vec(mean(model.turbulence.nu_t; dims=2))
            cb = bed_friction_coefficient(Z0, model.dz / 2)
            @test isapprox(cb * u[1]^2, ustar^2; rtol=BALANCE_RTOL)
            k_mid = 10
            tau = (1e-6 + 0.5 * (nut[k_mid] + nut[k_mid + 1])) * (u[k_mid + 1] - u[k_mid]) / model.dz
            @test isapprox(tau, 9.81 * SLOPE * (H - k_mid * model.dz); rtol=BALANCE_RTOL)
            @test isapprox(model.turbulence.k[1, 1], ustar^2 / sqrt(KE_C_MU_VALUE); rtol=BALANCE_RTOL)
            @test isapprox(mean(u), ustar / 0.41 * (log(H / Z0) - 1); rtol=LOG_LAW_RTOL)
            z_peak = (argmax(nut) - 0.5) * model.dz / H
            @test NU_T_PEAK_RANGE[1] < z_peak < NU_T_PEAK_RANGE[2]
        end
    end

    @testset "test_verify_k_epsilon_vegetation_balance" begin
        # Homogeneous emergent rigid canopy (periodic channel, no shear): the k equation drives eps -> C_fk P_v and the eps
        # equation eps -> (C1 C_fe / C2) P_v; with C_fe = (C2/C1) C_fk (Lopez & Garcia 1997, p. 24) both agree. With the
        # published C_fe = 1.33 the ratio is 1.33 * 1.44 / 1.92 = 0.9975 (observed 0.9975). Before the factor C1 was applied
        # to the vegetation term, the ratio was 0.693 and k grew without bound. The surface condition for eps disturbs the
        # top two rows (0.995, 0.97 after 200 s), so the check uses the rows below.
        RATIO_TOL = 1e-3
        INTERIOR_ROWS = 1:8
        cfg = ModelConfig(length=1.0, depth=0.5, nx=4, nz=10, dt=0.01, nt=1, use_hydrostatic=true, periodic_x=true, variable_top_layer=false,
            pressure_solver_method="sparse", nu=1e-6, body_force_x=1e-3, turbulence_closure="k_epsilon", rho0=TURB_RHO0, rho_left=TURB_RHO0,
            rho_right=TURB_RHO0, rayleigh_damping=0.0, apply_froude_limiter=false, write_plots=false, write_animation=false,
            write_pressure_buoyancy_diagnostics=false)
        model = Model(cfg)
        fill!(model.state.rho, TURB_RHO0); refresh_density_metadata!(model)
        attach_vegetation!(model, VegetationModel(; patches=[VegetationPatch(x_start=0.0, x_end=1.0, stem_height=1.0, stem_diameter=0.01, stem_density=400.0)]))
        for _ in 1:20000
            step_forward!(model)
        end
        ke = model.turbulence
        @test isapprox(mean(ke.eps[INTERIOR_ROWS, :]) / mean(ke.veg_work[INTERIOR_ROWS, :]), ke.cfe * KE_C1_VALUE / KE_C2_VALUE; atol=RATIO_TOL)
    end

    @testset "test_validate_k_epsilon_canopies_against_lopez_garcia" begin
        # Submerged canopies of Dunn et al. (1996), runs 1, 9 (rigid) and 13, 17 (flexible, as rigid canopies of the deflected
        # height), measured C_DB, smooth bed (dunn1996_k_epsilon in experiments/vegetation/cases.jl). Reference: the computed
        # k-epsilon profiles of Lopez & Garcia (1997), same model equations and constants (Figs. 6.1, 6.2, 6.4, 6.5,
        # digitized, LG1997_PROFILES). Observed (nz = 40): velocity 0.92-1.03 of theirs at all heights (about 5 % lower
        # above the canopy); stress peak at the canopy top 0.97-1.05 of theirs (their peaks are partly hidden by the canopy
        # line); stress below 0.8 h_p 0.68-1.35 of theirs (ours lower near the bed, higher in the middle of the canopy).
        # Both models overpredict the measured mean velocity Q/(bH), ours by 1.05-1.20 for these runs; Lopez & Garcia
        # attribute this to secondary currents in the flume, which reduce the measured stress above the canopy.
        VELOCITY_RTOL = 0.10
        PEAK_RTOL = 0.08
        CANOPY_STRESS_RTOL = 0.40
        OVERPREDICTION_RANGE = (1.0, 1.3)
        for run in (1, 9, 13, 17)
            r = dunn1996_k_epsilon(run)
            g = LG1997_PROFILES[run]
            ours_u = linear_interpolation.(g.uz, Ref(r.z), Ref(r.u))
            @test all(abs.(ours_u ./ g.u .- 1) .< VELOCITY_RTOL)
            @test isapprox(maximum(r.tau), maximum(g.tau); rtol=PEAK_RTOL)
            low = g.tz .< 0.8 * r.run.hp
            ours_tau = linear_interpolation.(g.tz[low], Ref(r.zf), Ref(r.tau))
            @test all(abs.(ours_tau ./ g.tau[low] .- 1) .< CANOPY_STRESS_RTOL)
            @test OVERPREDICTION_RANGE[1] < mean(r.u) / r.run.U < OVERPREDICTION_RANGE[2]
        end
    end

    @testset "test_validate_k_epsilon_rigid_canopies_against_dunn_profiles" begin
        # Laboratory data: the 12 rigid runs of Dunn et al. (1996), horizontally averaged velocity profiles (Appendix C,
        # DUNN1996_PROFILES). The measured velocities are lower than any 2D model gives (depth mean: observed 1.09-1.29 times
        # the measured Q/(bH)), and the measured stress at the canopy top is only 0.12-0.83 of the 2D momentum balance
        # g S (H - h_p) that the model satisfies (secondary currents, Lopez & Garcia 1997, p. 58). The shape inside the
        # canopy is compared instead: u(z)/u(h_p) of the model against the measured u(z)/u(h_p) (h_p: measured point
        # nearest the canopy top). Observed (nz = 20, which changes the profiles by at most 3 % against nz = 40): 79 points,
        # median deviation 4.8 %, largest 21 % (sparse runs 6, 7, 12, where the model's velocity drops more in the canopy).
        SHAPE_MAX = 0.25
        SHAPE_MEDIAN = 0.07
        OVERPREDICTION_RANGE = (1.05, 1.35)
        deviations = Float64[]
        for run in 1:12
            r = dunn1996_k_epsilon(run; nz=20)
            p = DUNN1996_PROFILES[run]
            _, itop = findmin(abs.(p.z .- r.run.hp))
            ours = linear_interpolation.(p.z, Ref(r.z), Ref(r.u))
            inside = findall((p.z .< r.run.hp) .& (eachindex(p.z) .!= itop))
            append!(deviations, abs.((ours[inside] ./ ours[itop]) ./ (p.u[inside] ./ p.u[itop]) .- 1))
            @test OVERPREDICTION_RANGE[1] < mean(r.u) / r.run.U < OVERPREDICTION_RANGE[2]
        end
        @test maximum(deviations) < SHAPE_MAX
        @test median(deviations) < SHAPE_MEDIAN
    end

    @testset "test_validate_k_epsilon_plant_density_series_against_lopez_garcia" begin
        # Numerical experiment of Lopez & Garcia (1997), Figs. 6.16 and 6.18: S = 0.0036, h_p = 0.10 m, D = 6.4 mm,
        # q = 1.12 m²/s, smooth bed (their wall functions, E = 9), stem densities 1.6-160 stems/m² (cases 3, 5, 7, 9 of
        # lg1997_manning_case), run at their computed depth H. Reference values are theirs (digitized, LG1997_* in
        # experiments/vegetation/cases.jl).
        # - Momentum transfer to the bed u*_bed/sqrt(g H S) (Fig. 6.16): observed 0.98-1.00 of theirs (dz = 0.02 m; with
        #   dz = 0.01 m 1.00-1.08, the wall function depends on the first cell).
        # - Manning n = H^(5/3) S^(1/2) / q at their depth (Fig. 6.18): observed 0.95, 1.04, 1.14, 1.25 of theirs; the
        #   difference grows with density (1.53 at 1581 stems/m²): over dense canopies in deep water our profile above the
        #   canopy is rougher (effective z0 about h_p/2). At low density our n follows the smooth-bed log law (0.0092 at
        #   their depth), theirs is 0.011. Not resolved; the bound documents the present agreement.
        CASES = (3, 5, 7, 9)
        DZ = 0.02                   # m
        BED_SHEAR_RTOL = 0.08
        MANNING_RTOL = 0.30
        n_values = Float64[]
        for i in CASES
            r = lg1997_manning_case(i; dz=DZ)
            @test isapprox(r.bed_shear, LG1997_BED_SHEAR[i]; rtol=BED_SHEAR_RTOL)
            @test isapprox(r.n, LG1997_MANNING_N[i]; rtol=MANNING_RTOL)
            push!(n_values, r.n)
        end
        @test issorted(n_values)
    end

    @testset "test_verify_turbulence_closure_selection" begin
        # The default and the legacy flag keep their meaning; k_epsilon creates the state (with the fixed top layer as the
        # automatic default); invalid choices (unknown closure, bed geometry, explicit variable top layer, nz = 1) and a
        # change of the closure across the k-epsilon boundary after construction are rejected.
        @test turbulence_closure(ModelConfig()) == "none"
        @test turbulence_closure(ModelConfig(use_turbulence_model=true)) == "mixing_length"
        @test Model(ModelConfig(turbulence_closure="k_epsilon", write_plots=false)).turbulence isa KEpsilonState
        @test !use_variable_top_layer(ModelConfig(turbulence_closure="k_epsilon"))
        @test use_variable_top_layer(ModelConfig())
        @test Model(ModelConfig()).turbulence === nothing
        @test_throws ArgumentError KEpsilonState(1, 3)
        @test_throws ArgumentError Model(ModelConfig(turbulence_closure="k_omega"))
        @test_throws ArgumentError Model(ModelConfig(turbulence_closure="k_epsilon", variable_top_layer=false, bed_level=zeros(100)))
        @test_throws ArgumentError Model(ModelConfig(turbulence_closure="k_epsilon", variable_top_layer=true))

        model_ke = Model(ModelConfig(turbulence_closure="k_epsilon", variable_top_layer=false, write_plots=false))
        model_ke.config.turbulence_closure = "none"
        @test_throws ArgumentError step_forward!(model_ke)
        model_none = Model(ModelConfig(variable_top_layer=false, write_plots=false))
        model_none.config.turbulence_closure = "k_epsilon"
        @test_throws ArgumentError step_forward!(model_none)
    end

    @testset "test_verify_clear_vegetation_resets_k_epsilon_production" begin
        # After clear_vegetation! the vegetation production P_v of the k-epsilon closure is zero (it is otherwise only
        # written while vegetation is attached).
        cfg = ModelConfig(length=1.0, depth=1.0, nx=4, nz=4, dt=1e-4, nt=1, periodic_x=true, variable_top_layer=false,
            pressure_solver_method="sparse", turbulence_closure="k_epsilon", write_plots=false, write_animation=false,
            write_pressure_buoyancy_diagnostics=false)
        model = Model(cfg)
        fill!(model.state.rho, TURB_RHO0)
        refresh_density_metadata!(model)
        fill!(model.state.u, 0.1)
        attach_vegetation!(model, VegetationModel(; patches=[VegetationPatch(x_start=0.0, x_end=cfg.length,
            stem_height=0.8, stem_diameter=0.01, stem_density=100.0)]))
        step_forward!(model)
        @test maximum(model.turbulence.veg_work) > 0.0
        clear_vegetation!(model)
        step_forward!(model)
        @test all(iszero, model.turbulence.veg_work)
    end
end
