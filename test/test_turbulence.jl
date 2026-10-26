# k-epsilon turbulence closure, bed wall functions and turbulent density diffusion (src/turbulence.jl).
# Run with the full suite (runtests.jl) or on its own: julia --project=. test/test_turbulence.jl

include(joinpath(@__DIR__, "test_common.jl"))
# Dunn et al. (1996) run data and the Lopez & Garcia (1997) reference profile.
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
        TIMESTEPS = (0.4, 0.2, 0.1)
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

    @testset "test_validate_k_epsilon_dunn_run1_against_lopez_garcia" begin
        # Submerged rigid canopy, run 1 of Dunn et al. (1996) (H = 0.335 m, h_p = 0.117 m, a = 1.09 1/m, S = 0.0036), k-epsilon
        # with the vegetation terms of Lopez & Garcia (1997), measured C_DB = 1.01, smooth bed (z0 = 5e-6 m). Reference: their
        # computed k-epsilon profile (Fig. 6.1, read from the figure, LG1997_RUN1 in experiments/vegetation/cases.jl). From the
        # canopy top upwards our profile lies within 12 % of theirs (observed 0.90-1.03). Both 2D models overpredict the
        # measured depth-mean velocity Q/(bH) (ours 1.20 times; theirs about 1.3 times from their figure), which Lopez &
        # Garcia attribute to secondary currents in the flume.
        PROFILE_RTOL = 0.12
        OVERPREDICTION_RANGE = (1.1, 1.3)
        d = dunn1996_run(1)
        nz = 40
        dz = d.H / nz
        dt = min(0.01, 0.2 * dz^2 / 5e-3)
        cfg = vegetation_case_config(length=1.0, depth=d.H, nx=4, nz=nz, dt=dt, nt=round(Int, 400 / dt), use_hydrostatic=true, periodic_x=true,
            variable_top_layer=false, pressure_solver_method="sparse", nu=1e-6, body_force_x=9.81 * d.S, turbulence_closure="k_epsilon",
            bed_roughness_length=5e-6)
        model = uniform_density_model(cfg)
        attach_vegetation!(model, VegetationModel(; patches=[VegetationPatch(x_start=0.0, x_end=1.0, stem_height=d.hp, stem_diameter=DUNN1996_D,
            stem_density=d.a / DUNN1996_D, drag_coefficient=d.cdb)]))
        for _ in 1:cfg.nt
            step_forward!(model)
        end
        u = vec(mean(model.state.u[:, 1:end-1]; dims=2))
        z = [(k - 0.5) * dz for k in 1:nz]
        above = LG1997_RUN1_Z .>= 0.10
        ours = linear_interpolation.(LG1997_RUN1_Z[above], Ref(z), Ref(u))
        @test all(abs.(ours ./ LG1997_RUN1_U[above] .- 1) .< PROFILE_RTOL)
        @test OVERPREDICTION_RANGE[1] < mean(u) / d.U < OVERPREDICTION_RANGE[2]
    end

    @testset "test_verify_turbulence_closure_selection" begin
        # The default and the legacy flag keep their meaning; k_epsilon creates the state; invalid choices are rejected.
        @test turbulence_closure(ModelConfig()) == "none"
        @test turbulence_closure(ModelConfig(use_turbulence_model=true)) == "mixing_length"
        @test Model(ModelConfig(turbulence_closure="k_epsilon", write_plots=false)).turbulence isa KEpsilonState
        @test Model(ModelConfig()).turbulence === nothing
        @test_throws ArgumentError Model(ModelConfig(turbulence_closure="k_omega"))
        @test_throws ArgumentError Model(ModelConfig(turbulence_closure="k_epsilon", bed_level=zeros(100)))
    end
end
