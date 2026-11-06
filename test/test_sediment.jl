# Suspended cohesive sediment of the wetting-drying step (src/sediment.jl, issue #64): settling, erosion, mass budget,
# density coupling and the equilibrium profile in a channel. Docs section 31.

const SED_C0 = 0.5                 # kg/m^3, initial concentration of the settling column
const SED_WS = 1e-3                # m/s, settling velocity of the settling column
const SED_COLUMN_DEPTH = 2.0       # m
const SED_COLUMN_NZ = 10
const SED_COLUMN_DT = 1.0          # s
const SED_SETTLING_TIME = 400.0    # s, 0.2 H/w_s: the smeared clear-water front is still far from the bed
const SED_SETTLING_RTOL = 1e-4     # -, suspended mass against c0 (H - w_s t) (measured 1.4e-6; implicit upwind smearing)
const SED_LOCK_DC = 2.0            # kg/m^3, sediment excess of the dense half of the turbidity lock exchange
const SED_LOCK_TIME = 20.0         # s
const SED_CHANNEL_DEPTH = 1.0      # m
const SED_CHANNEL_U = 1.0          # m/s, mean velocity
const SED_CHANNEL_Z0 = 1e-3        # m
const SED_CHANNEL_WS = 2e-2        # m/s, Rouse number P about 0.5: a strongly varying profile
const SED_CHANNEL_TIME = 300.0     # s, six settling times H/w_s
const SED_ROUSE_RTOL = 0.1         # -, c(z)/c(H/2) against the Rouse profile in 0.15 H-0.75 H (measured -5.4 to +4.8 %;
                                   # the k-epsilon nu_t is not exactly parabolic)
const SED_ROUSE_HEIGHTS = (0.15, 0.3, 0.6, 0.75)   # -, z/H of the comparison

function sed_column_model()
    cfg = ModelConfig(; length=1.0, depth=SED_COLUMN_DEPTH, nx=2, nz=SED_COLUMN_NZ, dt=SED_COLUMN_DT, use_hydrostatic=true,
        nu=0.0, bed_roughness_length=1e-3, rayleigh_damping=0.0, apply_froude_limiter=false, wetting_drying=true,
        implicit_vertical_terms=true, theta=1.0, write_plots=false)
    model = Model(cfg)
    initialize_water_level!(model, SED_COLUMN_DEPTH; rho=1000.0)
    return model
end

# Uniform channel, 4 m, log profile with k and eps in equilibrium at the inlet (Rodi 1993), relaxed outflow at the depth
function sed_channel_model(; erosion_parameter=0.0, critical_erosion_stress=0.1, inflow_concentration=nothing,
    concentration=0.1)
    nz, nx, L = 20, 8, 4.0
    H = SED_CHANNEL_DEPTH
    dt = 0.6 * (L / nx) / (1.3 * SED_CHANNEL_U)
    cfg = ModelConfig(; length=L, depth=H, nx=nx, nz=nz, dt=dt, use_hydrostatic=true, nu=1e-6, rho0=1000.0,
        turbulence_closure="k_epsilon", bed_roughness_length=SED_CHANNEL_Z0, rayleigh_damping=0.0, apply_froude_limiter=false,
        momentum_advection_scheme="tvd2", transport_advection_scheme="tvd2", wetting_drying=true, implicit_vertical_terms=true,
        theta=1.0, write_plots=false)
    model = Model(cfg)
    initialize_water_level!(model, H; rho=1000.0)
    zc = ((1:nz) .- 0.5) .* (H / nz)
    shape = log.(zc ./ SED_CHANNEL_Z0)
    a = SED_CHANNEL_U / (sum(shape) / nz)            # m/s, u*/kappa
    ustar = 0.41 * a
    u_in = a .* shape
    k_in = ustar^2 .* (1 .- zc ./ H) ./ sqrt(KE_C_MU_VALUE)
    eps_in = ustar^3 .* (1 .- zc ./ H) ./ (0.41 .* zc)
    for k in 1:nz
        model.state.u[k, :] .= u_in[k]
        model.turbulence.k[k, :] .= k_in[k]
        model.turbulence.eps[k, :] .= eps_in[k]
    end
    set_x_inflow_outflow!(model, u_in; left_discharge=SED_CHANNEL_U * H, left_k=k_in, left_eps=eps_in, right_level=0.0,
        right_free=true, right_relaxation_time=20.0)
    sed = attach_sediment!(model; settling_velocity=SED_CHANNEL_WS, critical_erosion_stress=critical_erosion_stress,
        erosion_parameter=erosion_parameter, concentration=concentration, inflow_concentration=inflow_concentration,
        density_coupling=false)
    return model, sed
end

@testset "Suspended sediment (issue #64)" begin
    @testset "test_verify_sediment_settling_column" begin
        # flow at rest, no turbulence, no erosion: uniform c settles; until the clear-water front from the surface reaches
        # the bed the bottom cell keeps c0, so the suspended mass is c0 (H - w_s t) and the deposit w_s c0 t
        model = sed_column_model()
        sed = attach_sediment!(model; settling_velocity=SED_WS, critical_erosion_stress=1e3, erosion_parameter=0.0,
            concentration=SED_C0, density_coupling=false)
        m0 = suspended_sediment_mass(model)
        width = model.config.nx * model.dx
        for _ in 1:round(Int, SED_SETTLING_TIME / SED_COLUMN_DT)
            step_forward!(model)
        end
        m = suspended_sediment_mass(model)
        @test m ≈ SED_C0 * (SED_COLUMN_DEPTH - SED_WS * SED_SETTLING_TIME) * width rtol = SED_SETTLING_RTOL
        @test abs(m + sed.deposited - m0) <= TOL_IDENTITY * m0
        @test all(>=(0), sed.c)
        @test all(sed.bed_change .≈ sed.bed_change[1]) && sed.bed_change[1] > 0
    end

    @testset "test_verify_sediment_erosion_partheniades" begin
        # one step of the channel: the erosion flux is M (tau_b/tau_ce - 1) with the tau_b of the step, and a bed mass
        # smaller than M dt limits it
        M, tce = 1e-4, 0.5
        model, sed = sed_channel_model(; erosion_parameter=M, critical_erosion_stress=tce)
        step_forward!(model)
        @test all(sed.tau_b .> tce)
        @test sed.erosion ≈ M .* (sed.tau_b ./ tce .- 1) rtol = TOL_IDENTITY
        limited, sedl = sed_channel_model(; erosion_parameter=M, critical_erosion_stress=tce)
        sedl.bed_mass .= 0.1 * M * limited.config.dt
        step_forward!(limited)
        @test all(sedl.erosion .≈ 0.1 * M)
        @test all(sedl.bed_mass .+ sedl.bed_change .>= -TOL_IDENTITY * M)
    end

    @testset "test_verify_sediment_mass_budget" begin
        # channel with erosion, clean inflow and outflow: the suspended mass changes by inflow - outflow + erosion -
        # deposition, to round-off
        model, sed = sed_channel_model(; erosion_parameter=2e-4, critical_erosion_stress=0.2, inflow_concentration=0.0)
        m0 = suspended_sediment_mass(model)
        for _ in 1:200
            step_forward!(model)
        end
        m = suspended_sediment_mass(model)
        @test sed.eroded > 0 && sed.outflow > 0
        @test abs(m - (m0 + sed.inflow - sed.outflow + sed.eroded - sed.deposited)) <= TOL_MODEL_IDENTITY * max(m0, m)
        @test sum(sed.bed_change) * model.dx ≈ sed.deposited - sed.eroded rtol = TOL_IDENTITY
        @test all(>=(0), sed.c)
    end

    @testset "test_verify_sediment_clean_water_enters_at_the_open_boundary" begin
        # operator level: with the flow reversed at the right boundary face, the entering water carries
        # outside_concentration (0) instead of the concentration of the last cell; only the last cell differs, by
        # dt/dx q (c_out - c_nx)/h
        model, sed = sed_channel_model(; concentration=0.3)
        step_forward!(model)
        wd = model.wetdry
        nx = model.config.nx
        wd.q[:, nx + 1] .= -0.2 .* wd.hf[:, nx + 1]
        c_before = copy(sed.c)
        with, without = copy(c_before), copy(c_before)
        _, out_with = SweTransport2DV._wd_transport_scalar!(model, wd, with, nothing, 0.0, 1.0; rho_right=0.0)
        _, out_without = SweTransport2DV._wd_transport_scalar!(model, wd, without, nothing, 0.0, 1.0)
        dt, dx = model.config.dt, model.dx
        for k in 1:model.config.nz
            expected = dt / dx * (-wd.q[k, nx + 1]) * (0.0 - c_before[k, nx]) / wd.h_new[k, nx]
            @test isapprox(with[k, nx] - without[k, nx], expected; atol=TOL_IDENTITY)
        end
        @test with[:, 1:(nx - 1)] == without[:, 1:(nx - 1)]
        @test out_with == 0.0 && out_without < 0.0
    end

    @testset "test_verify_sediment_density_coupling_equals_salt" begin
        # lock exchange driven by sediment (no settling, no erosion, coupled) against the same driven by the equivalent
        # salt density rho + c (1 - rho/rho_s): the flows are the same (the transport is linear in the transported field)
        function lock(with_sediment)
            cfg = ModelConfig(; length=4.0, depth=0.5, nx=40, nz=10, dt=0.01, use_hydrostatic=false, nu=1e-6,
                bed_roughness_length=1e-4, rayleigh_damping=0.0, apply_froude_limiter=false, wetting_drying=true,
                implicit_vertical_terms=true, theta=1.0, write_plots=false)
            model = Model(cfg)
            xc = ((1:cfg.nx) .- 0.5) .* model.dx
            c = [x < 2.0 ? SED_LOCK_DC : 0.0 for _ in 1:cfg.nz, x in xc]
            rho_s = 2650.0
            if with_sediment
                initialize_water_level!(model, 0.5; rho=1000.0)
                attach_sediment!(model; settling_velocity=0.0, critical_erosion_stress=1e3, erosion_parameter=0.0,
                    concentration=c, sediment_density=rho_s)
            else
                initialize_water_level!(model, 0.5; rho=1000.0 .+ c .* (1 - 1000.0 / rho_s))
            end
            for _ in 1:round(Int, SED_LOCK_TIME / cfg.dt)
                step_forward!(model)
            end
            return model
        end
        a, b = lock(true), lock(false)
        @test maximum(abs.(a.state.u)) > 0.01
        @test maximum(abs.(a.state.u .- b.state.u)) <= TOL_MODEL_IDENTITY * maximum(abs.(b.state.u))
        @test maximum(abs.(a.wetdry.sediment.rho_mix .- b.state.rho)) <= TOL_MODEL_IDENTITY * 1000.0
    end

    @testset "test_validate_sediment_rouse_profile" begin
        # equilibrium of settling and turbulent diffusion in the channel (zero-gradient inflow concentration): against the
        # Rouse profile c/c_r = [(H - z)/z * z_r/(H - z_r)]^P, P = w_s sigma_c / (kappa u*), for the parabolic
        # nu_t = kappa u* z (1 - z/H) of the log-law channel (Rouse 1937); erosion balances deposition at the bed
        model, sed = sed_channel_model(; erosion_parameter=1e-4, critical_erosion_stress=0.1)
        for _ in 1:round(Int, SED_CHANNEL_TIME / model.config.dt)
            step_forward!(model)
        end
        i = model.config.nx ÷ 2
        H = SED_CHANNEL_DEPTH
        h = model.wetdry.h[:, i]
        z = cumsum(h) .- 0.5 .* h
        ustar = sqrt(sed.tau_b[i] / model.config.rho0)
        P = SED_CHANNEL_WS * SweTransport2DV.SED_SCHMIDT / (0.41 * ustar)
        cz(zz) = (j = searchsortedlast(z, zz); t = (zz - z[j]) / (z[j + 1] - z[j]); (1 - t) * sed.c[j, i] + t * sed.c[j + 1, i])
        zr = 0.5H
        for f in SED_ROUSE_HEIGHTS
            rouse = (((H - f * H) / (f * H)) * (zr / (H - zr)))^P
            @test cz(f * H) / cz(zr) ≈ rouse rtol = SED_ROUSE_RTOL
        end
        @test abs(sum(sed.erosion) - sum(sed.deposition)) <= 0.05 * sum(sed.erosion)   # near equilibrium at the bed
    end

    @testset "test_verify_sediment_argument_checks" begin
        model = sed_column_model()
        kw = (; settling_velocity=1e-3, critical_erosion_stress=0.1, erosion_parameter=1e-4)
        @test_throws ArgumentError attach_sediment!(model; kw..., settling_velocity=-1.0)
        @test_throws ArgumentError attach_sediment!(model; kw..., critical_erosion_stress=0.0)
        @test_throws ArgumentError attach_sediment!(model; kw..., concentration=-1.0)
        @test_throws ArgumentError attach_sediment!(model; kw..., inflow_concentration=zeros(3))
        @test_throws ArgumentError attach_sediment!(model; kw..., sediment_density=900.0)
        cfg = ModelConfig(; length=1.0, depth=1.0, nx=2, nz=4, dt=0.1, write_plots=false)
        @test_throws ArgumentError attach_sediment!(Model(cfg); kw...)
        smooth = ModelConfig(; length=1.0, depth=1.0, nx=2, nz=4, dt=0.1, wetting_drying=true, implicit_vertical_terms=true,
            write_plots=false)
        @test_throws ArgumentError attach_sediment!(Model(smooth); kw...)
        # without erosion the smooth bed is allowed (passive tracer, issue #11)
        @test attach_sediment!(Model(smooth); kw..., erosion_parameter=0.0) isa SedimentState
    end
end
