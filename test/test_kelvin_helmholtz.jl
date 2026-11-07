# Kelvin-Helmholtz instability of a tanh shear layer: linear growth rate against the Taylor-Goldstein mode (issue #22).
# Run with the full suite (runtests.jl) or on its own: julia --project=. test/test_kelvin_helmholtz.jl

include(joinpath(@__DIR__, "test_common.jl"))

# ---- Case -----------------------------------------------------------------------------------------------------------
const KH_DELTA = 0.1                # m, shear-layer half thickness delta
const KH_U = 0.05                   # m/s, velocity U (difference 2U across the layer)
const KH_KDELTA = 0.4446            # -, k delta of the most unstable mode (Michalke 1964)
const KH_DEPTH = 0.8                # m, 8 delta (the reference has the same walls)
const KH_J = 0.05                   # -, bulk Richardson number g drho delta / (2 rho0 U^2) of the stratified case
const KH_RHO0 = 1000.0              # kg/m^3, ModelConfig default
const KH_G = 9.81                   # m/s^2, ModelConfig default
const KH_SEED = 1e-3                # m, initial interface displacement (k a = 0.004: linear)
const KH_COURANT = 0.2              # -, U dt / dx on every grid
const KH_FIT_WINDOW = (5.0, 35.0)   # s, linear phase used for the rate (the mode grows by e^2.2 in it; saturation after 40 s)
const KH_CELLS = (8, 16)            # cells per delta (dx = dz), the grids of the test

# ---- Bounds (measured values in the comments; docs/TESTS_TECHNICAL_DOCUMENTATION.md, section 35) ---------------------
const KH_MICHALKE_RATE = 0.1897             # U/delta, largest rate, unbounded and unstratified (Michalke 1964)
const KH_MICHALKE_TOL = 5e-4                # U/delta (computed 0.18970 with walls at 20 delta on each side)
const KH_DEEP_DEPTH = 40.0                  # delta, depth of the "unbounded" reference
const KH_STEP_CONVERGENCE_RTOL = 1e-8       # -, rate with 2x the shooting steps
const KH_NEAR_NEUTRAL_J = 0.249             # -, J just below 1/4 (Miles-Howard bound) at k delta = 1/2
const KH_NEAR_NEUTRAL_MAX = 0.03            # -, rate there relative to J = 0 (computed 0.016)
const KH_RATE_FINE_MAX = 0.008              # -, relative rate error on the finest grid (observed -0.52 % stratified, -0.46 % at J = 0)
const KH_RATE_COARSE_MAX = 0.015            # -, same, coarsest grid (observed -1.02 %)
const KH_CONVERGENCE_RATIO = 0.7            # -, the error must drop at least by this factor per refinement (observed 0.51)
const KH_DIVERGENCE_TOL = 1e-14             # 1/s, discrete divergence of the seeded velocity (round-off)
const KH_MASS_RTOL = 1e-12                  # -, relative salt mass change
const KH_VOLUME_TOL = 1e-12                 # m^2, water volume change

kh_k() = KH_KDELTA / KH_DELTA
kh_length() = 2pi / kh_k()
kh_drho(J) = J * 2 * KH_RHO0 * KH_U^2 / (KH_G * KH_DELTA)   # kg/m^3

function kh_model(cells_per_delta, J)
    dz = KH_DELTA / cells_per_delta
    nx, nz = round(Int, kh_length() / dz), round(Int, KH_DEPTH / dz)
    cfg = ModelConfig(length=kh_length(), depth=KH_DEPTH, nx=nx, nz=nz, dt=KH_COURANT * (kh_length() / nx) / KH_U,
        use_hydrostatic=false, periodic_x=true, momentum_advection_scheme="tvd2", transport_advection_scheme="tvd2",
        rayleigh_damping=0.0, apply_froude_limiter=false, nu=0.0, rho_left=KH_RHO0, rho_right=KH_RHO0,
        write_plots=false, write_animation=false, write_pressure_buoyancy_diagnostics=false)
    model = Model(cfg)
    sigma, c = initialize_shear_layer_mode!(model; velocity=KH_U, thickness=KH_DELTA, density_difference=kh_drho(J),
        amplitude=KH_SEED)
    return model, sigma, c
end

# Amplitude (m/s) of the e^{i k x} component of w at the interface z0 (the w-interface at mid-depth).
function kh_mode_amplitude(model)
    nx, nz = model.config.nx, model.config.nz
    return 2 / nx * abs(sum(model.state.w[nz ÷ 2 + 1, i] * cis(-kh_k() * (i - 0.5) * model.dx) for i in 1:nx))
end

# Growth rate (1/s): least-squares slope of log(amplitude) over KH_FIT_WINDOW.
function kh_growth_rate(cells_per_delta, J)
    model, sigma, _ = kh_model(cells_per_delta, J)
    dt = model.config.dt
    mass0 = SweTransport2DV._constituent_mass(model)
    volume0 = water_volume(model.config.depth, model.state.eta, model.dx)
    ts, logs = Float64[], Float64[]
    for n in 1:round(Int, KH_FIT_WINDOW[2] / dt)
        step_forward!(model)
        t = n * dt
        t >= KH_FIT_WINDOW[1] && (push!(ts, t); push!(logs, log(kh_mode_amplitude(model))))
    end
    tm, lm = mean(ts), mean(logs)
    rate = sum((ts .- tm) .* (logs .- lm)) / sum((ts .- tm) .^ 2)
    return (; rate, sigma, dmass=abs(SweTransport2DV._constituent_mass(model) - mass0) / mass0,
        dvolume=abs(water_volume(model.config.depth, model.state.eta, model.dx) - volume0), model)
end

@testset "Kelvin-Helmholtz (issue #22)" begin
    @testset "test_verify_kelvin_helmholtz_tanh_mode" begin
        U, d = 1.0, 1.0
        # Michalke (1964): largest rate 0.1897 U/delta at k delta = 0.4446, unbounded and unstratified
        sigma, c, _, _ = kelvin_helmholtz_tanh_mode(; k=KH_KDELTA, velocity=U, thickness=d, depth=KH_DEEP_DEPTH)
        @test sigma ≈ KH_MICHALKE_RATE atol = KH_MICHALKE_TOL
        @test abs(real(c)) <= 1e-12   # antisymmetric profile: a standing (non-propagating) mode
        @test all(kelvin_helmholtz_tanh_mode(; k=kd, velocity=U, thickness=d, depth=KH_DEEP_DEPTH)[1] < sigma
                  for kd in (KH_KDELTA - 0.05, KH_KDELTA + 0.05))
        @test kelvin_helmholtz_tanh_mode(; k=KH_KDELTA, velocity=U, thickness=d, depth=KH_DEEP_DEPTH, steps=8000)[1] ≈ sigma rtol = KH_STEP_CONVERGENCE_RTOL
        # dimensional scaling: sigma ~ U/delta
        @test kelvin_helmholtz_tanh_mode(; k=KH_KDELTA / 0.1, velocity=0.05, thickness=0.1, depth=KH_DEEP_DEPTH * 0.1)[1] ≈ sigma * 0.05 / 0.1 rtol = 1e-10
        # stratification lowers the rate; it vanishes towards J = 1/4 (Miles 1961, Howard 1961)
        rates = [kelvin_helmholtz_tanh_mode(; k=KH_KDELTA, velocity=U, thickness=d, richardson=J, depth=8.0)[1] for J in (0.0, 0.05, 0.1, 0.2)]
        @test issorted(rates; rev=true)
        near = kelvin_helmholtz_tanh_mode(; k=0.5, velocity=U, thickness=d, richardson=KH_NEAR_NEUTRAL_J, depth=KH_DEEP_DEPTH)[1]
        @test 0 < near <= KH_NEAR_NEUTRAL_MAX * kelvin_helmholtz_tanh_mode(; k=0.5, velocity=U, thickness=d, depth=KH_DEEP_DEPTH)[1]
        @test near < kelvin_helmholtz_tanh_mode(; k=0.5, velocity=U, thickness=d, richardson=0.24, depth=KH_DEEP_DEPTH)[1]
        # the eigenfunction meets the walls and is normalised
        _, _, z, phi = kelvin_helmholtz_tanh_mode(; k=KH_KDELTA, velocity=U, thickness=d, richardson=0.05, depth=8.0)
        @test abs(phi[1]) == 0 && abs(phi[end]) <= 1e-8
        @test maximum(abs, phi) ≈ 1
        # beyond the neutral wavenumber (k delta about 0.96 at J = 0.05) there is no unstable mode
        @test_throws DomainError kelvin_helmholtz_tanh_mode(; k=1.1, velocity=U, thickness=d, richardson=0.05, depth=8.0)
        @test_throws ArgumentError kelvin_helmholtz_tanh_mode(; k=1.0, velocity=U, thickness=d, richardson=0.25, depth=8.0)
        @test_throws ArgumentError kelvin_helmholtz_tanh_mode(; k=1.0, velocity=U, thickness=d, depth=8.0, z0=9.0)
    end

    @testset "test_verify_shear_layer_mode_initial_state" begin
        model, sigma, _ = kh_model(8, KH_J)
        st, dx, dz = model.state, model.dx, model.dz
        nx, nz = model.config.nx, model.config.nz
        # discrete continuity of every cell, and the periodic seam face
        div = maximum(abs((st.u[k, i + 1] - st.u[k, i]) / dx + (st.w[k + 1, i] - st.w[k, i]) / dz) for i in 1:nx, k in 1:nz)
        @test div <= KH_DIVERGENCE_TOL
        @test st.u[:, end] == st.u[:, 1]
        @test all(st.w[1, :] .== 0) && maximum(abs, st.w[end, :]) <= 1e-12
        # mean density profile and the seeded displacement (interface height from the heavy-fluid depth)
        drho = kh_drho(KH_J)
        @test minimum(st.rho) >= KH_RHO0 - drho / 2 - 1e-9 && maximum(st.rho) <= KH_RHO0 + drho / 2 + 1e-9
        @test sigma ≈ kelvin_helmholtz_tanh_mode(; k=kh_k(), velocity=KH_U, thickness=KH_DELTA, richardson=KH_J, depth=KH_DEPTH)[1]
        @test_throws ArgumentError initialize_shear_layer_mode!(Model(ModelConfig(nx=20, nz=10)); velocity=KH_U,
            thickness=KH_DELTA, amplitude=KH_SEED)
    end

    # Measured relative rate errors (sigma = 0.07382 1/s at J = 0.05, 0.08808 1/s at J = 0): 8 cells per delta -1.02 %
    # (J = 0: -0.87 %), 16 cells -0.52 % (-0.46 %), 32 cells (experiment) -0.28 % (-0.27 %): first order, from below
    # (the numerical diffusion widens the shear layer).
    @testset "test_validate_kelvin_helmholtz_growth_rate" begin
        errors = Float64[]
        rate_fine = 0.0
        for cells in KH_CELLS
            r = kh_growth_rate(cells, KH_J)
            push!(errors, abs(r.rate - r.sigma) / r.sigma)
            rate_fine = r.rate
            @test r.dmass <= KH_MASS_RTOL
            @test r.dvolume <= KH_VOLUME_TOL
        end
        @test errors[1] <= KH_RATE_COARSE_MAX
        @test errors[end] <= KH_RATE_FINE_MAX
        @test errors[end] <= KH_CONVERGENCE_RATIO * errors[1]
        # control without stratification: as close to its own (larger) rate, and measurably faster than the stratified run
        r0 = kh_growth_rate(KH_CELLS[end], 0.0)
        @test abs(r0.rate - r0.sigma) / r0.sigma <= KH_RATE_FINE_MAX
        @test r0.rate > rate_fine
    end
end
