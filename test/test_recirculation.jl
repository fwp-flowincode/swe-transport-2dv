include("test_common.jl")

# Recirculation behind obstacles (issue #50): the horizontal eddy diffusion of the wetting-drying step and the
# backward-facing step of case ST1 of Nakagawa & Nezu (1987), as simulated by Luo et al. (2023, JFM 966 A18).
# See docs/TESTS_TECHNICAL_DOCUMENTATION.md, section 28.

const RC_L = 1.0                # m, channel of the kernel tests
const RC_H = 1.0                # m
const RC_NU_T = 1e-3            # m^2/s
const RC_DT = 1e-3              # s
const RC_NX = (40, 80, 160)
const RC_ORDER_MIN = 1.9        # -, second order on the uniform grid
# Step ST1: S = 2 cm, depth 6 cm downstream (expansion ratio 1.5), Re = 4380, Fr = 0.19
const ST1_STEP = 0.02           # m
const ST1_DEPTH = 0.06          # m
const ST1_Q = 0.00876           # m^2/s
const ST1_Z0 = 1e-5             # m, smooth bed, nu / (9 u*)
const ST1_RESOLUTION = 5        # -, cells per step height
const ST1_TIME = 90.0           # s, the start-up transient has decayed
const ST1_AVERAGE = 30.0        # s
const ST1_RELAXATION = 20.0     # s, reflects < 3 % of the 8 s mode of the reach
const ST1_XR_LES = 6.15         # step heights, Luo et al. (2023), Fig. 3
const ST1_XR_RTOL = 0.2         # -, the grid spread of section 28; this grid gives 6.18
const ST1_W_RATIO_MIN = 3.0     # -, largest w hydrostatic against non-hydrostatic (measured 4.1, and 12 at S/10)

function rc_model(nx; nz=4, bed=nothing)
    cfg = ModelConfig(; length=RC_L, depth=RC_H, nx=nx, nz=nz, dt=RC_DT, nu=1e-6, turbulence_closure="k_epsilon",
        bed_roughness_length=1e-3, rayleigh_damping=0.0, apply_froude_limiter=false, wetting_drying=true,
        implicit_vertical_terms=true, theta=1.0, bed_level=bed, write_plots=false)
    model = Model(cfg)
    wd = model.wetdry
    SweTransport2DV._wd_cell_thicknesses!(wd.h, model, wd)
    SweTransport2DV._wd_face_thicknesses!(model, wd)
    copyto!(wd.h_new, wd.h)
    fill!(wd.q, 0.0)
    fill!(model.state.w, 0.0)
    fill!(model.turbulence.nu_t, RC_NU_T)
    return model
end

function st1_mean(; hydro)
    dx = ST1_STEP / ST1_RESOLUTION
    dt = 0.4 * dx / 0.25
    length_m, x_step = 70 * ST1_STEP, 40 * ST1_STEP
    nx, nz = round(Int, length_m / dx), round(Int, ST1_DEPTH / dx)
    xc = ((1:nx) .- 0.5) .* dx
    cfg = ModelConfig(; length=length_m, depth=ST1_DEPTH, nx=nx, nz=nz, dt=dt, use_hydrostatic=hydro, nu=1e-6,
        turbulence_closure="k_epsilon", bed_roughness_length=ST1_Z0, rayleigh_damping=0.0, apply_froude_limiter=false,
        momentum_advection_scheme="tvd2", wetting_drying=true, implicit_vertical_terms=true, theta=1.0,
        bed_level=[x < x_step ? ST1_STEP : 0.0 for x in xc], write_plots=false)
    model = Model(cfg)
    fill!(model.state.rho, 1000.0)
    refresh_density_metadata!(model)
    initialize_water_level!(model, fill(ST1_DEPTH, nx))
    set_x_inflow_outflow!(model, ones(nz); left_discharge=ST1_Q, right_level=0.0, right_free=true,
        right_relaxation_time=ST1_RELAXATION)
    steps = round(Int, ST1_TIME / dt)
    start = steps - round(Int, ST1_AVERAGE / dt)
    ubar, hbar = zeros(nz, nx), zeros(nz, nx)
    wmax = 0.0
    for n in 1:steps
        step_forward!(model)
        n > start || continue
        u = model.state.u
        @views ubar .+= 0.5 .* (u[:, 1:nx] .+ u[:, 2:(nx + 1)])
        hbar .+= model.wetdry.h
        wmax = max(wmax, maximum(abs, model.state.w))
    end
    ubar ./= steps - start
    hbar ./= steps - start
    return reattachment_length(ubar, hbar, xc, x_step) / ST1_STEP, wmax
end

@testset "Recirculation behind obstacles" begin
    @testset "test_verify_wd_horizontal_eddy_diffusion_second_order" begin
        # MMS: u = sin(2 pi x / L), constant nu_t and h: the term must be nu_t d2u/dx2 = -nu_t (2 pi / L)^2 u
        errs = map(RC_NX) do nx
            model = rc_model(nx)
            xf = (0:nx) .* (RC_L / nx)
            model.state.u .= sin.(2pi .* xf ./ RC_L)'
            ke, wd = model.turbulence, model.wetdry
            maximum(abs(SweTransport2DV._wd_horizontal_eddy_diffusion(model, wd, ke, 2, f) +
                        RC_NU_T * (2pi / RC_L)^2 * sin(2pi * xf[f] / RC_L)) for f in 2:nx)
        end
        @test all(log2(errs[i] / errs[i + 1]) > RC_ORDER_MIN for i in 1:(length(errs) - 1))
    end

    @testset "test_verify_wd_horizontal_turbulence_diffusion_second_order_and_conservative" begin
        # no flow: the update of phi = 2 + cos(pi x / L) is dt nu_t/sigma d2phi/dx2; the walls have zero flux
        errs = map(RC_NX) do nx
            model = rc_model(nx)
            xc = ((1:nx) .- 0.5) .* (RC_L / nx)
            phi = repeat((2 .+ cos.(pi .* xc ./ RC_L))', model.config.nz, 1)
            out = similar(phi)
            SweTransport2DV._wd_advect_turbulence!(out, phi, model, model.wetdry, 0.0, 1.0)
            @test sum(out) ≈ sum(phi) rtol = TOL_IDENTITY
            exact = -RC_DT * RC_NU_T * (pi / RC_L)^2 .* cos.(pi .* xc ./ RC_L)
            maximum(abs.(out[2, 2:(nx - 1)] .- phi[2, 2:(nx - 1)] .- exact[2:(nx - 1)])) / (RC_DT * RC_NU_T)
        end
        @test all(log2(errs[i] / errs[i + 1]) > RC_ORDER_MIN for i in 1:(length(errs) - 1))
    end

    @testset "test_verify_wd_horizontal_diffusion_limits" begin
        # a huge nu_t: the update stays a convex combination of the neighbours
        nx = 20
        model = rc_model(nx)
        ke, wd = model.turbulence, model.wetdry
        fill!(ke.nu_t, 1e6)
        model.state.u .= 0.0
        model.state.u[2, 11] = 1.0
        f = 11
        @test 1.0 + RC_DT * SweTransport2DV._wd_horizontal_eddy_diffusion(model, wd, ke, 2, f) ≈
              1.0 - SweTransport2DV.WD_HORIZONTAL_DIFFUSION_MAX rtol = TOL_IDENTITY
        phi = zeros(model.config.nz, nx)
        phi[2, 10] = 1.0
        out = similar(phi)
        SweTransport2DV._wd_advect_turbulence!(out, phi, model, wd, 0.0, 1.0)
        @test all(>=(0.0), out) && maximum(out) <= 1.0
        @test sum(out) ≈ 1.0 rtol = TOL_IDENTITY
        # the switch, and a closed neighbouring face (a step): only the open side contributes
        model = rc_model(nx; bed=[i <= 10 ? 0.5 : 0.0 for i in 1:nx])
        ke, wd = model.turbulence, model.wetdry
        @test wd.hf[1, 11] == 0.0
        model.state.u[1, 12] = 1.0
        @test SweTransport2DV._wd_horizontal_eddy_diffusion(model, wd, ke, 1, 12) ≈
              -RC_NU_T * (nx / RC_L)^2 rtol = TOL_IDENTITY
    end

    @testset "test_verify_reattachment_length" begin
        # psi from a bed-velocity sign change at a known position: u(x) = (x - xr) at the bed, forward flow above
        xr = 0.637
        xc = collect(0.01:0.02:1.99)
        u = zeros(3, length(xc))
        u[1, :] .= xc .- xr
        u[2:3, :] .= 1.0
        @test reattachment_length(u, fill(0.1, size(u)), xc, 0.2) ≈ xr - 0.2 rtol = TOL_IDENTITY
        @test reattachment_length(fill(1.0, size(u)), fill(0.1, size(u)), xc, 0.2) == 0.0
    end

    xr_nh, w_nh = st1_mean(; hydro=false)
    @testset "test_validate_backward_step_reattachment_nonhydrostatic" begin
        @test abs(xr_nh - ST1_XR_LES) < ST1_XR_RTOL * ST1_XR_LES
    end

    @testset "test_validate_backward_step_hydrostatic_vertical_velocity" begin
        # without the non-hydrostatic pressure nothing limits w at the step
        _, w_hy = st1_mean(; hydro=true)
        @test w_hy > ST1_W_RATIO_MIN * w_nh
    end
end
