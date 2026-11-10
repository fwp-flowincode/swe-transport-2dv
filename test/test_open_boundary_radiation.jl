include("open_boundary_helpers.jl")

# Radiation of dispersive waves at the open faces, right and left (issue #86, docs/numerics_open_boundaries.md section 7):
# theta-weighted radiation (radiation_theta), radiation speed of a design period on the current
# (radiation_period) and radiation of the non-hydrostatic pressure of the face (nonhydro_open_face = "radiation"),
# wetting-drying step. Section 39 of docs/TESTS_TECHNICAL_DOCUMENTATION.md.
@testset "Open boundary: radiation of dispersive waves (issue #86)" begin
    @testset "test_verify_radiation_speed_dispersion_relation" begin
        h = RD_DEPTH
        for U in (-0.16, 0.0, 0.16), T in (0.8, 1.44, 3.0)
            cfg = ModelConfig(nx=10, nz=4, wetting_drying=true, use_hydrostatic=false, nonhydro_open_face="radiation",
                radiation_period=T, write_plots=false)
            c = SweTransport2DV._radiation_speed(cfg, U, h)
            omega = 2pi / T
            k = omega / c
            @test abs((omega - k * U)^2 - RD_G * k * tanh(k * h)) <= RD_RADIATION_SPEED_RTOL * RD_G * k * tanh(k * h)
            @test c < U + sqrt(RD_G * h)   # dispersive: slower than the long wave
        end
        cfg0 = ModelConfig(nx=10, nz=4, write_plots=false)
        @test SweTransport2DV._radiation_speed(cfg0, 0.1, h) == 0.1 + sqrt(RD_G * h)   # period 0: long-wave speed
    end

    @testset "test_verify_radiation_options_config_checks" begin
        @test_throws ArgumentError ModelConfig(nx=10, nz=4, nonhydro_open_face="dirichlet", write_plots=false) |> Model
        @test_throws ArgumentError Model(ModelConfig(nx=10, nz=4, use_hydrostatic=false, nonhydro_open_face="radiation",
            pressure_solver_method="sparse", write_plots=false))   # fixed grid
        @test_throws ArgumentError Model(ModelConfig(nx=10, nz=4, wetting_drying=true, use_hydrostatic=false,
            radiation_period=-1.0, nonhydro_open_face="radiation", write_plots=false))
        @test_throws ArgumentError Model(ModelConfig(nx=10, nz=4, wetting_drying=true, use_hydrostatic=false,
            radiation_period=1.0, write_plots=false))   # period without the pressure radiation
        @test_throws ArgumentError Model(ModelConfig(nx=10, nz=4, wetting_drying=true, use_hydrostatic=true,
            radiation_period=1.0, nonhydro_open_face="radiation", write_plots=false))   # hydrostatic
    end

    @testset "test_verify_radiation_face_lake_at_rest_and_volume_budget" begin
        # still water: no wave maker motion (amplitude 0 through steps of a run with U = 0 and no forcing)
        h = RD_DEPTH
        nx = 40
        cfg = ModelConfig(; nx=nx, nz=RD_NZ, length=10.0, depth=h, dt=0.01, theta=0.7, implicit_vertical_terms=true,
            nu=1e-6, use_hydrostatic=false, wetting_drying=true, bed_level=zeros(nx), write_plots=false, rd_new(1.2)...)
        model = Model(cfg)
        fill!(model.state.rho, 1000.0)
        refresh_density_metadata!(model)
        initialize_water_level!(model, fill(h, nx))
        set_x_inflow_outflow!(model, zeros(RD_NZ); right_free=true)
        for _ in 1:200
            step_forward!(model)
        end
        @test maximum(abs, model.state.u) == 0.0
        @test maximum(abs, water_depth(model) .- h) == 0.0
        @test maximum(abs, model.wetdry.q_right) == 0.0
        # waves: the water volume changes only by the boundary fluxes of the step (measured 8.8e-15 m^2)
        _, budget_error = rd_reflection(; kh=1.2, options=rd_new(1.2), budget=true, steps=600)
        @test budget_error < TOL_MODEL_IDENTITY * RD_DEPTH * RD_WAVELENGTHS * 2pi / (1.2 / RD_DEPTH)
    end

    @testset "test_verify_radiation_face_direct_and_multigrid_agree" begin
        # the pressure radiation only adds to the diagonal: the system stays SPD and both solvers apply
        m1, _ = rd_reflection(; kh=1.2, options=rd_new(1.2), nonhydro_solver="direct", steps=300)
        m2, _ = rd_reflection(; kh=1.2, options=rd_new(1.2), nonhydro_solver="multigrid", steps=300)
        @test m2.wetdry.nh_multigrid_solves > 0 && m2.wetdry.nh_fallbacks == 0
        @test maximum(abs, water_depth(m1) .- water_depth(m2)) < RD_SOLVER_TOL * RD_AMPLITUDE
    end

    @testset "test_validate_theta_weighted_radiation_long_wave" begin
        R_new, _ = rd_reflection(; kh=0.3, options=rd_new(0.3))
        R_old, _ = rd_reflection(; kh=0.3)
        @test R_new < RD_R_MAX_LONG
        @test R_old > RD_R_MIN_LONG_BASELINE
    end

    @testset "test_validate_pressure_radiation_dispersive_wave" begin
        R_new, _ = rd_reflection(; kh=1.2, options=rd_new(1.2))
        R_old, _ = rd_reflection(; kh=1.2)
        @test R_new < RD_R_MAX_DISPERSIVE
        @test R_old > RD_R_MIN_DISPERSIVE_BASELINE
    end

    @testset "test_validate_pressure_radiation_following_current" begin
        R, _ = rd_reflection(; kh=1.2, U=RD_CURRENT, options=rd_new(1.2))
        @test R < RD_R_MAX_CURRENT
    end

    @testset "test_verify_left_free_face_config_checks" begin
        h, nx = RD_DEPTH, 20
        wd_cfg = ModelConfig(nx=nx, nz=4, length=2.0, depth=h, dt=0.01, implicit_vertical_terms=true, wetting_drying=true,
            bed_level=zeros(nx), write_plots=false)
        m = Model(wd_cfg)
        @test_throws ArgumentError set_x_inflow_outflow!(m, zeros(4); left_free=true, left_discharge=0.1)
        @test_throws ArgumentError set_x_inflow_outflow!(m, zeros(4); left_free=true, left_level=0.0)   # no tau
        @test_throws ArgumentError set_x_inflow_outflow!(m, zeros(4); left_free=true, left_relaxation_time=10.0)   # no target
        @test_throws ArgumentError set_x_inflow_outflow!(m, zeros(4); left_relaxation_time=10.0)   # not free
        @test_throws ArgumentError set_x_inflow_outflow!(m, zeros(4); left_free=true, left_level=0.0, left_relaxation_time=0.0)
        fixed = Model(ModelConfig(nx=nx, nz=4, implicit_vertical_terms=true, write_plots=false))
        @test_throws ArgumentError set_x_inflow_outflow!(fixed, zeros(4); left_free=true)   # fixed grid
        set_x_inflow_outflow!(m, zeros(4); left_free=true, left_level=0.0, left_relaxation_time=10.0)
        @test m.wetdry.left_open !== nothing && m.wetdry.left_open.relaxation_time == 10.0
        clear_x_inflow_outflow!(m)
        @test m.wetdry.left_open === nothing
    end

    @testset "test_verify_left_free_face_mirror_symmetry" begin
        # the left face is the mirror of the right one: a symmetric hump stays symmetric (former and new options)
        @test rd_mirror_asymmetry() < RD_MIRROR_TOL
        @test rd_mirror_asymmetry(; options=rd_new(1.2)) < RD_MIRROR_TOL
    end

    @testset "test_verify_left_free_face_uniform_outflow_steady" begin
        # a current of 0.16 m/s to the left: velocity inflow at the right face, relaxed free left face (target 0); the
        # uniform flow is a steady state to round-off
        h, nx, U = RD_DEPTH, 40, -RD_CURRENT
        cfg = ModelConfig(; nx=nx, nz=RD_NZ, length=10.0, depth=h, dt=0.01, theta=0.7, implicit_vertical_terms=true,
            nu=1e-6, rayleigh_damping=0.0, apply_froude_limiter=false, use_hydrostatic=false, wetting_drying=true,
            bed_level=zeros(nx), write_plots=false, rd_new(1.2)...)
        model = Model(cfg)
        fill!(model.state.rho, 1000.0)
        refresh_density_metadata!(model)
        initialize_water_level!(model, fill(h, nx))
        model.state.u .= U
        set_x_inflow_outflow!(model, zeros(RD_NZ); left_free=true, left_level=0.0, left_relaxation_time=10.0,
            open_right=false, right_profile=fill(U, RD_NZ))
        for _ in 1:300
            step_forward!(model)
        end
        @test maximum(abs, model.state.u .- U) < TOL_MODEL_IDENTITY
        @test maximum(abs, water_depth(model) .- h) < TOL_MODEL_IDENTITY
    end

    @testset "test_validate_left_free_face_reflection_matches_right" begin
        # k h = 1.2 packet: the left face reflects as the right one, and the new options reduce the reflection there too
        new_left = rd_packet_deviation(; dir=-1, options=rd_new(1.2))
        new_right = rd_packet_deviation(; dir=1, options=rd_new(1.2))
        former_left = rd_packet_deviation(; dir=-1)
        @test new_left < RD_PACKET_MAX_NEW
        @test former_left > RD_PACKET_MIN_FORMER
        @test abs(new_left - new_right) <= RD_PACKET_MIRROR_RTOL * new_right
    end

    @testset "test_validate_residual_reflection_is_interior_dispersion_error" begin
        # the boundary radiates the linear-theory phase speed; the interior's phase speed differs by c_num/c - 1 =
        # k/k_num - 1, and the mismatch reflects about half of it
        R, kr = rd_reflection(; kh=2.0, options=rd_new(2.0))
        ratio = R / (abs(1 / kr - 1) / 2)
        @test RD_DISPERSION_RATIO_RANGE[1] < ratio < RD_DISPERSION_RATIO_RANGE[2]
    end
end
