# Flexible-vegetation stem model and its coupling to the flow.
# Run with the full suite (runtests.jl) or on its own: julia --project=. test/test_vegetation.jl

include(joinpath(@__DIR__, "test_common.jl"))

@testset "Vegetation" begin
    @testset "test_single_vegetation_stem_under_prescribed_flow" begin
        cfg = ModelConfig(length=20.0, depth=4.0, nx=80, nz=20, dt=0.01, nt=1, use_hydrostatic=true, rayleigh_damping=0.0, apply_froude_limiter=false, write_plots=false, write_animation=false, write_pressure_buoyancy_diagnostics=false)
        model = Model(cfg)

        stem = VegetationStem(base_x=cfg.length / 2.0, height=0.8 * cfg.depth, diameter=0.03, drag_coefficient=1.5, stiffness=1.0, damping=0.2, mass_per_length=0.8, sample_points=12, source_strength=1.0)
        attach_vegetation!(model, VegetationModel(stem))

        z_centers = collect(range(model.dz / 2.0, cfg.depth - model.dz / 2.0, length=cfg.nz))
        z0 = 0.05 * cfg.depth
        log_profile = log.(1 .+ 20.0 .* (z_centers .- z0) ./ (cfg.depth - z0))
        log_profile ./= maximum(log_profile)

        @views for i in 1:(cfg.nx + 1)
            model.state.u[:, i] .= 0.12 .* log_profile
        end
        fill!(model.state.w, 0.0)
        fill!(model.state.eta, 0.0)
        fill!(model.state.p, 0.0)

        source = zeros(size(model.state.u))
        vegetation_source!(source, model.vegetation, model.state.u, cfg.dt, model.dx, model.dz, cfg.rho0)

        @test model.vegetation.stems[1].angle > 0.0
        @test sum(source) < 0.0
        @test maximum(abs.(source)) > 0.0
    end

    @testset "test_single_vegetation_stem_tracks_time_oscillating_flow" begin
        cfg = ModelConfig(length=10.0, depth=2.0, nx=50, nz=12, dt=0.01, nt=1, use_hydrostatic=true, rayleigh_damping=0.0, apply_froude_limiter=false, write_plots=false, write_animation=false, write_pressure_buoyancy_diagnostics=false)
        model = Model(cfg)

        stem = VegetationStem(base_x=0.5 * cfg.length, height=0.75 * cfg.depth, diameter=0.025, drag_coefficient=1.6, stiffness=0.25, damping=0.05, mass_per_length=0.7, sample_points=10, source_strength=1.0)
        veg = VegetationModel(stem)

        angles = Float64[]
        net_sources = Float64[]
        source = zeros(size(model.state.u))
        z_centers = collect(range(model.dz / 2.0, cfg.depth - model.dz / 2.0, length=cfg.nz))

        for n in 1:120
            t = (n - 1) * cfg.dt
            amplitude = 0.45 * sin(2.0 * pi * 0.8 * t)
            for i in 1:(cfg.nx + 1)
                for k in 1:cfg.nz
                    model.state.u[k, i] = amplitude * (0.35 + 0.65 * z_centers[k] / cfg.depth)
                end
            end
            vegetation_source!(source, veg, model.state.u, cfg.dt, model.dx, model.dz, cfg.rho0)
            push!(angles, veg.stems[1].angle)
            push!(net_sources, sum(source))
        end

        @test maximum(angles) > 0.0
        @test maximum(abs.(angles)) > 1e-4
        @test maximum(net_sources) > 0.0
        @test minimum(net_sources) < 0.0
    end

    @testset "test_vegetation_damps_free_surface_wave_motion" begin
        cfg = ModelConfig(length=20.0, depth=4.0, nx=80, nz=20, dt=5e-4, nt=1, use_hydrostatic=true, momentum_advection_scheme="upwind1", rayleigh_damping=0.0, apply_froude_limiter=false, write_plots=false, write_animation=false, write_pressure_buoyancy_diagnostics=false)
        model_ref = Model(cfg)
        model_veg = Model(cfg)

        stem = VegetationStem(base_x=0.5 * cfg.length, height=0.85 * cfg.depth, diameter=0.03, drag_coefficient=1.8, stiffness=1.0, damping=0.25, mass_per_length=0.8, sample_points=12, source_strength=1.0)
        attach_vegetation!(model_veg, VegetationModel(stem))

        x_centers = collect(range(model_ref.dx / 2.0, cfg.length - model_ref.dx / 2.0, length=cfg.nx))
        wave_amp = 0.08
        for i in 1:cfg.nx
            model_ref.state.eta[i] = wave_amp * sin(2.0 * pi * x_centers[i] / cfg.length)
            model_veg.state.eta[i] = wave_amp * sin(2.0 * pi * x_centers[i] / cfg.length)
        end
        fill!(model_ref.state.u, 0.0)
        fill!(model_ref.state.w, 0.0)
        fill!(model_ref.state.p, 0.0)
        fill!(model_veg.state.u, 0.0)
        fill!(model_veg.state.w, 0.0)
        fill!(model_veg.state.p, 0.0)

        for _ in 1:250
            step_forward!(model_ref)
            step_forward!(model_veg)
        end

        amp_ref = maximum(abs, model_ref.state.eta)
        amp_veg = maximum(abs, model_veg.state.eta)

        @test model_veg.vegetation.stems[1].angle != 0.0
        @test amp_veg < amp_ref
    end

    @testset "test_vegetation_slows_lock_exchange_front" begin
        cfg_ref = ModelConfig(length=30.0, depth=4.0, nx=100, nz=20, dt=5e-4, nt=1, use_hydrostatic=false, momentum_advection_scheme="tvd2", rayleigh_damping=0.01, apply_froude_limiter=true, max_froude=0.9, write_plots=false, write_animation=false, write_pressure_buoyancy_diagnostics=false)
        cfg_veg = copy_config(cfg_ref)

        model_ref = Model(cfg_ref)
        model_veg = Model(cfg_veg)

        stem = VegetationStem(base_x=0.5 * cfg_ref.length, height=0.9 * cfg_ref.depth, diameter=0.035, drag_coefficient=2.0, stiffness=1.2, damping=0.35, mass_per_length=0.9, sample_points=14, source_strength=1.1)
        attach_vegetation!(model_veg, VegetationModel(stem))

        initialize_lock_exchange!(model_ref)
        initialize_lock_exchange!(model_veg)

        x_centers = collect(range(model_ref.dx / 2.0, cfg_ref.length - model_ref.dx / 2.0, length=cfg_ref.nx))
        threshold = cfg_ref.rho_right + 0.8 * (cfg_ref.rho_left - cfg_ref.rho_right)

        function rightmost_front(rho_row)
            idx = findall(>(threshold), rho_row)
            if isempty(idx)
                return NaN
            end
            j = maximum(idx)
            if j >= length(rho_row)
                return x_centers[end]
            end
            rj = rho_row[j]
            rjp = rho_row[j + 1]
            if rjp >= threshold || abs(rj - rjp) < 1e-14
                return x_centers[j]
            end
            return x_centers[j] + (rj - threshold) / (rj - rjp) * (x_centers[j + 1] - x_centers[j])
        end

        for _ in 1:600
            step_forward!(model_ref)
            step_forward!(model_veg)
        end

        x_ref = rightmost_front(@view model_ref.state.rho[1, :])
        x_veg = rightmost_front(@view model_veg.state.rho[1, :])

        @test isfinite(x_ref)
        @test isfinite(x_veg)
        @test x_veg < x_ref
        @test model_veg.vegetation.stems[1].angle != 0.0
    end
end
