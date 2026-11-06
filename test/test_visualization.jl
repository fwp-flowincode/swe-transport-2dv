# Figures of the model state (src/visualization.jl): the final-state plot, the animation and the pressure-buoyancy
# diagnostics are written for a small fixed grid, a bed with solid cells and a wetting-drying case with vegetation
# (issue #10: coverage of the plotting code, which the other tests switch off).

include("test_common.jl")

const VIS_NX = 24
const VIS_NZ = 6
const VIS_STEPS = 4
const VIS_FRAMES = 3

# Writes the figures of `model` into `dir` and returns the file names.
function vis_write_figures(model, dir, name; vegetation=nothing)
    cfg = model.config
    history = [copy_state(model.state)]
    eta_history = [copy(model.state.eta)]
    for _ in 1:(VIS_FRAMES - 1)
        for _ in 1:VIS_STEPS
            step_forward!(model)
        end
        push!(history, copy_state(model.state))
        push!(eta_history, copy(model.state.eta))
    end
    final = joinpath(dir, "$(name)_final.png")
    anim = joinpath(dir, "$(name)_animation.gif")
    diag = joinpath(dir, "$(name)_pressure_buoyancy.png")
    plot_state(model.state, cfg; title=name, filename=final, eta_prev=eta_history[end - 1], dt=cfg.dt, vegetation=vegetation)
    vegetation_history = vegetation === nothing ? nothing : fill(vegetation, VIS_FRAMES)
    create_animation(history, cfg; filename=anim, save_interval=VIS_STEPS, eta_prev_history=eta_history,
        frame_times=collect(0:(VIS_FRAMES - 1)) .* VIS_STEPS .* cfg.dt, vegetation_history=vegetation_history)
    plot_pressure_buoyancy_diagnostics(model.state, cfg, model.last_p_hydro, model.last_p_corr, model.last_buoy_w;
        filename=diag)
    return (final, anim, diag)
end

@testset "Visualization (issue #10)" begin
    mktempdir() do dir
        @testset "test_verify_plots_fixed_grid" begin
            cfg = ModelConfig(; nx=VIS_NX, nz=VIS_NZ, dt=0.01, use_hydrostatic=false, write_plots=false)
            for file in vis_write_figures(Model(cfg), dir, "fixed")
                @test isfile(file) && filesize(file) > 0
            end
        end

        @testset "test_verify_plots_bed_with_solid_cells" begin
            bed = [abs(i - VIS_NX ÷ 2) < 4 ? 2 * 4.0 / VIS_NZ : 0.0 for i in 1:VIS_NX]   # m, two cells high
            cfg = ModelConfig(; nx=VIS_NX, nz=VIS_NZ, dt=0.01, use_hydrostatic=false, pressure_solver_method="sparse",
                bed_level=bed, write_plots=false)
            for file in vis_write_figures(Model(cfg), dir, "bed")
                @test isfile(file) && filesize(file) > 0
            end
        end

        @testset "test_verify_plots_wetting_drying_with_vegetation" begin
            bed = [0.5 * (i - 0.5) / VIS_NX for i in 1:VIS_NX]
            bed = round.(bed ./ (1.0 / VIS_NZ)) .* (1.0 / VIS_NZ)   # m, whole cells
            cfg = ModelConfig(; length=6.0, depth=1.0, nx=VIS_NX, nz=VIS_NZ, dt=0.01, wetting_drying=true, bed_level=bed,
                write_plots=false)
            model = Model(cfg)
            initialize_water_level!(model, [i <= VIS_NX ÷ 3 ? 0.9 : 0.4 for i in 1:VIS_NX])
            stem = VegetationStem(base_x=4.0, height=0.3, diameter=0.01, drag_coefficient=1.2, bending_stiffness=Inf,
                lateral_spacing=0.1)
            for file in vis_write_figures(model, dir, "wetting_drying"; vegetation=VegetationModel(stem))
                @test isfile(file) && filesize(file) > 0
            end
        end
    end
end
