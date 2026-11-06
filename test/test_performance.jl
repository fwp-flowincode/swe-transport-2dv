# Performance overview, the reuse of the non-hydrostatic pressure factorization and the allocations of the
# wetting-drying step (issue #10). Docs section 32.

include("test_common.jl")

const PERF_NX = 20
const PERF_NZ = 4
const PERF_STEPS = 5
# bytes per hydrostatic wetting-drying step with k-epsilon and sediment (measured 1.4 KiB on 200 x 56 cells, 0.4 KiB here:
# boundary closures); a type instability in a kernel allocates per cell (10-30 MiB per step before issue #10)
const PERF_WD_STEP_BYTES_MAX = 16 * 1024
const PERF_DAM_NX = 60             # dam break over a sloping bed: the wet front changes the pressure pattern
const PERF_DAM_NZ = 10
const PERF_DAM_STEPS = 150
const PERF_DAM_DT = 0.01           # s

# Wetting-drying dam break over a sloping bed (non-hydrostatic): 0.45 m of bed rise over 6 m, water at 0.9 m behind the
# dam, 0.55 m in front of it, so that the front runs up the slope and cells become pressure unknowns.
function perf_dam_break()
    bed = [0.45 * (i - 0.5) / PERF_DAM_NX for i in 1:PERF_DAM_NX]
    bed = round.(bed ./ 0.1) .* 0.1   # m, whole cells (dz = 0.1 m)
    cfg = ModelConfig(; length=6.0, depth=1.0, nx=PERF_DAM_NX, nz=PERF_DAM_NZ, dt=PERF_DAM_DT, use_hydrostatic=false,
        wetting_drying=true, bed_level=bed, momentum_advection_scheme="tvd2", apply_froude_limiter=false,
        rayleigh_damping=0.0, write_plots=false)
    model = Model(cfg)
    initialize_water_level!(model, [i <= PERF_DAM_NX ÷ 3 ? 0.9 : 0.55 for i in 1:PERF_DAM_NX])
    return model
end

@testset "Performance (issue #10)" begin
    @testset "test_verify_performance_report" begin
        reset_performance!()
        @test isempty(performance_models())
        cfg = ModelConfig(; nx=PERF_NX, nz=PERF_NZ, dt=0.01, write_plots=false)
        model = Model(cfg)
        @test performance_models() == [model.performance]
        for _ in 1:PERF_STEPS
            step_forward!(model)
        end
        p = model.performance
        @test p.steps == PERF_STEPS
        @test p.step_time_ns > 0
        @test p.cells == PERF_NX * PERF_NZ
        @test p.label == "$(PERF_NX) x $(PERF_NZ), fixed grid, hydrostatic"
        @test p.timers === model.timers
        Model(cfg)   # built, no steps: not in the table
        s = sprint(io -> print_performance_report(io; title="Report"))
        for text in ("Report", "wall-clock time", "CPU time", "compilation", "garbage collection", "allocated",
                     "models 1, steps $(PERF_STEPS)", p.label, "us/cell/step", "Sections of all models", "step/")
            @test occursin(text, s)
        end
        cpu = process_cpu_time()
        @test isfinite(cpu) && cpu > 0

        # more models than table rows: the rest in one row; without the section timers
        for _ in 1:SweTransport2DV.PERFORMANCE_REPORT_MAX_MODELS
            step_forward!(Model(cfg))
        end
        s = sprint(io -> print_performance_report(io; timers=false))
        @test occursin("1 further models", s)
        @test !occursin("Sections of all models", s)
        reset_performance!()
        @test isempty(performance_models())
        s = sprint(io -> print_performance_report(io))
        @test occursin("models 0, steps 0", s)
        @test !occursin("us/cell/step", s)

        k_eps = Model(ModelConfig(; nx=PERF_NX, nz=PERF_NZ, dt=0.01, use_hydrostatic=false, turbulence_closure="k_epsilon",
            bed_roughness_length=1e-3, write_plots=false))
        @test k_eps.performance.label == "$(PERF_NX) x $(PERF_NZ), fixed grid, non-hydrostatic, k-epsilon"
        @test perf_dam_break().performance.label == "$(PERF_DAM_NX) x $(PERF_DAM_NZ), wetting-drying, non-hydrostatic"
    end

    @testset "test_verify_performance_format_bytes" begin
        fmt = SweTransport2DV._format_bytes
        @test fmt(512) == "512 B"
        @test fmt(512.4) == "512 B"
        @test fmt(2048) == "2.00 KiB"
        @test fmt(3 * 2.0^20) == "3.00 MiB"
        @test fmt(1.5 * 2.0^30) == "1.50 GiB"
    end

    @testset "test_verify_wd_pressure_factor_reuse" begin
        # the factor of the last step is reused (numeric factorization only) while the pattern is unchanged: the same
        # result, bit for bit, as a new symbolic analysis in every step
        reused, fresh = perf_dam_break(), perf_dam_break()
        for _ in 1:PERF_DAM_STEPS
            step_forward!(reused)
            fresh.wetdry.nh_factor = nothing   # forces the assembly with sparse and a new analysis
            step_forward!(fresh)
        end
        @test reused.state.u == fresh.state.u
        @test reused.state.eta == fresh.state.eta
        @test reused.wetdry.q_nh == fresh.wetdry.q_nh
        wd = reused.wetdry
        @test wd.nh_factorizations == PERF_DAM_STEPS
        @test 1 < wd.nh_symbolic_analyses < wd.nh_factorizations   # the front changed the pattern, but not every step
        @test fresh.wetdry.nh_symbolic_analyses == PERF_DAM_STEPS
    end

    @testset "test_verify_wd_step_allocations" begin
        # hydrostatic wetting-drying step with k-epsilon, sediment and an inflow: no allocation per cell
        nx, nz = PERF_NX, 2PERF_NZ
        cfg = ModelConfig(; length=20.0, depth=4.0, nx=nx, nz=nz, dt=0.05, use_hydrostatic=true, wetting_drying=true,
            turbulence_closure="k_epsilon", bed_roughness_length=1e-3, implicit_vertical_terms=true,
            momentum_advection_scheme="tvd2", transport_advection_scheme="tvd2", apply_froude_limiter=false,
            rayleigh_damping=0.0, write_plots=false)
        model = Model(cfg)
        initialize_water_level!(model, 4.0; rho=1000.0)
        set_x_inflow_outflow!(model, fill(0.2, nz); right_level=0.0, right_free=true, right_relaxation_time=20.0, left_rho=1000.0)
        attach_sediment!(model; settling_velocity=1e-3, critical_erosion_stress=0.1, erosion_parameter=1e-4)
        for _ in 1:3
            step_forward!(model)
        end
        @test @allocated(step_forward!(model)) < PERF_WD_STEP_BYTES_MAX
    end
end
