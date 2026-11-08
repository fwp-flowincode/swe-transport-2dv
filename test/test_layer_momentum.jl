# Momentum of layered bores in the wetting-drying step (issue #80): the momentum budget of the Stoker dam break, the
# column-form loss in layers and the remapped advection (ModelConfig.layer_momentum_advection = "remapped"; Kleptsova,
# Stelling & Pietrzak 2010), which reproduces the 1D bore in every number of layers. docs/TESTS_TECHNICAL_DOCUMENTATION.md,
# section 38.
# Run with the full suite (runtests.jl) or on its own: julia --project=. test/test_layer_momentum.jl

include(joinpath(@__DIR__, "test_common.jl"))

# ---- Case: the Stoker and Ritter dam breaks of test_dam_break.jl -------------------------------------------------------
const LM_LENGTH = 100.0       # m
const LM_X0 = 40.0            # m, dam
const LM_H0 = 1.0             # m, depth behind the dam
const LM_H1 = 0.2             # m, depth in front of the dam (Stoker)
const LM_GRID_DEPTH = 1.2     # m
const LM_NX = 400             # -, dx = 0.25 m
const LM_DT_DX = 0.01         # s/m
const LM_T = 5.0              # s, the waves have not reached the walls
const LM_LAYERS = (2, 3, 5, 10)

# ---- Bounds (measured values in the comments) ---------------------------------------------------------------------------
const LM_MOMENTUM_1D_RTOL = 0.01          # -, 1D momentum against the exact force impulse (observed +0.45 %, first order)
# remapped layers against 1D: the same to round-off (largest depth difference 2.1e-14 m upwind1, 4.7e-15 m tvd2)
const LM_REMAPPED_VS_1D_BORE = 1e-10      # m, bore position
const LM_REMAPPED_VS_1D_DEPTH = 1e-12     # m, largest depth difference
const LM_REMAPPED_SPREAD = 1e-9           # m/s, layer velocity spread
const LM_REMAPPED_MOMENTUM_RTOL = 1e-10   # -, momentum of the layered run against 1D
const LM_COLUMN_DEFICIT_RANGE = (0.003, 0.015)   # -, momentum deficit of the column form in 10 layers (observed 0.90 %;
                                                 # bore 54.602 m against 54.795 m in 1D, exact 54.847 m)

lm_rate(g=9.81) = 0.5 * g * (LM_H0^2 - LM_H1^2)   # m^3/s^2, exact momentum gain of the channel (pressure force)
lm_momentum(m) = sum(m.wetdry.hf .* m.state.u) * m.dx   # m^3/s

# Bore position (m): where the depth crosses (h_m + h1)/2, interpolated between the cells
function lm_bore(x, h)
    hm, _, _ = stoker_middle_state(; h0=LM_H0, h1=LM_H1)
    level = 0.5 * (hm + LM_H1)
    i = findlast(>(level), h)
    return x[i] + (h[i] - level) / (h[i] - h[i + 1]) * (x[i + 1] - x[i])
end

# Largest difference (m/s) between the layer velocities of a face (layers thicker than 1 cm)
lm_spread(m) = maximum(begin
    lo, hi = Inf, -Inf
    for k in 1:m.config.nz
        m.wetdry.hf[k, f] > 0.01 && (lo = min(lo, m.state.u[k, f]); hi = max(hi, m.state.u[k, f]))
    end
    hi > lo ? hi - lo : 0.0
end for f in 2:m.config.nx)

function lm_run(nz; h1=LM_H1, advection="column", scheme="upwind1", hydrostatic=true)
    dx = LM_LENGTH / LM_NX
    cfg = ModelConfig(; wetting_drying=true, use_hydrostatic=hydrostatic, momentum_advection_scheme=scheme,
        transport_advection_scheme=scheme, nu=0.0, rayleigh_damping=0.0, apply_froude_limiter=false, length=LM_LENGTH,
        depth=LM_GRID_DEPTH, nx=LM_NX, nz=nz, dt=LM_DT_DX * dx, layer_momentum_advection=advection, write_plots=false,
        write_animation=false, write_pressure_buoyancy_diagnostics=false)
    m = Model(cfg)
    x = [(i - 0.5) * dx for i in 1:LM_NX]
    initialize_water_level!(m, [xi < LM_X0 ? LM_H0 : h1 for xi in x]; rho=1000.0)
    M0 = lm_momentum(m)
    steps = round(Int, LM_T / cfg.dt)
    for _ in 1:steps
        step_forward!(m)
    end
    t = steps * cfg.dt
    h = water_depth(m)
    return (; m, x, h, t, momentum=(lm_momentum(m) - M0) / (lm_rate() * t), bore=h1 > 0 ? lm_bore(x, h) : NaN)
end

@testset "Layer momentum (issue #80)" begin
    @testset "test_verify_layer_momentum_budget_1d" begin
        # the 1D scheme gains the pressure-force impulse 1/2 g (h0^2 - h1^2) t, to first order
        r = lm_run(1)
        @test abs(r.momentum - 1) <= LM_MOMENTUM_1D_RTOL
    end

    for scheme in ("upwind1", "tvd2")
        r1 = lm_run(1; scheme=scheme)
        @testset "test_validate_remapped_layers_reproduce_1d_stoker_$(scheme)" begin
            for nz in LM_LAYERS
                r = lm_run(nz; scheme=scheme, advection="remapped")
                @test abs(r.bore - r1.bore) <= LM_REMAPPED_VS_1D_BORE
                @test maximum(abs, r.h .- r1.h) <= LM_REMAPPED_VS_1D_DEPTH
                @test abs(r.momentum - r1.momentum) <= LM_REMAPPED_MOMENTUM_RTOL
                @test lm_spread(r.m) <= LM_REMAPPED_SPREAD
            end
        end
    end

    @testset "test_validate_column_form_momentum_deficit" begin
        # the column form loses momentum where the bore lifts the surface through layer interfaces (section 38); a
        # conservative scheme for flows with vertical structure would close this gap
        r1, r10 = lm_run(1), lm_run(10)
        @test LM_COLUMN_DEFICIT_RANGE[1] <= r1.momentum - r10.momentum <= LM_COLUMN_DEFICIT_RANGE[2]
        @test r10.bore < r1.bore
    end

    @testset "test_validate_remapped_layers_reproduce_1d_ritter" begin
        r1 = lm_run(1; h1=0.0)
        r12 = lm_run(12; h1=0.0, advection="remapped")
        @test maximum(abs, r12.h .- r1.h) <= LM_REMAPPED_VS_1D_DEPTH
        @test lm_spread(r12.m) <= LM_REMAPPED_SPREAD
    end

    @testset "test_verify_remapped_single_layer_is_column_form" begin
        # one layer: the remapped form is not used (nz = 1), the run is the column form bit for bit
        a, b = lm_run(1; advection="column"), lm_run(1; advection="remapped")
        @test a.m.state.u == b.m.state.u
        @test a.h == b.h
    end

    @testset "test_verify_remapped_selection" begin
        # bed step and non-hydrostatic mode: the column form; flat bed, hydrostatic: remapped
        cfg(; kw...) = ModelConfig(; wetting_drying=true, nx=40, nz=6, length=10.0, depth=1.2, dt=0.01, nu=0.0,
            rayleigh_damping=0.0, apply_froude_limiter=false, write_plots=false, write_animation=false,
            write_pressure_buoyancy_diagnostics=false, layer_momentum_advection="remapped", implicit_vertical_terms=true, kw...)
        step_bed = [i <= 20 ? 0.2 : 0.0 for i in 1:40]
        m = Model(cfg(; use_hydrostatic=true, bed_level=step_bed))
        initialize_water_level!(m, 1.0; rho=1000.0)
        set_x_inflow_outflow!(m, ones(6); left_discharge=0.2, right_level=0.0, right_free=true, right_relaxation_time=5.0)
        step_forward!(m)
        # faces whose neighbourhood f-2..f+2 (faces and cells between them) reaches over the step between cells 20 and 21
        @test !any(m.wetdry.remap.active[20:23])
        @test m.wetdry.remap.active[19] && m.wetdry.remap.active[24]
        @test m.wetdry.remap.active[10] && m.wetdry.remap.active[30]
        mn = Model(cfg(; use_hydrostatic=false))
        initialize_water_level!(mn, 1.0; rho=1000.0)
        step_forward!(mn)
        @test !any(mn.wetdry.remap.active)
        @test_throws ArgumentError Model(ModelConfig(wetting_drying=true, layer_momentum_advection="sigma"))
    end
end
