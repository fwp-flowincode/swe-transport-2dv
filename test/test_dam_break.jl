# Dam breaks on a flat, frictionless bed with the wetting-drying step: Stoker (1957) on a wet bed (issue #11) and Ritter
# (1892) on a dry bed (issue #12), in 1D and in 2DV with 2, 3, 5 and 10 layers, hydrostatic and non-hydrostatic, with and
# without a transported constituent. Docs section 33.

include("test_common.jl")

const DB_LENGTH = 100.0       # m
const DB_X0 = 40.0            # m, dam position
const DB_H0 = 1.0             # m, depth behind the dam
const DB_H1 = 0.2             # m, depth in front of the dam (Stoker; h1/h0 = 0.2 as in SWASHES, bore Froude number 2.1)
const DB_GRID_DEPTH = 1.2     # m
const DB_T = 5.0              # s, comparison time (the waves have not reached the walls)
const DB_DT_DX = 0.01         # s/m, time step per grid spacing (advective Courant number below 0.1)
const DB_NX = 400             # -, dx = 0.25 m
const DB_NX_COARSE = 200      # -, dx = 0.5 m (convergence)
const DB_LAYERS = (2, 3, 5, 10)
const DB_TRACER_LAYERS = 5
const DB_LAYER_DEPTH_MIN = 1e-3   # m, face layers thinner than this are left out of the layer velocity spread

# Stoker. Measured at dx = 0.25 m (upwind1): 1D L1 0.42 % (0.74 % at dx = 0.5 m), bore -0.34 %; hydrostatic 2DV L1
# 0.42-0.52 %, bore -0.4 to -1.7 %, 2DV against 1D 0.01-0.23 %, layer spread up to 0.08 m/s; non-hydrostatic L1
# 0.54-1.03 %, bore -0.5 to -3.6 %, against 1D 0.28-0.88 %.
const STOKER_L1_MAX = 0.01             # -, relative L1 depth error (hydrostatic)
const STOKER_BORE_MAX = 0.01           # -, bore position error relative to the distance travelled (1D)
const STOKER_BORE_2DV_MAX = 0.025      # -, the same with layers (hydrostatic)
const STOKER_CONVERGENCE_RATIO = 0.8   # -, the L1 error must drop at least by this factor when dx is halved
const STOKER_2DV_VS_1D_MAX = 0.01      # -, relative L1 difference between the 2DV and the 1D depth (hydrostatic)
# m/s, layer velocities behind the bore (hydrostatic): the shock splits the momentum between the layers depending on the
# scheme (the layers below h1 start at rest, those above are filled from upstream), and nothing evens it out without
# viscosity; 4 % of the middle-state velocity of 1.8 m/s
const STOKER_LAYER_SPREAD_MAX = 0.1
# WHY looser non-hydrostatic bounds: the non-hydrostatic bore has a front of the width of the depth and lags the
# shallow-water bore by about 0.5 m (one middle-state depth) at dx = 0.25 m and at 0.125 m alike; the
# shallow-water solution is not its limit
const STOKER_NH_L1_MAX = 0.015
const STOKER_NH_BORE_MAX = 0.05
const STOKER_NH_VS_1D_MAX = 0.015

# Ritter (the bounds of test_wetting_drying.jl). Measured at dx = 0.25 m: L1 0.90 % (hydrostatic), 1.00-1.06 %
# (non-hydrostatic), front 10.3-11.3 %, hydrostatic layer spread at most 0.0012 m/s.
const RITTER_DB_L1_MAX = 0.03
const RITTER_DB_FRONT_MAX = 0.12
const RITTER_DB_FRONT_DEPTH = 0.01     # m, depth that marks the front
const RITTER_DB_SPREAD_MAX = 0.01      # m/s

const DB_CONSERVATION_RTOL = 1e-12     # -, water volume and constituent mass (closed channel; measured 1e-15)

# Dam break with depth `h1` in front of the dam (0: Ritter) at t = DB_T; returns the model, the cell centres (m), the
# depth (m), the analytical depth (m) and, with `tracer`, the change of the constituent mass relative to its initial
# value. The constituent is a passive tracer (settling_velocity = 0, no erosion, no density coupling), 1 kg/m^3 behind
# the dam.
function db_run(nx, nz; h1, hydrostatic=true, scheme="upwind1", tracer=false)
    cfg = ModelConfig(; wetting_drying=true, use_hydrostatic=hydrostatic, momentum_advection_scheme=scheme,
        transport_advection_scheme=scheme, nu=0.0, rayleigh_damping=0.0, apply_froude_limiter=false, length=DB_LENGTH,
        depth=DB_GRID_DEPTH, nx=nx, nz=nz, dt=DB_DT_DX * DB_LENGTH / nx, write_plots=false)
    model = Model(cfg)
    x = [(i - 0.5) * DB_LENGTH / nx for i in 1:nx]
    initialize_water_level!(model, [xi < DB_X0 ? DB_H0 : h1 for xi in x]; rho=1000.0)
    if tracer
        attach_sediment!(model; settling_velocity=0.0, critical_erosion_stress=1.0, erosion_parameter=0.0,
            density_coupling=false, concentration=[x[i] < DB_X0 ? 1.0 : 0.0 for k in 1:nz, i in 1:nx])
    end
    mass0 = tracer ? suspended_sediment_mass(model) : 1.0
    volume0 = sum(water_depth(model))
    steps = round(Int, DB_T / cfg.dt)
    for _ in 1:steps
        step_forward!(model)
    end
    t = steps * cfg.dt
    h = water_depth(model)
    exact = [stoker_dam_break(xi, t; h0=DB_H0, h1=h1, x0=DB_X0, g=cfg.g)[1] for xi in x]
    @test abs(sum(h) - volume0) <= DB_CONSERVATION_RTOL * volume0
    @test all(isfinite, model.state.u)
    dmass = tracer ? abs(suspended_sediment_mass(model) - mass0) / mass0 : 0.0
    return (; model, x, h, exact, t, dmass)
end

db_l1(r) = sum(abs, r.h .- r.exact) / sum(r.exact)

# Bore position error of a Stoker run relative to the distance travelled: the numerical bore is where the depth crosses
# (h_m + h1)/2, interpolated linearly between the cell centres.
function db_bore_error(r; h1=DB_H1)
    hm, _, s = stoker_middle_state(; h0=DB_H0, h1=h1, g=r.model.config.g)
    level = 0.5 * (hm + h1)
    i = findlast(>(level), r.h)
    x_bore = r.x[i] + (r.h[i] - level) / (r.h[i] - r.h[i + 1]) * (r.x[i + 1] - r.x[i])
    return (x_bore - (DB_X0 + s * r.t)) / (s * r.t)
end

# Front position error of a Ritter run relative to the distance travelled (the front is the last cell deeper than
# RITTER_DB_FRONT_DEPTH; the analytical position of that depth is x0 + (2 c0 - 3 sqrt(g h_f)) t).
function db_front_error(r)
    g = r.model.config.g
    travel = (2 * sqrt(g * DB_H0) - 3 * sqrt(g * RITTER_DB_FRONT_DEPTH)) * r.t
    return abs(r.x[findlast(>(RITTER_DB_FRONT_DEPTH), r.h)] - (DB_X0 + travel)) / travel
end

# Largest difference (m/s) between the layer velocities of a face (layers thicker than DB_LAYER_DEPTH_MIN).
function db_layer_spread(model)
    spread = 0.0
    for f in 2:model.config.nx
        lo, hi = Inf, -Inf
        for k in 1:model.config.nz
            if model.wetdry.hf[k, f] > DB_LAYER_DEPTH_MIN
                lo, hi = min(lo, model.state.u[k, f]), max(hi, model.state.u[k, f])
            end
        end
        hi > lo && (spread = max(spread, hi - lo))
    end
    return spread
end

@testset "Dam breaks (issues #11, #12)" begin
    @testset "test_verify_stoker_reference_solution" begin
        g, h0, h1, x0, t = 9.81, DB_H0, DB_H1, DB_X0, DB_T
        hm, um, s = stoker_middle_state(; h0=h0, h1=h1, g=g)
        c0, cm = sqrt(g * h0), sqrt(g * hm)
        @test h1 < hm < h0 && um > 0 && s > um
        # Rankine-Hugoniot conditions of the bore: mass and momentum
        @test isapprox(s * (hm - h1), hm * um; rtol=TOL_IDENTITY)
        @test isapprox(s * hm * um, hm * um^2 + g / 2 * (hm^2 - h1^2); rtol=TOL_IDENTITY)
        # Riemann invariant u + 2 c through the rarefaction
        @test isapprox(um + 2cm, 2c0; rtol=TOL_IDENTITY)
        # the solution: still water, rarefaction joining the middle state continuously, bore, still water
        @test stoker_dam_break(x0 - c0 * t - 1e-9, t; h0=h0, h1=h1, x0=x0, g=g) == (h0, 0.0)
        tail = x0 + (um - cm) * t
        hl, ul = stoker_dam_break(tail - 1e-9, t; h0=h0, h1=h1, x0=x0, g=g)
        @test isapprox(hl, hm; rtol=1e-8) && isapprox(ul, um; rtol=1e-8)
        @test stoker_dam_break(x0 + s * t - 1e-9, t; h0=h0, h1=h1, x0=x0, g=g) == (hm, um)
        @test stoker_dam_break(x0 + s * t + 1e-9, t; h0=h0, h1=h1, x0=x0, g=g) == (h1, 0.0)
        # volume: the released water fills the domain between the rarefaction head and the bore
        xs = range(0.0, DB_LENGTH, length=400001)
        volume = sum(stoker_dam_break(x, t; h0=h0, h1=h1, x0=x0, g=g)[1] for x in xs) * step(xs)
        @test isapprox(volume, h0 * x0 + h1 * (DB_LENGTH - x0); rtol=1e-5)   # midpoint sum over two jumps
        # dry bed: the Ritter solution
        @test stoker_dam_break(45.0, t; h0=h0, h1=0.0, x0=x0, g=g) == ritter_dam_break(45.0, t; h0=h0, x0=x0, g=g)
        @test_throws ArgumentError stoker_middle_state(; h0=h0, h1=h0)
        @test_throws ArgumentError stoker_middle_state(; h0=h0, h1=0.0)
        @test_throws ArgumentError stoker_dam_break(45.0, 0.0; h0=h0, h1=h1)
    end

    @testset "test_validate_stoker_wet_dam_break_1d" begin
        coarse = db_run(DB_NX_COARSE, 1; h1=DB_H1)
        fine = db_run(DB_NX, 1; h1=DB_H1)
        @test db_l1(fine) <= STOKER_L1_MAX
        @test db_l1(fine) <= STOKER_CONVERGENCE_RATIO * db_l1(coarse)
        @test abs(db_bore_error(fine)) <= STOKER_BORE_MAX
        # tvd2 (measured L1 0.28 %, bore -0.10 %)
        tvd = db_run(DB_NX, 1; h1=DB_H1, scheme="tvd2")
        @test db_l1(tvd) < db_l1(fine)
        @test abs(db_bore_error(tvd)) <= STOKER_BORE_MAX
    end

    h1d = db_run(DB_NX, 1; h1=DB_H1).h
    @testset "test_validate_stoker_wet_dam_break_2dv_hydrostatic_nz$(nz)" for nz in DB_LAYERS
        r = db_run(DB_NX, nz; h1=DB_H1)
        @test db_l1(r) <= STOKER_L1_MAX
        @test abs(db_bore_error(r)) <= STOKER_BORE_2DV_MAX
        @test sum(abs, r.h .- h1d) / sum(h1d) <= STOKER_2DV_VS_1D_MAX
        @test db_layer_spread(r.model) <= STOKER_LAYER_SPREAD_MAX
    end

    @testset "test_validate_stoker_wet_dam_break_2dv_nonhydrostatic_nz$(nz)" for nz in DB_LAYERS
        r = db_run(DB_NX, nz; h1=DB_H1, hydrostatic=false)
        @test db_l1(r) <= STOKER_NH_L1_MAX
        @test abs(db_bore_error(r)) <= STOKER_NH_BORE_MAX
        @test sum(abs, r.h .- h1d) / sum(h1d) <= STOKER_NH_VS_1D_MAX
    end

    @testset "test_validate_ritter_dry_dam_break_2dv_$(hydrostatic ? "hydrostatic" : "nonhydrostatic")_nz$(nz)" for
            hydrostatic in (true, false), nz in DB_LAYERS
        r = db_run(DB_NX, nz; h1=0.0, hydrostatic=hydrostatic)
        @test db_l1(r) <= RITTER_DB_L1_MAX
        @test db_front_error(r) <= RITTER_DB_FRONT_MAX
        hydrostatic && @test db_layer_spread(r.model) <= RITTER_DB_SPREAD_MAX
    end

    # the passive constituent is conserved and does not change the flow
    @testset "test_verify_dam_break_constituent_mass_$(h1 > 0 ? "stoker" : "ritter")_$(hydrostatic ? "hydrostatic" : "nonhydrostatic")" for
            h1 in (DB_H1, 0.0), hydrostatic in (true, false)
        with = db_run(DB_NX, DB_TRACER_LAYERS; h1=h1, hydrostatic=hydrostatic, tracer=true)
        without = db_run(DB_NX, DB_TRACER_LAYERS; h1=h1, hydrostatic=hydrostatic)
        @test with.dmass <= DB_CONSERVATION_RTOL
        @test with.h == without.h
        @test with.model.state.u == without.model.state.u
        c = with.model.wetdry.sediment.c
        @test all(>=(0), c) && maximum(c) <= 1.0 + DB_CONSERVATION_RTOL   # monotone transport: no new extrema
    end
end
