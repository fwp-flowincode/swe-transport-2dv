# Wetting and drying (issue #31, docs/plan_issue_31_wetting_drying.md): cut-cell layer geometry, lake at rest,
# conservation of water and constituents, Ritter dam break on a dry bed and Carrier-Greenspan runup on a planar beach.
# "1D" is the single-layer model (nz = 1), "2DV" the multi-layer model (nz > 1); both use the same wet/dry time step.
# Run with the full suite (runtests.jl) or on its own: julia --project=. test/test_wetting_drying.jl

include(joinpath(@__DIR__, "test_common.jl"))

# Wetting-drying configuration with the numerical stabilisers off (they are not part of the tested physics).
function wd_config(; kwargs...)
    defaults = (wetting_drying=true, use_hydrostatic=true, momentum_advection_scheme="upwind1", nu=0.0,
        rayleigh_damping=0.0, apply_froude_limiter=false, rho_left=1000.0, rho_right=1000.0,
        write_plots=false, write_animation=false, write_pressure_buoyancy_diagnostics=false)
    return ModelConfig(; merge(defaults, kwargs)...)
end

# Cell-centre coordinates (m) of a configuration.
wd_x_centres(cfg) = [(i - 0.5) * cfg.length / cfg.nx for i in 1:cfg.nx]

@testset "Wetting and drying" begin
    # Named tolerances (no hardcoded tolerances inside the tests).
    TOL_REST_VELOCITY = 1e-13      # m/s, lake at rest: largest velocity after many steps (machine precision)
    TOL_REST_LEVEL = 1e-13         # m, lake at rest: largest change of the water level
    TOL_VOLUME_REL = 1e-12         # -, relative water-volume change in a closed basin (round-off)
    TOL_MASS_REL = 1e-12           # -, relative constituent-mass change in a closed basin (round-off)
    TOL_DENSITY_BOUND = 1e-9       # kg/m^3, allowed overshoot of the initial density bounds (round-off)
    WET_DEPTH = 1e-3               # m, a column counts as wet above this depth (a drained beach keeps a thin film)

    @testset "test_verify_wd_layer_thickness_kernel" begin
        nz, dz = 5, 0.2
        h = zeros(nz)
        for (zeta, zb) in ((0.73, 0.0), (0.73, 0.31), (0.31, 0.31), (0.2, 0.5), (1.37, 0.05), (0.4, 0.4 - 1e-9), (0.6, 0.6))
            layer_thicknesses!(h, zeta, zb, dz)
            @test all(>=(0.0), h)
            @test isapprox(sum(h), max(zeta - zb, 0.0); atol=TOL_IDENTITY)
        end
        # flat bed, surface inside the top layer: the variable top layer (dz + eta on top, dz below)
        eta = -0.03
        layer_thicknesses!(h, nz * dz + eta, 0.0, dz)
        @test all(isapprox.(h[1:(nz - 1)], dz; atol=TOL_IDENTITY))
        @test isapprox(h[nz], dz + eta; atol=TOL_IDENTITY)
        # the top layer is open above the grid top
        layer_thicknesses!(h, nz * dz + 0.15, 0.0, dz)
        @test isapprox(h[nz], dz + 0.15; atol=TOL_IDENTITY)
        # bed cuts layer 2, surface cuts layer 4
        layer_thicknesses!(h, 0.75, 0.33, dz)
        @test h[1] == 0.0 && h[5] == 0.0
        @test isapprox(h[2], 0.4 - 0.33; atol=TOL_IDENTITY)
        @test isapprox(h[3], dz; atol=TOL_IDENTITY)
        @test isapprox(h[4], 0.75 - 0.6; atol=TOL_IDENTITY)
    end

    @testset "test_verify_wd_configuration_checks" begin
        @test_throws ArgumentError Model(wd_config(nx=20, nz=4, use_hydrostatic=false))
        @test_throws ArgumentError Model(wd_config(nx=20, nz=4, momentum_advection_scheme="tvd2"))
        @test_throws ArgumentError Model(wd_config(nx=20, nz=4, periodic_x=true, pressure_solver_method="sparse"))
        @test_throws ArgumentError Model(wd_config(nx=20, nz=4, bed_level=fill(-0.1, 20)))
        # a single layer (1D) and a bed that is not a whole number of cells are accepted
        model = Model(wd_config(nx=20, nz=1, depth=2.0, bed_level=collect(range(0.0, 2.5, length=20))))
        @test model isa Model
    end

    # Bed for the lake at rest (m above the grid bottom): a beach rising above the still water level on the right, an
    # island and a submerged bump; none of the levels is a whole number of cells.
    function lake_bed(cfg)
        x = wd_x_centres(cfg)
        L = cfg.length
        beach = @. max(0.0, (x - 0.7 * L) / (0.3 * L)) * 1.4 * cfg.depth
        island = @. 1.15 * cfg.depth * exp(-((x - 0.3 * L) / (0.04 * L))^2)
        bump = @. 0.45 * cfg.depth * exp(-((x - 0.5 * L) / (0.05 * L))^2)
        return @. 0.037 + beach + island + bump
    end

    @testset "test_verify_wd_lake_at_rest_$(label)" for (label, nz, stratified) in
            (("1d", 1, false), ("2dv", 10, false), ("2dv_stratified", 10, true))
        NSTEPS = 500                 # -
        STILL_LEVEL_FRACTION = 0.8   # -, still water level / grid depth
        cfg = wd_config(length=50.0, depth=2.0, nx=200, nz=nz, dt=0.05)
        cfg.bed_level = lake_bed(cfg)
        model = Model(cfg)
        level = STILL_LEVEL_FRACTION * cfg.depth
        if stratified
            # two-layer stratification in z (horizontally uniform), interface at mid-depth of the still water
            dz = cfg.depth / cfg.nz
            rho = [((k - 0.5) * dz < 0.5 * level ? 1010.0 : 1000.0) for k in 1:cfg.nz, i in 1:cfg.nx]
            initialize_water_level!(model, level; rho=rho)
        else
            initialize_water_level!(model, level; rho=1000.0)
        end
        depth0 = water_depth(model)
        dry0 = depth0 .== 0.0
        @test any(dry0) && any(.!dry0)   # the case does contain dry cells (beach top and island)

        for _ in 1:NSTEPS
            step_forward!(model)
        end
        zeta = cfg.depth .+ model.state.eta
        @test maximum(abs, model.state.u) <= TOL_REST_VELOCITY
        @test maximum(abs, zeta[.!dry0] .- level) <= TOL_REST_LEVEL
        @test all(water_depth(model)[dry0] .== 0.0)
    end

    # Closed basin with beaches on both sides; a tilted surface sloshes, floods the right beach and drains again.
    function sloshing_model(nz)
        cfg = wd_config(length=40.0, depth=1.5, nx=160, nz=nz, dt=0.02)
        x = wd_x_centres(cfg)
        cfg.bed_level = @. 0.013 + 1.6 * ((x - 0.5 * cfg.length) / (0.5 * cfg.length))^2
        model = Model(cfg)
        level = @. 0.9 + 0.25 * (0.5 * cfg.length - x) / (0.5 * cfg.length)
        dz = cfg.depth / cfg.nz
        # two water masses in x (and in z for 2DV) of different density
        rho = [(x[i] < 0.4 * cfg.length ? 1004.0 : 1000.0) + ((k - 0.5) * dz < 0.5 ? 2.0 : 0.0) for k in 1:cfg.nz, i in 1:cfg.nx]
        initialize_water_level!(model, level; rho=rho)
        return model
    end

    @testset "test_verify_wd_volume_and_constituent_conservation_$(label)" for (label, nz) in (("1d", 1), ("2dv", 10))
        NSTEPS = 1000   # -, 20 s, more than half a sloshing period
        model = sloshing_model(nz)
        diag0 = compute_diagnostics(model)
        rho_lo, rho_hi = extrema(model.state.rho)
        wet0 = water_depth(model) .> WET_DEPTH
        wet_changed = falses(model.config.nx)
        min_depth = Inf
        rho_overshoot = 0.0
        for _ in 1:NSTEPS
            step_forward!(model)
            depth = water_depth(model)
            min_depth = min(min_depth, minimum(depth))
            wet_changed .|= (depth .> WET_DEPTH) .!= wet0
            rho_overshoot = max(rho_overshoot, maximum(model.state.rho) - rho_hi, rho_lo - minimum(model.state.rho))
        end
        diag = compute_diagnostics(model)
        @test abs(diag["water_volume"] - diag0["water_volume"]) <= TOL_VOLUME_REL * diag0["water_volume"]
        @test abs(diag["constituent_mass"] - diag0["constituent_mass"]) <= TOL_MASS_REL * diag0["constituent_mass"]
        @test min_depth >= 0.0
        @test rho_overshoot <= TOL_DENSITY_BOUND
        @test count(wet_changed) >= 5   # cells did flood and dry
        @test all(isfinite, model.state.u)
    end

    # Depth positivity (docs/depth_positivity.md): the limited continuity step keeps every depth >= 0 for any time step.
    # Dam break onto a dry bed and the sloshing basin with time steps far beyond the advective CFL limit, where the
    # limiter must act; the depth never becomes negative and volume and mass stay conserved.
    @testset "test_verify_wd_depth_positivity_large_time_step_$(label)" for (label, nz) in (("1d", 1), ("2dv", 10))
        CFL_TARGET = 3.0          # -, advective Courant number 2 c0 dt / dx of the dam break (positivity is unconditional)
        NSTEPS = 200              # -
        CLAMP_REL_MAX = 1e-14     # -, volume added by setting round-off negative depths to zero, relative to the total
        cfg = wd_config(length=40.0, depth=1.2, nx=80, nz=nz)
        dx = cfg.length / cfg.nx
        cfg.dt = CFL_TARGET * dx / (2 * sqrt(cfg.g * 1.0))
        x = wd_x_centres(cfg)
        cfg.bed_level = @. 0.05 * max(x - 0.5 * cfg.length, 0.0)   # a dry beach behind the dry flat bed
        model = Model(cfg)
        dz = cfg.depth / cfg.nz
        rho = [(x[i] < 10.0 ? 1003.0 : 1000.0) + ((k - 0.5) * dz < 0.4 ? 1.0 : 0.0) for k in 1:cfg.nz, i in 1:cfg.nx]
        initialize_water_level!(model, [xi < 10.0 ? 1.0 : 0.0 for xi in x]; rho=rho)
        diag0 = compute_diagnostics(model)
        min_depth = Inf
        theta_min = 1.0
        for _ in 1:NSTEPS
            step_forward!(model)
            min_depth = min(min_depth, minimum(cfg.depth .+ model.state.eta .- model.wetdry.bed))
            theta_min = min(theta_min, minimum(model.wetdry.theta))
        end
        diag = compute_diagnostics(model)
        @test theta_min < 1.0        # the limiter did act
        @test min_depth >= 0.0
        @test model.wetdry.clamped_volume <= CLAMP_REL_MAX * diag0["water_volume"]
        @test abs(diag["water_volume"] - diag0["water_volume"]) <= TOL_VOLUME_REL * diag0["water_volume"]
        @test abs(diag["constituent_mass"] - diag0["constituent_mass"]) <= TOL_MASS_REL * diag0["constituent_mass"]
        @test all(isfinite, model.state.u)
    end

    @testset "test_verify_wd_negative_depth_is_rejected" begin
        cfg = wd_config(length=10.0, depth=1.0, nx=20, nz=4, dt=0.01)
        cfg.bed_level = fill(0.3, cfg.nx)
        model = Model(cfg)
        initialize_water_level!(model, 0.8; rho=1000.0)
        model.state.eta[7] = 0.2 - cfg.depth   # water level below the bed: depth -0.1 m
        @test_throws DomainError step_forward!(model)
    end

    @testset "test_verify_ritter_reference_solution" begin
        RITTER_QUADRATURE_RTOL = 1e-5   # -, midpoint sum over the jump at x0 - c0 t: error ~ step * h0 / 2
        # Ritter (1892): the rarefaction conserves the released volume, joins h0 at x0 - c0 t and ends at x0 + 2 c0 t
        h0, x0, g, t = 1.0, 40.0, 9.81, 5.0
        c0 = sqrt(g * h0)
        xs = range(0.0, 100.0, length=200001)
        h = [ritter_dam_break(x, t; h0=h0, x0=x0, g=g)[1] for x in xs]
        volume = sum(h) * step(xs)
        @test isapprox(volume, h0 * x0; rtol=RITTER_QUADRATURE_RTOL)
        @test ritter_dam_break(x0 - c0 * t - 1e-9, t; h0=h0, x0=x0, g=g) == (h0, 0.0)
        @test ritter_dam_break(x0 + 2 * c0 * t + 1e-9, t; h0=h0, x0=x0, g=g)[1] == 0.0
        # at the dam the depth is 4/9 h0 and the velocity 2/3 c0
        hd, ud = ritter_dam_break(x0, t; h0=h0, x0=x0, g=g)
        @test isapprox(hd, 4h0 / 9; rtol=TOL_IDENTITY) && isapprox(ud, 2c0 / 3; rtol=TOL_IDENTITY)
    end

    @testset "test_verify_carrier_greenspan_reference_solution" begin
        # The hodograph solution must satisfy the nonlinear shallow-water equations over a sloping bed (x offshore,
        # depth = slope * x + eta, u offshore): eta_t + ((slope x + eta) u)_x = 0 and u_t + u u_x + g eta_x = 0.
        CG_RESIDUAL_MAX = 1e-6   # -, residual relative to the scales of the terms (central differences: O(DELTA^2))
        DELTA = 1e-4             # -, relative finite-difference step (of the length and time scales)
        cg = CarrierGreenspanWave(amplitude=0.4, slope=0.05, length_scale=20.0)
        L, T = cg.length_scale, carrier_greenspan_time_scale(cg)
        Z, U = cg.slope * L, sqrt(cg.g * cg.slope * L)
        dx, dt = DELTA * L, DELTA * T
        for x in (0.3L, 0.5L, 0.9L), t in (0.1T, 0.7T, 1.9T)
            s(x, t) = carrier_greenspan_solution(cg, x, t)
            eta_t = (s(x, t + dt)[1] - s(x, t - dt)[1]) / (2dt)
            flux(x) = (cg.slope * x + s(x, t)[1]) * s(x, t)[2]
            mass = eta_t + (flux(x + dx) - flux(x - dx)) / (2dx)
            u = s(x, t)[2]
            u_t = (s(x, t + dt)[2] - s(x, t - dt)[2]) / (2dt)
            u_x = (s(x + dx, t)[2] - s(x - dx, t)[2]) / (2dx)
            eta_x = (s(x + dx, t)[1] - s(x - dx, t)[1]) / (2dx)
            momentum = u_t + u * u_x + cg.g * eta_x
            @test abs(mass) <= CG_RESIDUAL_MAX * Z * U / L   # m/s, Z U / L: scale of the mass terms
            @test abs(momentum) <= CG_RESIDUAL_MAX * U / T   # m/s^2, U / T: scale of the momentum terms
        end
        # linear limit: the shoreline oscillates with amplitude A/4 (vertical) about the still water line
        small = CarrierGreenspanWave(amplitude=1e-4, slope=0.05, length_scale=20.0)
        rise = maximum(carrier_greenspan_shoreline(small, t)[2] for t in range(0.0, pi * T, length=401))
        @test isapprox(rise, small.amplitude / 4 * Z; rtol=1e-3)
        # dry beyond the shoreline
        @test !carrier_greenspan_solution(cg, -0.2L, 0.0)[3]
    end

    # ---- Ritter dam break on a dry bed --------------------------------------------------------------------------
    RITTER_H0 = 1.0             # m, initial depth behind the dam
    RITTER_X0 = 40.0            # m, dam position
    RITTER_T = 5.0              # s, comparison time (before the waves reach the walls)
    RITTER_L1_REL_MAX = 0.03    # -, accepted relative L1 depth error at dx = 0.25 m
    # WHY 12 %: the scheme is first order; the thin tip of the rarefaction (h ~ (x_f - x)^2) lags by numerical diffusion
    # and converges slowly (12.7 %, 11.3 %, 7.5 % of the travel distance at dx = 0.5, 0.25, 0.125 m; the measured front
    # is quantised by dx), while the depth converges at first order (L1 1.4 %, 0.9 %, 0.6 %). The test therefore also
    # requires the front error to decrease under refinement.
    RITTER_FRONT_REL_MAX = 0.12 # -, accepted front-position error relative to the distance travelled, at dx = 0.25 m
    RITTER_FRONT_DEPTH = 0.01   # m, depth that marks the front (numerical and analytical)
    RITTER_CONVERGENCE_RATIO = 0.8   # -, the L1 error must drop at least by this factor when dx is halved
    RITTER_2DV_VS_1D_MAX = 0.01      # -, relative L1 difference between the 2DV and the 1D depth

    function run_ritter(nx, nz)
        grid_depth = 1.2 * RITTER_H0
        cfg = wd_config(length=100.0, depth=grid_depth, nx=nx, nz=nz, dt=0.005 * 200 / nx)
        model = Model(cfg)
        x = wd_x_centres(cfg)
        initialize_water_level!(model, [xi < RITTER_X0 ? RITTER_H0 : 0.0 for xi in x]; rho=1000.0)
        nsteps = round(Int, RITTER_T / cfg.dt)
        for _ in 1:nsteps
            step_forward!(model)
        end
        h = water_depth(model)
        h_exact = [ritter_dam_break(xi, nsteps * cfg.dt; h0=RITTER_H0, x0=RITTER_X0, g=cfg.g)[1] for xi in x]
        l1 = sum(abs, h .- h_exact) / sum(h_exact)
        front(hh) = x[findlast(>(RITTER_FRONT_DEPTH), hh)]
        c0 = sqrt(cfg.g * RITTER_H0)
        front_exact = RITTER_X0 + (2c0 - 3 * sqrt(cfg.g * RITTER_FRONT_DEPTH)) * nsteps * cfg.dt
        front_error = abs(front(h) - front_exact) / (front_exact - RITTER_X0)
        return model, h, l1, front_error
    end

    @testset "test_validate_wd_ritter_dry_dam_break_1d" begin
        _, _, l1_coarse, front_coarse = run_ritter(200, 1)
        _, _, l1_fine, front_fine = run_ritter(400, 1)
        @test l1_fine <= RITTER_L1_REL_MAX
        @test l1_fine <= RITTER_CONVERGENCE_RATIO * l1_coarse
        @test front_fine <= RITTER_FRONT_REL_MAX
        @test front_fine < front_coarse
    end

    @testset "test_validate_wd_ritter_dry_dam_break_2dv" begin
        model, h2dv, l1, front_error = run_ritter(400, 12)
        _, h1d, _, _ = run_ritter(400, 1)
        @test l1 <= RITTER_L1_REL_MAX
        @test front_error <= RITTER_FRONT_REL_MAX
        @test sum(abs, h2dv .- h1d) / sum(h1d) <= RITTER_2DV_VS_1D_MAX
        @test all(isfinite, model.state.u)
    end

    # ---- Carrier-Greenspan standing wave on a planar beach ------------------------------------------------------
    CG_AMPLITUDE = 0.4          # -, Carrier-Greenspan amplitude A (non-breaking for A < 1)
    CG_SLOPE = 0.05             # -, beach slope
    CG_LENGTH = 20.0            # m, length scale = distance of the offshore boundary from the still shoreline
    CG_STILL_LEVEL = 1.2        # m, still water level above the grid bottom
    CG_ONSHORE = 4.0            # m, dry beach included landward of the still shoreline
    CG_SHORE_DEPTH = 1e-3       # m, depth that marks the numerical shoreline
    # WHY at dx = 0.05 m: the shoreline error converges at first order (14.6 mm at dx = 0.1 m, 9.2 mm at 0.05 m, largest
    # at maximum rundown, where the thin film on the beach drains); 10 % of the runup amplitude needs dx = 0.05 m.
    CG_SHORE_REL_MAX = 0.1      # -, accepted shoreline-elevation error relative to the runup amplitude A/4 * slope * L
    CG_ETA_REL_MAX = 0.05       # -, accepted surface-elevation error (wet cells) relative to the same amplitude
    CG_CONVERGENCE_RATIO = 0.8  # -, the shoreline error must drop at least by this factor when dx is halved
    CG_ETA_EVERY = 10           # -, steps between comparisons of the surface elevation (the reference is costly)

    # One wave period on a planar beach, forced by the analytical velocity at the offshore boundary; returns the largest
    # shoreline-elevation error, surface-elevation error (m) and the runup amplitude (m).
    function run_carrier_greenspan(nx, nz, dt)
        cg = CarrierGreenspanWave(amplitude=CG_AMPLITUDE, slope=CG_SLOPE, length_scale=CG_LENGTH)
        T = carrier_greenspan_time_scale(cg)
        amplitude = cg.amplitude / 4 * cg.slope * cg.length_scale   # m, vertical runup amplitude (linear)
        cfg = wd_config(length=CG_LENGTH + CG_ONSHORE, depth=1.5, nx=nx, nz=nz, dt=dt)
        x = wd_x_centres(cfg)
        x_off(xm) = CG_LENGTH - xm   # m, offshore distance from the still shoreline
        cfg.bed_level = @. CG_STILL_LEVEL - cg.slope * x_off(x)
        model = Model(cfg)
        # start at the phase of maximum runup, where u = 0 everywhere
        t0 = carrier_greenspan_rest_time(cg)
        level = map(x) do xi
            eta_a, _, wet = carrier_greenspan_solution(cg, x_off(xi), t0)
            wet ? CG_STILL_LEVEL + eta_a : 0.0
        end
        initialize_water_level!(model, level; rho=1000.0)
        set_x_inflow_outflow!(model, t -> -carrier_greenspan_solution(cg, CG_LENGTH, t0 + t)[2]; open_right=false)

        nsteps = round(Int, pi * T / cfg.dt)   # one period
        shore_error = 0.0
        eta_error = 0.0
        for n in 1:nsteps
            step_forward!(model)
            t = t0 + n * cfg.dt
            depth = water_depth(model)
            zeta = cfg.depth .+ model.state.eta
            i_shore = findlast(>(CG_SHORE_DEPTH), depth)
            shore_error = max(shore_error, abs(zeta[i_shore] - CG_STILL_LEVEL - carrier_greenspan_shoreline(cg, t)[2]))
            n % CG_ETA_EVERY == 0 || continue
            for i in 1:(i_shore - 2)
                eta_a, _, wet = carrier_greenspan_solution(cg, x_off(x[i]), t)
                wet && (eta_error = max(eta_error, abs(zeta[i] - CG_STILL_LEVEL - eta_a)))
            end
        end
        @test all(isfinite, model.state.u)
        return shore_error, eta_error, amplitude
    end

    @testset "test_validate_wd_carrier_greenspan_runup_1d" begin
        shore_coarse, _, _ = run_carrier_greenspan(240, 1, 0.02)
        shore_fine, eta_fine, amplitude = run_carrier_greenspan(480, 1, 0.01)
        @test shore_fine <= CG_SHORE_REL_MAX * amplitude
        @test shore_fine <= CG_CONVERGENCE_RATIO * shore_coarse
        @test eta_fine <= CG_ETA_REL_MAX * amplitude
    end

    @testset "test_validate_wd_carrier_greenspan_runup_2dv" begin
        shore, eta, amplitude = run_carrier_greenspan(480, 15, 0.01)
        @test shore <= CG_SHORE_REL_MAX * amplitude
        @test eta <= CG_ETA_REL_MAX * amplitude
    end
end
