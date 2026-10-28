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

# Pressure modes and advection schemes of the wet/dry step; the exactness and conservation tests run for all of them.
const WD_VARIANTS = (
    (tag="hydro_upwind1", use_hydrostatic=true, momentum_advection_scheme="upwind1", transport_advection_scheme="upwind1"),
    (tag="hydro_tvd2", use_hydrostatic=true, momentum_advection_scheme="tvd2", transport_advection_scheme="tvd2"),
    (tag="nonhydro_upwind1", use_hydrostatic=false, momentum_advection_scheme="upwind1", transport_advection_scheme="upwind1"),
    (tag="nonhydro_tvd2", use_hydrostatic=false, momentum_advection_scheme="tvd2", transport_advection_scheme="tvd2"),
)
wd_variant_kwargs(v) = (use_hydrostatic=v.use_hydrostatic, momentum_advection_scheme=v.momentum_advection_scheme,
    transport_advection_scheme=v.transport_advection_scheme)

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
        @test_throws ArgumentError Model(wd_config(nx=20, nz=4, momentum_advection_scheme="weno5"))
        @test_throws ArgumentError Model(wd_config(nx=20, nz=4, periodic_x=true, pressure_solver_method="sparse"))
        @test_throws ArgumentError Model(wd_config(nx=20, nz=4, bed_level=fill(-0.1, 20)))
        # a single layer (1D), a bed that is not a whole number of cells, the non-hydrostatic mode and tvd2 are accepted
        model = Model(wd_config(nx=20, nz=1, depth=2.0, bed_level=collect(range(0.0, 2.5, length=20))))
        @test model isa Model
        @test Model(wd_config(nx=20, nz=4; wd_variant_kwargs(WD_VARIANTS[end])...)) isa Model
    end

    @testset "test_verify_wd_cut_cell_output_masking" begin
        cfg = wd_config(nx=2, nz=2, depth=1.0)
        cfg.bed_level = [0.25, 0.0]
        model = Model(cfg)
        initialize_water_level!(model, [0.4, 0.0]; rho=1000.0)
        field = ones(cfg.nz, cfg.nx)
        masked = SweTransport2DV._blank_solid_cells(field, cfg; eta=model.state.eta)
        @test isfinite(masked[1, 1])
        @test isnan(masked[2, 1])
        @test all(isnan, masked[:, 2])
    end

    @testset "test_verify_wd_boundary_profiles_and_wet_flux" begin
        cfg = wd_config(nx=2, nz=1, depth=2.0, dt=0.001)
        cfg.bed_level = fill(0.2, cfg.nx)
        model = Model(cfg)
        initialize_water_level!(model, 0.7; rho=1000.0)
        set_x_inflow_outflow!(model, t -> 1.0)
        set_x_inflow_outflow!(model, [1.0]; open_right=false)
        @test model.wetdry.left_velocity === nothing
        step_forward!(model)
        @test isapprox(compute_diagnostics(model)["boundary_flux"], -0.5; atol=1e-12)

        vector_callback_model = Model(wd_config(nx=4, nz=2))
        set_x_inflow_outflow!(vector_callback_model, t -> [1.0])
        @test_throws ArgumentError step_forward!(vector_callback_model)
        right_profile_model = Model(wd_config(nx=4, nz=2))
        set_x_inflow_outflow!(right_profile_model, t -> 0.0; right_profile=[0.0])
        @test_throws ArgumentError step_forward!(right_profile_model)
    end

    @testset "test_verify_wd_float32_dry_columns" begin
        cfg = wd_config(nx=6, nz=2, depth=1.0, dt=0.001)
        cfg.bed_level = fill(0.2, cfg.nx)
        model = Model(cfg; T=Float32)
        initialize_water_level!(model, [0.2, 0.2, 0.21, 0.25, 0.3, 0.35]; rho=1000.0)
        @test all(>=(0.0), water_depth(model))
        for _ in 1:5
            step_forward!(model)
        end
        @test all(>=(0.0), water_depth(model))
        @test all(isfinite, model.state.eta)
    end

    @testset "test_verify_wd_pressure_components_include_nonhydrostatic_correction" begin
        cfg = wd_config(nx=20, nz=4, depth=1.0, dt=0.001, use_hydrostatic=false)
        model = Model(cfg)
        initialize_water_level!(model, 1.0 .+ 0.02 .* sin.(range(0, pi, length=cfg.nx)); rho=1000.0)
        step_forward!(model)
        p_hydro, p_corr, buoy_w = pressure_buoyancy_components(model)
        @test p_hydro == model.last_p_hydro
        @test p_corr == model.last_p_corr
        @test isapprox(p_hydro .+ p_corr, model.state.p; atol=1e-10, rtol=1e-10)
        @test maximum(abs, p_corr) > 0.0
        @test size(buoy_w) == (cfg.nz - 1, cfg.nx)
    end

    @testset "test_verify_wd_nz1_pressure_diagnostics_output" begin
        mktempdir() do output_dir
            cfg = wd_config(nx=8, nz=1, depth=2.0, nt=1, output_dir=output_dir,
                write_plots=false, write_animation=false, write_pressure_buoyancy_diagnostics=true)
            cfg.bed_level = fill(0.2, cfg.nx)
            model = Model(cfg)
            run!(model)
            @test isfile(joinpath(output_dir, "pressure_buoyancy_diagnostics.png"))
            @test isfile(joinpath(output_dir, "pressure_buoyancy_diagnostics.npz"))
        end
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

    @testset "test_verify_wd_lake_at_rest_$(label)_$(v.tag)" for (label, nz, stratified) in
            (("1d", 1, false), ("2dv", 10, false), ("2dv_stratified", 10, true)), v in WD_VARIANTS
        NSTEPS = 500                 # -
        STILL_LEVEL_FRACTION = 0.8   # -, still water level / grid depth
        cfg = wd_config(length=50.0, depth=2.0, nx=200, nz=nz, dt=0.05; wd_variant_kwargs(v)...)
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
    function sloshing_model(nz, v)
        cfg = wd_config(length=40.0, depth=1.5, nx=160, nz=nz, dt=0.02; wd_variant_kwargs(v)...)
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

    @testset "test_verify_wd_volume_and_constituent_conservation_$(label)_$(v.tag)" for (label, nz) in (("1d", 1), ("2dv", 10)),
            v in WD_VARIANTS
        NSTEPS = 1000   # -, 20 s, more than half a sloshing period
        model = sloshing_model(nz, v)
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

    # Depth positivity (docs/depth_positivity.md): the free-surface solve with the non-negative volume function V(eta)
    # (Casulli 2009) keeps every depth >= 0 for any time step. Dam break onto a dry bed and beach with a time step far
    # beyond the advective CFL limit, where the linear semi-implicit solve would drain columns below their bed: the
    # Newton iteration of the nonlinear solve is needed (more than one iteration), the depth never becomes negative and
    # volume and mass stay conserved.
    @testset "test_verify_wd_depth_positivity_large_time_step_$(label)_$(v.tag)" for (label, nz) in (("1d", 1), ("2dv", 10)),
            v in WD_VARIANTS
        CFL_TARGET = 6.0          # -, advective Courant number 2 c0 dt / dx; from about 6 on, columns fall dry within the solve in 1D and 2DV
        NSTEPS = 200              # -
        CLAMP_REL_MAX = 1e-14     # -, volume added by setting round-off negative depths to zero, relative to the total
        cfg = wd_config(length=40.0, depth=1.2, nx=80, nz=nz; wd_variant_kwargs(v)...)
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
        for _ in 1:NSTEPS
            step_forward!(model)
            min_depth = min(min_depth, minimum(cfg.depth .+ model.state.eta .- model.wetdry.bed))
        end
        diag = compute_diagnostics(model)
        @test model.wetdry.newton_iterations_max > 1   # columns fell dry within the solve
        @test min_depth >= 0.0
        @test model.wetdry.clamped_volume <= CLAMP_REL_MAX * diag0["water_volume"]
        @test abs(diag["water_volume"] - diag0["water_volume"]) <= TOL_VOLUME_REL * diag0["water_volume"]
        @test abs(diag["constituent_mass"] - diag0["constituent_mass"]) <= TOL_MASS_REL * diag0["constituent_mass"]
        @test all(isfinite, model.state.u)
    end

    # Operator-level manufactured solution for the momentum advection (docs/TESTS_TECHNICAL_DOCUMENTATION.md, section
    # "Wetting and drying: depth-uniform velocity"): a depth-uniform u(x) under a surface zeta(x) that crosses several
    # layer interfaces, with the interface fluxes W from the discrete layer continuity. Hydrostatic theory gives the same
    # tendency -u du/dx in every layer; the discrete tendency must converge to it at first order in the maximum norm in
    # every layer, as in 1D. (The layer-wise weight qbar_k/hbar_k had O(1) errors in thinly cut surface layers that did
    # not converge: 0.46, 0.87, 0.39, 0.17 m/s^2 for nx = 200 ... 1600 against |u du/dx| <= 0.27 m/s^2.)
    MMS_ORDER_MIN = 0.8          # -, observed order of the maximum error per grid halving (first-order scheme)
    MMS_2DV_VS_1D_MAX = 2.0      # -, the 2DV maximum error may exceed the 1D upwind1 one at most by this factor
    MMS_LAYER_MIN = 1e-3         # m, face layers thinner than this are left out (their weight in the momentum is nil)
    mms_zeta(x) = 0.6 + 0.3 * (1 - tanh((x - 50) / 15))    # m, from 1.2 m down to 0.6 m: crosses 6 interfaces
    mms_u(x) = 1.0 + 2.0 * tanh((x - 50) / 15)             # m/s, changes sign: inflow from both sides occurs
    mms_dudx(x) = 2.0 / 15 * (1 - tanh((x - 50) / 15)^2)   # 1/s

    function mms_advection_error(nx, nz, second)
        cfg = wd_config(length=100.0, depth=1.2, nx=nx, nz=nz, dt=0.001)
        model = Model(cfg)
        wd = model.wetdry
        dx = model.dx
        xc = wd_x_centres(cfg)
        xf = [(f - 1) * dx for f in 1:(nx + 1)]
        model.state.eta .= mms_zeta.(xc) .- cfg.depth
        for f in 2:nx, k in 1:nz
            model.state.u[k, f] = mms_u(xf[f])
        end
        SweTransport2DV._wd_cell_thicknesses!(wd.h, model, wd)
        SweTransport2DV._wd_face_thicknesses!(model, wd)
        model.state.u .*= wd.hf .> 0.0
        wd.q .= wd.hf .* model.state.u
        fill!(model.state.w, 0.0)
        for i in 1:nx
            zeta_t = -(sum(@view wd.q[:, i + 1]) - sum(@view wd.q[:, i])) / dx   # m/s, column continuity
            s = findlast(>(0.0), @view wd.h[:, i])                               # the surface layer takes the change
            for k in 1:nz
                model.state.w[k + 1, i] = model.state.w[k, i] - ((k == s ? zeta_t : 0.0) + (wd.q[k, i + 1] - wd.q[k, i]) / dx)
            end
        end
        err = 0.0
        for f in 3:(nx - 1), k in 1:nz
            wd.hf[k, f] > MMS_LAYER_MIN || continue
            err = max(err, abs(SweTransport2DV._wd_advection_tendency(model, wd, k, f, second) + mms_u(xf[f]) * mms_dudx(xf[f])))
        end
        return err
    end

    @testset "test_verify_wd_advection_depth_uniform_mms_$(scheme)" for scheme in ("upwind1", "tvd2")
        second = scheme == "tvd2"
        nxs = (100, 200, 400, 800)
        # WHY the 1D upwind1 error as reference for both schemes: at faces where a layer ends, tvd2 falls back to first order
        # (consistent stencils), so its 2DV maximum error is first order, while 1D tvd2 never falls back
        e1 = [mms_advection_error(nx, 1, false) for nx in nxs]
        e2 = [mms_advection_error(nx, 12, second) for nx in nxs]
        for j in 2:length(nxs)
            @test log2(e2[j - 1] / e2[j]) >= MMS_ORDER_MIN
            @test e2[j] <= MMS_2DV_VS_1D_MAX * e1[j]
        end
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

    RITTER_LAYER_DEPTH_MIN = 1e-3    # m, layers thinner than this are left out of the vertical velocity spread
    # m/s, accepted difference between the layer velocities (hydrostatic theory: 0; measured at t = 5 s, dx = 0.25 m:
    # 0.001 m/s upwind1, 0.003 m/s tvd2, of a largest velocity of about 5 m/s; 0.29 m/s before the column-consistent
    # advection weights, docs/TESTS_TECHNICAL_DOCUMENTATION.md)
    RITTER_LAYER_SPREAD_MAX = 0.01

    # Largest difference (m/s) between the layer velocities of a face (layers thicker than RITTER_LAYER_DEPTH_MIN);
    # hydrostatic theory keeps an initially depth-uniform velocity depth-uniform.
    function layer_velocity_spread(model)
        spread = 0.0
        for f in 2:model.config.nx
            lo, hi = Inf, -Inf
            for k in 1:model.config.nz
                if model.wetdry.hf[k, f] > RITTER_LAYER_DEPTH_MIN
                    lo, hi = min(lo, model.state.u[k, f]), max(hi, model.state.u[k, f])
                end
            end
            hi > lo && (spread = max(spread, hi - lo))
        end
        return spread
    end

    function run_ritter(nx, nz; v=WD_VARIANTS[1])
        grid_depth = 1.2 * RITTER_H0
        cfg = wd_config(length=100.0, depth=grid_depth, nx=nx, nz=nz, dt=0.005 * 200 / nx; wd_variant_kwargs(v)...)
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
        @test layer_velocity_spread(model) <= RITTER_LAYER_SPREAD_MAX
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
    function run_carrier_greenspan(nx, nz, dt; v=WD_VARIANTS[1])
        cg = CarrierGreenspanWave(amplitude=CG_AMPLITUDE, slope=CG_SLOPE, length_scale=CG_LENGTH)
        T = carrier_greenspan_time_scale(cg)
        amplitude = cg.amplitude / 4 * cg.slope * cg.length_scale   # m, vertical runup amplitude (linear)
        cfg = wd_config(length=CG_LENGTH + CG_ONSHORE, depth=1.5, nx=nx, nz=nz, dt=dt; wd_variant_kwargs(v)...)
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

    variant(tag) = only(v for v in WD_VARIANTS if v.tag == tag)

    # Measured at dx = 0.25 m: 1D L1 0.51 % (upwind1 0.91 %), front 5.6 % (11.3 %); 2DV L1 0.54 % (0.91 %), front 6.6 %
    # (11.3 %), layer spread 0.003 m/s (0.001 m/s).
    @testset "test_validate_wd_ritter_tvd2_improves_on_upwind1" begin
        for nz in (1, 12)
            model_up, _, l1_up, front_up = run_ritter(400, nz; v=variant("hydro_upwind1"))
            model_tvd, _, l1_tvd, front_tvd = run_ritter(400, nz; v=variant("hydro_tvd2"))
            @test l1_tvd < l1_up
            @test front_tvd < front_up
            nz > 1 && @test layer_velocity_spread(model_tvd) <= RITTER_LAYER_SPREAD_MAX
        end
    end

    @testset "test_validate_wd_ritter_dry_dam_break_2dv_$(tag)" for tag in ("nonhydro_upwind1", "nonhydro_tvd2")
        model, h, l1, front_error = run_ritter(400, 12; v=variant(tag))
        @test l1 <= RITTER_L1_REL_MAX
        @test front_error <= RITTER_FRONT_REL_MAX
        @test all(isfinite, model.state.u)
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

    # The shoreline improves slightly with tvd2 (14.3 mm vs 14.6 mm at dx = 0.1 m); the surface error away from the shore
    # does not (3.5 mm vs 3.3 mm): both are dominated by the wet/dry front, where tvd2 falls back to first order.
    @testset "test_validate_wd_carrier_greenspan_runup_tvd2" begin
        shore_up, _, _ = run_carrier_greenspan(240, 1, 0.02; v=variant("hydro_upwind1"))
        shore_tvd, eta_tvd, amplitude = run_carrier_greenspan(240, 1, 0.02; v=variant("hydro_tvd2"))
        @test shore_tvd < shore_up
        @test eta_tvd <= CG_ETA_REL_MAX * amplitude
    end

    @testset "test_validate_wd_carrier_greenspan_runup_2dv_nonhydro" begin
        # a long wave (k H << 1): the non-hydrostatic model must stay as close to the shallow-water solution
        shore, eta, amplitude = run_carrier_greenspan(480, 15, 0.01; v=variant("nonhydro_tvd2"))
        @test shore <= CG_SHORE_REL_MAX * amplitude
        @test eta <= CG_ETA_REL_MAX * amplitude
    end

    # ---- Non-hydrostatic: dispersion of a standing surface wave --------------------------------------------------
    # Same domain as test_validate_nonhydrostatic_dirichlet_surface_wave_matches_full_dispersion (k H = pi/2, where the
    # shallow-water and the full relation omega^2 = g k tanh(k H) differ by more than 25 %).
    NH_DISPERSION_REL_MAX = 0.02    # -, accepted relative period error on the finer grid
    NH_DISPERSION_GAP_MIN = 0.25    # -, relative difference of the two periods that the domain must resolve
    NH_WAVE_AMPLITUDE = 4e-5        # m, linear regime

    @testset "test_validate_wd_nonhydro_standing_wave_full_dispersion_$(scheme)" for scheme in ("upwind1", "tvd2")
        L, H, g = 8.0, 4.0, 9.81
        k_wave = pi / L
        period_shallow = 2pi / (k_wave * sqrt(g * H))
        period_exact = 2pi / sqrt(g * k_wave * tanh(k_wave * H))
        @test (period_exact - period_shallow) / period_shallow > NH_DISPERSION_GAP_MIN

        function wd_wave_period(nx, nz, dt)
            cfg = wd_config(length=L, depth=H, nx=nx, nz=nz, dt=dt, use_hydrostatic=false,
                momentum_advection_scheme=scheme, transport_advection_scheme=scheme)
            model = Model(cfg)
            x = wd_x_centres(cfg)
            initialize_water_level!(model, H .+ NH_WAVE_AMPLITUDE .* cos.(pi .* x ./ L); rho=1000.0)
            # first zero crossing of eta at the left wall: a quarter period
            t_prev, e_prev = 0.0, model.state.eta[1]
            for n in 1:round(Int, 0.6 * period_exact / dt)
                step_forward!(model)
                e = model.state.eta[1]
                if e <= 0.0
                    t = n * dt
                    return 4.0 * (t_prev + e_prev / (e_prev - e) * (t - t_prev))
                end
                t_prev, e_prev = n * dt, e
            end
            return NaN
        end

        period_coarse = wd_wave_period(40, 20, 0.004)
        period_fine = wd_wave_period(80, 40, 0.002)
        err_coarse = abs(period_coarse - period_exact) / period_exact
        err_fine = abs(period_fine - period_exact) / period_exact
        @test err_fine < err_coarse
        @test err_fine < NH_DISPERSION_REL_MAX
        @test abs(period_fine - period_exact) < abs(period_fine - period_shallow)
    end

    # Deep water: k H = 3, where the shallow-water period is 42 % too short (measured: 1.9 % error on 40 x 40 cells).
    NH_DEEP_KH = 3.0              # -
    NH_DEEP_REL_MAX = 0.03        # -, accepted relative period error
    @testset "test_validate_wd_nonhydro_standing_wave_deep_water" begin
        H, g = 1.0, 9.81
        L = pi * H / NH_DEEP_KH
        k_wave = pi / L
        period_exact = 2pi / sqrt(g * k_wave * tanh(k_wave * H))
        cfg = wd_config(length=L, depth=H, nx=40, nz=40, dt=period_exact / 400, use_hydrostatic=false)
        model = Model(cfg)
        initialize_water_level!(model, H .+ NH_WAVE_AMPLITUDE .* cos.(pi .* wd_x_centres(cfg) ./ L); rho=1000.0)
        t_prev, e_prev, period = 0.0, model.state.eta[1], NaN
        for n in 1:round(Int, 0.6 * period_exact / cfg.dt)
            step_forward!(model)
            e = model.state.eta[1]
            if e <= 0.0
                period = 4.0 * (t_prev + e_prev / (e_prev - e) * (n * cfg.dt - t_prev))
                break
            end
            t_prev, e_prev = n * cfg.dt, e
        end
        @test abs(period - period_exact) / period_exact < NH_DEEP_REL_MAX
    end

    # ---- Solitary wave runup on a plane beach (Synolakis 1987) ----------------------------------------------------
    # Non-breaking solitary wave, H/d = 0.019 on a 1:19.85 beach: maximum runup R/d = 2.831 sqrt(cot beta) (H/d)^(5/4)
    # (0.0890 m for d = 1 m). Measured (experiments/wetting_drying, mode solitary): 1D 0.888 and 0.938 of the law at
    # dx = 0.1 and 0.05 m, converging at first order to 1.00 at dx = 0.0125 m; 2DV non-hydrostatic 0.871 at dx = 0.1 m
    # (dispersion lowers the runup slightly).
    SOL_DEPTH = 1.0               # m, offshore depth d
    SOL_COT_BETA = 19.85          # -
    SOL_HEIGHT = 0.019            # -, H/d
    SOL_RUNUP_REL_MAX = 0.10      # -, accepted runup error at dx = 0.05 m (1D)
    SOL_RUNUP_NH_REL_MAX = 0.15   # -, accepted runup error at dx = 0.1 m (2DV non-hydrostatic)
    SOL_SHORE_DEPTH = 1e-3        # m, depth that marks the numerical shoreline
    SOL_T_END = 22.0              # s, past the maximum runup (about 18.5 s)

    function solitary_runup(dx, nz; kwargs...)
        offshore, onshore, still = 60.0, 5.0, 1.0   # m: left wall to still shoreline, dry beach, still level
        nx = round(Int, (offshore + onshore) / dx)
        cfg = wd_config(; length=offshore + onshore, depth=1.2, nx=nx, nz=nz, dt=0.1 * dx, kwargs...)
        x = wd_x_centres(cfg)
        x_toe = offshore - SOL_DEPTH * SOL_COT_BETA
        cfg.bed_level = [max(0.0, still - SOL_DEPTH + (xi - x_toe) / SOL_COT_BETA) for xi in x]
        model = Model(cfg)
        H = SOL_HEIGHT * SOL_DEPTH
        gamma = sqrt(3H / (4SOL_DEPTH^3))
        x1 = x_toe - acosh(sqrt(20.0)) / gamma
        eta(xx) = H * sech(gamma * (xx - x1))^2
        initialize_water_level!(model, [still + eta(xi) for xi in x]; rho=1000.0)
        c = sqrt(cfg.g * (SOL_DEPTH + H))
        for f in 2:nx, k in 1:nz
            model.state.u[k, f] = c * eta((f - 1) * model.dx) / (SOL_DEPTH + eta((f - 1) * model.dx))
        end
        runup = -Inf
        for _ in 1:round(Int, SOL_T_END / cfg.dt)
            step_forward!(model)
            i = findlast(>(SOL_SHORE_DEPTH), water_depth(model))
            runup = max(runup, cfg.depth + model.state.eta[i] - still)
        end
        @test all(isfinite, model.state.u)
        return runup
    end

    @testset "test_validate_wd_solitary_wave_runup" begin
        law = 2.831 * sqrt(SOL_COT_BETA) * SOL_HEIGHT^(5 / 4) * SOL_DEPTH
        err_coarse = abs(solitary_runup(0.1, 1) - law) / law
        err_fine = abs(solitary_runup(0.05, 1) - law) / law
        @test err_fine <= SOL_RUNUP_REL_MAX
        @test err_fine < err_coarse
        err_nh = abs(solitary_runup(0.1, 12; use_hydrostatic=false, momentum_advection_scheme="tvd2", transport_advection_scheme="tvd2") - law) / law
        @test err_nh <= SOL_RUNUP_NH_REL_MAX
    end
end
