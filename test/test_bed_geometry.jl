# Variable bed geometry (issue #3): staircase bed masks, input checks and the masked model paths.
# Run with the full suite (runtests.jl) or on its own: julia --project=. test/test_bed_geometry.jl

include(joinpath(@__DIR__, "test_common.jl"))
include(joinpath(@__DIR__, "../experiments/lock_exchange_obstacle/cases.jl"))

@testset "Bed geometry" begin
    @testset "test_verify_bed_geometry_masks_and_checks" begin
        nz, nx, dz = 10, 12, 0.1                     # -, -, m
        bed = zeros(nx)
        bed[6:7] .= 0.3                              # m, obstacle three cells high in columns 6-7
        geo = BedGeometry(bed, nz, nx, dz)

        @test geo.has_bed && geo.any_solid
        @test geo.k_bed == [1, 1, 1, 1, 1, 4, 4, 1, 1, 1, 1, 1]
        @test count(geo.wet_cell) == nz * nx - 2 * 3
        @test geo.bed_face_level[6] ≈ 0.3 && geo.bed_face_level[7] ≈ 0.3 && geo.bed_face_level[8] ≈ 0.3
        @test geo.k_bed_face[6] == 4 && geo.k_bed_face[5] == 1

        # face 6 (column 5 | obstacle) and face 8 (obstacle | column 8) are walls in rows 1-3: inactive for u, but
        # they take part in the stencils; face 7 lies inside the obstacle.
        @test !any(geo.u_open[1:3, 6]) && all(geo.u_active[1:3, 6]) && geo.u_open[4, 6]
        @test !any(geo.u_open[1:3, 8]) && all(geo.u_active[1:3, 8])
        @test !any(geo.u_active[1:3, 7]) && geo.u_open[4, 7]
        @test geo.u_open[1, 1] && geo.u_open[1, nx + 1]   # outer faces next to wet cells

        # w: the bed face (k = k_bed) is active with w = 0, faces above are open, faces below are inactive
        @test geo.w_active[4, 6] && !geo.w_open[4, 6] && geo.w_open[5, 6] && geo.w_open[nz + 1, 6]
        @test !any(geo.w_active[1:3, 6])
        @test geo.w_active[1, 1] && !geo.w_open[1, 1]     # flat column: bottom face is the bed face

        zero_bed = BedGeometry(zeros(nx), nz, nx, dz)
        @test zero_bed.has_bed && !zero_bed.any_solid
        @test all(zero_bed.wet_cell) && all(zero_bed.u_open) && all(zero_bed.u_active)
        @test !BedGeometry(nothing, nz, nx, dz).has_bed

        @test_throws ArgumentError BedGeometry(fill(0.15, nx), nz, nx, dz)          # not a whole number of cells
        @test_throws ArgumentError BedGeometry(fill(0.9, nx), nz, nx, dz)           # fewer than two wet cells
        @test_throws ArgumentError BedGeometry(fill(-0.1, nx), nz, nx, dz)          # below the grid bottom
        @test_throws ArgumentError BedGeometry(zeros(nx - 1), nz, nx, dz)           # wrong length
        @test_throws ArgumentError BedGeometry(zeros(nx), nz, nx, dz; periodic_x=true)

        cfg = ModelConfig(nx=nx, nz=nz, length=1.2, depth=1.0, bed_level=bed)
        model = Model(cfg)
        @test model.geometry.k_bed == geo.k_bed
        cfg2 = copy_config(cfg)
        @test cfg2.bed_level == bed && cfg2.bed_level !== cfg.bed_level
        @test copy_config(ModelConfig()).bed_level === nothing

        float32_model = Model(ModelConfig(nx=nx, nz=20, length=1.2, depth=4.0, bed_level=fill(1.0, nx)); T=Float32)
        @test float32_model.geometry.k_bed == fill(6, nx)

        # plots leave the bed blank (NaN), flat beds are passed through unchanged
        field = fill(1000.0, nz, nx)
        blanked = SweTransport2DV._blank_solid_cells(field, cfg)
        @test all(isnan, blanked[.!geo.wet_cell]) && all(==(1000.0), blanked[geo.wet_cell])
        @test SweTransport2DV._blank_solid_cells(field, ModelConfig(nx=nx, nz=nz)) === field
    end

    @testset "test_verify_bed_mass_and_surface_pressure_accounting" begin
        bed = zeros(12)
        bed[6:7] .= 0.6
        model = Model(ModelConfig(nx=12, nz=20, length=12.0, depth=4.0, rho_left=1000.0, rho_right=1000.0, bed_level=bed))
        @test model.initial_constituent_mass ≈ 46.8 * 1000.0

        fill!(model.state.rho, 1000.0)
        refresh_density_metadata!(model)
        model.state.rho[.!model.geometry.wet_cell] .= 50_000.0
        @test compute_diagnostics(model)["constituent_mass"] ≈ 46.8 * 1000.0
        model.state.eta .= 0.1
        @test SweTransport2DV._constituent_mass(model) ≈ 48.0 * 1000.0

        fill!(model.state.p, 0.0)
        model.state.eta[5] = 0.1
        model.state.eta[8] = -0.1
        expected_force = model.config.rho0 * model.config.g * 0.2 * 0.6
        @test leo_obstacle_force(model) ≈ expected_force
    end

    BED_EQUIVALENCE_TOL_U = 1e-10        # m/s, raised bed vs shallower domain (round-off from summation order)
    BED_EQUIVALENCE_TOL_RHO = 1e-9       # kg/m^3
    BED_EQUIVALENCE_TOL_ETA = 1e-11      # m
    BED_VOLUME_TOL = 1e-8                # m^2, water-volume drift over a run (same as the legacy tests)
    SEICHE_BED_ORDER_MIN = 1.6           # -, observed order of the second-order free-surface scheme
    SEICHE_BED_ORDER_MAX = 2.4
    SEICHE_BED_REL_ERROR_MAX = 5e-4      # -, relative period error on the finer grid (measured 2.7e-4)

    # lock-exchange settings shared by the model-level tests below
    bed_test_config(; kwargs...) = ModelConfig(; nx=60, nz=20, dt=0.005, use_hydrostatic=true, transport_advection_scheme="tvd2",
        rayleigh_damping=0.0, apply_froude_limiter=false, nu=1e-4, write_plots=false, write_animation=false,
        write_pressure_buoyancy_diagnostics=false, kwargs...)

    @testset "test_verify_invalid_bed_velocities_are_masked_before_advection" begin
        bed = zeros(60)
        bed[28:33] .= 1.2
        cfg = bed_test_config(bed_level=bed)
        reference = Model(copy_config(cfg))
        perturbed = Model(copy_config(cfg))
        geo = perturbed.geometry
        perturbed.state.u[.!geo.u_open] .= 10.0
        perturbed.state.w[.!geo.w_open] .= 10.0

        step_forward!(reference)
        step_forward!(perturbed)
        @test perturbed.state.u == reference.state.u
        @test perturbed.state.w == reference.state.w
        @test perturbed.state.rho == reference.state.rho
        @test perturbed.state.eta == reference.state.eta
    end

    @testset "test_verify_flat_bed_reproduces_legacy_results_hydrostatic" begin
        # An explicit zero bed runs through all masked code paths; it must reproduce the legacy flat-bed results bit
        # for bit (with and without the variable top layer).
        for variable_top in (true, false)
            legacy = Model(bed_test_config(variable_top_layer=variable_top))
            masked = Model(bed_test_config(variable_top_layer=variable_top, bed_level=zeros(60)))
            for _ in 1:200
                step_forward!(legacy)
                step_forward!(masked)
            end
            @test masked.state.u == legacy.state.u
            @test masked.state.w == legacy.state.w
            @test masked.state.rho == legacy.state.rho
            @test masked.state.eta == legacy.state.eta
        end
    end

    @testset "test_verify_raised_flat_bed_equals_shallower_domain_hydrostatic" begin
        # A uniform bed of d = 1 m (5 cells of 0.2 m) under a 4 m domain is the same problem as a 3 m domain with 15
        # layers. Checks the masks, the per-face barotropic depth and the w reconstruction from the bed together.
        H, d, nz = 4.0, 1.0, 20
        nb = 5
        for variable_top in (true, false)
            raised = Model(bed_test_config(nz=nz, depth=H, variable_top_layer=variable_top, bed_level=fill(d, 60)))
            shallow = Model(bed_test_config(nz=nz - nb, depth=H - d, variable_top_layer=variable_top))
            for _ in 1:200
                step_forward!(raised)
                step_forward!(shallow)
            end
            @test maximum(abs, raised.state.u[(nb + 1):end, :] .- shallow.state.u) < BED_EQUIVALENCE_TOL_U
            @test maximum(abs, raised.state.rho[(nb + 1):end, :] .- shallow.state.rho) < BED_EQUIVALENCE_TOL_RHO
            @test maximum(abs, raised.state.eta .- shallow.state.eta) < BED_EQUIVALENCE_TOL_ETA
            @test all(==(0.0), raised.state.u[1:nb, :])
            @test all(==(0.0), raised.state.w[1:(nb + 1), :])
        end
    end

    @testset "test_verify_rest_state_over_obstacle_hydrostatic" begin
        # Horizontally stratified water at rest over an obstacle stays exactly at rest (no spurious currents at the
        # obstacle walls): the hydrostatic pressure at a z-level does not depend on the bed below.
        bed = zeros(60)
        bed[28:33] .= 1.2                       # m, obstacle 6 cells high
        for variable_top in (true, false)
            model = Model(bed_test_config(variable_top_layer=variable_top, bed_level=bed, nu=0.0))
            zc = [(k - 0.5) * model.dz for k in 1:model.config.nz]
            for i in 1:60, k in 1:model.config.nz
                model.state.rho[k, i] = 1000.0 + 0.5 * (1.0 - zc[k] / model.config.depth)
            end
            refresh_density_metadata!(model)
            for _ in 1:400
                step_forward!(model)
            end
            @test maximum(abs, model.state.u) == 0.0
            @test maximum(abs, model.state.w) == 0.0
            @test maximum(abs, model.state.eta) == 0.0
        end
    end

    @testset "test_verify_lock_exchange_over_obstacle_conserves_hydrostatic" begin
        bed = zeros(60)
        bed[28:33] .= 1.2
        for variable_top in (true, false)
            model = Model(bed_test_config(variable_top_layer=variable_top, bed_level=bed))
            d0 = compute_diagnostics(model)
            for _ in 1:2000
                step_forward!(model)
            end
            d1 = compute_diagnostics(model)
            geo = model.geometry
            @test abs(d1["water_volume"] - d0["water_volume"]) < BED_VOLUME_TOL
            @test abs(d1["constituent_mass"] - d0["constituent_mass"]) < TOL_MODEL_IDENTITY * abs(d0["constituent_mass"])
            @test minimum(model.state.rho[geo.wet_cell]) >= model.rho_min_init - TOL_IDENTITY
            @test maximum(model.state.rho[geo.wet_cell]) <= model.rho_max_init + TOL_IDENTITY
            @test all(==(0.0), model.state.u[.!geo.u_open])
            @test maximum(abs, model.state.u) > 0.05    # contrast: the gravity current is running
        end
    end

    @testset "test_validate_seiche_over_raised_bed" begin
        # Analytical solution: linear standing wave over a uniformly raised bed of height d,
        # eta = A cos(pi x/L) cos(omega t), period T = 2L / sqrt(g (H - d)). Zero crossing of eta in cell 1 at
        # omega t = pi/2 (as in test_verify_variable_top_layer_seiche_period_and_convergence).
        L, H, d, A, dt = 20.0, 4.0, 1.0, 1e-4, 0.005
        g = 9.81
        T_exact = 2.0 * L / sqrt(g * (H - d))
        function seiche_period(nx::Int)
            cfg = ModelConfig(nx=nx, nz=20, length=L, depth=H, dt=dt, use_hydrostatic=true, variable_top_layer=true,
                rayleigh_damping=0.0, apply_froude_limiter=false, momentum_advection_scheme="upwind1", bed_level=fill(d, nx))
            model = Model(cfg)
            x = [(i - 0.5) * model.dx for i in 1:nx]
            model.state.eta .= A .* cos.(pi .* x ./ L)
            fill!(model.state.rho, 1000.0)
            refresh_density_metadata!(model)
            t_prev, e_prev = 0.0, model.state.eta[1]
            for n in 1:round(Int, 0.5 * T_exact / dt)
                step_forward!(model)
                e = model.state.eta[1]
                if e <= 0.0
                    return 4.0 * (t_prev + e_prev / (e_prev - e) * (n * dt - t_prev))
                end
                t_prev, e_prev = n * dt, e
            end
            return NaN
        end
        errors = [abs(seiche_period(nx) - T_exact) / T_exact for nx in (20, 40)]
        @test errors[2] < SEICHE_BED_REL_ERROR_MAX
        @test SEICHE_BED_ORDER_MIN < log2(errors[1] / errors[2]) < SEICHE_BED_ORDER_MAX
    end

    BED_REST_TOL_NONHYDRO = 1e-9     # m/s and m: round-off of the projection in a resting, stratified state
    BED_DIVERGENCE_TOL = 1e-10       # 1/s, divergence left by the masked projection (velocity scale u/dx ~ 0.1 1/s)
    POISSON_ORDER_MIN = 1.8          # -, observed order of the 5-point Laplacian (expected 2)
    POISSON_ORDER_MAX = 2.2

    @testset "test_verify_flat_bed_reproduces_legacy_results_nonhydrostatic" begin
        # Zero bed (no solid cells): the rectangular Poisson problem is kept, so both pressure solvers reproduce the
        # legacy results bit for bit.
        for method in ("spectral", "sparse")
            legacy = Model(bed_test_config(use_hydrostatic=false, pressure_solver_method=method))
            masked = Model(bed_test_config(use_hydrostatic=false, pressure_solver_method=method, bed_level=zeros(60)))
            for _ in 1:200
                step_forward!(legacy)
                step_forward!(masked)
            end
            @test masked.state.u == legacy.state.u
            @test masked.state.rho == legacy.state.rho
            @test masked.state.eta == legacy.state.eta
        end
    end

    @testset "test_verify_raised_flat_bed_equals_shallower_domain_nonhydrostatic" begin
        # Same comparison as in the hydrostatic mode, now through the masked Poisson problem on the wet cells.
        nb = 5
        raised = Model(bed_test_config(use_hydrostatic=false, pressure_solver_method="sparse", bed_level=fill(1.0, 60)))
        shallow = Model(bed_test_config(use_hydrostatic=false, pressure_solver_method="sparse", nz=15, depth=3.0))
        for _ in 1:200
            step_forward!(raised)
            step_forward!(shallow)
        end
        @test maximum(abs, raised.state.u[(nb + 1):end, :] .- shallow.state.u) < BED_EQUIVALENCE_TOL_U
        @test maximum(abs, raised.state.rho[(nb + 1):end, :] .- shallow.state.rho) < BED_EQUIVALENCE_TOL_RHO
        @test maximum(abs, raised.state.eta .- shallow.state.eta) < BED_EQUIVALENCE_TOL_ETA
    end

    @testset "test_verify_rest_state_and_projection_over_obstacle_nonhydrostatic" begin
        bed = zeros(60)
        bed[28:33] .= 1.2
        # resting stratified state stays at rest (to the round-off of the projection)
        rest = Model(bed_test_config(use_hydrostatic=false, pressure_solver_method="sparse", bed_level=bed, nu=0.0))
        zc = [(k - 0.5) * rest.dz for k in 1:rest.config.nz]
        for i in 1:60, k in 1:rest.config.nz
            rest.state.rho[k, i] = 1000.0 + 0.5 * (1.0 - zc[k] / rest.config.depth)
        end
        refresh_density_metadata!(rest)
        for _ in 1:400
            step_forward!(rest)
        end
        @test maximum(abs, rest.state.u) < BED_REST_TOL_NONHYDRO
        @test maximum(abs, rest.state.w) < BED_REST_TOL_NONHYDRO
        @test maximum(abs, rest.state.eta) < BED_REST_TOL_NONHYDRO

        # lock exchange over the obstacle: the projected velocities are divergence-free in every wet cell
        model = Model(bed_test_config(use_hydrostatic=false, pressure_solver_method="sparse", bed_level=bed))
        d0 = compute_diagnostics(model)
        for _ in 1:2000
            step_forward!(model)
        end
        d1 = compute_diagnostics(model)
        geo = model.geometry
        divergence = div_x_u(model.ws.u_hat, model.dx) .+ div_z_w(model.ws.w_hat, model.dz)
        @test maximum(abs, divergence[geo.wet_cell]) < BED_DIVERGENCE_TOL
        @test abs(d1["water_volume"] - d0["water_volume"]) < BED_VOLUME_TOL
        @test abs(d1["constituent_mass"] - d0["constituent_mass"]) < TOL_MODEL_IDENTITY * abs(d0["constituent_mass"])
        @test all(==(0.0), model.state.u[.!geo.u_open])
        @test maximum(abs, model.state.u) > 0.03    # contrast: the gravity current is running

        # the spectral solver cannot work on a domain with solid cells
        @test_throws ArgumentError step_forward!(Model(bed_test_config(use_hydrostatic=false, bed_level=bed)))
    end

    @testset "test_verify_masked_poisson_manufactured_solution" begin
        # Manufactured solution on the wet cells above a raised flat bed of height d: p = cos(pi x/L) cos(pi (z-d)/(H-d))
        # has zero normal derivative at the side walls, the bed and the top, and Laplacian
        # -pi^2 (1/L^2 + 1/(H-d)^2) p. The masked Neumann problem (Neumann walls at the solid cells, gauge pin at the
        # first wet cell, mean removal over the wet cells) must reproduce it with second-order accuracy.
        L, H, d = 2.0, 1.0, 0.25
        function poisson_error(n::Int)
            nx, nz = 2n, n
            dx, dz = L / nx, H / nz
            geo = BedGeometry(fill(d, nx), nz, nx, dz)
            A = build_poisson_matrix((nz, nx), dx, dz; wet=geo.wet_cell)
            pin = findfirst(vec(permutedims(geo.wet_cell)))
            p_exact = zeros(nz, nx)
            rhs = zeros(nz * nx)
            for k in 1:nz, i in 1:nx
                geo.wet_cell[k, i] || continue
                x, z = (i - 0.5) * dx, (k - 0.5) * dz
                p_exact[k, i] = cos(pi * x / L) * cos(pi * (z - d) / (H - d))
                rhs[(k - 1) * nx + i] = -pi^2 * (1 / L^2 + 1 / (H - d)^2) * p_exact[k, i]
            end
            wet_flat = vec(permutedims(geo.wet_cell))
            rhs[wet_flat] .-= sum(rhs[wet_flat]) / count(wet_flat)
            rhs[pin] = 0.0
            p = permutedims(reshape(A \ rhs, nx, nz))
            p_wet, e_wet = p[geo.wet_cell], p_exact[geo.wet_cell]
            return maximum(abs, (p_wet .- sum(p_wet) / length(p_wet)) .- (e_wet .- sum(e_wet) / length(e_wet)))
        end
        errors = [poisson_error(n) for n in (16, 32, 64)]
        @test POISSON_ORDER_MIN < log2(errors[1] / errors[2]) < POISSON_ORDER_MAX
        @test POISSON_ORDER_MIN < log2(errors[2] / errors[3]) < POISSON_ORDER_MAX
    end
end
