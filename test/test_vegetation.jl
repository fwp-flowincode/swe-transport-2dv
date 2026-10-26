# Vegetation: canopy patches and individual stems, rigid and flexible (issue #8, docs/plan_issue_8_vegetation.md).
# Run with the full suite (runtests.jl) or on its own: julia --project=. test/test_vegetation.jl

include(joinpath(@__DIR__, "test_common.jl"))

# Shared settings, the 1D canopy reference and the reconfiguration helper of the vegetation experiments.
include(joinpath(@__DIR__, "..", "experiments", "vegetation", "cases.jl"))

vegetation_test_config(; kwargs...) = vegetation_case_config(; kwargs...)

@testset "Vegetation" begin
    @testset "test_verify_rigid_patch_drag_per_unit_volume" begin
        # A rigid patch removes 1/2 C_D a u|u| per unit volume below the stem height, the canopy-top cell weighted by the
        # fraction below the stem height and the patch-edge faces by the fraction of their control interval inside.
        NX, NZ = 20, 10
        DX, DZ = 0.5, 0.2                       # m
        STEM_HEIGHT = 0.9                       # m: rows 1-4 full, row 5 half
        X_START, X_END = 2.25, 6.0              # m: face 6 (x = 2.5) and 13 (x = 6.0) at the edges
        CD, B, N = 1.2, 0.01, 300.0             # -, m, stems/m^2
        patch = VegetationPatch(x_start=X_START, x_end=X_END, stem_height=STEM_HEIGHT, stem_diameter=B, stem_density=N, drag_coefficient=CD)
        veg = VegetationModel(; patches=[patch])
        rng = MersenneTwister(8)
        u = 0.3 .* randn(rng, NZ, NX + 1)        # m/s, both signs
        accel = zeros(NZ, NX + 1)
        vegetation_source!(accel, veg, u, DX, DZ, VEG_RHO0)
        a = patch_frontal_area(patch)
        expected = zeros(NZ, NX + 1)
        for i in 1:(NX + 1), k in 1:NZ
            x = (i - 1) * DX
            w_x = clamp((min(x + DX / 2, X_END) - max(x - DX / 2, X_START)) / DX, 0.0, 1.0)
            w_z = clamp((STEM_HEIGHT - (k - 1) * DZ) / DZ, 0.0, 1.0)
            expected[k, i] = -0.5 * CD * a * u[k, i] * abs(u[k, i]) * w_x * w_z
        end
        @test maximum(abs, accel .- expected) < TOL_IDENTITY
        @test accel[5, 8] ≈ 0.5 * expected[5, 8] / 0.5   # row 5 is half covered (consistency of the expectation)
    end

    @testset "test_verify_patch_stiffness_must_be_positive" begin
        # Argument check: a patch needs EI > 0 (Inf for rigid); zero, negative and NaN stiffness are rejected.
        for stiffness in (0.0, -1.0, NaN)
            patch = VegetationPatch(x_start=0.0, x_end=1.0, stem_height=1.0, stem_diameter=0.01,
                stem_density=100.0, bending_stiffness=stiffness)
            @test_throws ArgumentError VegetationModel(; patches=[patch])
        end
    end

    @testset "test_verify_emergent_drag_up_to_top_of_momentum_layers" begin
        # Emergent rigid vegetation is loaded up to the top of the momentum layers and the force is divided by their
        # thickness h (dz below, the top-layer thickness in the top row):
        # - fixed top layer (linear free surface): water level nz dz, top row thickness dz, whatever the surface elevation;
        # - variable top layer: water level (nz-1) dz + h_top_face (individual stems: interpolated to the base), top row
        #   thickness h_top_face.
        # Checked: the stem force 1/2 rho C_D b u² (water level), the volume-integrated sink equals the stem force (Newton's
        # third law with the layer thicknesses), and a patch removes exactly 1/2 C_D a u|u| per unit volume in the top row.
        NX, NZ, DX, DZ = 10, 20, 0.2, 0.05      # -, -, m, m
        U0, CD, B, SPACING = 0.2, 1.0, 0.01, 0.4  # m/s, -, m, m
        eta = [0.015, -0.01, 0.005, 0.0, 0.01, -0.005, 0.003, 0.008, -0.002, 0.004]      # m
        h_top_face = zeros(NX + 1)
        top_layer_face_thickness!(h_top_face, eta, DZ)
        u = fill(U0, NZ, NX + 1)
        for variable_top in (false, true)
            h_faces = variable_top ? h_top_face : nothing
            layer(k, i) = (variable_top && k == NZ) ? h_top_face[i] : DZ
            stem = VegetationStem(base_x=0.63, height=1.3, diameter=B, drag_coefficient=CD, bending_stiffness=Inf,
                lateral_spacing=SPACING)
            veg = VegetationModel(stem)
            accel = zeros(size(u))
            vegetation_source!(accel, veg, u, DX, DZ, VEG_RHO0; h_top_face=h_faces)
            t = 0.63 / DX - 3                   # base between faces 4 and 5
            water_level = variable_top ? (NZ - 1) * DZ + (1 - t) * h_top_face[4] + t * h_top_face[5] : NZ * DZ
            @test isapprox(stem.force_x, 0.5 * VEG_RHO0 * CD * B * U0^2 * water_level; rtol=TOL_IDENTITY)
            sink = sum(-accel[k, i] * VEG_RHO0 * DX * layer(k, i) * SPACING for i in 1:(NX + 1), k in 1:NZ)
            @test isapprox(sink, stem.force_x; rtol=TOL_IDENTITY)

            patch = VegetationPatch(x_start=0.0, x_end=NX * DX, stem_height=2.0, stem_diameter=B, stem_density=100.0,
                drag_coefficient=CD)
            patch_veg = VegetationModel(; patches=[patch])
            vegetation_source!(accel, patch_veg, u, DX, DZ, VEG_RHO0; h_top_face=h_faces)
            rate = 0.5 * CD * patch_frontal_area(patch) * U0
            @test isapprox(accel[NZ, 2], -rate * U0; rtol=TOL_IDENTITY)
            @test isapprox(patch_veg.drag_rate[NZ, 2], rate; rtol=TOL_IDENTITY)
        end
    end

    @testset "test_verify_stem_drag_equals_fluid_momentum_sink" begin
        # Newton's third law for individual flexible stems in a sheared flow: the momentum removed from the fluid (per
        # metre width, over the lateral spacing) equals the stem drag.
        NX, NZ = 30, 20
        DX, DZ = 0.2, 0.05                      # m
        SPACING = 0.4                           # m
        stems = [VegetationStem(base_x=x, height=0.7, diameter=0.008, drag_coefficient=1.0, bending_stiffness=2e-3, lateral_spacing=SPACING)
                 for x in (1.23, 3.0, 4.71)]
        veg = VegetationModel(stems...)
        z = [(k - 0.5) * DZ for k in 1:NZ]
        u = repeat(0.4 .* (z ./ (NZ * DZ)) .^ (1 / 7), 1, NX + 1)
        accel = zeros(NZ, NX + 1)
        vegetation_source!(accel, veg, u, DX, DZ, VEG_RHO0)
        momentum_sink = -sum(accel) * VEG_RHO0 * DX * DZ * SPACING     # N
        @test isapprox(momentum_sink, sum(s.force_x for s in stems); rtol=TOL_IDENTITY)
        @test all(stem_tip_deflection(s) > 0 for s in stems)                    # bent downstream
        @test all(s.force_x < 0.5 * VEG_RHO0 * 1.0 * 0.008 * 0.7 * 0.4^2 for s in stems)   # below the rigid upper bound
    end

    @testset "test_verify_cantilever_small_deflection_second_order" begin
        # Small deflection under a uniform load q = 1/2 rho C_D b U^2: tip deflection q l^4 / (8 EI) (beam theory,
        # e.g. Timoshenko & Gere). Observed relative errors 1.0e-2, 2.5e-3, 6.3e-4 for 10, 20, 40 segments.
        U, B, CD, L, EI = 0.05, 0.01, 1.0, 0.5, 1.0      # m/s, m, -, m, N m^2  (Cauchy number 0.005)
        SEGMENTS = (10, 20, 40)
        ORDER_MIN = 1.9
        q = 0.5 * VEG_RHO0 * CD * B * U^2
        errors = Float64[]
        for nseg in SEGMENTS
            theta = zeros(nseg)
            converged, force, _ = update_stem_posture!(theta, (k, x) -> U; length=L, diameter=B, drag_coefficient=CD,
                bending_stiffness=EI, rho0=VEG_RHO0, dz=1e-3, nz=10_000, water_level=10.0)
            @test converged
            tip = L / nseg * sum(sin, theta)
            push!(errors, abs(tip / (q * L^4 / (8EI)) - 1))
        end
        @test all(log2.(errors[1:end-1] ./ errors[2:end]) .> ORDER_MIN)
    end

    @testset "test_validate_flexible_stem_reconfiguration" begin
        # Reconfiguration number R = F / (1/2 rho C_D b l U^2) (= effective length l_e/l of Luhar & Nepf 2011) against
        # the Cauchy number Ca = rho C_D b U^2 l^3 / (2 EI): R -> 1 for Ca << 1 (rigid) and R ~ Ca^(-1/3) for Ca >> 1
        # (drag ~ U^(4/3); Alben et al. 2002, Gosselin et al. 2010). Observed: R = 0.9997 at Ca = 0.1, local slope -0.334
        # between Ca = 300 and 3000 (80 segments), R Ca^(1/3) = 0.88 at Ca = 1000. Buoyancy keeps the stem upright while
        # Ca < B (B = buoyancy per length l^3 / EI; Luhar & Nepf 2011).
        B, CD, L, EI = 0.01, 1.0, 0.5, 1e-3
        SEGMENTS = 80
        SLOPE_EXPECTED = -1 / 3
        SLOPE_TOL = 0.02
        RIGID_TOL = 1e-3
        velocity_for(ca) = sqrt(2ca * EI / (VEG_RHO0 * CD * B * L^3))
        function reconfiguration(ca; buoyancy=0.0)
            theta = zeros(SEGMENTS)
            u = velocity_for(ca)
            converged, force, _ = update_stem_posture!(theta, (k, x) -> u; length=L, diameter=B, drag_coefficient=CD,
                bending_stiffness=EI, net_buoyancy=buoyancy, rho0=VEG_RHO0, dz=1e-3, nz=10_000, water_level=10.0)
            @test converged
            return force / (0.5 * VEG_RHO0 * CD * B * L * u^2)
        end
        @test abs(reconfiguration(0.1) - 1) < RIGID_TOL
        slope = log(reconfiguration(3000.0) / reconfiguration(300.0)) / log(10.0)
        @test abs(slope - SLOPE_EXPECTED) < SLOPE_TOL
        # buoyancy number B_n = 100 >> Ca = 10: nearly rigid; without buoyancy the same Ca halves the drag
        buoyancy = 100 * EI / L^3
        @test reconfiguration(10.0; buoyancy=buoyancy) > 0.95
        @test reconfiguration(10.0) < 0.6
    end

    @testset "test_verify_elastica_against_independent_reference" begin
        # Single stem in a uniform current: the model's discrete elastica (segments, Newton) against the continuous
        # elastica integrated independently from the tip (RK4, 20000 steps; elastica_reference_uniform in
        # experiments/vegetation/cases.jl). Observed: second order in the number of segments for R and the tip position at
        # all Cauchy numbers; relative R errors at 80 segments 5e-6 (Ca = 1), 4.2e-4 (100), 1.9e-3 (1000), 7.4e-4
        # (Ca = 300, B = 30). The reference gives R Ca^(1/3) = 0.890 at large Ca.
        CASES = ((1.0, 0.0), (100.0, 0.0), (1000.0, 0.0), (300.0, 30.0))    # (Ca, B)
        SEGMENTS = (40, 80)
        ORDER_MIN = 1.8
        R_RTOL = 3e-3                       # -, at 80 segments
        TIP_TOL = 1e-3                      # stem lengths, at 80 segments
        for (cauchy, buoyancy) in CASES
            ref = elastica_reference_uniform(cauchy; buoyancy_number=buoyancy)
            errors_r = Float64[]; errors_x = Float64[]
            for nseg in SEGMENTS
                r, theta = reconfiguration_number(cauchy; n_segments=nseg, buoyancy_number=buoyancy)
                push!(errors_r, abs(r / ref.R - 1))
                push!(errors_x, abs(sum(sin, theta) / nseg - ref.tip_x))
            end
            @test log2(errors_r[1] / errors_r[2]) > ORDER_MIN
            @test log2(errors_x[1] / errors_x[2]) > ORDER_MIN
            @test errors_r[end] < R_RTOL
            @test errors_x[end] < TIP_TOL
        end
    end

    @testset "test_verify_luhar_nepf_2011_published_postures" begin
        # Luhar & Nepf (2011) use the same blade model (drag on the velocity normal to the blade, cos²θ per unit length,
        # buoyancy, elastica). Published values (p. 2006-2007, Fig. 2):
        # (a) neutrally buoyant blade at Ca = 32: effective length l_e/l = 0.30, deflected height h/l = 0.61 (observed
        #     0.2995 and 0.6079 at 80 segments);
        # (b) zero stiffness at B⁻¹Ca = 3.2: the blade stays straight at the angle of their Eq. 12, sinθ = (B⁻¹Ca) cos²θ,
        #     i.e. cos²θ = (sqrt(1 + 4r²) - 1)/(2r²): θ = 58.9°, l_e/l = cos³θ = 0.138, h/l = cosθ = 0.517 (published 59°,
        #     0.14, 0.52). Our model needs EI > 0; at fixed B⁻¹Ca it approaches the limit as B grows, the error falling
        #     about as B^(-1/2) (stiffness boundary layer of thickness ~ l B^(-1/2) at the base): h/l errors 0.079, 0.028,
        #     0.009 for B = 1e2, 1e3, 1e4 (160 segments).
        PUBLISHED_TOL = 0.005               # -, the published values are rounded to two digits
        RATIO = 3.2
        BUOYANCY_NUMBERS = (1e2, 1e3, 1e4)
        REDUCTION_MIN = 2.5                 # -, error reduction per decade of B (B^(-1/2): 3.2)
        LIMIT_TOL = 0.015                   # -, at B = 1e4
        r, theta = reconfiguration_number(32.0; n_segments=80)
        @test abs(r - 0.30) < PUBLISHED_TOL
        @test abs(sum(cos, theta) / 80 - 0.61) < PUBLISHED_TOL
        c = sqrt((sqrt(1 + 4RATIO^2) - 1) / (2RATIO^2))       # cos θ of the straight blade
        @test abs(rad2deg(acos(c)) - 59) < 0.5 && abs(c^3 - 0.14) < PUBLISHED_TOL && abs(c - 0.52) < PUBLISHED_TOL
        errors_h = Float64[]; errors_le = Float64[]
        for bn in BUOYANCY_NUMBERS
            r, theta = reconfiguration_number(RATIO * bn; n_segments=160, buoyancy_number=bn)
            push!(errors_h, abs(sum(cos, theta) / 160 - c))
            push!(errors_le, abs(r - c^3))
        end
        @test all(errors_h[1:end-1] ./ errors_h[2:end] .> REDUCTION_MIN)
        @test all(errors_le[1:end-1] ./ errors_le[2:end] .> REDUCTION_MIN)
        @test errors_h[end] < LIMIT_TOL && errors_le[end] < LIMIT_TOL
    end

    @testset "test_validate_flexible_canopy_against_coupled_reference" begin
        # Flexible submerged canopy in a periodic channel (stem length H/2, a = 4 1/m, EI = 2e-5 N m^2, constant nu):
        # steady velocity profile and stem posture against the independent coupled 1D reference
        # (flexible_canopy_reference: continuous elastica in the velocity along the stem, iterated with the 1D momentum
        # equation). The model starts from the reference state and runs 1000 s (from rest the flexible canopy needs about
        # 6000 s to become steady). Observed: max velocity errors 5.4e-5 and 1.6e-5 m/s, tip errors (x, z) (0.67, 0.60) mm
        # and (0.16, 0.13) mm for nz = 20 and 40 (surface velocity 0.098 m/s, deflected height 0.283 m).
        H, HV, CD, B, N, G, NU, EI = 1.0, 0.5, 1.0, 0.01, 400.0, 2e-4, 1e-3, 2e-5
        U_TOL = 1e-4                        # m/s, nz = 20
        TIP_TOL = 1.5e-3                    # m, nz = 20
        REDUCTION_MIN = 2.0                 # -, error ratio nz = 20 / nz = 40
        T_RUN = 1000.0                      # s
        ref = flexible_canopy_reference(; H=H, stem_length=HV, a=N * B, b=B, cd=CD, ei=EI, G=G, nu=NU, M=500, steps=1000)
        s_ref = collect(range(0, HV, length=length(ref.theta)))
        errors = Dict{Int,NTuple{3,Float64}}()
        for nz in (20, 40)
            cfg = vegetation_test_config(length=4.0, depth=H, nx=8, nz=nz, dt=0.1, nt=round(Int, T_RUN / 0.1), use_hydrostatic=true,
                periodic_x=true, variable_top_layer=false, pressure_solver_method="sparse", nu=NU, body_force_x=G)
            model = uniform_density_model(cfg)
            vegetation = VegetationModel(; patches=[VegetationPatch(x_start=0.0, x_end=cfg.length, stem_height=HV, stem_diameter=B,
                stem_density=N, drag_coefficient=CD, bending_stiffness=EI, n_segments=nz)])
            attach_vegetation!(model, vegetation)
            z = [(k - 0.5) * model.dz for k in 1:nz]
            for i in 1:(cfg.nx + 1), k in 1:nz
                model.state.u[k, i] = linear_interpolation(z[k], ref.z, ref.u)
            end
            s_mid = [(j - 0.5) * HV / nz for j in 1:nz]
            vegetation.patch_theta[1] = repeat([linear_interpolation(s, s_ref, ref.theta) for s in s_mid], 1, cfg.nx + 1)
            for _ in 1:cfg.nt
                step_forward!(model)
            end
            u = vec(mean(model.state.u[:, 1:end-1]; dims=2))
            theta = vegetation.patch_theta[1][:, 1]
            ds = HV / nz
            errors[nz] = (maximum(abs, u .- linear_interpolation.(z, Ref(ref.z), Ref(ref.u))), abs(ds * sum(sin, theta) - ref.tip_x),
                abs(ds * sum(cos, theta) - ref.tip_z))
        end
        @test errors[20][1] < U_TOL
        @test errors[20][2] < TIP_TOL && errors[20][3] < TIP_TOL
        @test all(errors[20][j] / errors[40][j] > REDUCTION_MIN for j in 1:3)
    end

    @testset "test_validate_luhar_nepf_2011_blade_drag" begin
        # Laboratory data: drag of flexible model blades (HDPE: stiffness-dominated; silicone foam: buoyancy-dominated)
        # in uniform flow, Luhar & Nepf (2011), forces read from their Fig. 4 (LN2011_FORCES in experiments/vegetation/
        # cases.jl). Compared: measured forces of at least 5 mN (5 times the load-cell resolution) of the 25 cm HDPE blade
        # and the 5 cm and 25 cm foam blades. The 5 cm HDPE blade is left out: the authors report that their model (the
        # same as ours) overpredicts it, attributing this to a lower drag coefficient of short plates; our model gives
        # 1.22-1.40 times the measured force there. Observed: median deviation 7 %, largest 39 % (25 cm HDPE at 11 cm/s,
        # where the model is also above the data in their Fig. 4); 10 % is the load-cell accuracy.
        FORCE_MIN = 0.005           # N
        MEDIAN_TOL = 0.10           # -
        MAX_TOL = 0.45              # -
        ratios = Float64[]
        for (material, len) in (("HDPE", 0.25), ("foam", 0.05), ("foam", 0.25))
            for (j, u) in enumerate(LN2011_VELOCITIES)
                measured = LN2011_FORCES[(material, len)][j]
                (isnan(measured) || measured < FORCE_MIN) && continue
                push!(ratios, luhar_nepf_blade(material, len, u).force / measured)
            end
        end
        deviations = abs.(ratios .- 1)
        @test length(ratios) >= 15
        @test sort(deviations)[cld(length(deviations), 2)] < MEDIAN_TOL
        @test maximum(deviations) < MAX_TOL
        # the reconstructed blade parameters reproduce their Table 1 (Ca = 320, B = 0.62 for the 25 cm HDPE blade at 32 cm/s)
        check = luhar_nepf_blade("HDPE", 0.25, 0.32)
        @test isapprox(check.cauchy, 320; rtol=0.03) && isapprox(check.buoyancy, 0.62; rtol=0.02)
    end

    @testset "test_validate_emergent_rigid_canopy_uniform_velocity" begin
        # Periodic channel, body force G, emergent rigid patch over the whole channel, free-slip bed: the steady velocity
        # is uniform with 1/2 C_D a U^2 = G (exact). Observed relative error 4e-12 in both modes.
        G, CD, B, N = 1e-3, 1.0, 0.01, 400.0
        REL_TOL = 1e-8
        for use_hydrostatic in (true, false)
            cfg = vegetation_test_config(length=10.0, depth=1.0, nx=20, nz=10, dt=0.1, nt=3000, use_hydrostatic=use_hydrostatic,
                periodic_x=true, variable_top_layer=false, pressure_solver_method="sparse", nu=1e-6, body_force_x=G)
            model = uniform_density_model(cfg)
            attach_vegetation!(model, VegetationModel(; patches=[VegetationPatch(x_start=0.0, x_end=cfg.length, stem_height=2 * cfg.depth,
                stem_diameter=B, stem_density=N, drag_coefficient=CD)]))
            for _ in 1:cfg.nt
                step_forward!(model)
            end
            @test isapprox(mean(model.state.u), sqrt(2G / (CD * N * B)); rtol=REL_TOL)
        end
    end

    @testset "test_validate_submerged_rigid_canopy_profile" begin
        # Submerged rigid canopy (stem height H/2) in a periodic channel with constant viscosity: profile against the
        # independent 1D reference (canopy_reference_profile): slow flow in the canopy, fast flow above, shear layer at
        # the canopy top (Nepf 2012). Observed max errors 1.35e-3, 3.51e-4, 8.87e-5 m/s for nz = 10, 20, 40 (orders
        # 1.94, 1.98), identical in both modes.
        H, HV, CD, B, N, G, NU = 1.0, 0.5, 1.0, 0.01, 400.0, 2e-4, 1e-3
        GRIDS = (10, 20, 40)
        ORDER_MIN = 1.8
        T_END = 2000.0              # s, two viscous times H^2/nu
        zr, ur = canopy_reference_profile(; H=H, hv=HV, a=N * B, cd=CD, G=G, nu=NU)
        for use_hydrostatic in (true, false)
            errors = Float64[]
            for nz in (use_hydrostatic ? GRIDS : GRIDS[1:2])
                cfg = vegetation_test_config(length=4.0, depth=H, nx=8, nz=nz, dt=0.05, nt=round(Int, T_END / 0.05), use_hydrostatic=use_hydrostatic,
                    periodic_x=true, variable_top_layer=false, pressure_solver_method="sparse", nu=NU, body_force_x=G)
                model = uniform_density_model(cfg)
                attach_vegetation!(model, VegetationModel(; patches=[VegetationPatch(x_start=0.0, x_end=cfg.length, stem_height=HV,
                    stem_diameter=B, stem_density=N, drag_coefficient=CD)]))
                for _ in 1:cfg.nt
                    step_forward!(model)
                end
                z = [(k - 0.5) * model.dz for k in 1:nz]
                push!(errors, maximum(abs, vec(mean(model.state.u; dims=2)) .- linear_interpolation.(z, Ref(zr), Ref(ur))))
            end
            @test all(log2.(errors[1:end-1] ./ errors[2:end]) .> ORDER_MIN)
        end
    end

    @testset "test_verify_individual_stems_match_patch" begin
        # 25 rigid emergent stems at random positions with the lateral spacing of the patch density give the same
        # uniform velocity as the patch, and their total drag balances the body force (observed 1.4e-6 and exact).
        G, CD, B, N = 1e-3, 1.0, 0.01, 400.0
        NSTEMS = 25
        REL_TOL = 1e-4
        cfg = vegetation_test_config(length=10.0, depth=1.0, nx=40, nz=10, dt=0.05, nt=4000, use_hydrostatic=true, periodic_x=true,
            variable_top_layer=false, pressure_solver_method="sparse", nu=1e-6, body_force_x=G)
        model = uniform_density_model(cfg)
        rng = MersenneTwister(1)
        spacing = NSTEMS / (N * cfg.length)       # m
        stems = [VegetationStem(base_x=cfg.length * rand(rng), height=2.0, diameter=B, drag_coefficient=CD, lateral_spacing=spacing) for _ in 1:NSTEMS]
        attach_vegetation!(model, VegetationModel(stems...))
        for _ in 1:cfg.nt
            step_forward!(model)
        end
        @test isapprox(mean(model.state.u[:, 1:end-1]), sqrt(2G / (CD * N * B)); rtol=REL_TOL)
        @test isapprox(sum(s.force_x for s in stems) / spacing, VEG_RHO0 * G * cfg.depth * cfg.length; rtol=REL_TOL)
    end

    @testset "test_verify_flexible_patch_momentum_balance" begin
        # Flexible submerged patch in a periodic channel: in the steady state the total drag balances the body force,
        # the stems bend, and the reduced drag gives a larger velocity than the rigid patch. Observed: drag / (rho G H L)
        # = 0.99997 after 1000 s and 1 - 1e-6 after 1400 s (flexible); tip angle 0.75 rad.
        G, CD, B, N, EI = 1e-3, 1.0, 0.01, 200.0, 5e-4
        REL_TOL = 1e-4
        T_END = 1200.0              # s
        cfgs = Dict{Bool,Any}()
        means = Dict{Bool,Float64}()
        for flexible in (false, true)
            cfg = vegetation_test_config(length=4.0, depth=1.0, nx=8, nz=20, dt=0.05, nt=round(Int, T_END / 0.05), use_hydrostatic=true,
                periodic_x=true, variable_top_layer=false, pressure_solver_method="sparse", nu=5e-3, body_force_x=G)
            model = uniform_density_model(cfg)
            patch = VegetationPatch(x_start=0.0, x_end=cfg.length, stem_height=0.6, stem_diameter=B, stem_density=N, drag_coefficient=CD,
                bending_stiffness=flexible ? EI : Inf)
            attach_vegetation!(model, VegetationModel(; patches=[patch]))
            accel = zeros(size(model.state.u))
            for _ in 1:cfg.nt
                step_forward!(model)
            end
            vegetation_source!(accel, model.vegetation, model.state.u, model.dx, model.dz, VEG_RHO0; periodic_x=true)
            # momentum balance per unit width: drag = rho G H L
            @test isapprox(-sum(accel[:, 1:end-1]) * VEG_RHO0 * model.dx * model.dz, VEG_RHO0 * G * cfg.depth * cfg.length; rtol=REL_TOL)
            means[flexible] = mean(model.state.u)
            if flexible
                @test maximum(model.vegetation.patch_theta[1]) > 0.1      # rad, bent downstream
            end
        end
        @test means[true] > means[false]
    end

    @testset "test_validate_patch_head_loss" begin
        # Emergent rigid patch (length L) in a channel with prescribed inflow and outflow U: head loss across the patch
        # g d_eta = 1/2 C_D a L U^2 from the momentum balance (free-slip bed). Observed relative error -3.1e-4 in both
        # modes (the depth change across the patch is neglected in the balance).
        U, LP, A, CD = 0.1, 4.0, 1.0, 1.0
        REL_TOL = 3e-3
        for use_hydrostatic in (true, false)
            cfg = vegetation_test_config(length=40.0, depth=2.0, nx=200, nz=10, dt=0.02, nt=round(Int, 600 / 0.02), use_hydrostatic=use_hydrostatic,
                variable_top_layer=false, nu=1e-6)
            model = uniform_density_model(cfg)
            fill!(model.state.u, U)
            set_x_inflow_outflow!(model, fill(U, cfg.nz); open_right=false, right_profile=fill(U, cfg.nz))
            attach_vegetation!(model, VegetationModel(; patches=[VegetationPatch(x_start=18.0, x_end=18.0 + LP, stem_height=5.0,
                stem_diameter=0.01, stem_density=A / 0.01, drag_coefficient=CD)]))
            for _ in 1:cfg.nt
                step_forward!(model)
            end
            x = [(i - 0.5) * model.dx for i in 1:cfg.nx]
            head_loss = mean(model.state.eta[(x .> 10) .& (x .< 15)]) - mean(model.state.eta[(x .> 25) .& (x .< 30)])
            @test isapprox(head_loss, 0.5 * CD * A * LP * U^2 / cfg.g; rtol=REL_TOL)
        end
    end

    @testset "test_verify_explicit_drag_stability_check" begin
        # dt * 1/2 C_D a |u| above VEGETATION_EXPLICIT_DRAG_LIMIT is rejected (here 0.5 * 1 * 40 * 0.5 * 0.2 = 2).
        cfg = vegetation_test_config(length=4.0, depth=1.0, nx=8, nz=5, dt=0.2, nt=1, use_hydrostatic=true, variable_top_layer=false)
        model = uniform_density_model(cfg)
        fill!(model.state.u, 0.5)
        model.state.u[:, 1] .= 0.0; model.state.u[:, end] .= 0.0
        attach_vegetation!(model, VegetationModel(; patches=[VegetationPatch(x_start=0.0, x_end=4.0, stem_height=2.0, stem_diameter=0.1,
            stem_density=400.0, drag_coefficient=1.0)]))
        @test_throws ArgumentError step_forward!(model)
    end

    @testset "test_validate_wave_height_decay_over_rigid_canopy" begin
        # Regular linear waves (H0/h = 0.005, kh = 0.2, shallow water) through an emergent rigid canopy (a = 10 1/m) over
        # 60 m: the wave height decays as H0 / (1 + beta x) of Dalrymple, Kirby & Hwang (1984) (dalrymple_decay_coefficient
        # in experiments/vegetation/cases.jl). Compared: the decay with canopy divided by the decay of the same run without
        # canopy (removes the small numerical damping of the implicit free surface). Observed deviations 0.25 %, 0.34 %,
        # 0.70 % at 20, 40, 60 m into the canopy (hydrostatic).
        REL_TOL = 0.015
        DEPTH, KH, WAVE_HEIGHT, A = 0.5, 0.2, 0.0025, 10.0      # m, -, m, 1/m
        CANOPY = (10.0, 70.0)                                    # m
        setup = (; depth=DEPTH, kh=KH, wave_height=WAVE_HEIGHT, use_hydrostatic=true, canopy=CANOPY, cells_per_wavelength=30, steps_per_period=2000)
        bare = run_wave_flume(; frontal_area=0.0, setup...)
        vegetated = run_wave_flume(; frontal_area=A, setup...)
        i0 = findfirst(>=(CANOPY[1]), vegetated.x)
        beta = dalrymple_decay_coefficient(; cd=1.0, a=A, H0=vegetated.H[i0], k=vegetated.k, h=DEPTH, alpha=1.0)
        for x in (30.0, 50.0, 70.0)
            i = findfirst(>=(x), vegetated.x)
            ratio = (vegetated.H[i] / vegetated.H[i0]) / (bare.H[i] / bare.H[i0])
            @test isapprox(ratio, 1 / (1 + beta * (vegetated.x[i] - vegetated.x[i0])); rtol=REL_TOL)
        end
    end

    @testset "test_vegetation_damps_free_surface_wave_motion" begin
        # Replaces the old single-stem test, whose stem changed the wave amplitude by about 1e-15 (old model: relaxation
        # factor 0.35 on the force and no stem density). An emergent rigid patch around a velocity antinode of the
        # standing wave (node of eta at x = 5 m) removes wave energy: after 10 s (about two wave periods 2L/sqrt(gH)) the total wave energy (potential + kinetic) is
        # clearly smaller than without vegetation. (The amplitude at one instant is not a fair measure: the drag also
        # shifts the phase.)
        ENERGY_RATIO_MAX = 0.8      # -, observed 0.69
        cfg = vegetation_test_config(length=20.0, depth=4.0, nx=80, nz=20, dt=5e-3, nt=1, use_hydrostatic=true, variable_top_layer=false,
            momentum_advection_scheme="upwind1")
        model_ref = uniform_density_model(cfg)
        model_veg = uniform_density_model(cfg)
        attach_vegetation!(model_veg, VegetationModel(; patches=[VegetationPatch(x_start=2.0, x_end=8.0, stem_height=5.0,
            stem_diameter=0.02, stem_density=100.0, drag_coefficient=1.0)]))
        x = [(i - 0.5) * model_ref.dx for i in 1:cfg.nx]
        for m in (model_ref, model_veg)
            m.state.eta .= 0.08 .* cos.(2pi .* x ./ cfg.length)
        end
        for _ in 1:2000
            step_forward!(model_ref); step_forward!(model_veg)
        end
        # J/m per unit density: potential g eta^2 / 2 plus depth-integrated kinetic u^2 / 2
        energy(m) = sum(0.5 * cfg.g * m.state.eta[i]^2 + 0.5 * m.dz * sum(abs2, @view m.state.u[:, i]) for i in 1:cfg.nx) * m.dx
                @test energy(model_veg) < ENERGY_RATIO_MAX * energy(model_ref)
    end

    @testset "test_validate_canopy_lock_exchange_drag_regime" begin
        # Lock exchange through a dense emergent canopy (a = 80 1/m, H = 0.2 m, g' = 0.02 m/s^2): the front decelerates
        # from the inertial regime (x_f ~ t) towards the quadratic-drag regime x_f ~ t^(2/3) (run_canopy_lock_exchange in
        # experiments/vegetation/cases.jl). Observed exponents (hydrostatic, 2 cm cells): 0.80 (10-50 s), 0.71
        # (50-150 s), 0.685 (150-400 s, before the front reaches the wall).
        EXPONENT_DRAG = 2 / 3
        EXPONENT_TOL = 0.04
        run = run_canopy_lock_exchange(; frontal_area=80.0, use_hydrostatic=true)
        early = power_law_exponent(run.t, run.front, 10.0, 50.0)
        late = power_law_exponent(run.t, run.front, 150.0, 400.0)
        @test early > late
        @test abs(late - EXPONENT_DRAG) < EXPONENT_TOL
    end

    @testset "test_vegetation_slows_lock_exchange_front" begin
        # Replaces the old single-stem test (front position unchanged to the printed digits in the old model). An
        # emergent rigid patch in the right half delays the dense front (after 60 s, when the current has entered it).
        cfg = vegetation_test_config(length=30.0, depth=4.0, nx=100, nz=20, dt=5e-3, nt=1, use_hydrostatic=false, rayleigh_damping=0.0)
        cfg.rho_left = 1000.722; cfg.rho_right = 999.972
        model_ref = Model(cfg); model_veg = Model(copy_config(cfg))
        attach_vegetation!(model_veg, VegetationModel(; patches=[VegetationPatch(x_start=15.0, x_end=30.0, stem_height=5.0,
            stem_diameter=0.01, stem_density=400.0, drag_coefficient=1.0)]))
        initialize_lock_exchange!(model_ref); initialize_lock_exchange!(model_veg)
        threshold = cfg.rho_right + 0.5 * (cfg.rho_left - cfg.rho_right)
        front(m) = (idx = findall(>(threshold), @view m.state.rho[1, :]); (maximum(idx) - 0.5) * m.dx)
        for _ in 1:12000
            step_forward!(model_ref); step_forward!(model_veg)
        end
        @test front(model_veg) < front(model_ref) - 1.0     # m
    end
end
