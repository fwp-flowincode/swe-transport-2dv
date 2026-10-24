# Variable bed geometry (issue #3): staircase bed masks, input checks and the masked model paths.
# Run with the full suite (runtests.jl) or on its own: julia --project=. test/test_bed_geometry.jl

include(joinpath(@__DIR__, "test_common.jl"))

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
    end
end
