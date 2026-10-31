include("test_common.jl")

# WHY: every random draw is seeded inside its own testset, so the files are independent of each other and of their order.
@testset "2DV SWE Transport Tests" begin
    include("test_operators_and_diagnostics.jl")
    include("test_advection.jl")
    include("test_pressure.jl")
    include("test_free_surface.jl")
    include("test_model.jl")
    include("test_vegetation.jl")
    include("test_turbulence.jl")
    include("test_nonhydrostatic.jl")
    include("test_variable_top_layer.jl")
    include("test_bed_geometry.jl")
    include("test_wetting_drying.jl")
    include("test_theta_scheme.jl")
    include("test_implicit_vertical.jl")
    include("test_sloped_channel.jl")
    include("test_open_boundaries.jl")
    include("test_obstacles.jl")
    include("test_sluice_gate.jl")
end
