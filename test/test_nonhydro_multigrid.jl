# Column-line multigrid for the non-hydrostatic pressure of the wetting-drying step (issue #71, src/multigrid.jl): the
# operator and the solution against the assembled system and its direct solution, the lake at rest, convergence that does
# not depend on the grid, the fallback to the direct solver. The standing-wave dispersion (test_wetting_drying.jl) and the
# backward step (test_recirculation.jl) are also run with the multigrid.
# Run with the full suite (runtests.jl) or on its own: julia --project=. test/test_nonhydro_multigrid.jl

include(joinpath(@__DIR__, "test_common.jl"))
using Logging

const MG_SOLUTION_RTOL = 1e-7       # -, |x - x_direct| / |x_direct| per solve at the default tolerance 1e-8 (observed 1e-9)
const MG_OPERATOR_RTOL = 1e-13      # -, grid operator against the assembled matrix (round-off)
const MG_ITERATIONS_MAX = 20        # -, CG iterations at the default tolerance (observed 11-15)
const MG_ITERATIONS_SPREAD = 4      # -, iterations on the finer grid at most this many above the coarser one
const MG_STEPS = 20                 # -, steps before the system is examined
const MG_TOO_FEW_ITERATIONS = 2    # -, an iteration cap CG cannot meet the tolerance in: the direct solver takes over

# Plane-jet set-up of test_plane_jet.jl, small: the half slot on the bed, k-epsilon, a co-flow
function mg_jet(cells; solver="multigrid", max_iterations=SweTransport2DV.NH_MULTIGRID_MAX_ITERATIONS)
    b0, u0, ua, depth, len = 0.05, 0.5, 0.075, 1.2, 2.25
    dz = b0 / 2 / cells
    nz, nx = round(Int, depth / dz), round(Int, len / (2dz))
    dx = len / nx
    cfg = ModelConfig(; length=len, depth=depth, nx=nx, nz=nz, dt=0.5 * dx / u0, use_hydrostatic=false, nu=1e-6,
        turbulence_closure="k_epsilon", bed_roughness_length=0.0, rayleigh_damping=0.0, apply_froude_limiter=false,
        momentum_advection_scheme="tvd2", transport_advection_scheme="tvd2", wetting_drying=true,
        implicit_vertical_terms=true, write_plots=false, write_animation=false, write_pressure_buoyancy_diagnostics=false,
        nonhydro_solver=solver, nonhydro_solver_max_iterations=max_iterations)
    model = Model(cfg)
    initialize_water_level!(model, fill(depth, nx))
    zc = ((1:nz) .- 0.5) .* dz
    u_in = [z < b0 / 2 ? u0 : ua for z in zc]
    k_in = [z < b0 / 2 ? 1.5 * (0.05u0)^2 : 1.5 * (0.05ua)^2 for z in zc]
    eps_in = SweTransport2DV.KE_C_MU^0.75 .* k_in .^ 1.5 ./ (0.1b0)
    fill!(model.state.u, ua)
    fill!(model.turbulence.k, k_in[end])
    fill!(model.turbulence.eps, eps_in[end])
    set_x_inflow_outflow!(model, u_in; left_discharge=sum(u_in) * dz, left_k=k_in, left_eps=eps_in, right_level=0.0,
        right_free=true, right_relaxation_time=10.0)
    return model
end

# Wetting-drying cases with dry cells and a moving surface: a Stoker dam break and a sloping beach
function mg_stoker()
    cfg = ModelConfig(; length=100.0, depth=1.2, nx=400, nz=10, dt=0.0025, use_hydrostatic=false, nu=0.0,
        rayleigh_damping=0.0, apply_froude_limiter=false, wetting_drying=true, momentum_advection_scheme="tvd2",
        write_plots=false, write_animation=false, write_pressure_buoyancy_diagnostics=false, nonhydro_solver="multigrid")
    model = Model(cfg)
    x = [(i - 0.5) * 0.25 for i in 1:400]
    initialize_water_level!(model, [xi < 40.0 ? 1.0 : 0.2 for xi in x]; rho=1000.0)
    return model
end
function mg_beach()
    cfg = ModelConfig(; length=24.0, depth=1.5, nx=240, nz=15, dt=0.02, use_hydrostatic=false, nu=0.0,
        rayleigh_damping=0.0, apply_froude_limiter=false, wetting_drying=true, momentum_advection_scheme="tvd2",
        write_plots=false, write_animation=false, write_pressure_buoyancy_diagnostics=false, nonhydro_solver="multigrid",
        bed_level=[1.2 - 0.05 * (20.0 - (i - 0.5) * 0.1) for i in 1:240])
    model = Model(cfg)
    x = [(i - 0.5) * 0.1 for i in 1:240]
    # a wave on the beach: the shoreline moves, columns fall dry and flood
    initialize_water_level!(model, [max(1.2 + 0.05 * cos(pi * xi / 24.0), 1.2 - 0.05 * (20.0 - xi)) for xi in x]; rho=1000.0)
    return model
end

# The assembled system of the last step (triplets of _wd_nonhydrostatic_correction!) and the grid operator on it
function mg_system(model)
    wd = model.wetdry
    n = maximum(wd.nh_index)
    A = sparse(wd.nh_rows, wd.nh_cols, wd.nh_vals, n, n)
    to_vector(g) = [g[k, i] for i in axes(wd.nh_index, 2), k in axes(wd.nh_index, 1) if wd.nh_index[k, i] > 0][
        sortperm([wd.nh_index[k, i] for i in axes(wd.nh_index, 2), k in axes(wd.nh_index, 1) if wd.nh_index[k, i] > 0])]
    return A, wd.nh_rhs[1:n], to_vector(wd.nh_multigrid.x), to_vector
end

@testset "Non-hydrostatic multigrid (issue #71)" begin
    @testset "test_verify_multigrid_operator_and_solution_$(name)" for (name, make) in
            (("jet", () -> mg_jet(2)), ("stoker", mg_stoker), ("beach", mg_beach))
        model = make()
        for _ in 1:MG_STEPS
            step_forward!(model)
        end
        wd = model.wetdry
        @test wd.nh_multigrid_solves == MG_STEPS
        @test wd.nh_fallbacks == 0
        @test wd.nh_iterations_max <= MG_ITERATIONS_MAX
        A, b, x, to_vector = mg_system(model)
        # the grid operator of the fine level is the assembled matrix
        L = wd.nh_multigrid.levels[1]
        v = zeros(size(L.d))
        Random.seed!(71)
        for i in axes(v, 2), k in axes(v, 1)
            wd.nh_index[k, i] > 0 && (v[k, i] = randn())
        end
        y = similar(v)
        SweTransport2DV._mg_apply!(y, L, v)
        Av = A * to_vector(v)
        @test maximum(abs, to_vector(y) .- Av) <= MG_OPERATOR_RTOL * maximum(abs, Av)
        # the multigrid solution is the direct one within the tolerance
        xd = cholesky(Symmetric(A)) \ b
        @test norm(x .- xd) <= MG_SOLUTION_RTOL * norm(xd)
        @test norm(A * x .- b) <= model.config.nonhydro_solver_tolerance * norm(b)
    end

    @testset "test_verify_multigrid_iterations_independent_of_grid" begin
        its = Int[]
        for cells in (2, 4)
            model = mg_jet(cells)
            for _ in 1:MG_STEPS
                step_forward!(model)
            end
            push!(its, model.wetdry.nh_iterations_max)
        end
        @test its[2] <= its[1] + MG_ITERATIONS_SPREAD
        @test maximum(its) <= MG_ITERATIONS_MAX
    end

    @testset "test_verify_multigrid_lake_at_rest" begin
        cfg = ModelConfig(; length=24.0, depth=1.5, nx=60, nz=15, dt=0.05, use_hydrostatic=false, nu=0.0,
            rayleigh_damping=0.0, apply_froude_limiter=false, wetting_drying=true, write_plots=false, write_animation=false,
            write_pressure_buoyancy_diagnostics=false, nonhydro_solver="multigrid",
            bed_level=[0.6 + 0.4 * sin(2pi * (i - 0.5) / 60) for i in 1:60])
        model = Model(cfg)
        initialize_water_level!(model, 1.2; rho=1000.0)
        for _ in 1:MG_STEPS
            step_forward!(model)
        end
        @test model.wetdry.nh_multigrid_solves == MG_STEPS
        @test all(==(0.0), model.wetdry.q_nh)
        @test maximum(abs, model.state.u) == 0.0
    end

    @testset "test_verify_multigrid_fallback_to_direct" begin
        # too few iterations allowed: the direct solver takes over (warned once), the step is the direct one
        mg = mg_jet(2; max_iterations=MG_TOO_FEW_ITERATIONS)
        direct = mg_jet(2; solver="direct")
        with_logger(NullLogger()) do
            step_forward!(mg)
        end
        step_forward!(direct)
        @test mg.wetdry.nh_fallbacks == 1
        @test mg.wetdry.q_nh == direct.wetdry.q_nh
        @test mg.state.u == direct.state.u
    end

    @testset "test_verify_nonhydro_solver_options" begin
        @test_throws ArgumentError Model(ModelConfig(wetting_drying=true, nonhydro_solver="cg"))
        @test_throws ArgumentError Model(ModelConfig(wetting_drying=true, nonhydro_solver_tolerance=0.0))
        @test_throws ArgumentError Model(ModelConfig(wetting_drying=true, nonhydro_solver_tolerance=0.1))
        @test_throws ArgumentError Model(ModelConfig(wetting_drying=true, nonhydro_solver_max_iterations=0))
        # auto: the direct solver below NH_MULTIGRID_MIN_UNKNOWNS, the multigrid above
        small = mg_stoker()
        small.config.nonhydro_solver = "auto"
        step_forward!(small)
        @test small.wetdry.nh_multigrid_solves == 0
        large = mg_jet(4; solver="auto")
        step_forward!(large)
        @test maximum(large.wetdry.nh_index) >= SweTransport2DV.NH_MULTIGRID_MIN_UNKNOWNS
        @test large.wetdry.nh_multigrid_solves == 1
    end
end
