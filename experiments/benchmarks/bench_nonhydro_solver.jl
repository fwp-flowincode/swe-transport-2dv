# Issue #71: time per step of the wetting-drying non-hydrostatic step with the direct (sparse Cholesky) and the
# column-line multigrid pressure solver, on the jet basin of section 31 (200 x 56 cells, issue #10's benchmark) and on the
# plane jet of section 36 at about 1e4, 1e5 and 7e5 unknowns.
#
# Usage: julia --project=<repository> experiments/benchmarks/bench_nonhydro_solver.jl   (about 15 min)

include(joinpath(@__DIR__, "..", "jet_sediment", "jet_sediment.jl"))   # basin_model; its main runs only as a script
using Printf, Logging

const WARMUP_STEPS = 60      # -, steps before the timing (the jet develops; compilation)
const BASIN_STEPS = 200      # -, timed steps of the basin
const JET_STEPS = Dict(2 => 100, 6 => 40, 18 => 6)   # -, timed steps of the plane jet per grid (cells across the half slot)

const PJ_B0, PJ_U0, PJ_UA, PJ_DEPTH, PJ_LENGTH = 0.05, 0.5, 0.075, 1.2, 2.25   # m, m/s, m/s, m, m (test_plane_jet.jl)

function plane_jet(cells, solver)
    dz = PJ_B0 / 2 / cells
    nz, nx = round(Int, PJ_DEPTH / dz), round(Int, PJ_LENGTH / (2dz))
    dx = PJ_LENGTH / nx
    cfg = ModelConfig(; length=PJ_LENGTH, depth=PJ_DEPTH, nx=nx, nz=nz, dt=0.5 * dx / PJ_U0, use_hydrostatic=false,
        nu=1e-6, turbulence_closure="k_epsilon", bed_roughness_length=0.0, rayleigh_damping=0.0,
        apply_froude_limiter=false, momentum_advection_scheme="tvd2", transport_advection_scheme="tvd2",
        wetting_drying=true, implicit_vertical_terms=true, write_plots=false, nonhydro_solver=solver)
    model = Model(cfg)
    initialize_water_level!(model, fill(PJ_DEPTH, nx))
    zc = ((1:nz) .- 0.5) .* dz
    u_in = [z < PJ_B0 / 2 ? PJ_U0 : PJ_UA for z in zc]
    k_in = [z < PJ_B0 / 2 ? 1.5 * (0.05PJ_U0)^2 : 1.5 * (0.05PJ_UA)^2 for z in zc]
    eps_in = KE_C_MU_VALUE^0.75 .* k_in .^ 1.5 ./ (0.1PJ_B0)
    fill!(model.state.u, PJ_UA)
    fill!(model.turbulence.k, k_in[end])
    fill!(model.turbulence.eps, eps_in[end])
    set_x_inflow_outflow!(model, u_in; left_discharge=sum(u_in) * dz, left_k=k_in, left_eps=eps_in, right_level=0.0,
        right_free=true, right_relaxation_time=10.0)
    return model
end

function time_steps(model, warmup, steps)
    for _ in 1:warmup
        step_forward!(model)
    end
    t = @elapsed for _ in 1:steps
        step_forward!(model)
    end
    return 1000 * t / steps   # ms per step
end

disable_logging(Logging.Warn)
println("case                                   unknowns   direct (ms/step)   multigrid (ms/step)   ratio   CG iterations")
function basin(solver)
    model = basin_model(1.0, "nonhydrostatic")[1]
    model.config.nonhydro_solver = solver
    return model
end
CASES = [("jet basin 200 x 56 (section 31)", basin, BASIN_STEPS);
         [("plane jet, $(c) cells/half slot", s -> plane_jet(c, s), JET_STEPS[c]) for c in (2, 6, 18)]]
for (label, make, steps) in CASES
    times = Dict{String,Float64}()
    unknowns, iterations = 0, 0
    for solver in ("direct", "multigrid")
        model = make(solver)
        times[solver] = time_steps(model, WARMUP_STEPS, steps)
        unknowns = maximum(model.wetdry.nh_index)
        solver == "multigrid" && (iterations = model.wetdry.nh_iterations_max)
    end
    @printf("%-38s %9d   %16.1f   %19.1f   %5.2f   %d\n", label, unknowns, times["direct"], times["multigrid"],
        times["multigrid"] / times["direct"], iterations)
end
