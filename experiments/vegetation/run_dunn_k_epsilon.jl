# Dunn, Lopez & Garcia (1996) rigid runs 1-12 with the k-epsilon closure (src/turbulence.jl): depth-mean velocity of the
# model against the measured U = Q / (0.91 m H), for C_D = 1 and for the measured bulk drag coefficient of each run.
# Periodic channel per run (depth, slope, canopy height and frontal area from the report, see DUNN1996_RIGID in
# cases.jl), smooth flume bed (Manning n = 0.011, roughness length z0 = 5e-6 m). Status: see README (vegetation terms of
# the k-epsilon model are an open question).
#
# Usage: julia --project=<path to the repository> experiments/vegetation/run_dunn_k_epsilon.jl
# Run time: about 4 min.

using SweTransport2DV
using Printf
using Statistics

include(joinpath(@__DIR__, "cases.jl"))

const BED_ROUGHNESS = 5e-6      # m, smooth flume bed (Strickler k_s = (21.1 n)^6 for n = 0.011, z0 = k_s/30)
const NZ = 40                   # -
const T_END = 400.0             # s (converged: 800 s and nz = 80 change the result by less than 0.6 %)
const NU_T_SCALE = 5e-3         # m^2/s, upper estimate of nu_t for the explicit diffusion limit of dt

function dunn_k_epsilon(run::Int; drag_coefficient)
    d = dunn1996_run(run)
    cd = drag_coefficient === :measured ? d.cdb : drag_coefficient
    dz = d.H / NZ
    dt = min(0.01, 0.2 * dz^2 / NU_T_SCALE)
    cfg = vegetation_case_config(length=1.0, depth=d.H, nx=4, nz=NZ, dt=dt, nt=round(Int, T_END / dt), use_hydrostatic=true, periodic_x=true,
        variable_top_layer=false, pressure_solver_method="sparse", nu=1e-6, body_force_x=9.81 * d.S, turbulence_closure="k_epsilon",
        bed_roughness_length=BED_ROUGHNESS)
    model = uniform_density_model(cfg)
    attach_vegetation!(model, VegetationModel(; patches=[VegetationPatch(x_start=0.0, x_end=1.0, stem_height=d.hp, stem_diameter=DUNN1996_D,
        stem_density=d.a / DUNN1996_D, drag_coefficient=cd)]))
    for _ in 1:cfg.nt
        step_forward!(model)
    end
    return mean(model.state.u[:, 1:end-1]), d.U
end

for drag_coefficient in (1.0, :measured)
    ratios = Float64[]
    for run in 1:12
        u_model, u_measured = dunn_k_epsilon(run; drag_coefficient=drag_coefficient)
        push!(ratios, u_model / u_measured)
        @printf("C_D = %-8s run %2d: model %.3f m/s, measured %.3f m/s, ratio %.3f\n", string(drag_coefficient), run, u_model, u_measured, ratios[end])
    end
    @printf("C_D = %s: mean ratio %.3f, mean |ratio - 1| %.3f, max |ratio - 1| %.3f\n", string(drag_coefficient), mean(ratios),
        mean(abs.(ratios .- 1)), maximum(abs.(ratios .- 1)))
end
