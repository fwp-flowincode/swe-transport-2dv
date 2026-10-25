# Vegetation patches in a channel with prescribed inflow and outflow:
# 1. emergent rigid patch: water level along the channel and the head loss against the momentum balance
#    g d_eta = 1/2 C_D a L U² (hydrostatic and non-hydrostatic);
# 2. submerged flexible patch, non-hydrostatic, mixing-length turbulence: the flow spins up from rest, accelerates over
#    the canopy, and the stems bend (animation).
#
# The script writes to output_barotropic/vegetation_figures/:
# - vegetation_patch_head_loss.png   water level along the channel with the head-loss reference (case 1)
# - vegetation_patch_flexible.gif    u and the stem shapes against time (case 2)
# - vegetation_patch_flexible.png    u, stem shapes and profiles at the end (case 2)
# - vegetation_patch.npz             water levels, final velocity field and stem angles
# The figures in figures/ are copies (the GIF reduced in size).
#
# Usage: julia --project=<path to the repository> experiments/vegetation/plot_vegetation_patch.jl
# Run time: about 10 min.

# WHY: batch run without a display; same GR workstation choice as src/visualization.jl.
get!(ENV, "GKSwstype", "100")

using SweTransport2DV
using Plots
using NPZ
using Printf
using Statistics

include(joinpath(@__DIR__, "cases.jl"))

# case 1: emergent rigid patch
const HL_LENGTH = 40.0          # m, channel
const HL_DEPTH = 2.0            # m
const HL_U = 0.1                # m/s, inflow = outflow velocity
const HL_PATCH = (18.0, 22.0)   # m, patch extent (L = 4 m)
const HL_A = 1.0                # 1/m, frontal area per volume (b = 0.01 m, n = 100 stems/m^2)
const HL_T_END = 600.0          # s
# case 2: submerged flexible patch
const FX_LENGTH = 20.0          # m
const FX_DEPTH = 1.0            # m
const FX_NX, FX_NZ = 200, 40    # -
const FX_U = 0.2                # m/s, inflow = outflow velocity
const FX_PATCH = (6.0, 12.0)    # m
const FX_STEM_HEIGHT = 0.4      # m
const FX_B = 0.005              # m, stem diameter
const FX_N = 1000.0             # stems/m^2 (a = 5 1/m)
const FX_EI = 1e-4              # N m^2 (Ca = rho C_D b U^2 l^3 / (2 EI) = 6.4 at U = 0.2 m/s)
const FX_MIXING_LENGTH = 0.05   # m
const FX_T_END = 60.0           # s
const FX_DT = 0.01              # s
const FRAME_INTERVAL_S = 0.5    # s
const STEM_DRAW_EVERY = 10      # draw every 10th representative stem of the patch
const FRAME_SIZE_PX = (900, 300)
const FIGURE_MARGIN_MM = 4Plots.mm
const OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "output_barotropic", "vegetation_figures")

mkpath(OUTPUT_DIR)
out = Dict{String,Any}()

println("Case 1: emergent rigid patch, head loss")
p_hl = plot(xlabel="x (m)", ylabel="η (mm)", title="Emergent rigid patch (a = 1 1/m, L = 4 m, U = 0.1 m/s)", legend=:topright,
    size=(800, 380), left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM)
head_loss_ref = 0.5 * HL_A * (HL_PATCH[2] - HL_PATCH[1]) * HL_U^2 / 9.81      # m
for use_hydrostatic in (true, false)
    cfg = vegetation_case_config(length=HL_LENGTH, depth=HL_DEPTH, nx=200, nz=10, dt=0.02, nt=round(Int, HL_T_END / 0.02),
        use_hydrostatic=use_hydrostatic, variable_top_layer=false, nu=1e-6)
    model = uniform_density_model(cfg)
    fill!(model.state.u, HL_U)
    set_x_inflow_outflow!(model, fill(HL_U, cfg.nz); open_right=false, right_profile=fill(HL_U, cfg.nz))
    attach_vegetation!(model, VegetationModel(; patches=[VegetationPatch(x_start=HL_PATCH[1], x_end=HL_PATCH[2], stem_height=5.0,
        stem_diameter=0.01, stem_density=HL_A / 0.01, drag_coefficient=1.0)]))
    for _ in 1:cfg.nt
        step_forward!(model)
    end
    x = [(i - 0.5) * model.dx for i in 1:cfg.nx]
    head_loss = mean(model.state.eta[(x .> 10) .& (x .< 15)]) - mean(model.state.eta[(x .> 25) .& (x .< 30)])
    @printf("  %s: head loss %.4e m, momentum balance %.4e m (relative difference %.1e)\n",
        use_hydrostatic ? "hydrostatic" : "non-hydrostatic", head_loss, head_loss_ref, head_loss / head_loss_ref - 1)
    plot!(p_hl, x, 1e3 .* model.state.eta; linewidth=2, linestyle=use_hydrostatic ? :solid : :dash,
        color=use_hydrostatic ? :steelblue : :crimson, label=use_hydrostatic ? "hydrostatic" : "non-hydrostatic")
    out["eta_head_loss_$(use_hydrostatic ? "hydro" : "nonhydro")"] = copy(model.state.eta)
    out["x_head_loss"] = x
end
level_up = mean(out["eta_head_loss_hydro"][(out["x_head_loss"] .> 10) .& (out["x_head_loss"] .< 15)])
plot!(p_hl, [10, 15], 1e3 .* [level_up, level_up]; color=:black, linewidth=3, alpha=0.4, label="upstream mean")
plot!(p_hl, [25, 30], 1e3 .* [level_up - head_loss_ref, level_up - head_loss_ref]; color=:black, linewidth=3, alpha=0.4,
    linestyle=:dot, label="upstream − ½ C_D a L U² / g")
vspan!(p_hl, collect(HL_PATCH); color=:forestgreen, alpha=0.15, label="patch")
savefig(p_hl, joinpath(OUTPUT_DIR, "vegetation_patch_head_loss.png"))

println("Case 2: submerged flexible patch, non-hydrostatic, mixing-length turbulence")
cfg = vegetation_case_config(length=FX_LENGTH, depth=FX_DEPTH, nx=FX_NX, nz=FX_NZ, dt=FX_DT, nt=round(Int, FX_T_END / FX_DT),
    use_hydrostatic=false, nu=1e-6, use_turbulence_model=true, turbulence_mixing_length=FX_MIXING_LENGTH, max_turbulent_viscosity=5e-3)
model = uniform_density_model(cfg)
inflow = fill(FX_U, cfg.nz)
set_x_inflow_outflow!(model, inflow; open_right=false, right_profile=inflow)
patch = VegetationPatch(x_start=FX_PATCH[1], x_end=FX_PATCH[2], stem_height=FX_STEM_HEIGHT, stem_diameter=FX_B, stem_density=FX_N,
    drag_coefficient=1.0, bending_stiffness=FX_EI, n_segments=20)
attach_vegetation!(model, VegetationModel(; patches=[patch]))
xf = [(i - 1) * model.dx for i in 1:(cfg.nx + 1)]          # m, u-faces
zc = [(k - 0.5) * model.dz for k in 1:cfg.nz]              # m
patch_faces = [i for i in eachindex(xf) if FX_PATCH[1] <= xf[i] <= FX_PATCH[2]]

function frame_plot(model, t)
    p = heatmap(xf, zc, model.state.u; color=:viridis, clims=(0.0, 1.8 * FX_U), xlabel="x (m)", ylabel="z (m)",
        colorbar_title="u (m/s)", title=@sprintf("Submerged flexible patch, t = %.1f s", t), xlims=(0, FX_LENGTH), ylims=(0, FX_DEPTH),
        size=FRAME_SIZE_PX, left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM)
    theta = model.vegetation.patch_theta[1]
    ds = patch.stem_height / patch.n_segments
    for i in patch_faces[1:STEM_DRAW_EVERY:end]
        th = size(theta, 2) >= i ? theta[:, i] : zeros(patch.n_segments)
        plot!(p, xf[i] .+ [0.0; cumsum(ds .* sin.(th))], [0.0; cumsum(ds .* cos.(th))]; color=:white, linewidth=1.5, label="")
    end
    return p
end

anim = Animation()
frame(anim, frame_plot(model, 0.0))
frame_stride = round(Int, FRAME_INTERVAL_S / cfg.dt)
t0 = time()
for n in 1:cfg.nt
    step_forward!(model)
    n % frame_stride == 0 && frame(anim, frame_plot(model, n * cfg.dt))
end
theta = model.vegetation.patch_theta[1]
i_mid = patch_faces[end ÷ 2 + 1]
ds = patch.stem_height / patch.n_segments
@printf("  %.0f s wall | mid-patch stem: deflected height %.3f m of %.2f m, tip angle %.2f rad | max u %.3f m/s\n",
    time() - t0, ds * sum(cos, theta[:, i_mid]), patch.stem_height, theta[end, i_mid], maximum(model.state.u))
gif(anim, joinpath(OUTPUT_DIR, "vegetation_patch_flexible.gif"); fps=10)

p_end = frame_plot(model, FX_T_END)
p_prof = plot(xlabel="u (m/s)", ylabel="z (m)", title="Profiles at t = $(Int(FX_T_END)) s", legend=:bottomright, ylims=(0, FX_DEPTH))
for (x, color) in zip((4.0, 9.0, 11.5, 16.0), (:gray, :forestgreen, :darkgreen, :steelblue))
    i = round(Int, x / model.dx) + 1
    plot!(p_prof, model.state.u[:, i], zc; color=color, linewidth=2, label="x = $(x) m")
end
hline!(p_prof, [FX_STEM_HEIGHT]; color=:forestgreen, linestyle=:dot, label="stem length")
savefig(plot(p_end, p_prof; layout=grid(1, 2, widths=[0.7, 0.3]), size=(1200, 360), left_margin=FIGURE_MARGIN_MM,
    bottom_margin=FIGURE_MARGIN_MM), joinpath(OUTPUT_DIR, "vegetation_patch_flexible.png"))

out["u_flexible"] = copy(model.state.u)
out["theta_flexible"] = copy(theta)
npzwrite(joinpath(OUTPUT_DIR, "vegetation_patch.npz"), out)
println("Saved: vegetation_patch_head_loss.png, vegetation_patch_flexible.{gif,png} in ", abspath(OUTPUT_DIR))
