# Steady flow through canopies in a periodic channel driven by a body force (channel slope), constant viscosity:
# emergent rigid canopy (exact uniform velocity), submerged rigid canopy (1D reference, grid convergence) and a submerged
# flexible canopy (elastica) next to the rigid one, in the hydrostatic and the non-hydrostatic mode.
#
# The script writes to output_barotropic/vegetation_figures/:
# - vegetation_channel_profiles.png      velocity profiles against the references; flexible vs. rigid with stem shape
# - vegetation_channel_convergence.png   submerged rigid canopy: profile error against dz
# - vegetation_channel.npz               profiles and errors
# The figures in figures/ are copies.
#
# Usage: julia --project=<path to the repository> experiments/vegetation/plot_vegetation_channel.jl
# Run time: about 12 min.

# WHY: batch run without a display; same GR workstation choice as src/visualization.jl.
get!(ENV, "GKSwstype", "100")

using SweTransport2DV
using Plots
using NPZ
using Printf
using Statistics

include(joinpath(@__DIR__, "cases.jl"))

const H = 1.0                   # m, depth
const L = 4.0                   # m, channel length (periodic)
const NX = 8                    # -, the flow is uniform in x
const CD = 1.0                  # -
const B = 0.01                  # m, stem diameter
const N_STEMS = 400.0           # stems/m^2, a = 4 1/m
const NU = 1e-3                 # m^2/s
const G_EMERGENT = 1e-3         # m/s^2, body force of the emergent case
const G_SUBMERGED = 2e-4        # m/s^2, body force of the submerged cases
const HV = 0.5                  # m, stem length of the submerged canopy
const EI_FLEXIBLE = 2e-5        # N m^2, bending stiffness of the flexible canopy (Ca about 170 at the canopy-top velocity)
const NZ = 40                   # -, cells over the depth of the profile runs
const CONVERGENCE_GRIDS = (10, 20, 40, 80)
const T_END = 3000.0            # s, three viscous times H^2/nu
const T_END_FLEXIBLE = 7000.0   # s, the flexible canopy needs about 6000 s to become steady (posture-flow feedback)
const FIGURE_SIZE_PX = (1200, 460)
const CONVERGENCE_SIZE_PX = (560, 440)
const FIGURE_MARGIN_MM = 5Plots.mm
const MODE_TITLE = Dict(true => "hydrostatic", false => "non-hydrostatic")
const MODE_STYLE = Dict(true => :solid, false => :dash)
const MODE_COLOR = Dict(true => :steelblue, false => :crimson)
const OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "output_barotropic", "vegetation_figures")

"""
    channel_run(; use_hydrostatic, nz, G, stem_height, bending_stiffness=Inf) -> (z, u, model)

Steady periodic-channel run with a patch over the whole channel; returns the cell centres (m), the mean velocity
profile (m/s) and the model.
"""
function channel_run(; use_hydrostatic, nz, G, stem_height, bending_stiffness=Inf)
    dt = nz >= 80 ? 0.025 : 0.1      # s, explicit viscous limit 2 nu dt / dz^2 <= 0.32
    t_end = isfinite(bending_stiffness) ? T_END_FLEXIBLE : T_END
    cfg = vegetation_case_config(length=L, depth=H, nx=NX, nz=nz, dt=dt, nt=round(Int, t_end / dt), use_hydrostatic=use_hydrostatic,
        periodic_x=true, variable_top_layer=false, pressure_solver_method="sparse", nu=NU, body_force_x=G)
    model = uniform_density_model(cfg)
    attach_vegetation!(model, VegetationModel(; patches=[VegetationPatch(x_start=0.0, x_end=L, stem_height=stem_height, stem_diameter=B,
        stem_density=N_STEMS, drag_coefficient=CD, bending_stiffness=bending_stiffness, n_segments=40)]))
    t0 = time()
    for _ in 1:cfg.nt
        step_forward!(model)
    end
    z = [(k - 0.5) * model.dz for k in 1:nz]
    @printf("  %s nz=%d G=%.0e hv=%.2f EI=%g: %.0f s\n", MODE_TITLE[use_hydrostatic], nz, G, stem_height, bending_stiffness, time() - t0)
    return z, vec(mean(model.state.u[:, 1:end-1]; dims=2)), model
end

mkpath(OUTPUT_DIR)
a = N_STEMS * B
out = Dict{String,Any}()

println("Emergent rigid canopy")
u_exact = sqrt(2G_EMERGENT / (CD * a))
emergent = Dict(m => channel_run(; use_hydrostatic=m, nz=NZ, G=G_EMERGENT, stem_height=2H) for m in (true, false))
for m in (true, false)
    @printf("  %s: max |u - U| / U = %.2e\n", MODE_TITLE[m], maximum(abs, emergent[m][2] .- u_exact) / u_exact)
end

println("Submerged rigid canopy")
zr, ur = canopy_reference_profile(; H=H, hv=HV, a=a, cd=CD, G=G_SUBMERGED, nu=NU)
submerged = Dict(m => channel_run(; use_hydrostatic=m, nz=NZ, G=G_SUBMERGED, stem_height=HV) for m in (true, false))
errors = Float64[]
for nz in CONVERGENCE_GRIDS
    z, u, _ = nz == NZ ? submerged[true] : channel_run(; use_hydrostatic=true, nz=nz, G=G_SUBMERGED, stem_height=HV)
    push!(errors, maximum(abs, u .- linear_interpolation.(z, Ref(zr), Ref(ur))))
end
@printf("  errors %s m/s, orders %s\n", string(round.(errors, sigdigits=3)), string(round.(log2.(errors[1:end-1] ./ errors[2:end]), digits=2)))

println("Submerged flexible canopy")
flexible = Dict(m => channel_run(; use_hydrostatic=m, nz=NZ, G=G_SUBMERGED, stem_height=HV, bending_stiffness=EI_FLEXIBLE) for m in (true, false))
theta = flexible[true][3].vegetation.patch_theta[1][:, 1]
ds = HV / length(theta)
xs = [0.0; cumsum(ds .* sin.(theta))]; zs = [0.0; cumsum(ds .* cos.(theta))]
u_top = linear_interpolation(HV, flexible[true][1], flexible[true][2])
cauchy = VEG_RHO0 * CD * B * u_top^2 * HV^3 / (2EI_FLEXIBLE)
@printf("  deflected height %.3f m of %.2f m, tip angle %.2f rad, Ca (canopy-top velocity) = %.1f; depth-mean u flexible %.4f vs rigid %.4f m/s\n",
    zs[end], HV, theta[end], cauchy, mean(flexible[true][2]), mean(submerged[true][2]))
flex_ref = flexible_canopy_reference(; H=H, stem_length=HV, a=a, b=B, cd=CD, ei=EI_FLEXIBLE, G=G_SUBMERGED, nu=NU)
for m in (true, false)
    zf, uf, mf = flexible[m]
    th = mf.vegetation.patch_theta[1][:, 1]
    dsf = HV / length(th)
    err = maximum(abs, uf .- linear_interpolation.(zf, Ref(flex_ref.z), Ref(flex_ref.u)))
    @printf("  %s vs coupled reference: max |u - u_ref| = %.2e m/s (%.3f %% of the surface velocity), tip (%.4f, %.4f) m vs (%.4f, %.4f) m\n",
        MODE_TITLE[m], err, 100 * err / flex_ref.u[end], dsf * sum(sin, th), dsf * sum(cos, th), flex_ref.tip_x, flex_ref.tip_z)
end

# Figures
p1 = plot(xlabel="u (m/s)", ylabel="z (m)", title="Emergent rigid", legend=:bottomright, ylims=(0, H), xlims=(0.0, 0.04))
vline!(p1, [u_exact]; color=:black, linewidth=3, alpha=0.4, label="exact √(2G/(C_D a))")
for m in (true, false)
    plot!(p1, emergent[m][2], emergent[m][1]; color=MODE_COLOR[m], linestyle=MODE_STYLE[m], linewidth=2, label=MODE_TITLE[m])
end
p2 = plot(xlabel="u (m/s)", ylabel="z (m)", title="Submerged rigid (h_v = H/2)", legend=:bottomright, ylims=(0, H))
plot!(p2, ur, zr; color=:black, linewidth=3, alpha=0.4, label="1D reference")
for m in (true, false)
    plot!(p2, submerged[m][2], submerged[m][1]; color=MODE_COLOR[m], linestyle=MODE_STYLE[m], linewidth=2, label=MODE_TITLE[m])
end
hline!(p2, [HV]; color=:forestgreen, linestyle=:dot, label="stem height")
p3 = plot(xlabel="u (m/s)", ylabel="z (m)", title="Submerged flexible vs. rigid", legend=:bottomright, ylims=(0, H))
plot!(p3, submerged[true][2], submerged[true][1]; color=:gray, linewidth=2, label="rigid")
plot!(p3, flex_ref.u, flex_ref.z; color=:black, linewidth=3, alpha=0.4, label="coupled 1D reference")
for m in (true, false)
    plot!(p3, flexible[m][2], flexible[m][1]; color=MODE_COLOR[m], linestyle=MODE_STYLE[m], linewidth=2, label="flexible, $(MODE_TITLE[m])")
end
xscale = 0.6 * maximum(flexible[true][2]) / max(xs[end], eps())
plot!(p3, xscale .* xs, zs; color=:forestgreen, linewidth=3, label="stem shape (x not to scale)")
ds_ref = HV / (length(flex_ref.theta) - 1)
plot!(p3, xscale .* [0.0; cumsum(ds_ref .* sin.(flex_ref.theta[1:end-1]))], [0.0; cumsum(ds_ref .* cos.(flex_ref.theta[1:end-1]))];
    color=:black, linestyle=:dot, linewidth=2, label="reference stem shape")
savefig(plot(p1, p2, p3; layout=(1, 3), size=FIGURE_SIZE_PX, left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM),
    joinpath(OUTPUT_DIR, "vegetation_channel_profiles.png"))

dz = [H / nz for nz in CONVERGENCE_GRIDS]
pc = plot(dz, errors; xscale=:log10, yscale=:log10, marker=:circle, linewidth=2, label="submerged rigid canopy",
    xlabel="dz (m)", ylabel="max |u − u_ref| (m/s)", title="Grid convergence (hydrostatic)", legend=:bottomright,
    size=CONVERGENCE_SIZE_PX, left_margin=FIGURE_MARGIN_MM, bottom_margin=FIGURE_MARGIN_MM)
plot!(pc, dz, errors[1] .* (dz ./ dz[1]) .^ 2; color=:gray, linestyle=:dot, label="second order")
savefig(pc, joinpath(OUTPUT_DIR, "vegetation_channel_convergence.png"))

npzwrite(joinpath(OUTPUT_DIR, "vegetation_channel.npz"), Dict("z" => submerged[true][1], "u_submerged_rigid" => submerged[true][2],
    "u_submerged_flexible" => flexible[true][2], "u_emergent" => emergent[true][2], "z_ref" => zr, "u_ref" => ur,
    "convergence_dz" => dz, "convergence_error" => errors, "stem_x" => xs, "stem_z" => zs))
println("Saved: vegetation_channel_profiles.png, vegetation_channel_convergence.png in ", abspath(OUTPUT_DIR))
