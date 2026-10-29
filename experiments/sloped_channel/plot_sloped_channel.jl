# Figures and animation of the sloped channel of issue #14 (see README.md; tables: run_cases.jl).
#
# The script writes to output_barotropic/sloped_channel_figures/ and copies the files to figures/:
# - sloped_channel_spinup.png       spin-up U(t) against U_eq tanh(t/T); error against dt (orders 1 and 2)
# - sloped_channel_profiles.png     steady u(z): constant viscosity against the parabola; k-epsilon with the roughness
#                                   length of the Chezy coefficient against the log law
# - sloped_channel_backwater.png    M1 / M2 curves against the gradually-varied-flow equation, and the difference
# - sloped_channel_sloped_bed.png   real bed slope (cut cells): velocity section, bed and surface
# - sloped_channel_longitudinal.png real bed slope: water surface, energy line, bed, depth and velocity along x
# - sloped_channel_backwater.gif    outlet level raised from 0 to 0.3 m: the backwater moves upstream
# - sloped_channel_longitudinal.gif real bed slope: lake at rest, inflow ramped up to q, surface tilts to normal flow
#
# Usage: julia --project=<path to the repository> experiments/sloped_channel/plot_sloped_channel.jl
# Run time: about 1 min.

# WHY: batch run without a display; same GR workstation choice as src/visualization.jl.
get!(ENV, "GKSwstype", "100")

using SweTransport2DV
using Plots
using Printf

const C_CHEZY = 50.0     # m^(1/2)/s
const DEPTH = 4.0        # m
const SLOPE = 1e-4       # -
const NU = 0.02          # m^2/s
const G = 9.81           # m/s^2, ModelConfig default
const KAPPA = 0.41       # -, von Karman constant (as KE_KARMAN of src/turbulence.jl)
const LENGTH = 1000.0    # m
const Q = 4.0            # m^2/s
const U_EQ = chezy_equilibrium_velocity(C_CHEZY, DEPTH, SLOPE)   # m/s
const BACKWATER_LEVEL = 0.3            # m, outlet level above / below the normal depth
const SPINUP_END = 4000.0              # s, about four friction time scales
const STEADY_END = 20000.0             # s, steady profiles
const ANIMATION_END = 6000.0           # s
const ANIMATION_RAMP = 600.0           # s, outlet level raised linearly (a step would send a bore, see README)
const ANIMATION_INTERVAL_S = 150.0     # s, simulated time between frames
const ANIMATION_FPS = 8                # frames/s of the GIF
const FRAME_SIZE_PX = (640, 420)
const FIGURE_SIZE_PX = (1100, 420)
const GUIDE_OFFSET = 0.5               # -, factor of the order guide lines below the data
const FIGURE_MARGIN_MM = 5Plots.mm
const OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "output_barotropic", "sloped_channel_figures")
const FIGURES_DIR = joinpath(@__DIR__, "figures")

function finish!(model)
    fill!(model.state.rho, 1000.0)
    refresh_density_metadata!(model)
    return model
end

function steady_profile(nz)
    dz = DEPTH / nz
    P = [chezy_viscous_profile((k - 0.5) * dz, C_CHEZY, DEPTH, SLOPE, NU) for k in 1:nz]
    return P .+ (U_EQ - sum(P) / nz)
end

function periodic_channel(; nz=10, dt, implicit, kwargs...)
    cfg = ModelConfig(; nx=4, nz=nz, length=40.0, depth=DEPTH, dt=dt, periodic_x=true, pressure_solver_method="sparse",
        variable_top_layer=false, implicit_vertical_terms=implicit, body_force_x=G * SLOPE, rayleigh_damping=0.0,
        apply_froude_limiter=false, write_plots=false, kwargs...)
    return finish!(Model(cfg))
end

function open_channel(; nx=50, nz=8, dt=10.0, wd=false, right_level=0.0)
    cfg = ModelConfig(nx=nx, nz=nz, length=LENGTH, depth=DEPTH, dt=dt, implicit_vertical_terms=true,
        chezy_coefficient=C_CHEZY, nu=NU, body_force_x=G * SLOPE, rayleigh_damping=0.0, apply_froude_limiter=false,
        momentum_advection_scheme="tvd2", wetting_drying=wd, bed_level=wd ? zeros(nx) : nothing, write_plots=false)
    model = finish!(Model(cfg))
    P = steady_profile(nz)
    for i in axes(model.state.u, 2)
        model.state.u[:, i] .= P
    end
    set_x_inflow_outflow!(model, P; left_discharge=Q, right_level=right_level)
    return model, P
end

function save_figure(fig, name)
    path = joinpath(OUTPUT_DIR, name)
    savefig(fig, path)
    cp(path, joinpath(FIGURES_DIR, name); force=true)
    println("wrote ", path)
end

function spinup_figure()
    t_ref = range(0, SPINUP_END, length=200)
    p1 = plot(t_ref, [chezy_spinup_velocity(t, C_CHEZY, DEPTH, SLOPE) for t in t_ref]; color=:black, lw=2,
        label="U_eq tanh(t/T)", xlabel="t (s)", ylabel="depth-averaged U (m/s)", title="Spin-up from rest", legend=:bottomright)
    for (implicit, dt, style) in ((false, 2.0, :steelblue), (true, 20.0, :crimson))
        model = periodic_channel(; dt=dt, implicit=implicit, chezy_coefficient=C_CHEZY, nu=NU)
        ts, Us = [0.0], [0.0]
        stride = round(Int, 100.0 / dt)
        for n in 1:round(Int, SPINUP_END / dt)
            step_forward!(model)
            n % stride == 0 && (push!(ts, n * dt); push!(Us, sum(model.state.u[:, 1]) / model.config.nz))
        end
        scatter!(p1, ts, Us; color=style, ms=3, msw=0,
            label=(implicit ? "implicit" : "explicit") * @sprintf(", dt = %g s", dt))
    end
    exact = chezy_spinup_velocity(1000.0, C_CHEZY, DEPTH, SLOPE)
    p2 = plot(xscale=:log10, yscale=:log10, xlabel="dt (s)", ylabel="error of U(1000 s) (m/s)", title="Order in time",
        legend=:topleft)
    for (implicit, dts, color) in ((false, (2.0, 1.0, 0.5), :steelblue), (true, (20.0, 10.0, 5.0), :crimson))
        errors = map(collect(dts)) do dt
            model = periodic_channel(; dt=dt, implicit=implicit, chezy_coefficient=C_CHEZY, nu=NU)
            for _ in 1:round(Int, 1000.0 / dt)
                step_forward!(model)
            end
            abs(sum(model.state.u[:, 1]) / model.config.nz - exact)
        end
        order = implicit ? 2 : 1
        plot!(p2, collect(dts), errors; marker=:circle, color=color, label=implicit ? "implicit" : "explicit")
        # guide shifted by GUIDE_OFFSET so that it does not hide the data
        plot!(p2, collect(dts), GUIDE_OFFSET * errors[1] .* (collect(dts) ./ dts[1]) .^ order; color=color, ls=:dash,
            label="slope $order")
    end
    save_figure(plot(p1, p2; layout=(1, 2), size=FIGURE_SIZE_PX, margin=FIGURE_MARGIN_MM), "sloped_channel_spinup.png")
end

function profiles_figure()
    nz = 20
    dz = DEPTH / nz
    zc = ((1:nz) .- 0.5) .* dz
    model = periodic_channel(; nz=nz, dt=10.0, implicit=true, chezy_coefficient=C_CHEZY, nu=NU)
    for _ in 1:round(Int, STEADY_END / model.config.dt)
        step_forward!(model)
    end
    zf = range(0, DEPTH, length=200)
    p1 = plot([chezy_viscous_profile(z, C_CHEZY, DEPTH, SLOPE, NU) for z in zf], zf; color=:black, lw=2,
        label="parabola (constant nu)", xlabel="u (m/s)", ylabel="z (m)", title="Chezy friction, nu = $(NU) m²/s",
        legend=:topleft)
    scatter!(p1, model.state.u[:, 1], zc; color=:crimson, ms=4, msw=0, label="model, nz = $nz")
    vline!(p1, [U_EQ]; color=:gray, ls=:dash, label="U = C sqrt(H i_b)")

    z0 = chezy_roughness_length(C_CHEZY, DEPTH)
    ke = periodic_channel(; nz=nz, dt=2.0, implicit=true, nu=1e-6, turbulence_closure="k_epsilon",
        bed_roughness_length=z0, momentum_advection_scheme="tvd2")
    for _ in 1:round(Int, STEADY_END / ke.config.dt)
        step_forward!(ke)
    end
    u_star = sqrt(G * DEPTH * SLOPE)   # m/s, bed shear velocity of the uniform flow
    zl = range(z0 * 50, DEPTH, length=200)
    p2 = plot(u_star / KAPPA .* log.(zl ./ z0), zl; color=:black, lw=2, label="log law, u_* = sqrt(g H i_b)",
        xlabel="u (m/s)", ylabel="z (m)", title=@sprintf("k-epsilon, z0 = %.1f mm", 1e3 * z0), legend=:topleft)
    scatter!(p2, ke.state.u[:, 1], zc; color=:steelblue, ms=4, msw=0,
        label=@sprintf("model, U = %.3f m/s", sum(ke.state.u[:, 1]) / nz))
    vline!(p2, [U_EQ]; color=:gray, ls=:dash, label="Chezy U = 1 m/s")
    save_figure(plot(p1, p2; layout=(1, 2), size=FIGURE_SIZE_PX, margin=FIGURE_MARGIN_MM), "sloped_channel_profiles.png")
end

function backwater_figure()
    nx = 50
    dx = LENGTH / nx
    xc = ((1:nx) .- 0.5) .* dx
    p1 = plot(xlabel="x (m)", ylabel="depth h (m)", title="Backwater (M1) and drawdown (M2)", legend=:right)
    p2 = plot(xlabel="x (m)", ylabel="h - h_ref (mm)", title="Difference to the reference", legend=:bottomleft)
    hline!(p1, [DEPTH]; color=:gray, ls=:dot, label="normal depth")
    for eta_R in (BACKWATER_LEVEL, -BACKWATER_LEVEL)
        href = [chezy_backwater_depth(x, Q, C_CHEZY, SLOPE, LENGTH, DEPTH + eta_R) for x in xc]
        plot!(p1, xc, href; color=:black, lw=2, label=eta_R > 0 ? "gradually varied flow (RK4)" : "")
        for (wd, color, marker) in ((false, :crimson, :circle), (true, :steelblue, :utriangle))
            model, P = open_channel(; nx=nx, dt=dx / 2, wd=wd, right_level=eta_R)
            wd ? initialize_water_level!(model, href) : (model.state.eta .= href .- DEPTH)
            for i in axes(model.state.u, 2)
                model.state.u[:, i] .= P .* (DEPTH / href[min(i, nx)])
            end
            for _ in 1:round(Int, STEADY_END / model.config.dt)
                step_forward!(model)
            end
            h = DEPTH .+ model.state.eta
            label = eta_R > 0 ? (wd ? "wetting-drying step" : "variable top layer") : ""
            scatter!(p1, xc[1:2:end], h[1:2:end]; color=color, marker=marker, ms=3, msw=0, label=label)
            plot!(p2, xc, 1e3 .* (h .- href); color=color, ls=eta_R > 0 ? :solid : :dash,
                label=@sprintf("%s, eta_R = %+.1f m", wd ? "wetting-drying" : "variable top", eta_R))
        end
    end
    save_figure(plot(p1, p2; layout=(1, 2), size=FIGURE_SIZE_PX, margin=FIGURE_MARGIN_MM), "sloped_channel_backwater.png")
end

# Real bed slope in the wetting-drying step (no body force): the bed falls from i_b L at x = 0 to 0 at x = L, the grid is
# i_b L deeper than H, the outlet level is the normal depth above the bed extrapolated to x = L. Starts at normal flow
# (start = :normal) or from a lake at the outlet level at rest (start = :lake). Returns the model, the cell centres (m),
# the bed (m), the inflow shape and the outlet level eta_R (m).
const SLOPED_BED_NX = 50
const SLOPED_BED_NZ = 11
function sloped_bed_model(; start=:normal)
    nx, nz = SLOPED_BED_NX, SLOPED_BED_NZ
    dx = LENGTH / nx
    grid_depth = DEPTH + SLOPE * LENGTH
    xc = ((1:nx) .- 0.5) .* dx
    bed = SLOPE .* (LENGTH .- xc)
    cfg = ModelConfig(nx=nx, nz=nz, length=LENGTH, depth=grid_depth, dt=dx / 2, implicit_vertical_terms=true,
        chezy_coefficient=C_CHEZY, nu=NU, rayleigh_damping=0.0, apply_froude_limiter=false,
        momentum_advection_scheme="tvd2", wetting_drying=true, bed_level=bed, write_plots=false)
    model = finish!(Model(cfg))
    # inflow shape: the steady parabola at the centres of the (cut) layers of the inlet face at normal depth
    h = zeros(nz)
    layer_thicknesses!(h, DEPTH + bed[1], model.wetdry.bed_face[1], model.dz)
    zc = cumsum(h) .- h ./ 2
    shape = [h[k] > 0.0 ? chezy_viscous_profile(zc[k], C_CHEZY, DEPTH, SLOPE, NU) : 0.0 for k in 1:nz]
    eta_R = DEPTH - grid_depth
    if start == :normal
        initialize_water_level!(model, bed .+ DEPTH)
        for i in axes(model.state.u, 2)
            model.state.u[:, i] .= shape
        end
        set_x_inflow_outflow!(model, shape; left_discharge=Q, right_level=eta_R)
    else
        initialize_water_level!(model, DEPTH)   # horizontal surface at the outlet level, no flow
        fill!(model.state.u, 0.0)
        set_x_inflow_outflow!(model, shape; left_discharge=0.0, right_level=eta_R)
    end
    return model, xc, bed, shape, eta_R
end

# Water level (m), depth (m), depth-averaged velocity (m/s) and discharge (m^2/s) per column of the wetting-drying model.
function longitudinal_state(model, bed)
    depth = water_depth(model)
    level = bed .+ depth
    nx = model.config.nx
    q = [0.5 * (sum(model.wetdry.hf[:, i] .* model.state.u[:, i]) + sum(model.wetdry.hf[:, i + 1] .* model.state.u[:, i + 1]))
         for i in 1:nx]
    return level, depth, q ./ depth, q
end

# Longitudinal panels: water surface (and energy line) above, bed below, on two zoomed axes with the same vertical span,
# so the slopes are comparable; depth and depth-averaged velocity against the normal values.
const LONG_SPAN = 0.2    # m, vertical span of the zoomed level and bed axes (holds U²/2g = 0.05 m above the surface)
function longitudinal_panels(xc, bed, level, depth, U; title="", energy=true, span=LONG_SPAN)
    p_level = plot(xc, level; color=:steelblue, lw=3, label="water surface", ylims=(DEPTH - 0.03, DEPTH - 0.03 + span),
        ylabel="z (m)", title=title, legend=:topright)
    energy && plot!(p_level, xc, level .+ U .^ 2 ./ (2G); color=:darkorange, lw=2, label="energy line z + U²/2g")
    plot!(p_level, xc, bed .+ DEPTH; color=:black, ls=:dash, lw=1, label="bed + H (normal flow)")
    p_bed = plot(xc, bed; color=:saddlebrown, lw=3, fillrange=-1.0, fillalpha=0.3, fillcolor=:saddlebrown, label="bed",
        ylims=(-0.03, -0.03 + span), ylabel="z (m)", legend=:topright)
    p_depth = plot(xc, depth; color=:steelblue, lw=2, label="depth h", ylims=(DEPTH - 0.12, DEPTH + 0.12), ylabel="h (m)",
        legend=:bottomright)
    hline!(p_depth, [DEPTH]; color=:gray, ls=:dot, label="H = 4 m")
    p_U = plot(xc, U; color=:crimson, lw=2, label="U = q / h", ylims=(0.0, 1.2), xlabel="x (m)", ylabel="U (m/s)",
        legend=:bottomright)
    hline!(p_U, [U_EQ]; color=:gray, ls=:dot, label="U = C sqrt(H i_b)")
    return p_level, p_bed, p_depth, p_U
end

function sloped_bed_longitudinal_figure()
    model, xc, bed, _, _ = sloped_bed_model()
    for _ in 1:round(Int, STEADY_END / model.config.dt)
        step_forward!(model)
    end
    level, depth, U, _ = longitudinal_state(model, bed)
    panels = longitudinal_panels(xc, bed, level, depth, U; title="Uniform flow on the slope i_b = 1e-4 (cut cells)")
    fig = plot(panels...; layout=grid(4, 1, heights=[0.3, 0.3, 0.2, 0.2]), size=(800, 760), margin=FIGURE_MARGIN_MM,
        link=:x)
    save_figure(fig, "sloped_channel_longitudinal.png")
end

const LONG_ANIMATION_END = 8000.0      # s
const LONG_ANIMATION_RAMP = 1000.0     # s, inflow raised linearly from 0 to q
const LONG_ANIMATION_SPAN = 0.28       # m, level and bed axes; the surface overshoots to about 4.22 m at t ~ 1000 s
function sloped_bed_animation()
    model, xc, bed, shape, eta_R = sloped_bed_model(; start=:lake)
    dt = model.config.dt
    function draw(t, q_in)
        level, depth, U, _ = longitudinal_state(model, bed)
        panels = longitudinal_panels(xc, bed, level, depth, U; energy=false, span=LONG_ANIMATION_SPAN,
            title=@sprintf("t = %4.0f s, inflow q = %.2f m²/s", t, q_in))
        return plot(panels...; layout=grid(4, 1, heights=[0.3, 0.3, 0.2, 0.2]), size=(640, 640), margin=3Plots.mm,
            link=:x)
    end
    anim = Animation()
    frame(anim, draw(0.0, 0.0))
    stride = round(Int, ANIMATION_INTERVAL_S / dt)
    for n in 1:round(Int, LONG_ANIMATION_END / dt)
        t = n * dt
        q_in = Q * min(t / LONG_ANIMATION_RAMP, 1.0)
        set_x_inflow_outflow!(model, shape; left_discharge=q_in, right_level=eta_R)
        step_forward!(model)
        n % stride == 0 && frame(anim, draw(t, q_in))
    end
    path = joinpath(OUTPUT_DIR, "sloped_channel_longitudinal.gif")
    gif(anim, path; fps=ANIMATION_FPS)
    cp(path, joinpath(FIGURES_DIR, "sloped_channel_longitudinal.gif"); force=true)
    @printf("wrote %s (%.0f kB)\n", path, filesize(path) / 1024)
end

function sloped_bed_figure()
    model, xc, bed, _, _ = sloped_bed_model()
    nx, nz = SLOPED_BED_NX, SLOPED_BED_NZ
    grid_depth = DEPTH + SLOPE * LENGTH
    cfg = model.config
    for _ in 1:round(Int, STEADY_END / cfg.dt)
        step_forward!(model)
    end
    # cell-centred velocity of the wet cells (NaN: dry or empty)
    wd = model.wetdry
    ucell = fill(NaN, nz, nx)
    for i in 1:nx, k in 1:nz
        wd.h[k, i] > 0.0 && (ucell[k, i] = 0.5 * (model.state.u[k, i] + model.state.u[k, i + 1]))
    end
    zcell = ((1:nz) .- 0.5) .* model.dz
    level = grid_depth .+ model.state.eta
    p1 = heatmap(xc, zcell, ucell; color=:viridis, clims=(0.0, 1.2), colorbar_title="u (m/s)", xlabel="x (m)",
        ylabel="z (m)", title="Real bed slope i_b = 1e-4, cut cells")
    plot!(p1, xc, bed; color=:saddlebrown, lw=4, label="bed (falls 0.1 m)")
    plot!(p1, xc, level; color=:white, lw=2, label="surface", legend=:right, ylims=(0, grid_depth + 0.2))
    depth = water_depth(model)
    p2 = plot(xc, 1e3 .* (depth .- DEPTH); color=:crimson, label="depth - H", xlabel="x (m)", ylabel="mm",
        title="Deviation from normal flow")
    U = [sum(wd.hf[:, i] .* model.state.u[:, i]) / max(sum(wd.hf[:, i]), eps()) for i in 1:nx]
    plot!(p2, xc, 1e3 .* (U .- U_EQ); color=:steelblue, label="U - U_eq (mm/s)")
    save_figure(plot(p1, p2; layout=(1, 2), size=FIGURE_SIZE_PX, margin=FIGURE_MARGIN_MM), "sloped_channel_sloped_bed.png")
end

function backwater_animation()
    nx = 50
    dx = LENGTH / nx
    xc = ((1:nx) .- 0.5) .* dx
    xf = (0:nx) .* dx
    model, _ = open_channel(; nx=nx, dt=dx / 2)
    href = [chezy_backwater_depth(x, Q, C_CHEZY, SLOPE, LENGTH, DEPTH + BACKWATER_LEVEL) for x in xc]
    P = steady_profile(model.config.nz)
    nz, dz = model.config.nz, model.dz
    function draw(t)
        transport = [sum(model.state.u[1:(nz - 1), f]) * dz + (dz + model.state.eta[min(f, nx)]) * model.state.u[nz, f]
                     for f in 1:(nx + 1)]
        p1 = plot(xc, href .- DEPTH; color=:black, ls=:dash, label="steady backwater curve", ylims=(-0.05, 0.4),
            ylabel="eta (m)", title=@sprintf("Outlet level raised to %.1f m, t = %4.0f s", BACKWATER_LEVEL, t),
            legend=:topleft)
        plot!(p1, xc, model.state.eta; color=:crimson, lw=2, label="model")
        p2 = plot(xf, transport; color=:steelblue, lw=2, ylims=(3.0, 5.0), xlabel="x (m)", ylabel="q (m²/s)",
            label="discharge", legend=:topright)
        hline!(p2, [Q]; color=:gray, ls=:dot, label="inflow q")
        return plot(p1, p2; layout=(2, 1), size=FRAME_SIZE_PX, margin=FIGURE_MARGIN_MM)
    end
    anim = Animation()
    frame(anim, draw(0.0))
    stride = round(Int, ANIMATION_INTERVAL_S / model.config.dt)
    for n in 1:round(Int, ANIMATION_END / model.config.dt)
        t = n * model.config.dt
        level = BACKWATER_LEVEL * min(t / ANIMATION_RAMP, 1.0)
        set_x_inflow_outflow!(model, P; left_discharge=Q, right_level=level)
        step_forward!(model)
        n % stride == 0 && frame(anim, draw(t))
    end
    path = joinpath(OUTPUT_DIR, "sloped_channel_backwater.gif")
    gif(anim, path; fps=ANIMATION_FPS)
    cp(path, joinpath(FIGURES_DIR, "sloped_channel_backwater.gif"); force=true)
    @printf("wrote %s (%.0f kB)\n", path, filesize(path) / 1024)
end

mkpath(OUTPUT_DIR)
mkpath(FIGURES_DIR)
spinup_figure()
profiles_figure()
backwater_figure()
sloped_bed_figure()
sloped_bed_longitudinal_figure()
backwater_animation()
sloped_bed_animation()
