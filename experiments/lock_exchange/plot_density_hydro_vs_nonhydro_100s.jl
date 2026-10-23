# Lock-exchange density field in the xz-plane at t = 100 s: hydrostatic vs. non-hydrostatic,
# tvd2 vs. first-order upwind (upwind1) momentum advection, and upwind1 vs. tvd2 density transport.
#
# All runs use an identical configuration (grid, damping, Froude limiter, pressure-solver
# setting); only `use_hydrostatic`, `momentum_advection_scheme`, `transport_advection_scheme` and the
# time step differ, as listed in CASES. Each run writes one stand-alone density heatmap (PNG) at
# t = T_END_S with identical colour limits, so the figures are directly comparable.
#
# The settings are those of the lock-exchange case description: inviscid (nu = 0), rho0 = (rho1 + rho2)/2,
# a minimal Rayleigh damping, and time steps from the rule of analysis/lock_exchange/lock_exchange.py
# (time_step_bounds): U dt/min(dx, dz) <= 0.05 and, hydrostatic, max|w| dt/dz <= 0.5.
#
# WHY: the hydrostatic cases use a 5x smaller time step than the non-hydrostatic one. In hydrostatic
# mode w is only diagnosed from continuity and reaches ~2.5-5 m/s in single columns at the front;
# with dt = 0.01 s the vertical CFL number |w|*dt/dz (dz = 0.02 m) exceeded 1 (up to 2.5 for tvd2,
# 3.8 for upwind1), which produced grid-scale shocks ahead of the front, spurious mixing (density
# drift in undisturbed regions via the density clamp / mass correction) and a stalled front. With
# dt = 0.002 s the vertical CFL stays below ~0.5. The non-hydrostatic w is ~0.1 m/s (CFL < 0.05).
# The run loop tracks the vertical CFL every step and warns if it reaches VERTICAL_CFL_LIMIT.
#
# WHY: the hydrostatic cases run on a coarser grid (600 x 80, dx = dz = 0.05 m) than the non-hydrostatic ones
# (1500 x 200, 0.02 m). The hydrostatic solution develops no Kelvin-Helmholtz billows that would need resolving, and
# the diagnosed |w| ~ 0.25 U H/dz drops with dz, so time_step_bounds in analysis/lock_exchange/lock_exchange.py allows
# dt = 0.01 s there (default; maximum 0.02 s) instead of 0.002 s on 1500 x 200: about 50 times less work.
#
# WHY: Rayleigh damping r = 0.002 1/s (damping time 500 s, five times the simulation time): the smallest value
# that removes the vertical streaks / ripples of the inviscid hydrostatic solution (seen on 120 x 20 and 300 x 40
# at any time step). It lowers the front Froude number by ~0.5-1 % (hydrostatic) and ~2.5 % (non-hydrostatic,
# 300 x 40: 0.479 -> 0.467); the former r = 0.01 1/s lowered the non-hydrostatic value by ~11 % (-> 0.425).
#
# WHY: the hydrostatic cases also use the variable-thickness top layer (ModelConfig.variable_top_layer; now the
# library default for use_hydrostatic=true, set explicitly below for clarity). Fixed layers with a zero density flux
# through the top face while the surface moves (w_top = d eta/dt) acted as a spurious source of rho*eta_t*dt/dz per
# step in the top row (rho ~ 1000 kg/m^3), which inverted the top-row density (dense water appeared far from the
# front) and, through the density clamp and the global mass correction, drifted the density in undisturbed regions.
# The non-hydrostatic mode keeps the fixed geometry (the variable top layer is hydrostatic only). Its surface moves too
# (section 14 of the technical documentation), so its density transport advances the top row in advective form,
# which removes the same spurious source (_density_top_row_advective_form!, docs section 4).
#
# WHY: the time loop calls step_forward! directly (as validate_lock_exchange_front_speed.jl does)
# instead of run!, so that the figures show only the density with fixed colour limits and the vertical CFL is
# tracked every step. (run!'s built-in animation used to draw ~8000 velocity arrows per frame as separate plot
# series, ~1.4 h per case; it now draws at most ~60 x 12 arrows as one series, about 1 s per frame on 1500 x 200.)
#
# Lock-exchange gravity-current references:
# - Benjamin (1968), J. Fluid Mech., doi:10.1017/S0022112068002708
# - Shin et al. (2004), J. Fluid Mech., doi:10.1017/S0022112004009172
# - Jankowski dissertation notes in ../../literature/lock_exchange_jaceks_diss.md
#
# Usage (any working directory):
#   julia --project=<path to swe_transport_2dv_jl> experiments/lock_exchange/plot_density_hydro_vs_nonhydro_100s.jl [anim] [case ...]
# Without case names all cases run; otherwise only the named ones, e.g. `nonhydro_tvd2_upwind1`
# (case name = <mode>_<momentum scheme>_<transport scheme>, see CASES). The extra argument `anim` additionally writes an
# animated GIF of the density field (same axes/colour scale as the PNG, elapsed time as title,
# one frame every ANIMATION_INTERVAL_S of simulated time) next to the PNG.

# WHY: batch run without a display; same GR workstation choice as src/visualization.jl.
get!(ENV, "GKSwstype", "100")

using SweTransport2DV
using Plots
using NPZ
using Printf

include(joinpath(@__DIR__, "cases.jl"))

const T_END_S = 100.0                  # s, time at which the density field is plotted
const DT_NONHYDRO_S = 0.01             # s, time step of the non-hydrostatic cases
const DT_HYDRO_S = 0.01                # s, time step of the hydrostatic cases on their grid (vertical-CFL limit, see header)
const VERTICAL_CFL_LIMIT = 1.0         # -, stability limit of the explicit vertical advection (unsplit upwind/TVD)
const NX_NONHYDRO = 1500               # -, cells in x of the non-hydrostatic cases (dx = 0.02 m for L = 30 m)
const NZ_NONHYDRO = 200                # -, cells in z of the non-hydrostatic cases (dz = 0.02 m for H = 4 m)
const NX_HYDRO = 600                   # -, cells in x of the hydrostatic cases (dx = 0.05 m)
const NZ_HYDRO = 80                    # -, cells in z of the hydrostatic cases (dz = 0.05 m)
const FIGURE_SIZE_PX = (1200, 340)     # px, width x height of each PNG / animation frame
const FIGURE_MARGIN_MM = 6Plots.mm     # left/bottom/right margin so labels are not clipped
const COLORMAP = :viridis              # same colormap as plot_state in src/visualization.jl
const DENSITY_LABEL = "Density (kg/m³)"
const ANIMATION_FLAG = "anim"          # command-line argument that switches the GIF output on
const ANIMATION_INTERVAL_S = 1.0       # s, simulated time between animation frames
const ANIMATION_FPS = 10               # frames/s of the GIF (same as the built-in animation)
const PROGRESS_INTERVAL_S = 10.0       # s, simulated time between progress lines

const WRITE_ANIMATION = ANIMATION_FLAG in ARGS
const SELECTED_CASES = filter(!=(ANIMATION_FLAG), ARGS)   # empty -> all cases

# WHY: anchored at this file, not the cwd, so outputs land in the project's output_barotropic/
# regardless of where julia is started (the existing experiment scripts use cwd-relative paths).
const OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "output_barotropic", "lock_exchange_density_100s")

# (mode tag, momentum advection scheme, transport advection scheme, use_hydrostatic, time step in s, nx, nz)
# The README documents the tvd2/tvd2 cases; the others are comparisons (momentum scheme, transport scheme).
const CASES = (
    ("hydro", "tvd2", "tvd2", true, DT_HYDRO_S, NX_HYDRO, NZ_HYDRO),
    ("nonhydro", "tvd2", "tvd2", false, DT_NONHYDRO_S, NX_NONHYDRO, NZ_NONHYDRO),
    ("hydro", "upwind1", "tvd2", true, DT_HYDRO_S, NX_HYDRO, NZ_HYDRO),
    ("nonhydro", "tvd2", "upwind1", false, DT_NONHYDRO_S, NX_NONHYDRO, NZ_NONHYDRO),
)

# Shared settings (cases.jl); grid, time step, mode and advection schemes are overridden per case below.
base_cfg = lock_exchange_config(; use_hydrostatic=false, nx=NX_NONHYDRO, nz=NZ_NONHYDRO, dt=DT_NONHYDRO_S, t_end=T_END_S)

mkpath(OUTPUT_DIR)

gprime = base_cfg.g * (base_cfg.rho_left - base_cfg.rho_right) / base_cfg.rho0
@printf(
    "Lock-exchange | g'=%.4e m/s^2 | U_ref=%.4f m/s | t_end=%.1f s\n",
    gprime,
    0.5 * sqrt(gprime * base_cfg.depth),
    T_END_S,
)

"""
    density_heatmap(rho, cfg; title="")

Build the xz heatmap of the cell-centred density `rho` (kg/m³, size `nz × nx`).
The colour scale spans `[cfg.rho_right, cfg.rho_left]` (kg/m³), the initial density extrema.
"""
function density_heatmap(rho::AbstractMatrix{<:Real}, cfg::ModelConfig; title::AbstractString="")
    dx = cfg.length / cfg.nx    # m
    dz = cfg.depth / cfg.nz     # m
    x_centers = range(0.5 * dx, cfg.length - 0.5 * dx, length=cfg.nx)   # m
    z_centers = range(0.5 * dz, cfg.depth - 0.5 * dz, length=cfg.nz)    # m

    # WHY: fixed limits (not data min/max) keep the colour scale identical for all runs and frames.
    return heatmap(
        x_centers,
        z_centers,
        rho;
        color=COLORMAP,
        clims=(cfg.rho_right, cfg.rho_left),
        xlabel="x (m)",
        ylabel="z (m)",
        colorbar_title=DENSITY_LABEL,
        title=title,
        xlims=(0.0, cfg.length),
        ylims=(0.0, cfg.depth),
        aspect_ratio=:auto,
        size=FIGURE_SIZE_PX,
        left_margin=FIGURE_MARGIN_MM,
        bottom_margin=FIGURE_MARGIN_MM,
        right_margin=FIGURE_MARGIN_MM,
    )
end

for (mode_tag, scheme, transport_scheme, use_hydrostatic, dt_case, nx_case, nz_case) in CASES
    if !isempty(SELECTED_CASES) && !("$(mode_tag)_$(scheme)_$(transport_scheme)" in SELECTED_CASES)
        continue
    end

    cfg = copy_config(base_cfg)
    cfg.use_hydrostatic = use_hydrostatic
    cfg.momentum_advection_scheme = scheme
    cfg.transport_advection_scheme = transport_scheme
    cfg.nx = nx_case
    cfg.nz = nz_case
    cfg.dt = dt_case
    cfg.nt = round(Int, T_END_S / dt_case)
    # WHY: the variable-thickness top layer removes the zero-flux inconsistency at the moving free surface (see header);
    # it is implemented for the hydrostatic mode only (the non-hydrostatic projection assumes a fixed geometry).
    cfg.variable_top_layer = use_hydrostatic

    dt_tag = replace(@sprintf("dt%.4f", cfg.dt), "." => "p")   # same format as case_tag in experiments/common/helpers.jl
    stem = @sprintf("lock_exchange_density_%s_mom-%s_trans-%s_nx%d_nz%d_%s_t%ds", mode_tag, scheme, transport_scheme, nx_case, nz_case, dt_tag, round(Int, T_END_S))
    frame_stride = round(Int, ANIMATION_INTERVAL_S / cfg.dt)      # steps between frames
    progress_stride = round(Int, PROGRESS_INTERVAL_S / cfg.dt)    # steps between progress lines
    max_cfl_z = 0.0                                               # -, running max of |w|*dt/dz over all steps

    println("\n=== Lock exchange, $(mode_tag), momentum $(scheme), transport $(transport_scheme), $(nx_case) x $(nz_case) cells, dt=$(cfg.dt) s (nt=$(cfg.nt)), t_end=$(T_END_S) s ===")
    t0 = time()
    model = Model(cfg)
    anim = WRITE_ANIMATION ? Animation() : nothing
    if WRITE_ANIMATION
        frame(anim, density_heatmap(model.state.rho, cfg; title=@sprintf("t = %.0f s", 0.0)))
    end

    for n in 1:cfg.nt
        step_forward!(model)
        # PERF: cfl_numbers scans u and w once (~0.3 ms) against ~25-50 ms per step; every step is checked
        # because the CFL peaks are single columns that are easily missed by sparse sampling.
        _, cfl_z, _ = cfl_numbers(model.state.u, model.state.w, cfg.dt, model.dx, model.dz)
        max_cfl_z = max(max_cfl_z, cfl_z)
        if WRITE_ANIMATION && n % frame_stride == 0
            frame(anim, density_heatmap(model.state.rho, cfg; title=@sprintf("t = %.0f s", n * cfg.dt)))
        end
        if n % progress_stride == 0
            @printf(
                "t=%6.1f s | max|u|=%.3e m/s | max|w|=%.3e m/s | max vertical CFL so far=%.3f | wall=%.0f s\n",
                n * cfg.dt,
                maximum(abs, model.state.u),
                maximum(abs, model.state.w),
                max_cfl_z,
                time() - t0,
            )
        end
    end
    @printf("CASE=%s_mom-%s_trans-%s_%s FULL_WALL_SECONDS=%.2f\n", mode_tag, scheme, transport_scheme, dt_tag, time() - t0)
    @printf("Max vertical CFL over all %d steps: %.3f (limit %.1f)\n", cfg.nt, max_cfl_z, VERTICAL_CFL_LIMIT)
    if max_cfl_z >= VERTICAL_CFL_LIMIT
        @warn "Vertical CFL limit reached; results of this case are not trustworthy (reduce dt)" case=stem max_cfl_z
    end

    rho = model.state.rho   # kg/m³
    @printf(
        "Density range at t=%.1f s: [%.6f, %.6f] kg/m^3 (initial: [%.6f, %.6f])\n",
        cfg.nt * cfg.dt,
        minimum(rho),
        maximum(rho),
        cfg.rho_right,
        cfg.rho_left,
    )

    # Order of the outputs: raw field, then GIF, then PNG, so a failure to write one file
    # (e.g. a PNG held open by an image viewer on Windows) cannot cost the multi-minute results.
    # Raw field kept so the figures can be restyled without repeating the run.
    npz_path = joinpath(OUTPUT_DIR, stem * ".npz")
    npzwrite(npz_path, Dict("rho" => copy(rho), "t_s" => cfg.nt * cfg.dt))
    println("Saved density field: $(abspath(npz_path))")

    if WRITE_ANIMATION
        gif_path = joinpath(OUTPUT_DIR, stem * ".gif")
        gif(anim, gif_path; fps=ANIMATION_FPS)
        println("Saved animation: $(abspath(gif_path))")
    end

    png_path = joinpath(OUTPUT_DIR, stem * ".png")
    savefig(density_heatmap(rho, cfg), png_path)
    println("Saved density plot: $(abspath(png_path))")
end
