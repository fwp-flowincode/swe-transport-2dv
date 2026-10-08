using SweTransport2DV
using NPZ
using Plots
using Printf
using Statistics

include(joinpath(@__DIR__, "common.jl"))

function logarithmic_profile(z_centers::AbstractVector{<:Real}, depth::Real)
    z0 = 0.05 * depth
    profile = log.(1 .+ 20.0 .* (z_centers .- z0) ./ (depth - z0))
    profile ./= maximum(profile)
    return profile
end

function build_model(
    ;
    output_subdir::String,
    nu_value::Real,
    use_turbulence::Bool,
    turbulence_mixing_length::Real,
    vegetation_turbulence_strength::Real,
    canopy_mode::Symbol,
)
    cfg = ModelConfig(
        nx=120,
        nz=24,
        dt=0.005,
        nt=14400,
        use_hydrostatic=true,
        momentum_advection_scheme="upwind1",
        nu=Float64(nu_value),
        use_turbulence_model=use_turbulence,
        turbulence_mixing_length=Float64(turbulence_mixing_length),
        vegetation_turbulence_strength=Float64(vegetation_turbulence_strength),
        vegetation_turbulence_spread_cells=1,
        rayleigh_damping=0.01,
        apply_froude_limiter=true,
        max_froude=0.9,
        write_plots=true,
        write_animation=true,
        write_pressure_buoyancy_diagnostics=true,
        save_interval_seconds=2.0,
        output_dir=vegetation_output_dir(joinpath("uniform_flow_stiff_short", output_subdir)),
    )
    cfg.rho_left = cfg.rho0
    cfg.rho_right = cfg.rho0

    model = Model(cfg)
    fill!(model.state.rho, cfg.rho0)
    fill!(model.state.w, 0.0)
    fill!(model.state.eta, 0.0)
    fill!(model.state.p, 0.0)

    z_centers = collect(range(model.dz / 2.0, cfg.depth - model.dz / 2.0, length=cfg.nz))
    profile = logarithmic_profile(z_centers, cfg.depth)
    u_ref = 0.08

    for i in 1:(cfg.nx + 1)
        model.state.u[:, i] .= u_ref .* profile
    end

    refresh_density_metadata!(model)
    set_x_inflow_outflow!(model, u_ref .* profile; open_right=false, right_profile=u_ref .* profile)

    stems = if canopy_mode == :high
        base_positions = [0.50, 0.55, 0.60] .* cfg.length
        [
            VegetationStem(
                base_x=x,
                height=0.5 * cfg.depth,
                diameter=0.05,
                drag_coefficient=2.8,
                stiffness=1.25,
                damping=0.35,
                mass_per_length=0.8,
                sample_points=14,
                source_strength=2.4,
            ) for x in base_positions
        ]
    else
        [
            VegetationStem(
                base_x=0.55 * cfg.length,
                height=0.5 * cfg.depth,
                diameter=0.03,
                drag_coefficient=1.8,
                stiffness=1.25,
                damping=0.35,
                mass_per_length=0.8,
                sample_points=14,
                source_strength=1.2,
            ),
        ]
    end
    attach_vegetation!(model, VegetationModel(stems...))

    return model
end

function _u_profile_at_x(model, x_section::Real)
    cfg = model.config
    i = clamp(round(Int, x_section / model.dx), 1, cfg.nx)
    u_center = 0.5 .* (model.state.u[:, i] .+ model.state.u[:, i + 1])
    x_cell = (i - 0.5) * model.dx
    return u_center, x_cell
end

function save_vertical_profile!(model, case_name::String)
    cfg = model.config
    stem_idx = cld(length(model.vegetation.stems), 2)
    stem = model.vegetation.stems[stem_idx]
    z_centers = collect(range(model.dz / 2.0, cfg.depth - model.dz / 2.0, length=cfg.nz))

    x_before = max(model.dx, stem.config.base_x - 1.0)
    x_after = min(cfg.length - model.dx, stem.config.base_x + 1.0)

    u_before, x_before_cell = _u_profile_at_x(model, x_before)
    u_stem, x_stem_cell = _u_profile_at_x(model, stem.config.base_x)
    u_after, x_after_cell = _u_profile_at_x(model, x_after)

    below_mask = z_centers .<= stem.config.height
    above_mask = z_centers .> stem.config.height

    mean_below = any(below_mask) ? mean(u_stem[below_mask]) : NaN
    mean_above = any(above_mask) ? mean(u_stem[above_mask]) : NaN
    speedup = isfinite(mean_below) && mean_below != 0.0 ? mean_above / mean_below : NaN

    p_profile = plot(
        u_before,
        z_centers;
        xlabel="u (m/s)",
        ylabel="z (m)",
        linewidth=2,
        label=@sprintf("before stem x=%.2f m", x_before_cell),
        title="U-velocity sections before and after stem",
    )
    plot!(p_profile, u_after, z_centers; linewidth=2, label=@sprintf("after stem x=%.2f m", x_after_cell))
    plot!(p_profile, u_stem, z_centers; linewidth=2, linestyle=:dash, label=@sprintf("stem x=%.2f m", x_stem_cell))
    hline!(p_profile, [stem.config.height]; color=:black, linestyle=:dot, linewidth=1, label=@sprintf("stem height %.2f m", stem.config.height))

    p_sections = plot(
        [0.0, cfg.length],
        [0.0, 0.0];
        xlabel="x (m)",
        ylabel="section marker",
        ylims=(-0.2, 1.0),
        linewidth=0,
        legend=:top,
        title="Section locations",
    )
    scatter!(p_sections, [x_before_cell], [0.5]; markersize=7, label=@sprintf("before x=%.2f", x_before_cell))
    scatter!(p_sections, [x_stem_cell], [0.5]; markersize=7, label=@sprintf("stem x=%.2f", x_stem_cell))
    scatter!(p_sections, [x_after_cell], [0.5]; markersize=7, label=@sprintf("after x=%.2f", x_after_cell))
    vline!(p_sections, [x_before_cell, x_stem_cell, x_after_cell]; color=[:blue :black :red], linestyle=:dash, label="")

    profile_plot = plot(p_profile, p_sections; layout=(1, 2), size=(1200, 500))
    profile_plot_path = joinpath(cfg.output_dir, "u_sections_profile_$(case_name).png")
    savefig(profile_plot, profile_plot_path)

    profile_path = joinpath(cfg.output_dir, "vertical_profile_$(case_name).npz")
    npzwrite(profile_path, Dict(
        "z" => z_centers,
        "u_before" => u_before,
        "u_stem" => u_stem,
        "u_after" => u_after,
        "stem_height" => stem.config.height,
        "stem_x" => x_stem_cell,
        "x_before" => x_before_cell,
        "x_after" => x_after_cell,
    ))

    @printf("Saved vertical profile: %s\n", profile_path)
    @printf("Saved section profile plot: %s\n", profile_plot_path)
    @printf("Final vertical-profile means at stem x=%.2f m | below stem=%.4f m/s, above stem=%.4f m/s, above/below ratio=%.3f\n", stem.config.base_x, mean_below, mean_above, speedup)
end

function run_variant(
    case_name::String;
    nu_value::Real,
    description::String,
    output_subdir::String=case_name,
    use_turbulence::Bool=false,
    turbulence_mixing_length::Real=0.0,
    vegetation_turbulence_strength::Real=0.0,
    canopy_mode::Symbol=:baseline,
)
    model = build_model(
        output_subdir=output_subdir,
        nu_value=nu_value,
        use_turbulence=use_turbulence,
        turbulence_mixing_length=turbulence_mixing_length,
        vegetation_turbulence_strength=vegetation_turbulence_strength,
        canopy_mode=canopy_mode,
    )
    tag = case_tag(model.config)

    @printf("Running vegetation stiff-short uniform-flow case (%s) | tag=%s\n", case_name, tag)
    @printf("Case note: %s\n", description)

    t0 = time()
    run!(model)
    t1 = time()

    output_dir = model.config.output_dir
    outputs = Dict(
        "final" => joinpath(output_dir, "final_state_$(tag).png"),
        "animation" => joinpath(output_dir, "animation_$(tag).gif"),
        "pressure_buoyancy_png" => joinpath(output_dir, "pressure_buoyancy_diagnostics_$(tag).png"),
        "pressure_buoyancy_npz" => joinpath(output_dir, "pressure_buoyancy_diagnostics_$(tag).npz"),
    )
    save_vertical_profile!(model, case_name)

    @printf("CASE=vegetation_uniform_flow_stiff_short_%s FULL_WALL_SECONDS=%.2f\n", case_name, t1 - t0)
    print_case_outputs(outputs; title="Vegetation stiff-short uniform-flow outputs ($(case_name))")
end

run_variant("baseline";
    nu_value=1e-6,
    description="Half-depth stiff stem; no turbulence model",
    use_turbulence=false,
)

run_variant("enhanced_mixing";
    nu_value=2.5e-4,
    description="Half-depth stiff stem with turbulence model and vegetation-induced turbulent mixing",
    output_subdir="emix",
    use_turbulence=true,
    turbulence_mixing_length=0.25,
    vegetation_turbulence_strength=1.5,
)

run_variant("high_canopy_turb_off";
    nu_value=1e-6,
    description="High canopy effect (3 stems, larger diameter/drag/source), turbulence OFF",
    output_subdir="hc_off",
    use_turbulence=false,
    canopy_mode=:high,
)

run_variant("high_canopy_turb_on";
    nu_value=2.5e-4,
    description="High canopy effect (3 stems, larger diameter/drag/source), turbulence ON",
    output_subdir="hc_on",
    use_turbulence=true,
    turbulence_mixing_length=0.25,
    vegetation_turbulence_strength=2.5,
    canopy_mode=:high,
)
