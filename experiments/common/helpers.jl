using Printf

function case_tag(cfg)
    mode_tag = cfg.use_hydrostatic ? "hydro" : "nonhydro"
    scheme_tag = cfg.momentum_advection_scheme
    return replace(
        @sprintf(
            "%s_%s_dt%.4f_L%.1f_H%.1f_dx%.3f_dz%.3f",
            mode_tag,
            scheme_tag,
            cfg.dt,
            cfg.length,
            cfg.depth,
            cfg.length / cfg.nx,
            cfg.depth / cfg.nz,
        ),
        "." => "p",
    )
end

function copy_case_outputs!(prefix::AbstractString, tag::AbstractString; output_dir::AbstractString="output_barotropic")
    final_src = joinpath(output_dir, "final_state_$(tag).png")
    anim_src = joinpath(output_dir, "animation_$(tag).gif")
    pb_png_src = joinpath(output_dir, "pressure_buoyancy_diagnostics_$(tag).png")
    pb_npz_src = joinpath(output_dir, "pressure_buoyancy_diagnostics_$(tag).npz")

    final_dst = joinpath(output_dir, "$(prefix)_final_state_$(tag).png")
    anim_dst = joinpath(output_dir, "$(prefix)_animation_$(tag).gif")
    pb_png_dst = joinpath(output_dir, "$(prefix)_pressure_buoyancy_$(tag).png")
    pb_npz_dst = joinpath(output_dir, "$(prefix)_pressure_buoyancy_$(tag).npz")

    for (src, dst) in ((final_src, final_dst), (anim_src, anim_dst), (pb_png_src, pb_png_dst), (pb_npz_src, pb_npz_dst))
        if isfile(src)
            cp(src, dst; force=true)
        end
    end

    return Dict(
        "final" => abspath(final_dst),
        "animation" => abspath(anim_dst),
        "pressure_buoyancy_png" => abspath(pb_png_dst),
        "pressure_buoyancy_npz" => abspath(pb_npz_dst),
    )
end

function print_case_outputs(outputs::Dict{String, String}; title::String="Outputs")
    println("$(title):")
    println("  $(outputs["final"])")
    println("  $(outputs["animation"])")
    println("  $(outputs["pressure_buoyancy_png"])")
    println("  $(outputs["pressure_buoyancy_npz"])")
    return nothing
end
