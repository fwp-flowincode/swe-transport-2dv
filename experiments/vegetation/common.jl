using Printf
using NPZ

include(joinpath(@__DIR__, "..", "common", "helpers.jl"))

function vegetation_output_dir(case_name::AbstractString)
    return abspath(joinpath(@__DIR__, "..", "..", "output_barotropic", "vegetation", case_name))
end

function run_custom_case!(model; on_step=nothing)
    history = State{Float64}[copy_state(model.state)]
    history_eta_prev = Vector{Vector{Float64}}([copy(model.state.eta)])
    history_time = Float64[0.0]
    vegetation_history = model.vegetation === nothing ? nothing : Any[copy_vegetation(model.vegetation)]

    save_interval = max(1, Int(round(model.config.save_interval_seconds / model.config.dt)))
    eta_prev_for_plot = copy(model.state.eta)

    for i in 1:model.config.nt
        if on_step !== nothing
            on_step(model, i, (i - 1) * model.config.dt)
        end

        copyto!(eta_prev_for_plot, model.state.eta)
        step_forward!(model)

        if mod(i, save_interval) == 0
            push!(history, copy_state(model.state))
            push!(history_eta_prev, copy(eta_prev_for_plot))
            push!(history_time, i * model.config.dt)
            if vegetation_history !== nothing
                push!(vegetation_history, copy_vegetation(model.vegetation))
            end
        end
    end

    return history, history_eta_prev, history_time, vegetation_history, eta_prev_for_plot, save_interval
end

function save_custom_case!(prefix::AbstractString, title::AbstractString, model, history, history_eta_prev, history_time, vegetation_history, eta_prev_for_plot, save_interval)
    output_dir = isempty(strip(model.config.output_dir)) ? "output_barotropic" : model.config.output_dir
    mkpath(output_dir)
    tag = case_tag(model.config)

    default_final = joinpath(output_dir, "final_state_$(tag).png")
    default_anim = joinpath(output_dir, "animation_$(tag).gif")
    default_pb_plot = joinpath(output_dir, "pressure_buoyancy_diagnostics_$(tag).png")
    default_pb_npz = joinpath(output_dir, "pressure_buoyancy_diagnostics_$(tag).npz")

    plot_state(model.state, model.config; title=title, filename=default_final, eta_prev=eta_prev_for_plot, dt=model.config.dt, vegetation=model.vegetation)
    create_animation(history, model.config; filename=default_anim, save_interval=save_interval, eta_prev_history=history_eta_prev, frame_times=history_time, vegetation_history=vegetation_history)

    p_hydro_final, p_corr_final, buoy_w_final = pressure_buoyancy_components(model)
    plot_pressure_buoyancy_diagnostics(model.state, model.config, p_hydro_final, p_corr_final, buoy_w_final; filename=default_pb_plot)
    npzwrite(default_pb_npz, Dict(
        "p_total" => model.state.p,
        "p_hydro" => p_hydro_final,
        "p_corr" => p_corr_final,
        "buoy_w" => buoy_w_final,
    ))

    outputs = copy_case_outputs!(prefix, tag; output_dir=output_dir)
    print_case_outputs(outputs; title=title)
    return outputs
end