using SweTransport2DV
using Printf
using Statistics

# Reference basis (wet, small-amplitude limit of shallow-water equations):
# c = sqrt(g*H) for right/left-propagating gravity waves.
# This case checks early-time front speed before wall reflection.
# References:
# - Stoker (1957), Water Waves
# - Whitham (1974), Linear and Nonlinear Waves

cfg = ModelConfig(
    length=20.0,
    depth=4.0,
    nx=800,
    nz=120,
    dt=0.01,
    nt=600,
    use_hydrostatic=true,
    # WHY: see experiments/surface_dam_break/run_surface_dam_break_dt001_60s_anim.jl -- eta_jump/2 exceeds dz here too.
    variable_top_layer=false,
    momentum_advection_scheme="upwind1",
    rayleigh_damping=0.0,
    apply_froude_limiter=true,
    max_froude=0.9,
    write_plots=false,
    write_animation=false,
    write_pressure_buoyancy_diagnostics=false,
    pressure_solver_method="spectral",
    output_dir=joinpath("output_barotropic", "validation", "surface_dam_break_front_speed"),
)

eta_jump = 0.08
x0 = 0.5 * cfg.length
H0 = cfg.depth
c_ref = sqrt(cfg.g * H0)
t_reflect = (cfg.length - x0) / c_ref
t_window = 0.8 * t_reflect

model = Model(cfg)
initialize_surface_dam_break!(model; rho_uniform=1000.0, eta_jump=eta_jump, split_fraction=0.5)

x_centers = collect(range(model.dx / 2.0, cfg.length - model.dx / 2.0, length=cfg.nx))
dx = cfg.length / cfg.nx
mid = Int(round(cfg.nx / 2))

function right_gradient_front_x(eta::AbstractVector{<:Real}, x::AbstractVector{<:Real}, mid::Int, dx::Real)
    n = length(eta)
    if n < 3
        return NaN
    end

    grad = similar(eta, n - 1)
    @inbounds for i in 1:(n - 1)
        grad[i] = (eta[i + 1] - eta[i]) / dx
    end

    i0 = clamp(mid, 1, n - 1)
    right_grad = @view grad[i0:end]
    if isempty(right_grad)
        return NaN
    end

    local_min = argmin(right_grad)
    ig = i0 + local_min - 1
    return 0.5 * (x[ig] + x[ig + 1])
end

front_errors = Float64[]

for n in 1:cfg.nt
    step_forward!(model)
    if n % 20 == 0
        t = n * cfg.dt
        if t <= t_window
            x_num = right_gradient_front_x(model.state.eta, x_centers, mid, dx)
            x_ref = x0 + c_ref * t
            push!(front_errors, abs(x_num - x_ref))
        end
        if t > t_window
            break
        end
    end
end

if isempty(front_errors)
    error("No comparable front samples were found for dam-break validation")
end

mae = mean(front_errors)
@printf("VALIDATION=surface_dam_break_front_speed samples=%d MAE_m=%.4f c_ref=%.4f t_window=%.3f\n", length(front_errors), mae, c_ref, t_window)

# Early-time wet-wave speed check on this grid.
if mae > 0.80
    error(@sprintf("Surface dam-break validation failed: MAE %.4f m exceeds tolerance 0.80 m", mae))
end

println("Surface dam-break validation passed.")
