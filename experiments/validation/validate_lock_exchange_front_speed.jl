using SweTransport2DV
using Printf
using Statistics

# Lock-exchange front-speed references:
# - Benjamin (1968), J. Fluid Mech., doi:10.1017/S0022112068002708
# - Shin et al. (2004), J. Fluid Mech., doi:10.1017/S0022112004009172
# - Jankowski dissertation notes: front-speed range about 0.44-0.59*sqrt(g'*H)

cfg = ModelConfig(
    length=30.0,
    depth=4.0,
    nx=300,
    nz=80,
    dt=0.02,
    nt=2500,
    use_hydrostatic=false,
    momentum_advection_scheme="tvd2",
    rayleigh_damping=0.01,
    apply_froude_limiter=true,
    max_froude=0.9,
    write_plots=false,
    write_animation=false,
    write_pressure_buoyancy_diagnostics=false,
    pressure_solver_method="spectral",
    output_dir=joinpath("output_barotropic", "validation", "lock_exchange_front_speed"),
)

model = Model(cfg)
initialize_lock_exchange!(model; rho_left=1000.722, rho_right=999.972)

x_centers = collect(range(model.dx / 2.0, cfg.length - model.dx / 2.0, length=cfg.nx))
rho_threshold = 0.5 * (cfg.rho_left + cfg.rho_right)

function rightmost_front_x(rho_row::AbstractVector{<:Real}, x::AbstractVector{<:Real}, threshold::Real)
    idx = findall(>(threshold), rho_row)
    if isempty(idx)
        return NaN
    end

    j = maximum(idx)
    if j >= length(rho_row)
        return x[end]
    end

    rj = rho_row[j]
    rjp = rho_row[j + 1]
    if rjp >= threshold || abs(rj - rjp) < 1e-14
        return x[j]
    end

    return x[j] + (rj - threshold) / (rj - rjp) * (x[j + 1] - x[j])
end

times = Float64[]
front_x = Float64[]
sample_stride = 20

for n in 1:cfg.nt
    step_forward!(model)
    if n % sample_stride == 0
        t = n * cfg.dt
        if t >= 2.0 && t <= 40.0
            xf = rightmost_front_x(@view(model.state.rho[1, :]), x_centers, rho_threshold)
            if isfinite(xf)
                push!(times, t)
                push!(front_x, xf)
            end
        end
    end
end

if length(times) < 12
    error("Insufficient lock-exchange front samples for validation")
end

x_mean = mean(front_x)
t_mean = mean(times)
num = sum((times .- t_mean) .* (front_x .- x_mean))
den = sum((times .- t_mean) .^ 2)
if den <= 0.0
    error("Degenerate lock-exchange time window for speed fit")
end
u_num = num / den

gprime = cfg.g * (cfg.rho_left - cfg.rho_right) / cfg.rho0
u_scale = sqrt(gprime * cfg.depth)
coeff_num = u_num / u_scale

coeff_min = 0.40
coeff_max = 0.62
@printf(
    "VALIDATION=lock_exchange_front_speed samples=%d coeff_num=%.3f coeff_ref_range=[%.2f,%.2f] u_num=%.4f u_scale=%.4f\n",
    length(times),
    coeff_num,
    coeff_min,
    coeff_max,
    u_num,
    u_scale,
)

if !(coeff_min <= coeff_num <= coeff_max)
    error(
        @sprintf(
            "Lock-exchange validation failed: coefficient %.3f outside [%.2f, %.2f]",
            coeff_num,
            coeff_min,
            coeff_max,
        ),
    )
end

println("Lock-exchange front-speed validation passed.")
