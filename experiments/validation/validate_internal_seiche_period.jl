using SweTransport2DV
using FFTW
using Printf
using Statistics

# Two-layer long-wave internal seiche reference:
# c = sqrt(g' * h1*h2/(h1+h2)); T = 2L/c, with g' = g*(rho_bottom-rho_top)/rho0.
# The initializer uses cos(2*pi*x/L), which corresponds to mode-2 spatial structure,
# so the expected period for this specific setup is T = L/c.
# References:
# - Gill (1982), Atmosphere-Ocean Dynamics
# - Cushman-Roisin and Beckers (2011), Introduction to Geophysical Fluid Dynamics

cfg = ModelConfig(
    length=8.0,
    depth=4.0,
    nx=320,
    nz=120,
    dt=0.02,
    nt=12000,
    use_hydrostatic=true,
    momentum_advection_scheme="upwind1",
    rayleigh_damping=0.0002,
    apply_froude_limiter=true,
    max_froude=0.9,
    write_plots=false,
    write_animation=false,
    write_pressure_buoyancy_diagnostics=false,
    pressure_solver_method="spectral",
    output_dir=joinpath("output_barotropic", "validation", "internal_seiche_period"),
)

rho_top = 998.50
rho_bottom = 1001.50
interface_height_fraction = 0.50
interface_tilt_amplitude_fraction = 0.03

h1 = (1.0 - interface_height_fraction) * cfg.depth
h2 = interface_height_fraction * cfg.depth
gp = cfg.g * (rho_bottom - rho_top) / cfg.rho0
c = sqrt(gp * h1 * h2 / (h1 + h2))
T_ref = cfg.length / c

model = Model(cfg)
initialize_internal_seiche!(
    model;
    rho_top=rho_top,
    rho_bottom=rho_bottom,
    interface_height_fraction=interface_height_fraction,
    interface_tilt_amplitude_fraction=interface_tilt_amplitude_fraction,
)

signal = Float64[]
times = Float64[]

ic = Int(round(cfg.nx / 2))
rho_mid = 0.5 * (rho_top + rho_bottom)
z_centers = collect(range(model.dz / 2.0, cfg.depth - model.dz / 2.0, length=cfg.nz))

function interface_height_from_column(rho_col::AbstractVector{<:Real}, zc::AbstractVector{<:Real}, rho_mid::Real)
    nz = length(rho_col)
    for k in 2:nz
        a = rho_col[k - 1] - rho_mid
        b = rho_col[k] - rho_mid
        if (a >= 0.0 && b <= 0.0) || (a <= 0.0 && b >= 0.0)
            if abs(b - a) < 1e-12
                return zc[k]
            end
            t = -a / (b - a)
            return zc[k - 1] + t * (zc[k] - zc[k - 1])
        end
    end
    return NaN
end

sample_stride = 5
for n in 1:cfg.nt
    step_forward!(model)
    if n % sample_stride == 0
        push!(times, n * cfg.dt)
        zint = interface_height_from_column(@view(model.state.rho[:, ic]), z_centers, rho_mid)
        if isfinite(zint)
            push!(signal, zint)
        else
            push!(signal, interface_height_fraction * cfg.depth)
        end
    end
end

y = signal .- mean(signal)
n = length(y)
if n < 32
    error("Insufficient samples for spectral period estimate")
end

dt_sample = sample_stride * cfg.dt
Y = rfft(y)
amps = abs.(Y)
nf = length(amps)
freqs = collect(0:(nf - 1)) ./ (n * dt_sample)

amps[1] = 0.0
for i in eachindex(freqs)
    if freqs[i] < 1.0 / (4.0 * T_ref) || freqs[i] > 3.0 / T_ref
        amps[i] = 0.0
    end
end

imax = argmax(amps)
f_num = freqs[imax]
if f_num <= 0.0
    error("Failed to estimate dominant internal-seiche frequency")
end

T_num = 1.0 / f_num
rel_err = abs(T_num - T_ref) / T_ref
@printf("VALIDATION=internal_seiche_period T_ref_s=%.2f T_num_s=%.2f rel_err=%.3f\n", T_ref, T_num, rel_err)

if rel_err > 0.20
    error(@sprintf("Internal seiche period validation failed: rel_err %.3f exceeds tolerance 0.20", rel_err))
end

println("Internal seiche period validation passed.")
