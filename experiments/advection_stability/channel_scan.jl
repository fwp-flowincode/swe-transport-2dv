# Issue #37: growth of a small surface perturbation on a uniform current in the full model.
#
# Periodic channel (L = 1 m, H = 0.5 m, nx = 16), fixed grid (variable_top_layer = false, as every periodic and every
# non-hydrostatic run), no viscosity, no friction; surface perturbation A cos(2 pi x / L) with A = 1e-8 m, so the run is
# linear. Prints the final surface amplitude over A (below 1: decays; "blow-up": above BLOWUP_LEVEL or not finite)
# against the advective Courant number C = U dt / dx, the scheme, the Froude number Fr = U / sqrt(g H) and nz.
#
# Run: julia --project experiments/advection_stability/channel_scan.jl [theta]   (default theta = 0.7; a few minutes)

using SweTransport2DV, Printf

const GRAVITY = 9.81           # m/s^2, as ModelConfig's default g
const PERTURBATION_AMP = 1e-8  # m, initial surface perturbation
const BLOWUP_LEVEL = 1e-2      # m, surface amplitude counted as blow-up

function growth(; scheme, nz, Fr, C, theta, hydrostatic=true, L=1.0, H=0.5, nx=16, t_end=200.0)
    U = Fr * sqrt(GRAVITY * H)   # m/s
    dx = L / nx
    dt = C * dx / U
    cfg = ModelConfig(nx=nx, nz=nz, length=L, depth=H, dt=dt, theta=theta, periodic_x=true, pressure_solver_method="sparse",
        variable_top_layer=false, use_hydrostatic=hydrostatic, momentum_advection_scheme=scheme, nu=0.0,
        rayleigh_damping=0.0, apply_froude_limiter=false)
    model = Model(cfg)
    fill!(model.state.rho, 1000.0)
    refresh_density_metadata!(model)
    fill!(model.state.u, U)
    x = [(i - 0.5) * dx for i in 1:nx]
    model.state.eta .= PERTURBATION_AMP .* cos.(2pi .* x ./ L)
    amp = PERTURBATION_AMP
    for _ in 1:round(Int, t_end / dt)
        step_forward!(model)
        amp = maximum(abs, model.state.eta)
        (!isfinite(amp) || amp > BLOWUP_LEVEL) && return Inf
    end
    return amp / PERTURBATION_AMP
end

function main(theta)
    courants = (0.1, 0.3, 0.5, 0.7, 0.9)
    println("final |eta| / A, theta = $theta (hydrostatic 200 s, non-hydrostatic 50 s)")
    for hydrostatic in (true, false), (scheme, Fr) in (("upwind1", 0.2), ("tvd2", 0.2), ("tvd2", 0.5)), nz in (2, 8, 16)
        print(rpad("$(hydrostatic ? "hydrostatic" : "non-hydro  ") $scheme Fr=$Fr nz=$nz", 36))
        for C in courants
            g = growth(; scheme=scheme, nz=nz, Fr=Fr, C=C, theta=theta, hydrostatic=hydrostatic,
                t_end=hydrostatic ? 200.0 : 50.0)
            print(isfinite(g) ? @sprintf(" C=%.1f:%9.1e", C, g) : @sprintf(" C=%.1f:  blow-up", C))
        end
        println()
    end
end

main(isempty(ARGS) ? 0.7 : parse(Float64, ARGS[1]))
