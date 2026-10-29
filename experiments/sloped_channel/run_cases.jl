# Issue #14: uniform and gradually varied flow in a sloped channel with Chezy friction.
#
# Channel of the issue: L = 1000 m, H = 4 m, i_b = 1e-4, q = 4 m^2/s, C = 50 m^(1/2)/s, normal flow U = C sqrt(H i_b) = 1 m/s
# (Fr = 0.16). Constant vertical viscosity NU = 0.02 m^2/s (of the order kappa u_* H / 6 = 0.017 of the turbulent flow).
# Prints five tables (copied into README.md):
#   1. spin-up of the periodic channel from rest against U(t) = U_eq tanh(t/T): error and order in time;
#   2. open channel (discharge inlet, water-level outlet, slope as body force) from 0.8 times the normal flow;
#   3. backwater (M1) and drawdown (M2) curves against the gradually-varied-flow equation;
#   4. real sloped bed (wetting-drying cut cells, no body force);
#   5. log-law friction with the roughness length of the Chezy coefficient and k-epsilon (periodic channel).
#
# Run: julia --project experiments/sloped_channel/run_cases.jl   (about two minutes)

using SweTransport2DV, Printf

const C_CHEZY = 50.0     # m^(1/2)/s
const DEPTH = 4.0        # m
const SLOPE = 1e-4       # -
const NU = 0.02          # m^2/s
const G = 9.81           # m/s^2, ModelConfig default
const LENGTH = 1000.0    # m
const Q = 4.0            # m^2/s
const U_EQ = chezy_equilibrium_velocity(C_CHEZY, DEPTH, SLOPE)   # m/s

function finish!(model)
    fill!(model.state.rho, 1000.0)
    refresh_density_metadata!(model)
    return model
end

# Discrete steady profile of nz layers (parabola of chezy_viscous_profile with the layer mean U_EQ).
function steady_profile(nz)
    dz = DEPTH / nz
    P = [chezy_viscous_profile((k - 0.5) * dz, C_CHEZY, DEPTH, SLOPE, NU) for k in 1:nz]
    return P .+ (U_EQ - sum(P) / nz)
end

function spinup_table()
    println("1. Periodic channel, spin-up from rest: U(1000 s) against U_eq tanh(t/T) = ",
        @sprintf("%.6f", chezy_spinup_velocity(1000.0, C_CHEZY, DEPTH, SLOPE)), " m/s")
    println("| friction | dt (s) | error (m/s) | order |")
    println("|---|---|---|---|")
    exact = chezy_spinup_velocity(1000.0, C_CHEZY, DEPTH, SLOPE)
    for (implicit, dts) in ((false, (2.0, 1.0, 0.5)), (true, (20.0, 10.0, 5.0)))
        previous = NaN
        for dt in dts
            cfg = ModelConfig(nx=4, nz=10, length=40.0, depth=DEPTH, dt=dt, periodic_x=true, pressure_solver_method="sparse",
                variable_top_layer=false, implicit_vertical_terms=implicit, chezy_coefficient=C_CHEZY, nu=NU,
                body_force_x=G * SLOPE, rayleigh_damping=0.0, apply_froude_limiter=false, write_plots=false)
            model = finish!(Model(cfg))
            for _ in 1:round(Int, 1000.0 / dt)
                step_forward!(model)
            end
            err = abs(sum(model.state.u[:, 1]) / cfg.nz - exact)
            order = isnan(previous) ? "" : @sprintf("%.2f", log2(previous / err))
            @printf("| %s | %g | %.2e | %s |\n", implicit ? "implicit" : "explicit", dt, err, order)
            previous = err
        end
    end
    println()
end

function open_channel(; nx, nz, dt, wd=false, right_level=0.0, start=1.0)
    cfg = ModelConfig(nx=nx, nz=nz, length=LENGTH, depth=DEPTH, dt=dt, implicit_vertical_terms=true,
        chezy_coefficient=C_CHEZY, nu=NU, body_force_x=G * SLOPE, rayleigh_damping=0.0, apply_froude_limiter=false,
        momentum_advection_scheme="tvd2", wetting_drying=wd, bed_level=wd ? zeros(nx) : nothing, write_plots=false)
    model = finish!(Model(cfg))
    P = steady_profile(nz)
    for i in axes(model.state.u, 2)
        model.state.u[:, i] .= start .* P
    end
    set_x_inflow_outflow!(model, P; left_discharge=Q, right_level=right_level)
    return model, P
end

function relaxation_table()
    println("2. Open channel (q at the inlet, eta = 0 at the outlet), start 0.8 times the normal flow: max |u - u_eq| (m/s)")
    println("| grid | t = 5000 s | 10000 s | 15000 s | 20000 s |")
    println("|---|---|---|---|---|")
    for (name, wd) in (("variable top layer", false), ("wetting-drying step", true))
        model, P = open_channel(; nx=50, nz=10, dt=10.0, wd=wd, start=0.8)
        row = String[]
        for n in 1:2000
            step_forward!(model)
            n % 500 == 0 && push!(row, @sprintf("%.1e", maximum(abs.(model.state.u .- P))))
        end
        println("| ", name, " | ", join(row, " | "), " |")
    end
    println()
end

function backwater_table()
    println("3. Backwater (M1) and drawdown (M2) curves: steady depth against the gradually-varied-flow equation (RK4)")
    println("| outlet level eta_R (m) | grid | nx | h(0) model (m) | h(0) reference (m) | max abs(h - h_ref) (m) |")
    println("|---|---|---|---|---|---|")
    nz = 8
    for eta_R in (0.3, -0.3), wd in (false, true), nx in (25, 50, 100)
        dx = LENGTH / nx
        xc = ((1:nx) .- 0.5) .* dx
        href = [chezy_backwater_depth(x, Q, C_CHEZY, SLOPE, LENGTH, DEPTH + eta_R) for x in xc]
        model, P = open_channel(; nx=nx, nz=nz, dt=dx / 2, wd=wd, right_level=eta_R)
        wd ? initialize_water_level!(model, href) : (model.state.eta .= href .- DEPTH)
        for i in axes(model.state.u, 2)
            model.state.u[:, i] .= P .* (DEPTH / href[min(i, nx)])
        end
        for _ in 1:round(Int, 20000.0 / (dx / 2))
            step_forward!(model)
        end
        h = DEPTH .+ model.state.eta
        @printf("| %+.1f | %s | %d | %.5f | %.5f | %.1e |\n", eta_R, wd ? "wetting-drying" : "variable top layer", nx, h[1],
            href[1], maximum(abs.(h .- href)))
    end
    println()
end

function sloped_bed_table()
    println("4. Real bed slope (wetting-drying cut cells, no body force): deviation from normal flow")
    println("| nx | max abs(depth - H) (m) | U(L/2) - U_eq (m/s) | q(L/2) (m^2/s) |")
    println("|---|---|---|---|")
    nz = 11
    grid_depth = DEPTH + SLOPE * LENGTH
    for nx in (25, 50, 100)
        dx = LENGTH / nx
        xc = ((1:nx) .- 0.5) .* dx
        bed = SLOPE .* (LENGTH .- xc)
        cfg = ModelConfig(nx=nx, nz=nz, length=LENGTH, depth=grid_depth, dt=dx / 2, implicit_vertical_terms=true,
            chezy_coefficient=C_CHEZY, nu=NU, rayleigh_damping=0.0, apply_froude_limiter=false,
            momentum_advection_scheme="tvd2", wetting_drying=true, bed_level=bed, write_plots=false)
        model = finish!(Model(cfg))
        initialize_water_level!(model, bed .+ DEPTH)
        h = zeros(nz)
        layer_thicknesses!(h, DEPTH + bed[1], model.wetdry.bed_face[1], model.dz)
        zc = cumsum(h) .- h ./ 2
        shape = [h[k] > 0.0 ? chezy_viscous_profile(zc[k], C_CHEZY, DEPTH, SLOPE, NU) : 0.0 for k in 1:nz]
        for i in axes(model.state.u, 2)
            model.state.u[:, i] .= shape
        end
        set_x_inflow_outflow!(model, shape; left_discharge=Q, right_level=DEPTH - grid_depth)
        for _ in 1:round(Int, 20000.0 / cfg.dt)
            step_forward!(model)
        end
        depth = water_depth(model)
        mid = nx ÷ 2
        q_mid = sum(model.wetdry.hf[:, mid] .* model.state.u[:, mid])
        @printf("| %d | %.1e | %+.1e | %.6f |\n", nx, maximum(abs.(depth .- DEPTH)), q_mid / depth[mid] - U_EQ, q_mid)
    end
    println()
end

function log_law_table()
    z0 = chezy_roughness_length(C_CHEZY, DEPTH)
    @printf("5. Log-law friction with z0 = chezy_roughness_length(C, H) = %.2e m and k-epsilon (periodic, 20000 s)\n", z0)
    println("| nz | U (m/s) | U / U_eq - 1 |")
    println("|---|---|---|")
    for nz in (10, 20, 40)
        # WHY implicit vertical terms: nu_t ~ 0.02 m^2/s breaks the explicit diffusion limit nu dt / dz^2 < 1/2 for nz >= 20
        cfg = ModelConfig(nx=4, nz=nz, length=40.0, depth=DEPTH, dt=2.0, periodic_x=true, pressure_solver_method="sparse",
            variable_top_layer=false, implicit_vertical_terms=true, nu=1e-6, body_force_x=G * SLOPE, turbulence_closure="k_epsilon",
            bed_roughness_length=z0, rayleigh_damping=0.0, apply_froude_limiter=false, momentum_advection_scheme="tvd2",
            write_plots=false)
        model = finish!(Model(cfg))
        for _ in 1:round(Int, 20000.0 / cfg.dt)
            step_forward!(model)
        end
        U = sum(model.state.u[:, 1]) / nz
        @printf("| %d | %.4f | %+.3f |\n", nz, U, U / U_EQ - 1)
    end
    println()
end

spinup_table()
relaxation_table()
backwater_table()
sloped_bed_table()
log_law_table()
