include("test_common.jl")

# Uniform flow in a sloped channel with Chezy friction (issue #14): C = 50 m^(1/2)/s, H = 4 m, i_b = 1e-4 give the normal
# flow U = C sqrt(H i_b) = 1 m/s, q = 4 m^2/s. The slope is the body force g i_b (ModelConfig(body_force_x)); the Chezy
# stress g |U| U / C^2 acts on the depth-averaged velocity (ModelConfig(chezy_coefficient)).

const CHEZY_C = 50.0          # m^(1/2)/s
const CHEZY_DEPTH = 4.0       # m
const CHEZY_SLOPE = 1e-4      # -, bed slope i_b (the bed falls in +x)
const CHEZY_NU = 0.02         # m^2/s, of the order kappa u_* H / 6 = 0.017 of the turbulent channel
const CHEZY_G = 9.81          # m/s^2, ModelConfig default
const CHEZY_RELAX_TIME = 1e4  # s, about 10 friction time scales T = 1019 s (and 12 viscous times H^2 / nu)
const CHEZY_RELAX_TOL = 1e-8  # m/s, distance from the steady profile after CHEZY_RELAX_TIME (decays like exp(-t/T))
const CHEZY_ORDER_TOL = 0.05  # -, tolerance of an observed order of convergence in time
const CHEZY_SPINUP_TIME = 1000.0  # s, about one friction time scale: U = 0.75 U_eq, far from the equilibrium

# Periodic channel of nx = 4 columns (uniform flow, no horizontal structure), fixed grid.
function chezy_channel(; implicit, dt, nz=10, nu=CHEZY_NU)
    cfg = ModelConfig(nx=4, nz=nz, length=40.0, depth=CHEZY_DEPTH, dt=dt, periodic_x=true, pressure_solver_method="sparse",
        variable_top_layer=false, implicit_vertical_terms=implicit, chezy_coefficient=CHEZY_C, nu=nu,
        body_force_x=CHEZY_G * CHEZY_SLOPE, rayleigh_damping=0.0, apply_froude_limiter=false,
        momentum_advection_scheme="upwind1", write_plots=false)
    model = Model(cfg)
    fill!(model.state.rho, 1000.0)
    refresh_density_metadata!(model)
    return model
end

# Discrete steady profile at the layer centres: the parabola of chezy_viscous_profile plus the constant that makes its
# layer mean the Chezy velocity (central differences of a quadratic are exact, so only the mean differs).
function chezy_discrete_profile(nz; nu=CHEZY_NU)
    dz = CHEZY_DEPTH / nz
    P = [chezy_viscous_profile((k - 0.5) * dz, CHEZY_C, CHEZY_DEPTH, CHEZY_SLOPE, nu) for k in 1:nz]
    return P .+ (chezy_equilibrium_velocity(CHEZY_C, CHEZY_DEPTH, CHEZY_SLOPE) - sum(P) / nz)
end

depth_mean(model) = sum(model.state.u[:, 1]) / model.config.nz

# Open channel of length CHANNEL_LENGTH: discharge q at the left (shape = discrete steady profile), water level at the
# right (water-level boundary), slope as body force on a flat bed; variable top layer (vt = nothing), fixed grid
# (vt = false) or wetting and drying (wd = true, flat bed).
const CHANNEL_LENGTH = 1000.0   # m
const CHANNEL_DISCHARGE = 4.0   # m^2/s, issue #14
function open_channel(; nx=20, nz=8, dt=20.0, vt=nothing, wd=false, scheme="tvd2", start=1.0, right_level=0.0,
    discharge=CHANNEL_DISCHARGE)
    cfg = ModelConfig(nx=nx, nz=nz, length=CHANNEL_LENGTH, depth=CHEZY_DEPTH, dt=dt, implicit_vertical_terms=true,
        chezy_coefficient=CHEZY_C, nu=CHEZY_NU, body_force_x=CHEZY_G * CHEZY_SLOPE, rayleigh_damping=0.0,
        apply_froude_limiter=false, momentum_advection_scheme=scheme, variable_top_layer=wd ? nothing : vt,
        wetting_drying=wd, bed_level=wd ? zeros(nx) : nothing, write_plots=false)
    model = Model(cfg)
    fill!(model.state.rho, 1000.0)
    refresh_density_metadata!(model)
    P = chezy_discrete_profile(nz)
    for i in axes(model.state.u, 2)
        model.state.u[:, i] .= start .* P
    end
    set_x_inflow_outflow!(model, P; left_discharge=discharge, right_level=right_level)
    return model, P
end

const OPEN_CHANNEL_VARIANTS = ((vt=nothing, wd=false), (vt=false, wd=false), (vt=nothing, wd=true))
const OPEN_CHANNEL_RELAX_TIME = 2e4   # s, from 0.8 times the normal flow to CHEZY_RELAX_TOL (decay ~ exp(-t / 1500 s))
const LAKE_FILL_TIME = 3e4            # s, about 50 periods 4 L / sqrt(g H) of the open-closed basin
const LAKE_FILL_TOL = 1e-4            # m, |eta - eta_R| after LAKE_FILL_TIME (theta and Chezy damping)
const BACKWATER_LEVEL = 0.3           # m, |eta_R - 0| of the M1 / M2 curves (below dz = 0.5 m of the variable top layer)
const BACKWATER_TOL = 1e-4            # m, model depth against the RK4 backwater curve (measured <= 4e-5; the top-row
                                      # vertical advection with dz instead of h_top gave 3e-4 to 6e-4)
const SLOPED_BED_TOL = 1e-4           # -, relative deviation of depth and depth-averaged velocity from normal flow

# Water volume (m^2) and boundary discharges of the last step (m^2/s; theta fluxes, which moved the level).
function channel_volume(model)
    model.wetdry === nothing || return sum(water_depth(model)) * model.dx
    return sum(model.state.eta) * model.dx
end
function boundary_discharges(model)
    wd = model.wetdry
    wd === nothing || return (sum(wd.q[:, 1]), sum(wd.q[:, end]))
    nz, dz, ws = model.config.nz, model.dz, model.ws
    vt = use_variable_top_layer(model.config)
    Q(f) = sum(ws.u_theta[1:(nz - 1), f]) * dz + (vt ? ws.h_top_face[f] : dz) * ws.u_theta[nz, f]
    return (Q(1), Q(model.config.nx + 1))
end

@testset "Sloped channel with Chezy friction (issue #14)" begin
    @testset "test_verify_chezy_reference_values_of_issue_14" begin
        U = chezy_equilibrium_velocity(CHEZY_C, CHEZY_DEPTH, CHEZY_SLOPE)
        @test U ≈ 1.0 rtol = TOL_IDENTITY
        @test U * CHEZY_DEPTH ≈ 4.0 rtol = TOL_IDENTITY   # q (m^2/s) of the issue
        T = CHEZY_C * sqrt(CHEZY_DEPTH / CHEZY_SLOPE) / CHEZY_G
        @test chezy_spinup_velocity(T, CHEZY_C, CHEZY_DEPTH, CHEZY_SLOPE) ≈ tanh(1.0) * U rtol = TOL_IDENTITY
        # the profile has the Chezy velocity as its depth average and no stress at the surface
        nu = CHEZY_NU
        prof(z) = chezy_viscous_profile(z, CHEZY_C, CHEZY_DEPTH, CHEZY_SLOPE, nu)
        n = 2000   # Simpson points: exact for the parabola
        h = CHEZY_DEPTH / n
        simpson = h / 3 * (prof(0.0) + prof(CHEZY_DEPTH) + sum((isodd(j) ? 4 : 2) * prof(j * h) for j in 1:(n - 1)))
        @test simpson / CHEZY_DEPTH ≈ U rtol = TOL_IDENTITY
        # log-law roughness of the same Chezy coefficient: (sqrt(g)/kappa)(ln(H/z0) - 1) = C
        z0 = chezy_roughness_length(CHEZY_C, CHEZY_DEPTH; g=CHEZY_G)
        @test sqrt(CHEZY_G) / SweTransport2DV.KE_KARMAN * (log(CHEZY_DEPTH / z0) - 1) ≈ CHEZY_C rtol = TOL_IDENTITY
    end

    @testset "test_verify_chezy_column_depth_integrated_balance" begin
        # Flux form, Chezy only: the vertical terms move momentum within the column, so with Ubar = dZ'u / H
        #   dZ'u = dZ'uhat - dt gamma_mean Ubar   and   dZ'r = H / (1 + dt gamma_mean / H)   (1'A = dZ', file header),
        # and the Sherman-Morrison solution solves the rank-one-updated system A' u = dZ .* uhat exactly.
        rng = MersenneTwister(14)
        for m in (1, 2, 7), dt in (0.1, 10.0)
            dZ = 0.1 .+ rand(rng, m)
            W = 0.2 .* randn(rng, m + 1)
            nu = 0.01 .* rand(rng, m + 1)
            uhat = randn(rng, m)
            gamma_mean = 0.05
            H = sum(dZ)
            u, r, work = zeros(m), zeros(m), zeros(m, 4)
            SweTransport2DV.implicit_vertical_column!(u, r, uhat, dZ, W, nu, 0.0, dt, m, view(work, :, 1),
                view(work, :, 2), view(work, :, 3); gamma_mean=gamma_mean, zprime=view(work, :, 4))
            @test isapprox(sum(dZ .* u), sum(dZ .* uhat) - dt * gamma_mean * sum(dZ .* u) / H; atol=TOL_IDENTITY)
            @test isapprox(sum(dZ .* r), H / (1 + dt * gamma_mean / H); atol=TOL_IDENTITY)
            # dense check: A from its inverse (columns A^{-1} e_j from uhat = e_j / dZ_j), A' = A + e_1 v'
            Ainv = zeros(m, m)
            for j in 1:m
                e = zeros(m)
                e[j] = 1.0 / dZ[j]
                SweTransport2DV.implicit_vertical_column!(view(Ainv, :, j), r, e, dZ, W, nu, 0.0, dt, m,
                    view(work, :, 1), view(work, :, 2), view(work, :, 3))
            end
            A = inv(Ainv)
            A[1, :] .+= dt * gamma_mean .* dZ ./ H
            @test maximum(abs.(u .- A \ (dZ .* uhat))) < TOL_IDENTITY * max(1.0, maximum(abs, uhat))
        end
        @test_throws ArgumentError SweTransport2DV.implicit_vertical_column!(zeros(1), zeros(1), ones(1), ones(1), zeros(2),
            zeros(2), 0.0, 1.0, 1, zeros(1), zeros(1), zeros(1); gamma_mean=0.1)
    end

    @testset "test_verify_chezy_channel_profile_is_steady_state" begin
        # The discrete parabola with the Chezy mean is a steady state of both paths (to round-off), and a perturbed
        # profile relaxes to it; the free surface stays flat.
        for implicit in (false, true)
            nz = 10
            P = chezy_discrete_profile(nz)
            model = chezy_channel(; implicit=implicit, dt=2.0, nz=nz)
            for i in axes(model.state.u, 2)
                model.state.u[:, i] .= P
            end
            for _ in 1:200
                step_forward!(model)
            end
            @test maximum(abs.(model.state.u .- P)) < TOL_MODEL_IDENTITY
            @test maximum(abs, model.state.eta) < TOL_MODEL_IDENTITY

            model = chezy_channel(; implicit=implicit, dt=2.0, nz=nz)
            for i in axes(model.state.u, 2)
                model.state.u[:, i] .= P .+ 0.1 .* sin.(pi .* (1:nz) ./ nz)
            end
            for _ in 1:round(Int, CHEZY_RELAX_TIME / 2.0)
                step_forward!(model)
            end
            @test maximum(abs.(model.state.u .- P)) < CHEZY_RELAX_TOL
        end
    end

    @testset "test_verify_chezy_spinup_order_in_time" begin
        # Spin-up from rest against U(t) = U_eq tanh(t/T): the explicit friction (forward Euler, c_f |U^n| U^n) is
        # first order; the implicit one, c_f |U^n| U^{n+1}, approximates U^2 by the geometric mean of the two levels
        # (midpoint-like), second order.
        exact = chezy_spinup_velocity(CHEZY_SPINUP_TIME, CHEZY_C, CHEZY_DEPTH, CHEZY_SLOPE)
        for (implicit, dts, order) in ((false, (2.0, 1.0, 0.5), 1.0), (true, (20.0, 10.0, 5.0), 2.0))
            errors = map(dts) do dt
                model = chezy_channel(; implicit=implicit, dt=dt)
                for _ in 1:round(Int, CHEZY_SPINUP_TIME / dt)
                    step_forward!(model)
                end
                abs(depth_mean(model) - exact)
            end
            observed = log2.(errors[1:end-1] ./ errors[2:end])
            @test all(o -> abs(o - order) < CHEZY_ORDER_TOL, observed)
        end
    end

    @testset "test_verify_top_row_uniform_flow_with_implicit_vertical_advection" begin
        # Variable top layer h_top != dz, implicit vertical terms: for a uniform u the explicit top-row correction must
        # be the vertical flux that the implicit column moves into the top row, +u W / h_top with the column thickness
        # h_top (implicit_vertical_column!, flux form: dZ_top (u - uhat) = dt W u_{nz-1} for W > 0), so that the two
        # cancel. Issue #14: with dz instead of h_top a uniform flow had the tendency u W (1/dz - 1/h_top).
        nz, nx, dx, dz, U = 4, 6, 10.0, 0.5, 0.8
        u = fill(U, nz, nx + 1)
        w = zeros(nz + 1, nx)
        w[nz, :] .= 0.01 .* (1:nx)   # m/s, at the interface below the top layer
        h_top = [dz + 0.3 * sin(i) for i in 1:(nx + 1)]
        ws_adv = SweTransport2DV.AdvectionWorkspace(nz, nx + 1)
        ws_int = SweTransport2DV.InterpolationWorkspace(nz, nx)
        out = zeros(nz, nx + 1)
        advection_term_u!(out, u, w, dx, dz, "upwind1", ws_adv, ws_int; top_row_advective_form=true, vertical=false,
            h_top_vertical=h_top)
        out_dz = zeros(nz, nx + 1)
        advection_term_u!(out_dz, u, w, dx, dz, "upwind1", ws_adv, ws_int; top_row_advective_form=true, vertical=false)
        for f in 2:nx
            W = 0.5 * (w[nz, f - 1] + w[nz, f])   # the column's W below the top layer
            @test abs(out[nz, f] - U * W / h_top[f]) < TOL_IDENTITY
            @test abs(out_dz[nz, f] - U * W / h_top[f]) > 1e3 * TOL_IDENTITY   # the former scaling with dz
            @test maximum(abs, out[1:(nz - 1), f]) < TOL_IDENTITY             # no vertical part below (implicit)
        end
    end

    @testset "test_verify_level_boundary_keeps_open_channel_equilibrium" begin
        # Normal flow with q = 4 m^2/s at the inlet and eta_R = 0 at the outlet is a steady state of the variable top
        # layer, the fixed grid and the wetting-drying step, with both momentum schemes (the boundary face has an open,
        # divergence-free control volume: with a closed one it was unstable, ~0.012 1/s independently of dt).
        for v in OPEN_CHANNEL_VARIANTS, scheme in ("upwind1", "tvd2")
            model, P = open_channel(; vt=v.vt, wd=v.wd, scheme=scheme)
            for _ in 1:500
                step_forward!(model)
            end
            @test maximum(abs.(model.state.u .- P)) < TOL_MODEL_IDENTITY
            @test maximum(abs, model.state.eta) < TOL_MODEL_IDENTITY
        end
    end

    @testset "test_verify_level_boundary_converges_to_normal_flow" begin
        # From 0.8 times the normal flow the open channel returns to it; the boundary transports reproduce the volume
        # change of every step (to round-off) and the inflow is the prescribed discharge.
        for v in OPEN_CHANNEL_VARIANTS
            model, P = open_channel(; vt=v.vt, wd=v.wd, start=0.8)
            budget_error = 0.0
            for _ in 1:round(Int, OPEN_CHANNEL_RELAX_TIME / model.config.dt)
                V0 = channel_volume(model)
                step_forward!(model)
                Qin, Qout = boundary_discharges(model)
                budget_error = max(budget_error, abs(channel_volume(model) - V0 - model.config.dt * (Qin - Qout)))
            end
            @test budget_error < TOL_MODEL_IDENTITY * CHEZY_DEPTH * CHANNEL_LENGTH
            @test maximum(abs.(model.state.u .- P)) < CHEZY_RELAX_TOL
            @test maximum(abs, model.state.eta) < CHEZY_RELAX_TOL
            v.wd || @test model.last_kinematic_mismatch < TOL_MODEL_IDENTITY
        end
    end

    @testset "test_verify_level_boundary_fills_lake_to_prescribed_level" begin
        # Closed inlet, no slope: a lake at the boundary level stays at rest exactly; a lake below it fills up to it.
        for v in OPEN_CHANNEL_VARIANTS
            model, _ = open_channel(; vt=v.vt, wd=v.wd, right_level=0.1)
            model.config.body_force_x = 0.0
            fill!(model.state.u, 0.0)
            set_x_inflow_outflow!(model, zeros(model.config.nz); open_right=false, right_level=0.1)
            if v.wd
                initialize_water_level!(model, CHEZY_DEPTH + 0.1)
            else
                fill!(model.state.eta, 0.1)
            end
            for _ in 1:200
                step_forward!(model)
            end
            @test maximum(abs.(model.state.eta .- 0.1)) < TOL_MODEL_IDENTITY
            @test maximum(abs, model.state.u) < TOL_MODEL_IDENTITY

            model, _ = open_channel(; vt=v.vt, wd=v.wd, right_level=0.1)
            model.config.body_force_x = 0.0
            fill!(model.state.u, 0.0)
            set_x_inflow_outflow!(model, zeros(model.config.nz); open_right=false, right_level=0.1)
            for _ in 1:round(Int, LAKE_FILL_TIME / model.config.dt)
                step_forward!(model)
            end
            @test maximum(abs.(model.state.eta .- 0.1)) < LAKE_FILL_TOL
        end
    end

    @testset "test_verify_discharge_boundary_transport" begin
        # The inflow face carries exactly q with the face thicknesses of time n, also while the inlet level moves.
        for v in OPEN_CHANNEL_VARIANTS
            model, _ = open_channel(; vt=v.vt, wd=v.wd, start=0.8)
            for _ in 1:20
                step_forward!(model)
                if v.wd
                    @test sum(model.wetdry.hf[:, 1] .* model.state.u[:, 1]) ≈ CHANNEL_DISCHARGE rtol = TOL_IDENTITY
                else
                    nz, dz = model.config.nz, model.dz
                    h_top = v.vt === false ? dz : dz + model.ws.eta_old[1]
                    @test sum(model.state.u[1:(nz - 1), 1]) * dz + h_top * model.state.u[nz, 1] ≈ CHANNEL_DISCHARGE rtol = TOL_IDENTITY
                end
            end
        end
    end

    @testset "test_validate_backwater_curve_matches_gradually_varied_flow" begin
        # Outlet level above (M1) and below (M2) the normal depth: the steady depth follows the gradually-varied-flow
        # equation with the Chezy friction (RK4 reference). Started from the reference profile (a level step at the
        # outlet sends a bore that, reflected at the discharge inlet, leaves the accuracy range of the variable top
        # layer, |eta| < dz). Variable top layer and wetting-drying step; the fixed grid keeps the depth H in the
        # continuity equation and is no backwater model.
        nx, nz = 50, 8
        dx = CHANNEL_LENGTH / nx
        xc = ((1:nx) .- 0.5) .* dx
        for eta_R in (BACKWATER_LEVEL, -BACKWATER_LEVEL), wd in (false, true)
            href = [chezy_backwater_depth(x, CHANNEL_DISCHARGE, CHEZY_C, CHEZY_SLOPE, CHANNEL_LENGTH, CHEZY_DEPTH + eta_R)
                    for x in xc]
            model, P = open_channel(; nx=nx, nz=nz, dt=dx / 2, wd=wd, right_level=eta_R)
            if wd
                initialize_water_level!(model, href)
            else
                model.state.eta .= href .- CHEZY_DEPTH
            end
            for i in axes(model.state.u, 2)
                model.state.u[:, i] .= P .* (CHEZY_DEPTH / href[min(i, nx)])
            end
            for _ in 1:round(Int, OPEN_CHANNEL_RELAX_TIME / model.config.dt)
                step_forward!(model)
            end
            @test maximum(abs.(CHEZY_DEPTH .+ model.state.eta .- href)) < BACKWATER_TOL
        end
    end

    @testset "test_validate_sloped_bed_channel_normal_flow" begin
        # Real bed slope instead of the body force (wetting-drying cut cells; the bed falls by i_b L = 0.1 m, a quarter
        # of a layer): with q at the inlet and the normal depth at the outlet the flow is uniform, depth H and U = 1 m/s.
        nx, nz = 50, 11
        dx = CHANNEL_LENGTH / nx
        grid_depth = CHEZY_DEPTH + CHEZY_SLOPE * CHANNEL_LENGTH
        xc = ((1:nx) .- 0.5) .* dx
        bed = CHEZY_SLOPE .* (CHANNEL_LENGTH .- xc)
        cfg = ModelConfig(nx=nx, nz=nz, length=CHANNEL_LENGTH, depth=grid_depth, dt=dx / 2, implicit_vertical_terms=true,
            chezy_coefficient=CHEZY_C, nu=CHEZY_NU, rayleigh_damping=0.0, apply_froude_limiter=false,
            momentum_advection_scheme="tvd2", wetting_drying=true, bed_level=bed, write_plots=false)
        model = Model(cfg)
        fill!(model.state.rho, 1000.0)
        refresh_density_metadata!(model)
        initialize_water_level!(model, bed .+ CHEZY_DEPTH)
        # inflow shape: the steady parabola at the centres of the (cut) layers of the inlet face, height above its bed
        h = zeros(nz)
        layer_thicknesses!(h, CHEZY_DEPTH + bed[1], model.wetdry.bed_face[1], model.dz)
        z_centre = cumsum(h) .- h ./ 2
        shape = [h[k] > 0.0 ? chezy_viscous_profile(z_centre[k], CHEZY_C, CHEZY_DEPTH, CHEZY_SLOPE, CHEZY_NU) : 0.0 for k in 1:nz]
        for i in axes(model.state.u, 2)
            model.state.u[:, i] .= shape
        end
        # outlet: normal depth above the bed extrapolated to x = L (the bed of the boundary face is that of cell nx)
        set_x_inflow_outflow!(model, shape; left_discharge=CHANNEL_DISCHARGE, right_level=CHEZY_DEPTH - grid_depth)
        for _ in 1:round(Int, OPEN_CHANNEL_RELAX_TIME / cfg.dt)
            step_forward!(model)
        end
        depth = water_depth(model)
        @test maximum(abs.(depth .- CHEZY_DEPTH)) < SLOPED_BED_TOL * CHEZY_DEPTH
        U_mid = sum(model.wetdry.hf[:, nx ÷ 2] .* model.state.u[:, nx ÷ 2]) / depth[nx ÷ 2]
        @test abs(U_mid - chezy_equilibrium_velocity(CHEZY_C, CHEZY_DEPTH, CHEZY_SLOPE)) < SLOPED_BED_TOL
    end

    @testset "test_verify_open_boundary_config_checks" begin
        model, P = open_channel()
        @test_throws ArgumentError set_x_inflow_outflow!(model, P; right_level=0.0, right_profile=P)
        @test_throws ArgumentError set_x_inflow_outflow!(model, -P; left_discharge=4.0)
        @test_throws ArgumentError set_x_inflow_outflow!(model, P; right_level=NaN)
        for cfg in (ModelConfig(nx=8, nz=4, implicit_vertical_terms=false),
                    ModelConfig(nx=8, nz=4, implicit_vertical_terms=true, use_hydrostatic=false),
                    ModelConfig(nx=8, nz=4, implicit_vertical_terms=true, periodic_x=true, pressure_solver_method="sparse",
                        variable_top_layer=false))
            @test_throws ArgumentError set_x_inflow_outflow!(Model(cfg), ones(4); right_level=0.0)
        end
    end

    @testset "test_verify_chezy_config_checks" begin
        base = (nx=4, nz=4, periodic_x=true, pressure_solver_method="sparse", variable_top_layer=false)
        @test_throws ArgumentError Model(ModelConfig(; base..., chezy_coefficient=-1.0))
        @test_throws ArgumentError Model(ModelConfig(; base..., chezy_coefficient=50.0, bed_roughness_length=1e-3))
        @test_throws ArgumentError Model(ModelConfig(; base..., chezy_coefficient=50.0, turbulence_closure="k_epsilon"))
        @test_throws ArgumentError Model(ModelConfig(nx=4, nz=4, chezy_coefficient=50.0, bed_level=[0.0, 1.0, 1.0, 0.0]))
        @test_throws ArgumentError Model(ModelConfig(nx=4, nz=4, chezy_coefficient=50.0, wetting_drying=true))
        @test_throws ArgumentError chezy_roughness_length(0.0, 4.0)
    end
end
