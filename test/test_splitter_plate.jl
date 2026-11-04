# Inflow of density, k and eps per layer at the left boundary face, and a reduced splitter plate (issue #42): two streams
# of different velocity and density enter a channel, wetting-drying step, k-epsilon, relaxed outflow. Docs section 30.

const SP_LENGTH = 4.0           # m
const SP_NX = 40                # -, dx = 0.1 m
const SP_LAYERS = (4, 4)        # -, layers of the lower and the upper stream
const SP_DZ = 0.05              # m
const SP_U = (0.3, 0.5)         # m/s, lower and upper stream
const SP_RHO = (1031.0, 1016.0) # kg/m^3, lower and upper stream
const SP_K = (2e-4, 1e-4)       # m^2/s^2, inflow k of the streams
const SP_EPS = (2e-5, 5e-6)     # m^2/s^3, inflow eps of the streams
const SP_Z0 = 4e-5              # m, bed roughness length (Chezy 65 at 0.574 m)
const SP_COURANT = 0.6          # -, u_max dt / dx
const SP_TAU = 20.0             # s, relaxation time of the outflow
const SP_RUN_TIME = 60.0        # s, 7.5 transit times of the upper stream
const SP_STEADY_WINDOW = 20.0   # s, mean discharge over the end of the run
const SP_DISCHARGE_RTOL = 1e-2  # -, mean outflow against the inflow in the steady state

sp_nz() = sum(SP_LAYERS)
sp_layers(values) = vcat(fill(values[1], SP_LAYERS[1]), fill(values[2], SP_LAYERS[2]))

# Two-stream channel; closure "k_epsilon" or "none" (then nu = nu_none and no bed friction)
function sp_model(; closure="k_epsilon", nu_none=0.0)
    nz = sp_nz()
    depth = nz * SP_DZ
    cfg = ModelConfig(; length=SP_LENGTH, depth=depth, nx=SP_NX, nz=nz, dt=SP_COURANT * (SP_LENGTH / SP_NX) / maximum(SP_U),
        use_hydrostatic=true, nu=closure == "none" ? nu_none : 1e-6, rho0=sum(SP_RHO) / 2, turbulence_closure=closure,
        bed_roughness_length=closure == "none" ? 0.0 : SP_Z0, rayleigh_damping=0.0, apply_froude_limiter=false,
        momentum_advection_scheme="tvd2", transport_advection_scheme="tvd2", wetting_drying=true,
        implicit_vertical_terms=true, theta=1.0, write_plots=false)
    model = Model(cfg)
    rho_in = sp_layers(SP_RHO)
    u_in = sp_layers(SP_U)
    # WHY the initial density of the upper stream everywhere: the inflow must bring the lower density in by itself
    initialize_water_level!(model, depth; rho=SP_RHO[2])
    for k in 1:nz
        model.state.u[k, :] .= u_in[k]
    end
    kw = closure == "k_epsilon" ? (; left_k=sp_layers(SP_K), left_eps=sp_layers(SP_EPS)) : (;)
    set_x_inflow_outflow!(model, u_in; left_discharge=sum(u_in) * SP_DZ, left_rho=rho_in, kw..., right_level=0.0,
        right_free=true, right_relaxation_time=SP_TAU)
    return model, rho_in
end

# Runs `time` (s) and returns the excess-density budget error (kg/m, against SP_RHO[2]), the cumulated excess inflow
# (kg/m), the density range over all steps (kg/m^3) and the mean inflow and outflow (m^2/s) over SP_STEADY_WINDOW
function sp_run!(model, rho_in, time)
    wd = model.wetdry
    nz, nx = size(model.state.rho)
    dt = model.config.dt
    mass(m) = sum(m.state.rho .* m.wetdry.h) * m.dx
    volume(m) = sum(m.wetdry.h) * m.dx
    mass0, volume0 = mass(model), volume(model)
    flux_mass = flux_volume = excess = 0.0
    lo, hi = Inf, -Inf
    rho_old = similar(model.state.rho)
    q_in = q_out = 0.0
    window = 0
    steps = round(Int, time / dt)
    for n in 1:steps
        copyto!(rho_old, model.state.rho)
        step_forward!(model)
        # face densities of the transport: prescribed at the inflow face, upwind (the last cell) at the outflow face
        for k in 1:nz
            q_l, q_r = wd.q[k, 1], wd.q[k, nx + 1]
            rho_l = q_l > 0 ? rho_in[k] : rho_old[k, 1]
            flux_mass += dt * (q_l * rho_l - q_r * rho_old[k, nx])
            flux_volume += dt * (q_l - q_r)
            excess += dt * q_l * (rho_l - SP_RHO[2])
        end
        wet = wd.h .> 0
        lo, hi = min(lo, minimum(model.state.rho[wet])), max(hi, maximum(model.state.rho[wet]))
        if n > steps - round(Int, SP_STEADY_WINDOW / dt)
            q_in += sum(view(wd.q, :, 1))
            q_out += sum(view(wd.q, :, nx + 1))
            window += 1
        end
    end
    budget = (mass(model) - mass0 - flux_mass) - SP_RHO[2] * (volume(model) - volume0 - flux_volume)
    return budget, excess, (lo, hi), q_in / window, q_out / window
end

@testset "Splitter plate inflow (issue #42)" begin
    @testset "test_verify_inflow_density_per_layer_enters_the_first_cell" begin
        # operator level: the transport step carries left_rho[k] through the inlet face (first order there: no cell
        # behind the face), i.e. the first cell changes by dt/dx q_k (rho_in,k - rho_1,k)/h against the inflow with the
        # density of the first cell, and no other cell changes
        # WHY not a steady run: the density difference drives a baroclinic flow at the interface, so no closure-free
        # run is pure advection
        model, rho_in = sp_model()
        step_forward!(model)
        rho1 = copy(model.state.rho)
        with = deepcopy(model)
        without = deepcopy(model)
        without.x_boundary_rho_left = nothing
        SweTransport2DV._wd_transport_density!(with, with.wetdry)
        SweTransport2DV._wd_transport_density!(without, without.wetdry)
        wd = model.wetdry
        dt, dx = model.config.dt, model.dx
        for k in 1:sp_nz()
            expected = dt / dx * wd.q[k, 1] * (rho_in[k] - rho1[k, 1]) / wd.h_new[k, 1]
            @test isapprox(with.state.rho[k, 1] - without.state.rho[k, 1], expected; atol=TOL_IDENTITY * rho_in[k])
        end
        @test with.state.rho[:, 2:end] == without.state.rho[:, 2:end]
        @test any(abs.(rho_in .- rho1[:, 1]) .> 1.0)   # the check is not trivial: the inflow differs from the first cell
    end

    @testset "test_verify_inflow_k_eps_per_layer_enter_the_first_cell" begin
        # operator level: the upwind advection of the step carries left_k / left_eps through the inlet face, i.e. the
        # first cell changes by dt/dx q_k (phi_in - phi_1)/h against the zero-gradient inflow
        model, _ = sp_model()
        step_forward!(model)
        wd, ke = model.wetdry, model.turbulence
        dt, dx = model.config.dt, model.dx
        for (phi, inflow, floor, sigma) in ((ke.k, wd.left_k, SweTransport2DV.KE_K_MIN, SweTransport2DV.KE_SIGMA_K),
            (ke.eps, wd.left_eps, SweTransport2DV.KE_EPS_MIN, SweTransport2DV.KE_SIGMA_EPS))
            with = similar(phi)
            without = similar(phi)
            SweTransport2DV._wd_advect_turbulence!(with, phi, model, wd, floor, sigma; inflow=inflow)
            SweTransport2DV._wd_advect_turbulence!(without, phi, model, wd, floor, sigma)
            for k in 1:sp_nz()
                expected = dt / dx * wd.q[k, 1] * (inflow[k] - phi[k, 1]) / wd.h_new[k, 1]
                @test isapprox(with[k, 1] - without[k, 1], expected; rtol=TOL_IDENTITY, atol=TOL_IDENTITY * abs(phi[k, 1]))
            end
            @test with[:, 2:end] == without[:, 2:end]
        end
    end

    @testset "test_verify_two_streams_density_bounds_and_salt_budget" begin
        # k-epsilon mixes the two streams: the density stays within the two inflow densities, and the mass of the
        # density excess over the upper stream changes exactly by the boundary fluxes
        model, rho_in = sp_model()
        budget, excess, (lo, hi), _, _ = sp_run!(model, rho_in, SP_RUN_TIME)
        @test SP_RHO[2] - TOL_MODEL_IDENTITY * SP_RHO[2] <= lo
        @test hi <= SP_RHO[1] + TOL_MODEL_IDENTITY * SP_RHO[1]
        @test abs(budget) <= TOL_MODEL_IDENTITY * excess
    end

    @testset "test_validate_two_streams_steady_discharge" begin
        model, rho_in = sp_model()
        _, _, _, q_in, q_out = sp_run!(model, rho_in, SP_RUN_TIME)
        @test abs(q_out - q_in) <= SP_DISCHARGE_RTOL * q_in
        @test all(isfinite, model.state.u)
    end

    @testset "test_verify_inflow_per_layer_argument_checks" begin
        model, rho_in = sp_model()
        u_in = sp_layers(SP_U)
        nz = sp_nz()
        q = sum(u_in) * SP_DZ
        @test_throws ArgumentError set_x_inflow_outflow!(model, u_in; left_discharge=q, left_rho=rho_in[1:end-1])
        @test_throws ArgumentError set_x_inflow_outflow!(model, u_in; left_discharge=q, left_k=fill(1e-4, nz))
        @test_throws ArgumentError set_x_inflow_outflow!(model, u_in; left_discharge=q, left_k=fill(1e-4, nz),
            left_eps=fill(1e-5, nz - 1))
        @test_throws ArgumentError set_x_inflow_outflow!(model, u_in; left_discharge=q, left_k=fill(1e-4, nz),
            left_eps=fill(0.0, nz))
        plain, _ = sp_model(; closure="none")
        @test_throws ArgumentError set_x_inflow_outflow!(plain, u_in; left_discharge=q, left_k=fill(1e-4, nz),
            left_eps=fill(1e-5, nz))
        clear_x_inflow_outflow!(model)
        @test model.wetdry.left_k === nothing && model.wetdry.left_eps === nothing && model.x_boundary_rho_left === nothing
    end
end
