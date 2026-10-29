include("test_common.jl")

# Open boundaries for transcritical flow (issue #46): free outflow at the right boundary (right_free, radiation condition
# for subcritical outflow, extrapolation for supercritical outflow) and supercritical inflow (left_discharge with
# left_level). See docs/TESTS_TECHNICAL_DOCUMENTATION.md, section 22.

const OB_G = 9.81                 # m/s^2, ModelConfig default
const OB_C = 50.0                 # m^(1/2)/s, Chezy coefficient
const OB_NU = 0.02                # m^2/s, of the order kappa u_* H / 6 of both channels
const OB_RELAX_TOL = 1e-6         # m/s and m, distance from normal flow after the relaxation run
const OB_FIXED_GRID_RELAX_TOL = 1e-4   # m/s, the same on the fixed grid: the neutral level mode relaxes slower (3e-5)
const OB_REFLECTION_MAX = 0.02    # -, reflected amplitude over the incoming one at the free outflow (measured 0.007)
const OB_REFLECTION_LEVEL_MIN = 0.4   # -, the same with a prescribed level (complete reflection, damped by theta = 0.7)
const OB_SLOPED_BED_TOL = 1e-3    # m, depth deviation from normal flow over the real slope (measured 3.4e-4)

# Supercritical channel: H = 1 m, i_b = 0.01, U = C sqrt(H i_b) = 5 m/s, Fr = 1.6 (theta >= theta_for_froude(1.6) = 0.81).
const SC_DEPTH, SC_SLOPE, SC_LENGTH = 1.0, 0.01, 200.0
# Subcritical channel of issue #14: H = 4 m, i_b = 1e-4, U = 1 m/s, Fr = 0.16.
const SUB_DEPTH, SUB_SLOPE, SUB_LENGTH = 4.0, 1e-4, 1000.0

function ob_profile(H, slope, nz)
    dz = H / nz
    P = [chezy_viscous_profile((k - 0.5) * dz, OB_C, H, slope, OB_NU) for k in 1:nz]
    return P .+ (chezy_equilibrium_velocity(OB_C, H, slope) - sum(P) / nz)
end

# Channel with the slope as body force on a flat bed, discharge inlet, free outflow; inlet level for the supercritical case.
function ob_channel(; supercritical, vt=nothing, wd=false, start=1.0, inlet_level=supercritical)
    H, slope, L = supercritical ? (SC_DEPTH, SC_SLOPE, SC_LENGTH) : (SUB_DEPTH, SUB_SLOPE, SUB_LENGTH)
    nx, nz = supercritical ? (40, 8) : (50, 8)
    dt, theta = supercritical ? (0.5, 0.9) : (10.0, 0.7)
    cfg = ModelConfig(nx=nx, nz=nz, length=L, depth=H, dt=dt, theta=theta, implicit_vertical_terms=true,
        chezy_coefficient=OB_C, nu=OB_NU, body_force_x=OB_G * slope, rayleigh_damping=0.0, apply_froude_limiter=false,
        momentum_advection_scheme="tvd2", variable_top_layer=wd ? nothing : vt, wetting_drying=wd,
        bed_level=wd ? zeros(nx) : nothing, write_plots=false)
    model = Model(cfg)
    fill!(model.state.rho, 1000.0)
    refresh_density_metadata!(model)
    P = ob_profile(H, slope, nz)
    for i in axes(model.state.u, 2)
        model.state.u[:, i] .= start .* P
    end
    q = chezy_equilibrium_velocity(OB_C, H, slope) * H
    set_x_inflow_outflow!(model, P; left_discharge=q, left_level=inlet_level ? 0.0 : nothing, right_free=true)
    return model, P
end

# Right-going long wave of amplitude A (m) in still water of depth H (m), left wall; right boundary free or level.
function ob_wave(; wd, free, nx=100, sigma=50.0, A=0.01, H=4.0, L=1000.0)
    c = sqrt(OB_G * H)
    dx = L / nx
    cfg = ModelConfig(nx=nx, nz=4, length=L, depth=H, dt=dx / c, theta=0.7, implicit_vertical_terms=true, nu=1e-6,
        rayleigh_damping=0.0, apply_froude_limiter=false, momentum_advection_scheme="tvd2", wetting_drying=wd,
        bed_level=wd ? zeros(nx) : nothing, write_plots=false)
    model = Model(cfg)
    fill!(model.state.rho, 1000.0)
    refresh_density_metadata!(model)
    x0 = 0.6 * L
    xc = ((1:nx) .- 0.5) .* dx
    eta0 = A .* exp.(-((xc .- x0) ./ sigma) .^ 2)
    wd ? initialize_water_level!(model, H .+ eta0) : (model.state.eta .= eta0)
    for f in 1:(nx + 1)
        model.state.u[:, f] .= A * exp(-(((f - 1) * dx - x0) / sigma)^2) * sqrt(OB_G / H)
    end
    free ? set_x_inflow_outflow!(model, zeros(4); right_free=true) :
        set_x_inflow_outflow!(model, zeros(4); open_right=false, right_level=0.0)
    # until the wave has left and a reflection would have travelled back to the middle of the basin
    nsteps = round(Int, (L - x0 + 0.5 * L) / c / cfg.dt)
    return model, A, nsteps
end

water_volume_of(model) = model.wetdry === nothing ? sum(model.state.eta) * model.dx : sum(water_depth(model)) * model.dx
function theta_boundary_discharges(model)
    wd = model.wetdry
    wd === nothing || return (sum(wd.q[:, 1]), sum(wd.q[:, end]))
    nz, dz, ws = model.config.nz, model.dz, model.ws
    vt = use_variable_top_layer(model.config)
    Q(f) = sum(ws.u_theta[1:(nz - 1), f]) * dz + (vt ? ws.h_top_face[f] : dz) * ws.u_theta[nz, f]
    return (Q(1), Q(model.config.nx + 1))
end

@testset "Open boundaries for transcritical flow (issue #46)" begin
    @testset "test_verify_supercritical_channel_with_inlet_level_and_free_outflow" begin
        # Normal flow at Fr = 1.6 with discharge and level at the inlet and a free outflow is a steady state, and a
        # perturbed start (0.8 times the velocity) returns to it: supercritical, the perturbation leaves downstream.
        # Fixed grid: no inlet level (left_level needs the level-dependent inlet thicknesses); the velocity returns to
        # normal flow, the mean level is a neutral mode there (depth H at every face) and is not checked.
        for (vt, wd, inlet_level) in ((nothing, false, true), (nothing, true, true), (false, false, false))
            model, P = ob_channel(; supercritical=true, vt=vt, wd=wd, inlet_level=inlet_level)
            for _ in 1:200
                step_forward!(model)
            end
            @test maximum(abs.(model.state.u .- P)) < TOL_MODEL_IDENTITY
            @test maximum(abs, model.state.eta) < TOL_MODEL_IDENTITY
            model, P = ob_channel(; supercritical=true, vt=vt, wd=wd, inlet_level=inlet_level, start=0.8)
            for _ in 1:2000
                step_forward!(model)
            end
            @test maximum(abs.(model.state.u .- P)) < (vt === false ? OB_FIXED_GRID_RELAX_TOL : OB_RELAX_TOL)
            if inlet_level
                @test maximum(abs, model.state.eta) < OB_RELAX_TOL
                @test surface_froude_number(model)[1] ≈ chezy_equilibrium_velocity(OB_C, SC_DEPTH, SC_SLOPE) / sqrt(OB_G * SC_DEPTH) rtol = OB_RELAX_TOL
            end
        end
    end

    @testset "test_verify_free_outflow_subcritical_converges_to_normal_flow" begin
        # Subcritical channel of issue #14 with a discharge inlet and a free outflow: no level is prescribed anywhere,
        # the discharge and the friction set the normal depth. Steady state to round-off; from 0.8 times the normal flow
        # within OB_RELAX_TOL after 40000 s (variable top layer, wetting and drying).
        for (vt, wd) in ((nothing, false), (nothing, true))
            model, P = ob_channel(; supercritical=false, vt=vt, wd=wd)
            for _ in 1:300
                step_forward!(model)
            end
            @test maximum(abs.(model.state.u .- P)) < TOL_MODEL_IDENTITY
            @test maximum(abs, model.state.eta) < TOL_MODEL_IDENTITY
            model, P = ob_channel(; supercritical=false, vt=vt, wd=wd, start=0.8)
            budget_error = 0.0
            for _ in 1:4000
                V0 = water_volume_of(model)
                step_forward!(model)
                Qin, Qout = theta_boundary_discharges(model)
                budget_error = max(budget_error, abs(water_volume_of(model) - V0 - model.config.dt * (Qin - Qout)))
            end
            @test maximum(abs.(model.state.u .- P)) < OB_RELAX_TOL
            @test maximum(abs, model.state.eta) < OB_RELAX_TOL
            @test budget_error < TOL_MODEL_IDENTITY * SUB_DEPTH * SUB_LENGTH
            wd || @test model.last_kinematic_mismatch < TOL_MODEL_IDENTITY
        end
    end

    @testset "test_validate_free_outflow_transmits_long_wave" begin
        # A right-going long wave leaves through the free outflow: the implicit radiation condition reflects < 2 %
        # (measured 0.7 %); a prescribed level reflects it completely (> 40 % left after the theta = 0.7 damping).
        for wd in (false, true)
            model, A, nsteps = ob_wave(; wd=wd, free=true)
            for _ in 1:nsteps
                step_forward!(model)
            end
            @test maximum(abs, model.state.eta) / A < OB_REFLECTION_MAX
            model, A, nsteps = ob_wave(; wd=wd, free=false)
            for _ in 1:nsteps
                step_forward!(model)
            end
            @test maximum(abs, model.state.eta) / A > OB_REFLECTION_LEVEL_MIN
        end
    end

    @testset "test_validate_free_outflow_over_real_bed_slope" begin
        # Issue #14 slope as a real bed (wetting-drying cut cells) with a free outflow: the outlet level relaxes to the
        # bed-parallel continuation, and the face bed is extrapolated to x = L (with the bed of the last column the
        # whole channel rose by 1.7 cm). Depth within OB_SLOPED_BED_TOL of normal flow (measured 3.4e-4 m).
        nx, nz = 50, 11
        dx = SUB_LENGTH / nx
        grid_depth = SUB_DEPTH + SUB_SLOPE * SUB_LENGTH
        xc = ((1:nx) .- 0.5) .* dx
        bed = SUB_SLOPE .* (SUB_LENGTH .- xc)
        cfg = ModelConfig(nx=nx, nz=nz, length=SUB_LENGTH, depth=grid_depth, dt=10.0, implicit_vertical_terms=true,
            chezy_coefficient=OB_C, nu=OB_NU, rayleigh_damping=0.0, apply_froude_limiter=false,
            momentum_advection_scheme="tvd2", wetting_drying=true, bed_level=bed, write_plots=false)
        model = Model(cfg)
        fill!(model.state.rho, 1000.0)
        refresh_density_metadata!(model)
        initialize_water_level!(model, bed .+ SUB_DEPTH)
        h = zeros(nz)
        layer_thicknesses!(h, SUB_DEPTH + bed[1], model.wetdry.bed_face[1], model.dz)
        zc = cumsum(h) .- h ./ 2
        shape = [h[k] > 0 ? chezy_viscous_profile(zc[k], OB_C, SUB_DEPTH, SUB_SLOPE, OB_NU) : 0.0 for k in 1:nz]
        for i in axes(model.state.u, 2)
            model.state.u[:, i] .= shape
        end
        set_x_inflow_outflow!(model, shape; left_discharge=SUB_DEPTH * chezy_equilibrium_velocity(OB_C, SUB_DEPTH, SUB_SLOPE),
            right_free=true)
        for _ in 1:3000
            step_forward!(model)
        end
        @test maximum(abs.(water_depth(model) .- SUB_DEPTH)) < OB_SLOPED_BED_TOL
    end

    @testset "test_verify_boundary_froude_warnings" begin
        # A prescribed level at a supercritical outflow and a prescribed inlet level with subcritical inflow over-specify
        # the flow: warned once per model. Free outflow and supercritical inflow with level: no warning.
        model, P = ob_channel(; supercritical=true)
        set_x_inflow_outflow!(model, P; left_discharge=5.0, left_level=0.0, right_level=0.0)
        @test_logs (:warn, r"supercritical outflow .* right_free=true") step_forward!(model)
        @test_logs step_forward!(model)
        model, P = ob_channel(; supercritical=false)
        set_x_inflow_outflow!(model, P; left_discharge=4.0, left_level=0.0, right_free=true)
        @test_logs (:warn, r"subcritical inflow .* prescribe the discharge only") step_forward!(model)
        model, _ = ob_channel(; supercritical=true)
        @test_logs step_forward!(model)
    end

    @testset "test_verify_open_boundary_46_config_checks" begin
        model, P = ob_channel(; supercritical=false)
        @test_throws ArgumentError set_x_inflow_outflow!(model, P; left_level=0.0)                       # no discharge
        @test_throws ArgumentError set_x_inflow_outflow!(model, P; right_free=true, right_level=0.0)
        @test_throws ArgumentError set_x_inflow_outflow!(model, P; right_free=true, right_profile=P)
        @test_throws ArgumentError set_x_inflow_outflow!(model, P; left_discharge=4.0, left_level=NaN)
        fixed, Pf = ob_channel(; supercritical=false, vt=false)
        @test_throws ArgumentError set_x_inflow_outflow!(fixed, Pf; left_discharge=4.0, left_level=0.0)  # fixed grid
        for cfg in (ModelConfig(nx=8, nz=4, implicit_vertical_terms=false),
                    ModelConfig(nx=8, nz=4, implicit_vertical_terms=true, use_hydrostatic=false))
            @test_throws ArgumentError set_x_inflow_outflow!(Model(cfg), ones(4); right_free=true)
        end
    end
end
