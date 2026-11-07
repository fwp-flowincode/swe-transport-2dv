# Turbulent plane jet with k-epsilon (issue #25): the thin-shear-layer reference (free-jet limit, momentum integral) and
# the 2DV model against it. Half-jet: the frictionless bed is the symmetry plane of the jet; a co-flow supplies the
# entrainment of the confined jet (docs/TESTS_TECHNICAL_DOCUMENTATION.md, section 36).
# Run with the full suite (runtests.jl) or on its own: julia --project=. test/test_plane_jet.jl

include(joinpath(@__DIR__, "test_common.jl"))

# ---- Case ------------------------------------------------------------------------------------------------------------
const PJ_B0 = 0.05                # m, slot height of the full jet (the half slot b0/2 lies on the bed)
const PJ_U0 = 0.5                 # m/s, slot velocity (U0 b0 / nu = 25000)
const PJ_COFLOW = 0.075           # m/s, co-flow (15 % of U0): suppresses the recirculation of the confined jet
const PJ_DEPTH = 1.2              # m, 24 b0
const PJ_LENGTH = 2.25            # m, 45 b0
const PJ_NU = 1e-6                # m^2/s
const PJ_INTENSITY = 0.05         # -, turbulence intensity of the inflow
const PJ_MIXING_LENGTH = 0.1      # -, inflow length scale / b0 (eps = C_mu^(3/4) k^(3/2) / l)
const PJ_RELAXATION = 10.0        # s, relaxation of the outflow level
const PJ_T_END = 150.0            # s, five flow-throughs of the co-flow (the start-up circulation is gone)
const PJ_T_AVERAGE = 30.0         # s, mean over the end of the run
const PJ_COURANT = 0.5            # -, U0 dt / dx
const PJ_STATIONS = (10, 20, 30)  # b0, comparison stations (self-similar range of the reference)

# ---- Bounds (measured values in the comments; section 36) -------------------------------------------------------------
const PJ_FREE_COFLOW = 0.02       # m/s, weak co-flow of the free-jet limit (momentum length J/Ua^2 = 31 m >> x)
const PJ_FREE_DEPTH = 8.0         # m
const PJ_FREE_FIT = (15, 40)      # b0, fit range of the free-jet limit
const PJ_S_RANGE = (0.095, 0.115) # -, measured plane jets 0.10-0.11 (Pope 2000); reference 0.0973 without the time-scale bound
const PJ_EXPONENT_TOL = 0.03      # -, decay exponent within this of -1/2 (reference -0.509; model, 3 cells, -0.514)
const PJ_EXPONENT_TOL_COARSE = 0.05   # -, same, non-hydrostatic with 2 cells (observed -0.535)
const PJ_MOMENTUM_RTOL = 0.05     # -, change of J over 10-30 b0 against the momentum integral (observed 2.9 %; the change
                                  # is 2.6 % of J, so 0.07 % of J; without the free-stream term it would be zero)
const PJ_GRID_RTOL = 0.02         # -, reference with half the cells (first-order march; observed up to 1.3 %)
# model against the reference, 3 cells across the half slot (hydrostatic) and 2 cells (non-hydrostatic)
const PJ_UC_RTOL = (0.05, 0.08)           # -, centreline excess (observed -4.0, -2.5, -1.0 %; 2 cells -6.8, -4.8, -3.8 %)
const PJ_HALF_WIDTH_RTOL = (0.10, 0.15)   # -, half-width (observed +7.5, +5.0, +1.6 %; 2 cells +11.1, +7.7, +2.4 %)
const PJ_J_RTOL = (0.03, 0.03)            # -, excess momentum flux (observed within 0.5 %; 2 cells within 2.0 %)
const PJ_BACKFLOW_MAX = 0.01      # -, smallest velocity excess / U0 above the jet (no recirculation)
const PJ_MODE_RTOL = 0.03         # -, non-hydrostatic against hydrostatic on the same grid (2 cells: within 1.8 % in the
                                  # centreline excess, 2.4 % in J)
const PJ_EDGE_FACTOR = 3.0        # -, edge of the jet at this many half-widths (Gaussian excess 0.2 % of the centreline)
const PJ_EDGE_ITERATIONS = 2      # -, iterations of the edge and the half-width

pj_k(u) = 1.5 * (PJ_INTENSITY * u)^2                                            # m^2/s^2
pj_eps(k) = SweTransport2DV.KE_C_MU^0.75 * k^1.5 / (PJ_MIXING_LENGTH * PJ_B0)    # m^2/s^3
pj_inlet(coflow) = (k_slot=pj_k(PJ_U0), eps_slot=pj_eps(pj_k(PJ_U0)), k_ambient=pj_k(coflow), eps_ambient=pj_eps(pj_k(coflow)))

function pj_reference(; coflow=PJ_COFLOW, depth=PJ_DEPTH, x_end=maximum(PJ_STATIONS) * PJ_B0, kwargs...)
    return plane_jet_thin_shear_layer(; velocity=PJ_U0, half_slot=PJ_B0 / 2, coflow=coflow, depth=depth, nu=PJ_NU,
        pj_inlet(coflow)..., x_end=x_end, kwargs...)
end

# least-squares slope of y against x
pj_slope(x, y) = sum((x .- mean(x)) .* (y .- mean(y))) / sum((x .- mean(x)) .^ 2)

# spreading rate, virtual origin (m) and decay exponent of a jet (x, centreline excess, half-width; m, m/s, m) over [a, b]
function pj_similarity(x, uc, yh, a, b)
    sel = (a .<= x .<= b)
    S = pj_slope(x[sel], yh[sel])
    x0 = mean(x[sel]) - mean(yh[sel]) / S
    return S, x0, pj_slope(log.(x[sel] .- x0), log.(uc[sel]))
end

# value of the series y (along x) at the station xs
pj_at(x, y, xs) = y[argmin(abs.(x .- xs))]

"""
    pj_profile(u, z) -> (centreline excess, half-width, excess momentum flux, free stream, edge cell)

Jet measures of a velocity profile u (m/s) at the cell centres z (m, uniform spacing) from the bed: the free stream Ue
(m/s) is u at the edge z = PJ_EDGE_FACTOR y1/2 of the jet (two iterations from the top value), the centreline excess
u(0) - Ue, the half-width y1/2 (m) and the excess momentum flux J = 2 int_0^edge u (u - Ue) dz (m^3/s^2) of the full jet.
WHY the edge, not the top: the non-hydrostatic outer flow is not uniform (the entrainment sink at the bed curves it; at
30 b0, 0.069 m/s at 0.3 m against 0.056 m/s at the surface), the thin-shear-layer free stream is the velocity at the
edge of the shear layer.
"""
function pj_profile(u, z)
    dz = z[2] - z[1]
    ue, half, edge = u[end], 0.0, length(u)
    for _ in 1:PJ_EDGE_ITERATIONS
        excess = u .- ue
        j = findfirst(<(excess[1] / 2), excess)
        half = z[j - 1] + (excess[j - 1] - excess[1] / 2) / (excess[j - 1] - excess[j]) * dz
        edge = min(length(u), round(Int, PJ_EDGE_FACTOR * half / dz + 0.5))
        ue = u[edge]
    end
    excess = u .- ue
    return excess[1], half, 2 * sum(u[1:edge] .* excess[1:edge]) * dz, ue, edge
end

"""
    pj_model(cells_per_half_slot; hydrostatic) -> NamedTuple

The 2DV half-jet (wetting-drying step for the per-layer inlet k and eps), mean u over the last PJ_T_AVERAGE s of
PJ_T_END, and at every face the jet measures of `pj_profile` (centreline excess, half-width, excess momentum flux, free
stream) and the largest backflow below the jet edge (relative to U0).
"""
function pj_model(cells_per_half_slot; hydrostatic)
    dz = PJ_B0 / 2 / cells_per_half_slot
    nz, nx = round(Int, PJ_DEPTH / dz), round(Int, PJ_LENGTH / (2dz))
    dx = PJ_LENGTH / nx
    dt = PJ_COURANT * dx / PJ_U0
    cfg = ModelConfig(; length=PJ_LENGTH, depth=PJ_DEPTH, nx=nx, nz=nz, dt=dt, use_hydrostatic=hydrostatic, nu=PJ_NU,
        turbulence_closure="k_epsilon", bed_roughness_length=0.0, rayleigh_damping=0.0, apply_froude_limiter=false,
        momentum_advection_scheme="tvd2", transport_advection_scheme="tvd2", wetting_drying=true,
        implicit_vertical_terms=true, write_plots=false, write_animation=false, write_pressure_buoyancy_diagnostics=false)
    model = Model(cfg)
    initialize_water_level!(model, fill(PJ_DEPTH, nx))
    zc = ((1:nz) .- 0.5) .* dz
    slot = zc .< PJ_B0 / 2
    inlet = pj_inlet(PJ_COFLOW)
    u_in = [s ? PJ_U0 : PJ_COFLOW for s in slot]
    k_in = [s ? inlet.k_slot : inlet.k_ambient for s in slot]
    eps_in = [s ? inlet.eps_slot : inlet.eps_ambient for s in slot]
    ke = model.turbulence
    fill!(model.state.u, PJ_COFLOW)
    fill!(ke.k, inlet.k_ambient)
    fill!(ke.eps, inlet.eps_ambient)
    set_x_inflow_outflow!(model, u_in; left_discharge=sum(u_in) * dz, left_k=k_in, left_eps=eps_in, right_level=0.0,
        right_free=true, right_relaxation_time=PJ_RELAXATION)
    steps = round(Int, PJ_T_END / dt)
    start = steps - round(Int, PJ_T_AVERAGE / dt)
    ubar = zeros(nz, nx + 1)
    for n in 1:steps
        step_forward!(model)
        n > start && (ubar .+= model.state.u)
    end
    ubar ./= steps - start
    x = collect(0:nx) .* dx
    uc, yh, J, ue, backflow = similar(x), similar(x), similar(x), similar(x), Inf
    for i in eachindex(x)
        uc[i], yh[i], J[i], ue[i], edge = pj_profile(ubar[:, i], zc)
        backflow = min(backflow, minimum(ubar[1:edge, i] .- ue[i]) / PJ_U0)
    end
    return (; x, uc, yh, J, ue, backflow, model, ubar, zc)
end

@testset "Plane jet (issue #25)" begin
    @testset "test_verify_plane_jet_thin_shear_layer" begin
        r = pj_reference()
        # momentum integral of the confined jet, dJ/dx = -(dUe/dx) Q_excess, integrated over the self-similar range
        i1, i2 = (argmin(abs.(r.x .- xs * PJ_B0)) for xs in extrema(PJ_STATIONS))
        dUe = diff(r.free_stream[i1:i2])
        Qmid = 0.5 .* (r.excess_flux[i1:(i2 - 1)] .+ r.excess_flux[(i1 + 1):i2])
        @test r.momentum_flux[i2] - r.momentum_flux[i1] ≈ -sum(dUe .* Qmid) rtol = PJ_MOMENTUM_RTOL
        # the pressure gradient keeps the volume flux of the channel
        @test sum(r.u) * (r.z[2] - r.z[1]) ≈ (PJ_U0 * PJ_B0 / 2 + PJ_COFLOW * (PJ_DEPTH - PJ_B0 / 2)) rtol = 1e-12
        # grid convergence: half the cells
        coarse = pj_reference(; cells_per_half_slot=10)
        for xs in PJ_STATIONS
            @test pj_at(coarse.x, coarse.centreline_excess, xs * PJ_B0) ≈ pj_at(r.x, r.centreline_excess, xs * PJ_B0) rtol = PJ_GRID_RTOL
            @test pj_at(coarse.x, coarse.half_width, xs * PJ_B0) ≈ pj_at(r.x, r.half_width, xs * PJ_B0) rtol = PJ_GRID_RTOL
        end
        @test_throws DomainError pj_reference(; coflow=0.002, depth=1.0)   # the co-flow cannot feed the jet
        @test_throws ArgumentError pj_reference(; coflow=1.0)
    end

    # The free-jet limit of the reference: weak co-flow in a deep channel. Without the time-scale bound (standard
    # k-epsilon) S = 0.0973 over 15-40 b0 (0.1009 over 10-30 b0), exponent -0.509; with the bound (the model default)
    # S = 0.0922, exponent -0.513.
    @testset "test_validate_plane_jet_spreading_rate" begin
        for realizability in (false, true)
            r = pj_reference(; coflow=PJ_FREE_COFLOW, depth=PJ_FREE_DEPTH, x_end=PJ_FREE_FIT[2] * PJ_B0,
                realizability=realizability)
            S, _, n = pj_similarity(r.x, r.centreline_excess, r.half_width, (PJ_FREE_FIT .* PJ_B0)...)
            realizability || @test PJ_S_RANGE[1] <= S <= PJ_S_RANGE[2]
            realizability && @test S < PJ_S_RANGE[1]   # the bound lowers nu_t in the jet ((k/eps) |S| = 3.3 > 2.72)
            @test abs(n + 0.5) <= PJ_EXPONENT_TOL
        end
    end

    # The model against the reference (the same jet, co-flow and channel; model default: time-scale bound on) at 10, 20,
    # 30 b0. Hydrostatic with 3 cells across the half slot (dz = b0/6): centreline excess -4.0, -2.5, -1.0 %, half-width
    # +7.5, +5.0, +1.6 %, momentum flux within 0.5 %; S = 0.062 (reference 0.065), exponent -0.514. Non-hydrostatic with
    # 2 cells (dz = b0/4; 3 cells take 4 min): -6.8, -4.8, -3.8 %, +11.1, +7.7, +2.4 %, within 2.0 %; with 3 cells it is
    # the hydrostatic result within 0.5 % (experiments/jet/plane_jet.jl).
    ref = pj_reference()
    runs = Dict((true, 3) => pj_model(3; hydrostatic=true), (true, 2) => pj_model(2; hydrostatic=true),
        (false, 2) => pj_model(2; hydrostatic=false))
    for (hydrostatic, cells, level) in ((true, 3, 1), (false, 2, 2))
        m = runs[(hydrostatic, cells)]
        @testset "test_validate_plane_jet_against_thin_shear_layer_$(hydrostatic ? "hydrostatic" : "nonhydrostatic")" begin
            @test m.backflow > -PJ_BACKFLOW_MAX
            for xs in PJ_STATIONS
                @test pj_at(m.x, m.uc, xs * PJ_B0) ≈ pj_at(ref.x, ref.centreline_excess, xs * PJ_B0) rtol = PJ_UC_RTOL[level]
                @test pj_at(m.x, m.yh, xs * PJ_B0) ≈ pj_at(ref.x, ref.half_width, xs * PJ_B0) rtol = PJ_HALF_WIDTH_RTOL[level]
                @test pj_at(m.x, m.J, xs * PJ_B0) ≈ pj_at(ref.x, ref.momentum_flux, xs * PJ_B0) rtol = PJ_J_RTOL[level]
            end
            _, _, n = pj_similarity(m.x, m.uc, m.yh, (extrema(PJ_STATIONS) .* PJ_B0)...)
            @test abs(n + 0.5) <= (level == 1 ? PJ_EXPONENT_TOL : PJ_EXPONENT_TOL_COARSE)
        end
    end
    @testset "test_validate_plane_jet_hydrostatic_equals_nonhydrostatic" begin
        # a slender jet: the non-hydrostatic pressure is negligible (measured against the jet edge, section 36)
        h, nh = runs[(true, 2)], runs[(false, 2)]
        for xs in PJ_STATIONS
            @test pj_at(nh.x, nh.uc, xs * PJ_B0) ≈ pj_at(h.x, h.uc, xs * PJ_B0) rtol = PJ_MODE_RTOL
            @test pj_at(nh.x, nh.J, xs * PJ_B0) ≈ pj_at(h.x, h.J, xs * PJ_B0) rtol = PJ_MODE_RTOL
        end
    end
end
