# Analytical reference solutions: shallow-water equations with wetting and drying (issue #31), uniform channel flow
# with Chezy friction (issue #14).

const CG_NEWTON_TOL = 1e-13          # -, Newton tolerance on the dimensionless hodograph equations
const CG_NEWTON_MAXITER = 60         # -, Newton iterations before giving up
const CG_SMALL_SIGMA = 1e-4          # -, below this sigma the Bessel quotients use their Taylor series
const BESSEL_QUADRATURE_POINTS = 64  # -, trapezoid points of the Bessel integral (exponential convergence, |x| <~ 20)

"""
    ritter_dam_break(x, t; h0, x0=0.0, g=9.81) -> (h, u)

Ritter (1892) dam break on a dry, flat, frictionless bed: depth `h` (m) and velocity `u` (m/s) at position `x` (m) and
time `t` (s) > 0 after the removal of a dam at `x0` (m) with still water of depth `h0` (m) for x < x0. With
c0 = sqrt(g h0) and xi = (x - x0)/t:

    h = h0,                     u = 0                 for xi <= -c0,
    h = (2 c0 - xi)^2 / (9 g),  u = 2/3 (xi + c0)     for -c0 < xi < 2 c0,
    h = 0,                      u = 0                 for xi >= 2 c0.

Source: Ritter (1892), Z. Ver. Deutsch. Ing. 36, 947-954; see also Delestre et al. (2013), SWASHES, Sec. 4.1.2.
"""
function ritter_dam_break(x::Real, t::Real; h0::Real, x0::Real=0.0, g::Real=9.81)
    t > 0 || throw(ArgumentError("t must be > 0"))
    c0 = sqrt(g * h0)
    xi = (x - x0) / t
    xi <= -c0 && return (float(h0), 0.0)
    xi >= 2c0 && return (0.0, 0.0)
    return ((2c0 - xi)^2 / (9g), 2 * (xi + c0) / 3)
end

"""
    CarrierGreenspanWave(; amplitude, slope, length_scale, g=9.81)

Periodic standing wave on a planar beach (Carrier & Greenspan 1958). `amplitude` A (-) is the dimensionless amplitude of
the potential (non-breaking for A < 1), `slope` (-) the beach slope and `length_scale` L (m) the horizontal scale. The
scales are: length L, elevation slope*L, velocity sqrt(g slope L), time T = sqrt(L / (g slope)); the period is pi T.
"""
Base.@kwdef struct CarrierGreenspanWave
    amplitude::Float64
    slope::Float64
    length_scale::Float64
    g::Float64 = 9.81
end

"Time scale T = sqrt(L / (g slope)) (s) of a Carrier-Greenspan wave; the period is pi T."
carrier_greenspan_time_scale(cg::CarrierGreenspanWave) = sqrt(cg.length_scale / (cg.g * cg.slope))

"Time (s) of maximum runup, lambda = pi/2, where u = 0 everywhere (a state of rest to start a model from)."
carrier_greenspan_rest_time(cg::CarrierGreenspanWave) = pi / 4 * carrier_greenspan_time_scale(cg)

# Bessel function of the first kind J_n(x) from its integral (1/2pi) int_0^2pi cos(n tau - x sin tau) d tau; the
# trapezoid rule on the periodic integrand converges exponentially (Trefethen & Weideman 2014, SIAM Rev. 56, 385-458).
function _bessel_j(n::Int, x::Real)
    s = 0.0
    for j in 0:(BESSEL_QUADRATURE_POINTS - 1)
        tau = 2pi * j / BESSEL_QUADRATURE_POINTS
        s += cos(n * tau - x * sin(tau))
    end
    return s / BESSEL_QUADRATURE_POINTS
end

# J1(s)/s and its derivative (J0(s) - 2 J1(s)/s)/s, with Taylor series near s = 0 (J1(s)/s = 1/2 - s^2/16 + ...).
function _bessel_j1_over_s(s::Real)
    if s < CG_SMALL_SIGMA
        return 0.5 - s^2 / 16, -s / 8
    end
    j0, j1 = _bessel_j(0, s), _bessel_j(1, s)
    return j1 / s, (j0 - 2j1 / s) / s
end

# Dimensionless hodograph solution for phi = A J0(sigma) cos(lambda) (x offshore, bed -x, u offshore):
#   u = phi_sigma / sigma,  eta = phi_lambda / 4 - u^2 / 2,  x = sigma^2/16 - eta,  t = lambda/2 + u,
# with (sigma phi_sigma)_sigma = sigma phi_lambda_lambda (derived from the Riemann invariants u +- 2c - t; Carrier &
# Greenspan 1958, equation numbers not checked; written in this sign convention). Returns the fields and their partial derivatives.
function _cg_fields(A::Real, s::Real, lam::Real)
    q, dq = _bessel_j1_over_s(s)
    j0 = _bessel_j(0, s)
    j1 = s < CG_SMALL_SIGMA ? s / 2 : _bessel_j(1, s)
    u = -A * q * cos(lam)
    u_s = -A * dq * cos(lam)
    u_l = A * q * sin(lam)
    eta = A / 4 * j0 * sin(lam) - u^2 / 2
    eta_s = -A / 4 * j1 * sin(lam) - u * u_s
    eta_l = A / 4 * j0 * cos(lam) - u * u_l
    return u, u_s, u_l, eta, eta_s, eta_l
end

"""
    carrier_greenspan_shoreline(cg, t) -> (x_shore, eta_shore, u_shore)

Shoreline of the Carrier-Greenspan wave at time `t` (s): offshore distance from the still shoreline (m, negative =
runup onto the beach), surface elevation (m) above the still water level and velocity (m/s, positive offshore).
"""
function carrier_greenspan_shoreline(cg::CarrierGreenspanWave, t::Real)
    A = cg.amplitude
    ts = t / carrier_greenspan_time_scale(cg)
    # at sigma = 0: t = lambda/2 - A/2 cos(lambda), monotone in lambda for A < 1
    lam = 2ts
    for _ in 1:CG_NEWTON_MAXITER
        f = lam / 2 - A / 2 * cos(lam) - ts
        lam -= f / (0.5 + A / 2 * sin(lam))
        abs(f) < CG_NEWTON_TOL && break
    end
    u, _, _, eta, _, _ = _cg_fields(A, 0.0, lam)
    Z = cg.slope * cg.length_scale
    return (-eta * cg.length_scale, eta * Z, u * sqrt(cg.g * Z))
end

"""
    carrier_greenspan_solution(cg, x, t) -> (eta, u, wet)

Carrier-Greenspan standing wave at offshore distance `x` (m, from the still shoreline; the still depth is slope*x) and
time `t` (s): surface elevation `eta` (m) above the still water level, velocity `u` (m/s, positive offshore) and whether
the point is wet. Dry points return the bed elevation -slope*x and u = 0. The hodograph variables are found by Newton's
method from x = sigma^2/16 - eta and t = lambda/2 + u.
Source: Carrier & Greenspan (1958), J. Fluid Mech. 4, 97-109.
"""
function carrier_greenspan_solution(cg::CarrierGreenspanWave, x::Real, t::Real)
    A = cg.amplitude
    T = carrier_greenspan_time_scale(cg)
    Z = cg.slope * cg.length_scale
    xs, ts = x / cg.length_scale, t / T
    x_shore = carrier_greenspan_shoreline(cg, t)[1] / cg.length_scale
    if xs <= x_shore
        return (-cg.slope * x, 0.0, false)
    end
    s = 4 * sqrt(xs - x_shore)
    lam = 2ts
    converged = false
    for _ in 1:CG_NEWTON_MAXITER
        u, u_s, u_l, eta, eta_s, eta_l = _cg_fields(A, s, lam)
        f1 = s^2 / 16 - eta - xs
        f2 = lam / 2 + u - ts
        if abs(f1) < CG_NEWTON_TOL && abs(f2) < CG_NEWTON_TOL
            converged = true
            break
        end
        a11, a12 = s / 8 - eta_s, -eta_l
        a21, a22 = u_s, 0.5 + u_l
        det = a11 * a22 - a12 * a21
        ds = (f1 * a22 - a12 * f2) / det
        dl = (a11 * f2 - a21 * f1) / det
        s = abs(s - ds)   # WHY abs: sigma = 4c >= 0; the solution is even in sigma, so a reflected step stays valid
        lam -= dl
    end
    converged || error("carrier_greenspan_solution: Newton did not converge at x = $x m, t = $t s")
    u, _, _, eta, _, _ = _cg_fields(A, s, lam)
    return (eta * Z, u * sqrt(cg.g * Z), true)
end

# Uniform open-channel flow with Chezy friction (issue #14). The channel slope i_b (-, > 0: the bed falls in +x) is the
# body force g i_b of ModelConfig(body_force_x) or a real bed slope; c_f = g / C^2 (-). Depth-integrated momentum
# balance of a uniform flow U (m/s, depth-averaged) in water of depth H (m):
#   dU/dt = g i_b - c_f |U| U / H.

"""
    chezy_equilibrium_velocity(C, depth, slope) -> Float64

Velocity U = C sqrt(H i_b) (m/s) of uniform flow (normal flow) with the Chezy coefficient `C` (m^(1/2)/s), water depth
`depth` H (m) and channel slope `slope` i_b (-, >= 0); the discharge per unit width is q = U H (m^2/s).
Example (issue #14): C = 50, H = 4 m, i_b = 1e-4 gives U = 1 m/s, q = 4 m^2/s. Source: Chow (1959), Open-Channel
Hydraulics, ch. 5 (Chezy formula).
"""
function chezy_equilibrium_velocity(C::Real, depth::Real, slope::Real)
    (C > 0 && depth > 0 && slope >= 0) || throw(ArgumentError("C, depth must be > 0 and slope >= 0"))
    return C * sqrt(depth * slope)
end

"""
    chezy_spinup_velocity(t, C, depth, slope; g=9.81) -> Float64

Depth-averaged velocity U(t) (m/s) of a uniform channel flow that starts from rest at t = 0 (s): the solution of
dU/dt = g i_b - c_f U^2 / H, U(0) = 0, with c_f = g / C^2,

    U(t) = U_eq tanh(t / T),   U_eq = C sqrt(H i_b),   T = H / (c_f U_eq) = C sqrt(H / i_b) / g.

Arguments as in `chezy_equilibrium_velocity`; T = 1019 s for C = 50, H = 4 m, i_b = 1e-4.
"""
function chezy_spinup_velocity(t::Real, C::Real, depth::Real, slope::Real; g::Real=9.81)
    t >= 0 || throw(ArgumentError("t must be >= 0"))
    U_eq = chezy_equilibrium_velocity(C, depth, slope)
    U_eq > 0 || return 0.0
    T = depth * C^2 / (g * U_eq)
    return U_eq * tanh(t / T)
end

"""
    chezy_viscous_profile(z, C, depth, slope, nu; g=9.81) -> Float64

Velocity u(z) (m/s) at the height `z` (m, 0 <= z <= H above the bed) of uniform channel flow with the constant
vertical viscosity `nu` (m^2/s), the Chezy stress c_f U^2 at the bed and no stress at the surface: the shear stress
nu du/dz = g i_b (H - z) gives the parabola

    u(z) = U + (g i_b / nu) (H z - z^2 / 2 - H^2 / 3),   U = C sqrt(H i_b) its depth average.

The slip velocity at the bed is u(0) = U - g i_b H^2 / (3 nu); it is negative (the lowest layer reverses) for
nu < g i_b H^2 / (3 U), i.e. the depth-averaged Chezy law needs a turbulent viscosity (ModelConfig(chezy_coefficient)).
On the layer grid the discrete steady state equals this parabola at the layer centres up to a constant (the central
differences of a quadratic are exact), which `test_verify_chezy_channel_profile_*` uses.
"""
function chezy_viscous_profile(z::Real, C::Real, depth::Real, slope::Real, nu::Real; g::Real=9.81)
    nu > 0 || throw(ArgumentError("nu must be > 0"))
    U = chezy_equilibrium_velocity(C, depth, slope)
    return U + g * slope / nu * (depth * z - z^2 / 2 - depth^2 / 3)
end

const GVF_RK4_STEPS_PER_METRE = 4.0   # 1/m, RK4 steps of chezy_backwater_depth per metre of integration distance

"""
    chezy_backwater_depth(x, q, C, slope, x_end, depth_end; g=9.81) -> Float64

Water depth h(x) (m) of steady gradually varied flow with the discharge `q` (m^2/s), Chezy coefficient `C`
(m^(1/2)/s) and channel slope `slope` i_b (-), for the depth `depth_end` (m) prescribed at the downstream end `x_end`
(m, x <= x_end): the solution of the gradually-varied-flow equation (depth-uniform velocity, Boussinesq coefficient 1)

    dh/dx = (i_b - i_f) / (1 - Fr^2),   i_f = q^2 / (C^2 h^3),   Fr^2 = q^2 / (g h^3),

integrated upstream from x_end with the classical RK4 method (GVF_RK4_STEPS_PER_METRE steps per metre, fourth order;
the profile is smooth for subcritical flow, Fr < 1). depth_end above the normal depth gives the M1 (backwater) curve,
below it the M2 (drawdown) curve. Source: Chow (1959), Open-Channel Hydraulics, ch. 9-10.
"""
function chezy_backwater_depth(x::Real, q::Real, C::Real, slope::Real, x_end::Real, depth_end::Real; g::Real=9.81)
    x <= x_end || throw(ArgumentError("x must be <= x_end (integration upstream)"))
    f(h) = (slope - q^2 / (C^2 * h^3)) / (1 - q^2 / (g * h^3))
    q^2 / (g * depth_end^3) < 1 || throw(ArgumentError("the flow at x_end must be subcritical"))
    n = max(1, ceil(Int, GVF_RK4_STEPS_PER_METRE * (x_end - x)))
    step = (x - x_end) / n   # m, negative
    h = float(depth_end)
    for _ in 1:n
        k1 = f(h)
        k2 = f(h + step / 2 * k1)
        k3 = f(h + step / 2 * k2)
        k4 = f(h + step * k3)
        h += step / 6 * (k1 + 2k2 + 2k3 + k4)
    end
    return h
end

# Steady frictionless flow over a bed obstacle (issue #47): Bernoulli along the channel, q^2 / (2 g h^2) + h + z_b = E,
# with the specific discharge q (m^2/s), depth h (m), bed z_b (m) and energy head E (m). Chow (1959), Open-Channel
# Hydraulics, ch. 3 (specific energy) and 15 (hydraulic jump); test cases of SWASHES (Delestre et al. 2013), see
# swashes_bump_bed.

const BERNOULLI_BISECTION_STEPS = 200   # -, bisection steps of bernoulli_depth (interval halves to machine precision)
const JUMP_SEARCH_SAMPLES = 100_000     # -, sample points of hydraulic_jump_position between the crest and x_max

"""
    critical_depth(q; g=9.81) -> Float64

Critical depth h_c = (q^2/g)^(1/3) (m) of the specific discharge `q` (m^2/s): Fr = 1, the minimum 3/2 h_c of the specific
energy q^2/(2 g h^2) + h. Source: Chow (1959), Open-Channel Hydraulics, ch. 3.
"""
critical_depth(q::Real; g::Real=9.81) = cbrt(q^2 / g)

"""
    bernoulli_depth(head, q; supercritical=false, g=9.81) -> Float64

Depth h (m) with the specific energy q^2/(2 g h^2) + h = `head` (m, the energy head above the bed, E - z_b) for the
specific discharge `q` (m^2/s, > 0): the subcritical root (h >= h_c) or, with `supercritical=true`, the supercritical
root (h <= h_c). Requires head >= 3/2 h_c (otherwise the flow cannot pass with this energy). Bisection on the monotone
branch, BERNOULLI_BISECTION_STEPS steps.
"""
function bernoulli_depth(head::Real, q::Real; supercritical::Bool=false, g::Real=9.81)
    q > 0 || throw(ArgumentError("q must be > 0"))
    hc = critical_depth(q; g=g)
    specific_energy(h) = q^2 / (2g * h^2) + h
    head >= specific_energy(hc) * (1 - eps()) || throw(DomainError(head, "energy head below the critical value 3/2 h_c = $(1.5hc) m"))
    lo, hi = supercritical ? (hc * eps(), hc) : (hc, max(head, hc))
    for _ in 1:BERNOULLI_BISECTION_STEPS
        mid = 0.5 * (lo + hi)
        # subcritical branch: E increases with h; supercritical branch: E decreases with h
        too_high = specific_energy(mid) > head
        if supercritical == too_high
            lo = mid
        else
            hi = mid
        end
    end
    return 0.5 * (lo + hi)
end

"""
    conjugate_depth(h1, q; g=9.81) -> Float64

Depth h2 (m) downstream of a hydraulic jump with the upstream depth `h1` (m) and the specific discharge `q` (m^2/s)
(momentum conservation, Belanger): h2 / h1 = (sqrt(1 + 8 Fr1^2) - 1) / 2, Fr1^2 = q^2 / (g h1^3). Symmetric: the
conjugate of a subcritical depth is the supercritical one. Source: Chow (1959), Open-Channel Hydraulics, ch. 15.
"""
function conjugate_depth(h1::Real, q::Real; g::Real=9.81)
    Fr2 = q^2 / (g * h1^3)
    return 0.5 * h1 * (sqrt(1 + 8 * Fr2) - 1)
end

"""
    swashes_bump_bed(x) -> Float64

Bed level z_b (m) of the SWASHES bump at the position `x` (m) of a 25 m channel: 0.2 - 0.05 (x - 10)^2 for 8 < x < 12,
else 0 (crest 0.2 m at x = 10 m). Source: Delestre et al. (2013), SWASHES, Int. J. Numer. Methods Fluids 72, 269-300,
section 3.1.1 (cases: subcritical q = 4.42 m^2/s with the outlet depth 2 m, transcritical q = 1.53 m^2/s, with shock
q = 0.18 m^2/s and the outlet depth 0.33 m).
"""
swashes_bump_bed(x::Real) = 8 < x < 12 ? 0.2 - 0.05 * (x - 10)^2 : 0.0

"""
    bump_flow_depth(x, q; bed, x_crest, regime, outlet_depth=nothing, g=9.81) -> Float64

Depth h (m) at `x` (m) of steady frictionless flow with the discharge `q` (m^2/s) over the bed `bed(x)` (m) whose crest
is at `x_crest` (m), outlet bed 0. `regime`:
- `:subcritical`: E = q^2/(2 g h_out^2) + h_out from `outlet_depth` h_out (m), subcritical root everywhere;
- `:transcritical`: critical flow at the crest, E = 3/2 h_c + bed(x_crest); subcritical upstream, supercritical
  downstream of the crest;
- `:shock`: as `:transcritical` up to the hydraulic jump at `hydraulic_jump_position`, subcritical with the outlet energy
  of `outlet_depth` downstream of it.
"""
function bump_flow_depth(x::Real, q::Real; bed::Function, x_crest::Real, regime::Symbol, outlet_depth=nothing,
    g::Real=9.81)
    if regime == :subcritical
        E = q^2 / (2g * outlet_depth^2) + outlet_depth
        return bernoulli_depth(E - bed(x), q; g=g)
    end
    E1 = 1.5 * critical_depth(q; g=g) + bed(x_crest)
    regime in (:transcritical, :shock) || throw(ArgumentError("regime must be :subcritical, :transcritical or :shock"))
    x <= x_crest && return bernoulli_depth(E1 - bed(x), q; g=g)
    if regime == :shock
        xs = hydraulic_jump_position(q; bed=bed, x_crest=x_crest, outlet_depth=outlet_depth, g=g)
        if x >= xs
            E2 = q^2 / (2g * outlet_depth^2) + outlet_depth
            return bernoulli_depth(E2 - bed(x), q; g=g)
        end
    end
    return bernoulli_depth(E1 - bed(x), q; supercritical=true, g=g)
end

"""
    hydraulic_jump_position(q; bed, x_crest, outlet_depth, g=9.81, x_max=x_crest + 1e3) -> Float64

Position (m) of the hydraulic jump downstream of the crest of `bed` for transcritical flow with the discharge `q`
(m^2/s) and the subcritical outlet depth `outlet_depth` (m, outlet bed 0): where the conjugate depth of the supercritical
depth (energy 3/2 h_c + bed(x_crest)) equals the subcritical depth of the outlet energy. Bisection between the crest and
the downstream foot of the obstacle (searched up to `x_max`, m), where the subcritical outlet branch exists
(E2 - bed >= 3/2 h_c); the first sign change of the depth mismatch on JUMP_SEARCH_SAMPLES points is refined by bisection.
Throws if the jump is not on the obstacle.
"""
function hydraulic_jump_position(q::Real; bed::Function, x_crest::Real, outlet_depth::Real, g::Real=9.81,
    x_max::Real=x_crest + 1e3)
    hc = critical_depth(q; g=g)
    E1 = 1.5 * hc + bed(x_crest)
    E2 = q^2 / (2g * outlet_depth^2) + outlet_depth
    exists(x) = E2 - bed(x) >= 1.5 * hc   # subcritical downstream depth with the outlet energy
    mismatch(x) = conjugate_depth(bernoulli_depth(E1 - bed(x), q; supercritical=true, g=g), q; g=g) -
                  bernoulli_depth(E2 - bed(x), q; g=g)
    xs = range(x_crest, x_max, length=JUMP_SEARCH_SAMPLES)
    previous = nothing
    for (j, x) in enumerate(xs)
        exists(x) || continue
        m = mismatch(x)
        if previous !== nothing && sign(m) != sign(previous[2])
            lo, hi = previous[1], x
            mlo = previous[2]
            for _ in 1:BERNOULLI_BISECTION_STEPS
                mid = 0.5 * (lo + hi)
                sign(mismatch(mid)) == sign(mlo) ? (lo = mid) : (hi = mid)
            end
            return 0.5 * (lo + hi)
        end
        previous = (x, m)
        bed(x) > 0 || j == 1 || break   # past the obstacle: both depths are constant, no jump position
    end
    throw(DomainError(outlet_depth, "no hydraulic jump on the obstacle for this outlet depth"))
end

# Steady frictionless flow under a sluice gate (issue #48) with the opening a (m) and the contraction coefficient C_c (-):
# the jet contracts to the depth C_c a (vena contracta). Upstream of it the energy head is conserved, downstream of it
# the momentum flux (Henderson 1966, Open Channel Flow, ch. 6; Chow 1959, ch. 17; equation numbers not checked).

"""
    sluice_gate_free_head(q, a; contraction=0.61, g=9.81) -> Float64

Upstream energy head E_1 = C_c a + q^2 / (2 g (C_c a)^2) (m above the bed) of the free flow under a sluice gate with the
opening `a` (m), the specific discharge `q` (m^2/s) and the contraction coefficient `contraction` (-): Bernoulli from
the upstream section to the vena contracta, where the depth is C_c a and the pressure hydrostatic. Equivalent to
q = C_c a sqrt(2 g (E_1 - C_c a)).
"""
function sluice_gate_free_head(q::Real, a::Real; contraction::Real=0.61, g::Real=9.81)
    yc = contraction * a
    return yc + q^2 / (2g * yc^2)
end

"""
    sluice_gate_modular_limit(q, a; contraction=0.61, g=9.81) -> Float64

Largest tailwater depth (m) of free flow under a sluice gate: the conjugate depth of the vena contracta C_c a
(`conjugate_depth`). A deeper tailwater pushes the hydraulic jump against the gate and submerges the jet.
"""
sluice_gate_modular_limit(q::Real, a::Real; contraction::Real=0.61, g::Real=9.81) = conjugate_depth(contraction * a, q; g=g)

"""
    sluice_gate_submerged_head(q, a, tailwater; contraction=0.61, g=9.81) -> Float64

Upstream energy head E_1 (m above the bed) of the flow under a sluice gate with the tailwater depth `tailwater` (m),
opening `a` (m), specific discharge `q` (m^2/s) and contraction coefficient `contraction` (-). Submerged flow
(tailwater above `sluice_gate_modular_limit`): the depth y_s at the vena contracta (jet of depth C_c a below dead water)
follows from the momentum balance between the vena contracta and the tailwater,

    y_s^2 / 2 + q^2 / (g C_c a) = y_t^2 / 2 + q^2 / (g y_t),

and E_1 = y_s + q^2 / (2 g (C_c a)^2) from Bernoulli up to the vena contracta. Below the modular limit the free head
(`sluice_gate_free_head`). Henderson (1966), Open Channel Flow, ch. 6.
"""
function sluice_gate_submerged_head(q::Real, a::Real, tailwater::Real; contraction::Real=0.61, g::Real=9.81)
    yc = contraction * a
    tailwater <= sluice_gate_modular_limit(q, a; contraction=contraction, g=g) &&
        return sluice_gate_free_head(q, a; contraction=contraction, g=g)
    ys = sqrt(tailwater^2 + 2q^2 / g * (1 / tailwater - 1 / yc))
    return ys + q^2 / (2g * yc^2)
end
