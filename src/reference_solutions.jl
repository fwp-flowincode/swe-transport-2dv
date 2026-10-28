# Analytical reference solutions of the shallow-water equations with wetting and drying (issue #31).

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
