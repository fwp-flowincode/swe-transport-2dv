# Linear stability of the theta scheme with explicit momentum advection (issue #34).
#
# Von Neumann analysis of the 1D shallow-water equations linearised about a uniform flow U > 0 on the C-grid
# (docs/plan_issue_34_theta_scheme.md, section 3.3):
#   u^{n+1} = A u^n - g dt/dx D-[theta eta^{n+1} + (1 - theta) eta^n]
#   eta^{n+1} = B eta^n - H dt/dx D+[theta u^{n+1} + (1 - theta) u^n]
# with the explicit advection operator A (first-order upwind or Lax-Wendroff, the smooth limit of the MC-limited tvd2)
# and B = A when the surface is advected as well (variable top layer, wetting and drying) or B = 1 (fixed layer
# thicknesses). Per Fourier mode phi = k dx the eigenvalues lambda solve
#   (alpha - lambda)(beta - lambda) + mu^2 (1 - theta + theta lambda)^2 = 0,   mu = 2 C_g sin(phi/2),
# alpha, beta the advection factors, C = U dt/dx, C_g = sqrt(g H) dt/dx, Fr = C / C_g.
# Long waves with alpha = beta = exp(-i C phi) give |lambda|^2 <= 1 iff 2 (1 - theta) Fr <= 2 theta - 1, i.e.
#   Fr <= theta_froude_max(theta) = (2 theta - 1) / (2 (1 - theta)).
# The script prints the numerically computed minimal theta over (C, Fr) and checks that the closed form is sufficient
# (spectral radius <= 1) and sharp (> 1 just above it).
#
# Run: julia --project experiments/theta_scheme/stability_analysis.jl

using Printf
using SweTransport2DV: theta_froude_max

const N_PHI = 2001               # -, wavenumbers sampled in (0, pi]
const STABLE_TOL = 1e-12         # -, spectral radius above 1 + STABLE_TOL counts as unstable (round-off of the roots)
const BISECTION_STEPS = 40       # -, bisection steps for the minimal theta (interval 1/2 / 2^40)
const SHARPNESS_FACTOR = 1.05    # -, Fr = SHARPNESS_FACTOR * Fr_max must be unstable (Lax-Wendroff, surface advected)

# advection factor of one explicit step for the Fourier mode phi (Courant number C)
advection_factor(C, phi, scheme) =
    scheme == :upwind ? 1 - C * (1 - cis(-phi)) : 1 - im * C * sin(phi) - C^2 * (1 - cos(phi))

"""Spectral radius over all sampled wavenumbers of one step (dimensionless)."""
function spectral_radius(C, Cg, theta; scheme=:lw, eta_adv=true, nphi=N_PHI)
    r = 0.0
    a, b = 1 - theta, theta
    for phi in range(pi / nphi, pi; length=nphi)
        alpha = advection_factor(C, phi, scheme)
        beta = eta_adv ? alpha : one(alpha)
        mu2 = (2 * Cg * sin(phi / 2))^2
        # quadratic A2 lambda^2 + A1 lambda + A0 = 0
        A2 = 1 + mu2 * b^2
        A1 = -(alpha + beta) + 2 * mu2 * a * b
        A0 = alpha * beta + mu2 * a^2
        d = sqrt(A1^2 - 4 * A2 * A0 + 0im)
        r = max(r, abs((-A1 + d) / (2 * A2)), abs((-A1 - d) / (2 * A2)))
    end
    return r
end

"""Smallest stable theta in [1/2, 1] (NaN if theta = 1 is unstable)."""
function theta_min(C, Cg; kw...)
    stable(theta) = spectral_radius(C, Cg, theta; kw...) <= 1 + STABLE_TOL
    stable(1.0) || return NaN
    stable(0.5) && return 0.5
    lo, hi = 0.5, 1.0
    for _ in 1:BISECTION_STEPS
        mid = (lo + hi) / 2
        stable(mid) ? (hi = mid) : (lo = mid)
    end
    return hi
end

const COURANTS = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0)
const FROUDES = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0)

for scheme in (:lw, :upwind), eta_adv in (true, false)
    println("\nminimal theta, advection $(scheme == :lw ? "Lax-Wendroff (tvd2, smooth)" : "upwind1"), ",
        eta_adv ? "u and eta advected" : "u advected only (fixed layer thicknesses)")
    @printf("%8s", "Fr \\ C"); foreach(C -> @printf("%7.2f", C), COURANTS); @printf("%10s\n", "closed")
    for Fr in FROUDES
        @printf("%8.2f", Fr)
        for C in COURANTS
            @printf("%7.3f", theta_min(C, C / Fr; scheme=scheme, eta_adv=eta_adv))
        end
        @printf("%10.3f\n", (1 + 2Fr) / (2 * (1 + Fr)))
    end
end

println("\nclosed form Fr_max(theta) = (2 theta - 1) / (2 (1 - theta)):")
for theta in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0)
    @printf("  theta = %.2f   Fr_max = %.3f\n", theta, theta_froude_max(theta))
end

# sufficiency: radius <= 1 for Fr <= Fr_max, all C in (0, 1], both schemes, both continuity forms
worst_inside = 0.0
best_outside = Inf
for theta in (0.52, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0)
    fr_top = min(theta_froude_max(theta), 5.0)
    for scheme in (:lw, :upwind), eta_adv in (true, false)
        for C in range(0.02, 1.0; length=25), Fr in range(fr_top / 24, fr_top; length=24)
            global worst_inside = max(worst_inside, spectral_radius(C, C / Fr, theta; scheme, eta_adv, nphi=721))
        end
    end
    theta < 1 || continue
    for C in (0.1, 0.5, 0.9)
        global best_outside = min(best_outside,
            spectral_radius(C, C / (SHARPNESS_FACTOR * fr_top), theta; scheme=:lw, eta_adv=true))
    end
end
@printf("\nlargest spectral radius for Fr <= Fr_max(theta): 1 + %.1e (sufficient)\n", worst_inside - 1)
@printf("smallest spectral radius for Fr = %.2f Fr_max (Lax-Wendroff, eta advected): 1 + %.1e (sharp)\n",
    SHARPNESS_FACTOR, best_outside - 1)
