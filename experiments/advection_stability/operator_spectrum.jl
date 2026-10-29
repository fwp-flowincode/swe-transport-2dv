# Issue #37: von Neumann (matrix) analysis of the discrete momentum advection around a uniform current.
#
# Periodic channel, fixed grid (variable_top_layer = false), uniform current U > 0, first-order upwind. Around U the
# upwind directions are fixed, so the tendency is a quadratic polynomial of the perturbation u' (fluxes v*phi), and the
# central difference L u' = (A(U + u', w') - A(U - u', -w')) / 2 is its exact linear part, with w'
# from the layer continuity (w'_1 = 0 at the bed, w'_{k+1} = w'_k - dz du'_k/dx; the top value w'_{nz+1} = d(eta)/dt is
# not zero on the fixed grid). The script builds L column by column and reports the forward-Euler amplification
# max |eig(I - dt L)| against the advective Courant number C = U dt / dx and nz, for
#   - "legacy":     w interpolated via the layer centres ((w[k-1] + 2 w[k] + w[k+1]) / 4), no top-row correction
#                   (the scheme before issue #37);
#   - "top only":   legacy face velocities with the top-row advective form;
#   - "consistent": divergence-free face velocities (_w_at_u_interfaces!) with the top-row advective form (current).
# Gravity waves are left out: this isolates the advection operator. Values above 1 + AMPLIFICATION_TOL grow.
#
# Run: julia --project experiments/advection_stability/operator_spectrum.jl   (about a minute)

using SweTransport2DV, LinearAlgebra, Printf
const S = SweTransport2DV

const AMPLIFICATION_TOL = 1e-10   # -, round-off margin of the spectral radius
const PERTURBATION = 1e-3         # m/s, size of the unit perturbations (central differences: exact for the quadratic fluxes)

# Layer continuity on the fixed grid: w (m/s, (nz+1) x nx) of the periodic u (m/s, nz x (nx+1)).
function continuity_w(u, dx, dz)
    nz, nxp1 = size(u)
    nx = nxp1 - 1
    w = zeros(nz + 1, nx)
    for i in 1:nx, k in 1:nz
        w[k + 1, i] = w[k, i] - dz * (u[k, i + 1] - u[k, i]) / dx
    end
    return w
end

# Advective tendency (m/s^2) of u for one of the three variants.
function tendency(u, w, dx, dz, variant)
    nz, nxp1 = size(u)
    nx = nxp1 - 1
    out = zeros(nz, nxp1)
    ws_adv = S.AdvectionWorkspace(nz, nxp1)
    ws_interp = S.InterpolationWorkspace(nz, nx)
    if variant == :consistent
        S.advection_term_u!(out, u, w, dx, dz, "upwind1", ws_adv, ws_interp; top_row_advective_form=true, periodic_x=true)
    else
        S._interpolate_w_to_u!(ws_interp.w_u, w, ws_interp.w_center; periodic_x=true)
        S._advective_flux_divergence!(@view(out[:, 1:nx]), @view(u[:, 1:nx]), @view(u[:, 1:nx]),
            @view(ws_interp.w_u[:, 1:nx]), dx, dz, "upwind1", ws_adv, true)
        variant == :top_only && S._top_row_advective_form!(@view(out[:, 1:nx]), @view(u[:, 1:nx]), dx, dz, ws_adv; periodic_x=true)
    end
    return out[:, 1:nx]
end

# Linear operator L (1/s) on the nz * nx independent velocities (the seam column nx+1 repeats column 1).
function advection_operator(U, nz, nx, dx, dz, variant)
    n = nz * nx
    L = zeros(n, n)
    base = fill(U, nz, nx + 1)
    for j in 1:n
        k, i = (j - 1) % nz + 1, (j - 1) ÷ nz + 1
        f = map((1, -1)) do sign
            u = copy(base)
            u[k, i] += sign * PERTURBATION
            i == 1 && (u[k, nx + 1] += sign * PERTURBATION)
            tendency(u, continuity_w(u, dx, dz), dx, dz, variant)
        end
        L[:, j] = vec(f[1] .- f[2]) ./ (2 * PERTURBATION)
    end
    return L
end

function main()
    U, H, L_x, nx = 0.5, 0.5, 1.0, 16   # m/s, m, m, -
    dx = L_x / nx
    println("max |eig(I - dt L)|, upwind1, periodic fixed grid, U = $U m/s, H = $H m, nx = $nx")
    println(rpad("variant", 12), rpad("nz", 4), join([rpad(@sprintf("C=%.2f", C), 14) for C in (0.1, 0.25, 0.5, 0.75, 1.0)]))
    for variant in (:legacy, :top_only, :consistent), nz in (1, 2, 4, 8, 16)
        dz = H / nz
        Lop = advection_operator(U, nz, nx, dx, dz, variant)
        lambdas = eigvals(Lop)
        row = map((0.1, 0.25, 0.5, 0.75, 1.0)) do C
            dt = C * dx / U
            maximum(abs, 1 .- dt .* lambdas)
        end
        flags = [r > 1 + AMPLIFICATION_TOL ? "*" : " " for r in row]
        println(rpad(string(variant), 12), rpad(string(nz), 4), join([rpad(@sprintf("%.8f%s", r, f), 14) for (r, f) in zip(row, flags)]))
    end
    println("* grows")
end

main()
