# Implicit vertical terms of the momentum equation (issue #34, ModelConfig(implicit_vertical_terms=true)).
#
# Per u-face column the vertical viscosity, the vertical momentum advection (first-order upwind with the interface
# velocities of time n) and the linearised bed friction gamma = c_b |u^n_1| form one tridiagonal matrix A,
#   A u^{n+1} = dZ .* G - theta g dt d(eta^{n+1})/dx dZ,
# and the free surface is coupled through the column response r = A^{-1} dZ (Casulli & Cheng 1992, Int. J. Numer.
# Methods Fluids 15, 629-648; equation numbers not checked):
#   u^{n+1} = A^{-1}(dZ .* G) - theta g dt d(eta^{n+1})/dx r,   effective depth of the face: sum_k dZ_k r_k.
# A is an M-matrix (A^{-1} >= 0, so r >= 0 and the effective depth is >= 0, which keeps the free-surface matrix an
# M-matrix and the positivity proof of the wetting-drying solve valid). Without friction the effective depth equals the
# water depth: in flux form the columns of A sum to dZ (1' A = dZ'), in advective form its rows do (A 1 = dZ), and either
# gives dZ' A^{-1} dZ = sum(dZ). Backward Euler: first order in time, no time-step restriction (robustness, issue #34).

"""
    implicit_vertical_column!(u, r, uhat, dZ, W, nu, gamma, dt, m, cprime, dprime, rprime; advective=false, h_adv=dZ)

One implicit step of the vertical terms of a u-face column with `m` open layers, numbered from the bed (k = 1) to the
top (k = m); entries beyond `m` are not touched:

    dZ_k (u_k - uhat_k) = -dt V_k(u) + dt D_k(u) - dt gamma u_1 delta_{k1}.

- `dZ` (m): layer thicknesses; `uhat` (m/s): explicit predictor; `gamma` (m/s): linearised bed friction on layer 1.
- Advection with the interface velocities `W` (m/s, length >= m+1; W[j] lies below layer j, W[1] and W[m+1] are not
  used: no flux through the bed and the top):
  - flux form (`advective=false`, fixed grid, conserves sum(dZ u)): V_k = F_{k+1} - F_k,
    F_j = max(W_j, 0) u_{j-1} + min(W_j, 0) u_j;
  - advective form (`advective=true`, wetting and drying, keeps a uniform u): V_k = (dZ_k / h_adv_k) [max(W_k, 0)
    (u_k - u_{k-1}) - min(W_{k+1}, 0)(u_k - u_{k+1})], with the advection thickness `h_adv` (m) of the explicit scheme.
- Viscosity `nu` (m^2/s, at the interfaces, length >= m+1, W-like indexing): D_k = nu_{k+1}(u_{k+1} - u_k)/dzf_{k+1} -
  nu_k (u_k - u_{k-1})/dzf_k, dzf_j = (dZ_{j-1} + dZ_j)/2, zero stress at the bed and the top.

Writes `u` (m/s) = A^{-1}(dZ .* uhat) and the column response `r` (-) = A^{-1} dZ. `cprime`, `dprime`, `rprime`: work
vectors of length >= m. Thomas algorithm without pivoting: A is diagonally dominant (by columns in flux form, by rows
in advective form).
"""
function implicit_vertical_column!(u::AbstractVector{<:Real}, r::AbstractVector{<:Real}, uhat::AbstractVector{<:Real},
    dZ::AbstractVector{<:Real}, W::AbstractVector{<:Real}, nu::AbstractVector{<:Real}, gamma::Real, dt::Real, m::Int,
    cprime::AbstractVector{<:Real}, dprime::AbstractVector{<:Real}, rprime::AbstractVector{<:Real};
    advective::Bool=false, h_adv::AbstractVector{<:Real}=dZ)
    m >= 1 || return nothing
    @inbounds for k in 1:m
        lower = 0.0   # coefficient of u_{k-1} in row k
        upper = 0.0   # coefficient of u_{k+1} in row k
        diag = dZ[k]
        if k > 1
            v = dt * nu[k] / (0.5 * (dZ[k - 1] + dZ[k]))
            diag += v
            lower -= v
            if advective
                s = dt * dZ[k] / h_adv[k] * max(W[k], 0.0)
                diag += s
                lower -= s
            else
                diag -= dt * min(W[k], 0.0)   # outflow downwards through interface k
                lower -= dt * max(W[k], 0.0)  # inflow from below
            end
        end
        if k < m
            v = dt * nu[k + 1] / (0.5 * (dZ[k] + dZ[k + 1]))
            diag += v
            upper -= v
            if advective
                s = -dt * dZ[k] / h_adv[k] * min(W[k + 1], 0.0)
                diag += s
                upper -= s
            else
                diag += dt * max(W[k + 1], 0.0)   # outflow upwards through interface k+1
                upper += dt * min(W[k + 1], 0.0)  # inflow from above
            end
        end
        k == 1 && (diag += dt * gamma)
        # forward elimination (Thomas), both right-hand sides
        if k == 1
            denom = diag
            dprime[k] = dZ[k] * uhat[k] / denom
            rprime[k] = dZ[k] / denom
        else
            denom = diag - lower * cprime[k - 1]
            dprime[k] = (dZ[k] * uhat[k] - lower * dprime[k - 1]) / denom
            rprime[k] = (dZ[k] - lower * rprime[k - 1]) / denom
        end
        cprime[k] = upper / denom
    end
    u[m] = dprime[m]
    r[m] = rprime[m]
    @inbounds for k in (m - 1):-1:1
        u[k] = dprime[k] - cprime[k] * u[k + 1]
        r[k] = rprime[k] - cprime[k] * r[k + 1]
    end
    return nothing
end
