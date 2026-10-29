# k-epsilon turbulence closure with buoyancy and vegetation terms, bed wall functions and turbulent density diffusion
# (issue #8). Selected with ModelConfig(turbulence_closure="k_epsilon").
#
# Transport equations (standard high-Reynolds-number k-epsilon; Launder & Spalding 1974, Rodi 1993), per unit mass:
#     dk/dt + div(u k) = d/dz((nu + nu_t/sigma_k) dk/dz) + d/dx((nu + nu_t/sigma_k) dk/dx) + P + G + C_fk P_v - eps,
#     deps/dt + div(u eps) = (same diffusion with sigma_eps) + eps/k (C1 (P + C_fe P_v) + C3 max(G, 0) - C2 eps),
#     nu_t = C_mu k² / eps,
# with the shear production P = nu_t [(du/dz + dw/dx)² + 2 (du/dx)² + 2 (dw/dz)²], the buoyancy term
# G = -(nu_t / sigma_rho) N², N² = -(g/rho0) d(rho)/dz (Rodi 1987; C3 = 0 in stable, 1 in unstable stratification),
# and the vegetation production P_v = work of the drag force per unit mass, -a_veg u (m²/s³), i.e. 1/2 C_D a |u|³ for a
# rigid canopy, with the weights C_fk = 1.0 and C_fe = 1.33 of the k-epsilon vegetation model of Lopez & Garcia (1997,
# Eqs. 26-27 and p. 58; WES Technical Report WRP-CP-10), who tested it on the flume data of Dunn, Lopez & Garcia (1996).
# C1 multiplies the vegetation term in the eps equation; C_fe = (C2/C1) C_fk keeps a homogeneous canopy in balance
# (Lopez & Garcia 1997, p. 24, after Burke 1982): eps = C_fk P_v from both equations.
#
# Numerics per time step (after the flow and density update): explicit first-order upwind advection (conservative,
# closed walls, periodic or zero-gradient inflow), then per column an implicit (backward Euler) vertical diffusion with
# the sink terms linearised implicitly (Patankar): eps/k* k in the k equation, C2 eps*/k* eps in the eps equation, and a
# negative buoyancy term in the k equation; the sources are explicit. This keeps k and eps positive for any dt. Bed: with
# a bed roughness length z0 > 0 the lowest cell takes the log-law wall functions k = u*²/sqrt(C_mu), eps = u*³/(kappa z1)
# (Rodi 1993), u*² = c_b u1², c_b = (kappa / ln(z1/z0))², z1 = dz/2; otherwise zero flux. Surface: zero flux for k;
# for eps the Dirichlet value eps_s = C_mu^(3/4) k_s^(3/2) / (kappa 0.07 H) (Rodi 1993; default, as in Lopez & Garcia
# 1997) or zero flux.
# Density: implicit vertical diffusion with K = nu_t / sigma_rho at the w-faces (conservative, layer thicknesses).

const KE_C_MU = 0.09            # -, Launder & Spalding (1974)
const KE_C1 = 1.44              # -
const KE_C2 = 1.92              # -
const KE_C_MU_VALUE = KE_C_MU     # exported aliases for tests and scripts
const KE_C1_VALUE = KE_C1
const KE_C2_VALUE = KE_C2
const KE_SIGMA_K = 1.0          # -
const KE_SIGMA_EPS = 1.3        # -
const KE_SIGMA_RHO = 0.7        # -, turbulent Schmidt number of density (Rodi 1993)
const KE_C3_STABLE = 0.0        # -, Rodi (1987)
const KE_C3_UNSTABLE = 1.0      # -
const KE_VEG_CFK = 1.0          # -, Lopez & Garcia (1997), p. 58
const KE_VEG_CFE = 1.33         # -, Lopez & Garcia (1997), p. 58; = (C2/C1) C_fk
const KE_KARMAN = 0.41          # -, von Karman constant
const KE_K_MIN = 1e-10          # m^2/s^2, floor of k
const KE_EPS_MIN = 1e-14        # m^2/s^3, floor of eps
const KE_K_INIT = 1e-8          # m^2/s^2, initial k (quiescent)
const KE_NU_T_INIT = 1e-6       # m^2/s, initial nu_t (sets the initial eps)
const KE_SURFACE_LENGTH = 0.07   # -, free-surface dissipation length 0.07 H of Rodi (1993) (surface_condition = :rodi)

"""
    KEpsilonState(nz, nx; T=Float64)

State of the k-epsilon closure on the cell centres (nz x nx): turbulent kinetic energy `k` (m²/s²), dissipation `eps`
(m²/s³), eddy viscosity `nu_t` (m²/s), and the vegetation production `veg_work` (m²/s³, set by step_forward!).
"""
mutable struct KEpsilonState{T<:Real}
    k::Matrix{T}
    eps::Matrix{T}
    nu_t::Matrix{T}
    veg_work::Matrix{T}
    production::Matrix{T}
    buoyancy::Matrix{T}
    k_adv::Matrix{T}
    eps_adv::Matrix{T}
    lower::Vector{T}
    diag::Vector{T}
    upper::Vector{T}
    rhs::Vector{T}
    sol::Vector{T}
    cfk::Float64                # -, vegetation weight in the k equation
    cfe::Float64                # -, vegetation weight in the eps equation
    surface_condition::Symbol   # :zero_flux or :rodi (eps_s = C_mu^(3/4) k_s^(3/2) / (kappa 0.07 H))
end

function KEpsilonState(nz::Int, nx::Int; T::Type{<:Real}=Float64, cfk::Real=KE_VEG_CFK, cfe::Real=KE_VEG_CFE, surface_condition::Symbol=:rodi)
    nz >= 2 || throw(ArgumentError("k-epsilon requires nz >= 2"))
    surface_condition in (:zero_flux, :rodi) || throw(ArgumentError("surface_condition must be :zero_flux or :rodi"))
    k = fill(T(KE_K_INIT), nz, nx)
    eps = fill(T(KE_C_MU * KE_K_INIT^2 / KE_NU_T_INIT), nz, nx)
    nu_t = fill(T(KE_NU_T_INIT), nz, nx)
    z() = zeros(T, nz, nx)
    return KEpsilonState{T}(k, eps, nu_t, z(), z(), z(), z(), z(), zeros(T, nz), zeros(T, nz), zeros(T, nz), zeros(T, nz), zeros(T, nz),
        Float64(cfk), Float64(cfe), surface_condition)
end

"""
    bed_friction_coefficient(z0, z1) -> Float64

Quadratic bed-friction coefficient c_b = (kappa / ln(z1/z0))² (-) of the log law for the roughness length `z0` (m) and
the height `z1` (m) of the lowest velocity point; 0 for z0 <= 0 (free-slip bed).
"""
bed_friction_coefficient(z0::Real, z1::Real) = z0 > 0 ? (KE_KARMAN / log(z1 / z0))^2 : 0.0

"""
    chezy_roughness_length(C, depth; g=9.81) -> Float64

Roughness length z0 (m) of the log law that gives the Chezy coefficient `C` (m^(1/2)/s, > 0) in water of depth `depth`
(m): the depth average of u(z) = (u_*/kappa) ln(z/z0) over z0 <= z <= H is U = (u_*/kappa)(ln(H/z0) - 1 + z0/H), and
C = sqrt(g) U / u_* gives, for z0 << H,

    z0 = H exp(-(1 + kappa C / sqrt(g))).

For the log-law friction (`ModelConfig(bed_roughness_length=...)`, e.g. with k-epsilon), which then reproduces the
Chezy equilibrium if the velocity profile is logarithmic. Example: C = 50, H = 4 m gives z0 = 2.1 mm. Same relation as the
2D/3D roughness conversion of Delft3D-FLOW (User Manual, section 9.4; equation numbers not checked).
"""
function chezy_roughness_length(C::Real, depth::Real; g::Real=9.81)
    (C > 0 && depth > 0) || throw(ArgumentError("C and depth must be > 0"))
    return depth * exp(-(1 + KE_KARMAN * C / sqrt(g)))
end

# Solves the tridiagonal system (lower, diag, upper) x = rhs in place into `sol` (Thomas algorithm; diagonally dominant).
function _solve_tridiagonal_column!(sol, lower, diag, upper, rhs, n)
    @inbounds begin
        c_prev = upper[1] / diag[1]
        sol[1] = rhs[1] / diag[1]
        upper_mod = upper        # reuse: upper[j] becomes the modified super-diagonal
        upper_mod[1] = c_prev
        for j in 2:n
            m = diag[j] - lower[j] * upper_mod[j - 1]
            upper_mod[j] = j < n ? upper[j] / m : 0.0
            sol[j] = (rhs[j] - lower[j] * sol[j - 1]) / m
        end
        for j in (n - 1):-1:1
            sol[j] -= upper_mod[j] * sol[j + 1]
        end
    end
    return sol
end

"""
    implicit_vertical_diffusion!(phi, diffusivity_w, thickness, dt, dz; first=1, wall_value=nothing, ws)

One backward-Euler step of d(h phi)/dt = d/dz(K d phi/dz) per column for the cell-centred field `phi` (nz x nx), with
the diffusivity `diffusivity_w(k, i)` (m²/s) at the face between rows k and k+1 (k = 1 .. nz-1), the layer thicknesses
`thickness(k, i)` (m) and zero flux through the bed and the surface. Conservative: sum_k h_k phi_k is unchanged.
"""
function implicit_vertical_diffusion!(phi::AbstractMatrix{<:Real}, diffusivity_w, thickness, dt::Real, dz::Real, ws::KEpsilonState)
    nz, nx = size(phi)
    nz >= 2 || return phi
    lower, diag, upper, rhs, sol = ws.lower, ws.diag, ws.upper, ws.rhs, ws.sol
    @inbounds for i in 1:nx
        for k in 1:nz
            h = thickness(k, i)
            kd = k > 1 ? dt * diffusivity_w(k - 1, i) / dz : 0.0      # face below (distance between centres: dz)
            ku = k < nz ? dt * diffusivity_w(k, i) / dz : 0.0         # face above
            lower[k] = -kd
            upper[k] = -ku
            diag[k] = h + kd + ku
            rhs[k] = h * phi[k, i]
        end
        _solve_tridiagonal_column!(sol, lower, diag, upper, rhs, nz)
        for k in 1:nz
            phi[k, i] = sol[k]
        end
    end
    return phi
end

# Conservative first-order upwind advection of the cell-centred scalar `phi` with the face velocities u (nz x (nx+1))
# and w ((nz+1) x nx); closed outer faces have u = 0 / w = 0; at inflow faces (u > 0 at i = 1, u < 0 at i = nx+1) the
# boundary cell value is used (zero gradient); periodic_x wraps.
function _advect_upwind_scalar!(out, phi, u, w, dt, dx, dz, periodic_x)
    nz, nx = size(phi)
    @inbounds for i in 1:nx, k in 1:nz
        il = periodic_x ? mod1(i - 1, nx) : max(i - 1, 1)
        ir = periodic_x ? mod1(i + 1, nx) : min(i + 1, nx)
        ul, ur = u[k, i], u[k, i + 1]
        flux_l = ul >= 0 ? ul * phi[k, il] : ul * phi[k, i]
        flux_r = ur >= 0 ? ur * phi[k, i] : ur * phi[k, ir]
        wb, wt = w[k, i], w[k + 1, i]
        flux_b = k == 1 ? 0.0 : (wb >= 0 ? wb * phi[k - 1, i] : wb * phi[k, i])
        flux_t = k == nz ? 0.0 : (wt >= 0 ? wt * phi[k, i] : wt * phi[k + 1, i])
        # WHY the divergence correction phi * div(u): keeps a uniform field uniform when the discrete velocity field is
        # not exactly divergence-free in a cell (free surface, inflow), i.e. advective form for the non-solenoidal part.
        div = (ur - ul) / dx + ((k == nz ? 0.0 : wt) - (k == 1 ? 0.0 : wb)) / dz
        out[k, i] = phi[k, i] - dt * ((flux_r - flux_l) / dx + (flux_t - flux_b) / dz - phi[k, i] * div)
    end
    return out
end

"""
    k_epsilon_step!(ke, u, w, rho, dt, dx, dz, nu, g, rho0; periodic_x=false, bed_roughness=0.0)

Advances k and eps by one time step for the velocities `u`, `w` (m/s) and the density `rho` (kg/m³) of the new time
level (see the file header) and updates `ke.nu_t`. `ke.veg_work` (m²/s³) holds the vegetation production of the step.
"""
function k_epsilon_step!(ke::KEpsilonState, u::AbstractMatrix{<:Real}, w::AbstractMatrix{<:Real}, rho::AbstractMatrix{<:Real},
    dt::Real, dx::Real, dz::Real, nu::Real, g::Real, rho0::Real; periodic_x::Bool=false, bed_roughness::Real=0.0)
    nz, nx = size(ke.k)
    k, eps, nu_t = ke.k, ke.eps, ke.nu_t
    # cell-centred horizontal velocity
    uc(kk, i) = 0.5 * (u[kk, i] + u[kk, i + 1])
    @inbounds for i in 1:nx, kk in 1:nz
        k_up, k_dn = min(kk + 1, nz), max(kk - 1, 1)
        dudz = (uc(k_up, i) - uc(k_dn, i)) / ((k_up - k_dn) * dz)
        il = periodic_x ? mod1(i - 1, nx) : max(i - 1, 1)
        ir = periodic_x ? mod1(i + 1, nx) : min(i + 1, nx)
        wc(ii) = 0.5 * (w[kk, ii] + w[kk + 1, ii])
        dwdx = ir == il ? 0.0 : (wc(ir) - wc(il)) / ((ir > il ? ir - il : ir - il + nx) * dx)
        dudx = (u[kk, i + 1] - u[kk, i]) / dx
        dwdz = (w[kk + 1, i] - w[kk, i]) / dz
        ke.production[kk, i] = nu_t[kk, i] * ((dudz + dwdx)^2 + 2 * dudx^2 + 2 * dwdz^2)
        n2 = -(g / rho0) * (rho[k_up, i] - rho[k_dn, i]) / ((k_up - k_dn) * dz)
        ke.buoyancy[kk, i] = -(nu_t[kk, i] / KE_SIGMA_RHO) * n2
    end
    # advection (explicit upwind)
    _advect_upwind_scalar!(ke.k_adv, k, u, w, dt, dx, dz, periodic_x)
    _advect_upwind_scalar!(ke.eps_adv, eps, u, w, dt, dx, dz, periodic_x)
    @. k = max(ke.k_adv, KE_K_MIN)
    @. eps = max(ke.eps_adv, KE_EPS_MIN)
    # implicit vertical diffusion with sources and implicit sinks, column by column
    lower, diag, upper, rhs, sol = ke.lower, ke.diag, ke.upper, ke.rhs, ke.sol
    z1 = 0.5 * dz
    cb = bed_friction_coefficient(bed_roughness, z1)
    wall = cb > 0
    first = wall ? 2 : 1
    @inbounds for i in 1:nx
        for (field, sigma) in ((:k, KE_SIGMA_K), (:eps, KE_SIGMA_EPS))
            phi = field === :k ? k : eps
            n = nz - first + 1
            for j in 1:n
                kk = j + first - 1
                ratio = eps[kk, i] / k[kk, i]                              # 1/s, old (after advection) eps/k
                kd = kk > first ? dt * (nu + 0.5 * (nu_t[kk, i] + nu_t[kk - 1, i]) / sigma) / dz^2 : 0.0
                ku = kk < nz ? dt * (nu + 0.5 * (nu_t[kk, i] + nu_t[kk + 1, i]) / sigma) / dz^2 : 0.0
                p, gb, pv = ke.production[kk, i], ke.buoyancy[kk, i], ke.veg_work[kk, i]
                if field === :k
                    source = p + max(gb, 0.0) + ke.cfk * pv
                    sink = ratio + max(-gb, 0.0) / k[kk, i]               # implicit: eps/k and negative buoyancy
                else
                    c3 = gb > 0 ? KE_C3_UNSTABLE : KE_C3_STABLE
                    source = ratio * (KE_C1 * (p + ke.cfe * pv) + c3 * max(gb, 0.0))
                    sink = KE_C2 * ratio
                end
                lower[j] = -kd
                upper[j] = -ku
                diag[j] = 1 + kd + ku + dt * sink
                rhs[j] = phi[kk, i] + dt * source
            end
            if wall
                # lowest cell: wall function value (Dirichlet), coupled through the face above it
                u1 = abs(uc(1, i))
                ustar2 = cb * u1^2
                bed_value = field === :k ? ustar2 / sqrt(KE_C_MU) : (sqrt(ustar2))^3 / (KE_KARMAN * z1)
                phi[1, i] = max(bed_value, field === :k ? KE_K_MIN : KE_EPS_MIN)
                kd = dt * (nu + 0.5 * (nu_t[2, i] + nu_t[1, i]) / sigma) / dz^2
                diag[1] += kd
                rhs[1] += kd * phi[1, i]
            end
            if field === :eps && ke.surface_condition === :rodi
                # free-surface dissipation (Rodi 1993): Dirichlet value in the top cell from its k
                eps_s = KE_C_MU^0.75 * k[nz, i]^1.5 / (KE_KARMAN * KE_SURFACE_LENGTH * nz * dz)
                lower[n] = 0.0
                diag[n] = 1.0
                rhs[n] = eps_s
            end
            _solve_tridiagonal_column!(sol, lower, diag, upper, rhs, n)
            floor_value = field === :k ? KE_K_MIN : KE_EPS_MIN
            for j in 1:n
                phi[j + first - 1, i] = max(sol[j], floor_value)
            end
        end
    end
    @. nu_t = KE_C_MU * k^2 / eps
    return ke
end

"""
    turbulent_viscosity_u!(nu_u, ke)

Eddy viscosity at the u-faces (m²/s, nz x (nx+1)): mean of the two neighbouring cells (the outer faces take their only
neighbour).
"""
function turbulent_viscosity_u!(nu_u::AbstractMatrix{<:Real}, ke::KEpsilonState; periodic_x::Bool=false)
    nz, nx = size(ke.nu_t)
    @inbounds for i in 1:(nx + 1), kk in 1:nz
        il = i == 1 ? (periodic_x ? nx : 1) : i - 1
        ir = i == nx + 1 ? (periodic_x ? 1 : nx) : i
        nu_u[kk, i] = 0.5 * (ke.nu_t[kk, il] + ke.nu_t[kk, ir])
    end
    return nu_u
end
