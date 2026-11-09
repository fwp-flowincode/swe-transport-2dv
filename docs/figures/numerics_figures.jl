# Figures of docs/numerics_wetting_drying.md and docs/numerics_open_boundaries.md that are not taken from tests or
# experiments: a sketch of the cut-cell geometry and the linear analysis of the outflow boundary.
#
# - wd_cut_cells.png             cells cut by the bed and the surface, face thicknesses from the upwind level
# - ob_reflection_analysis.png   reflection of the free outflow: the discrete radiation condition with the theta scheme
#                                (analysis and simulated check), the rejected lagged variant, and dispersive waves
#                                against the long-wave speed
# - ob_characteristic_analysis.png  reflection of the characteristic open face (section 8): the weight beta, a third
#                                point, the limited beta at small Courant numbers, and a current through the face
#
# Usage: julia --project=<path to the repository> docs/figures/numerics_figures.jl   (about a minute)
#
# WHY: batch run without a display; same GR workstation choice as src/visualization.jl.
ENV["GKSwstype"] = "100"
using LinearAlgebra, Plots, Printf

const FIGURE_DIR = @__DIR__

# ---------------------------------------------------------------------------------------------------------------------
# Cut-cell sketch
# ---------------------------------------------------------------------------------------------------------------------

const SK_DZ = 1.0      # layer thickness (sketch units)
const SK_NZ = 5
const SK_DX = 1.6      # column width (sketch units)
const SK_BED = [0.0, 0.35, 1.3, 2.1, 2.75, 3.6]          # bed level of each column
const SK_LEVEL = [3.55, 3.5, 3.42, 3.3, 3.12, 3.6]       # water level (the last column is dry: level = bed)

function cut_cells_sketch()
    nx = length(SK_BED)
    p = plot(; size=(900, 500), legend=:topleft, framestyle=:box, grid=false, xlims=(-0.6, nx * SK_DX + 0.2),
        ylims=(-0.25, SK_NZ * SK_DZ + 1.4), xlabel="x", ylabel="z", aspect_ratio=:equal, margin=4Plots.mm,
        legendfontsize=8, ticks=false)
    for i in 1:nx
        x0, x1 = (i - 1) * SK_DX, i * SK_DX
        zb, zeta = SK_BED[i], SK_LEVEL[i]
        # wet part of every layer: h_k = max(0, min(z_{k+1}, zeta) - max(z_k, zb))
        for k in 1:SK_NZ
            lo, hi = max((k - 1) * SK_DZ, zb), min(k * SK_DZ, zeta)
            hi > lo || continue
            plot!(p, Shape([x0, x1, x1, x0], [lo, lo, hi, hi]); color=k % 2 == 0 ? :lightskyblue : :lightblue,
                linecolor=:steelblue, linewidth=0.5, label="")
        end
        plot!(p, Shape([x0, x1, x1, x0], [-0.25, -0.25, zb, zb]); color=:burlywood, linecolor=:saddlebrown, label="")
        zeta > zb && plot!(p, [x0, x1], [zeta, zeta]; color=:navy, linewidth=2, label="")
    end
    for k in 0:SK_NZ
        plot!(p, [0, nx * SK_DX], [k * SK_DZ, k * SK_DZ]; color=:gray, linestyle=:dot, linewidth=0.8, label="")
    end
    # faces: bed of the higher neighbour, level of the upwind column (flow to the right here)
    for f in 2:nx
        xf = (f - 1) * SK_DX
        zbf = max(SK_BED[f - 1], SK_BED[f])
        zetaf = SK_LEVEL[f - 1]
        if zetaf - zbf > 1e-6
            plot!(p, [xf, xf], [zbf, zetaf]; color=:crimson, linewidth=3,
                label=(f == 2 ? "open face (flow to the right): bed of the higher neighbour, upwind level" : ""))
        end
    end
    plot!(p, [-0.3, -0.3], [1.0, 2.0]; color=:black, arrow=:both, linewidth=1, label="")
    annotate!(p, [(0.5SK_DX, SK_LEVEL[1] + 0.25, text("ζ₁", 9, :navy)), (2.5SK_DX, 1.75, text("h_{k,i}", 8)),
        (5.5SK_DX, 3.85, text("dry column", 8)), (3.5SK_DX, 2.35, text("cut by the bed", 7)),
        (4.5SK_DX, 3.3, text("cut by the surface", 7)), (-0.45, 1.5, text("Δz", 8, :right)),
        (5.0SK_DX, 4.4, text("closed face: upwind level\nbelow the face bed", 7))])
    plot!(p, [NaN], [NaN]; color=:navy, linewidth=2, label="water level ζ_i of the column")
    plot!(p, [NaN], [NaN]; color=:gray, linestyle=:dot, label="fixed interfaces z_k = (k-1)Δz")
    savefig(p, joinpath(FIGURE_DIR, "wd_cut_cells.png"))
end

# ---------------------------------------------------------------------------------------------------------------------
# Reflection of the free outflow (linear, no mean flow, flat bed)
# ---------------------------------------------------------------------------------------------------------------------
# Units: g = H = dx = 1, so c0 = 1 and dt = Cr (gravity-wave Courant number). Interior of the theta scheme
# (Casulli & Cattani 1994), for a mode with the factor lambda per step and Theta = theta lambda + 1 - theta:
#     (lambda - 1) u_{j+1/2} = -dt Theta (eta_{j+1} - eta_j),   (lambda - 1) eta_j = -dt Theta (u_{j+1/2} - u_{j-1/2}).
# Boundary face at x = L, half a cell from the last centre j = n:
#     (lambda - 1) u_b = -2 dt Theta (eta_b - eta_n)                     its own momentum equation (as in the model)
#     (lambda - 1) eta_b = -2 dt Theta_r (eta_b - eta_n)                 radiation; Theta_r = lambda: backward Euler (model)
#     (lambda - 1) eta_b = -2 dt (lambda eta_b - eta_n)                 radiation with eta_n of time n ("lagged")
# Theta_r = theta lambda + 1 - theta is the radiation condition weighted like the interior (not in the model).
# Plane waves eta_j = exp(i kappa j), u_{j+1/2} = Gamma exp(i kappa (j + 1/2)), Gamma = -dt Theta 2i sin(kappa/2)/(lambda - 1),
# dispersion (lambda - 1)^2 = -4 Cr^2 Theta^2 sin^2(kappa/2). With a = 1 the right-going and b the left-going amplitude
# at x = L, R = b.

const RA_KDX_MAX = 1.6       # -, largest k dx of the plot (8 cells per wavelength at 0.79)
const RA_MONO_CELLS = 60     # cells of the monochromatic check
const RA_MONO_RAMP = 50      # periods of the wave-maker ramp
const RA_MONO_RUN = 400      # periods in all; the last 100 are analysed

function reflection(theta, cr, omega_dt; lagged=false, radiation_theta=1.0)
    lambda = exp(-im * omega_dt)
    Theta = theta * lambda + 1 - theta
    s = sin(omega_dt / 2) * exp(-im * omega_dt / 2) / (cr * Theta)   # sin(kappa/2), right-going root
    kappa = 2asin(s)
    Gamma = -cr * Theta * 2im * sin(kappa / 2) / (lambda - 1)
    # unknowns (b, eta_b); eta_n = a e^{-i kappa/2} + b e^{i kappa/2}, u_b = Gamma (a - b), a = 1
    en_a, en_b = exp(-im * kappa / 2), exp(im * kappa / 2)
    # momentum of the boundary face: (lambda - 1) Gamma (1 - b) + 2 cr Theta (eta_b - eta_n) = 0
    m11 = -(lambda - 1) * Gamma - 2cr * Theta * en_b
    m12 = 2cr * Theta
    r1 = -(lambda - 1) * Gamma + 2cr * Theta * en_a
    # radiation: (lambda - 1) eta_b + 2 cr (Tb eta_b - Tn eta_n) = 0
    Tb = lagged ? lambda : radiation_theta * lambda + 1 - radiation_theta
    Tn = lagged ? one(lambda) : Tb
    m21 = -2cr * Tn * en_b
    m22 = (lambda - 1) + 2cr * Tb
    r2 = 2cr * Tn * en_a
    b = (r1 * m22 - m12 * r2) / (m11 * m22 - m12 * m21)
    return real(kappa), abs(b), kappa
end

function reflection_curve(theta, cr; kwargs...)
    ks, rs = Float64[], Float64[]
    for omega_dt in range(1e-4, 0.999pi; length=4000)
        k, r, _ = reflection(theta, cr, omega_dt; kwargs...)
        # WHY the monotone check: with strong damping (theta = 1, Cr = 1) Re(kappa) has a maximum below k dx = pi
        (isfinite(k) && k <= RA_KDX_MAX && (isempty(ks) || k > ks[end])) || break
        push!(ks, k)
        push!(rs, r)
    end
    return ks, rs
end

# omega dt of the mode with Re(kappa) = kdx (bisection; Re(kappa) increases with omega)
function find_omega(theta, cr, kdx; kwargs...)
    lo, hi = 1e-6, 0.999pi
    for _ in 1:100
        mid = 0.5(lo + hi)
        reflection(theta, cr, mid; kwargs...)[1] < kdx ? (lo = mid) : (hi = mid)
    end
    return 0.5(lo + hi)
end

# Check of the analysis: the scheme above (radiation backward Euler, as in the model) driven by a wave maker at the left
# until periodic; the complex amplitude of every cell from the last 100 periods, fitted by a e^{i kappa j} + b e^{-i kappa j}
# over the right half. Returns |R| = |b/a| at x = L.
function simulated_reflection(theta, cr, kdx)
    nx = RA_MONO_CELLS
    om_dt = find_omega(theta, cr, kdx)
    dt = cr
    om = om_dt / dt
    mu = 2dt
    kappa = reflection(theta, cr, om_dt)[3]
    N = 2nx
    M = zeros(N, N)
    ie(j) = j
    iu(f) = nx + f - 1          # faces 2..nx+1
    for j in 1:nx
        M[ie(j), ie(j)] = 1.0
        M[ie(j), iu(j + 1)] += theta * dt
        j > 1 && (M[ie(j), iu(j)] -= theta * dt)
    end
    for f in 2:nx
        M[iu(f), iu(f)] = 1.0
        M[iu(f), ie(f)] += theta * dt
        M[iu(f), ie(f - 1)] -= theta * dt
    end
    # boundary face: eta_b^{n+1} - eta_n^{n+1} = (eta_b^n - eta_n^{n+1}) / (1 + mu)
    M[iu(nx + 1), iu(nx + 1)] = 1.0
    M[iu(nx + 1), ie(nx)] -= theta * dt / 0.5 / (1 + mu)
    F = lu(M)
    eta, u, etab = zeros(nx), zeros(nx + 1), 0.0
    period = round(Int, 2pi / om_dt)
    nsteps = RA_MONO_RUN * period
    acc = zeros(ComplexF64, nx)
    rhs = zeros(N)
    for n in 1:nsteps
        t = n * dt
        u1 = min(1.0, n / (RA_MONO_RAMP * period)) * cos(om * t)   # wave maker at the left face
        for j in 1:nx
            inflow = j == 1 ? dt * (theta * u1 + (1 - theta) * u[1]) : 0.0
            rhs[ie(j)] = eta[j] - (1 - theta) * dt * (u[j + 1] - (j == 1 ? 0.0 : u[j])) + inflow
        end
        for f in 2:nx
            rhs[iu(f)] = u[f] - (1 - theta) * dt * (eta[f] - eta[f - 1])
        end
        rhs[iu(nx + 1)] = u[nx + 1] - (1 - theta) * dt / 0.5 * (etab - eta[nx]) - theta * dt / 0.5 * etab / (1 + mu)
        sol = F \ rhs
        eta .= view(sol, 1:nx)
        u[2:end] .= view(sol, (nx + 1):N)
        u[1] = u1
        etab = (etab + mu * eta[nx]) / (1 + mu)
        n > nsteps - 100 * period && (acc .+= eta .* exp(im * om * t))
    end
    js = (nx ÷ 2):nx
    ab = hcat(exp.(im * kappa .* js), exp.(-im * kappa .* js)) \ acc[js]
    return abs(ab[2] * exp(-im * kappa * (nx + 0.5)) / (ab[1] * exp(im * kappa * (nx + 0.5))))
end

function reflection_figure()
    p1 = plot(; xlabel="k Δx", ylabel="|R|", yscale=:log10, ylims=(1e-5, 1.2), xlims=(0, RA_KDX_MAX),
        title="free outflow, long waves (no mean flow)", titlefontsize=10, legend=:bottomright, legendfontsize=7)
    k = range(1e-3, RA_KDX_MAX; length=400)
    plot!(p1, k, tan.(k ./ 4) .^ 2; color=:black, linewidth=2.5,
        label="semi-discrete tan²(kΔx/4) (= radiation weighted with θ)")
    for (theta, cr, style, color) in ((0.7, 1.0, :solid, 1), (0.7, 0.5, :dash, 1), (1.0, 1.0, :solid, 2), (0.5, 1.0, :solid, 3))
        ks, rs = reflection_curve(theta, cr)
        plot!(p1, ks, rs; color=color, linestyle=style, linewidth=1.5, label=@sprintf("model: θ = %.1f, Cr = %.1f", theta, cr))
    end
    ks, rs = reflection_curve(0.7, 1.0; lagged=true)
    plot!(p1, ks, rs; color=:gray, linewidth=1.5, label="θ = 0.7, Cr = 1, η_n of time n (rejected design)")
    for (theta, cr, color) in ((0.7, 1.0, 1), (0.5, 1.0, 3), (0.7, 0.5, 1))
        kd = [0.1, 0.2, 0.4, 0.8]
        rs = [simulated_reflection(theta, cr, x) for x in kd]
        scatter!(p1, kd, rs; color=color, markershape=:circle, markersize=4, label="")
        @printf("simulated theta %.1f Cr %.1f: %s\n", theta, cr, join([@sprintf("k dx %.1f |R| %.2e (analysis %.2e)", kd[i], rs[i],
            reflection(theta, cr, find_omega(theta, cr, kd[i]))[2]) for i in eachindex(kd)], ", "))
    end
    scatter!(p1, [NaN], [NaN]; color=:white, markerstrokecolor=:black, label="simulated (wave maker, fitted)")
    # dispersive waves leaving through the long-wave condition: R = (sqrt(gH) - c)/(sqrt(gH) + c), c^2 = g tanh(kH)/k
    kH = range(1e-3, 4; length=400)
    c = sqrt.(tanh.(kH) ./ kH)
    p2 = plot(kH, (1 .- c) ./ (1 .+ c); xlabel="k H", ylabel="|R|", title="dispersive wave, radiation with √(gH)",
        titlefontsize=10, color=:black, linewidth=2, label="(√(gH) - c)/(√(gH) + c), c² = g tanh(kH)/k", legend=:topleft,
        legendfontsize=7)
    savefig(plot(p1, p2; layout=(1, 2), size=(1150, 440), margin=5Plots.mm), joinpath(FIGURE_DIR, "ob_reflection_analysis.png"))
    for (theta, cr) in ((0.7, 1.0), (0.7, 0.5), (1.0, 1.0), (0.5, 1.0))
        @printf("analysis theta %.1f Cr %.1f: |R| at k dx = 0.1, 0.2, 0.4: %s\n", theta, cr,
            join([@sprintf("%.2e", reflection(theta, cr, find_omega(theta, cr, kd))[2]) for kd in (0.1, 0.2, 0.4)], ", "))
    end
    @printf("lagged theta 0.7 Cr 1: |R| at k dx = 0.1, 0.2: %.3f, %.3f\n",
        (reflection(0.7, 1.0, find_omega(0.7, 1.0, kd; lagged=true); lagged=true)[2] for kd in (0.1, 0.2))...)
    @printf("theta-weighted radiation, theta 0.7 Cr 1: |R| at k dx = 0.1: %.2e (tan^2: %.2e)\n",
        reflection(0.7, 1.0, find_omega(0.7, 1.0, 0.1; radiation_theta=0.7); radiation_theta=0.7)[2], tan(0.025)^2)
end

# ---------------------------------------------------------------------------------------------------------------------
# Characteristic open face (docs/numerics_open_boundaries.md section 8): normal-mode reflection, with a current
# ---------------------------------------------------------------------------------------------------------------------
# Normal-mode analysis in the sense of Gustafsson, Kreiss & Sundstroem (1972) and Vichnevetsky (1987). Units
# g = H = dx = 1 (c = 1, U = Froude number), dt = Cr. Interior: the theta scheme above with a current U. The continuity
# flux is Q_f = u_f + U eta_up(f), with the upwind level as in the face thicknesses of the wetting-drying step. The
# momentum advection is first-order upwind. All terms are theta-weighted; the model's advection is explicit, so this is
# an approximation. Cells j <= 0 (j = 0 the last cell N); face index j = face j + 1/2; face 0 is the open face with
# eta_b and u_b. The equations whose stencil reaches eta_b or u_b:
#   continuity of cell 0 (open-face flux u_b + U eta_up, eta_up = eta_0 for outflow and eta_b for inflow, or eta_b for
#     both with face_level_b = true, the present model);
#   momentum of face -1 when U < 0 (its upwind stencil reaches u_b);
#   natural:   (lambda-1) u_b + dt Theta [U adv_b + (eta_b - eta_0)/(1/2)]
#              + cb (lambda-1) [beta eta_b + gamma eta_{-1} - (beta + gamma) eta_0] = 0,
#              adv_b = u_b - u_{-1} for outflow, 0 for inflow (zero-gradient flux through the outer face);
#   essential: (lambda-1)(c eta_b - H u_b) = 0   (Delta F = 0, eps = 0; u-form, c the intrinsic speed).
# Modes eta_j = A z^j, u_j = B z^j are the roots of the cubic det M(z) = 0. The incident mode is the right-going physical
# one; the reflected modes have |z| > 1 (bounded for j -> -infinity). Their number is always the number of equations
# minus 2 (checked). R compares the left-going physical mode with the incident one, both at the face x = 1/2. The
# frequency is that of the continuous right-going wave, omega = (U + c) k.

const CA_KDX = range(0.02, 1.2; length=120)   # -, k dx of the curves
const CA_FROUDE = 0.07                         # -, U/c of Klopman's flume (0.16 m/s in 0.5 m depth)

function _ca_modes(lam, dt, theta, U)
    Th = theta * lam + 1 - theta
    a, e = lam - 1, dt * Th
    # cubic in z, coefficients (z^3, z^2, z, 1), from det M(z) z^2 (U >= 0) or det M(z) z (U < 0)
    p = U >= 0 ?
        [-e^2, (a + e * U)^2 + 2e^2, -2(a + e * U) * e * U - e^2, e^2 * U^2] :
        [e^2 * U^2, 2e * U * (a - e * U) - e^2, (a - e * U)^2 + 2e^2, -e^2]
    p = p[findfirst(x -> abs(x) > 1e-14 * maximum(abs, p), p):end]
    n = length(p) - 1
    comp = zeros(ComplexF64, n, n)            # companion matrix of the monic polynomial
    comp[1, :] .= -p[2:end] ./ p[1]
    for i in 2:n
        comp[i, i - 1] = 1
    end
    zs = eigvals(comp)
    vec = map(zs) do z
        m11 = U >= 0 ? a + e * U * (1 - 1 / z) : a + e * U * (z - 1)
        m12 = e * (1 - 1 / z)
        abs(m12) > abs(m11) ? ComplexF64[1, -m11 / m12] : ComplexF64[-m12 / m11, 1]
    end
    return zs, vec, Th
end

function characteristic_reflection(kdx; theta=0.7, cr=1.0, U=0.0, beta=0.0, gamma=0.0, cb=1.0, face_level_b=false)
    dt = cr
    lam = exp(-im * (U + 1) * kdx * dt)
    zs, vs, Th = _ca_modes(lam, dt, theta, U)
    inc = argmin(abs.(log.(zs) .- im * kdx))
    refl = [i for i in eachindex(zs) if i != inc && abs(zs[i]) > 1 - 1e-9]
    nr = length(refl)
    n = nr + 2                                  # unknowns: reflected amplitudes, eta_b, u_b; x[n + 1] = 1 (incident)
    a, e = lam - 1, dt * Th
    field(x, w, j) = (w == 2 && j >= 0) ? x[nr + 2] :
                     x[n + 1] * vs[inc][w] * zs[inc]^j + sum((x[i] * vs[refl[i]][w] * zs[refl[i]]^j for i in 1:nr); init=0im)
    eta(x, j) = field(x, 1, j)
    u(x, j) = field(x, 2, j)
    etab(x) = x[nr + 1]
    eta_up(x, f) = f == 0 ? ((U >= 0 && !face_level_b) ? eta(x, 0) : etab(x)) : (U >= 0 ? eta(x, f) : eta(x, f + 1))
    Q(x, f) = u(x, f) + U * eta_up(x, f)
    adv(x, f) = U >= 0 ? u(x, f) - u(x, f - 1) : u(x, f + 1) - u(x, f)
    eqs = Function[x -> a * eta(x, 0) + e * (Q(x, 0) - Q(x, -1))]
    U < 0 && push!(eqs, x -> a * u(x, -1) + e * (U * adv(x, -1) + eta(x, 0) - eta(x, -1)))
    push!(eqs, x -> a * u(x, 0) + e * (U * (U >= 0 ? u(x, 0) - u(x, -1) : 0.0) + (etab(x) - eta(x, 0)) / 0.5) +
                    cb * a * (beta * etab(x) + gamma * eta(x, -1) - (beta + gamma) * eta(x, 0)))
    push!(eqs, x -> a * (etab(x) - u(x, 0)))
    length(eqs) == n || error("normal-mode count: $nr reflected modes for $(length(eqs)) equations")
    A = zeros(ComplexF64, n, n)
    rhs = zeros(ComplexF64, n)
    for (r, eq) in enumerate(eqs)
        x0 = zeros(ComplexF64, n + 1)
        x0[n + 1] = 1
        rhs[r] = -eq(x0)
        for col in 1:n
            x1 = zeros(ComplexF64, n + 1)
            x1[col] = 1
            A[r, col] = eq(x1)
        end
    end
    x = A \ rhs
    phys = argmin([abs(abs(zs[i]) - 1) for i in refl])
    i = refl[phys]
    return abs(x[phys] * vs[i][1] / vs[inc][1] * sqrt(zs[i] / zs[inc]))
end

function characteristic_figure()
    p1 = plot(; xlabel="k Δx", ylabel="|R|", yscale=:log10, ylims=(1e-8, 0.3), xlims=(0, maximum(CA_KDX)),
        title="characteristic face, no current (θ = 0.7)", titlefontsize=10, legend=:bottomright, legendfontsize=7)
    k = collect(CA_KDX)
    plot!(p1, k, characteristic_reflection.(k); color=:black, linewidth=2.5, label="β = 0 (= θ-weighted radiation), Cr = 1")
    plot!(p1, k, characteristic_reflection.(k; beta=-0.5); color=1, linewidth=2, label="β = -1/2 (box), Cr = 1")
    plot!(p1, k, characteristic_reflection.(k; beta=-2 / 3, gamma=-1 / 12); color=2, linewidth=1.5,
        label="β = -2/3, γ = -1/12 (third point), Cr = 1")
    plot!(p1, k, characteristic_reflection.(k; cr=0.2, beta=-2 * 0.7 * 0.2); color=1, linestyle=:dash, linewidth=1.5,
        label="Cr = 0.2, β limited to -2θCr (w ≥ 0)")
    plot!(p1, k, k .^ 2 ./ 16; color=:gray, linestyle=:dot, label="(kΔx)²/16, (kΔx)³/64")
    plot!(p1, k, k .^ 3 ./ 64; color=:gray, linestyle=:dot, label="")
    p2 = plot(; xlabel="k Δx", ylabel="|R|", yscale=:log10, ylims=(1e-8, 0.3), xlims=(0, maximum(CA_KDX)),
        title=@sprintf("with a current (Fr = ±%.2f), β = -1/2, Cr = 1", CA_FROUDE), titlefontsize=10,
        legend=:bottomright, legendfontsize=7)
    F = CA_FROUDE
    plot!(p2, k, characteristic_reflection.(k; U=F, beta=-0.5, cb=1 - F, face_level_b=true); color=:gray, linewidth=2,
        label="outflow, face thickness from η_b (present model)")
    plot!(p2, k, characteristic_reflection.(k; U=F, beta=-0.5, cb=1 - F); color=1, linewidth=2,
        label="outflow, upwind thickness η_N, cb = c - |U|")
    plot!(p2, k, characteristic_reflection.(k; U=-F, beta=-0.5, cb=1 - F); color=2, linewidth=2,
        label="inflow (upwind = η_b), cb = c - |U|")
    plot!(p2, k, characteristic_reflection.(k; U=F); color=1, linestyle=:dash, linewidth=1.5, label="outflow, upwind, β = 0")
    plot!(p2, k, characteristic_reflection.(k; beta=-0.5); color=:black, linestyle=:dot, linewidth=1.5, label="no current, β = -1/2")
    savefig(plot(p1, p2; layout=(1, 2), size=(1150, 440), margin=5Plots.mm), joinpath(FIGURE_DIR, "ob_characteristic_analysis.png"))

    kd = (0.1, 0.2, 0.4, 0.8)
    row(f) = join([@sprintf("%.2e", f(x)) for x in kd], ", ")
    @printf("characteristic face, |R| at k dx = %s:\n", join(kd, ", "))
    # consistency with the analysis of section 5: beta = 0 is the theta-weighted radiation condition
    @printf("  beta 0, theta 0.7, Cr 1:           %s\n", row(x -> characteristic_reflection(x)))
    @printf("  section-5 theta-weighted radiation: %s (same discrete mode, other derivation)\n",
        row(x -> reflection(0.7, 1.0, 1.0 * x; radiation_theta=0.7)[2]))   # omega dt = Cr k dx
    for (theta, cr) in ((0.7, 1.0), (0.7, 0.5), (0.5, 1.0), (1.0, 1.0), (0.7, 2.0))
        @printf("  beta -1/2, theta %.1f, Cr %.1f:     %s\n", theta, cr, row(x -> characteristic_reflection(x; theta, cr, beta=-0.5)))
    end
    @printf("  beta -2/3, gamma -1/12:            %s\n", row(x -> characteristic_reflection(x; beta=-2 / 3, gamma=-1 / 12)))
    for cr in (0.1, 0.2, 0.3)
        @printf("  Cr %.1f, beta limited to %.2f:     %s\n", cr, -2 * 0.7 * cr, row(x -> characteristic_reflection(x; cr, beta=-2 * 0.7 * cr)))
    end
    for U in (F, -F, 0.2, -0.2)
        @printf("  U %+.2f: beta 0 %s | beta -1/2 cb c %s | cb c-|U| %s | face level eta_b, beta -1/2 %s\n", U,
            row(x -> characteristic_reflection(x; U)), row(x -> characteristic_reflection(x; U, beta=-0.5)),
            row(x -> characteristic_reflection(x; U, beta=-0.5, cb=1 - abs(U))),
            row(x -> characteristic_reflection(x; U, beta=-0.5, cb=1 - abs(U), face_level_b=true)))
    end
end

cut_cells_sketch()
reflection_figure()
characteristic_figure()
