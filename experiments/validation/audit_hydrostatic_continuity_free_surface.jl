using SweTransport2DV
using Printf

# Free-surface continuity references:
# - Stoker (1957), Water Waves
# - Gill (1982), Atmosphere-Ocean Dynamics
#
# Checks:
# 1) In-domain divergence residual from reconstructed (u,w)
# 2) Top-row divergence residual
# 3) Kinematic free-surface mismatch max |w_top - d(eta)/dt|
# 4) Closed-boundary volume and constituent-mass drift

function main()
    cfg = ModelConfig(
        length=30.0,
        depth=4.0,
        nx=120,
        nz=40,
        dt=0.001,
        nt=1,
        use_hydrostatic=true,
        # WHY: this audit checks the LEGACY fixed-grid closure identity (div=0 on every row including the top one).
        # The (now-default) variable top layer closes a different identity in the top row instead (see
        # test/TESTS_TECHNICAL_DOCUMENTATION.md section 13 and test_verify_variable_top_layer_closes_kinematic_and_volume_budgets
        # in test/test_variable_top_layer.jl), so it is pinned off here explicitly.
        variable_top_layer=false,
        momentum_advection_scheme="upwind1",
        rayleigh_damping=0.0,
        apply_froude_limiter=false,
        write_plots=false,
        write_animation=false,
        write_pressure_buoyancy_diagnostics=false,
        pressure_solver_method="spectral",
        output_dir=joinpath("output_barotropic", "validation", "hydrostatic_continuity_audit"),
    )

    model = Model(cfg)
    d0 = compute_diagnostics(model)

    max_div = 0.0
    max_top_div = 0.0
    max_kinematic_mismatch = 0.0

    for n in 1:400
        step_forward!(model)
        div = div_x_u(model.state.u, model.dx) .+ div_z_w(model.state.w, model.dz)
        max_div = max(max_div, maximum(abs, div))
        max_top_div = max(max_top_div, maximum(abs, @view div[end, :]))
        max_kinematic_mismatch = max(max_kinematic_mismatch, model.last_kinematic_mismatch)
    end

    d1 = compute_diagnostics(model)
    vol_err = d1["water_volume"] - d0["water_volume"]
    const_err = d1["constituent_mass"] - d0["constituent_mass"]
    bnd_flux = d1["boundary_flux"]

    @printf(
        "AUDIT=hydrostatic_continuity max_div=%.3e max_top_div=%.3e max_kinematic=%.3e vol_err=%.3e const_err=%.3e boundary_flux=%.3e\n",
        max_div,
        max_top_div,
        max_kinematic_mismatch,
        vol_err,
        const_err,
        bnd_flux,
    )

    if max_div > 1e-10
        error(@sprintf("Hydrostatic continuity audit failed: max_div %.3e > 1e-10", max_div))
    end
    if max_top_div > 1e-10
        error(@sprintf("Hydrostatic continuity audit failed: max_top_div %.3e > 1e-10", max_top_div))
    end
    if max_kinematic_mismatch > 1e-10
        error(@sprintf("Hydrostatic continuity audit failed: max_kinematic %.3e > 1e-10", max_kinematic_mismatch))
    end
    if abs(vol_err) > 1e-8
        error(@sprintf("Hydrostatic continuity audit failed: volume error %.3e exceeds 1e-8", vol_err))
    end
    if abs(const_err) > 1e-8
        error(@sprintf("Hydrostatic continuity audit failed: constituent error %.3e exceeds 1e-8", const_err))
    end
    if abs(bnd_flux) > 1e-10
        error(@sprintf("Hydrostatic continuity audit failed: boundary flux %.3e exceeds 1e-10", bnd_flux))
    end

    println("Hydrostatic continuity/free-surface audit passed.")
end

main()
