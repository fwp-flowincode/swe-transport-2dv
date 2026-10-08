using SweTransport2DV
using Printf

# Benchmark baseline: lock-exchange buoyancy current (Benjamin, 1968).

cases = [
    ("dx0p01", 3000, 400),
    ("dx0p02", 1500, 200),
]
steps = 50
results = Dict{String, Float64}()

for (name, nx, nz) in cases
    cfg = ModelConfig(
        nx=nx,
        nz=nz,
        dt=0.01,
        nt=steps,
        use_hydrostatic=false,
        momentum_advection_scheme="tvd2",
        rayleigh_damping=0.01,
        apply_froude_limiter=true,
        max_froude=0.9,
        write_plots=false,
        write_animation=false,
        write_pressure_buoyancy_diagnostics=false,
        pressure_solver_method="spectral",
    )
    model = Model(cfg)
    t0 = time()
    for _ in 1:steps
        step_forward!(model)
    end
    t1 = time()
    sps = (t1 - t0) / steps
    results[name] = sps
    @printf("%s_S_PER_STEP=%.6f\n", name, sps)
end

speedup = results["dx0p01"] / results["dx0p02"]
@printf("DX0P02_VS_DX0P01_SPEEDUP=%.3f\n", speedup)
