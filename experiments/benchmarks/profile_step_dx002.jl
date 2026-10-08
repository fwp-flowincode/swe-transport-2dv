using SweTransport2DV
using Profile

# Profiling baseline: lock-exchange gravity current (Benjamin, 1968).

cfg = ModelConfig(
    nx=1500,
    nz=200,
    dt=0.01,
    nt=40,
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

Profile.clear()
@profile begin
    for _ in 1:cfg.nt
        step_forward!(model)
    end
end

println("Julia Profile output:")
Profile.print(format=:flat, maxdepth=20)
