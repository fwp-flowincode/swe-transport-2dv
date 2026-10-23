# Kelvin–Helmholtz shear-layer experiment

Run the full-resolution case from the repository root with:

```sh
julia --project=. experiments/kelvin_helmholtz/run_kelvin_helmholtz_dt001_30s_anim.jl
```

The lateral domain is periodic, so the imposed shear and perturbed interface continue smoothly across the x seam. The
Froude limiter is disabled because the prescribed shear starts above its usual 0.9 threshold; applying it would scale
down the instability-driving velocity jump. The spectral pressure solver uses periodic Fourier modes in x and the
existing cosine transform in z.

The following lower-resolution 30-second preview is included for review:

![Final Kelvin–Helmholtz state](figures/periodic_kh_final.png)

![Kelvin–Helmholtz evolution](figures/periodic_kh_animation.gif)
