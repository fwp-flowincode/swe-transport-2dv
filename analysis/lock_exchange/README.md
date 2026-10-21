# Lock exchange: bulk reference values and evaluation of model runs

Python tools for the full-depth lock-exchange case after Jankowski (1999).
There is no analytical solution for the evolution of the lock exchange; the
2DV model of this repository (`experiments/lock_exchange/`) provides the
reference. These tools

* compute the bulk estimates of theory: densities from the equation of state,
  reduced gravity, front speed U = Fr √(g′d) and front positions, available
  potential energy, bulk Richardson number;
* evaluate model output: front positions and fitted front speed, nose angle,
  heavy-fluid volume;
* give time-step limits for model runs (Courant and diffusion limits).

```
python -m pip install numpy pytest
python -m pytest analysis/lock_exchange        # tests
```

`density_field` is an idealised two-layer picture for comparison and tests,
not a solution of the flow.
