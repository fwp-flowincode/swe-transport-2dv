# Packages and named tolerances shared by the test files (no hardcoded tolerances inside the tests).
# Included by runtests.jl and at the top of every test_*.jl file, so that each file also runs on its own:
#   julia --project=. test/test_advection.jl
# WHY: re-including is harmless: `using` is idempotent and redefining a const with an identical value is allowed silently.

using Test
using Random
using LinearAlgebra
using SparseArrays
using Statistics
using SweTransport2DV

const TOL_IDENTITY = 1e-11          # discrete identities that hold to round-off (kernel level)
const TOL_MODEL_IDENTITY = 1e-9     # same, after many steps of the full model (round-off accumulates, mass ~ 1e5 kg/m)
