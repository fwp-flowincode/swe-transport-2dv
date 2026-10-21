"""Verification tests for lock_exchange.py: the bulk reference values of the lock-exchange
case (full-depth lock exchange after Jankowski 1999, tables of its case description) and the
evaluation of model output (front speed fit, nose angle, heavy-fluid volume, time steps)."""

import numpy as np
import pytest

from lock_exchange import (
    FR_BENJAMIN,
    LockExchangeCalculator,
    LockExchangeParameters,
)

# Relative tolerance for closed-form identities (floating-point round-off only) [-].
RTOL_EXACT = 1e-12
# Absolute tolerance for values printed in the document with 2 decimals [m].
ATOL_DOC_POSITION_M = 0.01
# Relative tolerance for values printed in the document with 3 significant digits [-].
RTOL_DOC_3SIG = 5e-3
# Relative tolerance for the midpoint-rule potential energy integral [-].
RTOL_QUADRATURE = 1e-6


@pytest.fixture
def calc():
    return LockExchangeCalculator(LockExchangeParameters(), froude=FR_BENJAMIN)


def test_verify_densities_table2(calc):
    """EOS rho(s) = rho_2 (1 + 0.75e-3 s) reproduces Table 2."""
    assert calc.rho_heavy == pytest.approx(1000.722, abs=1e-3)
    assert calc.delta_rho == pytest.approx(0.75, abs=1e-3)
    assert calc.rho_0 == pytest.approx(1000.347, abs=1e-3)
    assert calc.g_prime == pytest.approx(7.36e-3, rel=RTOL_DOC_3SIG)


def test_verify_front_speed_and_wall_time(calc):
    """U = 0.5 sqrt(g'd) ~ 0.086 m/s and t_wall = x_lock / U ~ 175 s (doc, 'Front position')."""
    U = calc.front_speed()
    assert U == pytest.approx(0.5 * np.sqrt(calc.g_prime * 4.0), rel=RTOL_EXACT)
    assert U == pytest.approx(0.086, rel=RTOL_DOC_3SIG)
    assert calc.wall_time() == pytest.approx(15.0 / U, rel=RTOL_EXACT)
    assert calc.wall_time() == pytest.approx(175.0, rel=RTOL_DOC_3SIG)


def test_verify_front_positions_table5(calc):
    """Heavy-front positions of Table 5."""
    expected = {0: 15.0, 20: 16.72, 40: 18.44, 60: 20.16, 80: 21.88, 100: 23.6}
    rows = calc.front_position_table()
    assert [r["t [s]"] for r in rows] == list(map(float, expected))
    for row in rows:
        # WHY: Table 5 was computed with the rounded speed U = 0.086 m/s, so the
        # deviation grows linearly in t; allow the rounding error U_round_err * t.
        tol = ATOL_DOC_POSITION_M + 5e-4 * row["t [s]"]
        assert row["x_heavy [m]"] == pytest.approx(expected[int(row["t [s]"])], abs=tol)
        assert row["x_heavy [m]"] + row["x_light [m]"] == pytest.approx(2 * 15.0, rel=RTOL_EXACT)


def test_verify_bulk_richardson(calc):
    """Ri_0 = g'(d/2) / (2U)^2 = 1/(8 Fr^2) = 1/2 for Fr = 0.5."""
    assert calc.bulk_richardson() == pytest.approx(0.5, rel=RTOL_EXACT)
    assert calc.bulk_richardson(0.47) == pytest.approx(1.0 / (8 * 0.47 ** 2), rel=RTOL_EXACT)


def _potential_energy_per_area(rho, z, dz, L):
    """Midpoint-rule integral of rho g z over the x-z plane divided by L [J/m^2]."""
    dx = L / rho.shape[1]
    return np.sum(rho * 9.81 * z[:, None]) * dx * dz / L


def test_verify_available_potential_energy(calc):
    """dE = 1/8 dRho g H^2 equals the integrated PE difference between initial and layered states."""
    p = calc.params
    nx, nz = 600, 400
    x = (np.arange(nx) + 0.5) * p.L / nx
    z = (np.arange(nz) + 0.5) * p.H / nz
    dz = p.H / nz
    rho_initial = calc.density_field(0.0, x, z)
    rho_layered = np.where(z[:, None] < 0.5 * p.H, calc.rho_heavy, p.rho_light) * np.ones((1, nx))
    dE_num = (_potential_energy_per_area(rho_initial, z, dz, p.L)
              - _potential_energy_per_area(rho_layered, z, dz, p.L))
    assert dE_num == pytest.approx(calc.available_potential_energy(), rel=RTOL_QUADRATURE)


@pytest.mark.parametrize("t", [0.0, 20.0, 57.3, 100.0])
def test_verify_density_field_conserves_heavy_volume(calc, t):
    """Idealised two-layer field keeps the heavy-fluid area x_lock * H per unit width."""
    p = calc.params
    nx, nz = 3000, 400
    x = (np.arange(nx) + 0.5) * p.L / nx
    z = (np.arange(nz) + 0.5) * p.H / nz
    rho = calc.density_field(t, x, z)
    heavy_area = np.sum(rho == calc.rho_heavy) * (p.L / nx) * (p.H / nz)
    # WHY: fronts fall inside cells, so the discrete area error is at most one column each.
    assert heavy_area == pytest.approx(p.x_lock * p.H, abs=2 * (p.L / nx) * p.H)


def test_verify_fit_front_speed_recovers_theory(calc):
    """A synthetic series on the theoretical line returns Fr = 0.5 exactly."""
    t = np.linspace(0.0, 100.0, 11)
    fit = calc.fit_front_speed(t, calc.front_positions(t)["x_heavy"])
    assert fit["Fr_model [-]"] == pytest.approx(FR_BENJAMIN, rel=1e-10)
    assert fit["in_target_range"]


@pytest.mark.parametrize("angle_deg", [45.0, 60.0, 90.0])
def test_verify_nose_angle_recovers_synthetic_wedge(angle_deg):
    """A sharp wedge front with a known slope is measured to within one cell's resolution."""
    from lock_exchange import nose_angle_deg
    L, H, nx, nz = 30.0, 4.0, 3000, 200
    x = (np.arange(nx) + 0.5) * L / nx
    z = (np.arange(nz) + 0.5) * H / nz
    x_toe = 20.0
    # heavy below the plane x = x_toe - z / tan(angle)
    X, Z = np.meshgrid(x, z)
    rho = np.where(X < x_toe - Z / np.tan(np.radians(angle_deg)), 1000.722, 999.972)
    # WHY: crossings are interpolated between cell centres, so the error is O(dx/(0.2 H)) in slope.
    assert nose_angle_deg(rho, x, z, 1000.722, 999.972, H) == pytest.approx(angle_deg, abs=1.0)


def test_verify_heavy_volume_of_initial_state():
    """Heavy-fluid volume of the step initial condition equals x_lock * H."""
    from lock_exchange import heavy_volume
    calc = LockExchangeCalculator()
    p = calc.params
    nx, nz = 120, 20
    x = (np.arange(nx) + 0.5) * p.L / nx
    z = (np.arange(nz) + 0.5) * p.H / nz
    rho = calc.density_field(0.0, x, z)
    vol = heavy_volume(rho, p.L / nx, p.H / nz, calc.rho_heavy, p.rho_light)
    assert vol == pytest.approx(p.x_lock * p.H, rel=RTOL_EXACT)


@pytest.mark.parametrize("nx, nz, dt_default_expected", [(120, 20, 1.0), (300, 40, 0.18), (600, 80, 0.05),
                                                           (1500, 200, 0.02), (120, 200, 0.02)])
def test_verify_time_step_bounds_document_setup(calc, nx, nz, dt_default_expected):
    """Document grid keeps dt = 1 s; fine or flat-cell grids default to U dt/min(dx, dz) = 0.1; max stays at 0.5."""
    from lock_exchange import COURANT_MAX, time_step_bounds
    p = calc.params
    dx, dz, U = p.L / nx, p.H / nz, calc.front_speed()
    b = time_step_bounds(dx, dz, p.H, U)
    assert b["default"] == pytest.approx(dt_default_expected, abs=1e-9)
    assert b["step"] <= b["default"] <= b["max"]
    assert U * b["max"] / dx <= COURANT_MAX + 1e-9
    if b["max"] < 10.0 * b["default"] - 1e-9:  # not capped at DT_MAX_OVER_DEFAULT
        assert U * b["max"] / dx > COURANT_MAX * 0.85  # rounding down to the step loses < 15 %
    assert b["max"] / b["step"] <= 200  # slider stays usable
    assert U * b["default"] / min(dx, dz) <= 0.45 + 1e-9


def test_verify_time_step_bounds_viscous_limit(calc):
    """With viscosity, the maximum also respects the explicit diffusion limit nu dt (1/dx^2 + 1/dz^2) <= 0.4."""
    from lock_exchange import DIFFUSION_NUMBER_MAX, time_step_bounds
    dx, dz, nu = 0.25, 0.005, 1e-4
    b = time_step_bounds(dx, dz, 1.0, 0.04, nu=nu)
    assert nu * b["max"] * (1 / dx ** 2 + 1 / dz ** 2) <= DIFFUSION_NUMBER_MAX + 1e-9
