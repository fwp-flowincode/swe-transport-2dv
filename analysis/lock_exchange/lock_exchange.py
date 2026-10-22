"""Bulk reference values and evaluation tools for the lock-exchange case.

There is no analytical solution for the evolution of the lock exchange; the 2DV model of this
repository computes it. This module provides the bulk estimates of theory against which model
runs are evaluated, and the evaluation of model output (front positions and speed, nose angle,
heavy-fluid volume, time-step limits). Case: the full-depth lock exchange after Jankowski (1999);
table numbers below refer to the tables of its case description:

- domain / physical / numerical parameters (Tables 1-3, after Jankowski 1999),
- linear salinity equation of state rho(s) = rho_2 (1 + beta s),
- reduced gravity g' = g * delta_rho / rho_0,
- available potential energy  dE = 1/8 (rho_1 - rho_2) g H^2,
- front speed U = Fr * sqrt(g' d) and front positions x_lock +/- U t (Table 5),
- 60 deg nose angle (von Karman 1940; Benjamin 1968),
- bulk Richardson number Ri_0 = g' (d/2) / (2U)^2 vs. the Miles-Howard limit 1/4.

Coordinates: x [m] along the channel, z [m] height above the bed (0 <= z <= H).
The heavy fluid starts on the left (x < x_lock) and advances to the right along the bed.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

# Gravitational acceleration [m/s^2] as used in the case description (Table 2).
# WHY: the doc fixes g = 9.81 m/s^2 (not the CODATA standard 9.80665) so that the
# tabulated reference values (g', U, Table 5) are reproduced exactly.
G_ACCEL = 9.81

# Critical gradient Richardson number for inviscid stratified shear flow [-]
# (Miles 1961; Howard 1961): unstable to small disturbances when Ri < 1/4.
RI_CRITICAL = 0.25

# Nose angle at the stagnation point on the bed [deg] (von Karman 1940; Benjamin 1968).
NOSE_ANGLE_DEG = 60.0

# Froude number for the ideal, energy-conserving current, Fr = U / sqrt(g' d) [-]
# (Benjamin 1968).
FR_BENJAMIN = 0.5

# Literature Froude numbers (case description, Table 4).
FROUDE_LITERATURE: List[Dict] = [
    {"condition": "Ideal, energy-conserving (symmetric)", "fr": 0.50, "source": "Benjamin (1968)"},
    {"condition": "Closed conduit", "fr": 0.44, "source": "Barr (Jankowski, 1999)"},
    {"condition": "Free surface - underflow", "fr": 0.47, "source": "Jankowski (1999)"},
    {"condition": "Free surface - overflow", "fr": 0.59, "source": "Jankowski (1999)"},
]

# Acceptance range for the front Froude number in this validation [-] (doc, "Front propagation speed").
FR_TARGET_MIN = 0.47
FR_TARGET_MAX = 0.50
# Absolute tolerance on the target-range check [-].
# WHY: a model exactly at Fr = 0.50 must pass despite least-squares round-off.
FR_TARGET_ATOL = 1e-6

# Grid resolutions named in the case description (nx, nz).
GRID_PRESETS: Dict[str, tuple] = {
    "Baseline (Table 3)": (120, 20),
    "Non-hydrostatic KH (suggested)": (600, 80),
    "Non-hydrostatic KH (Figure 2)": (1500, 200),
}

# Minimum number of grid cells across the interfacial shear layer for the fastest-growing
# K-H mode to develop [-] (doc, "Kelvin-Helmholtz instability": "~8-10 cells").
KH_MIN_CELLS_ACROSS_SHEAR_LAYER = 8


@dataclass
class LockExchangeParameters:
    """Parameters of the lock-exchange case (case description, Tables 1-3).

    Attributes:
        L: channel length [m] (Table 1: 30.0).
        H: channel depth d = H [m] (Table 1: 4.0).
        W: channel width [m] (Table 1: 3.0); the case is invariant in y.
        x_lock: initial gate position [m] (Table 1: 15.0).
        rho_light: light fluid density rho_2 [kg/m^3] (Table 2: 999.972).
        eos_beta: linear haline contraction coefficient beta [1/psu] in
            rho(s) = rho_2 (1 + beta s); 0.75e-3 gives rho_1 = 1000.722 kg/m^3 at s = 1 psu.
        s_heavy: salinity of the heavy fluid [psu] (doc: 1 psu).
        g: gravitational acceleration [m/s^2] (Table 2: 9.81).
        t_end: total simulation time T [s] (Table 3: 100).
        dt: baseline time step [s] (Table 3: 1).
    """

    L: float = 30.0
    H: float = 4.0
    W: float = 3.0
    x_lock: float = 15.0
    rho_light: float = 999.972
    eos_beta: float = 0.75e-3
    s_heavy: float = 1.0
    g: float = G_ACCEL
    t_end: float = 100.0
    dt: float = 1.0


def density_from_salinity(s, rho_light: float, eos_beta: float):
    """Linear equation of state rho(s) = rho_2 (1 + beta s).

    Args:
        s: salinity [psu] (scalar or array).
        rho_light: reference (fresh) density rho_2 [kg/m^3].
        eos_beta: haline contraction coefficient beta [1/psu].

    Returns:
        density [kg/m^3], same shape as ``s``.
    """
    return rho_light * (1.0 + eos_beta * np.asarray(s, dtype=float))


@dataclass
class LockExchangeCalculator:
    """Analytical front kinematics and diagnostics for the full-depth lock exchange.

    Args:
        params: case parameters.
        froude: front Froude number Fr = U / sqrt(g' d) [-], typical range 0.44-0.59
            (Table 4); defaults to the energy-conserving value 0.5 (Benjamin 1968).
    """

    params: LockExchangeParameters = field(default_factory=LockExchangeParameters)
    froude: float = FR_BENJAMIN

    # ------------------------------------------------------------------ densities
    @property
    def rho_heavy(self) -> float:
        """Heavy fluid density rho_1 [kg/m^3] from the linear EOS at s = s_heavy."""
        p = self.params
        return float(density_from_salinity(p.s_heavy, p.rho_light, p.eos_beta))

    @property
    def delta_rho(self) -> float:
        """Density difference rho_1 - rho_2 [kg/m^3]."""
        return self.rho_heavy - self.params.rho_light

    @property
    def rho_0(self) -> float:
        """Reference density rho_0 = (rho_1 + rho_2) / 2 [kg/m^3]."""
        return 0.5 * (self.rho_heavy + self.params.rho_light)

    @property
    def g_prime(self) -> float:
        """Reduced gravity g' = g * delta_rho / rho_0 [m/s^2]."""
        return self.params.g * self.delta_rho / self.rho_0

    # ------------------------------------------------------------------ kinematics
    def front_speed(self, froude: Optional[float] = None) -> float:
        """Front speed U = Fr * sqrt(g' d) [m/s] (Benjamin 1968, Fr = 0.5 ideal)."""
        fr = self.froude if froude is None else froude
        return fr * np.sqrt(self.g_prime * self.params.H)

    def front_positions(self, t, froude: Optional[float] = None) -> Dict[str, np.ndarray]:
        """Heavy and light front positions [m] at times t [s].

        x_heavy(t) = x_lock + U t (bottom current, moving right),
        x_light(t) = x_lock - U t (surface current, moving left).
        Valid for t <= t_wall (fronts have not reached the end walls); positions are
        clipped to the domain [0, L].
        """
        t = np.asarray(t, dtype=float)
        U = self.front_speed(froude)
        p = self.params
        return {
            "x_heavy": np.clip(p.x_lock + U * t, 0.0, p.L),
            "x_light": np.clip(p.x_lock - U * t, 0.0, p.L),
        }

    def wall_time(self, froude: Optional[float] = None) -> float:
        """Time [s] at which the first front reaches an end wall, min(x_lock, L - x_lock) / U."""
        p = self.params
        return min(p.x_lock, p.L - p.x_lock) / self.front_speed(froude)

    def front_position_table(self, times: Optional[Sequence[float]] = None) -> List[Dict]:
        """Theoretical front positions (case description, Table 5).

        Args:
            times: output times [s]; defaults to 0, 20, ..., t_end.

        Returns:
            rows with t [s], x_heavy [m], x_light [m] for the active Fr and the
            heavy-front position for the target range bounds Fr = 0.47 and 0.50.
        """
        if times is None:
            times = np.arange(0.0, self.params.t_end + 1e-9, 20.0)
        rows = []
        for t in times:
            pos = self.front_positions(t)
            rows.append({
                "t [s]": float(t),
                "x_heavy [m]": float(pos["x_heavy"]),
                "x_light [m]": float(pos["x_light"]),
                f"x_heavy (Fr={FR_TARGET_MIN}) [m]": float(self.front_positions(t, FR_TARGET_MIN)["x_heavy"]),
                f"x_heavy (Fr={FR_TARGET_MAX}) [m]": float(self.front_positions(t, FR_TARGET_MAX)["x_heavy"]),
            })
        return rows

    # ------------------------------------------------------------------ energetics
    def available_potential_energy(self) -> float:
        """Available potential energy per unit plan area [J/m^2] (Jankowski 1999).

        dE = E_i - E_e = 1/8 (rho_1 - rho_2) g H^2, i.e. the difference between the
        side-by-side initial state and the stably layered equilibrium (interface at d/2),
        divided by the channel plan area L * W.
        """
        return 0.125 * self.delta_rho * self.params.g * self.params.H ** 2

    # ------------------------------------------------------------------ stability
    def bulk_richardson(self, froude: Optional[float] = None) -> float:
        """Bulk Richardson number Ri_0 = g' (d/2) / (2U)^2 [-].

        With U = Fr sqrt(g' d) this reduces to Ri_0 = 1 / (8 Fr^2), i.e. 1/2 for Fr = 0.5.
        Ri_0 > 1/4 means bulk-stable (Miles 1961; Howard 1961); K-H billows still form
        just behind the head where local shear is sharper (Turner 1973).
        """
        U = self.front_speed(froude)
        return self.g_prime * (0.5 * self.params.H) / (2.0 * U) ** 2

    def grid_metrics(self, nx: int, nz: int, dt: Optional[float] = None) -> Dict:
        """Grid spacings and the front Courant number for an nx x nz grid.

        Returns:
            dx [m], dz [m], aspect dx/dz [-], and front Courant number
            C = U dt / dx [-] (explicit advection of the density front requires C <= 1).
        """
        p = self.params
        dt = p.dt if dt is None else dt
        dx = p.L / nx
        dz = p.H / nz
        return {
            "nx": nx,
            "nz": nz,
            "dx [m]": dx,
            "dz [m]": dz,
            "dx/dz [-]": dx / dz,
            "Courant U*dt/dx [-]": self.front_speed() * dt / dx,
        }

    # ------------------------------------------------------------------ fields
    def density_field(self, t: float, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Idealised (hydrostatic two-layer) density field rho(x, z, t) [kg/m^3].

        A schematic picture for comparison and tests, not a solution of the flow.

        Initial state (t = 0): step at x_lock, heavy fluid on the left.
        For t > 0 the "rectangular" shallow-water solution described in the case
        description is used: full-depth heavy fluid for x < x_light, a two-layer region with
        the interface at z = d/2 for x_light <= x < x_heavy, and full-depth light fluid
        for x >= x_heavy. The heavy-fluid volume x_lock * H per unit width is conserved.

        Args:
            t: time [s].
            x: 1D x coordinates [m].
            z: 1D heights above the bed [m].

        Returns:
            array of shape (len(z), len(x)).
        """
        pos = self.front_positions(t)
        X, Z = np.meshgrid(np.asarray(x, float), np.asarray(z, float))
        heavy = (X < pos["x_light"]) | ((X < pos["x_heavy"]) & (Z < 0.5 * self.params.H))
        return np.where(heavy, self.rho_heavy, self.params.rho_light)

    def get_computed_parameters(self) -> Dict:
        """Derived quantities for display (units in the key names)."""
        return {
            "rho_heavy [kg/m^3]": self.rho_heavy,
            "delta_rho [kg/m^3]": self.delta_rho,
            "rho_0 [kg/m^3]": self.rho_0,
            "g_prime [m/s^2]": self.g_prime,
            "U [m/s]": self.front_speed(),
            "t_wall [s]": self.wall_time(),
            "dE [J/m^2]": self.available_potential_energy(),
            "Ri_0 [-]": self.bulk_richardson(),
        }

    # ------------------------------------------------------------------ comparison
    def fit_front_speed(self, t, x_front) -> Dict:
        """Least-squares front speed from a model time series of the heavy-front position.

        Fits x_front = x0 + U_model t and reports Fr_model = U_model / sqrt(g' d).

        Args:
            t: times [s] (at least two distinct values).
            x_front: heavy-front positions [m].
        """
        t = np.asarray(t, dtype=float)
        x_front = np.asarray(x_front, dtype=float)
        U_model, x0 = np.polyfit(t, x_front, 1)
        fr_model = U_model / np.sqrt(self.g_prime * self.params.H)
        return {
            "U_model [m/s]": float(U_model),
            "x0 [m]": float(x0),
            "Fr_model [-]": float(fr_model),
            "in_target_range": bool(FR_TARGET_MIN - FR_TARGET_ATOL <= fr_model <= FR_TARGET_MAX + FR_TARGET_ATOL),
        }


# ---------------------------------------------------------------------------- model-output analysis

# Height band above the bed used to fit the nose slope, as a fraction of H [-].
# WHY: the 60 deg result is local to the stagnation point; above roughly half the head height (head ~ H/2)
# the interface turns horizontal, so only the lowest ~H/5 is used.
NOSE_FIT_HEIGHT_FRACTION = 0.2
# Minimum number of cell rows inside that band for a nose-angle estimate [-].
NOSE_FIT_MIN_ROWS = 3


def rightmost_crossing(row: np.ndarray, x: np.ndarray, threshold: float) -> float:
    """Rightmost x [m] at which ``row`` drops below ``threshold`` (linear interpolation); NaN if never above."""
    above = np.nonzero(row > threshold)[0]
    if above.size == 0:
        return np.nan
    j = above[-1]
    if j == row.size - 1:
        return float(x[-1])
    r0, r1 = row[j], row[j + 1]
    return float(x[j] if r0 == r1 else x[j] + (r0 - threshold) / (r0 - r1) * (x[j + 1] - x[j]))


def nose_angle_deg(rho: np.ndarray, x: np.ndarray, z: np.ndarray, rho_heavy: float, rho_light: float,
                   H: float) -> float:
    """Slope of the c = 0.5 isoline at the leading edge of the heavy current, in degrees to the horizontal.

    For every cell row with z <= NOSE_FIT_HEIGHT_FRACTION * H the rightmost c = 0.5 crossing x_f(z) is found,
    and x_f = a + b z is fitted by least squares; the angle is atan2(1, -b), i.e. 90 deg for a vertical front,
    < 90 deg for a front leaning back over the current (theory: 60 deg; von Karman 1940, Benjamin 1968),
    > 90 deg for an overhanging (raised) nose.

    Args:
        rho: density [kg/m^3], shape (nz, nx), row 0 at the bed.
        x, z: cell-centre coordinates [m].
        rho_heavy, rho_light: end-member densities [kg/m^3].
        H: depth [m].

    Returns:
        angle [deg], or NaN if fewer than NOSE_FIT_MIN_ROWS rows lie in the band or no front exists.
    """
    threshold = 0.5 * (rho_heavy + rho_light)
    rows = np.nonzero(z <= NOSE_FIT_HEIGHT_FRACTION * H)[0]
    xf = np.array([rightmost_crossing(rho[k], x, threshold) for k in rows])
    ok = np.isfinite(xf)
    if ok.sum() < NOSE_FIT_MIN_ROWS:
        return np.nan
    b, _ = np.polyfit(z[rows][ok], xf[ok], 1)
    return float(np.degrees(np.arctan2(1.0, -b)))


def heavy_volume(rho: np.ndarray, dx: float, dz: float, rho_heavy: float, rho_light: float) -> float:
    """Heavy-fluid volume per unit width [m^2]: integral of c = (rho - rho_2)/(rho_1 - rho_2) over the x-z plane."""
    return float(np.sum((rho - rho_light) / (rho_heavy - rho_light)) * dx * dz)


def fit_front_speed_window(t: np.ndarray, x_front: np.ndarray, t_min: float, t_max: float,
                           g_prime: float, H: float) -> Dict:
    """Least-squares front speed over t_min <= t <= t_max (skips the initial acceleration and wall reflection).

    Returns U_model [m/s] (absolute value, so it works for both fronts), Fr_model = U_model / sqrt(g' H) [-],
    and the number of samples used; NaN values if fewer than two finite samples are in the window.
    """
    sel = (t >= t_min) & (t <= t_max) & np.isfinite(x_front)
    if sel.sum() < 2:
        return {"U_model [m/s]": np.nan, "Fr_model [-]": np.nan, "n_samples": int(sel.sum())}
    slope = np.polyfit(t[sel], x_front[sel], 1)[0]
    U = abs(float(slope))
    return {"U_model [m/s]": U, "Fr_model [-]": U / np.sqrt(g_prime * H), "n_samples": int(sel.sum())}


# ---------------------------------------------------------------------------- time-step limits (prototype 2DV model)

# Largest horizontal Courant number U_f * dt / dx allowed for the prototype 2DV model [-], with U_f the theoretical
# front speed 0.5 sqrt(g' H). WHY: local velocities reach about 2 U_f, and explicit advection needs a true Courant
# number <= 1; in a parameter scan (L 10-100 m, H 1-10 m, 40-1500 x 8-200 cells, drho 0.05-5 kg/m^3, both pressure
# modes) no run failed at 0.5, while a non-hydrostatic 600 x 80 run blew up at 0.75. The vertical spacing does not
# limit stability (runs with U_f dt/dz up to 250 stayed stable), only accuracy (below).
COURANT_MAX = 0.5
# Default Courant number U_f dt / min(dx, dz) on every grid [-]. WHY: measured on the case description's grid
# (120 x 20, inviscid, no damping, t = 100 s, reference dt = 0.0025 s): at the document's dt = 1 s (0.43) spikes
# form behind both fronts and the density field deviates by 3.4 % (hydrostatic) / 4.8 % (non-hydrostatic) of the
# density difference on average; at dt = 0.1 s (0.043) by 0.2 % / 2.0 %, without artefacts. On fine grids the
# front speed converges only below about 0.1 (600 x 80 hydrostatic: Fr = 0.39 at 0.17, 0.47 at 0.09; 1500 x 200:
# 0.37 at 0.21, 0.48 at 0.09).
COURANT_DEFAULT = 0.05
# Hydrostatic mode: largest vertical velocity relative to U_f H / dz [-]. WHY: there w is only diagnosed from
# continuity and grows with the vertical resolution; measured max|w| (inviscid, no damping, t <= 100 s): 0.42 m/s
# on 120 x 20 (dz = 0.2 m), 1.04 m/s on 300 x 40 (dz = 0.1 m), 2.5-5 m/s on 1500 x 200 (dz = 0.02 m), i.e. about
# 0.25 U_f H / dz with U_f = 0.086 m/s, H = 4 m.
W_HYDROSTATIC_OVER_UH_DZ = 0.25
# Hydrostatic mode: vertical Courant number max|w| dt / dz of the default and the maximum time step [-].
# WHY: explicit vertical advection; at max|w| dt/dz > 1 (1500 x 200, dt = 0.01 s: up to 2.5) grid-scale shocks,
# spurious mixing and a stalled front appear (experiments/lock_exchange/plot_density_hydro_vs_nonhydro_100s.jl).
VERTICAL_COURANT_DEFAULT = 0.5
VERTICAL_COURANT_MAX = 1.0
# Explicit diffusion limit nu * dt * (1/dx^2 + 1/dz^2) <= 1/2, used with a safety factor of 0.8 [-].
DIFFUSION_NUMBER_MAX = 0.4
# Largest time step offered relative to the default [-]. WHY: on flat-cell grids the stability limit can be
# thousands of times the accurate default (e.g. 29 s vs 0.011 s), which is useless; with the default of the case
# description's grid (0.1 s) the document's dt = 1 s stays selectable.
DT_MAX_OVER_DEFAULT = 10.0
# Smallest time step offered relative to the default [-] (e.g. 0.001 s on the case description's grid).
DT_MIN_OVER_DEFAULT = 0.01
# Mantissas of the offered time steps (1-2-5 series) [-].
DT_SERIES = (1.0, 2.0, 5.0)


def recommended_courant(dx: float, dz: float, H: float) -> float:
    """Default Courant number U_f dt / min(dx, dz) [-] for grid spacings dx, dz [m] and depth H [m].

    COURANT_DEFAULT on every grid (see the constant for the measurements); the arguments are kept so that callers
    need not change if the rule becomes grid dependent again.
    """
    return COURANT_DEFAULT


def _series(lo: float, hi: float) -> list:
    """Values m * 10^e (m in DT_SERIES) with lo <= value <= hi, ascending."""
    out = []
    for e in range(int(np.floor(np.log10(lo))) - 1, int(np.ceil(np.log10(hi))) + 1):
        for m in DT_SERIES:
            v = float(f"{m * 10.0 ** e:.12g}")
            if lo * (1 - 1e-9) <= v <= hi * (1 + 1e-9):
                out.append(v)
    return out


def time_step_bounds(dx: float, dz: float, H: float, front_speed: float, nu: float = 0.0,
                     hydrostatic: bool = False) -> Dict:
    """Time steps [s] offered for the model: a 1-2-5 series from a small minimum to the stability maximum.

    default: COURANT_DEFAULT * min(dx, dz) / U_f (at most the stability maximum), rounded down to the series;
             hydrostatic also at most VERTICAL_COURANT_DEFAULT dz / w with w = W_HYDROSTATIC_OVER_UH_DZ U_f H / dz.
    max:     min(COURANT_MAX * dx / U_f, DIFFUSION_NUMBER_MAX / (nu (1/dx^2 + 1/dz^2)),
             DT_MAX_OVER_DEFAULT * default), rounded down to the series; hydrostatic also at most
             VERTICAL_COURANT_MAX dz / w.
    min:     DT_MIN_OVER_DEFAULT * default, rounded down to the series.

    Returns {"options": [...], "default", "min", "max", "decimals"} (decimals: enough to show every option).

    Args:
        dx, dz: grid spacings [m]; H: depth [m]; front_speed: U_f = 0.5 sqrt(g' H) [m/s];
        nu: kinematic viscosity [m^2/s]; hydrostatic: the run includes the hydrostatic mode.
    """
    dt_stable = COURANT_MAX * dx / front_speed
    if nu > 0.0:
        dt_stable = min(dt_stable, DIFFUSION_NUMBER_MAX / (nu * (1.0 / dx ** 2 + 1.0 / dz ** 2)))
    dt_accurate = recommended_courant(dx, dz, H) * min(dx, dz) / front_speed
    if hydrostatic:
        w_max = W_HYDROSTATIC_OVER_UH_DZ * front_speed * H / dz
        dt_stable = min(dt_stable, VERTICAL_COURANT_MAX * dz / w_max)
        dt_accurate = min(dt_accurate, VERTICAL_COURANT_DEFAULT * dz / w_max)
    dt_default = min(dt_accurate, dt_stable)
    dt_max = min(dt_stable, DT_MAX_OVER_DEFAULT * dt_default)
    options = _series(DT_MIN_OVER_DEFAULT * dt_default / 5.0, dt_max)

    def down(value: float) -> float:
        return max(v for v in options if v <= value * (1 + 1e-9))

    lo, default, hi = down(DT_MIN_OVER_DEFAULT * dt_default), down(dt_default), down(dt_max)
    options = [v for v in options if lo <= v <= hi]
    decimals = max(0, max(len(f"{v:.12g}".split(".")[1]) if "." in f"{v:.12g}" else 0 for v in options))
    return {"options": options, "default": default, "min": lo, "max": hi, "decimals": decimals}
