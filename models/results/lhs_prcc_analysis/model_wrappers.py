"""Model interface for the LHS+PRCC analysis.

This module wraps the two evaluation back-ends used throughout the
sensitivity analysis:

1. The **closed-form reproduction number** :math:`R_0` (from the NGM derivation),
   reused directly from `../sensitivity_analysis/seirs_sei_r0.py`, evaluated at
   the transmission-season reference climate.

2. The **SEIRS-SEI ODE** (from `deploying ../seirs_sei/lmfit_optimization/
   lmfit_optimization_shared.simulate_year`) used to obtain the seasonal peak of
   infectious humans :math:`I_H`. The ODE consumes the calibrated per-year
   parameters and the 7-compartment state; the climate-response parameters that
   are module-level globals there (biting-rate, mortality, development,
   carrying-capacity terms) are overridden per-sample through a safe
   ``GlobalOverride`` context manager.

It also exposes the per-year calibrated values of the transmission coefficients
:math:`b_1, b_2` (and the rest of the refined parameter vector) via
`load_refined_params`.

Importing this module triggers the ODE back-end's one-time data loading
(climate + cases + population CSVs), which takes a few seconds; subsequent ODE
calls are in-memory and fast.
"""

from __future__ import annotations

import contextlib
import os
import sys

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))

_SENS_DIR = os.path.join(_HERE, "../sensitivity_analysis")
_LMFIT_DIR = os.path.join(_HERE,
                          "../../seirs_sei/lmfit_optimization")

sys.path.insert(0, _SENS_DIR)
sys.path.insert(0, _LMFIT_DIR)

from seirs_sei_r0 import (  # noqa: E402
    BASELINE as R0_BASELINE,
    REFERENCE_CLIMATE,
    N_POP,
    M_PRIME,
)
from sobol_analysis import BOUNDS, GROUPS  # noqa: E402

import lmfit_optimization_shared as _ode  # noqa: E402
from weekly_beta_optimization import load_refined_params  # noqa: E402


# --------------------------------------------------------------------------- #
# Global-override handling for the ODE back-end
# --------------------------------------------------------------------------- #
# Map our analysis parameter names -> (binding, key):
#   ("param", key): value passed inside the simulate_year `p` dict.
#   ("global", attr): value assigned to a module-level global of the ODE module
#                     before simulation (and restored afterwards).
_ODE_BINDING = {
    "H0":      ("param", "H0"),
    "k":       ("param", "k"),
    "phi":     ("param", "phi"),
    "tau_H":   ("param", "tau_H"),
    "gamma":   ("param", "gamma"),
    "omega":   ("param", "omega"),
    "b1":      ("param", "b1"),
    "b2":      ("param", "b2"),
    "M_prime": ("param", "M_prime"),
    "T_prime": ("global", "T_prime"),
    "D1":      ("global", "D1"),
    "R_L":     ("global", "R_L"),
    "DD":      ("global", "DD"),
    "Tmin":    ("global", "Tmin"),
    "Tmax_plasm": ("global", "critical_max_temp_plasm"),
    "Topt_plasm": ("global", "optimal_temp_plasm"),
    "A":       ("global", "A"),
    "B":       ("global", "B"),
    "C":       ("global", "C"),
    "B_E":     ("global", "B_E"),
    "c1":      ("global", "c1"),
    "c2":      ("global", "c2"),
    "Tmin_anop": ("global", "critical_min_temp_anop"),
    "Topt_anop": ("global", "optimal_temp_anop"),
    "Tmax_anop": ("global", "critical_max_temp_anop"),
}

# Names that are settable through the ODE back-end (used for the I_H ODE LHS).
ODE_NAMES = list(_ODE_BINDING.keys())


@contextlib.contextmanager
def override_ode_globals(values: dict):
    """Temporarily set module-level globals of the ODE back-end.

    ``values`` maps analysis parameter *names* to physical values (only the
    globally-bound ones are applied; parametrized ones are ignored here).
    Values are restored on exit.
    """
    saved = {}
    for name, val in values.items():
        binding, attr = _ODE_BINDING[name]
        if binding != "global":
            continue
        cur = getattr(_ode, attr)
        saved[attr] = cur
        setattr(_ode, attr, val)
    try:
        yield
    finally:
        for attr, cur in saved.items():
            setattr(_ode, attr, cur)


def ode_initial_state(year):
    """Per-year calibrated 7-compartment initial state (from refined params)."""
    _, state0, _, _, _ = load_refined_params(year)
    return state0


def ode_fixed_params(year):
    """Per-year calibrated dict passed to ``simulate_year`` (all 9 keys)."""
    fixed, _, _, b1, b2 = load_refined_params(year)
    p = dict(fixed)
    p["b1"], p["b2"] = b1, b2
    return p


# --------------------------------------------------------------------------- #
# Defensible "reference" parameter vector
# --------------------------------------------------------------------------- #
def reference_params(names, years=None):
    """A physiologically defensible baseline for each analysis parameter.

    For the calibrated epidemiological / demographic parameters (H0, k, phi,
    tau_H, gamma, omega, b1, b2, M_prime) the across-year *mean* of the refined
    per-year calibration is used. For the climate-response parameters the
    literature `BASELINE` from the closed-form R0 module is used. This avoids
    the pathological "parameter mid-point" baseline (e.g. gamma ~ 0.5, i.e. a
    2-day recovery) that would otherwise pin the ODE's I_H peak at an
    artificial ceiling.

    Returns {analysis_name: float}.
    """
    if years is None:
        years = list(range(2017, 2024))
    cal = calibrated_b1_b2_by_year(years)
    mean_cal = {c: float(cal[c].mean()) for c in
                ("H0", "k", "phi", "tau_H", "gamma", "omega", "b1", "b2",
                 "M_prime")}
    out = {}
    for n in names:
        if n in mean_cal:
            out[n] = mean_cal[n]
        elif n in R0_BASELINE:
            out[n] = float(R0_BASELINE[n])
        else:
            out[n] = float(np.mean(BOUNDS[n]))
    return out


# --------------------------------------------------------------------------- #
# Seasonal peak of I_H
# --------------------------------------------------------------------------- #
def ih_peak_for_year(year, p, global_overrides=None, **sim_kw):
    """Run the ODE for one year and return the peak of I_H.

    Parameters
    ----------
    year : int
        Simulation year (2017-2023).
    p : dict
        Parameter dict with at least the 9 keys used by ``simulate_year``
        (H0, k, phi, tau_H, gamma, omega, b1, b2, M_prime).
    global_overrides : dict, optional
        {analysis_name: value} for the module-global climate-response params
        (e.g. T_prime, D1, A, B, C, R_L, DD, Tmin, ...). Applied only for the
        duration of this call and restored afterwards.

    Returns
    -------
    float or np.nan
        Peak (max) of the daily I_H over the year; np.nan if integration fails.
    """
    if global_overrides:
        with override_ode_globals(global_overrides):
            return _ih_peak_for_year_impl(year, p, **sim_kw)
    return _ih_peak_for_year_impl(year, p, **sim_kw)


def _ih_peak_for_year_impl(year, p, return_diagnostics=False, **sim_kw):
    state0 = ode_initial_state(year)
    res = _ode.simulate_year(year, state0, p,
                             return_diagnostics=return_diagnostics, **sim_kw)
    if res is None:
        return np.nan
    IH = res[1]
    if not np.all(np.isfinite(IH)):
        return np.nan
    peak = float(np.max(IH))
    return peak


def seasonal_ih_peak(years, p, global_overrides=None):
    """Mean across years of each year's seasonal peak I_H.

    This smooths out single-year climate anomalies and gives a single scalar
    "typical seasonal peak of infectious humans" for the PRCC analysis.
    """
    peaks = [
        ih_peak_for_year(y, p, global_overrides) for y in years
    ]
    peaks = np.array([x for x in peaks if np.isfinite(x)])
    if len(peaks) == 0:
        return np.nan
    return float(peaks.mean())


# --------------------------------------------------------------------------- #
# Closed-form R0
# --------------------------------------------------------------------------- #
def r0_components_kwargs(row_names, row_values):
    """Translate physical parameter values into r0_components kwargs.

    ``M_prime`` is handled by name (the closed-form takes it as a top-level
    argument, not a keyword).
    """
    vals = dict(zip(row_names, row_values))
    from sobol_analysis import _NAME_TO_KW, _ZERO
    kwargs = {
        _NAME_TO_KW[n]: v for n, v in vals.items()
        if n not in _ZERO and n != "M_prime"
    }
    mprime = vals.get("M_prime", M_PRIME)
    return kwargs, mprime


def evaluate_r0_row(row_names, row_values, reference=REFERENCE_CLIMATE,
                    N=N_POP):
    """Evaluate the closed-form R0 for a single parameter row."""
    from seirs_sei_r0 import r0_components
    kwargs, mprime = r0_components_kwargs(row_names, row_values)
    c = r0_components(reference["T"], reference["R"], reference["H"],
                      N=N, M_prime=mprime, **kwargs)
    return c["R0"]


def evaluate_r0_rows(param_values, names=None):
    """Evaluate R0 for each row of ``param_values``.

    ``param_values`` may be an ndarray (columns = ``names``) or a DataFrame
    (columns used as names).
    """
    if isinstance(param_values, pd.DataFrame):
        names = list(param_values.columns)
        rows = param_values.values
    else:
        rows = param_values
    if names is None:
        names = list(BOUNDS.keys())
    out = np.array([evaluate_r0_row(names, row) for row in rows])
    return out


# --------------------------------------------------------------------------- #
# Per-year calibrated b1 / b2 (calibration uncertainty)
# --------------------------------------------------------------------------- #
def calibrated_b1_b2_by_year(years=None):
    """Return a DataFrame of per-year calibrated (b1, b2, gamma, M_prime, ...).

    Columns: year, b1, b2, gamma, omega, tau_H, M_prime, H0, k, phi.
    """
    if years is None:
        years = list(range(2017, 2024))
    rows = []
    for y in years:
        fixed, _, _, b1, b2 = load_refined_params(y)
        rows.append({
            "year": y, "b1": b1, "b2": b2,
            "gamma": fixed["gamma"], "omega": fixed["omega"],
            "tau_H": fixed["tau_H"], "M_prime": fixed["M_prime"],
            "H0": fixed["H0"], "k": fixed["k"], "phi": fixed["phi"],
        })
    return pd.DataFrame(rows)
