"""Global (variance-based / Sobol) sensitivity analysis of the SEIRS-SEI R0.

The analysis evaluates the closed-form reproduction number R0 of the SEIRS-SEI
model (see ``seirs_sei_r0.r0_components`` and Eq. R0 in the paper) at a fixed
reference environmental state (the mean climate over the transmission season,
2017-2023). Each model parameter is drawn independently over its physically /
biologically motivated range, and first-order (S1), second-order (S2) and
total-order (ST) Sobol indices are estimated with the Saltelli sampler.

Parameters that do not enter the closed-form R0 (the human intrinsic incubation
period tau_H and the immunity-waning rate omega) are tabulated separately and
assigned zero sensitivity, since R0 is evaluated at the disease-free equilibrium
and is therefore independent of them.
"""

import numpy as np
import pandas as pd
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

from seirs_sei_r0 import (
    BASELINE,
    REFERENCE_CLIMATE,
    N_POP,
    M_PRIME,
    r0_components,
)

# Logical groupings of the parameters used for reporting / colouring.
GROUPS = {
    "calibrated_epi": ["H0", "k", "phi", "gamma", "b1", "b2", "M_prime"],
    "human_immune": ["tau_H", "omega"],   # do NOT enter R0 (fixed zero sensitivity)
    "climate_rate": [
        "T_prime", "D1", "R_L", "DD", "Tmin",
        "Tmax_plasm", "Topt_plasm",
    ],
    "climate_mortality": ["A", "B", "C", "B_E", "c1", "c2"],
    "anop_suitability": [
        "Tmin_anop", "Topt_anop", "Tmax_anop",
    ],
}

# Parameter -> (lower bound, upper bound). Bounds are biologically/physically
# motivated ranges that bracket the baseline value used in the calibrated model.
BOUNDS = {
    # --- calibrated epidemiological parameters ---
    "H0":        (20.0, 75.0),      # humidity half-saturation threshold (%)
    "k":         (0.01, 1.0),       # humidity sigmoid steepness (%^-1)
    "phi":       (0.01, 0.5),       # baseline survival floor
    "tau_H":     (10.0, 20.0),      # human intrinsic incubation (days)  [not in R0]
    "gamma":     (1.0 / 150.0, 1.0),# human recovery rate (day^-1)
    "omega":     (1.0 / 300.0, 0.1),# immunity waning rate (day^-1)      [not in R0]
    "b1":        (0.001, 0.5),      # human -> mosquito transmission prob.
    "b2":        (0.01, 0.5),       # mosquito -> human transmission prob.
    "M_prime":   (50000.0, 500.0 * N_POP),  # carrying-capacity scale
    # --- temperature / rainfall dependent rates ---
    "T_prime":   (14.0, 20.0),      # biting-rate threshold T' (deg C)
    "D1":        (18.0, 32.0),      # biting-rate thermal scale (deg C day)
    "R_L":       (20.0, 50.0),      # rainfall optimum for habitat (mm)
    "DD":        (80.0, 140.0),     # sporogonic degree-days (deg C day)
    "Tmin":      (12.0, 17.0),      # Plasmodium min development temp (deg C)
    "Tmax_plasm": (27.0, 32.0),     # Plasmodium max development temp (deg C)
    "Topt_plasm": (21.0, 26.0),     # Plasmodium optimal dev temp (deg C)
    # --- adult mosquito mortality / larval development ---
    "A":         (-0.05, -0.01),    # mortality quadratic coeff.
    "B":         (1.0, 1.6),        # mortality linear coeff.
    "C":         (-5.5, -3.5),      # mortality intercept
    "B_E":       (100.0, 300.0),    # eggs per batch
    "c1":        (0.003, 0.008),    # larval development slope
    "c2":        (-0.09, -0.04),    # larval development intercept
    # --- Anopheles temperature suitability (carrying capacity) ---
    "Tmin_anop": (13.0, 19.0),      # Anopheles min temp (deg C)
    "Topt_anop": (22.0, 28.0),      # Anopheles optimal temp (deg C)
    "Tmax_anop": (30.0, 38.0),      # Anopheles max temp (deg C)
}


def build_problem(include_zero_params=True):
    """Return the SALib problem dict.

    If ``include_zero_params`` is True, tau_H and omega (which do not enter the
    closed-form R0) are included together with the active parameters so their
    identically-zero sensitivity is visible in the output tables/figures.
    """
    if include_zero_params:
        names = list(BOUNDS.keys())
    else:
        names = [n for n in BOUNDS if n not in ("tau_H", "omega")]
    bounds = [BOUNDS[n] for n in names]
    return {"num_vars": len(names), "names": names, "bounds": bounds}


_ACTIVE = set(GROUPS["calibrated_epi"]) | set(GROUPS["climate_rate"]) | \
          set(GROUPS["climate_mortality"]) | set(GROUPS["anop_suitability"])
_ZERO = set(GROUPS["human_immune"])

# Map our internal parameter names -> the keyword names accepted by r0_components.
_NAME_TO_KW = {
    "H0": "H0", "k": "k", "phi": "phi",
    "gamma": "gamma", "b1": "b1", "b2": "b2",
    "T_prime": "T_prime", "D1": "D1", "R_L": "R_L", "DD": "DD", "Tmin": "Tmin",
    "Tmax_plasm": "critical_max_temp_plasm",
    "Topt_plasm": "optimal_temp_plasm",
    "A": "A", "B": "B", "C": "C", "B_E": "B_E", "c1": "c1", "c2": "c2",
    "Tmin_anop": "critical_min_temp_anop",
    "Topt_anop": "optimal_temp_anop",
    "Tmax_anop": "critical_max_temp_anop",
}


def _active_kwargs(active):
    """Translate an {internal_name: value} mapping into r0_components kwargs.

    ``M_prime`` is passed separately by the caller and is dropped here, as is
    anything that does not enter R0.
    """
    return {
        _NAME_TO_KW[n]: v for n, v in active.items()
        if n not in _ZERO and n != "M_prime"
    }


def evaluate_R0(param_values, reference=REFERENCE_CLIMATE, N=N_POP,
                M_prime_base=M_PRIME):
    """Evaluate R0 for each row of ``param_values`` (columns match problem.names).

    ``param_values`` are the physical (already un-scaled) values produced by
    SALib (SALib returns them in the original units). Returns a 1-D array of R0.
    """
    names = None
    if isinstance(param_values, pd.DataFrame):
        names = list(param_values.columns)
        rows = param_values.values
    else:
        rows = param_values

    T = reference["T"]
    R = reference["R"]
    H = reference["H"]

    out = np.empty(len(rows))
    for i, row in enumerate(rows):
        active = dict(zip(names, row)) if names is not None else dict(
            zip([n for n in BOUNDS], row))
        Mprime = active.get("M_prime", M_prime_base)
        kwargs = _active_kwargs(active)
        try:
            c = r0_components(T, R, H, N=N, M_prime=Mprime, **kwargs)
            out[i] = c["R0"]
        except Exception:
            out[i] = np.nan
    return out


def run_sobol(n_samples=None, seed=42, include_zero_params=True):
    """Run the Sobol sensitivity analysis and return (Si dict, param_values, Y).

    Sampling uses the Saltelli scheme; ``n_samples`` is the base size N such that
    the total number of model evaluations is ``N * (2k + 2)`` (SALib convention).
    """
    problem = build_problem(include_zero_params)
    k = problem["num_vars"]
    if n_samples is None:
        # A pragmatic default: N ~ 512 yields a stable estimate while remaining
        # essentially instantaneous for the closed-form R0.
        n_samples = 512
    param_values = sobol_sample.sample(problem, n_samples, seed=seed)

    Y = evaluate_R0(param_values)
    Si = sobol_analyze.analyze(problem, Y, print_to_console=False)
    Si["names"] = problem["names"]
    return Si, param_values, Y


def indices_dataframe(Si):
    """Collapse a SALib results dict into a tidy pandas DataFrame for export."""
    names = Si["names"]
    rows = []
    for i, n in enumerate(names):
        rows.append({
            "parameter": n,
            "group": _group_of(n),
            "baseline": BASELINE.get(n, np.nan),
            "lower": BOUNDS[n][0],
            "upper": BOUNDS[n][1],
            "S1": Si["S1"][i],
            "S1_conf": Si["S1_conf"][i],
            "ST": Si["ST"][i],
            "ST_conf": Si["ST_conf"][i],
            "enters_R0": n not in _ZERO,
        })
    return pd.DataFrame(rows)


def _group_of(name):
    for g, members in GROUPS.items():
        if name in members:
            return g
    return "other"
