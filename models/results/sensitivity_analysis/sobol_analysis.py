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
#
# One group == one mechanism of the model, i.e. one of the elementary response
# functions or composite rates of Section "Mosquito population dynamics" and of
# the closed form of R0. Grouping by mechanism (rather than by "what kind of
# quantity it is") is what makes a group interpretable: no group mixes, say, a
# biting-rate threshold with a parasite development optimum.
GROUPS = {
    # Per-bite transmission probabilities (the force of infection coefficients)
    # and the carrying-capacity scale M' of K(t).
    "transmission": ["b1", "b2", "M_prime"],
    # Human latent period tau_H, recovery/immunity-acquisition rate gamma and
    # immunity waning omega. Only tau_H and omega are absent from R0: gamma
    # DOES enter the closed form, in the denominator of R0^H = a b2 / gamma.
    "human_immune": ["tau_H", "gamma", "omega"],
    # Humidity-dependent daily survival p(H) = (phi e^{-k(H-H0)} + 1) /
    # (e^{-k(H-H0)} + 1), which multiplies p(T) inside mu(T, H) = -log p(T, H)
    "humidity_survival": ["H0", "k", "phi"],
    # biting rate a(T) = max(0, (T - T') / D1)
    "biting_rate": ["T_prime", "D1"],
    # extrinsic incubation period tau_M(T) = DD / (T - Tmin) and the Briere
    # development rate b3M(T) (Tmax_plasm, Topt_plasm)
    "parasite_development": ["DD", "Tmin", "Tmax_plasm", "Topt_plasm"],
    # recruitment rate b_bar(T, R) and the rainfall response f_R(R): R_L sets
    # the rainfall optimum of the immature-stage survival factors, B_E is the
    # fecundity prefactor, and c1/c2 give the larval development time tau_L(T)
    # and the larval temperature-dependent survival p_LT(T)
    "vector_recruitment": ["R_L", "B_E", "c1", "c2"],
    # adult survival curve p_T(T) = exp(-1 / (A T^2 + B T + C)) and hence mu(T, H)
    "vector_survival": ["A", "B", "C"],
    # thermal suitability f_T(T) entering the carrying capacity K
    "anop_suitability": ["Tmin_anop", "Topt_anop", "Tmax_anop"],
}

# Short legend text for each group. The colours are per-module (each figure set
# keeps its own palette) but the taxonomy and the wording live here only.
GROUP_LABELS = {
    "transmission": "Transmission / capacity",
    "human_immune": "Human immunity",
    "humidity_survival": "Humidity-dependent survival",
    "biting_rate": "Biting rate",
    "parasite_development": "Parasite development",
    "vector_recruitment": "Vector recruitment",
    "vector_survival": "Adult vector survival",
    "anop_suitability": "Anopheles thermal suitability",
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
    # --- biting rate ---
    # NOTE: T' and Tmin_anop both equal 16 degC in the baseline, but they are
    # distinct parameters of distinct functions (a(T) vs f_T(T)) and are sampled
    # independently here. Merging them into one range would remove a real degree
    # of freedom from the box.
    "T_prime":   (14.0, 20.0),      # biting-rate threshold T' (deg C)
    "D1":        (18.0, 32.0),      # biting-rate thermal scale (deg C day)
    # --- recruitment / rainfall ---
    "R_L":       (20.0, 50.0),      # rainfall optimum for habitat (mm)
    # --- Plasmodium development (extrinsic incubation period) ---
    "DD":        (80.0, 140.0),     # sporogonic degree-days (deg C day)
    "Tmin":      (12.0, 17.0),      # Plasmodium min development temp (deg C)
    "Tmax_plasm": (27.0, 32.0),     # Plasmodium max development temp (deg C)
    "Topt_plasm": (21.0, 26.0),     # Plasmodium optimal dev temp (deg C)
    # --- adult vector survival (mortality) ---
    "A":         (-0.05, -0.01),    # mortality quadratic coeff.
    "B":         (1.0, 1.6),        # mortality linear coeff.
    "C":         (-5.5, -3.5),      # mortality intercept
    # --- larval development and fecundity (recruitment) ---
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
        names = [n for n in BOUNDS if n not in _ZERO]
    bounds = [BOUNDS[n] for n in names]
    return {"num_vars": len(names), "names": names, "bounds": bounds}


# tau_H and omega do not enter the closed form of R0, so they carry a fixed zero
# sensitivity there. They are still sampled by the ODE analyses (where they do
# have an effect), so this split is about the R0 problem only.
#
# NOTE: this is spelled out explicitly and NOT derived from GROUPS. gamma also
# lives in GROUPS["human_immune"] (it is the immunity-acquisition rate) but it
# DOES enter R0, through the denominator gamma of R0^H = a b2 / gamma, so tying
# this set to the group would wrongly zero out gamma.
_ZERO = {"tau_H", "omega"}
_ACTIVE = {n for members in GROUPS.values() for n in members} - _ZERO

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
