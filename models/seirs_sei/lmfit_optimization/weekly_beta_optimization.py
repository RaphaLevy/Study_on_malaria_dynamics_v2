"""Weekly free-force-of-infection (beta) calibration for 2017.

Diagnostic version of the lmfit calibration: instead of assuming the FoI is
climate-driven (`foi_h = a(T)*b2`, `foi_m = a(T)*b1`), the effective
transmission coefficients for human infection (beta_h) and mosquito infection
(beta_m) are free parameters estimated per week from the case data with lmfit.

All other parameters are fixed at the refined 2017 values saved in
`lmfit_results.json` (see `load_refined_2017`). The fitted per-week beta curves
can then be compared against the climate-driven beta of the current method
(`a(T)*b1`, `a(T)*b2`) to locate where the climate formulation deviates.

Reuses the ODE/data-loading from `lmfit_optimization_shared` (its
`simulate_year` accepts weekly beta arrays).
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import lmfit

from lmfit_optimization_shared import (
    climate_data,
    rural_cases_df,
    DEFAULT_PARAMS,
    DEFAULT_WEIGHT_KW,
    simulate_year,
    weighted_mse_for_year,
    observed_for_year,
    build_initial_state_2017,
    initial_state_2017,
    a,
    DATA_DIR,
    RESULTS_FILE as SHARED_RESULTS_FILE,
)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEEKLY_RESULTS_FILE = os.path.join(_SCRIPT_DIR, "weekly_beta_results.json")

# Upper bound for a free per-week transmission coefficient. The climate-driven
# effective beta peaks around a(T)*b ~ 0.5*0.09 ~ 0.05, so 3.0 gives ample
# headroom for a freely-estimated beta without destabilizing the ODE.
BETA_MAX = 3.0
BETA_MIN = 1e-9


# ---------------------------------------------------------------------------
# Per-year grid helpers
# ---------------------------------------------------------------------------
def days_in_year(year):
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    return len(
        climate_data[(climate_data["date"] >= start) & (climate_data["date"] <= end)]
    )


def n_weeks_in_year(year):
    return int(np.ceil(days_in_year(year) / 7))


def climate_beta_daily(year, b1, b2):
    """Daily effective climate-driven beta: a(T)*b2 (human), a(T)*b1 (mosquito).

    Uses the same raw `temp_med` that the ODE feeds into the biting rate a(T)."""
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    clim = climate_data[
        (climate_data["date"] >= start) & (climate_data["date"] <= end)
    ].reset_index(drop=True)
    T = clim["temp_med"].values
    bite = np.array([a(t) for t in T])
    return bite * b2, bite * b1


def climate_beta_weekly(year, b1, b2, n_weeks=None):
    """Weekly-mean climate-driven beta (seeding/plotting reference)."""
    bh, bm = climate_beta_daily(year, b1, b2)
    if n_weeks is None:
        n_weeks = n_weeks_in_year(year)
    w_idx = np.minimum(np.arange(len(bh)) // 7, n_weeks - 1)
    bh_w = np.array([bh[w_idx == w].mean() for w in range(n_weeks)])
    bm_w = np.array([bm[w_idx == w].mean() for w in range(n_weeks)])
    return bh_w, bm_w


def week_index_edges(year):
    """Start/end day-of-year (0-based, inclusive) for each weekly bucket."""
    n_days = days_in_year(year)
    n_weeks = n_weeks_in_year(year)
    edges = []
    for w in range(n_weeks):
        lo = w * 7
        hi = min(lo + 7, n_days) - 1
        edges.append((lo, hi))
    return edges


# ---------------------------------------------------------------------------
# Weights (matches compute_weighted_mse)
# ---------------------------------------------------------------------------
def weights_for_obs(obs, weight_kw=None):
    wkw = DEFAULT_WEIGHT_KW if weight_kw is None else weight_kw
    days_from_start = (obs["date"] - obs["date"].iloc[0]).dt.days.values.astype(float)
    decay = wkw["weight_decay"]
    if decay == "exponential":
        return wkw["initial_weight"] * np.exp(-wkw["decay_rate"] * days_from_start)
    elif decay == "linear":
        return wkw["initial_weight"] / (1 + wkw["decay_rate"] * days_from_start)
    elif decay == "step":
        return np.where(
            days_from_start <= wkw["step_threshold"],
            wkw["initial_weight"],
            wkw["step_later_weight"],
        )
    elif decay == "inverse":
        return wkw["initial_weight"] * (
            1.0 / (1.0 + days_from_start / wkw["inverse_scale"])
        )
    return np.ones_like(days_from_start)


# ---------------------------------------------------------------------------
# Fixed 2017 parameters (from the committed refined lmfit results)
# ---------------------------------------------------------------------------
def load_refined_2017(results_file=None):
    """Load the refined 2017 parameters from lmfit_results.json.

    Returns (fixed_params, state0, refined_ic, b1, b2). `fixed_params` holds
    H0,k,phi,tau_H,gamma,omega,M_prime (b1/b2 are replaced by the weekly beta);
    `state0` is built from the refined IC fractions."""
    results_file = SHARED_RESULTS_FILE if results_file is None else results_file
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            data = json.load(f)
        per_year = data.get("per_year", {}).get("2017", {})
        p = per_year.get("params")
        if p:
            fixed = {
                k: float(p[k])
                for k in ("H0", "k", "phi", "tau_H", "gamma", "omega", "M_prime")
            }
            b1, b2 = float(p["b1"]), float(p["b2"])
            fracs = {k: float(p[k]) for k in ("S_H0_frac", "E_H0_frac", "I_M0_ratio")}
            state0 = build_initial_state_2017(
                fracs["S_H0_frac"], fracs["E_H0_frac"], fracs["I_M0_ratio"]
            )
            if state0 is None:
                state0 = initial_state_2017.copy()
            return fixed, state0, fracs, b1, b2
    fixed = {
        k: float(DEFAULT_PARAMS[k])
        for k in ("H0", "k", "phi", "tau_H", "gamma", "omega")
    }
    fixed["M_prime"] = float(DEFAULT_PARAMS["M_prime"])
    return (
        fixed,
        initial_state_2017.copy(),
        None,
        float(DEFAULT_PARAMS["b1"]),
        float(DEFAULT_PARAMS["b2"]),
    )


# ---------------------------------------------------------------------------
# Objective (weighted residuals, for lmfit least_squares)
# ---------------------------------------------------------------------------
def _weekly_objective(
    params, year, state0, fixed, obs, obs_vals, obs_dates_int, weights, n_weeks
):
    beta_h = np.array([params[f"beta_h_{i}"].value for i in range(n_weeks)])
    beta_m = np.array([params[f"beta_m_{i}"].value for i in range(n_weeks)])
    res = simulate_year(year, state0, fixed, beta_h, beta_m)
    if res is None:
        return 1e9 * np.ones(len(obs_vals))
    dates, IH, _ = res
    if len(IH) == 0 or not np.all(np.isfinite(IH)):
        return 1e9 * np.ones(len(obs_vals))
    model_dates_int = pd.to_datetime(dates).astype(np.int64).values
    model_at_obs = interp1d(
        model_dates_int, IH, kind="linear", fill_value="extrapolate"
    )(obs_dates_int)
    return np.sqrt(weights) * (model_at_obs - obs_vals)


# ---------------------------------------------------------------------------
# lmfit least-squares fit of the per-week betas
# ---------------------------------------------------------------------------
def fit_weekly_betas(
    year,
    fixed,
    state0,
    seed_beta_h,
    seed_beta_m,
    weight_kw=None,
    max_nfev=1200,
    ftol=1e-8,
):
    """Fit a free per-week beta_h and beta_m (all other params fixed).

    Seeded from the climate-driven weekly beta (a(T)*b2, a(T)*b1). Uses lmfit's
    least_squares (Levenberg-Marquardt / trust-region) which is well suited to a
    medium number of parameters with a 365-point residual vector.

    Returns (result, beta_h_fit, beta_m_fit, wmse_fit, end_state)."""
    wkw = DEFAULT_WEIGHT_KW if weight_kw is None else weight_kw
    n_weeks = len(seed_beta_h)
    params = lmfit.Parameters()
    for i in range(n_weeks):
        params.add(
            f"beta_h_{i}",
            value=float(seed_beta_h[i]),
            min=BETA_MIN,
            max=BETA_MAX,
            vary=True,
        )
        params.add(
            f"beta_m_{i}",
            value=float(seed_beta_m[i]),
            min=BETA_MIN,
            max=BETA_MAX,
            vary=True,
        )

    obs = observed_for_year(year)
    obs_vals = obs["active_total"].values
    obs_dates_int = obs["date"].astype(np.int64).values
    weights = weights_for_obs(obs, wkw)

    def objective(params):
        return _weekly_objective(
            params, year, state0, fixed, obs, obs_vals, obs_dates_int, weights, n_weeks
        )

    result = lmfit.minimize(
        objective, params, method="least_squares", max_nfev=max_nfev, ftol=ftol
    )

    beta_h_fit = np.array([result.params[f"beta_h_{i}"].value for i in range(n_weeks)])
    beta_m_fit = np.array([result.params[f"beta_m_{i}"].value for i in range(n_weeks)])
    res = simulate_year(year, state0, fixed, beta_h_fit, beta_m_fit)
    if res is None:
        return result, beta_h_fit, beta_m_fit, None, None
    dates, IH, end_state = res
    wmse_fit = weighted_mse_for_year(dates, IH, obs, wkw)
    return result, beta_h_fit, beta_m_fit, wmse_fit, end_state


def weekly_beta_table(year, beta_h_fit, beta_m_fit, b1, b2, n_weeks=None):
    """DataFrame comparing fitted vs climate-driven weekly beta."""
    if n_weeks is None:
        n_weeks = n_weeks_in_year(year)
    bh_c, bm_c = climate_beta_weekly(year, b1, b2, n_weeks)
    edges = week_index_edges(year)
    start_days = np.array([e[0] for e in edges])
    return pd.DataFrame(
        {
            "week": np.arange(n_weeks),
            "start_day": start_days,
            "start_date": pd.to_datetime(f"{year}-01-01")
            + pd.to_timedelta(start_days, "D"),
            "beta_h_climate": bh_c,
            "beta_h_fitted": beta_h_fit,
            "beta_m_climate": bm_c,
            "beta_m_fitted": beta_m_fit,
        }
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_weekly_beta_fit(
    year=2017,
    weight_kw=None,
    max_nfev=1200,
    ftol=1e-8,
    save_json=True,
    results_file=None,
    verbose=True,
):
    """Fit free per-week betas for one year and save `weekly_beta_results.json`.

    Reports the baseline wMSE (climate-driven beta with the refined 2017
    parameters) and the wMSE after freeing the weekly betas, plus the fitted
    and climate weekly beta curves."""
    results_file = WEEKLY_RESULTS_FILE if results_file is None else results_file
    wkw = DEFAULT_WEIGHT_KW if weight_kw is None else weight_kw

    fixed, state0, fracs, b1, b2 = load_refined_2017()
    n_weeks = n_weeks_in_year(year)
    seed_h, seed_m = climate_beta_weekly(year, b1, b2, n_weeks)

    obs = observed_for_year(year)
    full = dict(fixed)
    full["b1"], full["b2"] = b1, b2
    res0 = simulate_year(year, state0, full)
    if res0 is None:
        raise RuntimeError("Baseline (climate-driven) simulation failed")
    dates0, IH0, _ = res0
    wmse0 = weighted_mse_for_year(dates0, IH0, obs, wkw)

    if verbose:
        print(f"Baseline wMSE (climate-driven beta, refined 2017 params): {wmse0:,.1f}")
        print(
            f"Fitting {2 * n_weeks} free weekly betas (beta_h, beta_m) seeded from climate..."
        )

    result, beta_h_fit, beta_m_fit, wmse1, end_state = fit_weekly_betas(
        year,
        fixed,
        state0,
        seed_h,
        seed_m,
        weight_kw=wkw,
        max_nfev=max_nfev,
        ftol=ftol,
    )

    if verbose:
        print(
            f"Fit done: wMSE = {wmse1:,.1f} "
            f"(nfev={result.nfev}, success={result.success})"
        )

    payload = {
        "year": year,
        "method": "lmfit least_squares (weekly free beta)",
        "n_weeks": n_weeks,
        "beta_max": BETA_MAX,
        "weight_kw": {k: v for k, v in wkw.items()},
        "fixed_params": {k: float(v) for k, v in fixed.items()},
        "refined_ic": fracs,
        "climate_params": {"b1": float(b1), "b2": float(b2)},
        "baseline_wmse": float(wmse0),
        "weekly_fit_wmse": float(wmse1) if wmse1 is not None else None,
        "chisqr": float(result.chisqr) if result.chisqr is not None else None,
        "nfev": int(getattr(result, "nfev", -1)),
        "success": bool(getattr(result, "success", False)),
        "beta_h_weekly_fitted": [float(v) for v in beta_h_fit],
        "beta_m_weekly_fitted": [float(v) for v in beta_m_fit],
        "beta_h_weekly_climate": [float(v) for v in seed_h],
        "beta_m_weekly_climate": [float(v) for v in seed_m],
        "end_state": end_state.tolist() if end_state is not None else None,
    }

    if save_json:
        with open(results_file, "w") as f:
            json.dump(payload, f, indent=2)
        if verbose:
            print(f"Saved results to {results_file}")

    return payload


def load_weekly_results(results_file=None):
    results_file = WEEKLY_RESULTS_FILE if results_file is None else results_file
    if not os.path.exists(results_file):
        return None
    with open(results_file, "r") as f:
        return json.load(f)
