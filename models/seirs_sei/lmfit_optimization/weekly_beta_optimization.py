"""Weekly free-force-of-infection (beta) calibration for any year.

Diagnostic version of the lmfit calibration: instead of assuming the FoI is
climate-driven (`foi_h = a(T)*b2`, `foi_m = a(T)*b1`), the effective
transmission coefficients for human infection (beta_h) and mosquito infection
(beta_m) are free parameters estimated per week from the case data with lmfit.

All other parameters are fixed at the refined values saved in
`lmfit_results.json` (see `load_refined_params`). The fitted per-week beta curves
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
    IM_MIN_VIABLE,
    im_extinction_penalty,
    simulate_year,
    weighted_mse_for_year,
    observed_for_year,
    observed_IH_start,
    build_initial_state,
    get_default_initial_state,
    get_year_population,
    get_default_params,
    pop_by_year,
    a,
    DATA_DIR,
    RESULTS_FILE as SHARED_RESULTS_FILE,
)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
#WEEKLY_RESULTS_FILE = os.path.join(_SCRIPT_DIR, "weekly_beta_results.json")

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
# Fixed parameters (from the committed refined lmfit results)
# ---------------------------------------------------------------------------
def load_refined_params(year, results_file=None):
    """Load the refined parameters for any year from lmfit_results.json.

    State reconstruction order:
      1. previous year's `end_state` with I_H reset to the first observed case
         (identical to what `run_yearly_fit` uses as this year's start state);
      2. IC fractions stored in `params` (genuinely fitted for the first year
         of a chain, e.g. 2017);
      3. IC fractions under `refine`;
      4. default IC fractions.
    """
    results_file = SHARED_RESULTS_FILE if results_file is None else results_file
    
    if not os.path.exists(results_file):
        print(f"Warning: Results file {results_file} not found. Using defaults for {year}.")
        return _get_default_params_for_year(year)
    
    try:
        with open(results_file, "r") as f:
            data = json.load(f)
        
        per_year = data.get("per_year", {}).get(str(year), {})
        p = per_year.get("params")
        
        if not p:
            print(f"Warning: No params found for {year}. Using defaults.")
            return _get_default_params_for_year(year)
        
        # Extract fixed parameters (always available)
        fixed = {}
        for k in ("H0", "k", "phi", "tau_H", "gamma", "omega", "M_prime"):
            if k in p:
                fixed[k] = float(p[k])
            else:
                print(f"Warning: Missing parameter {k} for year {year}. Using default.")
                if k in DEFAULT_PARAMS:
                    fixed[k] = float(DEFAULT_PARAMS[k])
                else:
                    # Use year-specific default for M_prime
                    if k == "M_prime":
                        fixed[k] = float(get_default_params(year)["M_prime"])
        
        # Extract b1, b2
        if "b1" not in p or "b2" not in p:
            print(f"Warning: Missing b1/b2 for {year}. Using defaults.")
            b1, b2 = float(DEFAULT_PARAMS["b1"]), float(DEFAULT_PARAMS["b2"])
        else:
            b1, b2 = float(p["b1"]), float(p["b2"])
        
        # Try to get IC fractions - they may not exist for years after 2017
        fracs = None
        state0 = None

        # First try: carry the previous year's simulated end-state. This is
        # exactly the start-of-year state the yearly pipeline used for this
        # year (`run_yearly_fit` carries end_state forward and resets I_H to
        # the first observed case), so it reproduces that trajectory. Stored
        # IC fractions are deliberately NOT preferred over this: for carried
        # years they follow different scale conventions (fractions of S+E+I,
        # E_M/I_H ratios) than `build_initial_state` expects.
        prev_per_year = data.get("per_year", {}).get(str(year - 1), {})
        prev_end = prev_per_year.get("end_state")
        if prev_end is not None and len(prev_end) == 7:
            state0 = np.array(prev_end, dtype=float)
            state0[2] = float(observed_IH_start(year))
            n_year = int(round(pop_by_year[year]))
            fracs = {
                "S_H0_frac": float(state0[0] / n_year),
                "E_H0_frac": float(state0[1] / n_year),
                "I_M0_ratio": float(state0[6] / (10 * n_year)),
            }
            return fixed, state0, fracs, b1, b2

        # Second try: extract from params if available (typically the first
        # year of a chain, e.g. 2017, whose IC fractions were genuinely fitted)
        if "S_H0_frac" in p and "E_H0_frac" in p and "I_M0_ratio" in p:
            fracs = {
                "S_H0_frac": float(p["S_H0_frac"]),
                "E_H0_frac": float(p["E_H0_frac"]),
                "I_M0_ratio": float(p["I_M0_ratio"])
            }
            state0 = build_initial_state(
                year, fracs["S_H0_frac"], fracs["E_H0_frac"], fracs["I_M0_ratio"]
            )
            if state0 is not None:
                return fixed, state0, fracs, b1, b2
        
        # Third try: use refine params if available
        refine = per_year.get("refine", {})
        refine_params = refine.get("params", {})
        if "S_H0_frac" in refine_params and "E_H0_frac" in refine_params and "I_M0_ratio" in refine_params:
            fracs = {
                "S_H0_frac": float(refine_params["S_H0_frac"]),
                "E_H0_frac": float(refine_params["E_H0_frac"]),
                "I_M0_ratio": float(refine_params["I_M0_ratio"])
            }
            state0 = build_initial_state(
                year, fracs["S_H0_frac"], fracs["E_H0_frac"], fracs["I_M0_ratio"]
            )
            if state0 is not None:
                return fixed, state0, fracs, b1, b2
        
        # Fourth try: use default IC fractions
        print(f"Warning: No state info for {year}. Using default IC fractions.")
        default_ic = get_default_ic_fractions(year)
        fracs = default_ic
        state0 = build_initial_state(
            year, fracs["S_H0_frac"], fracs["E_H0_frac"], fracs["I_M0_ratio"]
        )
        
        # Final fallback
        if state0 is None:
            print(f"Warning: Failed to build state for {year}. Using get_default_initial_state.")
            state0 = get_default_initial_state(year)
            default_ic = get_default_ic_fractions(year)
            fracs = default_ic
        
        return fixed, state0, fracs, b1, b2
        
    except Exception as e:
        print(f"Warning: Error loading params for {year}: {e}. Using defaults.")
        return _get_default_params_for_year(year)


def _get_default_params_for_year(year):
    """Get default parameters for a year (fallback)."""
    fixed = {
        k: float(DEFAULT_PARAMS[k])
        for k in ("H0", "k", "phi", "tau_H", "gamma", "omega")
    }
    fixed["M_prime"] = float(get_default_params(year)["M_prime"])
    state0 = get_default_initial_state(year)
    default_ic = get_default_ic_fractions(year)
    fracs = default_ic
    return (
        fixed,
        state0,
        fracs,
        float(DEFAULT_PARAMS["b1"]),
        float(DEFAULT_PARAMS["b2"]),
    )


def get_default_ic_fractions(year):
    """Get default initial condition fractions for a year."""
    N = get_year_population(year)
    state = get_default_initial_state(year)
    # M0 is 10 * N in get_default_initial_state
    M0 = 10 * N
    return {
        "S_H0_frac": float(state[0] / N) if N > 0 else 0.687,
        "E_H0_frac": float(state[1] / N) if N > 0 else 0.05,
        "I_M0_ratio": float(state[5] / M0) if M0 > 0 else 0.02
    }


# ---------------------------------------------------------------------------
# Objective (weighted residuals, for lmfit least_squares)
# ---------------------------------------------------------------------------
def _weekly_objective(
    params, year, state0, fixed, obs, obs_vals, obs_dates_int, weights, n_weeks
):
    beta_h = np.array([params[f"beta_h_{i}"].value for i in range(n_weeks)])
    beta_m = np.array([params[f"beta_m_{i}"].value for i in range(n_weeks)])
    res = simulate_year(year, state0, fixed, beta_h, beta_m, return_diagnostics=True)
    if res is None:
        return 1e9 * np.ones(len(obs_vals))
    dates, IH, _, diag = res
    if len(IH) == 0 or not np.all(np.isfinite(IH)):
        return 1e9 * np.ones(len(obs_vals))
    model_dates_int = pd.to_datetime(dates).astype(np.int64).values
    model_at_obs = interp1d(
        model_dates_int, IH, kind="linear", fill_value="extrapolate"
    )(obs_dates_int)
    resid = np.sqrt(weights) * (model_at_obs - obs_vals)
    penalty = im_extinction_penalty(diag["im_min"])
    if penalty > 0:
        # Offset the residual vector (constant offset keeps the same length
        # while giving least_squares a gradient that pushes I_M back up).
        resid = resid + np.sqrt(penalty)
    return resid


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
    """DataFrame comparing fitted vs climate-driven weekly beta for any year."""
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
    year,
    weight_kw=None,
    max_nfev=1200,
    ftol=1e-8,
    save_json=True,
    results_file=None,
    verbose=True,
):
    """Fit free per-week betas for any year."""
    results_file = f"weekly_beta_results_{year}.json" if results_file is None else results_file
    wkw = DEFAULT_WEIGHT_KW if weight_kw is None else weight_kw

    fixed, state0, fracs, b1, b2 = load_refined_params(year)
    n_weeks = n_weeks_in_year(year)
    seed_h, seed_m = climate_beta_weekly(year, b1, b2, n_weeks)

    obs = observed_for_year(year)
    full = dict(fixed)
    full["b1"], full["b2"] = b1, b2
    res0 = simulate_year(year, state0, full, return_diagnostics=True)
    if res0 is None:
        raise RuntimeError(f"Baseline (climate-driven) simulation failed for year {year}")
    dates0, IH0, _, diag0 = res0
    wmse0 = weighted_mse_for_year(dates0, IH0, obs, wkw)

    if verbose:
        print(f"Year {year}: Baseline wMSE (climate-driven beta): {wmse0:,.1f}")
        if diag0["im_min"] < IM_MIN_VIABLE:
            print(
                f"WARNING: the fixed {year} parameters drive I_M down to "
                f"{diag0['im_min']:.4g} (< {IM_MIN_VIABLE}); with no infected "
                "mosquitoes the weekly betas are weakly identified. Recalibrate "
                "the yearly params first."
            )
        print(f"Fitting {2 * n_weeks} free weekly betas (beta_h, beta_m)...")

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
        "baseline_im_min": float(diag0["im_min"]),
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


def load_weekly_results(year, results_file=None):
    """Load weekly beta results for a specific year."""
    if results_file is None:
        results_file = f"weekly_beta_results_{year}.json"
    if not os.path.exists(results_file):
        return None
    with open(results_file, "r") as f:
        return json.load(f)