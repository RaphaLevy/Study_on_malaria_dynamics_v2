"""Analysis of the uncertainty in the calibrated b1 / b2 transmission coefficients.

The weekly free-beta calibration recovers effective per-bite transmission
probabilities per week; the yearly chain additionally reports a scalar
``b1`` and ``b2`` per year (see ``lmfit_results.json``). These scalar values
**drift from year to year** (b1 climbs toward its upper bound while b2 falls),
which is the usual signature of the scalar climate-driven transmission form
compensating for residual temporal misfit.

This module characterises that across-year spread and propagates it through the
closed-form reproduction number R0 (and, if requested, the seasonal peak of
I_H), so we can quantify how much of the *model uncertainty* in a transmission
summary is attributable to the calibration uncertainty in ``b1`` / ``b2``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model_wrappers import (
    calibrated_b1_b2_by_year,
    evaluate_r0_rows,
    seasonal_ih_peak,
    ode_fixed_params,
    ODE_NAMES,
)


def across_year_b1b2_table(years=None, with_r0=True, with_ih_peak=False):
    """Per-year calibrated params + derived R0 (and optionally I_H peak)."""
    df = calibrated_b1_b2_by_year(years)
    # R0 for each year at the reference climate using that year's params.
    if with_r0:
        from sobol_analysis import _NAME_TO_KW, _ZERO
        r0s = []
        for _, r in df.iterrows():
            active = {k: float(r[k]) for k in
                      ("H0", "k", "phi", "gamma", "b1", "b2")}
            active["M_prime"] = float(r["M_prime"])
            kwargs = {
                _NAME_TO_KW[n]: v for n, v in active.items()
                if n not in _ZERO and n != "M_prime"
            }
            from seirs_sei_r0 import r0_components, REFERENCE_CLIMATE, N_POP
            c = r0_components(REFERENCE_CLIMATE["T"], REFERENCE_CLIMATE["R"],
                              REFERENCE_CLIMATE["H"], N=N_POP,
                              M_prime=active["M_prime"], **kwargs)
            r0s.append(c["R0"])
        df["R0"] = r0s

    if with_ih_peak:
        peaks = []
        for _, r in df.iterrows():
            p = {
                "H0": float(r["H0"]), "k": float(r["k"]), "phi": float(r["phi"]),
                "tau_H": float(r["tau_H"]), "gamma": float(r["gamma"]),
                "omega": float(r["omega"]), "b1": float(r["b1"]),
                "b2": float(r["b2"]), "M_prime": float(r["M_prime"]),
            }
            peaks.append(seasonal_ih_peak(list(df["year"]), p, {}))
        df["seasonal_IH_peak"] = peaks

    return df


def uncertainty_summary(df):
    """Mean / SD / range / CV of b1, b2 and derived R0 across the calibrated years."""
    rows = {}
    for col, label in (("b1", "b1 (mosq->human)"), ("b2", "b2 (human->mosq)"),
                       ("R0", "R0"), ("seasonal_IH_peak", "I_H peak")):
        if col not in df.columns:
            continue
        v = df[col].values.astype(float)
        v = v[np.isfinite(v)]
        rows[label] = {
            "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
            "min": float(v.min()), "max": float(v.max()),
            "cv_pct": float(100.0 * v.std(ddof=1) / v.mean()) if v.mean() != 0 else np.nan,
        }
    return pd.DataFrame(rows).T


def b1b2_correlation(df):
    r_i = df["b1"].rank()
    r_j = df["b2"].rank()
    from scipy.stats import spearmanr, pearsonr
    rho_s, p_s = spearmanr(df["b1"], df["b2"])
    r_p, p_p = pearsonr(df["b1"], df["b2"])
    return {"pearson_r": r_p, "pearson_p": p_p,
            "spearman_rho": rho_s, "spearman_p": p_s}


def main():
    df = across_year_b1b2_table(with_ih_peak=True)
    print(df.to_string())
    print()
    print("Uncertainty summary:")
    print(uncertainty_summary(df).to_string())
    print()
    print("b1-b2 correlation:", b1b2_correlation(df))


if __name__ == "__main__":
    main()
