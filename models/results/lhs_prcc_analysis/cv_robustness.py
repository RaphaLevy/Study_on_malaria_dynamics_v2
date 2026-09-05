"""Cross-validation robustness check for b1 / b2: the role of year-to-year level.

Motivation
----------
The week-level regressions predict the effective per-bite transmission
probabilities ``b1_eff`` and ``b2_eff`` from environmental covariates. Two
quite different R^2 figures have been reported:

* **Within-year (in-sample) R^2** -- a separate random forest per year. This is
  comparatively high, but each year's model is free to fit *its own* level,
  because it is trained and evaluated on the same year.
* **Leave-one-year-out CV R^2** (LOYO) -- a model trained on all other years
  and evaluated on a held-out year. This is much lower, because, among other
  things, the *level* of ``b1``/``b2`` shifts from year to year. A model
  trained on other years cannot reproduce the held-out year's offset, even if
  it captures the within-year environmental shape.

Robustness check implemented here
---------------------------------
We apply the user-proposed diagnostic: **standardize the target within each
year** (subtract the year mean, divide by the year standard deviation) before
training / testing, and re-measure both the within-year and the LOYO-CV R^2 in
that standardized space. If removing the per-year level makes the LOYO CV R^2
rise toward the within-year R^2, the year-to-year offset is a concrete,
technical explanation for the cross-year generalization gap. If, instead, the
gap persists, the shortfall reflects within-year (shaped) misfit rather than
intercept drift -- an honest, more defensible result.

Caveat (stated in the notebook and README)
------------------------------------------
The within-year-standardized analysis is a *diagnostic that quantifies the role
of the level effect*. In a genuine LOYO evaluation a held-out year's mean/std
cannot be computed from the held-out data without target leakage, so this is
not a deployable predictor but a controlled experiment that holds the year
mean/std fixed while leaving everything else unchanged.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import clone

_HERE = os.path.dirname(os.path.abspath(__file__))
REGRESSION_DIR = os.path.join(_HERE, "../../seirs_sei/regression")
DATA_FILE = os.path.join(REGRESSION_DIR, "data/weekly_regression_valid.csv")

FEATURE_COLS = [
    "temp_mean", "precip_mean", "umid_min_mean",
    "defor_clearcut_4wk", "defor_degradation_4wk",
    "defor_burnscar_4wk", "fire_counts_4wk",
    "sin_week", "cos_week",
] + [f"year_{y}" for y in range(2017, 2024)]

TARGET_B1 = "b1_eff"
TARGET_B2 = "b2_eff"

BASE_PARAMS = {
    "n_estimators": 600,
    "max_depth": 12,
    "min_samples_leaf": 4,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": 1,
}


def load_data():
    df = pd.read_csv(DATA_FILE, parse_dates=["start_date"])
    cols_needed = FEATURE_COLS + [TARGET_B1, TARGET_B2, "year"]
    return df.dropna(subset=cols_needed).copy()


def make_model(target, transformed):
    """RF regressor.

    ``inc_target_transform`` is True for the raw b2 target (right-skewed ->
    fitted in log1p space) and False when the target has already been pre-scaled
    (within-year standardized), where double-transforming would break the
    log1p/expm1 round-trip on negative values.
    """
    base = RandomForestRegressor(**BASE_PARAMS)
    if target == TARGET_B2 and not transformed:
        return TransformedTargetRegressor(regressor=base, func=np.log1p,
                                          inverse_func=np.expm1)
    return base


def within_year_zscore(y, groups):
    """Standardize y within each year (z = (y - mean)/std)."""
    y = np.asarray(y, dtype=float)
    z = np.empty_like(y)
    for g in np.unique(groups):
        m = groups == g
        mu = y[m].mean()
        sd = y[m].std(ddof=0)
        z[m] = (y[m] - mu) / sd if sd > 0 else 0.0
    return z


def within_year_cv_R2(X, y, groups, target, standardized):
    """Per-year in-sample R^2 (separate model trained+tested within each year)."""
    years = sorted(np.unique(groups))
    per_year = []
    for year in years:
        m = groups == year
        Xy, yy = X[m], y[m]
        if len(Xy) < 5:
            continue
        model = make_model(target, transformed=standardized)
        model.fit(Xy, yy)
        pred = model.predict(Xy)
        per_year.append({"year": int(year), "n": int(len(yy)),
                         "r2": float(r2_score(yy, pred)),
                         "mae": float(mean_absolute_error(yy, pred))})
    d = pd.DataFrame(per_year)
    return {"mode": "within_year", "standardized": bool(standardized),
            "mean_r2": float(d["r2"].mean()),
            "sd_r2": float(d["r2"].std(ddof=0)),
            "per_year": d}


def loyo_cv_R2(X, y, groups, target, transformed):
    """Leave-one-year-out CV (per-fold + pooled metrics)."""
    logo = LeaveOneGroupOut()
    years, fold_r2 = [], []
    yt, yp = [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        model = make_model(target, transformed=transformed)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        years.append(int(groups[test_idx][0]))
        fold_r2.append(r2_score(y[test_idx], pred))
        yt.extend(y[test_idx]); yp.extend(pred)
    yt, yp = np.array(yt), np.array(yp)
    return {"mode": "loyo", "fold_years": years, "fold_r2": fold_r2,
            "mean_r2": float(np.mean(fold_r2)),
            "overall_r2": float(r2_score(yt, yp)),
            "y_true": yt, "y_pred": yp}


def run_robustness(target="b1_eff", verbose=True):
    """Run the full robustness check for one target."""
    df = load_data()
    groups = df["year"].values
    X = df[FEATURE_COLS].values
    y = df[target].values.astype(float)
    y_std = within_year_zscore(y, groups)

    results = {
        "within_year_raw": within_year_cv_R2(X, y, groups, target, False),
        "within_year_std": within_year_cv_R2(X, y_std, groups, target, True),
        "loyo_raw": loyo_cv_R2(X, y, groups, target, transformed=False),
        "loyo_std": loyo_cv_R2(X, y_std, groups, target, transformed=True),
    }

    wy_raw = results["within_year_raw"]["mean_r2"]
    wy_std = results["within_year_std"]["mean_r2"]
    loyo_raw = results["loyo_raw"]["overall_r2"]
    loyo_std = results["loyo_std"]["overall_r2"]

    results["summary"] = {
        "target": target, "n_weeks": len(df),
        "within_year_r2_raw": wy_raw, "within_year_r2_std": wy_std,
        "loyo_overall_r2_raw": loyo_raw, "loyo_overall_r2_std": loyo_std,
        "loyo_mean_r2_raw": results["loyo_raw"]["mean_r2"],
        "loyo_mean_r2_std": results["loyo_std"]["mean_r2"],
        "gap_raw": wy_raw - loyo_raw, "gap_std": wy_std - loyo_std,
        "gap_narrowed": (wy_raw - loyo_raw) - (wy_std - loyo_std),
    }
    if verbose:
        s = results["summary"]
        print(f"[{target}] within-year R2: raw={s['within_year_r2_raw']:.3f} "
              f"std={s['within_year_r2_std']:.3f}")
        print(f"[{target}] LOYO R2: raw={s['loyo_overall_r2_raw']:.3f} "
              f"std={s['loyo_overall_r2_std']:.3f}")
        print(f"[{target}] gap {s['gap_raw']:.3f} -> {s['gap_std']:.3f} "
              f"(narrows by {s['gap_narrowed']:+.3f})")
    return results


def main(targets=(TARGET_B1, TARGET_B2)):
    for t in targets:
        run_robustness(t, verbose=True)


if __name__ == "__main__":
    main()
