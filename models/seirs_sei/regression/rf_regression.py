"""Stage 2: Random Forest regression of the weekly free b1 and b2 against environmental variables.

Trains two independent RandomForestRegressors (one for b1_eff, one for b2_eff)
-- i.e. the weekly free per-bite transmission probabilities recovered from the
weekly free-beta fit -- using leave-one-year-out cross-validation. b2 is
right-skewed so it is fit on the log1p scale (TransformedTargetRegressor) and
back-transformed via expm1 so predictions stay in the [0,1] b2 scale. Saves
trained models, feature importances, prediction plots, and cross-validation
metrics.

Requires scikit-learn (available in the 'ml' conda environment).
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut, cross_validate
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import clone

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_SCRIPT_DIR, "data")
_RESULTS_DIR = os.path.join(_SCRIPT_DIR, "results")
_FIG_DIR = os.path.join(_RESULTS_DIR, "figures")
os.makedirs(_RESULTS_DIR, exist_ok=True)
os.makedirs(_FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature / target definitions
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "temp_mean", "precip_mean", "umid_min_mean",
    "defor_clearcut_4wk", "defor_degradation_4wk",
    "defor_burnscar_4wk", "fire_counts_4wk",
    "sin_week", "cos_week",
] + [f"year_{y}" for y in range(2017, 2024)]

TARGET_B1 = "b1_eff"
TARGET_B2 = "b2_eff"


def load_data():
    """Load the valid weekly regression dataset."""
    path = os.path.join(_DATA_DIR, "weekly_regression_valid.csv")
    df = pd.read_csv(path, parse_dates=["start_date"])
    # Drop any rows with NaN in features or targets
    cols_needed = FEATURE_COLS + [TARGET_B1, TARGET_B2]
    df = df.dropna(subset=cols_needed).copy()
    return df


def get_groups(df):
    """Year labels for leave-one-year-out CV."""
    return df["year"].values


def leave_one_year_out_cv(X, y, groups, model, target_name):
    """Run LOYO CV and return per-fold and aggregate metrics."""
    logo = LeaveOneGroupOut()
    y_true_all = []
    y_pred_all = []
    fold_years = []
    fold_r2 = []
    fold_mae = []

    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        year = groups[test_idx][0]

        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        y_pred = model_clone.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        fold_years.append(year)
        fold_r2.append(r2)
        fold_mae.append(mae)

        print(f"  Year {year}: R²={r2:.4f}, MAE={mae:.6f}")

    overall_r2 = r2_score(y_true_all, y_pred_all)
    overall_mae = mean_absolute_error(y_true_all, y_pred_all)

    return {
        "fold_years": fold_years,
        "fold_r2": fold_r2,
        "fold_mae": fold_mae,
        "mean_r2": np.mean(fold_r2),
        "std_r2": np.std(fold_r2),
        "mean_mae": np.mean(fold_mae),
        "std_mae": np.std(fold_mae),
        "overall_r2": overall_r2,
        "overall_mae": overall_mae,
        "y_true": np.array(y_true_all),
        "y_pred": np.array(y_pred_all),
    }


def train_final_model(X, y, model):
    """Train on all data and return the fitted model."""
    model.fit(X, y)
    return model


def plot_feature_importance(model, feature_names, target_name, importance_type="mdi"):
    """Bar chart of feature importances."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # TransformedTargetRegressor wraps the RF in .regressor_
    forest = getattr(model, "regressor_", model)

    if importance_type == "mdi":
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
        sorted_idx = np.argsort(importances)
        ax.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx],
                xerr=std[sorted_idx], color="#4C72B0", edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Mean Decrease in Impurity")
        ax.set_title(f"RF Feature Importance — {target_name}")
    else:
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        sorted_idx = result.importances_mean.argsort()
        ax.barh(np.array(feature_names)[sorted_idx], result.importances_mean[sorted_idx],
                xerr=result.importances_std[sorted_idx], color="#4C72B0", edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Mean Accuracy Decrease")
        ax.set_title(f"Permutation Importance — {target_name}")

    plt.tight_layout()
    path = os.path.join(_FIG_DIR, f"rf_feature_importance_{target_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_predicted_vs_actual(cv_results, target_name):
    """Scatter plot of CV predicted vs actual values."""
    fig, ax = plt.subplots(figsize=(6, 6))
    y_true = cv_results["y_true"]
    y_pred = cv_results["y_pred"]

    ax.scatter(y_true, y_pred, alpha=0.5, s=30, edgecolors="black", linewidth=0.3, color="#4C72B0")

    # Perfect prediction line
    lims = [min(y_true.min(), y_pred.min()) * 0.9,
            max(y_true.max(), y_pred.max()) * 1.1]
    ax.plot(lims, lims, "--", color="red", linewidth=1, label="Perfect prediction")

    ax.set_xlabel(f"Actual {target_name}")
    ax.set_ylabel(f"Predicted {target_name}")
    ax.set_title(f"RF CV Predicted vs Actual — {target_name}\n"
                 f"R²={cv_results['overall_r2']:.4f}, MAE={cv_results['overall_mae']:.6f}")
    ax.legend()
    ax.set_aspect("equal")
    plt.tight_layout()
    path = os.path.join(_FIG_DIR, f"rf_predicted_vs_actual_{target_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_cv_by_year(cv_results, target_name):
    """Bar chart of per-fold R² and MAE by year."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    years = cv_results["fold_years"]
    r2s = cv_results["fold_r2"]
    maes = cv_results["fold_mae"]

    colors = ["#4C72B0" if r >= 0 else "#C44E52" for r in r2s]
    ax1.bar([str(y) for y in years], r2s, color=colors, edgecolor="black", linewidth=0.5)
    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_ylabel("R²")
    ax1.set_title(f"Per-Year R² — {target_name}")

    ax2.bar([str(y) for y in years], maes, color="#55A868", edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("MAE")
    ax2.set_title(f"Per-Year MAE — {target_name}")

    plt.tight_layout()
    path = os.path.join(_FIG_DIR, f"rf_cv_by_year_{target_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_partial_dependence(model, X, feature_names, target_name):
    """Partial dependence for the top 4 continuous features."""
    from sklearn.inspection import PartialDependenceDisplay

    continuous_features = ["temp_mean", "precip_mean", "umid_min_mean", "fire_counts_4wk"]
    feature_indices = [feature_names.index(f) for f in continuous_features if f in feature_names]

    if not feature_indices:
        return

    fig, axes = plt.subplots(1, len(feature_indices), figsize=(4 * len(feature_indices), 4))
    if len(feature_indices) == 1:
        axes = [axes]

    for ax, feat_idx in zip(axes, feature_indices):
        feat_name = feature_names[feat_idx]
        # Compute partial dependence manually
        X_grid = X.copy()
        feat_values = np.linspace(X[:, feat_idx].min(), X[:, feat_idx].max(), 50)
        pdp_values = []
        for val in feat_values:
            X_grid[:, feat_idx] = val
            pdp_values.append(model.predict(X_grid).mean())
        ax.plot(feat_values, pdp_values, color="#4C72B0", linewidth=2)
        ax.set_xlabel(feat_name)
        ax.set_ylabel(f"E[{target_name}]")
        ax.set_title(f"Partial Dependence: {feat_name}")

    plt.suptitle(f"Partial Dependence — {target_name}", y=1.02)
    plt.tight_layout()
    path = os.path.join(_FIG_DIR, f"rf_partial_dependence_{target_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading data...")
    df = load_data()
    print(f"  {len(df)} valid weeks across years {df['year'].min()}-{df['year'].max()}")

    X = df[FEATURE_COLS].values
    groups = get_groups(df)

    feature_names = FEATURE_COLS

    # Baseline RF parameters
    # n_jobs=1 (no parallel workers) to avoid the noisy
    # sklearn.utils.parallel.delayed/Parallel UserWarning seen on this machine;
    # the 357-row dataset is small enough that single-core training is fast.
    base_params = {
        "n_estimators": 600,
        "max_depth": 12,
        "min_samples_leaf": 4,
        "max_features": "sqrt",
        "random_state": 42,
        "n_jobs": 1,
    }

    # b2 is right-skewed (mean ~0.04, max ~0.30) and spends much of the year
    # near zero. Predicting in log1p space and back-transforming via expm1
    # improves the fit for the dominant low-value weeks without changing the
    # [0,1] support of the resulting b2 predictions. b1 is roughly symmetric
    # (~0.42) so it is modelled on the raw scale.
    all_metrics = {}

    for target, target_name in [(TARGET_B1, "b1_eff"), (TARGET_B2, "b2_eff")]:
        print(f"\n{'='*60}")
        print(f"Training RF for {target_name}")
        print(f"{'='*60}")

        y = df[target].values

        base = RandomForestRegressor(**base_params)
        if target == TARGET_B2:
            model = TransformedTargetRegressor(
                regressor=base, func=np.log1p, inverse_func=np.expm1)
        else:
            model = base

        print("\nLeave-One-Year-Out CV:")
        cv_results = leave_one_year_out_cv(X, y, groups, model, target_name)
        print(f"\n  Aggregate: R²={cv_results['overall_r2']:.4f}, MAE={cv_results['overall_mae']:.6f}")
        print(f"  Fold mean R²={cv_results['mean_r2']:.4f} ± {cv_results['std_r2']:.4f}")
        print(f"  Fold mean MAE={cv_results['mean_mae']:.6f} ± {cv_results['std_mae']:.6f}")

        # Train final model on all data
        print("\nTraining final model on all data...")
        final_base = RandomForestRegressor(**base_params)
        if target == TARGET_B2:
            final_model = TransformedTargetRegressor(
                regressor=final_base, func=np.log1p, inverse_func=np.expm1)
        else:
            final_model = final_base
        final_model = train_final_model(X, y, final_model)

        # Save model
        model_path = os.path.join(_RESULTS_DIR, f"{target_name}_rf_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(final_model, f)
        print(f"  Saved model: {model_path}")

        # Plots
        print("\nGenerating plots...")
        plot_feature_importance(final_model, feature_names, target_name)
        plot_predicted_vs_actual(cv_results, target_name)
        plot_cv_by_year(cv_results, target_name)
        plot_partial_dependence(final_model, X, feature_names, target_name)

        # Save metrics
        all_metrics[target_name] = {
            "fold_years": [int(x) for x in cv_results["fold_years"]],
            "fold_r2": [float(x) for x in cv_results["fold_r2"]],
            "fold_mae": [float(x) for x in cv_results["fold_mae"]],
            "mean_r2": float(cv_results["mean_r2"]),
            "std_r2": float(cv_results["std_r2"]),
            "mean_mae": float(cv_results["mean_mae"]),
            "std_mae": float(cv_results["std_mae"]),
            "overall_r2": float(cv_results["overall_r2"]),
            "overall_mae": float(cv_results["overall_mae"]),
        }

    # Save all metrics
    metrics_path = os.path.join(_RESULTS_DIR, "rf_cv_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved metrics: {metrics_path}")
    print("Done.")


if __name__ == "__main__":
    main()
