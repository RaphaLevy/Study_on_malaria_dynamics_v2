"""Generate all figures and tables for the LHS+PRCC robustness analysis.

Runs four analyses and writes outputs to ``./figures`` and ``./tables``:

A. LHS + PRCC of the closed-form reproduction number R0 (reference climate).
B. LHS + PRCC of the seasonal peak of infectious humans I_H (ODE simulation).
C. Uncertainty of the calibrated b1 / b2 (across-year variability -> R0),
   plus a joint (b1, b2) -> R0 contour and a global R0 tornado diagram.
D. Cross-validation-by-year R^2 robustness check for the b1/b2 regressions.

Importing this module triggers the ODE back-end's one-time data loading
(a few seconds). The ODE-based LHS is the slowest part (default 300 samples
x 7 years); the R0 and CV parts are fast.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_SENS = os.path.join(_HERE, "../sensitivity_analysis")
for _p in (_SENS,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sobol_analysis import BOUNDS, GROUPS

import model_wrappers as mw
from lhs_prcc import (
    lhs_sample, prcc_correlations, spearman_correlations,
    tornado_values, prcc_barh, tornado_fig, apply_paper_style,
    GROUP_COLORS, GROUP_LABELS,
)
from b1b2_uncertainty import (
    across_year_b1b2_table, uncertainty_summary, b1b2_correlation,
)
import cv_robustness as cvr

apply_paper_style()

_HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(_HERE, "figures")
TBL = os.path.join(_HERE, "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TBL, exist_ok=True)

REFERENCE = mw.REFERENCE_CLIMATE

# The active parameter set for the closed-form R0 (tau_H/omega do not enter R0).
R0_NAMES = [n for n in BOUNDS if n not in ("tau_H", "omega")]
# The full set the ODE actually uses (tau_H/omega enter the I_H dynamics).
ODE_NAMES = mw.ODE_NAMES


def eval_r0_rows(X, names):
    return mw.evaluate_r0_rows(np.asarray(X, dtype=float), names=list(names))


def eval_ih_peak_rows(X, names, years=None, verbose=False, progress_every=10):
    if years is None:
        years = list(range(2017, 2024))
    out = []
    X = np.asarray(X, dtype=float)
    for i, row in enumerate(X, 1):
        active = dict(zip(names, row))
        # split into p-dict params and global overrides
        p = {}
        globals_ = {}
        for n, v in active.items():
            binding, key = mw._ODE_BINDING[n]
            if binding == "param":
                p[key] = float(v)
            else:
                globals_[n] = float(v)
        # ensure M_prime is present
        if "M_prime" not in p:
            fixed0 = mw.ode_fixed_params(years[0])
            p["M_prime"] = fixed0["M_prime"]
        out.append(mw.seasonal_ih_peak(years, p, globals_))
        if verbose and i % progress_every == 0:
            print(f"  IH ODE rows: {i}/{len(X)}")
    return np.array(out, dtype=float)


def _prcc_table(X, Y, names, output):
    df = prcc_correlations(X, Y, names=names)
    df_spear = spearman_correlations(X, Y, names=names)
    df = df.merge(df_spear[["parameter", "spearman"]], on="parameter")
    df = df.sort_values("PRCC", key=np.abs, ascending=False)
    df.to_csv(output, index=False)
    return df


# --------------------------------------------------------------------------- #
# A. R0 LHS + PRCC
# --------------------------------------------------------------------------- #
def run_r0_prcc(n_samples=2000, seed=42):
    print("A. R0 LHS+PRCC ...")
    X = lhs_sample(R0_NAMES, n_samples=n_samples, seed=seed)
    Y = eval_r0_rows(X, R0_NAMES)
    df = _prcc_table(X, Y, R0_NAMES, os.path.join(TBL, "prcc_r0.csv"))

    # tornado (one-at-a-time), other params held at the calibrated reference
    ref = mw.reference_params(R0_NAMES)
    tornado_df, base_out = tornado_values(eval_r0_rows, R0_NAMES, baseline=ref)
    tornado_df.to_csv(os.path.join(TBL, "r0_tornado.csv"), index=False)

    # ---- PRCC chart: all parameters, significance stars, journal style ----
    fig, _ = prcc_barh(df, "PRCC of $\\mathcal{R}_0$",
                       "LHS + PRCC of the closed-form $\\mathcal{R}_0$")
    fig.savefig(os.path.join(FIG, "prcc_r0_barh.png")); plt.close(fig)

    # ---- Butterfly tornado (symlog axis: swing spans orders of magnitude) ----
    fig, _ = tornado_fig(tornado_df, base_out, use_symlog=True,
                         title="One-at-a-time tornado of $\\mathcal{R}_0$ "
                               "(calibrated reference)",
                         value_label="Change in $\\mathcal{R}_0$")
    fig.savefig(os.path.join(FIG, "r0_tornado.png")); plt.close(fig)

    return df, tornado_df, base_out, X, Y


# --------------------------------------------------------------------------- #
# B. Seasonal I_H peak LHS + PRCC (ODE)
# --------------------------------------------------------------------------- #
def run_ih_prcc(n_samples=300, seed=42):
    print(f"B. Seasonal I_H LHS+PRCC (n={n_samples}) ...")
    X = lhs_sample(ODE_NAMES, n_samples=n_samples, seed=seed)
    Y = eval_ih_peak_rows(X, ODE_NAMES)

    # The seasonal I_H peak is heavily right-skewed (extreme values when, e.g.,
    # gamma sits near its tiny lower bound). PRCC is computed on log10(peak)
    # for a stable, elasticity-style measure. The CSV retains a raw-peak table
    # via the spearman/raw columns for reference.
    Y_pos = np.where(np.isfinite(Y) & (Y > 0), Y, np.nan)
    Y_log = np.log10(Y_pos)
    df = _prcc_table(X, Y_log, ODE_NAMES, os.path.join(TBL, "prcc_ih_peak.csv"))

    n_valid = int(np.isfinite(Y).sum())
    chr_rows = []
    for m, arr in (("raw", Y), ("log10", Y_log)):
        a = arr[np.isfinite(arr)]
        chr_rows.append({
            "transform": m, "n_valid": int(len(a)),
            "mean": float(a.mean()) if len(a) else np.nan,
            "median": float(np.median(a)) if len(a) else np.nan,
            "sd": float(a.std(ddof=1)) if len(a) else np.nan,
            "min": float(a.min()) if len(a) else np.nan,
            "max": float(a.max()) if len(a) else np.nan,
        })
    pd.DataFrame(chr_rows).to_csv(os.path.join(TBL, "ih_peak_sampling.csv"), index=False)

    # ---- PRCC chart (on log10 peak): all parameters, stars, journal style ----
    fig, _ = prcc_barh(df, "PRCC of $\\log_{10}$(seasonal $I_H$ peak)",
                       "LHS + PRCC of the seasonal $I_H$ peak (ODE)")
    fig.savefig(os.path.join(FIG, "prcc_ih_peak_barh.png")); plt.close(fig)

    # ---- distribution (raw peak, log x) ----
    yv = Y[np.isfinite(Y)]
    if len(yv):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(yv, bins=50, color="#0072B2", edgecolor="black", linewidth=0.3)
        ax.axvline(np.median(yv), color="#D55E00", linestyle="--",
                   label=f"median = {np.median(yv):.0f}")
        ax.set_xscale("log")
        ax.set_xlabel("Mean seasonal peak I_H (across 2017-2023, log scale)")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of seasonal I_H peak over the LHS")
        ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "ih_peak_distribution.png"))
        plt.close(fig)

    return df, X, Y


# --------------------------------------------------------------------------- #
# C. b1 / b2 calibration uncertainty + joint (b1,b2) -> R0 contour
# --------------------------------------------------------------------------- #
def run_b1b2_uncertainty(X_r0=None, names_r0=None):
    print("C. b1/b2 calibration uncertainty ...")
    df = across_year_b1b2_table(with_ih_peak=True)
    df.to_csv(os.path.join(TBL, "calibrated_b1b2_by_year.csv"), index=False)
    summ = uncertainty_summary(df)
    summ.to_csv(os.path.join(TBL, "b1b2_uncertainty_summary.csv"))
    corr = b1b2_correlation(df)
    pd.DataFrame([corr]).to_csv(os.path.join(TBL, "b1b2_correlation.csv"), index=False)

    # ---- b1 & b2 by year (drift) ----
    x = df["year"]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(x, df["b1"], "-o", color="#0072B2", label="b1 (mosq->human)")
    ax1.axhline(BOUNDS["b1"][1], color="#0072B2", linestyle=":", alpha=0.6,
                label=f"b1 upper bound ({BOUNDS['b1'][1]:.2f})")
    ax1.set_ylabel("b1"); ax1.set_xlabel("Year")
    ax2 = ax1.twinx()
    ax2.plot(x, df["b2"], "-s", color="#D55E00", label="b2 (human->mosq)")
    ax2.set_ylabel("b2")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center left", fontsize=9, frameon=False)
    ax1.set_xticks(x)
    ax1.set_title("Calibrated b1 and b2 drift by year")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "b1b2_by_year.png")); plt.close(fig)

    # ---- R0 by year under each year's calibration; b1b2 joint contour ----
    # joint (b1, b2) -> R0 contour, other params at the calibrated reference
    base = mw.reference_params(R0_NAMES)
    b1g = np.linspace(BOUNDS["b1"][0], BOUNDS["b1"][1], 80)
    b2g = np.linspace(BOUNDS["b2"][0], BOUNDS["b2"][1], 80)
    B1, B2 = np.meshgrid(b1g, b2g)
    R0g = np.empty_like(B1)
    for i in range(len(b2g)):
        for j in range(len(b1g)):
            row = [base[n] if n not in ("b1", "b2") else
                   (B1[i, j] if n == "b1" else B2[i, j]) for n in R0_NAMES]
            R0g[i, j] = eval_r0_rows(np.array([row]), R0_NAMES)[0]

    fig, ax = plt.subplots(figsize=(7.5, 6))
    cf = ax.contourf(B1, B2, R0g, levels=30, cmap="viridis")
    cs = ax.contour(B1, B2, R0g, levels=[1, 2, 4, 8], colors="white", linewidths=0.8)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f")
    cs1 = ax.contour(B1, B2, R0g, levels=[1], colors="black", linewidths=1.2)
    ax.clabel(cs1, inline=True, fontsize=8, fmt="R0 = 1")
    ax.scatter(df["b1"], df["b2"], c="#D55E00", edgecolor="black", s=60, zorder=5,
               label="per-year calibration")
    for _, r in df.iterrows():
        ax.annotate(str(int(r["year"])), (r["b1"], r["b2"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=8)
    ax.set_xlabel("b1"); ax.set_ylabel("b2")
    ax.set_title("Joint (b1, b2) -> R0 contour with per-year calibrations")
    cb = fig.colorbar(cf, ax=ax); cb.set_label("R0")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "b1b2_r0_contour.png")); plt.close(fig)

    # ---- R0 under each year's calibration ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, df["R0"], "-o", color="#0072B2")
    ax.axhline(1, color="black", linestyle=":", label="R0 = 1")
    for _, r in df.iterrows():
        ax.annotate(f"b2 = {r['b2']:.3f}", (r['year'], r['R0']),
                    textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
    ax.set_xticks(x); ax.set_ylabel("R0 (reference climate)")
    ax.set_title("R0 under each year's calibrated parameters")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "b1b2_r0_by_year.png")); plt.close(fig)

    return df, summ, corr


# --------------------------------------------------------------------------- #
# D. CV robustness (within-year standardization)
# --------------------------------------------------------------------------- #
def run_cv_robustness():
    print("D. CV robustness (within-year standardization) ...")
    results = {}
    for target in (cvr.TARGET_B1, cvr.TARGET_B2):
        results[target] = cvr.run_robustness(target, verbose=True)

    rows = []
    for t, r in results.items():
        rows.append(r["summary"])
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(TBL, "cv_robustness_summary.csv"), index=False)

    # ---- grouped bar: within-year vs LOYO, raw vs standardized ----
    targets = [cvr.TARGET_B1, cvr.TARGET_B2]
    labels = ["b1", "b2"]
    ok = [results[t]["summary"] for t in targets]
    wy_raw = [ok[i]["within_year_r2_raw"] for i in range(len(targets))]
    loyo_raw = [ok[i]["loyo_overall_r2_raw"] for i in range(len(targets))]
    wy_std = [ok[i]["within_year_r2_std"] for i in range(len(targets))]
    loyo_std = [ok[i]["loyo_overall_r2_std"] for i in range(len(targets))]

    pos = np.arange(len(targets))
    w = 0.18
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(pos - 1.5 * w, wy_raw, w, label="within-year (raw)", color="#0072B2")
    ax.bar(pos - 0.5 * w, wy_std, w, label="within-year (std)", color="#56B4E9")
    ax.bar(pos + 0.5 * w, loyo_raw, w, label="LOYO (raw)", color="#D55E00")
    ax.bar(pos + 1.5 * w, loyo_std, w, label="LOYO (std)", color="#E69F00")
    ax.set_xticks(pos, labels)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("R²")
    ax.set_title("Within-year vs LOYO CV R² — effect of removing the year level")
    ax.legend(fontsize=9, frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "cv_robustness.png")); plt.close(fig)

    return summ, results


# --------------------------------------------------------------------------- #
def main(n_r0=2000, n_ih=300, seed=42):
    prcc_r0, tornado_df, base_out, Xr0, Yr0 = run_r0_prcc(n_r0, seed)
    print()
    prcc_ih, Xih, Yih = run_ih_prcc(n_ih, seed)
    print()
    b1b2_df, b1b2_summ, b1b2_corr = run_b1b2_uncertainty()
    print()
    cv_summ, cv_results = run_cv_robustness()
    print()
    print("All done.")
    print("Saved figures to:", FIG)
    print("Saved tables to:", TBL)


if __name__ == "__main__":
    main()
