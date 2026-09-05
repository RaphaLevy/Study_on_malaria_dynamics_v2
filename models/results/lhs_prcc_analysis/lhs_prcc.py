"""LHS sampling and PRCC (Partial Rank Correlation Coefficient) utilities.

Two complementary, rank-based sensitivity measures are implemented:

* **PRCC** (Partial Rank Correlation Coefficient): the partial correlation
  between each parameter and the output after the (rank-transformed) linear
  effect of all *other* parameters has been removed. PRCC is the standard
  screening measure for nonlinear but monotone models, and — unlike Sobol'
  indices, which need thousands of model runs — it is cheap enough to apply to
  a costly ODE output such as the seasonal peak of ``I_H``.

* **Spearman rank correlation** between each parameter and the output
  (first-order, marginal) for the tornado diagram.

Sampling uses SALib's Latin Hypercube Sampler (``SALib.sample.latin``), which
is stratified and produces well-spread samples for a modest number of runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SALib.sample import latin as latin_sampler

from sobol_analysis import BOUNDS, GROUPS


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def lhs_sample(names, n_samples=500, seed=42):
    """Latin-hypercube sample of ``n_samples`` rows over the parameter bounds.

    Returns an ndarray of shape (n_samples, len(names)) in *physical* units
    (SALib's latin sampler returns un-scaled values).
    """
    bounds = [BOUNDS[n] for n in names]
    problem = {"num_vars": len(names), "names": list(names), "bounds": bounds}
    X = latin_sampler.sample(problem, n_samples, seed=seed)
    return np.asarray(X, dtype=float)


def sample_to_frame(X, names):
    return pd.DataFrame(X, columns=list(names))


# --------------------------------------------------------------------------- #
# PRCC
# --------------------------------------------------------------------------- #
def _rank(x):
    # Average ranks for ties; n_percentile=0 disables percentile blocking.
    from scipy.stats import rankdata
    return rankdata(x, method="average")


def _partial_corr(ranks_X, ranks_y):
    """Partial correlation of each input rank column with y ranks.

    Computed by regressing each input and y on the remaining inputs (ranks),
    and correlating the residuals. Returns (prcc, p_value) per input using a
    t-distribution on n - k - 2 degrees of freedom.
    """
    from scipy.stats import t
    n, k = ranks_X.shape
    prcc = np.empty(k)
    pval = np.empty(k)
    for j in range(k):
        other = np.delete(np.arange(k), j)
        X_other = np.column_stack([np.ones(n), ranks_X[:, other]])
        X_j = ranks_X[:, j]
        # residualise both the parameter and the output on the others
        beta_j = np.linalg.lstsq(X_other, X_j, rcond=None)[0]
        beta_y = np.linalg.lstsq(X_other, ranks_y, rcond=None)[0]
        res_j = X_j - X_other @ beta_j
        res_y = ranks_y - X_other @ beta_y
        r = np.corrcoef(res_j, res_y)[0, 1]
        # r is undefined if either residual has zero variance
        if not np.isfinite(r):
            prcc[j] = np.nan
            pval[j] = np.nan
            continue
        prcc[j] = r
        df = n - k - 1
        if df > 0 and abs(r) < 1.0:
            stat = r * np.sqrt((n - k - 1) / (1.0 - r**2))
            pval[j] = 2.0 * t.sf(abs(stat), df)
        else:
            pval[j] = np.nan
    return prcc, pval


def prcc_correlations(X, Y, names=None):
    """Compute PRCC (with p-values) of model output ``Y`` on sample matrix ``X``.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Parameter samples (physical units).
    Y : ndarray, shape (n,)
        Model output.
    names : list, optional
        Parameter names (defaults to ``list(BOUNDS)`` if X is a row-sample).

    Returns
    -------
    DataFrame with columns parameter, group, PRCC, p_value.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if names is None:
        names = list(BOUNDS.keys())[: X.shape[1]]
    names = list(names)[: X.shape[1]]

    # Drop non-finite outputs (ODE failures / NaN) but keep sample structure.
    keep = np.isfinite(Y) & np.all(np.isfinite(X), axis=1)
    X_k, Y_k = X[keep], Y[keep]
    if Y_k.size == 0:
        raise ValueError("No finite outputs to compute PRCC on.")

    rank_X = np.column_stack([_rank(X_k[:, j]) for j in range(X_k.shape[1])])
    rank_Y = _rank(Y_k)
    prcc, pval = _partial_corr(rank_X, rank_Y)

    rows = []
    for j, n in enumerate(names):
        rows.append({
            "parameter": n,
            "group": _group_of(n),
            "lower": BOUNDS.get(n, (np.nan, np.nan))[0],
            "upper": BOUNDS.get(n, (np.nan, np.nan))[1],
            "PRCC": prcc[j],
            "p_value": pval[j],
            "n_used": int(len(Y_k)),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Marginal (Spearman) correlation for tornado diagrams
# --------------------------------------------------------------------------- #
def spearman_correlations(X, Y, names=None):
    """First-order (marginal) Spearman rank correlations for a tornado plot."""
    from scipy.stats import spearmanr
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if names is None:
        names = list(BOUNDS.keys())[: X.shape[1]]
    names = list(names)[: X.shape[1]]
    keep = np.isfinite(Y) & np.all(np.isfinite(X), axis=1)
    out = []
    for j, n in enumerate(names):
        rho, p = spearmanr(X[keep, j], Y[keep])
        out.append({"parameter": n, "group": _group_of(n),
                    "spearman": rho, "p_value": p})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Tornado (one-at-a-time over each parameter's range, other params at baseline)
# --------------------------------------------------------------------------- #
def tornado_values(eval_fn, names, baseline=None, n_points=100, seed=42):
    """One-at-a-time sweep of each parameter over its range.

    ``eval_fn(param_values, names)`` must return a 1-D array of the scalar
    output for each row of ``param_values`` (columns = ``names``). For each
    parameter, all others are held fixed at ``baseline`` while that parameter
    is swept linearly across its bounds; the min and max of the output give the
    bar extent for the tornado diagram.

    Returns a DataFrame with parameter, group, low, high, swing (= high-low),
    baseline output (for centring) and the low/high output values.
    """
    if baseline is None:
        baseline = {n: np.mean(BOUNDS[n]) for n in names}
    base_vec = np.array([baseline.get(n, np.mean(BOUNDS[n])) for n in names])

    base_out = float(np.asarray(eval_fn(base_vec.reshape(1, -1), names))[0])

    rows = []
    lo_grid = np.linspace(0, 1, n_points)
    for j, n in enumerate(names):
        lo, hi = BOUNDS[n]
        sweep = lo + (hi - lo) * lo_grid
        X = np.tile(base_vec, (n_points, 1))
        X[:, j] = sweep
        Y = np.asarray(eval_fn(X, names))
        Y = Y[np.isfinite(Y)]
        rows.append({
            "parameter": n, "group": _group_of(n),
            "lower": lo, "upper": hi,
            "low_out": float(np.min(Y)) if Y.size else np.nan,
            "high_out": float(np.max(Y)) if Y.size else np.nan,
            "swing": float(np.max(Y) - np.min(Y)) if Y.size else np.nan,
        })
    df = pd.DataFrame(rows).sort_values("swing", ascending=False)
    return df, base_out


# --------------------------------------------------------------------------- #
# Colouring / labels shared by figure code
# --------------------------------------------------------------------------- #
# Okabe-Ito (colourblind-safe) palette.
GROUP_COLORS = {
    "calibrated_epi": "#0072B2",
    "climate_rate": "#E69F00",
    "climate_mortality": "#009E73",
    "anop_suitability": "#CC79A7",
    "human_immune": "#999999",
}
GROUP_LABELS = {
    "calibrated_epi": "Calibrated epidemiological",
    "climate_rate": "Temperature / rainfall rates",
    "climate_mortality": "Mosquito mortality / larval",
    "anop_suitability": "Anopheles thermal suitability",
    "human_immune": "Human immunity",
}
_GROUP_ORDER = list(GROUP_COLORS.keys())


def _group_of(name):
    for g, members in GROUPS.items():
        if name in members:
            return g
    return "other"


def apply_paper_style():
    """Consistent, journal-ready matplotlib defaults (spines, grid, fonts)."""
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#cfcfcf",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.55,
        "axes.axisbelow": True,
        "axes.titleweight": "bold",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
    })


def sig_stars(p):
    """Conventional significance stars from a (one-sided-adjusted) p-value."""
    if p is None or not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def prcc_barh(df, output_label, title, figsize=(6.5, 9.5), ax=None,
              legend=True):
    """Publication-style horizontal bar chart of PRCC with significance stars.

    Shows *all* parameters (sorted by |PRCC|), colour-coded by group, with
    ``* / ** / ***`` markers for p<0.05 / p<0.01 / p<0.001 placed at the end of
    each bar and a legend outside the axes. Returns ``(fig, ax)``. If ``ax`` is
    given, draws into it (``fig`` returned is ``None``).
    """
    d = df[df["PRCC"].notna()].copy()
    d["star"] = [sig_stars(p) for p in d["p_value"]]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        own_fig = True
    else:
        fig, own_fig = None, False
    ypos = np.arange(len(d))[::-1]
    ax.barh(ypos, d["PRCC"], color=[GROUP_COLORS[g] for g in d["group"]],
            edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.9)
    for yi, (v, st) in enumerate(zip(d["PRCC"], d["star"])):
        if st:
            ax.text(v + (0.012 if v >= 0 else -0.012), yi, st,
                    ha="left" if v >= 0 else "right", va="center", fontsize=9)
    ax.set_yticks(ypos, d["parameter"])
    ax.set_xlim(-1.13, 1.13)
    ax.set_xlabel(output_label)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    if legend:
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=GROUP_COLORS[g],
                                 label=GROUP_LABELS[g])
                   for g in _GROUP_ORDER if (d["group"] == g).any()]
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(1.01, 0.0),
                  frameon=False, fontsize=9, title="Parameter group")
    if own_fig:
        fig.subplots_adjust(right=0.80)
    return fig, ax


def tornado_fig(df, base_out, figsize=(6.5, 9.5), use_symlog=False,
                title="One-at-a-time tornado", value_label="Change in output",
                ax=None):
    """Publication-style butterfly tornado centred on the baseline output.

    ``df`` is the frame returned by :func:`tornado_values` (sorted by swing,
    largest first). Red bars = move to lower bound, blue bars = move to upper
    bound. With ``use_symlog`` a symmetric-log axis shows parameters spanning
    several orders of magnitude in swing. Returns ``(fig, ax)``.
    """
    d = df.iloc[::-1].copy()  # largest swing on top
    ypos = np.arange(len(d))
    lo_d = (d["low_out"] - base_out).values.astype(float)
    hi_d = (d["high_out"] - base_out).values.astype(float)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        own_fig = True
    else:
        fig, own_fig = None, False
    ax.barh(ypos, lo_d, color="#D55E00", edgecolor="black", linewidth=0.4,
            label="to lower bound")
    ax.barh(ypos, hi_d, color="#0072B2", edgecolor="black", linewidth=0.4,
            label="to upper bound")
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_yticks(ypos, d["parameter"])
    ax.set_xlabel(f"{value_label} (baseline = {base_out:.2f})")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=9)
    if use_symlog:
        maxabs = max(abs(lo_d.min()) if lo_d.size else 0.0,
                     abs(hi_d.max()) if hi_d.size else 0.0)
        linthresh = max(0.05, 0.01 * maxabs)
        ax.set_xscale("symlog", linthresh=linthresh)
        ax.set_xlabel(f"{value_label} (baseline = {base_out:.2f}; symlog axis)")
    if own_fig:
        fig.tight_layout()
    return fig, ax
