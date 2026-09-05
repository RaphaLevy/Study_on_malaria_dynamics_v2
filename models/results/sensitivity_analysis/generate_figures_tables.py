"""Generate all figures and tables for the R0 sensitivity analysis.

Run from inside the ``sensitivity_analysis`` folder:

    python generate_figures_tables.py

Produces:
- figures/ : Sobol-index bar charts, S1-vs-ST scatter, R0 distribution, group bars
- tables/  : full and summary index tables (CSV)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sobol_analysis import (
    run_sobol,
    indices_dataframe,
    build_problem,
)
from seirs_sei_r0 import (
    r0_components,
    BASELINE,
    REFERENCE_CLIMATE,
    TRANSMISSION_SEASON_FRACTION,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
TBL = os.path.join(HERE, "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TBL, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})

GROUP_COLORS = {
    "calibrated_epi": "#1f77b4",
    "climate_rate": "#ff7f0e",
    "climate_mortality": "#2ca02c",
    "anop_suitability": "#9467bd",
    "human_immune": "#7f7f7f",
}
GROUP_LABELS = {
    "calibrated_epi": "Calibrated epidemiological",
    "climate_rate": "Temperature / rainfall rates",
    "climate_mortality": "Mosquito mortality / larval",
    "anop_suitability": "Anopheles thermal suitability",
    "human_immune": "Human immunity (not in R0)",
}


def baseline_components():
    c = r0_components(
        REFERENCE_CLIMATE["T"], REFERENCE_CLIMATE["R"], REFERENCE_CLIMATE["H"]
    )
    return c


def main():
    N_BASE = 4096
    print(f"Running Sobol sensitivity analysis (base N={N_BASE})...")
    Si, X, Y = run_sobol(n_samples=N_BASE, seed=42)
    df = indices_dataframe(Si)
    df = df.sort_values("ST", ascending=False).reset_index(drop=True)

    # ---- CSV tables ----
    df.to_csv(os.path.join(TBL, "sobol_indices_full.csv"), index=False)

    top = df[df["enters_R0"]].head(12)[
        ["parameter", "group", "baseline", "lower", "upper", "S1", "ST", "ST_conf"]
    ]
    top = top.copy()
    top["S1"] = top["S1"].round(4)
    top["ST"] = top["ST"].round(4)
    top["ST_conf"] = top["ST_conf"].round(4)
    upper_top = df[df["enters_R0"]].sort_values("S1", ascending=False).head(6)
    top_summary = pd.DataFrame({
        "quantity": ["R0_H", "R0_M", "R0"],
        "value": [c["R0_H"], c["R0_M"], c["R0"]],
        "value_desc": ["human-side factor", "mosquito-side factor", "combined sqrt product"],
    })
    top.to_csv(os.path.join(TBL, "sobol_top_parameters.csv"), index=False)
    top_summary.to_csv(os.path.join(TBL, "reference_R0_values.csv"), index=False)

    # Group summary (sum of ST, mean S1, and parameter count by group)
    g = df[df["enters_R0"]]
    gs = g.groupby("group").agg(
        sum_ST=("ST", "sum"),
        mean_S1=("S1", "mean"),
        n_params=("parameter", "count"),
    ).round(4)
    gs = gs.reindex(list(GROUP_COLORS.keys()))
    gs = gs.dropna(subset=["sum_ST"])
    gs.to_csv(os.path.join(TBL, "sobol_group_summary.csv"))

    # ---- Figure 1: total-order indices (horizontal bar, colored by group) ----
    fig, ax = plt.subplots(figsize=(8, 8))
    active = df[df["enters_R0"]]
    y_pos = np.arange(len(active))[::-1]
    colors = [GROUP_COLORS[g] for g in active["group"]]
    ax.barh(y_pos, active["ST"], color=colors, edgecolor="black", linewidth=0.4)
    ax.errorbar(active["ST"], y_pos, xerr=active["ST_conf"],
                fmt="none", ecolor="black", capsize=2, linewidth=0.7)
    ax.set_yticks(y_pos, active["parameter"])
    ax.set_xlabel("Total-order Sobol index ($S_T$)")
    ax.set_title("Global sensitivity of $R_0$ to model parameters (total-order)")
    handles = []
    for g in GROUP_COLORS:
        if any(active["group"] == g):
            handles.append(plt.Rectangle((0, 0), 1, 1,
                                         facecolor=GROUP_COLORS[g], label=GROUP_LABELS[g]))
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    ax.set_xlim(0, max(active["ST"].max() * 1.15, 0.5))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "sobST_barh.png"))
    plt.close(fig)

    # ---- Figure 2: first-order indices bar chart ----
    fig, ax = plt.subplots(figsize=(8, 8))
    active = df[df["enters_R0"]]
    y_pos = np.arange(len(active))[::-1]
    colors = [GROUP_COLORS[g] for g in active["group"]]
    ax.barh(y_pos, np.maximum(active["S1"], 0), color=colors,
            edgecolor="black", linewidth=0.4)
    ax.set_yticks(y_pos, active["parameter"])
    ax.set_xlabel("First-order Sobol index ($S_1$)")
    ax.set_title("Global sensitivity of $R_0$ to model parameters (first-order)")
    handles = []
    for g in GROUP_COLORS:
        if any(active["group"] == g):
            handles.append(plt.Rectangle((0, 0), 1, 1,
                                         facecolor=GROUP_COLORS[g], label=GROUP_LABELS[g]))
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "sobS1_barh.png"))
    plt.close(fig)

    # ---- Figure 3: S1 vs ST scatter ----
    fig, ax = plt.subplots(figsize=(7, 7))
    active = df[df["enters_R0"]]
    for g in ["calibrated_epi", "climate_rate", "climate_mortality", "anop_suitability"]:
        sub = active[active["group"] == g]
        ax.scatter(np.maximum(sub["S1"], 0), sub["ST"], s=80,
                   color=GROUP_COLORS[g], label=GROUP_LABELS[g],
                   edgecolor="black", linewidth=0.4, zorder=3)
        for _, r in sub.iterrows():
            ax.annotate(r["parameter"], (max(r["S1"], 0), r["ST"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=7)
    lim = max(active["ST"].max(), active["S1"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, label="S1 = ST")
    ax.set_xlabel("First-order index ($S_1$)")
    ax.set_ylabel("Total-order index ($S_T$)")
    ax.set_title("Interaction strength: $S_T$ vs $S_1$ for $R_0$")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "sob_S1_vs_ST.png"))
    plt.close(fig)

    # ---- Figure 4: grouped sum of ST (bar) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    gs = gs.reset_index()
    x = np.arange(len(gs))
    ax.bar(x, gs["sum_ST"], color=[GROUP_COLORS[g] for g in gs["group"]],
           edgecolor="black", linewidth=0.5)
    for xi, (_, row) in zip(x, gs.iterrows()):
        ax.text(xi, row["sum_ST"] + 0.01, f"n={int(row['n_params'])}",
                ha="center", fontsize=8)
    ax.set_xticks(x, [GROUP_LABELS[g] for g in gs["group"]], rotation=20, ha="right")
    ax.set_ylabel("Sum of total-order index ($S_T$, per group)")
    ax.set_title("Aggregated contribution of parameter groups to $R_0$ variance")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "sob_group_summary.png"))
    plt.close(fig)

    # ---- Figure 5: distribution of R0 across samples ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(Y, bins=60, color="#1f77b4", edgecolor="black", linewidth=0.3)
    ax.axvline(np.median(Y), color="red", linestyle="--",
               label=f"median R0 = {np.median(Y):.2f}")
    ax.axvline(1.0, color="black", linestyle=":", label="R0 = 1 (threshold)")
    ax.axvline(c["R0"], color="green", linestyle="-.", label="reference R0")
    ax.set_xlabel("$R_0$")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of $R_0$ across the sampled parameter space")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "R0_distribution.png"))
    plt.close(fig)

    # ---- Figure 6: reference R0 component summary ----
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.5))
    comps = pd.DataFrame({
        "component": ["R0_H (human)", "R0_M (vector)", "R0 (combined)"],
        "value": [c["R0_H"], c["R0_M"], c["R0"]],
    })
    for ax in axs:
        pass
    axs[0].bar(["R0_H", "R0_M", "R0"], [c["R0_H"], c["R0_M"], c["R0"]],
               color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    axs[0].axhline(1, color="k", linestyle="--", linewidth=0.8)
    axs[0].set_title("Reference reproduction numbers")
    axs[1].bar(["a\n(bites/night)", "mu\n(1/day)", "b3M\n(1/day)", "ell"],
               [c["a"], c["mu"], c["b3M"], c["ell"]], color="#1f77b4")
    axs[1].set_title("Reference biological rates")
    axs[2].bar(["b_bar", "K (x1e5)", "Lambda* (x1e4)"],
               [c["b_bar"], c["K"] / 1e5, c["Lambda_star"] / 1e4], color="#ff7f0e")
    axs[2].set_title("Reference recruitment / capacity")
    for ax in axs:
        for lbl in ax.get_xticklabels():
            lbl.set_fontsize(8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "reference_R0_components.png"))
    plt.close(fig)

    # Write a small metadata/params note as CSV
    meta = pd.DataFrame({
        "entry": ["reference_T", "reference_R", "reference_H",
                  "transmission_season_fraction", "n_parameters",
                  "n_samples_base_N", "baseline_R0"],
        "value": [REFERENCE_CLIMATE["T"], REFERENCE_CLIMATE["R"],
                  REFERENCE_CLIMATE["H"], TRANSMISSION_SEASON_FRACTION,
                  len(build_problem()["names"]), N_BASE, c["R0"]],
    })
    meta.to_csv(os.path.join(TBL, "analysis_metadata.csv"), index=False)

    print("Saved figures to:", FIG)
    print("Saved tables to:", TBL)
    print("Reference R0 =", round(c["R0"], 3),
          "(R0_H=%.3f, R0_M=%.3f)" % (c["R0_H"], c["R0_M"]))
    print("\nTop total-order contributors:")
    print(df[df["enters_R0"]].head(8)[["parameter", "S1", "ST", "ST_conf"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    c = baseline_components()
    main()
