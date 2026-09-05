"""Yearly time-series of the closed-form R0 (2017-2023) with R0=1 crossing marks.

Uses the daily 2017-2023 climate record (temperature = temp_med, rainfall =
precip_med, humidity = umid_min, matching ``models/seirs_sei/R0_Calculation.ipynb``)
and the *per-year* calibrated epidemiological parameters (b1, b2, gamma, M_prime,
H0, k, phi) from ``load_refined_params``. All climate-response parameters are the
fixed BASELINE values of ``seirs_sei_r0.py``, so the within-year variation of R0
is driven entirely by the environment and the year-to-year differences by the
calibration.

Following ``lmfit_optimization_shared.simulate_year`` the R0 curve is evaluated
on the 14-day trailing mean of the daily climate (SMOOTH_WINDOW = 14), so that
the R0 = 1 crossings identify the transmission-season onset/offset rather than
day-to-day rainfall noise. The raw daily series is overlaid for context.

Outputs
-------
- figures/r0_yearly.png : one panel per year, smoothed R0(t) curve with the
  R0 = 1 threshold, shaded above-1 episodes and markers (with dates) at each
  threshold crossing of the smoothed curve.
- tables/r0_yearly_summary.csv : per-year R0 statistics (raw and smoothed) and
  crossing dates.
- tables/r0_daily.csv : the full daily series (date, R0 raw, R0 smoothed).
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "../sensitivity_analysis"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "../../seirs_sei/lmfit_optimization"))

import model_wrappers as mw  # noqa: E402
import lmfit_optimization_shared as _ode  # noqa: E402
from seirs_sei_r0 import r0_components  # noqa: E402
from lhs_prcc import apply_paper_style, GROUP_COLORS  # noqa: E402

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
TAB_DIR = os.path.join(HERE, "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

YEARS = list(range(2017, 2024))
SMOOTH_WINDOW = _ode.SMOOTH_WINDOW  # 14 days, as in the calibrated ODE
BLUE = GROUP_COLORS["calibrated_epi"]
ORANGE = GROUP_COLORS["climate_rate"]
GREEN = GROUP_COLORS["climate_mortality"]
GRAY = "#8c8c8c"

CAL = mw.calibrated_b1_b2_by_year(YEARS)
CLIMATE = _ode.climate_data[["date", "temp_med", "precip_med", "umid_min"]].copy()


def daily_r0(year: int, smooth: bool) -> pd.DataFrame:
    """Daily R0 for one year with that year's calibrated params.

    ``smooth=True`` mirrors ``simulate_year`` by feeding the 14-day trailing
    mean of (temp_med, precip_med, umid_min) into the R0 response functions.
    """
    sub = CLIMATE[CLIMATE["date"].dt.year == year].reset_index(drop=True).copy()

    def _smooth(v):
        return (pd.Series(v).rolling(SMOOTH_WINDOW, min_periods=1,
                                     center=False).mean().to_numpy())

    T = _smooth(sub["temp_med"].to_numpy()) if smooth else sub["temp_med"].to_numpy()
    R = _smooth(sub["precip_med"].to_numpy()) if smooth else sub["precip_med"].to_numpy()
    H = _smooth(sub["umid_min"].to_numpy()) if smooth else sub["umid_min"].to_numpy()

    p = CAL[CAL["year"] == year].iloc[0]
    epi = dict(
        b1=float(p["b1"]), b2=float(p["b2"]), gamma=float(p["gamma"]),
        M_prime=float(p["M_prime"]), H0=float(p["H0"]),
        k=float(p["k"]), phi=float(p["phi"]),
    )
    N = round(_ode.pop_by_year[year])
    mprime = epi.pop("M_prime")
    vals = [
        r0_components(t, r, h, N=N, M_prime=mprime, **epi)["R0"]
        for t, r, h in zip(T, R, H)
    ]
    return pd.DataFrame({"date": sub["date"].to_numpy(),
                         "R0": np.asarray(vals, float)})


def find_crossings(dates, y, threshold=1.0):
    """Return (crossing_timestamp, is_up) for each crossing of the threshold."""
    d1, d2 = y[:-1], y[1:]
    out = []
    for k in np.nonzero((d1 < threshold) & (d2 >= threshold))[0]:
        frac = (threshold - d1[k]) / (d2[k] - d1[k])
        out.append((dates[k] + pd.Timedelta(frac, unit="D"), True))
    for k in np.nonzero((d1 > threshold) & (d2 <= threshold))[0]:
        frac = (d1[k] - threshold) / (d1[k] - d2[k])
        out.append((dates[k] + pd.Timedelta(frac, unit="D"), False))
    for k in np.nonzero(d2 == threshold)[0]:
        if d1[k] < threshold:
            out.append((dates[k], True))
        elif d1[k] > threshold:
            out.append((dates[k], False))
    out.sort(key=lambda x: x[0])
    return out


def _stat_block(y):
    return dict(r0_min=round(float(y.min()), 3),
                r0_mean=round(float(y.mean()), 3),
                r0_max=round(float(y.max()), 3),
                days_r0_above_1=int(np.sum(y >= 1.0)),
                frac_days_r0_above_1=round(float(np.mean(y >= 1.0)), 3))


def run():
    apply_paper_style()
    plt.rcParams["savefig.bbox"] = "tight"

    raw = {y: daily_r0(y, smooth=False) for y in YEARS}
    smo = {y: daily_r0(y, smooth=True) for y in YEARS}

    full = pd.DataFrame({
        "year": np.repeat(YEARS, [len(raw[y]) for y in YEARS]),
        "R0": np.concatenate([raw[y]["R0"].to_numpy() for y in YEARS]),
        "R0_smoothed": np.concatenate([smo[y]["R0"].to_numpy() for y in YEARS]),
    })
    dates = pd.to_datetime(CLIMATE["date"].to_numpy())
    full.insert(1, "date", dates)
    full.to_csv(os.path.join(TAB_DIR, "r0_daily.csv"), index=False)

    summary = []
    for y in YEARS:
        raw_y = raw[y]["R0"].to_numpy()
        smo_y = smo[y]["R0"].to_numpy()
        crossings = find_crossings(smo[y]["date"].to_numpy(), smo_y)
        up = [c[0].floor("D") for c in crossings if c[1]]
        dn = [c[0].floor("D") for c in crossings if not c[1]]
        row = {"year": y, "n_days": int(len(raw_y))}
        for tag, arr in (("", raw_y), ("_smoothed", smo_y)):
            row.update({k + tag: v for k, v in _stat_block(arr).items()})
        row.update({
            "n_crossings_smooth": len(crossings),
            "first_up_crossing_smooth": str(up[0].date()) if up else "",
            "last_down_crossing_smooth": str(dn[-1].date()) if dn else "",
            "up_crossing_dates_smooth": "|".join(d.strftime("%Y-%m-%d") for d in up),
            "down_crossing_dates_smooth": "|".join(d.strftime("%Y-%m-%d") for d in dn),
        })
        summary.append(row)
    summ = pd.DataFrame(summary)
    summ.to_csv(os.path.join(TAB_DIR, "r0_yearly_summary.csv"), index=False)

    # ------------------------- Figure --------------------------- #
    fig, axes = plt.subplots(len(YEARS), 1, figsize=(9.2, 13.6), sharex=False)
    for ax, y in zip(axes, YEARS):
        d_raw, d_smo = raw[y], smo[y]
        yv = d_smo["R0"].to_numpy()
        dates = pd.to_datetime(d_smo["date"])

        ax.plot(dates, d_raw["R0"], color=BLUE, lw=0.8, alpha=0.35,
                zorder=2, label="_nolegend_")
        ax.plot(dates, yv, color=BLUE, lw=1.8, zorder=3,
                label="Daily $R_0$ (14-day smoothed climate)")
        ax.axhline(1.0, color="#555555", lw=1.0, ls="--", alpha=0.85, zorder=1)
        ax.fill_between(dates, 1.0, yv, where=(yv >= 1.0),
                        color=GREEN, alpha=0.22, interpolate=True, zorder=1)

        crossings = find_crossings(dates.to_numpy(), yv)
        for ts, is_up in crossings:
            ax.scatter([ts], [1.0], s=48, marker="^" if is_up else "v",
                       color=ORANGE, edgecolor="k", lw=0.4, zorder=6)
            ax.annotate(ts.floor("D").strftime("%b %d"), (ts, 1.0),
                        textcoords="offset points",
                        xytext=(0, 9) if is_up else (0, -11),
                        ha="center", va="bottom" if is_up else "top",
                        fontsize=7.5, color=ORANGE, rotation=90, zorder=6)

        above = int(np.sum(yv >= 1.0))
        lo = float(np.nanmin(np.concatenate([d_raw["R0"].to_numpy(), yv])))
        hi = float(np.nanmax(np.concatenate([d_raw["R0"].to_numpy(), yv])))
        ax.set_ylim(min(0.0, lo * 0.95), max(1.6, hi * 1.08))
        ax.set_xlim(pd.Timestamp(y, 1, 1), pd.Timestamp(y, 12, 31))
        ax.set_title(f"{y}", loc="left", fontsize=13)
        ax.text(1.0, 0.97,
                f"season $R_0>1$: {above} of {len(yv)} days "
                f"({100 * above / len(yv):.0f}%)",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color="#333333")

    for ax in axes:
        ax.xaxis.set_major_locator(MonthLocator())
        ax.xaxis.set_major_formatter(DateFormatter("%b"))
        ax.tick_params(axis="y", labelsize=8.5)
    axes[-1].set_xlabel("Month")
    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False)

    handles = [
        plt.Line2D([], [], color=BLUE, lw=1.8,
                   label="$R_0$ on 14-day-smoothed climate"),
        plt.Line2D([], [], color=BLUE, lw=0.8, alpha=0.35,
                   label="Daily $R_0$ (raw climate)"),
        plt.Line2D([], [], color="#555555", lw=1.0, ls="--",
                   label="$R_0 = 1$ epidemic threshold"),
        plt.Line2D([], [], marker="^", color="w", mec="#333333",
                   mfc=ORANGE, lw=0, label="Crosses 1 (up)"),
        plt.Line2D([], [], marker="v", color="w", mec="#333333",
                   mfc=ORANGE, lw=0, label="Crosses 1 (down)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.965))

    out_path = os.path.join(FIG_DIR, "r0_yearly.png")
    if os.path.exists(out_path):
        os.remove(out_path)
    fig.savefig(out_path)
    plt.close(fig)
    print("wrote", out_path)

    print("\nPer-year summary (smoothed-14d R0):")
    cols = ["year", "r0_min_smoothed", "r0_mean_smoothed", "r0_max_smoothed",
            "days_r0_above_1_smoothed", "frac_days_r0_above_1_smoothed",
            "n_crossings_smooth", "first_up_crossing_smooth",
            "last_down_crossing_smooth"]
    print(summ[cols].to_string(index=False))
    print(f"\nOverall raw R0 range: [{full['R0'].min():.3f}, {full['R0'].max():.3f}]")
    print(f"Overall smoothed R0 range: [{full['R0_smoothed'].min():.3f}, "
          f"{full['R0_smoothed'].max():.3f}]")


if __name__ == "__main__":
    run()