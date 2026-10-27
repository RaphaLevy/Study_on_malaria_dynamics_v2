"""Yearly time-series of the closed-form R(t) (2017-2023) with R(t)=1 crossing marks.

Uses the daily 2017-2023 climate record (temperature = temp_med, rainfall =
precip_med, humidity = umid_min, matching ``models/seirs_sei/R0_Calculation.ipynb``)
and the *per-year* calibrated epidemiological parameters (b1, b2, gamma, M_prime,
H0, k, phi) from ``load_refined_params``. All climate-response parameters are the
fixed BASELINE values of ``seirs_sei_r0.py``, so the within-year variation of R(t)
is driven entirely by the environment and the year-to-year differences by the
calibration.

Two reproduction numbers are reported for each day:

- the **coupled** ``R0 = sqrt(R0_H * R0_M)`` (spectral radius of the next
  generation matrix, feeding back through both human and mosquito legs), and
- the **human component** ``R0_H = a(T) * b2 / gamma`` (the mosquito -> human
  leg in isolation). R_H(t) depends on temperature alone (through the biting
  rate a(T)) plus the calibrated b2, gamma, so its seasonality can be compared
  directly with the ODE ``I_H`` trajectory and with the coupled curve.

Following ``lmfit_optimization_shared.simulate_year`` the R(t) curve is evaluated
on the 14-day trailing mean of the daily climate (SMOOTH_WINDOW = 14), so the
R = 1 crossings (recorded in the summary table) identify the transmission-season
onset/offset rather than day-to-day rainfall noise. The raw daily series is
overlaid for context.

Outputs
-------
- figures/r0_yearly.png : one panel per year, smoothed coupled R(t) curve with the
  R(t) = 1 threshold and shaded above-1 episodes (no crossing markers).
- figures/rH_yearly.png : same layout for the human component R_H(t) alone.
- figures/rH_coupled_yearly.png : same layout with both R(t) and R_H(t) overlaid.
- figures/rH_coupled_2017.png : single-panel 2017 overlay (for comparison with I_H).
- tables/r0_yearly_summary.csv : per-year R(t) statistics (raw and smoothed) and
  crossing dates, plus the same statistics for R_H(t) (rh_* columns).
- tables/r0_daily.csv : the full daily series (date, R0 raw/smoothed,
  R0_H raw/smoothed).
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
BLUE = GROUP_COLORS["transmission"]
ORANGE = GROUP_COLORS["biting_rate"]
GREEN = GROUP_COLORS["vector_survival"]
GRAY = "#8c8c8c"

CAL = mw.calibrated_b1_b2_by_year(YEARS)
CLIMATE = _ode.climate_data[["date", "temp_med", "precip_med", "umid_min"]].copy()


def daily_r0(year: int, smooth: bool) -> pd.DataFrame:
    """Daily (coupled, human) reproduction numbers for one year.

    Returns both the coupled ``R0`` (NGM spectral radius sqrt(R0_H * R0_M)) and
    the human component ``R0_H = a(T) * b2 / gamma``, each for every day of the
    year with that year's calibrated params.

    ``smooth=True`` mirrors ``simulate_year`` by feeding the 14-day trailing
    mean of (temp_med, precip_med, umid_min) into the R(t) response functions.
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
    comps = [
        r0_components(t, r, h, N=N, M_prime=mprime, **epi)
        for t, r, h in zip(T, R, H)
    ]
    return pd.DataFrame({
        "date": sub["date"].to_numpy(),
        "R0": np.asarray([c["R0"] for c in comps], float),
        "R0_H": np.asarray([c["R0_H"] for c in comps], float),
    })


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


def _stat_block_rh(y):
    return dict(rh_min=round(float(y.min()), 3),
                rh_mean=round(float(y.mean()), 3),
                rh_max=round(float(y.max()), 3),
                rh_days_above_1=int(np.sum(y >= 1.0)),
                rh_frac_days_above_1=round(float(np.mean(y >= 1.0)), 3))


def _draw_panel(ax, year, raw, smo, key="R0", overlay_key=None,
                main_color=BLUE, overlay_color=ORANGE, sym="R(t)",
                sym_overlay="R_H(t)", threshold=1.0):
    """Draw one year panel: raw + smoothed main R curve, optional overlay."""
    d_raw, d_smo = raw[year], smo[year]
    yv = d_smo[key].to_numpy()
    yvo = d_smo[overlay_key].to_numpy() if overlay_key else None
    dates = pd.to_datetime(d_smo["date"])

    ax.plot(dates, d_raw[key].to_numpy(), color=main_color, lw=0.8, alpha=0.35,
            zorder=2, label="_nolegend_")
    ax.plot(dates, yv, color=main_color, lw=1.8, zorder=3,
            label=f"${sym}$ on 14-day-smoothed climate")
    if yvo is not None:
        ax.plot(dates, yvo, color=overlay_color, lw=1.5, zorder=3,
                label=f"${sym_overlay}$ on 14-day-smoothed climate")
    ax.axhline(threshold, color="#555555", lw=1.0, ls="--", alpha=0.85, zorder=1)
    ax.fill_between(dates, threshold, yv, where=(yv >= threshold),
                    color=GREEN, alpha=0.22, interpolate=True, zorder=1)

    above = int(np.sum(yv >= threshold))
    arrs = [d_raw[key].to_numpy(), yv]
    if yvo is not None:
        arrs.append(yvo)
    lo = float(np.nanmin(np.concatenate(arrs)))
    hi = float(np.nanmax(np.concatenate(arrs)))
    ax.set_ylim(min(0.0, lo * 0.95), max(1.6, hi * 1.08))
    ax.set_xlim(pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31))
    ax.set_title(f"{year}", loc="left", fontsize=13)
    ax.text(1.0, 0.97,
            f"Number of days with ${sym}>1$: {above} of {len(yv)} "
            f"({100 * above / len(yv):.0f}%)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="#333333")


def _year_handles(main_color=BLUE, overlay_color=ORANGE, sym="R(t)",
                  sym_overlay="R_H(t)", with_overlay=False):
    handles = [
        plt.Line2D([], [], color=main_color, lw=1.8,
                   label=f"${sym}$ on 14-day-smoothed climate"),
        plt.Line2D([], [], color=main_color, lw=0.8, alpha=0.35,
                   label=f"Daily ${sym}$ (raw climate)"),
        plt.Line2D([], [], color="#555555", lw=1.0, ls="--",
                   label=f"${sym} = 1$ epidemic threshold"),
    ]
    if with_overlay:
        handles.insert(1, plt.Line2D([], [], color=overlay_color, lw=1.5,
                                     label=f"${sym_overlay}$ on 14-day-smoothed climate"))
    return handles


def _yearly_figure(raw, smo, out_name, key="R0", overlay_key=None,
                   main_color=BLUE, overlay_color=ORANGE, sym="R(t)",
                   sym_overlay="R_H(t)", figsize=(9.2, 13.6), top_space=0.965):
    """Seven-panel yearly figure (one panel per year)."""
    apply_paper_style()
    plt.rcParams["savefig.bbox"] = "tight"

    fig, axes = plt.subplots(len(YEARS), 1, figsize=figsize, sharex=False)
    for ax, y in zip(axes, YEARS):
        _draw_panel(ax, y, raw, smo, key=key, overlay_key=overlay_key,
                    main_color=main_color, overlay_color=overlay_color,
                    sym=sym, sym_overlay=sym_overlay)
    for ax in axes:
        ax.xaxis.set_major_locator(MonthLocator())
        ax.xaxis.set_major_formatter(DateFormatter("%b"))
        ax.tick_params(axis="y", labelsize=8.5)
    axes[-1].set_xlabel("Month")
    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False)

    handles = _year_handles(main_color, overlay_color, sym, sym_overlay,
                            with_overlay=overlay_key is not None)
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, top_space))

    out_path = os.path.join(FIG_DIR, out_name)
    if os.path.exists(out_path):
        os.remove(out_path)
    fig.savefig(out_path)
    plt.close(fig)
    print("wrote", out_path)
    return out_path


def _single_year_figure(year, out_name, raw, smo, figsize=(9.2, 3.6),
                        key="R0", overlay_key=None, main_color=BLUE,
                        overlay_color=ORANGE, sym="R(t)", sym_overlay="R_H(t)",
                        top_space=0.92):
    """Single-year figure with month labels below the axis."""
    apply_paper_style()
    plt.rcParams["savefig.bbox"] = "tight"

    fig, ax = plt.subplots(figsize=figsize)
    _draw_panel(ax, year, raw, smo, key=key, overlay_key=overlay_key,
                main_color=main_color, overlay_color=overlay_color,
                sym=sym, sym_overlay=sym_overlay)
    ax.xaxis.set_major_locator(MonthLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%b"))
    ax.tick_params(axis="y", labelsize=8.5)
    ax.set_xlabel("Month")

    handles = _year_handles(main_color, overlay_color, sym, sym_overlay,
                            with_overlay=overlay_key is not None)
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, top_space))

    out_path = os.path.join(FIG_DIR, out_name)
    if os.path.exists(out_path):
        os.remove(out_path)
    fig.savefig(out_path)
    plt.close(fig)
    print("wrote", out_path)
    return out_path


def run():
    raw = {y: daily_r0(y, smooth=False) for y in YEARS}
    smo = {y: daily_r0(y, smooth=True) for y in YEARS}

    full = pd.DataFrame({
        "year": np.repeat(YEARS, [len(raw[y]) for y in YEARS]),
        "R0": np.concatenate([raw[y]["R0"].to_numpy() for y in YEARS]),
        "R0_smoothed": np.concatenate([smo[y]["R0"].to_numpy() for y in YEARS]),
        "RH": np.concatenate([raw[y]["R0_H"].to_numpy() for y in YEARS]),
        "RH_smoothed": np.concatenate([smo[y]["R0_H"].to_numpy() for y in YEARS]),
    })
    dates = pd.to_datetime(CLIMATE["date"].to_numpy())
    full.insert(1, "date", dates)
    full.to_csv(os.path.join(TAB_DIR, "r0_daily.csv"), index=False)

    summary = []
    for y in YEARS:
        raw_y = raw[y]["R0"].to_numpy()
        smo_y = smo[y]["R0"].to_numpy()
        rh_raw = raw[y]["R0_H"].to_numpy()
        rh_smo = smo[y]["R0_H"].to_numpy()
        crossings = find_crossings(smo[y]["date"].to_numpy(), smo_y)
        up = [c[0].floor("D") for c in crossings if c[1]]
        dn = [c[0].floor("D") for c in crossings if not c[1]]
        row = {"year": y, "n_days": int(len(raw_y))}
        for tag, arr in (("", raw_y), ("_smoothed", smo_y)):
            row.update({k + tag: v for k, v in _stat_block(arr).items()})
        for tag, arr in (("", rh_raw), ("_smoothed", rh_smo)):
            row.update({k + tag: v for k, v in _stat_block_rh(arr).items()})
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

    _yearly_figure(raw, smo, "r0_yearly.png", key="R0", sym="R(t)")

    print("\nPer-year summary (smoothed-14d R(t)):")
    cols = ["year", "r0_min_smoothed", "r0_mean_smoothed", "r0_max_smoothed",
            "days_r0_above_1_smoothed", "frac_days_r0_above_1_smoothed",
            "n_crossings_smooth", "first_up_crossing_smooth",
            "last_down_crossing_smooth"]
    print(summ[cols].to_string(index=False))
    print(f"\nOverall raw R(t) range: [{full['R0'].min():.3f}, {full['R0'].max():.3f}]")
    print(f"Overall smoothed R(t) range: [{full['R0_smoothed'].min():.3f}, "
          f"{full['R0_smoothed'].max():.3f}]")
    print(f"Overall smoothed R_H(t) range: [{full['RH_smoothed'].min():.3f}, "
          f"{full['RH_smoothed'].max():.3f}]")


def run_h(out_name="rH_yearly.png"):
    """Figure of the human reproduction number R_H(t) = a(T) b2 / gamma alone."""
    raw = {y: daily_r0(y, smooth=False) for y in YEARS}
    smo = {y: daily_r0(y, smooth=True) for y in YEARS}
    return _yearly_figure(raw, smo, out_name, key="R0_H", main_color=ORANGE,
                          sym="R_H(t)")


def run_h_coupled(out_name="rH_coupled_yearly.png"):
    """Figure overlaying the human R_H(t) and the coupled R(t) per year."""
    raw = {y: daily_r0(y, smooth=False) for y in YEARS}
    smo = {y: daily_r0(y, smooth=True) for y in YEARS}
    return _yearly_figure(raw, smo, out_name, key="R0", overlay_key="R0_H",
                          main_color=BLUE, overlay_color=ORANGE, sym="R(t)",
                          sym_overlay="R_H(t)")


def run_single_year(year=2017, out_name="r0_yearly_2017.png",
                    figsize=(9.2, 3.6)):
    """Single-year copy of the coupled ``run()`` figure, with month labels below.

    Identical smoothed/raw R(t) curves, R(t) = 1 threshold and above-threshold
    shading as ``run()``, but restricted to one year and drawn as a single
    panel so the x-axis month labels appear below the curve.
    """
    raw = {year: daily_r0(year, smooth=False)}
    smo = {year: daily_r0(year, smooth=True)}
    return _single_year_figure(year, out_name, raw, smo, figsize=figsize,
                               key="R0", sym="R(t)")


def run_h_coupled_single_year(year=2017, out_name="rH_coupled_2017.png",
                              figsize=(9.2, 3.6)):
    """Single-year overlay of R_H(t) and the coupled R(t).

    The compact panel used to compare the human transmission drive with the
    ODE ``I_H`` trajectory of the same year.
    """
    raw = {year: daily_r0(year, smooth=False)}
    smo = {year: daily_r0(year, smooth=True)}
    return _single_year_figure(year, out_name, raw, smo, figsize=figsize,
                               key="R0", overlay_key="R0_H", main_color=BLUE,
                               overlay_color=ORANGE, sym="R(t)",
                               sym_overlay="R_H(t)")


if __name__ == "__main__":
    run()