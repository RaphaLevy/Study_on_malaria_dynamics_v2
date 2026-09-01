"""Stage 1: Assemble the weekly regression dataset.

Loads the weekly free-beta calibration results (2017-2023), recovers the
per-week free b1 and b2 (the per-bite transmission probabilities) by dividing
the fitted betas by the biting rate a(T), and merges with weekly-averaged
environmental variables (temperature, precipitation, humidity, deforestation,
fire counts) plus temporal features. The targets are the weekly free b1 and
b2, regressed on the environmental drivers.

Output: data/weekly_regression_dataset.csv
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
DATA_DIR = os.path.join(_PROJECT_ROOT, "data_files", "data")
DETER_DIR = os.path.join(DATA_DIR, "deter_notification_data")
LMFIT_DIR = os.path.join(_SCRIPT_DIR, "..", "lmfit_optimization")
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants (matches lmfit_optimization_shared.py)
# ---------------------------------------------------------------------------
T_PRIME = 16.0   # threshold temperature for biting rate (deg C)
D1 = 24.8        # scaling divisor for a(T)

YEARS = list(range(2017, 2024))

# Fraction of beta_h/beta_m relative to climate seed below which the week
# is considered an optimizer artifact (mosquito extinction).
FILTER_THRESHOLD = 0.01  # 1%


# ---------------------------------------------------------------------------
# Biting rate
# ---------------------------------------------------------------------------
def a(Temp):
    """Biting rate: linear ramp above T_PRIME."""
    return max(0.0, (Temp - T_PRIME) / D1)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_weekly_beta_json(year):
    """Load the weekly beta results for a given year."""
    path = os.path.join(LMFIT_DIR, f"weekly_beta_results_{year}.json")
    with open(path, "r") as f:
        data = json.load(f)
    return data


def load_climate_data():
    """Load daily climate data, filtered to 2017-2023."""
    path = os.path.join(DATA_DIR, "climate_api_data_2016_2024.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    return df[(df["date"] >= "2017-01-01") & (df["date"] <= "2023-12-31")].copy()


def load_deforestation_data():
    """Load daily treated deforestation data, filtered to 2017-2023."""
    path = os.path.join(DETER_DIR, "treated_deter_deforestation_data_2016_2024.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    return df[(df["date"] >= "2017-01-01") & (df["date"] <= "2023-12-31")].copy()


def load_fire_data():
    """Load daily fire counts, filtered to 2017-2023."""
    path = os.path.join(DATA_DIR, "inpe_fire_counts_data_2016_2024.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    return df[(df["date"] >= "2017-01-01") & (df["date"] <= "2023-12-31")].copy()


def load_fire_trends_data():
    """Load monthly Mapbiomas fire trends (ha), filtered to 2017-2023."""
    path = os.path.join(DATA_DIR, "mapbiomas_fire_trends_data_2017_2023.csv")
    df = pd.read_csv(path)
    # Parse the YYYY-MM date column to a proper datetime (first day of month)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m")
    return df[["date", "year", "month", "fire_trends (ha)"]].copy()


# ---------------------------------------------------------------------------
# Week grid helpers (mirrors weekly_beta_optimization.py)
# ---------------------------------------------------------------------------
def days_in_year(year):
    start = datetime(year, 1, 1)
    end = datetime(year, 12, 31)
    return (end - start).days + 1


def n_weeks_in_year(year):
    return int(np.ceil(days_in_year(year) / 7))


def week_index_edges(year):
    """Return (lo, hi) day-of-year (0-based, inclusive) for each weekly bucket."""
    n_days = days_in_year(year)
    n_weeks = n_weeks_in_year(year)
    edges = []
    for w in range(n_weeks):
        lo = w * 7
        hi = min(lo + 7, n_days) - 1
        edges.append((lo, hi))
    return edges


# ---------------------------------------------------------------------------
# Core assembly
# ---------------------------------------------------------------------------
def build_weekly_betas():
    """Extract weekly free betas and compute the weekly free b1, b2.

    b1_eff and b2_eff are the per-week free per-bite probabilities recovered
    from the weekly free-beta fit by dividing out the biting rate a(T):
    b1_w = beta_m,w / a(T_w), b2_w = beta_h,w / a(T_w).
    """
    rows = []
    for year in YEARS:
        data = load_weekly_beta_json(year)
        n_w = data["n_weeks"]
        bh_fitted = np.array(data["beta_h_weekly_fitted"])
        bm_fitted = np.array(data["beta_m_weekly_fitted"])
        bh_climate = np.array(data["beta_h_weekly_climate"])
        bm_climate = np.array(data["beta_m_weekly_climate"])

        edges = week_index_edges(year)
        year_start = datetime(year, 1, 1)

        # Load climate data for this year to compute weekly a(T)
        clim_path = os.path.join(DATA_DIR, "climate_api_data_2016_2024.csv")
        clim = pd.read_csv(clim_path, parse_dates=["date"])
        clim = clim[(clim["date"] >= f"{year}-01-01") & (clim["date"] <= f"{year}-12-31")].reset_index(drop=True)
        T = clim["temp_med"].values
        a_daily = np.array([a(t) for t in T])

        for w in range(n_w):
            lo, hi = edges[w]
            # Days in this week bucket
            day_indices = list(range(lo, min(hi + 1, len(a_daily))))
            a_weekly = a_daily[day_indices].mean() if day_indices else 0.0

            # Start date of this week
            start_date = year_start + timedelta(days=lo)

            # Filter: check if fitted betas are above threshold of climate seed
            valid = True
            if a_weekly > 0:
                if bh_climate[w] > 0 and bh_fitted[w] / bh_climate[w] < FILTER_THRESHOLD:
                    valid = False
                if bm_climate[w] > 0 and bm_fitted[w] / bm_climate[w] < FILTER_THRESHOLD:
                    valid = False

            # Effective b1, b2 = fitted beta / a(T_weekly)
            b1_eff = bm_fitted[w] / a_weekly if a_weekly > 0 else np.nan
            b2_eff = bh_fitted[w] / a_weekly if a_weekly > 0 else np.nan

            # b1 and b2 are per-bite infection probabilities, so they are
            # physically constrained to [0, 1]. The weekly free-beta fit can
            # produce beta_m spikes (esp. early/late year) that inflate b1
            # beyond 1; clip to enforce the biological bound.
            b1_eff = np.clip(b1_eff, 0.0, 1.0) if np.isfinite(b1_eff) else np.nan
            b2_eff = np.clip(b2_eff, 0.0, 1.0) if np.isfinite(b2_eff) else np.nan

            rows.append({
                "year": year,
                "week": w,
                "start_date": start_date,
                "n_days_in_week": len(day_indices),
                "a_weekly": a_weekly,
                "beta_h_fitted": bh_fitted[w],
                "beta_m_fitted": bm_fitted[w],
                "beta_h_climate": bh_climate[w],
                "beta_m_climate": bm_climate[w],
                "b1_eff": b1_eff,
                "b2_eff": b2_eff,
                "valid": valid,
            })

    return pd.DataFrame(rows)


def build_environmental_features(betas_df):
    """Compute weekly-averaged environmental features aligned to the beta grid."""
    climate = load_climate_data()
    defor = load_deforestation_data()
    fire = load_fire_data()
    fire_trends = load_fire_trends_data()

    # Build a (year, month) -> fire_trends_ha lookup from the monthly Mapbiomas
    # data, so each week is assigned the fire-trends value of the month it falls in.
    fire_trends["raw_year"] = fire_trends["date"].dt.year
    fire_trends["raw_month"] = fire_trends["date"].dt.month
    fire_trends_lookup = dict(
        zip(zip(fire_trends["raw_year"], fire_trends["raw_month"]),
            fire_trends["fire_trends (ha)"])
    )

    # Merge all daily data on date
    daily = climate[["date", "temp_med", "precip_med", "umid_min"]].copy()
    daily = daily.merge(defor[["date", "clear_cut_primary", "forest_degradation",
                                "burn_scar", "total_degradation"]], on="date", how="left")
    daily = daily.merge(fire[["date", "fire_counts"]], on="date", how="left")
    daily = daily.fillna(0)

    # Add year and day-of-year
    daily["year"] = daily["date"].dt.year
    daily["doy"] = daily["date"].dt.dayofyear - 1  # 0-based

    # 4-week rolling sums for deforestation and fire counts (trailing, min_periods=1)
    roll_cols = ["clear_cut_primary", "forest_degradation", "burn_scar",
                 "total_degradation", "fire_counts"]
    for col in roll_cols:
        daily[f"{col}_4wk"] = daily[col].rolling(window=28, min_periods=1).sum()

    # Aggregate to weekly resolution matching the beta grid
    env_rows = []
    for _, row in betas_df.iterrows():
        year = int(row["year"])
        w = int(row["week"])
        edges = week_index_edges(year)
        lo, hi = edges[w]

        # Monthly fire-trends value for this week (from its start-date month)
        week_date = row["start_date"]
        fire_trends_ha = fire_trends_lookup.get((week_date.year, week_date.month), np.nan)

        # Filter daily data to this week
        mask = (daily["year"] == year) & (daily["doy"] >= lo) & (daily["doy"] <= hi)
        week_data = daily[mask]

        if len(week_data) == 0:
            env_rows.append({col: np.nan for col in [
                "temp_mean", "precip_mean", "umid_min_mean",
                "defor_clearcut_4wk", "defor_degradation_4wk",
                "defor_burnscar_4wk", "defor_total_4wk",
                "fire_counts_4wk", "fire_trends_ha"
            ]})
        else:
            env_rows.append({
                "temp_mean": week_data["temp_med"].mean(),
                "precip_mean": week_data["precip_med"].mean(),
                "umid_min_mean": week_data["umid_min"].mean(),
                "defor_clearcut_4wk": week_data["clear_cut_primary_4wk"].iloc[-1],
                "defor_degradation_4wk": week_data["forest_degradation_4wk"].iloc[-1],
                "defor_burnscar_4wk": week_data["burn_scar_4wk"].iloc[-1],
                "defor_total_4wk": week_data["total_degradation_4wk"].iloc[-1],
                "fire_counts_4wk": week_data["fire_counts_4wk"].iloc[-1],
                "fire_trends_ha": fire_trends_ha,
            })

    env_df = pd.DataFrame(env_rows)
    return env_df


def add_temporal_features(df):
    """Add sinusoidal week encoding and one-hot year indicators."""
    # Sinusoidal week encoding (captures seasonality)
    df["sin_week"] = np.sin(2 * np.pi * df["week"] / 52)
    df["cos_week"] = np.cos(2 * np.pi * df["week"] / 52)

    # One-hot year indicators
    for y in YEARS:
        df[f"year_{y}"] = (df["year"] == y).astype(int)

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Building weekly beta dataset...")
    betas_df = build_weekly_betas()
    print(f"  Total weeks: {len(betas_df)}")
    print(f"  Valid weeks (after filter): {betas_df['valid'].sum()}")

    print("Building environmental features...")
    env_df = build_environmental_features(betas_df)

    # Combine
    result = pd.concat([betas_df.reset_index(drop=True),
                        env_df.reset_index(drop=True)], axis=1)

    # Add temporal features
    result = add_temporal_features(result)

    # Separate valid and filtered rows
    valid_df = result[result["valid"]].copy()
    filtered_df = result[~result["valid"]].copy()

    print(f"\nValid weeks: {len(valid_df)}")
    print(f"Filtered weeks: {len(filtered_df)}")
    if len(filtered_df) > 0:
        print("Filtered weeks:")
        for _, r in filtered_df.iterrows():
            print(f"  {int(r['year'])} W{int(r['week']):02d} "
                  f"b1_eff={r['b1_eff']:.6f} b2_eff={r['b2_eff']:.6f} "
                  f"a_w={r['a_weekly']:.4f}")

    # Summary statistics for valid data
    print("\nSummary statistics (valid weeks):")
    print(f"  b1_eff: mean={valid_df['b1_eff'].mean():.4f}, "
          f"std={valid_df['b1_eff'].std():.4f}, "
          f"min={valid_df['b1_eff'].min():.4f}, max={valid_df['b1_eff'].max():.4f}")
    print(f"  b2_eff: mean={valid_df['b2_eff'].mean():.4f}, "
          f"std={valid_df['b2_eff'].std():.4f}, "
          f"min={valid_df['b2_eff'].min():.4f}, max={valid_df['b2_eff'].max():.4f}")

    # Save full dataset (with valid flag) and valid-only dataset
    output_all = os.path.join(OUTPUT_DIR, "weekly_regression_dataset.csv")
    output_valid = os.path.join(OUTPUT_DIR, "weekly_regression_valid.csv")

    # Columns to save (drop intermediate beta columns and valid flag)
    feature_cols = [
        "year", "week", "start_date", "n_days_in_week",
        "a_weekly", "b1_eff", "b2_eff",
        "beta_h_fitted", "beta_m_fitted",
        "beta_h_climate", "beta_m_climate",
        "valid",
        "temp_mean", "precip_mean", "umid_min_mean",
        "defor_clearcut_4wk", "defor_degradation_4wk",
        "defor_burnscar_4wk", "defor_total_4wk",
        "fire_counts_4wk", "fire_trends_ha",
        "sin_week", "cos_week",
    ] + [f"year_{y}" for y in YEARS]

    result[feature_cols].to_csv(output_all, index=False)
    valid_df[feature_cols].to_csv(output_valid, index=False)

    print(f"\nSaved: {output_all}")
    print(f"Saved: {output_valid}")
    print("Done.")


if __name__ == "__main__":
    main()
