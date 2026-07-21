import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import os
import warnings

warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-whitegrid")

# Path to data directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_SCRIPT_DIR, "../../../data_files/data")

# Load climate data
climate_2012_2015 = pd.read_csv(
    os.path.join(DATA_DIR, "climate_api_data_2012_2015.csv")
)
climate_2016_2024 = pd.read_csv(
    os.path.join(DATA_DIR, "climate_api_data_2016_2024.csv")
)

climate_2012_2015["date"] = pd.to_datetime(climate_2012_2015["date"])
climate_2016_2024["date"] = pd.to_datetime(climate_2016_2024["date"])

climate_full = pd.concat([climate_2012_2015, climate_2016_2024], ignore_index=True)
climate_full = climate_full[
    (climate_full["date"] >= "2012-01-01") & (climate_full["date"] <= "2023-12-31")
].reset_index(drop=True)

warmup_end_idx = (climate_full["date"] < "2017-01-01").sum()
num_total_days = len(climate_full)

print(f"Climate data 2012-2023: {num_total_days} days")
print(f"Warm-up (2012-2016): {warmup_end_idx} days")
print(f"Main period (2017-2023): {num_total_days - warmup_end_idx} days")

# Add smoothed columns
SMOOTH_WINDOW = 14
climate_full["temp_med_smooth"] = (
    climate_full["temp_med"].rolling(SMOOTH_WINDOW, min_periods=1, center=False).mean()
)
climate_full["precip_med_smooth"] = (
    climate_full["precip_med"]
    .rolling(SMOOTH_WINDOW, min_periods=1, center=False)
    .mean()
)
climate_full["umid_min_smooth"] = (
    climate_full["umid_min"].rolling(SMOOTH_WINDOW, min_periods=1, center=False).mean()
)

# Load cases data
warmup_cases_data = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "sivep_notification_data/treated_malaria_notification_data/cumulative_manaus_cases_2012_2015.csv",
    )
)
cases_data = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "sivep_notification_data/treated_malaria_notification_data/cumulative_manaus_cases_2016_2023.csv",
    )
)

warmup_cases_data["date"] = pd.to_datetime(warmup_cases_data["date"])
cases_data["date"] = pd.to_datetime(cases_data["date"])

# Load population data
pop_ibge = pd.read_csv(
    os.path.join(DATA_DIR, "ibge_manaus_rural_population_data_2000_2025.csv")
)
pop_by_year = dict(zip(pop_ibge["year"], pop_ibge["rural_population"]))

# Filter for analysis period
start_date = pd.to_datetime("2017-01-01")
end_date = pd.to_datetime("2023-12-31")
climate_data = climate_full[
    (climate_full["date"] >= start_date) & (climate_full["date"] <= end_date)
].reset_index(drop=True)

# Rural cases (41.54% of total for Manaus rural areas)
rural_cases_df = cases_data.copy()
cols = [
    "active_total",
    "active_symptomatic",
    "active_asymptomatic",
    "new_cases",
    "cumulative_cases",
    "active_per_new_case",
]
rural_cases_df[cols] = rural_cases_df[cols] * (41.54 / 100)

print("\nIBGE rural population by year (2017-2023):")
for y in range(2017, 2024):
    print(f"{y}: {pop_by_year[y]:.2f}")

# Model Parameters
T_prime = 16.0
B_E = 200
p_ME = 0.9
p_ML = 0.25
p_MP = 0.75
tau_E = 1
tau_P = 1
c1 = 0.00554
c2 = -0.06737
D1 = 24.8
b1 = 0.04
b2 = 0.09
A = -0.03
B = 1.31
C = -4.4
DD = 105
Tmin = 14.5
gamma = 1 / 120
R_L = 32.67
tau_H = 10
omega = 1 / 270
optimal_temp_plasm = 23.5
critical_max_temp_plasm = 29.8
optimal_temp_anop = 25
critical_max_temp_anop = 34
critical_min_temp_anop = 16

M_prime_warmup = 40 * round(pop_by_year[2012])
M_min = 50000

# Initial conditions for 2012 (warm-up)
N_warmup = round(pop_by_year[2012])
warmup_rural_cases_df = warmup_cases_data.copy()
warmup_rural_cases_df[cols] = warmup_rural_cases_df[cols] * (41.54 / 100)
warmup_I_H0 = round(warmup_rural_cases_df["active_total"].iloc[0])
warmup_E_H0 = round(N_warmup * 0.05)
warmup_R_H0 = round(N_warmup * 0.15)
warmup_S_H0 = N_warmup - warmup_E_H0 - warmup_I_H0 - warmup_R_H0

warmup_M_0 = 10 * N_warmup
warmup_I_M0 = round(warmup_M_0 * 0.01)
warmup_E_M0 = round(warmup_I_M0 * 3)
warmup_S_M0 = warmup_M_0 - warmup_E_M0 - warmup_I_M0
initial_state_2012 = np.array(
    [
        warmup_S_H0,
        warmup_E_H0,
        warmup_I_H0,
        warmup_R_H0,
        warmup_S_M0,
        warmup_E_M0,
        warmup_I_M0,
    ]
)

print(
    f"\nInitial human compartments in 2012: N={N_warmup}, S_H={warmup_S_H0}, E_H={warmup_E_H0}, I_H={warmup_I_H0}, R_H={warmup_R_H0}"
)
print(
    f"Initial mosquito compartments in 2012: M={warmup_M_0}, S_M={warmup_S_M0}, E_M={warmup_E_M0}, I_M={warmup_I_M0}"
)


# Helper functions
def tau_L(Temp):
    denom = c1 * Temp + c2
    return 1.0 / denom if denom > 0 else 100.0


def p_T(Temp, Humid, H0=58.0, k=0.25, phi=0.05):
    denominator = A * Temp**2 + B * Temp + C
    p_T_temp = np.where(denominator <= 0, 0.0, np.exp(-1 / denominator))
    p_H = phi + (1.0 - phi) / (1.0 + np.exp(-k * (Humid - H0)))
    return p_T_temp * p_H


def mu(Temp, Humid, H0=58.0, k=0.25, phi=0.05):
    p_val = p_T(Temp, Humid, H0, k, phi)
    return -np.log(p_val) if p_val > 0 else 1.0


def a(Temp):
    return max(0, (Temp - T_prime) / D1)


def tau_M(Temp):
    diff = Temp - Tmin
    return DD / diff if diff > 0 else 1000.0


def b_rate(Rain, Temp):
    tau_L_curr = tau_L(Temp)
    if tau_L_curr <= 0:
        return 0.0
    p_ER = (4 * p_ME / R_L**2) * Rain * (R_L - Rain) if 0 <= Rain <= R_L else 0.0
    p_LR = (4 * p_ML / R_L**2) * Rain * (R_L - Rain) if 0 <= Rain <= R_L else 0.0
    p_LT = np.exp(-(c1 * Temp + c2))
    p_PR = (4 * p_MP / R_L**2) * Rain * (R_L - Rain) if 0 <= Rain <= R_L else 0.0
    numer = B_E * p_ER * p_LR * p_LT * p_PR
    denom = tau_E + tau_L_curr + tau_P
    return numer / denom if denom > 0 else 0.0


def b3_briere_unscaled_plasm(Temp):
    return Temp * (Temp - Tmin) * (critical_max_temp_plasm - Temp) ** (1 / 2)


unscaled_peak_value_plasm = b3_briere_unscaled_plasm(optimal_temp_plasm)
desired_peak_rate_plasm = 1.0 / tau_M(optimal_temp_plasm)
eta_plasm = desired_peak_rate_plasm / unscaled_peak_value_plasm


def b3_briere_scaled_plasm(Temp):
    if Temp <= Tmin or Temp >= critical_max_temp_plasm:
        return 0.0
    return (
        eta_plasm * Temp * (Temp - Tmin) * ((critical_max_temp_plasm - Temp) ** (1 / 2))
    )


def temp_factor_briere_normalized_anop(Temp):
    if Temp <= critical_min_temp_anop or Temp >= critical_max_temp_anop:
        return 0.0
    raw = (
        Temp
        * (Temp - critical_min_temp_anop)
        * ((critical_max_temp_anop - Temp) ** 0.5)
    )
    opt_raw = (
        optimal_temp_anop
        * (optimal_temp_anop - critical_min_temp_anop)
        * ((critical_max_temp_anop - optimal_temp_anop) ** 0.5)
    )
    return min(1.0, raw / opt_raw)


def run_with_humidity(H0_val, k_val, phi_val, M_prime_val=M_prime_warmup):
    """Run warm-up (2012-2016) + main (2017-2023) with custom H0, k, phi."""

    # Custom mu function with the given humidity parameters
    def custom_p_T(Temp, Humid):
        denominator = A * Temp**2 + B * Temp + C
        p_T_temp = np.where(denominator <= 0, 0.0, np.exp(-1 / denominator))
        p_H = phi_val + (1.0 - phi_val) / (1.0 + np.exp(-k_val * (Humid - H0_val)))
        return p_T_temp * p_H

    def custom_mu(Temp, Humid):
        p_val = custom_p_T(Temp, Humid)
        return -np.log(p_val) if p_val > 0 else 1.0

    def make_ode_func(N, clim, M_prime_local):
        def ode_func(t, z):
            day_idx = int(t)
            day_idx = np.clip(day_idx, 0, len(clim) - 1)
            T_curr = clim.iloc[day_idx]["temp_med"]
            R_curr = clim.iloc[day_idx]["precip_med"]
            T_value = clim.iloc[day_idx]["temp_med_smooth"]
            R_value = clim.iloc[day_idx]["precip_med_smooth"]
            H_value = clim.iloc[day_idx]["umid_min_smooth"]

            mu_curr = custom_mu(T_value, H_value)
            a_curr = a(T_curr)
            tau_M_curr = tau_M(T_value)
            b3_m_curr = 1.0 / tau_M_curr if tau_M_curr > 0 else 0.0
            b3_h_curr = 1.0 / tau_H if tau_H > 0 else 0.0
            l_curr = np.exp(-mu_curr * tau_M_curr)
            b_curr = b_rate(R_value, T_value)
            b3_briere_curr = b3_briere_scaled_plasm(T_curr)

            S_H, E_H, I_H, R_H, S_M, E_M, I_M = z
            foi_h = a_curr * b2 * (I_M / N)
            foi_m = a_curr * b1 * (I_H / N)

            dShdt = -foi_h * S_H + omega * R_H
            dEhdt = foi_h * S_H - b3_h_curr * E_H
            dIhdt = b3_h_curr * E_H - gamma * I_H
            dRhdt = gamma * I_H - omega * R_H

            temp_factor = temp_factor_briere_normalized_anop(T_curr)
            habitat_creating_factor = 2 * R_curr / R_L
            habitat_flushing_factor = np.exp(1 - (2 * R_curr / R_L))
            rain_factor = habitat_creating_factor * habitat_flushing_factor

            K = max(M_prime_local * temp_factor * rain_factor, M_min)
            total_mosq = S_M + E_M + I_M
            density_factor = max(0, 1 - total_mosq / K) if K > 0 else 0.0
            mosquito_birth = b_curr * density_factor * K
            residual_birth_rate = mu_curr * M_min
            residual_recruitment = residual_birth_rate * max(0, 1 - total_mosq / M_min)
            mosquito_birth = mosquito_birth + residual_recruitment

            dSmdt = mosquito_birth - foi_m * S_M - mu_curr * S_M
            dEmdt = foi_m * S_M - (mu_curr + b3_briere_curr * l_curr) * E_M
            dImdt = b3_briere_curr * l_curr * E_M - mu_curr * I_M

            return [dShdt, dEhdt, dIhdt, dRhdt, dSmdt, dEmdt, dImdt]

        return ode_func

    years = list(range(2017, 2024))

    # Warm-up
    sol_wu = solve_ivp(
        make_ode_func(N_warmup, climate_full, M_prime_warmup),
        (0, warmup_end_idx),
        initial_state_2012,
        t_eval=np.arange(0, warmup_end_idx, 1),
        method="LSODA",
    )
    if not sol_wu.success:
        return np.array([]), np.array([])
    warmup_end = np.maximum(sol_wu.y[:, -1].copy(), 0)

    # Main simulation
    all_IH = []
    all_dates = []
    current_state = warmup_end
    for year in years:
        N = round(pop_by_year[year])
        start = f"{year}-01-01"
        end = f"{year}-12-31"
        year_climate = climate_data[
            (climate_data["date"] >= start) & (climate_data["date"] <= end)
        ].reset_index(drop=True)
        year_climate["temp_med_smooth"] = (
            year_climate["temp_med"]
            .rolling(SMOOTH_WINDOW, min_periods=1, center=False)
            .mean()
        )
        year_climate["precip_med_smooth"] = (
            year_climate["precip_med"]
            .rolling(SMOOTH_WINDOW, min_periods=1, center=False)
            .mean()
        )
        year_climate["umid_min_smooth"] = (
            year_climate["umid_min"]
            .rolling(SMOOTH_WINDOW, min_periods=1, center=False)
            .mean()
        )
        num_days = len(year_climate)
        sol = solve_ivp(
            make_ode_func(N, year_climate, M_prime_warmup),
            [0, num_days],
            current_state,
            t_eval=np.linspace(0, num_days, num_days * 10),
            method="LSODA",
        )
        if not sol.success:
            return np.array([]), np.array([])
        current_state = np.maximum(sol.y[:, -1].copy(), 0)
        t_interp = np.linspace(0, sol.t[-1], len(year_climate))
        IH_interp = interp1d(sol.t, sol.y[2])(t_interp)
        all_IH.extend(IH_interp)
        all_dates.extend(year_climate["date"].values)
    return np.array(all_dates), np.array(all_IH)


def compute_weighted_mse(
    dates,
    IH,
    rural_cases_df,
    weight_decay="exponential",
    decay_rate=0.005,
    initial_weight=5.0,
    step_threshold=365,
    step_later_weight=1.0,
    inverse_scale=365,
):
    """
    Compute weighted MSE giving more weight to early time periods.

    Parameters:
    -----------
    dates : array-like
        Model prediction dates
    IH : array-like
        Model predicted infectious humans
    rural_cases_df : DataFrame
        Observed data with 'date' and 'active_total' columns
    weight_decay : str
        'exponential', 'linear', 'step', or 'inverse'
    decay_rate : float
        For exponential: weight = exp(-decay_rate * days_from_start)
        For linear: weight = 1 / (1 + decay_rate * days_from_start)
    initial_weight : float
        Multiplier for the weight at time 0
    step_threshold : int
        Days threshold for step function weighting
    step_later_weight : float
        Weight after the step threshold
    inverse_scale : int
        Days for inverse decay weight to drop to half

    Returns:
    --------
    weighted_mse : float
        The weighted mean squared error
    """
    model_interp = interp1d(
        pd.to_datetime(dates).astype(np.int64),
        IH,
        kind="linear",
        fill_value="extrapolate",
    )
    obs_dates_int = rural_cases_df["date"].astype(np.int64).values
    model_at_obs = model_interp(obs_dates_int)
    observed = rural_cases_df["active_total"].values

    start_date = rural_cases_df["date"].iloc[0]
    days_from_start = (rural_cases_df["date"] - start_date).dt.days.values.astype(float)

    if weight_decay == "exponential":
        weights = initial_weight * np.exp(-decay_rate * days_from_start)
    elif weight_decay == "linear":
        weights = initial_weight / (1 + decay_rate * days_from_start)
    elif weight_decay == "step":
        weights = np.where(
            days_from_start <= step_threshold, initial_weight, step_later_weight
        )
    elif weight_decay == "inverse":
        weights = initial_weight * (1.0 / (1.0 + days_from_start / inverse_scale))
    else:
        weights = np.ones_like(observed)

    weighted_mse = np.sum(weights * (model_at_obs - observed) ** 2) / np.sum(weights)
    return weighted_mse


def objective_function_weighted(
    params,
    weight_decay="exponential",
    decay_rate=0.005,
    initial_weight=5.0,
    step_threshold=365,
    step_later_weight=1.0,
    inverse_scale=365,
):
    """
    Weighted objective function for calibration.

    Parameters:
    -----------
    params : list
        [H0, k, phi] parameters
    weight_decay : str
        Type of weight decay
    decay_rate : float
        Rate of weight decay
    initial_weight : float
        Initial weight multiplier
    step_threshold : int
        Days threshold for step function weighting
    step_later_weight : float
        Weight after the step threshold
    inverse_scale : int
        Days for inverse decay weight to drop to half

    Returns:
    --------
    weighted_mse : float
        The weighted mean squared error
    """
    H0_val, k_val, phi_val = params
    if H0_val < 20 or H0_val > 90:
        return 1e15
    if k_val < 0.01 or k_val > 1.0:
        return 1e15
    if phi_val < 0.001 or phi_val > 0.5:
        return 1e15

    dates, IH = run_with_humidity(H0_val, k_val, phi_val)
    if len(dates) == 0:
        return 1e15

    weighted_mse = compute_weighted_mse(
        dates,
        IH,
        rural_cases_df,
        weight_decay=weight_decay,
        decay_rate=decay_rate,
        initial_weight=initial_weight,
        step_threshold=step_threshold,
        step_later_weight=step_later_weight,
        inverse_scale=inverse_scale,
    )
    return weighted_mse


def objective_function_unweighted(params):
    """
    Standard (unweighted) objective function for comparison.
    """
    H0_val, k_val, phi_val = params
    if H0_val < 20 or H0_val > 90:
        return 1e15
    if k_val < 0.01 or k_val > 1.0:
        return 1e15
    if phi_val < 0.001 or phi_val > 0.5:
        return 1e15

    dates, IH = run_with_humidity(H0_val, k_val, phi_val)
    if len(dates) == 0:
        return 1e15

    model_interp = interp1d(
        pd.to_datetime(dates).astype(np.int64),
        IH,
        kind="linear",
        fill_value="extrapolate",
    )
    obs_dates_int = rural_cases_df["date"].astype(np.int64).values
    model_at_obs = model_interp(obs_dates_int)
    observed = rural_cases_df["active_total"].values
    mse = np.mean((model_at_obs - observed) ** 2)
    return mse


def plot_comparison(
    dates_default, IH_default, dates_opt, IH_opt, title_suffix="", figsize=(16, 10)
):
    """Plot comparison between default and optimized model fits."""
    fig, axes = plt.subplots(3, 1, figsize=figsize)

    # Plot 1: Model vs Observed
    ax1 = axes[0]
    ax1.plot(
        pd.to_datetime(dates_default),
        IH_default,
        "b-",
        linewidth=1.5,
        alpha=0.5,
        label="Default (H0=58, k=0.25, phi=0.05)",
    )
    ax1.plot(pd.to_datetime(dates_opt), IH_opt, "g-", linewidth=2, label="Optimized")
    ax1.plot(
        rural_cases_df["date"],
        rural_cases_df["active_total"],
        "r-",
        linewidth=1.0,
        alpha=0.7,
        label="Observed",
    )
    ax1.set_ylabel("Infectious Humans (I_H)")
    ax1.set_title(f"Model vs Observed - {title_suffix}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Residuals
    ax2 = axes[1]
    model_interp_opt = interp1d(
        pd.to_datetime(dates_opt).astype(np.int64),
        IH_opt,
        kind="linear",
        fill_value="extrapolate",
    )
    model_interp_default = interp1d(
        pd.to_datetime(dates_default).astype(np.int64),
        IH_default,
        kind="linear",
        fill_value="extrapolate",
    )
    obs_dates_int = rural_cases_df["date"].astype(np.int64).values
    residuals_opt = (
        model_interp_opt(obs_dates_int) - rural_cases_df["active_total"].values
    )
    residuals_default = (
        model_interp_default(obs_dates_int) - rural_cases_df["active_total"].values
    )

    ax2.plot(
        rural_cases_df["date"],
        residuals_opt,
        "g-",
        linewidth=1.0,
        alpha=0.7,
        label="Residuals (optimal)",
    )
    ax2.plot(
        rural_cases_df["date"],
        residuals_default,
        "b-",
        linewidth=1.0,
        alpha=0.5,
        label="Residuals (default)",
    )
    ax2.axhline(0, color="k", linestyle="--", linewidth=0.5)
    ax2.set_ylabel("Model - Observed")
    ax2.set_title("Residuals")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Weights visualization
    ax3 = axes[2]
    start_date = rural_cases_df["date"].iloc[0]
    days_from_start = (rural_cases_df["date"] - start_date).dt.days.values.astype(float)

    # Show different weight schemes
    decay_rates = [0.002, 0.005, 0.01]
    colors = ["orange", "red", "purple"]
    for dr, color in zip(decay_rates, colors):
        weights = 5.0 * np.exp(-dr * days_from_start)
        weights = weights / np.sum(weights) * len(weights)
        ax3.plot(
            rural_cases_df["date"],
            weights,
            color=color,
            linewidth=1.5,
            alpha=0.7,
            label=f"Exponential (rate={dr})",
        )

    ax3.set_ylabel("Weight")
    ax3.set_title("Weight Functions Over Time")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return fig
