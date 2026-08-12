import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import os
import json
import warnings

import lmfit

warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-whitegrid")

# Path to data directory (repo root, three levels up from lmfit_optimization/)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_SCRIPT_DIR, "../../../data_files/data")
RESULTS_FILE = os.path.join(_SCRIPT_DIR, "lmfit_results.json")

# ---------------------------------------------------------------------------
# Data loading (analysis period only: 2017-2023)
# ---------------------------------------------------------------------------
SMOOTH_WINDOW = 14

climate_2016_2024 = pd.read_csv(
    os.path.join(DATA_DIR, "climate_api_data_2016_2024.csv")
)
climate_2016_2024["date"] = pd.to_datetime(climate_2016_2024["date"])

start_date = pd.to_datetime("2017-01-01")
end_date = pd.to_datetime("2023-12-31")
climate_data = climate_2016_2024[
    (climate_2016_2024["date"] >= start_date) & (climate_2016_2024["date"] <= end_date)
].reset_index(drop=True)

cases_data = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "sivep_notification_data/treated_malaria_notification_data/cumulative_manaus_cases_2016_2023.csv",
    )
)
cases_data["date"] = pd.to_datetime(cases_data["date"])
cases_data = cases_data[
    (cases_data["date"] >= start_date) & (cases_data["date"] <= end_date)
].reset_index(drop=True)

pop_ibge = pd.read_csv(
    os.path.join(DATA_DIR, "ibge_manaus_rural_population_data_2000_2025.csv")
)
pop_by_year = dict(zip(pop_ibge["year"], pop_ibge["rural_population"]))

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

print(f"Climate data 2017-2023: {len(climate_data)} days")
print(f"Cases data 2017-2023: {len(rural_cases_df)} rows")
print(f"lmfit version: {lmfit.__version__}")

# ---------------------------------------------------------------------------
# Model Parameters
# ---------------------------------------------------------------------------
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

M_min = 50000


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Population, fixed carrying capacity and default parameter set
# ---------------------------------------------------------------------------
N_2017 = round(pop_by_year[2017])
M_PRIME_FIXED = 40 * N_2017

# Fixed 2017 initial conditions (ode_models proportions): E_H = 5% N, R_H = 15%
# N, I_H from notification data, I_M = 2% M_0, E_M = 3 x I_M0.
I_H0_DEF = round(rural_cases_df["active_total"].iloc[0])
E_H0_DEF = round(N_2017 * 0.05)
R_H0_DEF = round(N_2017 * 0.15)
S_H0_DEF = N_2017 - E_H0_DEF - I_H0_DEF - R_H0_DEF

M_0 = 10 * N_2017
I_M0_DEF = round(M_0 * 0.02)
E_M0_DEF = round(I_M0_DEF * 3)
S_M0_DEF = M_0 - E_M0_DEF - I_M0_DEF
initial_state_2017 = np.array(
    [S_H0_DEF, E_H0_DEF, I_H0_DEF, R_H0_DEF, S_M0_DEF, E_M0_DEF, I_M0_DEF]
)

# Default IC fractions (used as starting guesses for the 2017 IC fit)
S_H0_FRAC_DEF = S_H0_DEF / N_2017
E_H0_FRAC_DEF = E_H0_DEF / N_2017
I_M0_RATIO_DEF = I_M0_DEF / M_0

DEFAULT_PARAMS = dict(
    H0=58.0,
    k=0.25,
    phi=0.05,
    tau_H=10.0,
    gamma=1 / 120,
    omega=1 / 270,
    b1=0.04,
    b2=0.09,
    M_prime=M_PRIME_FIXED,
)

print(
    f"\n2017 direct start (no warm-up): N={N_2017}, M'={M_PRIME_FIXED} ({M_PRIME_FIXED / N_2017:.2f} x N)"
)
print(
    f"Initial human compartments in 2017: N={N_2017}, S_H={S_H0_DEF}, E_H={E_H0_DEF}, I_H={I_H0_DEF}, R_H={R_H0_DEF}"
)
print(
    f"Initial mosquito compartments in 2017: M={M_0}, S_M={S_M0_DEF}, E_M={E_M0_DEF}, I_M={I_M0_DEF}"
)


# ---------------------------------------------------------------------------
# Per-year helpers
# ---------------------------------------------------------------------------
def observed_for_year(year):
    """Observed rural cases (active_total) for a single year (2017-2023)."""
    return rural_cases_df[rural_cases_df["date"].dt.year == year].reset_index(drop=True)


def observed_IH_start(year):
    """First observed active_total (rural, rounded) for a year.

    Used as the I_H initial condition of that year instead of the previous
    year's simulated end-state."""
    return round(observed_for_year(year)["active_total"].iloc[0])


def build_initial_state_2017(
    S_H0_frac=S_H0_FRAC_DEF,
    E_H0_frac=E_H0_FRAC_DEF,
    I_M0_ratio=I_M0_RATIO_DEF,
):
    """Build the 2017 initial state from fitted fractions.

    I_H is fixed from the first observed data point; R_H is the residual of N
    (S + E + I). Returns None if any of S_H, E_H, R_H is negative."""
    N = N_2017
    I_H0v = round(rural_cases_df["active_total"].iloc[0])
    S_H0v = round(N * S_H0_frac)
    E_H0v = round(N * E_H0_frac)
    R_H0v = N - S_H0v - E_H0v - I_H0v
    if S_H0v < 0 or E_H0v < 0 or R_H0v < 0:
        return None
    M0 = 10 * N
    I_M0v = round(M0 * I_M0_ratio)
    E_M0v = round(I_M0v * 3)
    S_M0v = M0 - E_M0v - I_M0v
    return np.array([S_H0v, E_H0v, I_H0v, R_H0v, S_M0v, E_M0v, I_M0v])


# ---------------------------------------------------------------------------
# Single-year simulation
# ---------------------------------------------------------------------------
def simulate_year(year, state0, p, beta_h_weekly=None, beta_m_weekly=None):
    """Simulate one year (2017-2023) with parameter dict `p`.

    p keys: H0, k, phi, tau_H, gamma, omega, b1, b2, M_prime.
    Returns (dates, IH, end_state) or None if integration fails.

    If `beta_h_weekly`/`beta_m_weekly` are given (arrays of per-week effective
    transmission coefficients), they replace the climate-driven FoI terms
    `a(T)*b2` and `a(T)*b1` with a piecewise-constant week profile:
        foi_h = beta_h_weekly[w] * I_M / N
        foi_m = beta_m_weekly[w] * I_H / N
    Both must have the same length (one value per 7-day bucket, last bucket
    short). When they are None (default), the original climate-driven FoI is
    used and the fitted b1/b2 scalars apply."""
    N = round(pop_by_year[year])
    H0v, kv, phiv = p["H0"], p["k"], p["phi"]
    tau_Hv, gammav, omegav = p["tau_H"], p["gamma"], p["omega"]
    M_prime_v = p["M_prime"]
    use_weekly_beta = beta_h_weekly is not None and beta_m_weekly is not None
    if use_weekly_beta:
        b1v = b2v = None
    else:
        b1v, b2v = p["b1"], p["b2"]

    start = f"{year}-01-01"
    end = f"{year}-12-31"
    clim = climate_data[
        (climate_data["date"] >= start) & (climate_data["date"] <= end)
    ].reset_index(drop=True)
    clim["temp_med_smooth"] = (
        clim["temp_med"].rolling(SMOOTH_WINDOW, min_periods=1, center=False).mean()
    )
    clim["precip_med_smooth"] = (
        clim["precip_med"].rolling(SMOOTH_WINDOW, min_periods=1, center=False).mean()
    )
    clim["umid_min_smooth"] = (
        clim["umid_min"].rolling(SMOOTH_WINDOW, min_periods=1, center=False).mean()
    )

    def custom_p_T(Temp, Humid):
        denominator = A * Temp**2 + B * Temp + C
        p_T_temp = np.where(denominator <= 0, 0.0, np.exp(-1 / denominator))
        p_H = phiv + (1.0 - phiv) / (1.0 + np.exp(-kv * (Humid - H0v)))
        return p_T_temp * p_H

    def custom_mu(Temp, Humid):
        p_val = custom_p_T(Temp, Humid)
        return -np.log(p_val) if p_val > 0 else 1.0

    temp_med = clim["temp_med"].values
    precip_med = clim["precip_med"].values
    temp_smooth = clim["temp_med_smooth"].values
    precip_smooth = clim["precip_med_smooth"].values
    umid_smooth = clim["umid_min_smooth"].values
    n_days = len(clim)

    def ode_func(t, z):
        day_idx = int(t)
        day_idx = day_idx if day_idx < n_days else n_days - 1
        T_curr = temp_med[day_idx]
        R_curr = precip_med[day_idx]
        T_value = temp_smooth[day_idx]
        R_value = precip_smooth[day_idx]
        H_value = umid_smooth[day_idx]

        mu_curr = custom_mu(T_value, H_value)
        a_curr = a(T_curr)
        tau_M_curr = tau_M(T_value)
        b3_m_curr = 1.0 / tau_M_curr if tau_M_curr > 0 else 0.0
        b3_h_curr = 1.0 / tau_Hv if tau_Hv > 0 else 0.0
        l_curr = np.exp(-mu_curr * tau_M_curr)
        b_curr = b_rate(R_value, T_value)
        b3_briere_curr = b3_briere_scaled_plasm(T_curr)

        S_H, E_H, I_H, R_H, S_M, E_M, I_M = z
        if use_weekly_beta:
            week_idx = min(int(day_idx // 7), len(beta_h_weekly) - 1)
            foi_h = beta_h_weekly[week_idx] * (I_M / N)
            foi_m = beta_m_weekly[week_idx] * (I_H / N)
        else:
            foi_h = a_curr * b2v * (I_M / N)
            foi_m = a_curr * b1v * (I_H / N)

        dShdt = -foi_h * S_H + omegav * R_H
        dEhdt = foi_h * S_H - b3_h_curr * E_H
        dIhdt = b3_h_curr * E_H - gammav * I_H
        dRhdt = gammav * I_H - omegav * R_H

        temp_factor = temp_factor_briere_normalized_anop(T_curr)
        habitat_creating_factor = 2 * R_curr / R_L
        habitat_flushing_factor = np.exp(1 - (2 * R_curr / R_L))
        rain_factor = habitat_creating_factor * habitat_flushing_factor

        K = max(M_prime_v * temp_factor * rain_factor, M_min)
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

    num_days = len(clim)
    sol = solve_ivp(
        ode_func,
        [0, num_days],
        state0,
        t_eval=np.linspace(0, num_days, num_days * 10),
        method="LSODA",
    )
    if not sol.success:
        return None
    t_interp = np.linspace(0, sol.t[-1], len(clim))
    IH_interp = interp1d(sol.t, sol.y[2])(t_interp)
    end_state = np.maximum(sol.y[:, -1].copy(), 0)
    return clim["date"].values, IH_interp, end_state


# ---------------------------------------------------------------------------
# Weighted objective
# ---------------------------------------------------------------------------
DEFAULT_WEIGHT_KW = dict(
    weight_decay="exponential",
    decay_rate=0.0015,
    initial_weight=5.0,
    step_threshold=365,
    step_later_weight=1.0,
    inverse_scale=365,
)


def weighted_mse_for_year(dates, IH, obs, weight_kw=None):
    wkw = DEFAULT_WEIGHT_KW if weight_kw is None else weight_kw
    return compute_weighted_mse(dates, IH, obs, **wkw)


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
    """Weighted MSE giving more weight to early time periods (within a year)."""
    model_interp = interp1d(
        pd.to_datetime(dates).astype(np.int64),
        IH,
        kind="linear",
        fill_value="extrapolate",
    )
    obs_dates_int = rural_cases_df["date"].astype(np.int64).values
    model_at_obs = model_interp(obs_dates_int)
    observed = rural_cases_df["active_total"].values

    start_dt = rural_cases_df["date"].iloc[0]
    days_from_start = (rural_cases_df["date"] - start_dt).dt.days.values.astype(float)

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

    return np.sum(weights * (model_at_obs - observed) ** 2) / np.sum(weights)


# ---------------------------------------------------------------------------
# lmfit differential-evolution fitting
# ---------------------------------------------------------------------------
TOPIC_BOUNDS = {
    "humidity": {"H0": (20, 90), "k": (0.01, 1.0), "phi": (0.001, 0.5)},
    "human": {"tau_H": (10, 20), "gamma": (1/150, 1.0), "omega": (1/300, 0.1)},
    "foi": {"b1": (0.001, 0.5), "b2": (0.01, 0.5)},
}
IC_BOUNDS = {
    "S_H0_frac": (0.3, 0.9),
    "E_H0_frac": (0.001, 0.30),
    "I_M0_ratio": (0.0005, 0.15),
}

DEFAULT_DE_SETTINGS = dict(seed=42, popsize=6, max_nfev=800, tol=0.01, polish=False)


def m_prime_bounds(year):
    N = round(pop_by_year[year])
    return {"M_prime": (M_min, 500 * N)}


def _make_objective(year, state0, obs, topic_keys, carried, weight_kw):
    def objective(params):
        p = carried.copy()
        for key in topic_keys:
            p[key] = params[key].value
        res = simulate_year(year, state0, p)
        if res is None:
            return 1e15
        dates, IH, _ = res
        if len(IH) == 0 or not np.all(np.isfinite(IH)):
            return 1e15
        return weighted_mse_for_year(dates, IH, obs, weight_kw)

    return objective


def _ic_penalty(params, base_penalty=1e15):
    """Graded penalty for infeasible 2017 IC fractions (R_H < 0).

    R_H0 = N - S - E - I must stay >= 0, so S_frac + E_frac must not exceed
    1 - I_H0/N. A flat 1e15 lets DE "converge" onto the infeasible plateau;
    a graded penalty steers it back toward the feasible region."""
    n = N_2017
    i_h = round(rural_cases_df["active_total"].iloc[0])
    s = params["S_H0_frac"].value
    e = params["E_H0_frac"].value
    rem = (n - i_h) / n
    viol = max(0.0, (s + e) - rem) + max(0.0, -s) + max(0.0, -e)
    return base_penalty + viol * 1e14


def _make_objective_ic2017(obs, carried, weight_kw):
    def objective(params):
        state0 = build_initial_state_2017(
            params["S_H0_frac"].value,
            params["E_H0_frac"].value,
            params["I_M0_ratio"].value,
        )
        if state0 is None:
            return _ic_penalty(params)
        res = simulate_year(2017, state0, carried)
        if res is None:
            return 1e15
        dates, IH, _ = res
        if len(IH) == 0 or not np.all(np.isfinite(IH)):
            return 1e15
        return weighted_mse_for_year(dates, IH, obs, weight_kw)

    return objective


def _lmfit_params(bounds, start):
    params = lmfit.Parameters()
    for name, (lo, hi) in bounds.items():
        params.add(name, value=float(start[name]), min=lo, max=hi, vary=True)
    return params


def _fit(bounds, start, objective, de_settings, method="differential_evolution"):
    params = _lmfit_params(bounds, start)
    kwargs = dict(de_settings)
    if method == "nelder":
        for k in ("seed", "popsize", "tol", "polish"):
            kwargs.pop(k, None)
        kwargs.setdefault("max_nfev", 2000)
    result = lmfit.minimize(objective, params, method=method, **kwargs)
    fitted = {name: float(result.params[name].value) for name in bounds}
    return result, fitted


def _summarize(result, fitted):
    s = {name: float(val) for name, val in fitted.items()}
    if result.chisqr is not None:
        s["chisqr"] = float(result.chisqr)
        s["wMSE"] = float(np.sqrt(result.chisqr))
    s["nfev"] = int(getattr(result, "nfev", -1))
    s["success"] = bool(getattr(result, "success", False))
    return s


def fit_ic_2017(obs, carried, de_settings=None, weight_kw=None, method="nelder"):
    de_settings = DEFAULT_DE_SETTINGS if de_settings is None else de_settings
    start = dict(
        S_H0_frac=carried.get("S_H0_frac", S_H0_FRAC_DEF),
        E_H0_frac=carried.get("E_H0_frac", E_H0_FRAC_DEF),
        I_M0_ratio=carried.get("I_M0_ratio", I_M0_RATIO_DEF),
    )
    objective = _make_objective_ic2017(obs, carried, weight_kw)
    result, fitted = _fit(IC_BOUNDS, start, objective, de_settings, method=method)
    return result, fitted


def fit_topic(topic, year, state0, obs, carried, de_settings=None, weight_kw=None):
    de_settings = DEFAULT_DE_SETTINGS if de_settings is None else de_settings
    if topic == "m_prime":
        bounds = m_prime_bounds(year)
    else:
        bounds = TOPIC_BOUNDS[topic]
    start = {name: carried[name] for name in bounds}
    objective = _make_objective(year, state0, obs, list(bounds), carried, weight_kw)
    result, fitted = _fit(bounds, start, objective, de_settings)
    return result, fitted


def fit_year_joint(
    year,
    state0,
    obs,
    carried,
    include_ic,
    de_settings=None,
    weight_kw=None,
    method="nelder",
):
    """Joint refinement: fit all free parameters of a year at once.

    For 2017 this includes the IC fractions (IC was skipped for 2018+), plus all
    topic parameters (H0,k,phi,tau_H,gamma,omega,b1,b2,M_prime). Seeded from the
    sequential solution `carried`; `method` defaults to a local Nelder-Mead
    polish, which is stable from a good seed (differential_evolution on 12
    parameters is prone to landing in worse basins at modest budgets).
    Returns (result, fitted, bounds)."""
    de_settings = DEFAULT_DE_SETTINGS if de_settings is None else de_settings
    bounds = {}
    if include_ic:
        bounds.update(IC_BOUNDS)
    bounds.update(TOPIC_BOUNDS["humidity"])
    bounds.update(TOPIC_BOUNDS["human"])
    bounds.update(TOPIC_BOUNDS["foi"])
    bounds.update(m_prime_bounds(year))
    start = {name: carried[name] for name in bounds}

    def objective(params):
        p = carried.copy()
        for key in bounds:
            p[key] = params[key].value
        if include_ic:
            st = build_initial_state_2017(
                params["S_H0_frac"].value,
                params["E_H0_frac"].value,
                params["I_M0_ratio"].value,
            )
            if st is None:
                return _ic_penalty(params)
        else:
            st = state0
        res = simulate_year(year, st, p)
        if res is None:
            return 1e15
        dates, IH, _ = res
        if len(IH) == 0 or not np.all(np.isfinite(IH)):
            return 1e15
        return weighted_mse_for_year(dates, IH, obs, weight_kw)

    result, fitted = _fit(bounds, start, objective, de_settings, method=method)
    return result, fitted, bounds


# ---------------------------------------------------------------------------
# Yearly sequential fitting orchestration
# ---------------------------------------------------------------------------
def run_yearly_fit(
    years=None,
    de_settings=None,
    weight_kw=None,
    save_json=True,
    results_file=None,
    resume=False,
    refine=True,
    refine_method="nelder",
    refine_de_settings=None,
    verbose=True,
):
    """Fit topics sequentially for each year (2017-2023) and carry state forward.

    Per year: [2017 only: IC] -> humidity (H0,k,phi) -> human (tau_H,gamma,omega)
    -> FoI (b1,b2) -> M_prime. Each topic fixes the others at the best values
    found so far (carried from the previous year as starting guesses). The year
    is then simulated with the fully fitted parameters and its end-state is
    carried into the next year.

    The I_H component of each year's initial condition is always reset to the
    first observed `active_total` of that year (data), instead of the previous
    year's simulated end-state I_H. The remaining compartments (S_H, E_H, R_H,
    and the mosquito compartments) are carried forward from the previous year.

    If `refine=True` (default), each year is followed by a joint fit over all of
    that year's free parameters (IC fractions included for 2017), seeded from
    the sequential solution, to undo sub-optimality caused by the sequential
    order. `refine_method` defaults to a local Nelder-Mead polish (stable from a
    good seed; differential_evolution is available via `refine_method="differential_evolution"`).
    The refined parameters replace the sequential ones in the reported per-year
    params/trajectory (also stored under `refine`).

    Returns (results, trajectories) and saves results to `lmfit_results.json`."""
    years = list(range(2017, 2024)) if years is None else list(years)
    de_settings = DEFAULT_DE_SETTINGS if de_settings is None else de_settings
    refine_de_settings = (
        de_settings if refine_de_settings is None else refine_de_settings
    )
    if results_file is None:
        results_file = RESULTS_FILE
    wkw = DEFAULT_WEIGHT_KW if weight_kw is None else weight_kw

    if resume and os.path.exists(results_file):
        with open(results_file, "r") as f:
            saved = json.load(f)
        if verbose:
            print(f"Resumed: loaded existing results from {results_file}")
        return saved.get("per_year", {}), {}

    carried = dict(DEFAULT_PARAMS)
    state0 = initial_state_2017.copy()
    results = {}
    trajectories = {}

    for year in years:
        obs = observed_for_year(year)
        if verbose:
            print(f"\n=== Fitting year {year} ===")
        if year == 2017:
            ic_settings = dict(de_settings)
            ic_settings["max_nfev"] = 300
            result, fitted = fit_ic_2017(obs, carried, ic_settings, wkw)
            state0 = build_initial_state_2017(
                fitted["S_H0_frac"], fitted["E_H0_frac"], fitted["I_M0_ratio"]
            )
            if state0 is None:
                fitted = dict(
                    S_H0_frac=S_H0_FRAC_DEF,
                    E_H0_frac=E_H0_FRAC_DEF,
                    I_M0_ratio=I_M0_RATIO_DEF,
                )
                state0 = initial_state_2017.copy()
                if verbose:
                    print("  IC: fitted IC infeasible -> using default 2017 IC")
            carried.update(fitted)
            results.setdefault(str(year), {})["IC"] = _summarize(result, fitted)
            if verbose:
                print(
                    "IC: "
                    + ", ".join(f"{k}={v:.4f}" for k, v in fitted.items())
                    + f"  (chisqr={result.chisqr:.4g}, nfev={result.nfev})"
                )
        # I_H of the year starts from the first observed active_total (data),
        # not from the previous year's simulated end-state. Other compartments
        # keep the carried state.
        state0 = state0.copy()
        state0[2] = observed_IH_start(year)
        for topic in ["humidity", "human", "foi", "m_prime"]:
            result, fitted = fit_topic(
                topic, year, state0, obs, carried, de_settings, wkw
            )
            carried.update(fitted)
            results.setdefault(str(year), {})[topic] = _summarize(result, fitted)
            if verbose:
                print(
                    f"  {topic}: "
                    + ", ".join(f"{k}={v:.4g}" for k, v in fitted.items())
                    + f"  (chisqr={result.chisqr:.4g}, nfev={result.nfev})"
                )
        dates, IH, end_state = simulate_year(year, state0, carried)
        trajectories[year] = (dates, IH)
        results[str(year)]["weighted_mse"] = float(
            weighted_mse_for_year(dates, IH, obs, wkw)
        )
        results[str(year)]["params"] = {k: float(v) for k, v in carried.items()}
        results[str(year)]["end_state"] = end_state.tolist()
        if refine:
            result, fitted, bounds = fit_year_joint(
                year,
                state0,
                obs,
                carried,
                include_ic=(year == 2017),
                de_settings=refine_de_settings,
                weight_kw=wkw,
                method=refine_method,
            )
            st_refine = state0
            if year == 2017:
                st_refine = build_initial_state_2017(
                    fitted["S_H0_frac"],
                    fitted["E_H0_frac"],
                    fitted["I_M0_ratio"],
                )
                if st_refine is None:
                    st_refine = state0
                    for k in ("S_H0_frac", "E_H0_frac", "I_M0_ratio"):
                        fitted.pop(k, None)
                    if verbose:
                        print("  refine: refined IC infeasible -> kept sequential IC")
            carried.update(fitted)
            s = _summarize(result, fitted)
            results[str(year)]["refine"] = s
            dates, IH, end_state = simulate_year(year, st_refine, carried)
            trajectories[year] = (dates, IH)
            results[str(year)]["weighted_mse"] = float(
                weighted_mse_for_year(dates, IH, obs, wkw)
            )
            results[str(year)]["params"] = {k: float(v) for k, v in carried.items()}
            results[str(year)]["end_state"] = end_state.tolist()
            if verbose:
                print(
                    f"  refine ({len(bounds)} params): "
                    + ", ".join(f"{k}={v:.4g}" for k, v in fitted.items())
                    + f"  (chisqr={result.chisqr:.4g}, nfev={result.nfev})"
                )
        state0 = end_state.copy()
        if verbose:
            print(
                f"  -> year {year}: weighted MSE = {results[str(year)]['weighted_mse']:.4f}"
            )

    if save_json:
        payload = {
            "years": years,
            "method": "lmfit differential_evolution",
            "de_settings": {k: v for k, v in de_settings.items()},
            "weight_kw": {k: v for k, v in wkw.items()},
            "per_year": results,
        }
        with open(results_file, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        if verbose:
            print(f"\nSaved results to {results_file}")

    return results, trajectories


# ---------------------------------------------------------------------------
# Baseline (default params, fixed ode_models ICs)
# ---------------------------------------------------------------------------
def run_baseline(years=None, weight_kw=None):
    """Simulate the full period with default params and fixed 2017 ICs."""
    wkw = DEFAULT_WEIGHT_KW if weight_kw is None else weight_kw
    years = list(range(2017, 2024)) if years is None else list(years)
    state0 = initial_state_2017.copy()
    p = dict(DEFAULT_PARAMS)
    traj = {}
    mse = {}
    for year in years:
        res = simulate_year(year, state0, p)
        if res is None:
            traj[year] = None
            mse[year] = None
            continue
        dates, IH, end_state = res
        traj[year] = (dates, IH)
        mse[year] = weighted_mse_for_year(dates, IH, observed_for_year(year), wkw)
        state0 = end_state
    return traj, mse


def load_results(results_file=None):
    results_file = RESULTS_FILE if results_file is None else results_file
    if not os.path.exists(results_file):
        return None
    with open(results_file, "r") as f:
        return json.load(f)
