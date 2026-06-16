import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from scipy.interpolate import griddata
import warnings

warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-whitegrid")

import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_SCRIPT_DIR, "../../../data_files/data")

warmup_climate_data = pd.read_csv(os.path.join(DATA_DIR, "climate_api_data_2012_2015.csv"))
warmup_cases_data = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "sivep_notification_data/treated_malaria_notification_data/cumulative_manaus_cases_2012_2015.csv",
    )
)

climate_data = pd.read_csv(os.path.join(DATA_DIR, "climate_api_data_2016_2024.csv"))
cases_data = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "sivep_notification_data/treated_malaria_notification_data/cumulative_manaus_cases_2016_2023.csv",
    )
)

pop_ibge = pd.read_csv(
    os.path.join(DATA_DIR, "ibge_manaus_rural_population_data_2000_2025.csv")
)
pop_by_year = dict(zip(pop_ibge["year"], pop_ibge["rural_population"]))

defor_data = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "deter_notification_data/treated_deter_deforestation_data_2016_2024.csv",
    )
)
fires_data = pd.read_csv(os.path.join(DATA_DIR, "inpe_fire_counts_data_2016_2024.csv"))

warmup_climate_data["date"] = pd.to_datetime(warmup_climate_data["date"])
warmup_cases_data["date"] = pd.to_datetime(warmup_cases_data["date"])

climate_data["date"] = pd.to_datetime(climate_data["date"])
cases_data["date"] = pd.to_datetime(cases_data["date"])
defor_data["date"] = pd.to_datetime(defor_data["date"])
fires_data["date"] = pd.to_datetime(fires_data["date"])

warmup_start_date = pd.to_datetime("2012-01-01")
warmup_end_date = pd.to_datetime("2015-12-31")
warmup_climate_data = warmup_climate_data[
    (warmup_climate_data["date"] >= warmup_start_date) & (warmup_climate_data["date"] <= warmup_end_date)
].reset_index(drop=True)
warmup_cases_data = warmup_cases_data[
    (warmup_cases_data["date"] >= warmup_start_date) & (warmup_cases_data["date"] <= warmup_end_date)
].reset_index(drop=True)

pre_analysis_start = pd.to_datetime("2012-01-01")
pre_analysis_end = pd.to_datetime("2016-12-31")
pre_analysis_climate_data = pd.concat([warmup_climate_data, climate_data], ignore_index=True)
pre_analysis_cases_data = pd.concat([warmup_cases_data, cases_data], ignore_index=True)
pre_analysis_climate_data = pre_analysis_climate_data[
    (pre_analysis_climate_data["date"] >= pre_analysis_start) & (pre_analysis_climate_data["date"] <= pre_analysis_end)
].reset_index(drop=True)
pre_analysis_cases_data = pre_analysis_cases_data[
    (pre_analysis_cases_data["date"] >= pre_analysis_start) & (pre_analysis_cases_data["date"] <= pre_analysis_end)
].reset_index(drop=True)

climate_data = climate_data[
    (climate_data["date"] >= start_date) & (climate_data["date"] <= end_date)
].reset_index(drop=True)
cases_data = cases_data[
    (cases_data["date"] >= start_date) & (cases_data["date"] <= end_date)
].reset_index(drop=True)

start_date = pd.to_datetime("2017-01-01")
end_date = pd.to_datetime("2023-12-31")
climate_data = climate_data[
    (climate_data["date"] >= start_date) & (climate_data["date"] <= end_date)
].reset_index(drop=True)
cases_data = cases_data[
    (cases_data["date"] >= start_date) & (cases_data["date"] <= end_date)
].reset_index(drop=True)

print("Data loaded and filtered successfully!")
print("Climate data:", len(climate_data), "days")
print("Cases data:", len(cases_data), "days")
print("Deforestation data:", len(defor_data), "days")
print("Forest fires data:", len(fires_data), "days")

print("\nIBGE rural population by year (2017-2023):")
for y in range(2017, 2024):
    print(f"{y}: {pop_by_year[y]:.2f}")

#### Model Parameters
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


#### Helper functions
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


#### E/I ratio estimation
def exposed_to_infected_ratio(Temp, Humid, H0=58.0, k=0.25, phi=0.05):
    return (
        mu(Temp, Humid, H0=58.0, k=0.25, phi=0.05)
        * tau_M(Temp)
        * np.exp(mu(Temp, Humid, H0=58.0, k=0.25, phi=0.05) * tau_M(Temp))
        / (b3_briere_unscaled_plasm(Temp) / unscaled_peak_value_plasm)
    )


warmup_temp_med_0 = warmup_climate_data["temp_med"][0]
warmup_umid_min_0 = warmup_climate_data["umid_min"][0]
# umid_min_0 = climate_data["umid_min"][0]
warmup_initial_exposed_to_infected_ratio = exposed_to_infected_ratio(warmup_temp_med_0, warmup_umid_min_0)
print(
    f"The initial ratio of exposed to infected mosquitos in 2012 was estimated to be ~{round(warmup_initial_exposed_to_infected_ratio)}"
)

temp_med_0 = climate_data["temp_med"][0]
umid_min_0 = climate_data["umid_min"][0]
# umid_min_0 = climate_data["umid_min"][0]
initial_exposed_to_infected_ratio = exposed_to_infected_ratio(temp_med_0, umid_min_0)
print(
    f"The initial ratio of exposed to infected mosquitos was estimated to be ~{round(initial_exposed_to_infected_ratio)}"
)

#### Initial conditions
warmup_rural_cases_df = warmup_cases_data.copy()
cols = [
    "active_total",
    "active_symptomatic",
    "active_asymptomatic",
    "new_cases",
    "cumulative_cases",
    "active_per_new_case",
]
warmup_rural_cases_df[cols] = warmup_rural_cases_df[cols] * (41.54 / 100)
warmup_I_H0 = round(warmup_rural_cases_df["active_total"].iloc[0])
N_2012 = round(pop_by_year[2012])
warmup_E_H0 = round(N_2012 * 0.05)
warmup_R_H0 = round(N_2012 * 0.15)
warmup_S_H0 = N_2012 - warmup_E_H0 - warmup_I_H0 - warmup_R_H0

warmup_M_0 = 10 * N_2012
warmup_I_M0 = round(warmup_M_0 * 0.01)
warmup_E_M0 = round(warmup_I_M0 * 3)
warmup_S_M0 = warmup_M_0 - warmup_E_M0 - warmup_I_M0
initial_state_2012 = np.array([warmup_S_H0, warmup_E_H0, warmup_I_H0, warmup_R_H0, warmup_S_M0, warmup_E_M0, warmup_I_M0])

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
I_H0 = round(rural_cases_df["active_total"].iloc[0])
N_2017 = round(pop_by_year[2017])

E_H0 = round(N_2017 * 0.05)
R_H0 = round(N_2017 * 0.15)
S_H0 = N_2017 - E_H0 - I_H0 - R_H0

M_0 = 10 * N_2017
I_M0 = round(M_0 * 0.01)
E_M0 = round(I_M0 * 3)
S_M0 = M_0 - E_M0 - I_M0
initial_state_2017 = np.array([S_H0, E_H0, I_H0, R_H0, S_M0, E_M0, I_M0])

print(
    f"Initial human compartments in 2017: N={N_2017}, S_H={S_H0}, E_H={E_H0}, I_H={I_H0}, R_H={R_H0}"
)
print(
    f"Initial mosquito compartments in 2017: M={M_0}, S_M={S_M0}, E_M={E_M0}, I_M={I_M0}"
)

M_prime = M_0
M_min = 50000
permanent_factor = 0

# Smoothing window for climate data (days, matching larval development time)
SMOOTH_WINDOW = 14

# Toggle: use smoothed climate data for mosquito demographic rates (mu, tau_M, b_rate)
# to dampen daily oscillations. When False, all rates use current daily values.
USE_SMOOTH_CLIMATE = True

# Add smoothed climate columns for mosquito demographic rates
climate_data["temp_med_smooth"] = (
    climate_data["temp_med"].rolling(SMOOTH_WINDOW, min_periods=1, center=False).mean()
)
climate_data["precip_med_smooth"] = (
    climate_data["precip_med"]
    .rolling(SMOOTH_WINDOW, min_periods=1, center=False)
    .mean()
)
climate_data["umid_min_smooth"] = (
    climate_data["umid_min"].rolling(SMOOTH_WINDOW, min_periods=1, center=False).mean()
)


#### ODE system
def seirs_sei_ode(
    t,
    z,
    N,
    year_climate,
    T_prime,
    B_E,
    p_ME,
    p_ML,
    p_MP,
    tau_E,
    tau_P,
    c1,
    c2,
    D1,
    b1,
    b2,
    A,
    B,
    C,
    DD,
    Tmin,
    optimal_temp_plasm,
    critical_max_temp_plasm,
    optimal_temp_anop,
    critical_max_temp_anop,
    critical_min_temp_anop,
    gamma,
    R_L,
    M_prime,
    tau_H,
    omega,
    M_min,
    permanent_factor,
    use_smooth_climate=USE_SMOOTH_CLIMATE,
    b3_h=None,
    b3_m=None,
):

    day_idx = int(t)
    day_idx = np.clip(day_idx, 0, len(year_climate) - 1)
    T_curr = year_climate.iloc[day_idx]["temp_med"]
    R_curr = year_climate.iloc[day_idx]["precip_med"]
    H_curr = year_climate.iloc[day_idx]["umid_min"]

    if use_smooth_climate:
        T_smooth = year_climate.iloc[day_idx]["temp_med_smooth"]
        R_smooth = year_climate.iloc[day_idx]["precip_med_smooth"]
        H_smooth = year_climate.iloc[day_idx]["umid_min_smooth"]
        T_value = T_smooth
        R_value = R_smooth
        H_value = H_smooth
    else:
        T_value = T_curr
        R_value = R_curr
        H_value = H_curr

    mu_curr = mu(T_value, H_value)
    a_curr = a(T_curr)
    tau_M_curr = tau_M(T_value)

    if b3_m is None:
        b3_m_curr = 1.0 / tau_M_curr if tau_M_curr > 0 else 0.0
    else:
        b3_m_curr = b3_m

    if b3_h is None:
        b3_h_curr = 1.0 / tau_H if tau_H > 0 else 0.0
    else:
        b3_h_curr = b3_h

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

    K_permanent = M_prime * permanent_factor
    K_dynamic = M_prime * temp_factor * rain_factor
    K = max(K_permanent + K_dynamic, M_min)

    total_mosq = S_M + E_M + I_M
    density_factor = max(0, 1 - total_mosq / K) if K > 0 else 0.0
    mosquito_birth = b_curr * density_factor * K

    residual_birth_rate = mu_curr * M_min
    residual_recruitment = residual_birth_rate * max(0, 1 - total_mosq / M_min)
    mosquito_birth = mosquito_birth + residual_recruitment

    dSmdt = mosquito_birth - foi_m * S_M - mu_curr * S_M
    dEmdt = foi_m * S_M - (mu_curr + (b3_briere_curr * l_curr)) * E_M
    dImdt = b3_briere_curr * l_curr * E_M - mu_curr * I_M

    for d in [dShdt, dEhdt, dIhdt, dRhdt, dSmdt, dEmdt, dImdt]:
        d = max(d, -1e6)

    return [dShdt, dEhdt, dIhdt, dRhdt, dSmdt, dEmdt, dImdt]