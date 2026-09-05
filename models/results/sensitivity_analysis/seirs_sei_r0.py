"""Closed-form computation of the basic reproduction number R0 of the SEIRS-SEI model.

This module mirrors the environmental (temperature / rainfall / humidity) response
functions of the SEIRS-SEI model developed in this project
(see ``models/seirs_sei/ode_models/seirs_sei_ode_shared.py``) and evaluates the
semi-analytic reproduction number derived via the Next Generation Matrix method in
``models/seirs_sei/R0_Calculation.ipynb`` and in Section 5.2 of the paper:

    R0 = sqrt( a^2 b1 b2 Lambda* b3M ell / ( N gamma (b3M ell + mu) mu^2 ) )

where the recruitment at the disease free equilibrium is

    Lambda* = mu M* = mu * b_bar * K / (b_bar + mu),

with b_bar the temperature/rainfall-dependent per-capita recruitment and
K = M' * f_T(T) * f_R(R) the environmental carrying capacity.

The module is intentionally self-contained so that it can be imported both from a
stand-alone script and from the sensitivity-analysis notebook without triggering the
heavy data-loading side effects of the calibration modules.
"""

import numpy as np

# --------------------------------------------------------------------------- #
# Baseline parameters (values used in the calibrated model / paper)
# --------------------------------------------------------------------------- #
BASELINE = dict(
    # Thermal response of the biting rate  a(T) = max(0,(T-T')/D1)
    T_prime=16.0,
    D1=24.8,
    # Larval / adult life-history constants entering b(R,T)
    B_E=200.0,
    p_ME=0.9,
    p_ML=0.25,
    p_MP=0.75,
    tau_E=1.0,
    tau_P=1.0,
    c1=0.00554,
    c2=-0.06737,
    R_L=32.67,
    # Mosquito mortality / survival  mu(T,H) = -ln(p_T(T)*p_H(H))
    A=-0.03,
    B=1.31,
    C=-4.4,
    H0=58.0,
    k=0.25,
    phi=0.05,
    # Sporogonic development  tau_M(T) = DD/(T-Tmin)
    DD=105.0,
    Tmin=14.5,
    critical_max_temp_plasm=29.8,
    optimal_temp_plasm=23.5,
    # Anopheles temperature suitability for carrying capacity f_T(T)
    optimal_temp_anop=25.0,
    critical_max_temp_anop=34.0,
    critical_min_temp_anop=16.0,
    # Human epidemiological parameters
    tau_H=10.0,          # intrinsic incubation period (days)
    gamma=1.0 / 120.0,   # recovery rate
    omega=1.0 / 270.0,   # immunity waning rate
    b1=0.04,             # human -> mosquito transmission probability
    b2=0.09,             # mosquito -> human transmission probability
)

# Baseline carrying-capacity scale and population (rural Manaus, 2017)
M_PRIME = 55.99 * 8369.0
N_POP = 8369.0

# Reference environmental state used to evaluate R0 for the sensitivity analysis.
#
# By default we use the mean environmental state over the *transmission season*
# (the days in 2017-2023 on which the rainfall/temperature-dependent recruitment
# sustains R0 > 1). This represents the endemic operating point of the system and
# avoids the pathology of evaluating the bell-shaped rainfall-recruitment response
# b(R,T) at the arithmetic-mean daily rainfall (which is dominated by many dry days
# and drives R0 below the threshold). The values below were computed directly from
# the 2017-2023 daily climate record (see the sensitivity-analysis notebook).
REFERENCE_CLIMATE = dict(
    T=25.57,
    R=7.19,
    H=78.34,
)
# Fraction of study-period days in the transmission season (R0 > 1), for reporting.
TRANSMISSION_SEASON_FRACTION = 1018.0 / 2556.0


# --------------------------------------------------------------------------- #
# Environmental response functions
# --------------------------------------------------------------------------- #
def tau_L(Temp, c1=0.00554, c2=-0.06737):
    denom = c1 * Temp + c2
    return 1.0 / denom if denom > 0 else 100.0


def p_T(Temp, A=-0.03, B=1.31, C=-4.4):
    denominator = np.asarray(A * Temp**2 + B * Temp + C, dtype=float)
    # exp(-1/den) overflows when den -> 0^- ; set those to 0 directly.
    out = np.zeros_like(denominator)
    with np.errstate(divide="ignore", over="ignore"):
        out = np.where(denominator > 0, np.exp(-1.0 / denominator), 0.0)
    return out


def p_H(Humid, H0=58.0, k=0.25, phi=0.05):
    return phi + (1.0 - phi) / (1.0 + np.exp(-k * (Humid - H0)))


def p_surv(Temp, Humid, H0=58.0, k=0.25, phi=0.05, A=-0.03, B=1.31, C=-4.4):
    return p_T(Temp, A, B, C) * p_H(Humid, H0, k, phi)


def mu(Temp, Humid, H0=58.0, k=0.25, phi=0.05, A=-0.03, B=1.31, C=-4.4):
    """Daily mosquito mortality rate (deaths / day)."""
    p_val = np.asarray(p_surv(Temp, Humid, H0, k, phi, A, B, C), dtype=float)
    with np.errstate(divide="ignore"):
        logp = -np.log(p_val)
    return np.where(p_val > 0, logp, 1.0)


def a_bite(Temp, T_prime=16.0, D1=24.8):
    """Temperature-dependent biting rate (bites / mosquito / night)."""
    return np.maximum(0.0, (Temp - T_prime) / D1)


def tau_M_development(Temp, DD=105.0, Tmin=14.5):
    """Sporogonic development time in days."""
    diff = Temp - Tmin
    return np.where(diff > 0, DD / diff, 1000.0)


def b3_briere_scaled_plasm(
    Temp,
    Tmin=14.5,
    critical_max_temp_plasm=29.8,
    optimal_temp_plasm=23.5,
    DD=105.0,
):
    """Briere-type sporogonic development rate b3M(T) (day^-1)."""
    eta = _eta_plasm(Tmin, critical_max_temp_plasm, optimal_temp_plasm, DD)
    Temp = np.asarray(Temp, dtype=float)
    valid = (Temp > Tmin) & (Temp < critical_max_temp_plasm)
    out = np.zeros_like(Temp)
    with np.errstate(invalid="ignore"):
        val = eta * Temp * (Temp - Tmin) * ((critical_max_temp_plasm - Temp) ** 0.5)
    return np.where(valid, val, 0.0)


def _eta_plasm(Tmin, tmax, topt, DD):
    unscaled_peak = topt * (topt - Tmin) * ((tmax - topt) ** 0.5)
    desired_peak = 1.0 / (DD / (topt - Tmin)) if topt > Tmin else 1.0
    return desired_peak / unscaled_peak


def temp_factor_briere_normalized_anop(
    Temp,
    optimal_temp_anop=25.0,
    critical_max_temp_anop=34.0,
    critical_min_temp_anop=16.0,
):
    """Normalized temperature suitability f_T(T) in [0, 1]."""
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


def f_R_rain(Rain, R_L=32.67):
    """Rainfall habitat response f_R(R) (creation x flushing)."""
    return (2.0 * Rain / R_L) * np.exp(1.0 - 2.0 * Rain / R_L)


def b_rate(Rain, Temp, B_E=200.0, p_ME=0.9, p_ML=0.25, p_MP=0.75,
           tau_E=1.0, tau_P=1.0, c1=0.00554, c2=-0.06737, R_L=32.67):
    """Per-capita mosquito recruitment rate b(R,T) (day^-1)."""
    tL = tau_L(Temp, c1, c2)
    if tL <= 0:
        return 0.0

    def tri(R):
        return (4.0 * R * (R_L - R) / R_L**2) if 0 <= R <= R_L else 0.0

    p_ER = tri(Rain) * p_ME
    p_LR = tri(Rain) * p_ML
    p_LT = np.exp(-(c1 * Temp + c2))
    p_PR = tri(Rain) * p_MP
    numer = B_E * p_ER * p_LR * p_LT * p_PR
    denom = tau_E + tL + tau_P
    return numer / denom if denom > 0 else 0.0


# --------------------------------------------------------------------------- #
# R0 evaluation
# --------------------------------------------------------------------------- #
def r0_components(T, R, H, N=N_POP, M_prime=M_PRIME, **p):
    """Evaluate the components of R0 at a single environmental state (T, R, H).

    Returns a dict with the intermediate quantities and R0_H, R0_M and R0.
    """
    a = a_bite(T, p.get("T_prime", BASELINE["T_prime"]),
               p.get("D1", BASELINE["D1"]))
    mu_val = mu(T, H,
                p.get("H0", BASELINE["H0"]), p.get("k", BASELINE["k"]),
                p.get("phi", BASELINE["phi"]),
                p.get("A", BASELINE["A"]), p.get("B", BASELINE["B"]),
                p.get("C", BASELINE["C"]))
    tM = tau_M_development(T, p.get("DD", BASELINE["DD"]),
                           p.get("Tmin", BASELINE["Tmin"]))
    b3 = b3_briere_scaled_plasm(
        T, p.get("Tmin", BASELINE["Tmin"]),
        p.get("critical_max_temp_plasm", BASELINE["critical_max_temp_plasm"]),
        p.get("optimal_temp_plasm", BASELINE["optimal_temp_plasm"]),
        p.get("DD", BASELINE["DD"]))
    ell = np.exp(-mu_val * tM)

    b_bar = b_rate(
        R, T,
        p.get("B_E", BASELINE["B_E"]), p.get("p_ME", BASELINE["p_ME"]),
        p.get("p_ML", BASELINE["p_ML"]), p.get("p_MP", BASELINE["p_MP"]),
        p.get("tau_E", BASELINE["tau_E"]), p.get("tau_P", BASELINE["tau_P"]),
        p.get("c1", BASELINE["c1"]), p.get("c2", BASELINE["c2"]),
        p.get("R_L", BASELINE["R_L"]))

    f_T = temp_factor_briere_normalized_anop(
        T, p.get("optimal_temp_anop", BASELINE["optimal_temp_anop"]),
        p.get("critical_max_temp_anop", BASELINE["critical_max_temp_anop"]),
        p.get("critical_min_temp_anop", BASELINE["critical_min_temp_anop"]))
    f_R = f_R_rain(R, p.get("R_L", BASELINE["R_L"]))
    K = M_prime * f_T * f_R

    if b_bar + mu_val > 0:
        Lambda_star = b_bar * K * mu_val / (b_bar + mu_val)
    else:
        Lambda_star = 0.0

    gamma = p.get("gamma", BASELINE["gamma"])
    b1 = p.get("b1", BASELINE["b1"])
    b2 = p.get("b2", BASELINE["b2"])

    R0_H = a * b2 / gamma if gamma > 0 else 0.0
    denom_M = N * mu_val * (b3 * ell + mu_val) * mu_val
    R0_M = (Lambda_star * a * b1 * b3 * ell / denom_M
            if denom_M > 0 else 0.0)
    R0 = np.sqrt(R0_H * R0_M) if (R0_H > 0 and R0_M > 0) else 0.0

    return dict(
        a=a, mu=mu_val, tau_M=tM, b3M=b3, ell=ell,
        b_bar=b_bar, f_T=f_T, f_R=f_R, K=K, Lambda_star=Lambda_star,
        R0_H=R0_H, R0_M=R0_M, R0=float(R0),
    )


def r0(T, R, H, **p):
    """Return only the scalar R0 at environmental state (T, R, H)."""
    return r0_components(T, R, H, **p)["R0"]


if __name__ == "__main__":
    c = r0_components(
        REFERENCE_CLIMATE["T"], REFERENCE_CLIMATE["R"], REFERENCE_CLIMATE["H"]
    )
    print("R0 at reference mean climate:", c["R0"])
    print("R0_H:", round(c["R0_H"], 3), "R0_M:", round(c["R0_M"], 3))
    print("a:", round(c["a"], 4), "mu:", round(c["mu"], 4),
          "b3M:", round(c["b3M"], 4), "ell:", round(c["ell"], 4))
    print("b_bar:", round(c["b_bar"], 4), "K:", round(c["K"], 1),
          "Lambda*:", round(c["Lambda_star"], 1))
