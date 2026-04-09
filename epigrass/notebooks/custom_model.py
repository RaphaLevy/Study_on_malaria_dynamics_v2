"""
SEIRS-SEI Custom Model for Epigrass
Malaria model with:
- Human compartment: SEIRS (Susceptible-Exposed-Infectious-Recovered-Susceptible)
- Vector compartment: SEI (Susceptible-Exposed-Infectious)
- Climate integration for Amazon region (Manaus)
"""

import numpy as np

vnames = ["Sh", "Eh", "Ih", "Rh", "Sm", "Em", "Im"]


def Model(
    inits, simstep, totpop, theta=0, npass=0, bi=None, bp=None, values=None, model=None
):
    """
    SEIRS-SEI model for malaria transmission

    Human: SEIRS (S->E->I->R->S with waning immunity)
    Vector: SEI (S->E->I)

    Parameters from bp (site parameters):
    - beta: transmission rate from mosquito to human
    - alpha: non-linear exponent (mass action = 1)
    - sigma: human incubation rate (1/latent period)
    - gamma: recovery rate (1/infectious period)
    - omega: immunity loss rate (waning immunity)
    - mu_m: mosquito mortality rate
    - a: mosquito biting rate
    - b1: probability infection from human to mosquito
    - b2: probability infection from mosquito to human

    Climate parameters (from climate_data):
    - temperature affects: a, mu_m, incubation periods
    - precipitation affects: mosquito carrying capacity
    """

    if simstep == 0:
        Sh = bi.get("sh", bi.get(b"sh", totpop * 0.99))
        Eh = bi.get("eh", bi.get(b"eh", 0))
        Ih = bi.get("ih", bi.get(b"ih", totpop * 0.01))
        Rh = bi.get("rh", bi.get(b"rh", 0))
        Sm = bi.get("sm", bi.get(b"sm", 100))
        Em = bi.get("em", bi.get(b"em", 0))
        Im = bi.get("im", bi.get(b"im", 0))
    else:
        Sh, Eh, Ih, Rh, Sm, Em, Im = inits

    N = totpop

    beta = bp.get("beta", bp.get(b"beta", 0.001))
    alpha = bp.get("alpha", bp.get(b"alpha", 1.0))
    sigma = bp.get("sigma", bp.get(b"sigma", 1 / 12))  # 12 day incubation
    gamma = bp.get("gamma", bp.get(b"gamma", 1 / 180))  # 180 day infectious
    omega = bp.get("omega", bp.get(b"omega", 1 / 365))  # 1 year immunity
    mu_m = bp.get("mu_m", bp.get(b"mu_m", 1 / 30))  # 30 day lifespan
    a = bp.get("a", bp.get(b"a", 0.5))  # biting rate
    b1 = bp.get("b1", bp.get(b"b1", 0.5))  # prob human->mosquito
    b2 = bp.get("b2", bp.get(b"b2", 0.5))  # prob mosquito->human

    foi_human = a * b2 * (Im / N) ** alpha

    new_exposures_h = foi_human * Sh
    new_infections_h = sigma * Eh
    recoveries = gamma * Ih
    immunity_loss = omega * Rh

    dSh = -new_exposures_h + immunity_loss
    dEh = new_exposures_h - new_infections_h
    dIh = new_infections_h - recoveries
    dRh = recoveries - immunity_loss

    foi_mosquito = a * b1 * (Ih / N) ** alpha

    new_exposures_m = foi_mosquito * Sm
    new_infections_m = sigma * Em

    dSm = -new_exposures_m - mu_m * Sm
    dEm = new_exposures_m - new_infections_m - mu_m * Em
    dIm = new_infections_m - mu_m * Im

    Sh_pos = max(0, Sh + dSh)
    Eh_pos = max(0, Eh + dEh)
    Ih_pos = max(0, Ih + dIh)
    Rh_pos = max(0, Rh + dRh)
    Sm_pos = max(0, Sm + dSm)
    Em_pos = max(0, Em + dEm)
    Im_pos = max(0, Im + dIm)

    migrating_infected = Ih_pos

    return (
        [Eh_pos, Ih_pos, Sh_pos, Rh_pos, Em_pos, Im_pos, Sm_pos],
        new_exposures_h,
        migrating_infected,
    )
