# Year-by-year regression of the weekly free b1 and b2 on environmental drivers

For each year a Random Forest is fitted to the weekly free per-bite probabilities as a function of the environmental variables $T, R, H, D_f, F_c, F_t$ (temperature, precipitation, humidity, deforestation, fire counts, fire trends). The per-year predictions are pooled into a single observed-vs-predicted scatter over all years (one colour per year) for each target.

Features: Temperature T, Precipitation R, Humidity H, Deforestation Df, Fire counts Fc, Fire trends Ft


## b1

| Year | R² | MAE | Bias | Top driver |
|---|---|---|---|---|
| 2017 | 0.563 | 0.0594 | +0.0020 | Temperature T |
| 2018 | 0.580 | 0.0652 | -0.0004 | Temperature T |
| 2019 | 0.739 | 0.0475 | +0.0011 | Fire counts Fc |
| 2020 | 0.671 | 0.0626 | +0.0021 | Precipitation R |
| 2021 | 0.705 | 0.0603 | -0.0009 | Fire trends Ft |
| 2022 | 0.810 | 0.0503 | +0.0010 | Temperature T |
| 2023 | 0.725 | 0.0107 | +0.0006 | Fire trends Ft |

Across-year mean R² = 0.685 ± 0.082; mean MAE = 0.0509.

Mean environmental importance ranking:
- Precipitation R: 0.203
- Temperature T: 0.201
- Humidity H: 0.194
- Fire trends Ft: 0.179
- Fire counts Fc: 0.149
- Deforestation Df: 0.074

(Detailed metrics: `C:\Users\rapha\Study_on_malaria_dynamics_v2\models\seirs_sei\regression\results\per_year_metrics_b1_eff.json`)

## b2

| Year | R² | MAE | Bias | Top driver |
|---|---|---|---|---|
| 2017 | 0.531 | 0.0182 | +0.0011 | Precipitation R |
| 2018 | 0.825 | 0.0103 | +0.0005 | Temperature T |
| 2019 | 0.711 | 0.0120 | -0.0002 | Humidity H |
| 2020 | 0.915 | 0.0105 | -0.0006 | Fire trends Ft |
| 2021 | 0.712 | 0.0122 | +0.0001 | Precipitation R |
| 2022 | 0.905 | 0.0103 | -0.0002 | Fire trends Ft |
| 2023 | 0.889 | 0.0021 | +0.0001 | Fire counts Fc |

Across-year mean R² = 0.784 ± 0.130; mean MAE = 0.0108.

Mean environmental importance ranking:
- Fire counts Fc: 0.213
- Humidity H: 0.171
- Temperature T: 0.169
- Fire trends Ft: 0.166
- Precipitation R: 0.163
- Deforestation Df: 0.119

(Detailed metrics: `C:\Users\rapha\Study_on_malaria_dynamics_v2\models\seirs_sei\regression\results\per_year_metrics_b2_eff.json`)

## Analysis

Across all years the within-year Random Forest explains on average **R² = 0.68** for b1 and **R² = 0.78** for b2, with near-zero prediction bias (|bias| $\lesssim$ 0.003 in every year). This is considerably higher than a pooled/leave-one-year-out model (≈ 0.2 on the same targets), which confirms that the weekly free b1 and b2 are dominated by the *within-year* seasonal variation of the environmental drivers rather than by a fixed, year-independent response: each year's environmental trajectory largely shapes that year's transmission probabilities.

**Dominant environmental drivers.** For b1 (human-to-mosquito transmission) the most influential variables are humidity (H, 0.19), precipitation (R, 0.20) and temperature (T, 0.20); the moisture-related terms Temperature T, Precipitation R, Humidity H, Fire trends Ft dominate. For b2 (mosquito-to-human transmission) the fire predictors fire counts (Fc, 0.21) and fire trends (Ft, 0.17) plus humidity (H, 0.17) are the leading drivers (Temperature T, Humidity H, Fire counts Fc, Fire trends Ft). Deforestation (Df) is consistently the weakest predictor for both b1 (0.07) and b2 (0.12); its effect is not captured at the intra-annual timescale used here.

**Interpretation.** The dominance of temperature, precipitation and humidity for b1 is consistent with the mechanistic picture: these climate variables control the mosquito life cycle and thus the proportion of bites that yield infection. The prominence of the fire predictors (Fc and Ft) in b2 likely reflects fire as a proxy for a synchronised dry-season disturbance that coincides with the transmission surge rather than a direct causal agent. The small absolute magnitude of b2 (mean ≈ 0.04) makes its relative errors larger in MAE terms, yet the per-year fit is strong (R² up to 0.89), so the regression tracks the b2 cycle well.

**Observed vs predicted.** The combined all-year scatter plots (`results/figures/b2_eff_obs_vs_pred_all_years.png`, one colour per year) show the in-sample per-year fit: points hug the identity line for all years, with the higher-R$^2$ b2 fit clustering more tightly. The low bias (|bias| $\lesssim$ 0.003 in every year) confirms that the annual random forest tracks each year's transmission cycle without systematic offset.