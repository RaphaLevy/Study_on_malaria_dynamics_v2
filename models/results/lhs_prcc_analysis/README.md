# Robustness Analysis of the SEIRS-SEI Malaria Model — LHS + PRCC

This folder supplies a **Latin-Hypercube Sampling + Partial Rank Correlation
Coefficient (LHS + PRCC)** robustness analysis of the SEIRS-SEI malaria model,
complementing the variance-based (Sobol) sensitivity analysis of $\mathcal{R}_0$
in `../sensitivity_analysis/`. Whereas Sobol decomposes **variance**, PRCC is a
**rank-based, monotone** sensitivity measure that is cheap and robust to a
non-Gaussian, correlated design — well suited to the ODE-based outputs that
cannot be evaluated 200,000+ times.

Four complementary analyses are provided:

| label | Analysis | Output metric | Cost |
|-------|----------|---------------|------|
| **A** | LHS + PRCC of the closed-form $\mathcal{R}_0$ | $\mathcal{R}_0$ (analytical) | ~instant |
| **B** | LHS + PRCC of the **seasonal $I_H$ peak** via ODE simulation | mean seasonal peak of human infectious prevalence, averaged over 2017–2023 | ~6 min (n=400) |
| **C** | **Across-year calibration uncertainty** of $b_1,b_2$ propagated to $\mathcal{R}_0$ | $\mathcal{R}_0$ per year, drift of $b_1,b_2$, joint $(b_1,b_2)\to\mathcal{R}_0$ contour | ~seconds |
| **D** | **Cross-validation robustness** of the week-level regression: role of the year-to-year level | within-year vs leave-one-year-out $R^2$, raw vs within-year-standardized | ~seconds |

---

## Method

### Shared machinery

- **`model_wrappers.py`** bridges the two model modules:
  - the analytical $\mathcal{R}_0$ (`../sensitivity_analysis/seirs_sei_r0.py`);
  - the ODE integration of the mean seasonal $I_H$ peak
    (`../../seirs_sei/lmfit_optimization/lmfit_optimization_shared.simulate_year`),
    which reproduces the paper's calibrated adult-mosquito-infectious time series and
    returns the daily human-infectious curve; the peak is averaged across the seven
    transmission years 2017–2023.
  - Because the *climate-response* parameters (e.g. $A,B,T',D_1,R_L,\mathrm{DD}$) are
    module **globals** in the ODE back-end, an override/restore context manager
    (`override_ode_globals`) lets each LHS row set them independently without disturbing
    the host process.
- **`reference_params(names)`** returns a reference vector: the **calibrated per-year
  means** of the epidemiological parameters (from `lmfit_results.json`) together with the
  Sobol-analysis baseline for the climate parameters. **Calibrated means, not
  midpoints**, are used as the "other-parameters-held-fixed" baseline, because parameter
  midpoints sit far from the biologically fitted values (e.g. the $\gamma$ midpoint
  ~0.5/day corresponds to a 2-day recovery and pins the ODE $I_H$ peak at an artificial
  ceiling).

### A & B. LHS + PRCC of $\mathcal{R}_0$ and of the seasonal $I_H$ peak

1. **Sample.** $n$ parameter rows are drawn over the same 25-dim hypercube and ranges as
   the Sobol analysis (`sobol_analysis.BOUNDS`) using SALib's Latin hypercube sampler
   (`seed=42`).
2. **Evaluate.** For **A** each row is fed to the closed-form $\mathcal{R}_0$. For **B**
   each row drives seven ODE year-simulations and we take the **mean seasonal peak**
   $I_H$.
3. **Sensitivity.** PRCC is computed per parameter: rank-transform both the parameter
   column and the output, regress out all other parameters, and take the partial
   correlation of the residuals; significance is a $t$-test on the partial correlation.
   Because log10-scaling $I_H$ peak is heavily right-skewed (the plateau at ~917 vs
   extreme excursions when $\gamma$ is tiny), PRCC for **B** is evaluated on
   $\log_{10}(\text{peak } I_H)$; the raw distribution is kept for display.
4. **Tornado.** One-at-a-time sweeps of each parameter to its bounds with the others at
   the calibrated reference, reporting the swing in the output. For $\mathcal{R}_0$ this
   is a fast, visual complement to PRCC.

### C. Across-year calibration uncertainty → $\mathcal{R}_0$

The per-year calibrated transmission probabilities $b_1,b_2$ (and the other fitted
parameters) are read from `lmfit_results.json`. For each of the seven years we report:

- the calibrated $b_1,b_2$ and their **drift** toward the fit bounds (2017 $b_1\approx0.26$
  rising to $\approx0.50$, $b_2\approx0.035$ falling to $\approx0.01$);
- $\mathcal{R}_0$ evaluated at each year's calibrated vector at the reference climate, and
  the resulting mean / SD / CV across years;
- the Spearman / Pearson correlation between the per-year $b_1$ and $b_2$ (whether the
  two transmission probabilities are trading off against each other);
- a joint $(b_1,b_2)\to\mathcal{R}_0$ contour with all other parameters at the calibrated
  reference, and the calibrated $I_H$ peak per year.

### D. CV robustness of the week-level regression ($b_1,b_2$ $R^2$)

The week-level regressions report a high **within-year (in-sample)** $R^2$ and a much
lower **leave-one-year-out (LOYO)** $R^2$. To test whether the year-to-year *level* of
$b_1,b_2$ explains that gap, we apply the user-proposed diagnostic:

- **standardize the target within each year** ($z=(y-\mu_y)/\sigma_y$) before fitting and
  evaluating, and re-measure both the within-year $R^2$ and the LOYO $R^2$;
- if removing the per-year level closes most of the gap, the shortfall is a technical
  **intercept/level** effect; if the gap persists, it reflects genuine **within-year shape**
  misfit.

> **Honest caveat (stated in the notebook too).** Within-year standardization is a
> *diagnostic*, not a deployable estimator: in a true LOYO evaluation you cannot compute a
> held-out year's mean/std without target leakage. It is a controlled experiment that
> holds the year level fixed while leaving every other part of the pipeline unchanged.

---

## Key findings

**A. $\mathcal{R}_0$ PRCC (n=1000).** Dominant contributors are the mosquito-mortality
shape coefficients $A$ (+0.82) and $B$ (+0.63), then $M'$, $b_1$, $b_2$, $\gamma$ (−),
with $\tau_H$ and $\omega$ ≈ 0 — fully consistent with the Sobol result. The one-at-a-time
tornado (at the calibrated reference, base $\mathcal{R}_0=1.33$) shows the largest swings
for $M'$, $b_2$, $A$, $B$, $b_1$, $\gamma$.

**B. Seasonal $I_H$ peak PRCC (n=400, on $\log_{10}$ peak).** The peak is dominated by the
recovery rate $\gamma$ (PRCC −0.53: faster recovery ⇒ smaller epidemic peak) and mosquito
mortality $A$ (+0.44), then $b_2$ (+0.30), $B$ (+0.24), $b_1$ (+0.21); $H_0$, $M'$, $k$,
$\phi$, $\omega$ are secondary and the remaining climate-response parameters are
negligible. The sampling distribution (median = min ≈ 917) reveals that the seasonal peak
sits on a **plateau** for a large fraction of the parameter space and only moves when
$\gamma$ is small or transmission/mortality terms push it off — an informative, structural
property of the $I_H$ output.

**C. Across-year calibration uncertainty.** $b_1$ drifts from 0.26 (2017) to the upper fit
bound ≈0.50 while $b_2$ drifts from ≈0.035 down to ≈0.01, so the two transmission
probabilities trade off (Pearson $r=-0.83$, $p=0.02$). The across-year variability of
$b_1$ is ~19% CV and of $b_2$ ~35% CV. Propagated to $\mathcal{R}_0$ at the reference
climate, the calibrated vectors give $\mathcal{R}_0\in[0.74,\,1.33]$ (mean ≈ 0.96, 20% CV)
— i.e. the fitted transmission parameters, on their own, place $\mathcal{R}_0$
**near/under the epidemic threshold**, in contrast to the paper's reference value of
2.20 (which uses the more optimistic fixed $b_1=0.04$, $b_2=0.09$, $M'=468{,}580$
combination). This is a substantive tension worth foregrounding: the $b_1/b_2$
calibration sits on the fit bounds and materially modulates whether the modelled system is
endemic.

**D. CV robustness.** Removing the year level *improves LOYO $R^2$ for both targets* but
by very different amounts:
- **$b_1$**: LOYO $R^2$ 0.17 → 0.21; the within-vs-LOYO gap narrows only slightly
  (0.33 → 0.28). The year offset plays a **minor** role; the gap is mostly **within-year
  shape** misfit.
- **$b_2$**: LOYO $R^2$ 0.19 → 0.44; the gap narrows strongly (0.44 → 0.20). A large part
  of the cross-year gap for $b_2$ **is** the year-to-year level shift.

So the answer to the robustness question is honest and mixed: the year-level effect
explains much of the generalization gap for $b_2$ but little for $b_1$.

---

## Files

| File | Description |
|------|-------------|
| `model_wrappers.py` | ODE + $\mathcal{R}_0$ wrappers, ODE-global override context manager, calibrated per-year reference vector, calibrated $b_1/b_2$ by year. |
| `lhs_prcc.py` | LHS sampler, PRCC partial-correlation implementation, Spearman, one-at-a-time tornado, group colour/label maps. |
| `b1b2_uncertainty.py` | Across-year $b_1/b_2$ drift, per-year $\mathcal{R}_0$, Spearman/Pearson, standalone `main`. |
| `cv_robustness.py` | Week-level regression CV: within-year vs LOYO $R^2$ for raw and within-year-standardized targets. |
| `run_lhs_prcc.py` | Orchestrator: runs **A–D** and writes all figures and tables. |
| `LHS_PRCC_Analysis.ipynb` | Self-contained notebook reproducing the analysis and narrative for the paper. |
| `figures/` | `.png` figures (PRCC bar charts, tornado, $I_H$ distribution, $b_1/b_2$ drift + $\mathcal{R}_0$ + contour, CV grouped bars). |
| `tables/` | `.csv` tables (PRCC tables for A and B, tornado, $I_H$ sampling stats, per-year calibration + uncertainty summary + correlation, CV summary). |
| `README.md` | This file. |

## Reproducing

Run from this folder using the project virtual environment (which also needs `SALib`
and `scikit-learn`):

```bash
python run_lhs_prcc.py      # recompute + regenerate all figures/tables (full: n_r0=2000, n_ih=400)
jupyter notebook LHS_PRCC_Analysis.ipynb
```

The ODE-based part (B) takes a few minutes (n=400); the closed-form $\mathcal{R}_0$
part is near-instant. Everything is deterministic (`seed=42`).
