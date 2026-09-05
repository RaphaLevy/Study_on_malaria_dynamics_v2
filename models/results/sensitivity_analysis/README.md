# Sensitivity Analysis of the SEIRS-SEI Malaria Model — R0

This folder contains a **global (variance-based / Sobol) sensitivity analysis** of the
basic reproduction number $\mathcal{R}_0$ of the SEIRS-SEI malaria model developed for
the paper, together with all generated figures and tables.

## What is analysed

The reproduction number $\mathcal{R}_0$ is the single most important summary of
transmission risk. It has a closed form (derived via the Next Generation Matrix method
in `models/seirs_sei/R0_Calculation.ipynb` and Section 5.2 of the paper):

$$
\mathcal{R}_0 =
\sqrt{
\frac{a^2\, b_1\, b_2\, \Lambda^{*}\, b_{3M}\,\ell}
{N\,\gamma\,(b_{3M}\,\ell+\mu)\,\mu^{2}}
},
\qquad
\Lambda^{*} = \mu\, M^{*} = \mu\, \frac{\bar{b}\,\bar{K}}{\bar{b}+\mu},
\qquad
\bar{K} = M'\, f_T(T)\, f_R(R),
$$

where each quantity encodes an environmental response:

- $a(T)$ — temperature-dependent biting rate  $a=\max(0,(T-T')/D_1)$;
- $\mu(T,H)$ — mosquito mortality  $\mu=-\ln\big(p_T(T)\,p_H(H)\big)$, with the humidity
  sigmoid  $p_H(H;\;H_0,k,\varphi)$;
- $b_{3M}(T)$ — Brière sporogonic development rate;
- $\ell=e^{-\mu\,\tau_M(T)}$ — probability of surviving the extrinsic incubation period;
- $\bar{b}(R,T)$ — temperature/rainfall-dependent per-capita recruitment;
- $f_T(T),\,f_R(R)$ — temperature and rainfall habitat suitability driving the carrying
  capacity $K$;
- $b_1,\,b_2$ — per-bite transmission probabilities; $\gamma$ — human recovery;
  $N$ — human population.

Because $\mathcal{R}_0$ is evaluated at the **disease-free equilibrium**, it is
independent of the human intrinsic incubation period $\tau_H$ and of the immunity-waning
rate $\omega$. These two calibrated parameters are therefore assigned *zero* sensitivity
(they are tabulated but enter neither the formula nor the variance).

## Method

1. **Reference environmental state.** The closed-form $\mathcal{R}_0$ is evaluated at a
   single representative state. We use the **mean climate of the transmission season**
   (the days in 2017–2023 on which the model sustains $\mathcal{R}_0>1$):
   $T=25.57\,^{\circ}$C, $R=7.19$ mm, $H=78.34\%$. This avoids evaluating the
   bell-shaped rainfall-recruitment response at the arithmetic-mean daily rainfall
   (which is dominated by dry days and forces $\mathcal{R}_0<1$), and it corresponds to
   the endemic operating point of the system. At this reference,
   $\mathcal{R}_0 = 2.20$ (with $\mathcal{R}_0^H=4.17$, $\mathcal{R}_0^M=1.16$).

2. **Sampling.** Each of 25 parameters is drawn independently over a physically /
   biologically motivated range (Table bounds in `sobol_analysis.py`) using the Saltelli
   Sobol' sampler from [SALib](https://salib.readthedocs.io/), base size $N=4096$
   ($N\times(2k+2)=212{,}992$ model evaluations; instantaneous for the closed-form).

3. **Indices.** First-order ($S_1$), total-order ($S_T$) and second-order indices are
   estimated by the SALib Sobol' estimator, with 95% confidence intervals from
   bootstrap resampling (`num_resamples`).

## Key findings

- The **mosquito mortality response** — specifically the quadratic temperature
  coefficients $A$ and $B$ of the survival curve — and the **human recovery rate
  $\gamma$** are the dominant drivers of $\mathcal{R}_0$ uncertainty, with
  $S_T\approx 0.44$ each. Their $S_T \gg S_1$ indicates **strong interaction effects**:
  the impact of each is amplified through the coupled human–vector cycle and the
  density-dependent recruitment $\Lambda^{*}$.
- **$B$, $b_2$, $M'$, $b_1$, $k$, $H_0$** are secondary contributors.
- Because $\mathcal{R}_0 \propto 1/\gamma$ and depends on $\mu$ nonlinearly, control
  measures acting on vector mortality (e.g. insecticide/LLIN-driven survival reduction)
  and on shortening the human infectious period are predicted to have the largest
  leverage on transmission, consistent with the model structure.
- $\tau_H$ and $\omega$ have identically zero sensitivity by construction.

## Files

| File | Description |
|------|-------------|
| `seirs_sei_r0.py` | Self-contained closed-form $\mathcal{R}_0$ evaluation of the SEIRS-SEI model (mirrors `seirs_sei_ode_shared.py` equations). |
| `sobol_analysis.py` | Parameter bounds + SALib sampling/analysis wrappers and tidy export helpers. |
| `generate_figures_tables.py` | Script that runs the analysis and writes all figures and tables. |
| `Sensitivity_Analysis.ipynb` | Self-contained notebook reproducing the analysis and narrative for the paper. |
| `figures/` | `.png` figures (Sobol index bar charts, $S_1$ vs $S_T$, group summary, $\mathcal{R}_0$ distribution). |
| `tables/` | `.csv` tables (full indices, top contributors, group summary, reference $\mathcal{R}_0$, metadata). |
| `README.md` | This file. |

## Reproducing

Run from this folder (using the project virtual environment, which has `numpy`, `scipy`,
`pandas`, `matplotlib` and `SALib` — SALib may need `pip install SALib`):

```bash
python generate_figures_tables.py      # recompute + regenerate all figures/tables
jupyter notebook Sensitivity_Analysis.ipynb
```

The analysis is deterministic (fixed `seed=42`).
