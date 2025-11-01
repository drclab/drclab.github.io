+++
title = "PKPD 301: Drug-Disease Model Fundamentals"
date = "2025-11-01T02:00:00Z"
type = "post"
draft = true
math = true
tags = ["pkpd", "pharmacology", "modeling", "drug-disease"]
categories = ["posts"]
description = "Build a semimechanistic drug-disease model that captures natural disease turnover, drug modulation, and latent concentrations using ocular pharmacology as a case study."
+++

PKPD 201 and 202 connected exposure to effect using direct-effect and turnover pharmacodynamic models. PKPD 301 steps up to the drug-disease layer: we now model disease progression and drug action simultaneously so the natural history, placebo response, and treatment effect share one coherent set of parameters. The macular degeneration program described by Weber et&nbsp;al. (2018) is our running example—it provides a compact blueprint for building, calibrating, and diagnosing a semimechanistic drug-disease model.

## Logit scale for bounded clinical endpoints

Best corrected visual acuity (BCVA) is scored on the ETDRS letter chart, bounded between 0 and 100. Directly modeling a bounded response with Gaussian noise can force unrealistic extrapolations outside the physical range. Weber et&nbsp;al. resolve this by transforming each observation,

$$
R_j(t) = \operatorname{logit}\!\left(\frac{y_{jk}}{100}\right),
$$

which maps the measured letter score $y_{jk}$ to the real line. The turnover ODE works on $R_j(t)$, and predictions are mapped back through the inverse-logit before comparing to data. The transform also stabilizes variance near the boundaries, improving numerical performance in Stan or NONMEM implementations.

## Turnover backbone and steady state intuition

On the transformed scale, the disease trajectory follows a modified Jusko–Ko turnover model:

$$
\frac{dR_j(t)}{dt} = k_{\text{in},j} - k_{\text{out},j} \, R_j(t) - E_{\max,j} \, S_j\!\left(C_j(t)\right).
$$

- $k_{\text{in}}$ and $k_{\text{out}}$ define the natural disease turnover. Their ratio sets the placebo steady state: $\lim_{t \to \infty} R_j(t) = k_{\text{in},j} / k_{\text{out},j}$.
- $E_{\max}$ is the maximal drug-induced shift on the transformed scale.
- $S_j(C_j(t))$ is a Hill-type stimulation or inhibition term with $EC_{50}$ controlling potency. On a logit scale this produces logistic exposure-response profiles that respect the BCVA bounds.

The ratio $k_{\text{in}}/k_{\text{out}}$ is the placebo steady state. If the model drifts during placebo qualification, investigate whether baseline covariates, dropout trends, or measurement differences are forcing bias into these turnover terms. Because $E_{\max}$ and $EC_{50}$ act separately on the drug component, they can be interpreted alongside placebo diagnostics without conflating natural and treatment-driven changes.

## Latent ocular concentration

Ethical and logistical constraints prevent direct intraocular PK sampling. Instead, the concentration driving $S_j(C_j(t))$ is treated as latent but tied to dosing via mono-exponential elimination:

$$
C_j(t) = \frac{D_j}{V_\text{vit}} \, \exp\!\left(-\frac{\ln 2}{t_{1/2}} t\right),
$$

with vitreous volume fixed at 4 mL and half-life $t_{1/2} \approx 9$ days. Eliminating the need for direct PK samples keeps the model identifiable by leveraging known ocular physiology and dosing schedules (monthly or every eight weeks injections in the VIEW1/VIEW2 trials). In practice, you will:

- Encode regimen differences purely through dose amounts and frequency.
- Separate drug-specific parameters such as $t_{1/2}$ and $E_{\max}$ from disease parameters ($k_{\text{in}}$, $k_{\text{out}}$), enabling extrapolation to new agents while holding progression biology constant.
- Use posterior predictive checks on placebo and treated arms to verify that the latent concentration assumptions do not induce bias in late follow-up windows.

## Fitting workflow

1. **Initial conditions**: For each individual, set $R_j(0)$ based on the observed baseline BCVA (transformed). When baseline data span several visits, use the mean to avoid fitting noise as disease progression.
2. **Variation components**: Introduce inter-individual variability on $k_{\text{in}}$, $k_{\text{out}}$, $E_{\max}$, and $EC_{50}$ using log-normal distributions. Add residual variability on the transformed scale so model errors respect the bounded outcome after back-transformation.
3. **Estimation**: Stan handles the ODE with `integrate_ode_bdf`, while NLME tools (NONMEM, Monolix) can rely on their stiff ODE solvers. Monitor ODE tolerances because steep dosing pulses interact with the exponential decay.
4. **Back-transformation**: Always convert posterior predictive trajectories through the inverse-logit before reporting letter scores. This keeps clinical review grounded in familiar units.

## Model qualification signals

- **Steady-state alignment**: Compare posterior draws of $k_{\text{in}}/k_{\text{out}}$ against long-term placebo data or natural history cohorts.
- **Dose-regimen sensitivity**: Simulate q4w versus q8w regimens by adjusting the latent concentration and ensure predicted BCVA drop matches observed arm separation.
- **Posterior predictive overlays**: Compare inverse-logit predictions with individual spaghetti plots and means to confirm the ODE captures both acute gains and gradual decline.
- **Parameter correlations**: Inspect joint posteriors for $E_{\max}$ and $EC_{50}$—strong correlations can signal insufficient sampling near onset and may motivate richer PD sampling in future protocols.

## Where to deepen your build

- Gelman, S. Weber, et&nbsp;al. (2018). *Bayesian aggregation of average data: An application in drug development*. Annals of Applied Statistics 12(3):1583–1604. The Appendix contains Stan code for the macular degeneration drug-disease model.
- Jusko, W.J. and Ko, H.C. (1994). Classic reference for semimechanistic turnover models.
- Hart, W.M. (1992). Provides the 4 mL vitreous volume reference used for the latent concentration calculation.

Coming next in the PKPD 300 series: sensitivity analyses for drug-disease combinations and strategies for communicating heterogeneity to clinical teams.
