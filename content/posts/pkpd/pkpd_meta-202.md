+++
title = "PKPD_META 202: Hierarchical Linear Stress Test"
type = "post"
date = "2026-02-12"
math = true
draft = true
summary = "Section 4.1 of Gelman et al. (2018) shows how a toy hierarchical linear model validates the simulated-likelihood strategy before we plug it into nonlinear PKPD meta-analyses."
tags = ["pkpd", "bayesian", "meta-analysis", "stan"]
+++

Section 4.1 builds a synthetic hierarchical linear regression to sanity-check the workflow before we trust it inside a nonlinear drug-disease model. Two datasets—one internal with raw longitudinal measurements, one external reported only as visit-level means—share core population parameters $\phi$, while the external study gets its own shift vector $\delta$ on the intercept and linear slope. The quadratic effect and noise scale stay global so we can observe how pooling increases precision when the structure is shared.

## Model anatomy in miniature

Each patient $j$ contributes observations $y_{jk} \sim \mathcal{N}(\alpha_{j1} + \alpha_{j2} x_k + \beta x_k^2, \sigma_y^2)$ over 13 monthly visits. Individual intercepts and slopes follow $\alpha_j \sim \mathcal{N}(\mu_\alpha, \Sigma_\alpha)$ with no slope-intercept correlation, and the external cohort repeats the hierarchy but with $\mu_\alpha$ translated by $\delta$. Weakly informative unit-normal priors on $\mu_\alpha$, $\beta$, and $\delta$ alongside half-normal priors on $\Sigma_\alpha$ and $\sigma_y$ keep the fake-data truth in view without overpowering it. The authors deliberately set $\delta = (0.1, 0.1)$ to ensure the posterior has to recover a meaningful cross-study difference.

- $J = 100$ patients per data set, each with $T = 13$ visits at months $x_k = 0, 1/12, \ldots, 1$.
- $\mu_{\alpha_1} = 0.5$, $\sigma_{\alpha_1} = 0.1$ so intercepts live near 0.5 on the unit-scaled outcome.
- $\mu_{\alpha_2} = -0.2$, $\sigma_{\alpha_2} = 0.1$ implying a 10–30 point decline over the year.
- $\beta = -0.1$ to encode accelerated decline; no intercept-slope correlation.
- $\sigma_y = 0.05$ representing 5-point residual error on the original 0–100 scale.
- $\delta = (0.1, 0.1)$ introduces a sizable external shift the sampler must recover.

## Simulation-based likelihood check

Because the external data arrive as means $\bar{y}$, the Stan program plugs in the simulated-average likelihood: for each draw, simulate $\tilde{J}$ hypothetical patients under $\phi + \delta$, record the mean vector $\tilde{M}_s$ and scaled covariance $\tilde{\Sigma}_s / \bar{J}$, then add $\log \mathcal{N}(\bar{y} \mid \tilde{M}_s, \tilde{\Sigma}_s / \bar{J})$ to the target density. With this linear model, the team can integrate the averages analytically, so they chart the approximation against ground truth. Figure 2 shows the Monte Carlo band hugging the closed-form likelihood—its width contracting like $\tilde{J}^{-1/2}$—which certifies the simulated plug-in approach.

## What the diagnostics tell us

Four inference scenarios highlight how much the averages buy us:
- `local`: fit internal raw data only; wider intervals on $\mu_{\alpha2}$ and $\beta$.
- `approximate`: combine raw and average data through the simulated likelihood.
- `integrated`: same data but with the exact average-data likelihood.
- `full`: pretend we had the external raw data.

The approximate and integrated posteriors line up, confirming that the approximation is the right target. Variance components $\sigma_{\alpha1}$ and $\sigma_{\alpha2}$ shrink only when true raw external data are present; averages cannot inform them. Yet the shared fixed effects tighten, and the $\delta$ components are estimated as efficiently from means as from raw data once $\Sigma_\alpha$ is pinned down. The odd/even chain strategy—half the chains at $\tilde{J}$, the others at $2\tilde{J}$—acts as a built-in Monte Carlo diagnostic: diverging posteriors signal the need to raise $\tilde{J}$.

## PKPD meta takeaways

- Reserve $\delta$ for structural differences you expect across trials (dose strength, visit cadence) and keep shared curvature terms like $\beta$ coupled to reap precision gains.
- Use the linear toy as a unit test: verify that your simulated-average block reproduces the analytic likelihood before deploying it in nonlinear PKPD compartments.
- Do not expect averaged externals to identify between-patient variance components—lock those down with priors or internal data.
- Stagger $\tilde{J}$ across chains to monitor simulation-induced noise; treat mismatched posteriors as you would a failing $\hat{R}$ diagnostic.
