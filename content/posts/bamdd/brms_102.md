+++
title = "brms 102: Prior Specification & Diagnostics"
date = "2026-11-21T00:00:00Z"
type = "post"
draft = true
tags = ["brms", "bamdd", "bayesian", "priors"]
categories = ["bamdd"]
description = "Extended notes on how the BAMDD meta-analysis uses informative priors in brms, including defaults, customizations, and prior predictive checks."
math = true
+++

In [brms 101](../brms_101/) we introduced the likelihood for the BAMDD control-arm meta-analysis. This sequel condenses the prior discussion from the [`src/01b_basic_workflow.html`](http://localhost:8000/src/01b_basic_workflow.html) notes into three synchronized views: (1) the mathematical form, (2) the exact `brms` code, and (3) the realized values reported in the notebook cell output. Keeping the three aligned avoids accidental drift between documentation, scripts, and the fitted object.

## 1. Mathematical form

The data model is

$$
\begin{aligned}
r_i \mid \theta_i &\sim \operatorname{Binomial}(n_i, \theta_i), \\
\mathrm{logit}(\theta_i) &= \beta_0 + u_i,
\end{aligned}
$$

with priors

$$
\begin{aligned}
\beta_0 &\sim \mathcal{N}(0, 2^2), \\
u_i \mid \tau &\sim \mathcal{N}(0, \tau^2), \\
\tau &\sim \mathcal{N}^+(0, 1^2),
\end{aligned}
$$

where $\mathcal{N}^+$ denotes the half-normal distribution induced by truncating at zero. The intercept prior keeps the median response probability near 50% but limits logits to roughly $[-4, 4]$. The heterogeneity prior keeps study-level odds ratios within $\exp(\pm 2)$ unless data insist otherwise.

## 2. `brms` code

Matching the mathematical intent requires specifying the priors with explicit `class`, `coef`, and `group` entries. The notebook uses the following chunk so that the fit object carries the exact same distributions written above:

```r
prior_meta_fixed <- prior(normal(0, 2), class = Intercept)

prior_meta_random <- prior_meta_fixed +
  prior(
    normal(0, 1),
    class = sd,
    coef = Intercept,
    group = study
  )
```

Adding or removing varying effects later will not silently reassign this `normal(0, 1)` prior because it is tied to `class = sd`, `coef = "Intercept"`, and `group = "study"`.

## 3. Real values from the cell output

After fitting `fit_meta_random`, the notebook printed `prior_summary(fit_meta_random)` to verify the scales and truncations that Stan actually used. The resulting table (generated via `knitr::kable()` inside `brms_101.ipynb`) is reproduced below so the documentation, code, and output stay synchronized:

| prior                | class     | coef     | group | lb | source |
|----------------------|-----------|----------|-------|----|--------|
| `normal(0, 2)`       | Intercept |          |       |    | user   |
| `student_t(3,0,2.5)` | sd        |          |       | 0  | default |
|                      | sd        |          | study |    | default |
| `normal(0, 1)`       | sd        | Intercept | study |    | user   |

The duplicate `sd` rows come from `brms` first declaring the generic standard-deviation class (with its half-Student-*t* default) and then overriding the relevant entry with the half-normal we supplied. Confirming this table after every run is the quickest way to ensure the three prior forms above continue to match.
