+++
title = "brms 103: Model Formulation"
date = "2026-11-21T00:00:00Z"
type = "post"
draft = true
tags = ["brms", "bamdd", "bayesian", "model formulation"]
categories = ["bamdd"]
description = "A bridge between the mathematical statements and the brms formulas used for the BAMDD control-arm meta-analysis."
math = true
+++

With the priors (brms 102) and workflow (01b basic module) in place, the next step is to pin down the exact model statements so that the R formula, Stan program, and manuscript description all match. Two canonical forms cover nearly every BAMDD analysis:

- a pooled (fixed-effect) binomial logistic model for quick summaries, and
- a random-intercept (hierarchical) model that captures among-study heterogeneity.

Each subsection below shows (1) the generative story, (2) how `brms::bf()` encodes it, and (3) the key arguments that must appear whenever you refit the model in `brms_101.ipynb`.

## Pooled control-arm likelihood

**Mathematical form**

$$
\begin{aligned}
r_i \mid \theta_i &\sim \operatorname{Binomial}(n_i, \theta_i), \\\\
\mathrm{logit}(\theta_i) &= \beta_0,
\end{aligned}
$$

where $r_i$ is the number of responders in study $i$ and $n_i$ is the trial size.

**brms formula**

```r
model_meta_fixed <- bf(
  r | trials(n) ~ 1,
  family = binomial(link = "logit")
)
```

- `r | trials(n)` instructs `brms` to use the binomial PMF with count `r` and total `n` that already exist in `arm_data`.
- `~ 1` encodes the single log-odds parameter $\beta_0$.
- Leaving the formula this simple ensures that the posterior samples in `fit_meta_fixed` track the pooled log-odds seen in `brms_101.ipynb`.

## Random-intercept meta-analysis

**Mathematical form**

$$
\begin{aligned}
r_i \mid \theta_i &\sim \operatorname{Binomial}(n_i, \theta_i), \\\\
\mathrm{logit}(\theta_i) &= \beta_0 + u_i, \\\\
u_i &\sim \mathcal{N}(0, \tau^2),
\end{aligned}
$$

where $\beta_0$ is the grand mean and $\tau$ is the among-study standard deviation.

**brms formula**

```r
model_meta_random <- bf(
  r | trials(n) ~ 1 + (1 | study),
  family = binomial(link = "logit")
)
```

- `(1 | study)` adds a study-specific intercept $u_i$ for every site in `arm_data$study`.
- When paired with `prior(normal(0, 1), class = sd, coef = Intercept, group = study)` (see brms 102), this formula reproduces the hierarchical model summarized in `fit_meta_random`.
- Because `bf()` builds the full design matrix, the `get_prior()` calls shipped in the notebook surface the `sd` rows that correspond to $\tau$; check those tables anytime you change the grouping variables.

## Extending the formulation

The `01b_basic_workflow.html` chapter emphasizes that the likelihood and random-effects structure should be settled before layering on covariates. The same pairing of math + `bf()` works for each extension:

- **Study-level predictors.** Add them to the linear predictor on both the math side and the R formula, e.g.,

  $$
  \mathrm{logit}(\theta_i) = \beta_0 + \beta_1 x_{i} + u_i
  $$

  ```r
  bf(r | trials(n) ~ 1 + x + (1 | study))
  ```

- **Alternative grouping.** Swap `(1 | study)` for `(1 | trial/id)` or similar structures as soon as the generative story introduces nested or cross-classified effects.
- **Non-centered reparameterizations.** `brms` handles these internally when you use `(1 | study)`; no extra code is required, but documenting the corresponding $u_i$ definition prevents confusion when you hand off posterior draws.

Capturing each model in both languages—the math and the `brms` formula—ensures the notebook, HTML tutorial, and any downstream decision documents stay synchronized even as the BAMDD control-arm meta-analysis evolves.
