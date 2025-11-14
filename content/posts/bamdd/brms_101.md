+++
title = "brms 101: Basic Workflow"
date = "2026-11-14T00:38:07Z"
type = "post"
draft = false
tags = ["brms", "bamdd", "bayesian", "workflow"]
categories = ["bamdd"]
description = "Formal summary of the Basic Workflow module covering the observed data and the brms model specifications."
math = true
+++
## Data

Let $\{(r_i, n_i)\}_{i=1}^{8}$ denote the responder counts and total subjects for each study in the `RBesT::AS` historical control arms. The generative data statement is

$$
\begin{aligned}
D &= \{(r_i, n_i)\}_{i=1}^{8}, \\
r_i &\in \{0, 1, \ldots, n_i\}, \\
r_i \mid \theta_i &\sim \mathrm{Binomial}(n_i, \theta_i),
\end{aligned}
$$

with one observation per study (so the index $i$ already identifies the study).

| study   | $n_i$ (total) | $r_i$ (responders) |
|---------|---------------|--------------------|
| Study 1 | 107           | 23                 |
| Study 2 | 44            | 12                 |
| Study 3 | 51            | 19                 |
| Study 4 | 39            | 9                  |
| Study 5 | 139           | 39                 |
| Study 6 | 20            | 6                  |
| Study 7 | 78            | 9                  |
| Study 8 | 35            | 10                 |

## Model specifications

All models share the binomial likelihood above and the logit link $\mathrm{logit}(\theta) = \log(\theta / (1 - \theta))$.

### Intercept-only (fixed effect) model

$$
\begin{aligned}
r_i \mid \theta &\sim \mathrm{Binomial}(n_i, \theta), \\
\mathrm{logit}(\theta) &= \beta_0, \\
\beta_0 &\sim \mathcal{N}(0, 2^2).
\end{aligned}
$$

This reproduces the `bf(r | trials(n) ~ 1, family = binomial)` specification with the single prior obtained via `get_prior()` in the notebook. In brms (and the underlying `lme4` syntax) the `1` on the right-hand side simply denotes the intercept term. Writing `~ 1` forces the fixed-effects design matrix to contain only a column of ones, so every observation shares the same log-odds $\beta_0$ and no study-level predictors enter the model. Dropping the `1` or replacing it with `0` would remove the intercept altogether, while adding predictors (e.g., `~ 1 + x`) would extend the row-specific log-odds beyond this pooled baseline.

### Random-intercept hierarchical model

$$
\begin{aligned}
r_i \mid \theta_i &\sim \mathrm{Binomial}(n_i, \theta_i), \\
\mathrm{logit}(\theta_i) &= \beta_0 + u_i, \\
u_i &\sim \mathcal{N}(0, \tau^2), \\
\beta_0 &\sim \mathcal{N}(0, 2^2), \\
\tau &\sim \mathcal{N}^+(0, 1^2),
\end{aligned}
$$

where $\mathcal{N}^+$ denotes a half-normal prior as implemented with `prior(normal(0, 1), class = sd, coef = "Intercept", group = "study")`. An alternative sensitivity prior narrows the scale to $0.5$ while preserving the same structure, matching the `prior_meta_random_alt` updates demonstrated in the notebook.

The corresponding brms formula is `model_meta_random <- bf(r | trials(n) ~ 1 + (1 | study), family = binomial)`, highlighting how the random intercept term `(1 | study)` augments the fixed-effect baseline `~ 1`. Compared to the pooled model, this syntax introduces a study-level intercept deviation $u_i$ so each study receives its own $\theta_i$ while still sharing the overall $\beta_0$.
