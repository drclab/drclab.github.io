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

In [brms 101](../brms_101/) we walked through the likelihood for the BAMDD control-arm meta-analysis and compared pooled versus random-intercept fits. This follow-up distills what the [`src/01b_basic_workflow.html`](http://localhost:8000/src/01b_basic_workflow.html) module says about prior selection and ties the code chunks from `content/posts/bamdd/brms_101.ipynb` into a single priors playbook.

## Why we bother with explicit priors

`brms` will happily supply wide Student-*t* defaults such as `student_t(3, 0, 2.5)` on intercepts and standard deviations. That can be acceptable when data are abundant, but in our eight-study control arm we only have responder counts between 20 and 139 patients. The HTML tutorial stresses that model *and* data jointly define which parameters even exist (categorical predictors, varying intercepts, etc.), so stating priors is part of communicating the modeling assumptions rather than an optional flourish.

Formally, the shared binomial likelihood stays the same:

$$
\begin{aligned}
r_i \mid \theta_i &\sim \operatorname{Binomial}(n_i, \theta_i), \\
\mathrm{logit}(\theta_i) &= \beta_0 + u_i,
\end{aligned}
$$

where the fixed-effect model sets $u_i = 0$ and the hierarchical model infers $u_i \sim \mathcal{N}(0, \tau^2)$.

## Inspecting `brms` defaults with `get_prior()`

The notebook starts by calling

```r
get_prior(model_meta_fixed, arm_data)
get_prior(model_meta_random, arm_data)
```

replicating the tables shown in §2.4 of the HTML walkthrough. Two takeaways matter:

- The intercept-only model exposes one parameter, the log-odds $\beta_0$, with a default `student_t(3, 0, 2.5)` prior.
- The random-intercept model extends the parameter list with `sd` entries tied to the `study` grouping factor; leaving the defaults untouched implies a half-Student-*t* prior on $\tau$ that may overdisperse the heterogeneity.

| prior                | class     | coef | group | resp | dpar | nlpar | lb | ub | tag     | source  |
|----------------------|-----------|------|-------|------|------|-------|----|----|---------|---------|
| `student_t(3, 0, 2.5)` | Intercept |      |       |      |      |       |    |    | default | default |

| prior                | class | coef     | group | resp | dpar | nlpar | lb | ub | tag     | source  |
|----------------------|-------|----------|-------|------|------|-------|----|----|---------|---------|
| `student_t(3, 0, 2.5)` | Intercept |          |       |      |      |       |    |    | default | default |
| `student_t(3, 0, 2.5)` | sd    |          |       |      |      |       | 0  |    | default | default |
|                      | sd    |          | study |      |      |       |    |    | default | default |
|                      | sd    | Intercept | study |      |      |       |    |    | default | default |

Because `get_prior()` inspects the fully expanded design matrix, it returns identifiers for `class`, `coef`, and `group`. Those strings reappear in every `prior()` call. The HTML tutorial emphasizes being specific—matching on `class = sd`, `coef = "Intercept"`, and `group = "study"`—to avoid unintentionally reusing priors elsewhere in larger models.

## Custom priors used in BAMDD

The R notebook codifies the modeling decisions as

```r
prior_meta_fixed <- prior(normal(0, 2), class = Intercept)

prior_meta_random <- prior_meta_fixed +
  prior(normal(0, 1), class = sd, coef = Intercept, group = study)
```

On the logit scale, $\beta_0 \sim \mathcal{N}(0, 2^2)$ puts the 95% probability interval roughly between logits $-4$ and $4$, i.e., between 1.8% and 98% responder rates. That is still weakly informative compared to the RBesT arm data but removes the extremely fat tails of the default Student-*t*. Likewise, $\tau \sim \mathcal{N}^+(0, 1^2)$ (the half-normal induced by `class = sd`) shrinks study-to-study deviations toward 0 while still permitting odds ratios up to `exp(2)` for individual sites.

### Less-specific declarations (when desired)

The HTML page also illustrates a compact alternative:

```r
prior_meta_random <- prior_meta_fixed +
  prior(normal(0, 1), class = sd)
```

Leaving out `coef` and `group` makes *every* random-effect standard deviation share the same prior. That can be convenient for models with dozens of varying coefficients, but the meta-analysis sticks with the explicit variant above to keep the intent obvious to collaborators.

## Sensitivity to heterogeneity priors

Posterior inference in sparse meta-analyses can pivot on the heterogeneity prior. The notebook defines

```r
prior_meta_random_alt <- prior_meta_fixed +
  prior(normal(0, 0.5), class = sd, coef = Intercept, group = study)

fit_meta_random_alt <- update(fit_meta_random,
  prior = prior_meta_random_alt,
  control = list(adapt_delta = 0.95),
  seed = 6845736
)
```

and also reruns the narrower prior on a truncated dataset (`slice_head(arm_data, n = 4)`). Comparing the resulting posterior predictive checks (see the `ppc_intervals` plots in `brms_101.ipynb`) clarifies how halving the prior scale for $\tau$ makes the new-study forecasts cling more tightly to the pooled estimate. Treat these paired fits as a lightweight sensitivity grid before you commit to a prior that will be used downstream for operating characteristics.

## Summarizing priors inside a fitted object

Once a model is fit, `prior_summary(fit_meta_random)` prints the actual distributions and scales that ended up in Stan, consolidating the chained `prior()` statements, defaults for any unused parameters, and the implied truncation for standard deviations. Pulling that summary into markdown (e.g., via `knitr::kable()`) is a quick way to double-check that the priors you meant to use are truly attached.

| prior           | class     | coef     | group | resp | dpar | nlpar | lb | ub | tag     | source |
|-----------------|-----------|----------|-------|------|------|-------|----|----|---------|--------|
| `normal(0, 2)`  | Intercept |          |       |      |      |       |    |    |         | user   |
| `student_t(3,0,2.5)` | sd    |          |       |      |      |       | 0  |    |         | default |
|                 | sd        |          | study |      |      |       |    |    |         | default |
| `normal(0, 1)`  | sd        | Intercept | study |      |      |       |    |    |         | user   |

## Prior predictive diagnostics

Both the HTML workflow and the notebook show how to engage `brms`’ prior predictive mode:

```r
fit_prior_only <- update(
  fit_meta_random,
  sample_prior = "only",
  refresh = 0,
  silent = TRUE,
  seed = 5868467
)
```

Sampling from the prior alone makes it obvious whether the implied responder counts live on the same order of magnitude as the historical trials. Overlaying `posterior_predict()` draws from `fit_prior_only` onto the observed counts (for example, with `bayesplot::ppc_intervals`) is the fastest sanity check we have, and it usually reveals if the prior variance is so large that impossible studies dominate the predictive envelope.

## Where to go next

The logical continuation after settling on priors is to fold them into the multiplicity of posterior predictive checks documented in `brms_101`. From there you can graduate to the [`src/01c_priors.html`](http://localhost:8000/src/01c_priors.html) module, which expands the same vocabulary to custom link functions and multi-parameter priors. For now, keeping these base patterns in mind ensures that every BAMDD meta-analysis starts from reproducible and well-communicated prior assumptions.
