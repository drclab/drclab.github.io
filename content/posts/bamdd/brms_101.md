+++
title = "brms 101: Basic Workflow"
date = "2026-11-14T00:38:07Z"
type = "post"
draft = false
tags = ["brms", "bamdd", "bayesian", "workflow"]
categories = ["bamdd"]
description = "Notes from the Basic Workflow module outlining how we prep R, specify models, and critique fits with brms."
+++

A fresh pass through the **Basic workflow** module (http://localhost:8000/src/01b_basic_workflow.html) shows how to take a meta-analysis from raw binomial counts to posterior predictive checks with `brms`. These are the highlights we keep handy when onboarding new BAMDD contributors.

## Case Study Recap

- Work with the `RBesT::AS` control-arm data (8 historical studies with responders `r` out of `n`).
- Aim: estimate the subgroup-level responder rate while accounting for between-study heterogeneity.
- Two brms models matter: an intercept-only fixed effect and a random intercept model `r | trials(n) ~ 1 + (1 | study)` under a binomial family with logit link.
- Priors mirror the case study: `β ~ Normal(0, 2)` for the overall log-odds and a conservative half-normal prior on the study-level SD `τ`.

## Prepare the R Session

```r
library(brms)
library(posterior)
library(bayesplot)
library(ggplot2)
library(dplyr)
library(tidyr)
library(knitr)
here::i_am("src/01b_basic_workflow.qmd")
options(
  brms.backend = "cmdstanr",
  cmdstanr_write_stan_file_dir = here::here("_brms-cache")
)
dir.create(here::here("_brms-cache"), showWarnings = FALSE)
set.seed(5886935)
```

- Keep at least four CPU cores plus ~8 GB RAM so parallel chains and Stan compilation behave.
- Cache compiled Stan binaries via `cmdstanr_write_stan_file_dir`; it saves minutes every time you tweak data or restart R.
- Only enable `options(mc.cores = 4)` when a single-chain runtime exceeds a minute; otherwise parallel overhead dominates and slows simulation studies.
- Optional: source helper scripts with `source(here::here("src", "setup.R"))` to keep plotting/reporting defaults consistent.

## Model Formula & Families

- Put the brms formula in a variable (e.g., `model_meta_random <- bf(r | trials(n) ~ 1 + (1 | study), family = binomial)`).
- Left-hand side handles the response plus per-row metadata (`trials(n)` passes exposure); right-hand side holds fixed effects and random effects just like base `lme4` syntax.
- Switch to simpler `bf(r | trials(n) ~ 1, family = binomial)` for the fixed effect comparison or extend to non-linear/multi-parameter families as needed.
- `?brmsfamily` lists every supported likelihood; if the library lacks yours you can still register a custom family.

## Priors: Inspect, Then Set

```r
get_prior(model_meta_fixed, arm_data)
prior_meta_fixed  <- prior(normal(0, 2), class = Intercept)

get_prior(model_meta_random, arm_data)
prior_meta_random <- prior_meta_fixed +
  prior(normal(0, 1), class = sd, coef = "Intercept", group = "study")
```

- `get_prior()` needs both the formula and the data because categorical levels or multi-response terms inflate the parameter list.
- Use the columns `class`, `coef`, and `group` to surgically assign priors to specific random-effect SDs or intercepts; only fall back to broad `class = sd` rules when every grouping structure should share the same prior.
- Sample priors directly with `update(fit, sample_prior = "only")` to run prior predictive checks; it strips the likelihood but keeps the rest of the model pipeline identical.

## Fit & Iterate Quickly

```r
fit_meta_random <- brm(
  model_meta_random,
  data  = arm_data,
  prior = prior_meta_random,
  control = list(adapt_delta = 0.95),
  refresh = 0,
  silent = TRUE,
  seed = 5868467
)
```

- Bump `adapt_delta` above the 0.8 default whenever you want a more conservative NUTS trajectory (fewer divergences in exchange for runtime).
- Use `update(fit_meta_random, prior = prior_meta_random_alt, ...)` to re-use compiled models when testing alternative priors or arguments.
- `print(fit_meta_random)` surfaces R-hat, Bulk_ESS, and Tail_ESS; keep R-hat near 1.0 and aim for ESS > 200 before trusting summaries.
- `fitted()` vs `predict()`: the former gives posterior means for the latent response; the latter adds sampling noise so you see outcome-level dispersion.

## Posterior Predictive Checks

- `pp_check(fit_meta_random, type = "intervals", ndraws = NULL)` integrates with **bayesplot** and visualizes 50/90% credible intervals plus observed data.
- The random-effects model covers studies with unusual response counts (study 7) much better than the fixed-effect model—the predictive intervals actually include the observed `r`.
- Re-label studies as new (`mutate(study = paste0("new_", study))`) and call `posterior_predict(..., allow_new_levels = TRUE)` to gauge how much uncertainty remains when you do **not** condition on each historical dataset.
- Building a combined tibble of interval data for `ppc_intervals_data` and faceting by model makes the trade-off between bias and variance obvious for stakeholders.

## Takeaways for BAMDD Work

- Always start by writing the formula and priors down explicitly; `get_prior()` serves as a checklist before any `brm()` call.
- Cache Stan builds, set seeds, and gate parallelization on actual runtime so re-fitting is painless on shared infrastructure.
- Prior predictive checks plus posterior predictive checks (`pp_check`, `posterior_predict`, `ppc_intervals`) should happen before you compare leave-one-out scores—otherwise you risk chasing metrics on a misspecified model.
- Keep both the fixed and random effects fits around; comparing `fitted()` outputs quickly reveals how much shrinkage heterogeneity introduces, which informs whether the model can support borrowing for future studies.
