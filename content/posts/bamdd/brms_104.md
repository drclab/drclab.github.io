+++
title = "brms 104: Model Fitting"
date = "2026-11-21T00:00:00Z"
type = "post"
draft = true
tags = ["brms", "bamdd", "bayesian", "model fitting"]
categories = ["bamdd"]
description = "Step-by-step instructions for compiling and sampling the BAMDD control-arm models in brms, including prior-only runs and sensitivity refits."
math = true
+++

`brms_103` locked down the formulas; now we focus on actually sampling them in `brms`. Every command below is drawn from `content/posts/bamdd/brms_101.ipynb` and the `01b_basic_workflow` module so the rendered HTML, notebooks, and published results stay synchronized.

## 1. Shared sampler configuration

Both the pooled and random-intercept fits use the same engine options:

```r
control = list(adapt_delta = 0.95)
refresh = 0
silent = TRUE
seed = <fixed integer>
```

- `adapt_delta = 0.95` trades a small slowdown for fewer divergences in the sparse eight-study dataset.
- `refresh = 0` and `silent = TRUE` keep the notebook output clean (mirroring the HTML screenshots).
- Recording the seed beside every fit (e.g., `seed = 4658758` for the pooled model) ensures deterministic posterior summaries when colleagues re-run the notebook.

### Math connection

Sampling targets the posterior densities

$$
p(\beta_0 \mid \mathbf{r}, \mathbf{n}) \propto p(\mathbf{r} \mid \beta_0, \mathbf{n}) p(\beta_0)
$$

for the pooled model and

$$
p(\beta_0, \mathbf{u}, \tau \mid \mathbf{r}, \mathbf{n}) \propto p(\mathbf{r} \mid \beta_0, \mathbf{u}, \mathbf{n}) p(\mathbf{u} \mid \tau) p(\beta_0) p(\tau)
$$

for the hierarchical model. The `control` list tunes the NUTS sampler that explores these surfaces.

## 2. Pooled fit (`fit_meta_fixed`)

```r
fit_meta_fixed <- brm(
  model_meta_fixed,
  data = arm_data,
  prior = prior_meta_fixed,
  control = list(adapt_delta = 0.95),
  refresh = 0,
  silent = TRUE,
  seed = 4658758
)
```

- `model_meta_fixed` and `prior_meta_fixed` come from `brms_103` and `brms_102`, respectively.
- The resulting draws directly summarize the pooled log-odds $\beta_0$.
- Inspect `summary(fit_meta_fixed)` or `posterior_summary(fit_meta_fixed)` after sampling; R-hat values at 1.00 and bulk/tail ESS > 1000 are expected given the simplicity of the model.
- The notebook echoes Stan’s status output so it is captured in the published HTML. The relevant cell prints

  ```
  Start sampling

  Running MCMC with 4 parallel chains...

  Chain 1 finished in 0.0 seconds.
  Chain 2 finished in 0.0 seconds.
  Chain 3 finished in 0.0 seconds.
  Chain 4 finished in 0.0 seconds.

  All 4 chains finished successfully.
  Mean chain execution time: 0.0 seconds.
  Total execution time: 0.3 seconds.
  ```

  Keeping `silent = TRUE` while allowing this brief summary mirrors the notebook output exactly.

## 3. Random-intercept fit (`fit_meta_random`)

```r
fit_meta_random <- brm(
  model_meta_random,
  data = arm_data,
  prior = prior_meta_random,
  control = list(adapt_delta = 0.95),
  refresh = 0,
  silent = TRUE,
  seed = 5868467
)
```

- Adds the study-level intercepts $u_i$ and heterogeneity $\tau$ to the parameter vector.
- After fitting, run both `summary(fit_meta_random)` and `bayesplot::mcmc_pairs(fit_meta_random)` (as in the notebook) to verify no divergences occurred and to visualize the $\tau$ vs. $\beta_0$ correlation.
- The HTML tutorial’s posterior predictive checks (`ppc_intervals`) also rely on this object, so keep the object name stable.
- The cell output stored in `brms_101.ipynb` confirms the four chains complete cleanly:

  ```
  Start sampling

  Running MCMC with 4 parallel chains...

  Chain 1 finished in 0.2 seconds.
  Chain 2 finished in 0.2 seconds.
  Chain 3 finished in 0.2 seconds.
  Chain 4 finished in 0.2 seconds.

  All 4 chains finished successfully.
  Mean chain execution time: 0.2 seconds.
  Total execution time: 0.3 seconds.
  ```

  If you see anything other than “All 4 chains finished successfully,” revisit the `control` list or data inputs before trusting downstream diagnostics.

## 4. Sensitivity refits via `update()`

The notebook shows how to reuse Stan compilation while modifying priors or data slices:

```r
prior_meta_random_alt <- prior_meta_fixed +
  prior(normal(0, 0.5), class = sd, coef = Intercept, group = study)

fit_meta_random_alt <- update(
  fit_meta_random,
  prior = prior_meta_random_alt,
  control = list(adapt_delta = 0.95),
  refresh = 0,
  silent = TRUE,
  seed = 6845736
)

fit_meta_random_alt2 <- update(
  fit_meta_random,
  newdata = slice_head(arm_data, n = 4),
  control = list(adapt_delta = 0.95),
  refresh = 0,
  silent = TRUE,
  seed = 5868467
)
```

- `update()` keeps the compiled Stan program, saving minutes on each variant.
- The first refit tightens the heterogeneity prior (see brms 102 for motivation); the second trims the dataset to the first four studies, matching the HTML’s “small-sample stress test.”
- Document the rationale for each variant in the surrounding markdown cells so reviewers know why multiple fits appear in the notebook.
- The “small-sample” refit produced a single divergence in the notebook, which is also preserved in the HTML:

  ```
  Warning: 1 of 4000 (0.0%) transitions ended with a divergence.
  See https://mc-stan.org/misc/warnings for details.
  ```

  The workflow keeps this warning visible so readers understand why sensitivity analyses are necessary; if the divergence persists after additional tuning, mention it explicitly in the accompanying commentary.

## 5. Prior-only sampling

Before trusting posterior predictions, run the model with `sample_prior = "only"`:

```r
fit_meta_random_prior <- update(
  fit_meta_random,
  sample_prior = "only",
  control = list(adapt_delta = 0.95),
  refresh = 0,
  silent = TRUE,
  seed = 5868467
)
```

- This draws from $p(\mathbf{r}^{\text{rep}} \mid \text{priors})$ so you can overlay prior predictive intervals on the observed counts.
- In the HTML module the resulting object feeds `bayesplot::ppc_intervals()`; keep the naming consistent so readers can cross-reference code blocks and figures.
- The sampler status again appears in the notebook for traceability (all four chains finish in ≈0.1 s). Leaving this snippet intact provides reviewers with immediate confirmation that prior-only runs actually executed.

## 6. Post-fit diagnostics checklist

After each run (posterior or prior-only):

1. `summary(fit)` – confirm R-hat ≈ 1 and effective sample sizes are healthy.
2. `pp_check(fit, type = "intervals")` – reproduce the workflow plots.
3. `log_lik(fit)` – optional, but storing it enables PSIS-LOO comparisons later.
4. `posterior::as_draws_df(fit)` – tidy draws for any custom contrasts or meta-analytic summaries.

Keeping this routine in `brms_101.ipynb` ensures model fitting stays reproducible and reviews can trace every figure back to a specific `brm()` or `update()` call.
