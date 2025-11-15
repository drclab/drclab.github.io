+++
title = "brms 105: Model Summary & Analysis"
date = "2026-11-21T00:00:00Z"
type = "post"
draft = true
tags = ["brms", "bamdd", "bayesian", "model analysis"]
categories = ["bamdd"]
description = "Guidance on extracting posterior summaries and diagnostics from the BAMDD control-arm fits, aligning notebook output with reporting needs."
math = true
+++

With the models formulated (brms 103) and fit (brms 104), this final installment walks through the summaries and diagnostics already demonstrated in `content/posts/bamdd/brms_101.ipynb`. The goal is to keep the published markdown synchronized with the actual cell output so stakeholders can trace every estimate back to a specific computation.

## 1. Posterior summaries (`summary()`)

Running `summary(fit_meta_random)` inside the notebook produces the table below (verbatim from the stored cell output):

| parameter          | Estimate | Est.Error | l-95% CI | u-95% CI | Rhat | Bulk_ESS | Tail_ESS |
|--------------------|----------|-----------|----------|----------|------|----------|----------|
| `sd(Intercept)`    | 0.38     | 0.22      | 0.04     | 0.88     | 1.00 | 949      | 1130     |
| `Intercept`        | -1.11    | 0.19      | -1.50    | -0.72    | 1.00 | 1287     | 1412     |

- The table encapsulates both heterogeneity ($\tau = \text{sd(Intercept)}$) and the grand log-odds $\beta_0$.
- R-hat exactly equals 1.00 and both ESS columns exceed 900, so no additional sampling is required.
- When reporting, convert $\beta_0$ to a probability (`plogis(-1.11) ≈ 0.25`) and $\tau` to an odds ratio scale (`exp(0.38) ≈ 1.46`) if the audience expects more intuitive units.

For the pooled model, `summary(fit_meta_fixed)` yields the same column structure but only reports the intercept row. Include both tables (or their tidy equivalents via `broom.mixed::tidy()`) in any analysis memo to maintain a direct link to the notebook.

## 2. Posterior draws for downstream contrasts

Use `posterior::as_draws_df()` or `posterior::summarise_draws()` to move from the `brm` object into data frames:

```r
draws_random <- as_draws_df(fit_meta_random)
tibble(
  median_prob = plogis(draws_random$b_Intercept),
  heterogeneity = draws_random$sd_study__Intercept
) |> summarise(
  q50_prob = median(median_prob),
  q05_prob = quantile(median_prob, 0.05),
  q95_prob = quantile(median_prob, 0.95)
)
```

- This mirrors the calculations used in the HTML figures (e.g., the horizontal interval chart for $\theta_{\text{new}}$).
- Keeping the quantiles in probability space simplifies downstream decision analysis since the probability of response is often the clinical target.

## 3. Posterior predictive checks

`brms_101.ipynb` overlays observed responder counts with posterior predictive intervals via:

```r
ppc_intervals(
  y = arm_data$r,
  yrep = posterior_predict(fit_meta_random, ndraws = 200),
  x = arm_data$study
) + labs(title = "Posterior predictive intervals (random intercept)")
```

- The plot shows each study’s observed count against the 50% and 90% predictive envelopes. In the existing notebook the observed dots lie well within the 90% band, indicating no gross misfit.
- The exported figure below (decoded directly from `brms_101.ipynb`) matches the notebook output:

  ![Posterior predictive check, random-effects model](/img/posts/brms-105/ppc-random.png)

- Reproduce the same figure for the pooled model (`fit_meta_fixed`) to highlight how ignoring heterogeneity underestimates uncertainty for the sparsely populated trials.
- The fixed-effects version is also embedded so reviewers do not need to open the notebook to see the contrast:

  ![Posterior predictive check, fixed-effects model](/img/posts/brms-105/ppc-fixed.png)
- When working offline, save the plot objects (e.g., with `ggsave()`) and reference them in long-form reports so reviewers can recreate the plots from the same draws.
- For extrapolating to hypothetical new studies, the notebook reuses `posterior_predict()` with `allow_new_levels = TRUE`. The resulting `ppc_intervals()` call produces:

  ![Posterior predictive check for new studies, random-effects model](/img/posts/brms-105/ppc-random-new.png)

  Showing the new-study envelope alongside observed data clarifies how uncertainty inflates once site-specific effects are marginalized out.

## 4. Sensitivity comparison

The alternative fits (`fit_meta_random_alt`, `fit_meta_random_alt2`) illustrate how the posterior responds to prior tightening and smaller sample sizes:

| Fit                      | Key change                               | Notable output                                                |
|--------------------------|-------------------------------------------|----------------------------------------------------------------|
| `fit_meta_random_alt`    | $\tau \sim \mathcal{N}^+(0, 0.5^2)$       | Posterior mean `sd(Intercept)` drops toward 0.25; predictive bands narrow. |
| `fit_meta_random_alt2`   | Same prior, first 4 studies only          | Stan warns: “1 of 4000 transitions ended with a divergence.”   |

- Always mention the divergence warning in the analysis narrative; it signals that further tuning (e.g., raising `adapt_delta`) or re-scaling may be required if the four-study subset becomes the primary dataset.
- Comparing `posterior_predict()` outputs across these fits provides the fastest visual check of prior influence—overlay the envelopes to show stakeholders how assumptions drive forecasts.
- The final comparison plot from the notebook (shown below) stacks point-interval summaries for the random-effects fit, the random-effects fit projected to new studies, and the pooled fit. Include this figure when discussing sensitivity analyses so readers can visually trace how each assumption shifts the predictive intervals:

  ![Posterior predictive check across pooled and random-effects variants](/img/posts/brms-105/ppc-all-models.png)

## 5. Prior predictive recap

The prior-only object `fit_meta_random_prior` feeds a nearly identical `ppc_intervals()` call but uses `posterior_predict(..., draws = fit_meta_random_prior)` so only prior information contributes. Use the resulting figure to document that the priors allow counts that span the observed range (20–139 responders) without producing impossible extremes (e.g., near-zero or near-total response for every study). Include this figure whenever you justify the priors in regulatory or cross-team reviews.

## 6. Reporting checklist

Before finalizing any BAMDD model summary, confirm that the following artifacts are exported (or at least referenced) from the notebook:

1. `summary()` output for every fit (pooled, hierarchical, sensitivities).
2. Posterior predictive plots for observed vs. replicated data.
3. Prior predictive plot highlighting the same scale.
4. Notes about sampler diagnostics (divergences, R-hat) lifted directly from the notebook cell outputs.
5. Any custom contrasts or decision metrics computed from `as_draws_df()`.

Following the checklist ensures that the textual summary, figures, and code cells move together through review cycles. It also keeps `brms_101.ipynb` as the single source of truth for both numbers and narratives in the BAMDD workflow.
