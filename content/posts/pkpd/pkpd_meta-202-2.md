+++
title = "PKPD_META 202.2: Local-Only Stan Diagnostics"
type = "post"
date = "2026-08-12"
math = true
draft = true
summary = "Walk through the Stan switches, posterior draws, and bayesplot diagnostics that back the local-only PKPD_META 202 fit lifted directly from the supporting notebook."
tags = ["pkpd", "bayesian", "stan", "posterior"]
+++

This note follows [PKPD_META 202.1](./pkpd_meta-202-1/) but stays inside the `content/ipynb/pkpd_meta-202-local-only.ipynb` workflow that actually drives the local-only Stan run. Everything cited below—Stan code, posterior summaries, and diagnostic plots—comes from that notebook so the post remains in lockstep with the executable record.

## Stan switchboard for local-only runs

The notebook copies `linear_mvn_approx.stan` into a local `stan/` directory before compiling, then flips `local_data$fit_local <- 1L` (see the "Local sampling" chunk). That single flag is what tells the Stan program to drop the external cohort from the likelihood. The relevant block is reproduced from `content/baad-master/linear_mvn_approx.stan:70-149`:

```stan
transformed data {
  ...
  vector[T_prime] y_prime_bar = colMeans(y_prime);
}
transformed parameters {
  ...
  if (!fit_local) {
    if (fit_all == 1) {
      y_prime_log = normal_lpdf(to_vector(y_prime) | to_vector(y_prime_pred), sigma_y);
    } else if (fit_all == -1) {
      y_prime_log = multi_normal_lpdf(y_prime_bar | y_prime_pred_bar, Sigma_y_prime_bar);
    } else {
      y_prime_log = mvn_approx_lpdf(y_prime_bar | x_prime, J_prime, delta,
                                    mu_a, beta, sigma_a, sigma_y, xi_tilde);
    }
  } else {
    y_prime_log = 0;
  }
}
model {
  target += y_log;
  target += y_prime_log;
  ...
}
```

Because `fit_local = 1`, the sampler keeps drawing the nuisance shift `delta` under its prior, but it never evaluates an external-data likelihood term—`local y_prime_log = 0 ± 0` in the notebook’s `summarise_draws` output line 816 confirms that the log-likelihood contribution vanishes.

## Posterior draws cling to the simulator truth

The posterior summary below is taken from the `posterior::summarise_draws()` table in the notebook (cell 18) after restricting to the population parameters that matter for the local fit. The scale parameters were recorded on the log scale inside Stan; the table exponentiates those draws so they are comparable to the simulator inputs (`true_values` in cell 8).

| parameter   | mean  | sd    | p05  | p95  | truth |
| --- | --- | --- | --- | --- | --- |
| `mu_a[1]`   | 0.511 | 0.011 | 0.493 | 0.530 | 0.500 |
| `mu_a[2]`   | -0.189 | 0.019 | -0.221 | -0.157 | -0.200 |
| `beta`      | -0.099 | 0.016 | -0.126 | -0.073 | -0.100 |
| `sigma_a[1]`| 0.108 | 0.008 | 0.096 | 0.122 | 0.100 |
| `sigma_a[2]`| 0.099 | 0.009 | 0.085 | 0.113 | 0.100 |
| `sigma_y`   | 0.051 | 0.001 | 0.049 | 0.052 | 0.050 |

Two takeaways tie the results back to the gating logic above:

- Every posterior interval comfortably straddles the simulator truth, so the local-only fit remains well calibrated even without the external averages.
- The notebook rows `local y_log = 2033 ± 10.7` and `local y_prime_log = 0 ± 0` document that the sampler only updates on the internal patient matrix while treating the external block as prior-predictive noise.

## Local-chain diagnostics straight from bayesplot

The same notebook chunk (cells 21–31) saves the bayesplot objects that check mixing and marginal fits. Exported PNGs live under `static/img/pkpd/` so Hugo can reuse them without re-running the notebook.

![Trace plots for μₐ, β, and σ parameters gathered from all four chains. Tight vertical bands and overlapping colors show that the chains mix cleanly on the local data.](/img/pkpd/pkpd_meta-202-local-trace.png)

![Posterior density overlays by chain for the local parameters. The overlap demonstrates that no chain-specific modes appear even though only the internal cohort informs the fit.](/img/pkpd/pkpd_meta-202-local-density.png)

![80/95% credible intervals from `mcmc_intervals` with dashed red lines for the simulator truth. Every line sits inside the posterior bands, which reinforces that the local-only controls stay anchored to the known generating values.](/img/pkpd/pkpd_meta-202-local-intervals.png)

Together these figures show that nothing pathological happens when the model runs in “local only” mode: R-hat diagnostics embedded in the plots stay near 1, the density overlays are indistinguishable across chains, and the credible intervals retain the simulator ground truth as expected. That is exactly the behavior we need before introducing approximate or integrated likelihoods in the broader PKPD_META 202 workflow.
