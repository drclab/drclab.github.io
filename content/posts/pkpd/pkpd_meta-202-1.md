+++
title = "PKPD_META 202.1: Simulating the Hierarchical Stress Test"
type = "post"
date = "2026-02-03"
math = true
draft = true
summary = "Deep-dive on how the PKPD_META 202 synthetic datasets are generated, including the exact draws, sanity tables, and plots lifted from the supporting notebook."
tags = ["pkpd", "bayesian", "simulation", "stan"]
+++

This follow-up to [PKPD_META 202](../pkpd_meta-202/) stays entirely in the simulator. We recreate the internal patient trajectories and the external visit-level averages from `content/ipynb/pkpd_meta-202-sim.ipynb`, capture the notebook outputs, and explain how they supply every input to the Stan workflows.

## Fixed truth in the notebook

The notebook pins the population parameters to the exact values cited in Section 4.1, which keeps every rerun reproducible:

| parameter | value |
| --- | --- |
| `mu_a[1]` | 0.50 |
| `mu_a[2]` | -0.20 |
| `beta` | -0.10 |
| `sigma_a[1]` | 0.10 |
| `sigma_a[2]` | 0.10 |
| `sigma_y` | 0.05 |
| `delta[1]` | 0.10 |
| `delta[2]` | 0.10 |

The visit grid spans 13 evenly spaced months (`seq(0, 1, length.out = 13)`), and both internal and external cohorts simulate `J = 100` patients.

## Cohort simulator straight from the R chunk

The helper below is copied verbatim from the notebook and mirrors `linear_baad.R`. It draws subject-level intercepts and slopes from a diagonal covariance, injects the external shift `delta` when requested, and applies residual noise at each visit.

```r
simulate_cohort <- function(J, x, mu_alpha, sigma_alpha, beta, sigma_y, delta = c(0, 0)) {
  Sigma_alpha <- diag(sigma_alpha^2)
  alpha <- mvtnorm::rmvnorm(J, mu_alpha + delta, Sigma_alpha)
  T <- length(x)
  y <- matrix(NA_real_, nrow = J, ncol = T)
  for (k in seq_len(T)) {
    mean_k <- alpha[, 1] + alpha[, 2] * x[k] + beta * x[k]^2
    y[, k] <- rnorm(J, mean_k, sigma_y)
  }
  list(alpha = alpha, y = y)
}
```

Feeding the helper with and without the `delta` shift produces the notebook tibble `visit_grid`. Three representative visits show how the averages separate while keeping comparable dispersion:

| month | internal_mean | external_mean | internal_sd | external_sd |
| --- | --- | --- | --- | --- |
| 0.00 | 0.512 | 0.617 | 0.121 | 0.115 |
| 0.50 | 0.397 | 0.525 | 0.127 | 0.122 |
| 1.00 | 0.227 | 0.420 | 0.156 | 0.154 |

## Visual check: trajectories vs. shifted averages

![Internal trajectories with external mean overlay](/img/pkpd/pkpd_meta-202-1-trajectories.png)

The PNG above is exported directly from `pkpd_meta-202-sim.ipynb`. Each blue line is a simulated internal patient path, the solid black curve is the internal cohort mean, and the dashed red curve is the external mean after applying `delta = (0.1, 0.1)`. The overlay makes two facts visible: the curvature parameter `beta` is shared (means curve together), and the external cohort sits consistently higher because both intercept and slope receive the same positive shift.

## Packaging Stan data lists

After simulation, the notebook constructs four Stan-ready lists (`local`, `approximate`, `integrated`, `full`) that only toggle the logical flags used inside the model. The shared base object echoes the dimensions of the simulated arrays:

- `y` and `y_prime` are both `100 × 13`, i.e., patient × visit matrices.
- `J_tilde = 500` simulated patients feed the Monte Carlo likelihood approximation.
- `xi` holds the random seeds for the simulated-average block with dimensions `4 × 2 × 1000`, reflecting four inference scenarios, two summary statistics, and `2 * J_tilde` draws per scenario.

Those objects are what drive the inference comparison in the original post; this article simply documents their provenance so future PKPD_META installments can reuse or adapt the simulator with confidence.
