+++
title = "Stan 102: Trusting R-hat"
date = "2025-10-23T00:00:00Z"
type = "post"
draft = false
math = true
tags = ["stan", "bayesian diagnostics", "RHat"]
categories = ["posts"]
description = "Deep dive on the R-hat convergence diagnostic in Stan, how to compute it, and what to do when it misbehaves."
+++

R-hat is the first convergence diagnostic we read after fitting the Stan 101 logistic regression, and it stays on our dashboard as models grow. This follow-up focuses entirely on what R-hat measures, how Stan reports it, and the playbook we use when it drifts above a clean threshold.

## Why R-hat is mandatory

- Detects whether independent Markov chains explore the same posterior bulk.
- Flags sticky chains faster than eyeballing trace plots.
- Ships with every Stan fit, so you never need to bolt on extra tooling.
- Extends to transformed parameters and generated quantities, not just raw draws.

We still scan divergences and effective sample sizes, but R-hat is the quick, global check for “did the sampler finish its job?”

## How Stan computes R-hat

Stan follows the split-chain estimator from Vehtari, Gelman, Simpson (2021). For each parameter $\theta$, chains are split in half to double the number of sequences, then within- and between-chain variances are compared:

$$
\hat{R} = \sqrt{\frac{\hat{V}}{W}}, \qquad \hat{V} = \frac{N - 1}{N} W + \frac{1}{N} B,
$$

where

- $W$ is the mean of the within-chain variances,
- $B$ is the between-chain variance scaled by the number of draws per split chain $N$.

If chains mix, $B$ and $W$ agree and $\hat{R} \rightarrow 1$. Values above $1.01$ mean at least one chain is wandering in a different part of the posterior or stuck in a bad geometry.

## Reading R-hat in CmdStanPy

`CmdStanPy` computes R-hat as part of the summary. Pull it programmatically so you can wire alerts or tidy tables.

```python
from cmdstanpy import CmdStanModel
import pandas as pd

model = CmdStanModel(stan_file="models/logistic_regression.stan")
fit = model.sample(
    data=stan_data,
    iter_warmup=1000,
    iter_sampling=1000,
    chains=4,
    seed=20251030,
)

summary = fit.summary()
rhat = summary.loc[:, ["R_hat"]].reset_index(names="variable")
rhat[rhat["R_hat"] > 1.01]
```

The last line filters any parameter or transformed quantity with suspicious R-hat. For richer reporting, convert the draws to ArviZ and use `az.rhat()`:

```python
import arviz as az

idata = fit.to_inference_data()
az.rhat(idata, method="rank")
```

Rank-normalized R-hat stabilizes the diagnostic for heavy-tailed posteriors and is what Stan reports by default.

## When R-hat raises a flag

- **Thin effective samples:** Chains might alternate between modes but never overlap enough.
- **Centering pathologies:** Hierarchical models with centered parameterizations can trap chains.
- **Poor initialization:** Outlying initial values create long warmup transients that contaminate the split chains.
- **Posterior multimodality:** If the model has label switching or true multimodal structure, chains must visit every mode for R-hat to settle.

Looking at trace plots still helps diagnose the exact failure mode, but R-hat is the first red light.

## Recover from inflated R-hat

1. Re-run with more warmup (`iter_warmup=1500`) so adaptation fully settles.
2. Reparameterize: move centered hierarchies to non-centered forms or rescale parameters to the unit scale Stan prefers.
3. Tighten priors to rule out implausible regions that the sampler keeps exploring.
4. Increase `max_treedepth` only after geometry fixes—long treedepths alone rarely drop R-hat.
5. If the model is multimodal, add identifiability constraints or marginalized likelihoods that collapse redundant modes.

Track R-hat after each change; the goal is every reported value at or below `1.01` (some teams enforce `1.005` for production runs).

## Takeaways

- R-hat compares between- and within-chain variation, so it is sensitive to both stuck chains and genuine multimodality.
- Split-chain, rank-normalized R-hat is the default in Stan—treat `1.01` as a soft ceiling and investigate anything higher.
- Automate R-hat checks in pipelines so failing diagnostics halt downstream decisions before they touch production.
- If R-hat refuses to cooperate, reassess model geometry before simply adding iterations.
