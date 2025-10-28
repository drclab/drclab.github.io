+++
title = "Stan 103: Effective Sample Size"
date = "2025-10-24T00:00:00Z"
type = "post"
draft = false
math = true
tags = ["stan", "bayesian diagnostics", "ess"]
categories = ["posts"]
description = "Understanding bulk and tail effective sample sizes in Stan, how they are computed, and the levers we pull when they drop below healthy thresholds."
+++

Effective sample size (ESS) lives next to R-hat on every Stan dashboard we maintain. Stan 101 taught the sampler basics, Stan 102 covered why we trust R-hat; this third lesson explains what ESS measures, why Stan splits it into bulk and tail variants, and the workflow we follow when it nosedives.

## Why effective sample size matters

- Converts autocorrelated draws into an intuitive “independent draws” scale.
- Highlights parameters that mix slowly even when R-hat looks clean.
- Guides Monte Carlo error estimates for means, quantiles, and functions of the posterior.
- Guards against overconfident intervals produced from sticky chains.

When ESS stays high, Monte Carlo noise shrinks like $1/\sqrt{N_{\text{eff}}}$, keeping posterior summaries sharp without rerunning the model.

## Stan’s bulk vs tail ESS

Stan reports two numbers per parameter: `ess_bulk` for central tendency and `ess_tail` for extreme quantiles. Both stem from the autocorrelation-adjusted formula

$$
N_{\text{eff}} = \frac{MN}{1 + 2 \sum_{t=1}^{\infty} \rho_t},
$$

where $M$ is the number of chains, $N$ is the post-warmup draws per chain, and $\rho_t$ is the rank-normalized autocorrelation at lag $t$. Bulk ESS downweights the central ranks, while tail ESS is tuned to capture heavy tails and extreme quantiles. If either value drops below $0.1MN$, Stan emits a warning because your posterior summaries now carry large Monte Carlo error.

## Inspecting ESS in CmdStanPy

CmdStan and CmdStanPy both compute ESS during `summary()`. Pull both bulk and tail variants so dashboards can highlight shaky parameters.

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
ess = summary.loc[:, ["ESS_bulk", "ESS_tail"]].reset_index(names="variable")
low_ess = ess[(ess["ESS_bulk"] < 200) | (ess["ESS_tail"] < 200)]
low_ess
```

For richer diagnostics, convert to ArviZ and compute Monte Carlo error directly:

```python
import arviz as az

idata = fit.to_inference_data()
az.ess(idata, method="bulk")
az.ess(idata, method="tail")
```

ArviZ aligns with Stan’s rank-normalized ESS, making it safe to mix CLI summaries with notebook analyses.

## Diagnosing low ESS

- **Strong posterior correlations:** Narrow funnels or ridges force the sampler to move slowly along the constrained geometry.
- **Multimodality:** Chains that hop between modes inflate autocorrelation even if R-hat passes.
- **Poor tuning:** Short warmup or mis-specified step size leaves the sampler making tiny, correlated moves.
- **Heavy tails:** Parameters with diffuse priors can wander far, stretching the rank-based diagnostic.

Trace plots and pair plots usually expose whether correlations, funnels, or label switching drive the collapse.

## How to boost effective sample size

1. Reparameterize problematic blocks (e.g., non-center hierarchical scales, rotate funnels to unit scale).
2. Increase warmup so adaptation locks into a geometry that supports larger, less correlated moves.
3. Tighten or reshape priors to avoid implausible tails that soak up iterations without adding signal.
4. Raise `max_treedepth` only after geometry fixes; deeper trees by themselves seldom rescue ESS if correlations remain.
5. Add informative transformations or marginalizations (e.g., integrate out nuisance parameters) to trim autocorrelation sources.

Monitor both bulk and tail ESS after each change—tail ESS often lags behind and reveals persistent tail stickiness.

## Key reminders

- Bulk ESS controls Monte Carlo error for means; tail ESS does the same for quantiles—watch both.
- Treat `ESS_bulk < 400` or `ESS_tail < 400` as a nudge to revisit geometry before rerunning expensive chains.
- R-hat near `1.0` is necessary but not sufficient; ESS confirms the sampler gathered enough independent information.
- Automate ESS checks alongside R-hat so pipelines fail fast when either diagnostic leaves the safe zone.
