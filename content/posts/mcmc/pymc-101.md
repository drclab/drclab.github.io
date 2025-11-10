+++
title = "PyMC 101: Overview and Workflow"
slug = "pymc-101"
date = "2025-11-15T00:00:00Z"
type = "post"
draft = false
math = false
tags = ["pymc", "bayesian", "probabilistic-programming"]
categories = ["posts"]
description = "Guided tour of the official PyMC overview: model contexts, inference engines, predictive checks, and workflow tips for your first probabilistic programs."
+++

The [PyMC Overview](https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/pymc_overview.html) distills what you need to know before building Bayesian models in PyMC 5. This post retells that notebook-sized tour in prose so you can keep the big ideas handy while experimenting in a REPL or notebook.

## 1. Why PyMC is compelling

PyMC wraps mature numerical stacks (Aesara, NumPy, JAX backends) behind a concise Python API. You describe generative stories with familiar control flow, and PyMC translates them into differentiable computation graphs for gradient-based samplers. The overview highlights three reasons to start here:
- **Expressive random variables.** Every common probability distribution is exposed as a first-class Python object, and you can mix discrete and continuous components in the same model context.
- **Automatic inference selection.** `pm.sample()` inspects your model and picks a good default (usually NUTS); if a variable breaks the assumptions, PyMC automatically falls back to adaptive Metropolis or slices that piece out into a compound step method.
- **Tight ecosystem.** PyMC exports everything as an [ArviZ `InferenceData`](https://python.arviz.org/) object, so diagnostics, posterior summaries, and plotting pipelines are one line away.

## 2. Anatomy of a model context

Models in PyMC live inside a context manager. The overview shows a simple linear regression composed from priors, deterministic transformations, and an observed likelihood; assume `X` is a two-column NumPy array holding rainfall and sunlight features aligned with the target vector `y`:

```python
coords = {"obs_id": range(len(y)), "feature": ["rainfall", "sunlight"]}

with pm.Model(coords=coords) as linear_model:
    x = pm.MutableData("x", X, dims=("obs_id", "feature"))
    sigma = pm.HalfNormal("sigma", 1.0)
    beta = pm.Normal("beta", mu=0.0, sigma=5.0, dims="feature")
    alpha = pm.Normal("alpha", 0.0, 5.0)

    mu = alpha + (x * beta).sum(axis=1)
    target = pm.Normal("target", mu=mu, sigma=sigma, observed=y, dims="obs_id")
```

Key takeaways from the doc:
- Pass `coords` and `dims` as soon as possible so tensors stay labeled; ArviZ will recycle those names in diagnostics.
- Use `pm.MutableData`/`pm.Data` for any array you plan to update later—posterior predictive checks become one-liners when inputs are mutable nodes.
- Deterministic transforms (`mu` above) are regular Python expressions; PyMC traces them to keep symbolic gradients intact.

## 3. Running inference

`pm.sample()` orchestrates the inference stack. According to the overview, a typical session looks like this:

```python
with linear_model:
    idata = pm.sample(
        draws=1_000,
        tune=1_000,
        target_accept=0.9,
        random_seed=14,
    )
```

Under the hood PyMC will:
1. Initialize with `pm.init_nuts()` (jitter + adapt_diag) to find a stable mass matrix.
2. Run NUTS for purely continuous models, or build a compound kernel when discrete variables are present.
3. Record draws, tuning stats, divergences, and sampler configuration into `idata`.

If you install the optional JAX extras, the same API can delegate to BlackJAX or NumPyro samplers for GPU-accelerated runs via `pm.sampling_jax.sample_numpyro_nuts()`—the overview points out that parity between backends keeps your modeling code identical.

## 4. Prior and posterior predictive loops

The overview stresses that Bayesian workflow is iterative. PyMC bakes this in with explicit helpers:

```python
with linear_model:
    prior_checks = pm.sample_prior_predictive(random_seed=14)

with linear_model:
    posterior_checks = pm.sample_posterior_predictive(idata, random_seed=14)
```

- **Prior predictive draws** let you see whether your priors generate reasonable synthetic data before touching observations.
- **Posterior predictive draws** reuse fitted traces to simulate new outcomes, so you can validate fit quality or feed downstream decision rules.

Because both functions return `InferenceData` groups (`prior`, `posterior_predictive`, `predictions`), you can pass them directly into ArviZ plots like `az.plot_ppc()` or `az.plot_kde()` and keep the bookkeeping tidy.

## 5. Observing, intervening, and reusing models

One underappreciated section of the overview walks through *structural edits* you can make without redefining the graph:
- `pm.observe(model, {"y": y_data})` conditions the likelihood on observed targets and hands you a ready-to-sample context (great for wrapping into helper functions).
- `pm.do(model, {"x": forced_inputs})` performs Pearl-style interventions so you can reason about counterfactuals or run stress tests with fixed latent values.
- `pm.set_data({"x": new_matrix})` updates mutable nodes for forecasting, cross-validation folds, or posterior predictive simulations.

These hooks encourage a reusable modeling core: define the variables once, then drive experiments by swapping data or interventions instead of copying code between notebooks.

## 6. Inspecting the `InferenceData`

Every call to `pm.sample()` (and the predictive helpers) returns an `InferenceData` bundle. The overview reminds you to:
- Call `pm.stats.summary(idata, var_names=[...])` for quick means, HDIs, ESS, and `r_hat` diagnostics.
- Use `idata.log_likelihood` and `az.loo(idata)` to compare candidate models on held-out metrics.
- Slice by coordinates (`idata.posterior["beta"].sel(feature="sunlight")`) to inspect individual coefficients with their labeled dimensions.

## 7. Practical primer checklist

Drawing from the overview, here is a condensed checklist you can keep beside your notebook when learning PyMC:
1. **Name everything.** Provide `coords`, `dims`, and descriptive variable names so traces are self-documenting.
2. **Start with prior predictive checks** to ensure your story can plausibly generate the data scale you expect.
3. **Lean on defaults.** NUTS with `target_accept≈0.9` works for most continuous models; adjust only when diagnostics demand it.
4. **Use mutable data nodes** for anything you plan to update (forecast horizons, cross-validation folds, interventions).
5. **Export diagnostics early.** ArviZ plots will quickly reveal divergences, low ESS, or multimodality that deserves model revisions before you automate the analysis.

The official overview notebook remains the canonical, executable reference—pair it with this narrative and you will have the conceptual map plus runnable code for your first PyMC explorations.
