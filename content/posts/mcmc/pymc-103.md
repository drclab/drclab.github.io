+++ 
title = "PyMC 103: Linear Regression With Synthetic Plants"
slug = "pymc-103"
aliases = ["pymc101"]
date = "2025-11-17T18:36:20Z"
type = "post"
draft = false
math = false
tags = ["mcmc", "pymc", "bayesian"]
categories = ["posts"]
description = "Step-by-step tour of the PyMC notebook that builds a synthetic plant-growth regression model, runs NUTS, and makes posterior predictions."
+++

This post translates the notebook at `content/ipynb/pymc_101.ipynb` into a narrative walkthrough. We start with synthetic features, wrap them in a PyMC model, recover the parameters with NUTS, and finish by generating fresh predictions. The goal is to help you connect each code cell in the notebook to the corresponding idea in Bayesian workflow.

Learning goals:
- organize covariate data with coordinates so PyMC keeps dimensions straight,
- specify a linear regression with informative names, and
- move from prior predictive checks to posterior inference and posterior predictive draws.

## 0. Confirm your tooling

Open the notebook (or a Colab runtime) and make sure PyMC is available. Recording the version helps when collaborators try to reproduce your results.

```python
import pymc as pm
pm.__version__
```

_Notebook output:_

```text
'5.26.1'
```

I used PyMC 5.x; if you get something older, upgrade before running the rest of the notebook so that `pm.do`, `pm.observe`, and `pm.set_data` behave the same.

## 1. Sample synthetic covariates

The notebook manufactures 100 plant-growth trials with three features—sunlight, water, and soil nitrogen—so we control the ground truth.

```python
seed = 42
x_dist = pm.Normal.dist(shape=(100, 3))
x_data = pm.draw(x_dist, random_seed=seed)

coords = {
    "trial": range(100),
    "features": ["sunlight hours", "water amount", "soil nitrogen"],
}
```

`pm.draw` gives us a NumPy array of shape `(100, 3)`. Passing `coords` to the model keeps downstream arrays well-labeled when we inspect traces or posterior predictive samples.

## 2. Build the generative model

With data in hand, we write the regression in PyMC. The names match the science story: each feature controls plant growth through a coefficient stored in `betas`, and `sigma` captures leftover noise.

```python
with pm.Model(coords=coords) as generative_model:
    x = pm.Data("x", x_data, dims=["trial", "features"])

    betas = pm.Normal("betas", dims="features")
    sigma = pm.HalfNormal("sigma")

    mu = x @ betas
    plant_growth = pm.Normal("plant growth", mu, sigma, dims="trial")
```

`pm.Data` lets us later replace `x` without rebuilding the graph—handy for posterior predictive checks. The matrix multiply `x @ betas` automatically respects the named dimensions thanks to the coordinate metadata.

## 3. Generate synthetic observations

To test inference, we need ground-truth parameters. The notebook pins the coefficients and noise scale, then draws one set of outcomes from the prior predictive distribution.

```python
fixed_parameters = {"betas": [5, 20, 2], "sigma": 0.5}

with pm.do(generative_model, fixed_parameters):
    idata = pm.sample_prior_predictive(random_seed=seed)
    synthetic_y = idata.prior["plant growth"].sel(draw=0, chain=0)
```

_Notebook output:_

```text
Sampling: [plant growth]
```

`pm.do` is PyMC’s intervention helper—it overrides nodes without editing the model definition. Grabbing a single draw from `idata.prior` gives us a deterministic vector we can treat as observed data in the next step.

## 4. Condition on the fake experiment

Now we pretend the synthetic measurements were collected in the lab. `pm.observe` swaps the likelihood’s random variable with observed values, and `pm.sample` runs NUTS to recover the latent parameters.

```python
with pm.observe(generative_model, {"plant growth": synthetic_y}) as inference_model:
    idata = pm.sample(random_seed=seed)
    summary = pm.stats.summary(idata, var_names=["betas", "sigma"])
    print(summary)
```

_Notebook output:_

```text
Initializing NUTS using jitter+adapt_diag...
Multiprocess sampling (4 chains in 4 jobs)
NUTS: [sigma, betas]
Sampling 4 chains for 1_000 tune and 1_000 draw iterations (4_000 + 4_000 draws total) took 2 seconds.
                         mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  \
betas[sunlight hours]   4.973  0.054   4.875    5.076      0.001    0.001   
betas[water amount]    19.963  0.050  19.868   20.059      0.001    0.001   
betas[soil nitrogen]    1.996  0.056   1.897    2.103      0.001    0.001   
sigma                   0.512  0.038   0.443    0.582      0.001    0.001   

                       ess_bulk  ess_tail  r_hat  
betas[sunlight hours]    5458.0    3238.0    1.0  
betas[water amount]      4853.0    2804.0    1.0  
betas[soil nitrogen]     5103.0    3037.0    1.0  
sigma                    4906.0    3038.0    1.0  
```

Expect the posterior means to sit near `[5, 20, 2]` and `0.5`, with tight intervals because we generated 100 trials. The summary table confirms that the sampler can reconstruct the coefficients that produced the data.

## 5. Make posterior predictive checks

After inference we can reuse the fitted model to simulate new plants, possibly with different covariates. The notebook samples three fresh feature vectors, feeds them through the model, and extends the `InferenceData` object with predictive draws.

```python
new_x_data = pm.draw(pm.Normal.dist(shape=(3, 3)), random_seed=seed)
new_coords = coords | {"trial": [0, 1, 2]}

with inference_model:
    pm.set_data({"x": new_x_data}, coords=new_coords)
    pm.sample_posterior_predictive(
        idata,
        predictions=True,
        extend_inferencedata=True,
        random_seed=seed,
    )

pm.stats.summary(idata.predictions, kind="stats")
```

_Notebook output:_

```text
Sampling: [plant growth]
                   mean     sd  hdi_3%  hdi_97%
plant growth[0]  14.230  0.514  13.270   15.203
plant growth[1]  24.415  0.513  23.442   25.365
plant growth[2]  -6.749  0.517  -7.744   -5.808
```

Two important tricks are hiding here:
- `pm.set_data` swaps the design matrix so we can predict on new feature values.
- `extend_inferencedata=True` keeps everything in one `InferenceData`, which simplifies charting in ArviZ or exporting results.

The final `pm.stats.summary` reports the predictive mean and standard deviation for each of the three new trials. Their spreads reflect both observation noise (`sigma`) and posterior uncertainty in the betas.

## Where to go next

## PyMC primitives — quick reference

Below are short explanations and tips for the PyMC building blocks used in the notebook. These are practical notes you can keep next to the code when porting the notebook to your own analysis.

- pm.Model(coords=...)
    - The top-level context that holds random variables and metadata. Passing a `coords` mapping (like `{"trial": range(100), "features": [...]}`) attaches human-friendly names to array dimensions. When variables declare `dims`, PyMC aligns shapes using these coordinates which makes traces and predictions easier to slice and label.

- pm.Data(name, value, dims=...)
    - A mutable data wrapper for observed covariates or design matrices. Use `pm.Data("x", x_array, dims=["trial","features"])` so you can later swap `x` with `pm.set_data(...)` without rebuilding the model. `pm.Data` is the recommended pattern for prediction and cross-validation.

- pm.MutableData
    - An explicit mutable container (alias/variant in some PyMC versions). The notebook uses `pm.Data` + `pm.set_data`, which is the same workflow: declare once, update many times for predictions.

- pm.Distribution vs pm.<RVname>
    - `pm.Normal("betas", dims="features")` defines a random variable in the model and registers it in the model context. `pm.Normal.dist(shape=...)` returns a distribution object that you can sample from outside a model with `pm.draw(...)` (useful for synthetic data generation).

- pm.draw(dist, random_seed=...)
    - Draws a NumPy array from a distribution object (not from the model). We use this to create synthetic covariates and new prediction inputs before attaching them to the model.

- pm.do(model, interventions)
    - A convenience context that temporarily overrides nodes in a model (an intervention). In the notebook we `pm.do(generative_model, fixed_parameters)` so the prior predictive draw behaves like a controlled data generator without permanently changing the model.

- pm.observe(model, {"rv_name": values})
    - Wraps a model with observed data by replacing the likelihood's random variable with the supplied array. This produces an `inference_model` context you can call `pm.sample` from. It’s a compact alternative to rebuilding a model with `observed=` arguments.

- pm.sample(random_seed=..., tune=..., draws=...)
    - Runs the chosen sampler (NUTS by default for continuous models). Typical workflow: initialize, run multiple chains (the notebook uses 4), and inspect `pm.stats.summary` or ArviZ plots for convergence diagnostics (`r_hat`, ESS).

- pm.sample_prior_predictive / pm.sample_posterior_predictive
    - `sample_prior_predictive` generates data under prior assumptions; `sample_posterior_predictive` simulates data from the posterior (or predictive) distribution. Use `extend_inferencedata=True` when you want the draws attached to the same `InferenceData` object for downstream plotting.

- pm.set_data({...}, coords=...)
    - Replaces the arrays stored in `pm.Data` objects. When predicting, call this inside the model context and then run `pm.sample_posterior_predictive` to generate predictions for new covariates.

- pm.Deterministic("name", expr)
    - Register a named derived variable in the trace. Useful for quantities of scientific interest (ratios, contrasts, predicted outcomes) that you want stored in the `InferenceData` for plotting.

- pm.Potential("name", logp)
    - Add an unnormalized log-probability term to the model. Handy for non-standard penalties or for incorporating constraints that are difficult to express as standard distributions.

Tips and idioms

- Prefer short, slug-like variable names (e.g., `plant_growth`) so ArviZ indexing is simpler — spaces are allowed but require bracket-style selection in summaries.
- Use `coords` + `dims` liberally: they make multi-dimensional models (hierarchies, panel data) much easier to reason about and display.
- When debugging model shape errors, print shapes from `pm.draw` and double-check `dims` order; shape mismatches are usually a dims/coords issue.
- Keep the data-declaration (`pm.Data`) separate from model structure so you can re-use the model for prior predictive checks, posterior inference, and out-of-sample prediction with minimal code changes.

- Plug the notebook into `hugo server --buildDrafts --buildFuture` and render the Markdown side-by-side to ensure equations and code blocks look right.
- Replace the synthetic Normal draws with measurements from your project; only the `pm.Data` block and coordinate names need to change.
- Explore the full `idata` object in ArviZ (trace plots, posterior predictive plots) to develop intuition for diagnosing PyMC runs.
