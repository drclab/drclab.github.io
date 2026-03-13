+++
title = "MCMC 103: Exploratory Analysis of Bayesian Models with ArviZ"
slug = "mcmc-103"
aliases = ["mcmc103"]
date = "2026-03-15T00:00:00Z"
type = "post"
draft = false
math = true
tags = ["mcmc", "arviz", "bayesian", "diagnostics"]
categories = ["posts"]
description = "A deep dive into exploratory analysis of Bayesian models (EABM) and the redesigned, highly modular architecture of the ArviZ Python library."
+++

Welcome to MCMC 103! In [MCMC 101]({{< relref "mcmc-101.md" >}}) and [MCMC 102]({{< relref "mcmc-102.md" >}}), we explored the mechanics of Hamiltonian Monte Carlo and the leapfrog integrator, focusing on how to generate samples from a target distribution. In this post, we take the crucial next step: figuring out what to do with those samples. 

When working with Bayesian models, running the inference algorithm is only part of the job. You also need to diagnose the quality of the Markov chain Monte Carlo (MCMC) samples, critique your model's fits, compute various predictive checks, and compare models. The broader term for these activities is **Exploratory Analysis of Bayesian Models (EABM)**.

In this post, we'll examine EABM through the lens of **ArviZ**, a popular Python package that has recently undergone a major architectural redesign to become more modular, flexible, and powerful. ArviZ seamlessly integrates with essentially every major probabilistic programming language in Python, including PyMC, Stan (via cmdstanpy), NumPyro, Pyro, Bambi, and emcee.

## The Need for EABM and a Unified Library

Effectively leveraging Bayesian statistics in applied settings requires robust toolsets for uncertainty visualization, sampling diagnostics, model comparison, and checking. Rather than leaving each probabilistic programming language (PPL) to build its own plotting and diagnostic frameworks, **ArviZ** serves as a backend-agnostic hub. 

The Bayesian ecosystem continuously rapidly evolves. For example, modern implementations for computing effective sample size (`ess`), $\hat{R}$ (`rhat`), leave-one-out cross-validation (`loo`), and model comparisons have advanced significantly over the years. By decoupling the diagnostic logic from the specific inference engine, ArviZ ensures that you have access to state-of-the-art methodology regardless of the sampler that produced your draws.

## A Modular Architecture

The redesigned version of ArviZ emphasizes user control and tight integrations. Previously constrained by tight coupling, the new suite separates the library into multiple independent modular packages (`arviz`, `arviz-base`, `arviz-stats`, and `arviz-plots`).

This shift brings major improvements to three key areas:

### 1. Data Processing and I/O with Xarray DataTrees

The previous iterations of ArviZ relied on a custom `InferenceData` class to organize high-dimensional outputs in a structured, labeled format. EABM data is fundamentally hierarchical—containing unobserved parameter draws, prior predictive samples, observed data, and sample stats like step sizes or divergences.

The new architecture replaces custom objects with the powerful **`DataTree`** class from the `xarray` library. The `DataTree` handles rich nesting entirely natively and automatically supports all robust I/O formats baked into `xarray`. 

### 2. Tailored Statistical Interfaces

Statistical computations can quickly get complicated, depending heavily on the user's expertise level. ArviZ solves this by providing functions through two distinct parallel interfaces:

- **Low-level array interface:** Exposes functions with only `numpy` and `scipy` dependencies. This is perfect for library developers and advanced analysts who need to manage their own arrays, metadata, and axes.
- **High-level `xarray` interface:** Designed for standard end-users working with PPLs. It intelligently automates common tasks by using the named dimensions and metadata embedded right inside the `DataTree`.

### 3. Redesigned, Multi-Backend Plotting

Good exploratory analysis relies on informative visualizations. The rewrite brings modularity to the plotting functions across multiple levels:

- **High Level ("Batteries-included"):** ArviZ provides 37 ready-to-use plots to handle diagnostics, predictive checks, and trace analysis with sensible defaults.
- **Intermediate Level:** The new `PlotCollection` class drastically lowers the barrier to customizing default charts or building advanced, faceted visual models without duplicating layout logic.
- **Low Level:** Complete separation between the computational statistical work and the plotting logic. This clean break enables first-class support for multiple plotting backends, allowing users to choose from **matplotlib**, **Bokeh**, or **plotly**.

## Putting it Together: ArviZ in Action

Let's look at how seamless diagnostic functions are applied to MCMC output using the various interfaces.

If you are writing a custom sampler or strictly working with NumPy arrays, you can use the low-level functions directly. You just need to let ArviZ know how your array dimensions are structured (e.g., chains vs. draws):

```python
import numpy as np
from arviz_stats.base import array_stats

rng = np.random.default_rng()
# Creating an array: (chain, draw, variable)
samples = rng.normal(size=(4, 1000, 2)) 

# Compute Effective Sample Size
array_stats.ess(samples, chain_axis=0, draw_axis=1)
```

However, standard PPL users will mostly interact with the high-level API. When samples are loaded into an `xarray` `DataTree`, ArviZ automatically tracks the dimensions.

```python
import arviz as az# Initialize via DataTree
dt_samples = az.convert_to_datatree(samples)
az.ess(dt_samples)

# A sample "batteries-included" plot with customizations
az.style.use('arviz-variat')
dt = az.load_arviz_data("centered_eight")

pc = az.plot_dist(
    dt,
    kind="dot",      # Change default KDE to a quantile dot plot
    visuals={"dist": {"marker": "C6"}, "point_estimate_text": False},
    aes={"color": ["school"]}
)
pc.add_legend("school", loc="outside right upper")
```

The EABM workflow is critical to responsible statistical modeling. With the improved modularity and powerful xarray-backed internals, ArviZ equips you perfectly to critique, validate, and understand the output of whatever sampler you use. Happy exploring!
