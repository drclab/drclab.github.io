+++
title = "PyMC 204: Dimensionality and Coordinates"
slug = "pymc-204"
date = "2026-12-15T01:00:00Z"
type = "post"
draft = false
math = true
tags = ["pymc", "dims", "coords", "best-practices"]
categories = ["posts"]
description = "Understand PyMC's dimensionality bookkeeping system, how to register coordinates, and how to build readable multi-dimensional models."
+++

Dimensionality is the quiet hero of every PyMC model. Shapes determine whether tensors broadcast, sampling speed, and the clarity of generated traces. This post distills the lessons from the accompanying [PyMC 204 notebook](./pymc_204.ipynb) and PyMC's [dimensionality & coordinates guide](https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/dimensionality.html). You'll see how to introspect shapes, decide between implicit and explicit dimensions, and map model axes to named coordinates that make downstream analysis painless.

## 1. Inspect dimensionality before sampling

PyMC exposes a convenient `eval_rv_shapes` helper that resolves tensor shapes after all broadcasting rules are applied. Run it before sampling to surface silent mismatches:

```python
import pymc as pm

with pm.Model() as model:
    mu = pm.Normal("x", mu=0, shape=(3,))  # vector latent
    sigma = pm.HalfNormal("sigma")          # shared scalar scale
    obs = pm.Normal("y", mu=mu, sigma=sigma)

for rv, shape in model.eval_rv_shapes().items():
    print(f"{rv:>10} -> {shape}")
```

Typical output:

```
         x -> (3,)
     sigma -> ()
         y -> (3,)
```

`eval_rv_shapes` mirrors the technique used throughout the PyMC dimensionality tutorial and lets you catch rank mistakes early—long before NUTS raises a shape error.

## 2. Understand implicit vs. explicit shapes

PyMC will try to infer shapes whenever you pass concrete numpy arrays or python sequences as parameters. Being intentional about when you declare `shape=` keeps models readable, especially when collaboration grows.

```python
with pm.Model() as shape_demo:
    pm.Normal("scalar (support)")                # scalar by default
    pm.Normal("vector (implicit)", mu=[1, 2, 3])  # inferred 3-vector
    pm.Normal("vector (explicit)", shape=(4,))    # explicit length 4

pm.model_to_graphviz(shape_demo)
```

The graphviz output (previewed in the notebook) labels each node with its resolved dimensionality. Follow these heuristics from the docs:

- Prefer **implicit shapes** when passing observed arrays; PyMC copies the leading dimensions for you.
- Use **explicit `shape=` (or `size=`)** when the tensor's length is a modeling decision—e.g., number of clusters or spline knots.
- Keep a consistent tuple style (`shape=(n,)`) instead of bare integers; it avoids confusion between scalar vs. length-one vectors.

{{< figure src="/img/pymc-204/shape-demo.svg" alt="Graphviz diagram showing scalar, implicit vector, and explicit vector RVs along with their resolved shapes" caption="Figure 1. Graphviz output from the notebook clarifies exactly how each distribution resolves its dimensionality.">}}

## 3. Attach semantic coordinates to dimensions

Shapes alone are opaque. Coordinates attach labels to each axis so ArviZ summaries and posterior predictive arrays remain self-documenting.

```python
coords = {"year": [2020, 2021, 2022]}
with pm.Model(coords=coords) as revenue_model:
    pm.Normal("profit", mu=0, sigma=1, dims="year")
```

By referencing `dims="year"`, PyMC matches the `profit` random variable to the `year` coordinate. When you inspect posterior predictive draws you'll see `year=2020` rather than `dim_0`, exactly as recommended in the tutorial. Two tips:

1. Register `coords` **once** when instantiating the model so every downstream RV can look up the labels.
2. Use singular dimension names (e.g., `"year"` instead of `"years"`) to keep ArviZ summary keys short and predictable.

{{< figure src="/img/pymc-204/coords-year.svg" alt="Graphviz diagram showing a three-year coordinate dimension connected to the profit RV" caption="Figure 2. The coordinate-aware model annotates each axis with its semantic labels, making it obvious how the `profit` RV maps onto the `year` dimension.">}}

## 4. Mix implicit shapes and explicit dims

You can mix both strategies to keep broadcasting obvious. In the notebook we build a multi-index example:

```python
import numpy as np

coords = {
    "batch": [0, 1, 2, 3],
    "support": ["A", "B", "C"],
}

with pm.Model(coords=coords) as batching_model:
    pm.MvNormal("vector", mu=np.zeros(3), cov=np.eye(3), dims=("support",))
    pm.MvNormal(
        "matrix (implicit)", mu=np.zeros((4, 3)), cov=np.eye(3), dims=("batch", "support")
    )
    pm.MvNormal(
        "matrix (explicit)", mu=np.zeros(3), cov=np.eye(3), shape=(4, 3), dims=("batch", "support")
    )
```

{{< figure src="/img/pymc-204/batch-support.svg" alt="Graphviz diagram showing batch and support coordinates with both implicit and explicit matrix RVs" caption="Figure 3. By naming both the batch and support axes we keep every latent aligned with the observed data tensor and avoid silent broadcasting mistakes.">}}

Key takeaways mirrored from the guide:

- `dims` **only names axes**; PyMC still needs to know the size of each axis, either from coordinates (`len(coords["support"])`) or from `shape=`.
- Explicit `shape=(4, 3)` is useful when the mean parameter cannot easily reveal the intended size (e.g., when it's learned).
- Assigning `dims` to latent variables ensures posterior outputs share coordinate labels with observed data, making `az.plot_forest` and `az.summary` align automatically.

## 5. Debugging checklist

Whenever you hit `ValueError: Input dimension mis-match`, walk through this checklist inspired by the docs:

1. **Add `eval_rv_shapes()`** after model creation; confirm every RV has the expected tuple.
2. **Inspect coordinates** with `model.coords` and verify every dim referenced in `dims=` exists.
3. **Prefer tuples** in `shape` even for one-dimensional tensors.
4. **Align coordinate lengths** with numpy array axes—reshape or transpose upstream data instead of trusting broadcasting.
5. **Render the model graph** with `pm.model_to_graphviz(model)`; it annotates node ranks and highlights stray dimensions visually.

Mastering dimensionality and coordinates unlocks clearer mental models, fewer runtime errors, and richer labeled outputs. Treat this bookkeeping as part of the modeling workflow—not as an afterthought once sampling fails.
