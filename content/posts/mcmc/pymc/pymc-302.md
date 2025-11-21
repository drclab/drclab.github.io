+++
title = "PyMC 302: Dimensions Without Tears"
slug = "pymc-302"
date = "2025-11-21T01:00:00Z"
type = "post"
draft = false
math = true
tags = ["pymc", "dims"]
categories = ["posts"]
description = "Turn PyMC dims from metadata into executable structure: build dim-aware variables, mix them with legacy tensors, and vectorize spline models using the experimental pymc.dims API."
+++

PyMC gained string-based `dims` metadata back in 3.9, but the latest [dims module tutorial](https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/dims_module.html) shows how dims now **drive** modeling rather than sitting on the side. This post tracks that story through the local notebook at `content/posts/mcmc/pymc/pymc_302.ipynb`: we recreate its examples, explain what `pymc.dims` builds on (hint: `pytensor.xtensor`), and borrow the case study that vectorizes an unwieldy spline model down to a 40-node graph. Think of this as a survival guide for the experimental API so you can adopt it piecemeal today.

## 1. Why dims matter again

The tutorial recounts how dims debuted quietly in PyMC 3.9 yet became a core ingredient for aligning arrays, exporting InferenceData, and reasoning about multivariate shapes. The experimental `pymc.dims` namespace extends that power: every constructor (`Data`, `Deterministic`, `Potential`), distribution, and math helper there returns an **`XTensorVariable`** that carries named axes all the way through PyTensor. Under the hood it reuses `pytensor.xtensor`, so you get PyTensor graphs that understand strings like `"participant"` instead of raw integers. Those names unlock safer broadcasting, easier debugging, and cleaner handoffs to ArviZ.

## 2. Rewriting a categorical panel model the dim-first way

The notebook opens with a multinomial-response toy model—five participants, 20 trials, three items—first written with the classic API and then rewritten via `pymc.dims`:

```python
with pm.Model(coords=coords) as model:
    observed = pm.Data("observed_response", observed_response_np, dims=("participant", "trial"))
    participant_pref = pm.ZeroSumNormal("participant_preference", n_zerosum_axes=1, dims=("participant", "item"))
    time_effects = pm.Normal("time_effects", dims=("trial", "item"))

    trial_pref = pm.Deterministic(
        "trial_preference",
        participant_pref[:, None, :] + time_effects[None, :, :],
        dims=("participant", "trial", "item"),
    )
    pm.Categorical("response", p=pm.math.softmax(trial_pref, axis=-1), observed=observed, dims=("participant", "trial"))
```

Switching to `pymc.dims` eliminates all the manual broadcasting and axis bookkeeping:

```python
import pymc.dims as pmd

with pm.Model(coords=coords) as dmodel:
    observed = pmd.Data("observed_response", observed_response_np, dims=("participant", "trial"))
    participant_pref = pmd.ZeroSumNormal("participant_preference", core_dims="item", dims=("participant", "item"))
    time_effects = pmd.Normal("time_effects", dims=("item", "trial"))
    trial_pref = pmd.Deterministic("trial_preference", participant_pref + time_effects)

    pmd.Categorical(
        "response",
        p=pmd.math.softmax(trial_pref, dim="item"),
        core_dims="item",
        observed=observed,
    )
```

Both models report identical `point_logps()`, yet the dim-aware version mirrors the algebra in the tutorial: `participant_preference + time_effects` works because `"participant"`, `"trial"`, and `"item"` line up automatically, and `core_dims="item"` tells PyMC which axes belong to the categorical simplex. The payoff is a graphviz view without ad-hoc reshapes or `None` inserts.

## 3. Getting cozy with `XTensorVariable`s

Because `pymc.dims` rides on `pytensor.xtensor`, every random variable is an `XTensorVariable` that stores a tuple of dims. That makes operations like outer sums and renames safe without extra annotations:

```python
dims_normal = pmd.Normal.dist(mu=pmd.math.as_xtensor([0, 1, 2], dims=("a",)), sigma=1)
outer = dims_normal + dims_normal.rename({"a": "b"})
draw = pm.draw(outer, random_seed=rng)  # retains dims=('a', 'b')
```

Deterministics inherit dims automatically, but you can add explicit checks—or even use ellipses—when you want strong guarantees:

```python
with pm.Model(coords={"a": range(2), "b": range(5)}):
    x = pmd.Normal("x", dims=("a", "b"))
    auto = pmd.Deterministic("auto", x + 1)            # infers ('a', 'b')
    explicit = pmd.Deterministic("explicit", x + 1, dims=("a", "b"))
    transposed = pmd.Deterministic("swap", x + 1, dims=("b", "a"))
    suffix = pmd.Deterministic("ellipsis", x + 1, dims=(..., "a"))
```

Those tricks come straight from the "Redundant dims" section of the docs and fulfill two goals: 1) you can confirm PyMC inferred what you expect, and 2) you can intentionally reorder axes via ellipses or explicit tuples when downstream code demands it.

## 4. Mixing dims with the classic API

The tutorial stresses that `pymc.dims` is additive—you can drop into it gradually. When you need to feed a dimmed variable into an older distribution, convert it explicitly:

```python
with pm.Model(coords={"core1": range(3), "core2": range(3), "batch": range(5)}) as mixed_api_model:
    chol, _, _ = pm.LKJCholeskyCov("chol", eta=1, n=3, sd_dist=pm.Exponential.dist(1))
    chol_xr = pmd.as_xtensor(chol, dims=("core1", "core2"))

    mu = pmd.Normal("mu", dims=("batch", "core1"))
    pmd.MvNormal("y", mu=mu, chol=chol_xr, core_dims=("core1", "core2"))
```

To go the other direction—feeding an `XTensorVariable` into a legacy op—call `.values` or `pymc.dims.tensor_from_xtensor()` to recover a vanilla `TensorVariable`. The important bit is that you don’t have to choose between APIs; the doc’s “Combining dims with the old API” section and the notebook both illustrate this hybrid pattern.

## 5. Case study: vectorizing splines with dims

Dims shine when you refactor loop-heavy models. The tutorial’s spline example (ported verbatim in the notebook) starts with a for-loop per group. That unvectorized version produces **806 PyTensor nodes**—far too many for comfort and large enough to trigger the reported bug. Refactoring into NumPy-style tensor math drops the count to **38 nodes**, and swapping in dimmed tensors adds just two more (40 total) while delivering labeled axes and clearer broadcasting:

```python
with pm.Model(coords={"group": range(3), "knot": range(n_knots), "obs": range(N)}) as dims_splines_model:
    x = pmd.Data("x", x_np, dims=("obs",))
    knots = pmd.Data("knots", knots_np, dims=("knot",))
    group_idx = pmd.math.as_xtensor(group_idx_np, dims=("obs",))

    beta0 = pmd.HalfNormal("beta_0", sigma=sigma_beta0, dims=("group",))
    z = pmd.Normal("z", dims=("group", "knot"))

    delta = pmd.math.softmax(z, dim="knot")
    slopes = pmd.concat([beta0, beta0 * (1 - delta.isel(knot=slice(None, -1)).cumsum("knot"))], dim="knot")
    beta = pmd.Deterministic("beta", pmd.concat([beta0, slopes.diff("knot")], dim="knot"))

    X = pmd.math.maximum(0, x - knots)
    mu = (X * beta.isel(group=group_idx)).sum("knot")
    pmd.Normal("y", mu=mu, sigma=pmd.HalfCauchy("sigma", beta=1), observed=y_obs)
```

Everything from the group index (`group_idx`) to the `softmax` call declares its axis names, so the derived `mu` inherits `"obs"` cleanly and you can swap pieces without recoding broadcasting logic. The lesson: dims aren’t just metadata—they encourage you to write vectorized math that PyTensor can simplify aggressively.

## 6. Working with xarray and coordinates

`XTensorVariable`s evaluate to NumPy arrays, but they remember their dims, so converting to xarray is trivial:

```python
from xarray import DataArray

w = pm.dims.Normal.dist(dim_lengths={"a": 3})
outer = w + w.rename({"a": "b"})
DataArray(pm.draw(outer), dims=outer.dims)
```

You can slice (`isel`) and rename dims just like xarray objects before wrapping them back into deterministics:

```python
with pm.Model(coords={"a": [-3, -2, -1], "a*": [-2, -1]}):
    x = pmd.Normal("x", dims=("a",))
    y = pmd.Deterministic("y", x.isel(a=slice(1, None)).rename({"a": "a*"}))
```

One caution from the Q&A: dims keep track of **names**, not coordinate values. If you reverse a variable (`x[::-1]`), you must also update the coordinate array (e.g., via `.assign_coords({"a": [3, 2, 1]})`) or rename it to avoid silently mismatched axes. Think of dims as alignment labels—you still own the coordinate metadata.

## 7. Where to go next

- Open `content/posts/mcmc/pymc/pymc_302.ipynb` to execute every snippet above, including the `model.point_logps()` checks that confirm dimmed and classic models are identical.
- Read the rest of the [dims module tutorial](https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/dims_module.html) for answers on supported distributions, coordinate caveats, and future roadmap notes.
- Experiment with gradually dimming existing PyMC code: wrap your `pm.Data` calls first, then start replacing math helpers with `pmd.math` as you reach for more advanced broadcasting.

Named dimensions stopped being optional commentary; with `pymc.dims` they actively guard against shape bugs, unlock vectorization, and make PyMC graphs feel as transparent as the NumPy code you sketch on paper.
