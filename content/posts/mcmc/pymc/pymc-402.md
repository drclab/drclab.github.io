+++
title = "PyMC 402: Gaussian Processes"
slug = "pymc-402"
date = "2027-08-27T00:00:00Z"
type = "post"
draft = false
math = true
tags = ["pymc", "gaussian-processes", "bayesian", "regression", "kernels"]
categories = ["posts"]
description = "A practical guide to Gaussian Processes in PyMC covering mean and covariance functions, GP implementations, and additive GP models based on PyMC's official documentation."
+++

When your unknown isn't a scalar or vector but an entire **function**, Gaussian processes (GPs) offer a principled Bayesian prior over continuous function spaces. This **PyMC 402** guide distills the official [Gaussian Processes documentation](https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/Gaussian_Processes.html), walking through mean and covariance functions, GP implementations in PyMC, and additive GP models that decompose signals into interpretable components.

## 1. GP fundamentals: priors over functions

A Gaussian process defines a distribution over functions $f(x)$ by specifying:

$$f(x) \sim \text{GP}(m(x), k(x, x'))$$

where $m(x)$ is the **mean function** and $k(x, x')$ is the **covariance function** (or kernel). At any finite collection of points, the function values follow a multivariate normal distribution. This marginalization property makes GPs tractable for inference.

The **joint distribution** of observed function values $f(x)$ and predictions at new points $f(x_*)$ is:

$$\begin{bmatrix} f(x) \\ f(x_*) \end{bmatrix} \sim \mathcal{N}\left(\begin{bmatrix} m(x) \\ m(x_*) \end{bmatrix}, \begin{bmatrix} k(x,x') & k(x,x_*) \\ k(x_*,x) & k(x_*,x_*') \end{bmatrix}\right)$$

From this joint, we extract the **marginal** as $f(x) \sim \mathcal{N}(m(x), k(x, x'))$ and the **conditional** (predictive) distribution:

$$f(x_*) \mid f(x) \sim \mathcal{N}\big(k(x_*, x) k(x, x)^{-1} [f(x) - m(x)] + m(x_*), \, k(x_*, x_*) - k(x, x_*) k(x, x)^{-1} k(x, x_*)\big)$$

This machinery—marginalizing over training points and conditioning on new inputs—is what makes GPs so flexible for regression, interpolation, and uncertainty quantification.

**Further reading:** Rasmussen & Williams' [*Gaussian Processes for Machine Learning*](http://www.gaussianprocess.org/gpml/) and David MacKay's [GP introduction](http://www.inference.org.uk/mackay/gpB.pdf) provide comprehensive mathematical foundations.

## 2. Mean and covariance functions

PyMC's GP syntax separates **parameterization** from **evaluation**. When you instantiate a covariance function, you specify its hyperparameters (e.g., lengthscales) and which input dimensions it acts on, but you don't supply data yet.

### Example: Exponentiated quadratic kernel

```python
import pymc as pm

# Operates on columns 1 and 2 of a 3-column input matrix
ls = [2, 5]  # lengthscales for dimensions 1 and 2
cov_func = pm.gp.cov.ExpQuad(input_dim=3, ls=ls, active_dims=[1, 2])
```

Here `input_dim=3` declares the total number of columns in your data matrix, and `active_dims=[1, 2]` specifies which columns this kernel will use. The two-element `ls` gives each active dimension a separate lengthscale, controlling how quickly correlations decay along that axis.

### Combining kernels: algebra for covariance functions

PyMC kernels follow compositional rules that mirror kernel algebra:

- **Sum of kernels:** Captures independent sources of variation
  ```python
  cov_func = pm.gp.cov.ExpQuad(...) + pm.gp.cov.ExpQuad(...)
  ```

- **Product of kernels:** Models interactions or modulation
  ```python
  cov_func = pm.gp.cov.ExpQuad(...) * pm.gp.cov.Periodic(...)
  ```

- **Scaled kernels:** Adjusts overall variance
  ```python
  eta = 1.5
  cov_func = eta**2 * pm.gp.cov.Matern32(...)
  ```

After construction, evaluate the kernel by calling `cov_func(X, X)` to produce a covariance matrix. Because PyMC is built on PyTensor, you can define custom mean and covariance functions by writing PyTensor-compatible Python code. See the [GP Means and Covariances tutorial](https://www.pymc.io/projects/examples/en/latest/gaussian_processes/GP-MeansAndCovs.html) for examples.

## 3. GP implementations in PyMC

PyMC provides several GP classes (`gp.Latent`, `gp.Marginal`, etc.) that handle different inference strategies. All follow the same workflow:

1. **Instantiate** a GP with mean and covariance functions
2. **Add** GPs together for additive models (optional)
3. **Call** a method (`prior`, `marginal_likelihood`, or `conditional`) to create PyMC random variables

### Example: `gp.Latent` for latent function priors

```python
import pymc as pm
import numpy as np

# Define mean and covariance
mean_func = pm.gp.mean.Zero()
cov_func = pm.gp.cov.ExpQuad(input_dim=1, ls=1.0)

# Create GP object
gp = pm.gp.Latent(mean_func=mean_func, cov_func=cov_func)

# Specify prior over function f at inputs X
with pm.Model() as model:
    X = np.linspace(0, 10, 100)[:, None]  # column vector
    f = gp.prior("f", X=X)
    
    # Add likelihood, perform inference...
```

The `prior` method returns a PyMC random variable `f` representing the latent function values at the input locations `X`. If `X` is a PyTensor tensor or PyMC random variable, you must specify the `shape` argument.

### Conditional (predictive) distributions

After fitting the model, construct the predictive distribution at new points $X_*$:

```python
with model:
    f_star = gp.conditional("f_star", X_star)
```

This `conditional` call uses the trained GP to produce predictions with uncertainty bounds. The resulting `f_star` is a standard PyMC random variable that integrates seamlessly with posterior predictive sampling.

**Note:** `gp.Marginal` and similar classes replace the `prior` method with `marginal_likelihood`, which requires additional arguments like observed data `y` and noise parameter `sigma`. The `conditional` method works similarly across implementations but may need different inputs depending on the GP type.

## 4. Additive GPs: decomposing signals

Additive GPs express a complex function as the sum of simpler components, each with its own mean and covariance structure. This is particularly useful for separating trends, seasonal patterns, and noise.

### Mathematical foundation

Consider two independent GPs:

$$f_1(x) \sim \text{GP}(m_1(x), k_1(x, x')), \quad f_2(x) \sim \text{GP}(m_2(x), k_2(x, x'))$$

Their sum $f(x) = f_1(x) + f_2(x)$ is also a GP with:

$$f(x) \sim \text{GP}(m_1(x) + m_2(x), k_1(x, x') + k_2(x, x'))$$

The joint distribution of all components and their sum is a block-structured multivariate normal. PyMC tracks these marginals automatically, enabling you to decompose fitted models into individual contributions.

### Implementation pattern

```python
with pm.Model() as model:
    # Define two independent GPs
    gp1 = pm.gp.Marginal(mean_func1, cov_func1)
    gp2 = pm.gp.Marginal(mean_func2, cov_func2)
    
    # Combine them
    gp = gp1 + gp2
    
    # Fit the combined model
    f = gp.marginal_likelihood("f", X=X, y=y, sigma=sigma)
    
    idata = pm.sample(1000)
```

### Extracting component conditionals

To predict each component separately, provide the full fitting context via the `given` argument:

```python
with model:
    # Conditional for f1 component
    f1_star = gp1.conditional(
        "f1_star", X_star,
        given={"X": X, "y": y, "sigma": sigma, "gp": gp}
    )
    
    # Conditional for f2 component
    f2_star = gp2.conditional(
        "f2_star", X_star,
        given={"X": X, "y": y, "sigma": sigma, "gp": gp}
    )
    
    # Conditional for combined f1 + f2 (no `given` needed)
    f_star = gp.conditional("f_star", X_star)
```

**Key insight:** The combined GP `gp` caches training data when you call `marginal_likelihood`, so its `conditional` needs no extra arguments. Individual components `gp1` and `gp2` were never fitted directly, so their conditionals require the full context in the `given` dict.

This decomposition lets you visualize how much each GP component contributes to the observed data—critical for interpretable models that separate long-term trends from periodic variations or noise.

## 5. Practical workflow summary

1. **Choose kernels** that encode domain knowledge (e.g., `ExpQuad` for smooth functions, `Periodic` for seasonal data, `Matern` for rougher signals)
2. **Combine kernels** additively or multiplicatively to capture complex structure
3. **Instantiate** a GP implementation (`Latent`, `Marginal`, etc.) with your mean and covariance functions
4. **Fit** the model by calling `prior` or `marginal_likelihood` inside a PyMC context
5. **Predict** at new points using the `conditional` method
6. **Diagnose** with ArviZ trace plots and posterior predictive checks

PyMC's GP machinery handles the matrix inversions, Cholesky decompositions, and marginalizations automatically, letting you focus on model design and interpretation.

## 6. Where to go next

- **Explore the [PyMC GP examples gallery](https://www.pymc.io/projects/examples/en/latest/gallery.html#gaussian-processes)** for applied notebooks covering GP regression, classification, and time series decomposition
- **Experiment with custom kernels** by writing PyTensor functions that return valid covariance matrices
- **Scale up** using sparse approximations (`gp.MarginalSparse`, `gp.LatentKron`) for datasets too large for exact GP inference
- **Dive deeper** into additive decompositions to model hierarchical structure or multi-output GPs for correlated outputs

Gaussian processes transform function estimation from curve fitting into principled probabilistic inference. With PyMC's composable kernel algebra and automatic marginalization, you can build interpretable models that quantify uncertainty at every prediction.
