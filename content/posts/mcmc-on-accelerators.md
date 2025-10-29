+++
title = "MCMC on Accelerators"
date = "2025-10-29T00:00:00Z"
type = "post"
draft = true
math = true
tags = ["mcmc", "jax", "bayesian-inference"]
categories = ["posts"]
description = "Follow Hoffman et al.'s patterns for modern hardware by running parallel Hamiltonian Monte Carlo with JAX inside Google Colab."
+++

Modern numerical hardware loves parallel work, yet many Bayesian workflows still idle along on CPUs. After studying Hoffman, Sountsov, and Carroll’s playbook on *Running Markov Chain Monte Carlo on Modern Hardware*, the notebook in this repo now walks through the critical pieces that make Hamiltonian Monte Carlo (HMC) accelerator-ready: install the right JAX tooling, express a hierarchical logistic regression, move everything to an unconstrained parameterization, and wire up the leapfrog integrator. The cells below mirror `MCMC_Jax.ipynb` and remain copy-paste friendly for Google Colab.

## 1. Bootstrap the runtime

Colab images do not ship with NumPyro, so we start by installing it and confirming the deployed JAX build.

```python
!pip install numpyro

import jax

print("JAX version:", jax.__version__)
```

If you requested a GPU runtime, `nvidia-smi` confirms that the accelerator is available:

```python
!nvidia-smi
```

Remaining on CPU is fine—the later sampling code will simply take longer.

## 2. Load and standardize the data

Hoffman et al. highlight sparse logistic regression with hierarchical shrinkage. The notebook stays self-contained by using scikit-learn’s Breast Cancer dataset.

```python
import jax.numpy as jnp
import numpyro
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

tfd = numpyro.distributions

dataset = load_breast_cancer()
scaler = StandardScaler()
X = scaler.fit_transform(dataset.data).astype("float32")
y = dataset.target.astype("float32")

X = jnp.asarray(X)
y = jnp.asarray(y)
n_features = X.shape[1]
print("Dataset shape:", X.shape, y.shape)
```

```text
Dataset shape: (569, 30) (569,)
```

Standardizing the predictors keeps the Gamma shrinkage priors well behaved when we later evolve the system with Hamiltonian dynamics.

## 3. Write the hierarchical log density

The prior mirrors the chapter: a Gamma global scale `τ`, per-feature Gamma local scales `λ`, and standard normal base weights `β`. NumPyro’s distribution namespace keeps everything differentiable and numerically stable.

```python
def joint_log_prob(x, y, tau, lamb, beta):
    lp = tfd.Gamma(0.5, 0.5).log_prob(tau)
    lp += tfd.Gamma(0.5, 0.5).log_prob(lamb).sum()
    lp += tfd.Normal(0.0, 1.0).log_prob(beta).sum()
    logits = x @ (tau * lamb * beta)
    lp += tfd.Bernoulli(logits=logits).log_prob(y).sum()
    return lp
```

To keep things concrete we draw a random coefficient vector before evaluating the joint density. JAX’s functional PRNG API makes reruns painless.

```python
from jax import random

key = random.key(0)
beta = random.uniform(key, (30,), minval=0.0, maxval=1.0)
print(beta)

joint_log_prob(X, y, 1.0, 1.0, beta)
```

```text
[0.947667   0.9785799  0.33229148 0.46866846 0.5698887  0.16550303
 0.3101946  0.68948054 0.74676657 0.17101455 0.9853538  0.02528262
 0.6400418  0.56269085 0.8992138  0.93453753 0.8341402  0.7256162
 0.5098531  0.02765214 0.03148878 0.9580188  0.5188192  0.79221416
 0.5522419  0.6113529  0.8931755  0.75499094 0.21164179 0.22934973]
Array(-4869.8623, dtype=float32)
```

## 4. Switch to an unconstrained parameterization

HMC works best when every parameter lives in $\mathbb{R}^n$. We therefore exponentiate the positive scales and add the appropriate log-determinant Jacobian correction.

```python
def unconstrained_joint_log_prob(x, y, z):
    ndims = x.shape[-1]
    unc_tau, unc_lamb, beta = jnp.split(z, [1, 1 + ndims])
    unc_tau = unc_tau.reshape([])
    tau = jnp.exp(unc_tau)
    lamb = jnp.exp(unc_lamb)
    ldj = unc_tau + unc_lamb.sum()
    return joint_log_prob(x, y, tau, lamb, beta) + ldj

target_log_prob = lambda z: unconstrained_joint_log_prob(X, y, z)
```

Packing `τ`, `λ`, and `β` into a single vector lets the integrator evolve one array while keeping the slices readable when we need per-parameter summaries. Reshaping `unc_tau` into a scalar avoids accidental broadcasting quirks later.

## 5. Autodiff for Hamiltonian dynamics

One line of `jax.value_and_grad` provides both the log density and its gradient—exactly the pattern Section 3 of the chapter recommends for hardware-friendly HMC.

```python
target_log_prob_and_grad = jax.value_and_grad(target_log_prob)

dim = 1 + n_features + n_features  # tau + lamb + beta
z_init = jnp.zeros((dim,))

logp, grad = target_log_prob_and_grad(z_init)
print("Initial log-density:", float(logp))
print("Gradient L2 norm:", float(jnp.linalg.norm(grad)))
```

```text
Initial log-density: -465.95599365234375
Gradient L2 norm: 803.6372680664062
```

Lastly, since we're interested in just the posterior of this model, we can condition it via partially applying the `x` and `y` arguments with some data. For this example, we'll use the German Credit dataset (Hofmann, 1994), $N_{\text{observations}} = 1000, N_{\text{features}} = 24$:

```python
from functools import partial

target_log_prob = partial(unconstrained_joint_log_prob, x_data, y_data)
```

The resulting `target_log_prob` function represents an unnormalized version of the posterior $p(z \mid x, y)$; it is a function only of $z$.

So far, the model code does not look so different from R or plain NumPy code. Now, we will show capabilities which are more unique to JAX and similar toolkits. First, since HMC requires the gradient of the log-density, we will use JAX to compute it using automatic differentiation. JAX exposes this feature as a one-line program transformation:

```python
import jax

target_log_prob_and_grad = jax.value_and_grad(target_log_prob)
tlp, tlp_grad = target_log_prob_and_grad(z)
```

## 6. Implement the leapfrog step

With gradients in hand we can code the symplectic integrator that powers HMC. The leapfrog kernel mirrors Algorithm 1 in the chapter and sets the stage for `jax.lax.scan` and parallel chains in a follow-up revision.

```python
def leapfrog_step(target_log_prob_and_grad, step_size, i, leapfrog_state):
    z, m, tlp, tlp_grad = leapfrog_state
    m += 0.5 * step_size * tlp_grad
    z += step_size * m
    tlp, tlp_grad = target_log_prob_and_grad(z)
    m += 0.5 * step_size * tlp_grad
    return z, m, tlp, tlp_grad
```

**What happens inside:** the first half-step nudges the momentum using the current gradient, the full step updates position with that momentum, and the second half-step refreshes the momentum with the gradient at the new position. The symmetry keeps the integrator reversible and energy preserving, which is key for high acceptance rates once we wrap this kernel in a Metropolis correction.

## Where we go next

The current notebook stops right after the leapfrog integrator so the remaining pieces—looping leapfrog updates with `jax.lax.fori_loop`, vectorizing chains with `jax.vmap`, and summarizing draws with ArviZ—can build on a tested base. In the meantime you can experiment with different step sizes, sanity-check gradients, or drop in BlackJAX’s NUTS kernel to compare against the hand-rolled integrator.
