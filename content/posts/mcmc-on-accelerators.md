+++
title = "MCMC on Accelerators"
date = "2025-10-29T00:00:00Z"
type = "post"
draft = false
math = true
tags = ["mcmc", "jax", "bayesian-inference"]
categories = ["posts"]
description = "Follow Hoffman et al.'s patterns for modern hardware by running parallel Hamiltonian Monte Carlo with JAX inside Google Colab."
+++

Modern numerical hardware loves parallel work, yet a lot of Bayesian workflows still idle along on CPUs. After studying Hoffman, Sountsov, and Carroll’s playbook on *Running Markov Chain Monte Carlo on Modern Hardware*, I wanted a notebook that mirrors their patterns but stays approachable for anyone opening a fresh Google Colab tab. The mission here is to set up a hierarchical logistic regression, differentiate it automatically, and run Hamiltonian Monte Carlo (HMC) with many chains in parallel. Every cell below is copy-paste ready for Colab and produces the same outputs you see here.

Along the way we will:

- lean on automatic differentiation so we can express the model once and get gradients “for free” à la Section 3 of the chapter,
- surface the three axes of parallelism the authors emphasize—chains, data, and parameters,
- and finish with diagnostics that confirm the accelerator actually bought us better effective sample sizes per second.

## Hierarchical Logistic Regression in Math

Let $\{(x_i, y_i)\}_{i=1}^N$ be covariate/label pairs with $y_i \in \{0, 1\}$ and $x_i \in \mathbb{R}^p$. A hierarchical logistic regression expresses each binary outcome through a Bernoulli likelihood with a logit that shares information across predictors via global and local shrinkage scales:

$$
\begin{aligned}
  y_i \mid \beta, \lambda, \tau &\sim \text{Bernoulli}(\sigma(\eta_i)), \\
  \eta_i &= x_i^\top (\tau\, \lambda \odot \beta), \\
  \beta_j &\sim \mathcal{N}(0, 1), \\
  \lambda_j &\sim \text{Gamma}(\alpha_\lambda, \beta_\lambda), \\
  \tau &\sim \text{Gamma}(\alpha_\tau, \beta_\tau),
\end{aligned}
$$

where $\sigma(z) = 1 / (1 + e^{-z})$ is the logistic link and $\odot$ denotes elementwise multiplication.

### Understanding the linear predictor

The key expression $\eta_i = x_i^\top (\tau\, \lambda \odot \beta)$ deserves unpacking because it encodes the hierarchical shrinkage structure:

- **$\beta \in \mathbb{R}^p$** are raw regression coefficients drawn independently from standard normals. On their own they carry no shrinkage.
- **$\lambda \in \mathbb{R}^p_+$** are *local* scale parameters (one per feature) drawn from Gamma priors. When $\lambda_j$ is small, feature $j$ gets shrunk aggressively toward zero; when $\lambda_j$ is large, that feature escapes regularization. This lets different predictors have different degrees of influence.
- **$\tau \in \mathbb{R}_+$** is the *global* scale parameter that controls overall sparsity. A small $\tau$ shrinks the entire coefficient vector uniformly, while a large $\tau$ lets the local scales dominate.
- **$\lambda \odot \beta$** is the Hadamard (elementwise) product, yielding $(\lambda_1 \beta_1, \lambda_2 \beta_2, \ldots, \lambda_p \beta_p)^\top$. Each coefficient gets modulated by its own local scale.
- **$\tau (\lambda \odot \beta)$** applies the global scale to every modulated coefficient, producing the final effective regression weights.
- **$x_i^\top (\tau\, \lambda \odot \beta)$** is the standard linear predictor $\sum_{j=1}^p x_{ij} \cdot \tau \lambda_j \beta_j$, but now each term in the sum carries both local and global regularization.

This factorization—standard normal base weights times local scales times a global scale—yields a *horseshoe-like* prior that concentrates mass near zero yet admits heavy tails. When a predictor is truly relevant, the posterior for its $\lambda_j$ will drift away from zero, letting the signal through. Irrelevant predictors stay pinned near zero by small $\lambda_j$ values and the global $\tau$. The Gamma priors on $\lambda$ and $\tau$ ensure positivity and provide flexible tail behavior, making the model robust to high-dimensional binary classification settings where most features are noise.

The rest of this post maps that model into differentiable code that runs efficiently on accelerators.

## 1. Launch a Colab runtime

Open a new [Google Colab notebook](https://colab.research.google.com/), optionally switch the runtime to GPU (`Runtime → Change runtime type → GPU`), and run the first cell to confirm JAX sees the accelerator-bound backend:

```python
import jax

print("JAX version:", jax.__version__)
print("Backend:", jax.default_backend())
print("Devices:", jax.devices())
```

```text
JAX version: 0.8.0
Backend: gpu
Devices: [CudaDevice(id=0)]
```

If you grabbed a GPU runtime, a quick `nvidia-smi` check verifies Colab handed you an accelerator:

```bash
!nvidia-smi
```

```text
Wed Oct 29 15:59:42 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   43C    P0             27W /   70W |     110MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
+-----------------------------------------------------------------------------------------+
```
If you stay on CPU, the same notebook still runs—it just takes a bit longer to finish the sampling section.

## 2. Load data and express the posterior

Hoffman et al. showcase a sparse logistic regression with global and local shrinkage parameters. We will mirror that structure on the Breast Cancer dataset (569 observations, 30 standardized features) so everything stays self-contained in Colab.

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




We code the hierarchical prior the chapter uses: a global Gamma scale `τ`, per-feature Gamma scales `λ`, and standard normal weights `β`. Combining them yields sparse logistic regression coefficients through a Hadamard product.

```python
def joint_log_prob(x, y, tau, lamb, beta):
    lp = tfd.Gamma(0.5, 0.5).log_prob(tau)
    lp += tfd.Gamma(0.5, 0.5).log_prob(lamb).sum()
    lp += tfd.Normal(0.0, 1.0).log_prob(beta).sum()
    logits = x @ (tau * lamb * beta)
    lp += tfd.Bernoulli(logits=logits).log_prob(y).sum()
    return lp
```


NumPyro’s distribution library keeps the log-density numerically robust while staying up to date with modern JAX releases. The Hoffman et al. chapter originally leans on TensorFlow Probability, but that substrate lags behind the current JAX version, so NumPyro serves as a compatible drop-in: its distributions carry stable special-function code, vectorize cleanly across parameters and observations, and remain differentiable via `jax.grad`. That means we get the chapter’s “write it once, run it fast on accelerators” workflow without re-deriving log-density formulas or worrying about underflow.

To keep the walkthrough concrete, the notebook draws a random coefficient vector before evaluating the joint density:

```python
from jax import random

key = random.key(0)           # initialize PRNG key
beta = random.uniform(key, (30,), minval=0.0, maxval=1.0)

print(beta)
```

```text
[0.947667   0.9785799  0.33229148 0.46866846 0.5698887  0.16550303
 0.3101946  0.68948054 0.74676657 0.17101455 0.9853538  0.02528262
 0.6400418  0.56269085 0.8992138  0.93453753 0.8341402  0.7256162
 0.5098531  0.02765214 0.03148878 0.9580188  0.5188192  0.79221416
 0.5522419  0.6113529  0.8931755  0.75499094 0.21164179 0.22934973]
```

```python
joint_log_prob(X, y, 1.0, 1.0, beta)
```

```text
Array(-4869.8623, dtype=float32)
```

Like the authors, we transform constrained variables (the positive scales) to an unconstrained vector before sampling. The log-determinant of the transform is the Jacobian correction the paper calls out in Equation (3.1).

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

The call to `jnp.split(z, [1, 1 + ndims])` slices the flattened parameter vector into the three blocks we need for the model: the first index `[0]` holds the single unconstrained scalar for `τ`, the next `ndims` entries `[1:1+ndims]` capture the per-feature unconstrained scales `λ`, and everything past that index `1 + ndims` becomes the unconstrained regression coefficients `β`. Working with this packed vector simplifies the HMC implementation because the integrator only has to advance one array while we retain the ability to recover each parameter by segment.

Immediately after the split, `unc_tau` still has shape `(1,)`. Calling `unc_tau.reshape([])` collapses that single-element array into a true scalar, which keeps downstream algebra cleaner: exponentiating it with `jnp.exp` returns a scalar `tau`, and broadcasting into expressions like `tau * lamb * beta` behaves exactly like the theoretical derivation rather than relying on implicit length-one dimensions.


## 3. Automatic gradients for Hamiltonian dynamics

One line of `jax.value_and_grad` gives us both the log-density and its gradient, exactly how Section 3.1 suggests accelerating HMC development.

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
## 4. Implement and JIT-compile HMC

The chapter’s HMC pseudocode swaps Python loops for `jax.lax.fori_loop` and `jax.lax.scan` so XLA can stage everything for accelerators. We follow the same pattern and JIT the kernels so compilation is a one-time cost.


Set tuning parameters once and keep them explicit. For a production workflow you would adapt these, but fixed hyperparameters keep the notebook readable.


## 5. Run many chains in parallel

Section 4 of the paper highlights chain, data, and model parallelism. Our log-density already leverages data and model axes (matrix multiplies happen across observations and features); now we let `jax.vmap` hand us chain parallelism for free. Each chain gets its own seed and random start to keep them independent.



We now have 8 × 1,000 draws in the unconstrained parameterization. To interpret the samples we split them back into `τ`, `λ`, and `β`, mirroring the change-of-variables section from the paper.



## 6. Summarize with ArviZ

The chapter’s Table 1 reports effective sample sizes (ESS) per second to compare accelerators. We can emulate that by passing the vectorized chains into ArviZ. Run this cell on both CPU and GPU runtimes to see the wall-clock and ESS differences yourself.



If you time the notebook (`%%time` before the `jax.vmap` cell is enough), you will see the GPU runtime match the speedup trends reported in Figure 1 of the chapter: wall-clock goes down while ESS per second climbs as we add chains.

## Takeaways

- Express the posterior once and let automatic differentiation power gradient-based samplers; JAX mirrors the workflow Hoffman et al. describe while staying NumPy-friendly.
- Chain, data, and model parallelism fall out naturally when arrays carry explicit axes. `jax.vmap` gives independent chains without rewriting the kernel.
- Modern accelerators dramatically boost ESS per second for HMC workloads. Try doubling `num_chains` to watch throughput continue to rise until you saturate the device.
- The same notebook scaffolds richer experiments: swap in BlackJAX’s adaptive NUTS, stream diagnostics with ArviZ, or log directly to TensorBoard—all without leaving Colab.
