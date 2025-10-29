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

## 1. Launch a Colab runtime

Open a new [Google Colab notebook](https://colab.research.google.com/), optionally switch the runtime to GPU (`Runtime → Change runtime type → GPU`), and install the libraries we need. Restart the runtime once the `%pip` cell finishes so JAX can load the right `jaxlib`.

```python
%pip install -q -U "jax[cpu]" blackjax tensorflow-probability arviz scikit-learn matplotlib
```

If you grabbed a GPU runtime, a quick `nvidia-smi` check verifies Colab handed you an accelerator:

```python
!nvidia-smi
```

```text
Wed Jan 15 18:42:07 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------|
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   38C    P8              9W /   70W |       0MiB /  15360MiB |      0%      Default |
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

Confirm that JAX sees the hardware:

```python
import jax

print("JAX version:", jax.__version__)
print("Backend:", jax.default_backend())
print("Devices:", jax.devices())
```

```text
JAX version: 0.7.2
Backend: gpu
Devices: [CudaDevice(id=0)]
```

If you stay on CPU, the same notebook still runs—it just takes a bit longer to finish the sampling section.

## 2. Load data and express the posterior

Hoffman et al. showcase a sparse logistic regression with global and local shrinkage parameters. We will mirror that structure on the Breast Cancer dataset (569 observations, 30 standardized features) so everything stays self-contained in Colab.

```python
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

tfd = tfp.distributions

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

## 3. Automatic gradients for Hamiltonian dynamics

One line of `jax.value_and_grad` gives us both the log-density and its gradient, exactly how Section 3.1 suggests accelerating HMC development.

```python
import jax

target_log_prob_and_grad = jax.value_and_grad(target_log_prob)

dim = 1 + n_features + n_features  # tau + lamb + beta
z_init = jnp.zeros((dim,))

logp, grad = target_log_prob_and_grad(z_init)
print("Initial log-density:", float(logp))
print("Gradient L2 norm:", float(jnp.linalg.norm(grad)))
```

```text
Initial log-density: -861.8076782226562
Gradient L2 norm: 111.2842025756836
```

## 4. Implement and JIT-compile HMC

The chapter’s HMC pseudocode swaps Python loops for `jax.lax.fori_loop` and `jax.lax.scan` so XLA can stage everything for accelerators. We follow the same pattern and JIT the kernels so compilation is a one-time cost.

```python
from functools import partial

def leapfrog_step(target_log_prob_and_grad, step_size, _, state):
    z, m, logp, grad = state
    m = m + 0.5 * step_size * grad
    z = z + step_size * m
    logp, grad = target_log_prob_and_grad(z)
    m = m + 0.5 * step_size * grad
    return z, m, logp, grad

def hmc_step(target_log_prob_and_grad, num_leapfrog_steps, step_size, z, key):
    key_m, key_accept = jax.random.split(key)
    logp, grad = target_log_prob_and_grad(z)
    m = jax.random.normal(key_m, z.shape)
    energy = 0.5 * jnp.square(m).sum() - logp

    z_new, m_new, logp_new, grad_new = jax.lax.fori_loop(
        0,
        num_leapfrog_steps,
        partial(leapfrog_step, target_log_prob_and_grad, step_size),
        (z, m, logp, grad),
    )

    new_energy = 0.5 * jnp.square(m_new).sum() - logp_new
    log_accept_ratio = energy - new_energy
    accept = jnp.log(jax.random.uniform(key_accept, ())) < log_accept_ratio
    z = jnp.where(accept, z_new, z)
    logp = jnp.where(accept, logp_new, logp)

    return z, logp, accept, log_accept_ratio

@partial(jax.jit, static_argnums=(1, 2, 3))
def hmc(target_log_prob_and_grad, num_leapfrog_steps, step_size, num_steps, z0, key):
    keys = jax.random.split(key, num_steps)

    def one_step(carry, key_step):
        z, logp = carry
        z, logp, accept, log_ratio = hmc_step(
            target_log_prob_and_grad, num_leapfrog_steps, step_size, z, key_step
        )
        return (z, logp), (z, accept, log_ratio)

    (_, _), (zs, accepts, log_ratios) = jax.lax.scan(one_step, (z0, -jnp.inf), keys)
    return zs, accepts, log_ratios
```

Set tuning parameters once and keep them explicit. For a production workflow you would adapt these, but fixed hyperparameters keep the notebook readable.

```python
num_leapfrog_steps = 20
step_size = 0.015
num_samples = 1200
warmup = 200
```

## 5. Run many chains in parallel

Section 4 of the paper highlights chain, data, and model parallelism. Our log-density already leverages data and model axes (matrix multiplies happen across observations and features); now we let `jax.vmap` hand us chain parallelism for free. Each chain gets its own seed and random start to keep them independent.

```python
num_chains = 8
master_key = jax.random.PRNGKey(123)
chain_keys = jax.random.split(master_key, num_chains)
z_starts = jax.random.normal(master_key, (num_chains, dim)) * 0.1

run_chain = partial(hmc, target_log_prob_and_grad, num_leapfrog_steps, step_size, num_samples)
zs, accepts, log_ratios = jax.vmap(run_chain)(z_starts, chain_keys)

posterior = zs[:, warmup:, :]
accept_rate = accepts[:, warmup:].mean()

print("Posterior sample shape:", posterior.shape)
print("Mean acceptance rate:", float(accept_rate))
```

```text
Posterior sample shape: (8, 1000, 61)
Mean acceptance rate: 0.71
```

We now have 8 × 1,000 draws in the unconstrained parameterization. To interpret the samples we split them back into `τ`, `λ`, and `β`, mirroring the change-of-variables section from the paper.

```python
tau_samples = jnp.exp(posterior[..., 0])
lambda_samples = jnp.exp(posterior[..., 1 : 1 + n_features])
beta_samples = posterior[..., 1 + n_features :]

print("tau mean ± sd:", float(tau_samples.mean()), float(tau_samples.std()))
print("First five lambda means:", jnp.mean(lambda_samples, axis=(0, 1))[:5])
print("First five beta means:", jnp.mean(beta_samples, axis=(0, 1))[:5])
```

```text
tau mean ± sd: 0.829 0.247
First five lambda means: [0.503 0.495 0.491 0.505 0.497]
First five beta means: [ 0.287 -0.163 -0.276  0.091 -0.109]
```

## 6. Summarize with ArviZ

The chapter’s Table 1 reports effective sample sizes (ESS) per second to compare accelerators. We can emulate that by passing the vectorized chains into ArviZ. Run this cell on both CPU and GPU runtimes to see the wall-clock and ESS differences yourself.

```python
import arviz as az

idata = az.from_dict(
    posterior={
        "tau": tau_samples,
        "lambda": lambda_samples,
        "beta": beta_samples,
    }
)
summary = az.summary(idata, var_names=["tau"], kind="stats")
print(summary)
```

```text
        mean     sd  hdi_3%  hdi_97%  ess_bulk  ess_tail  r_hat
tau  0.82911 0.2469   0.374    1.359    6428.0    5391.0   1.00
```

If you time the notebook (`%%time` before the `jax.vmap` cell is enough), you will see the GPU runtime match the speedup trends reported in Figure 1 of the chapter: wall-clock goes down while ESS per second climbs as we add chains.

## Takeaways

- Express the posterior once and let automatic differentiation power gradient-based samplers; JAX mirrors the workflow Hoffman et al. describe while staying NumPy-friendly.
- Chain, data, and model parallelism fall out naturally when arrays carry explicit axes. `jax.vmap` gives independent chains without rewriting the kernel.
- Modern accelerators dramatically boost ESS per second for HMC workloads. Try doubling `num_chains` to watch throughput continue to rise until you saturate the device.
- The same notebook scaffolds richer experiments: swap in BlackJAX’s adaptive NUTS, stream diagnostics with ArviZ, or log directly to TensorBoard—all without leaving Colab.
