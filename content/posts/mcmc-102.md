+++
title = "MCMC 102: Hamiltonian Monte Carlo Chains"
slug = "mcmc-102"
aliases = ["mcmc102"]
date = "2025-10-28T00:00:00Z"
type = "post"
draft = true
math = true
tags = ["mcmc", "hmc", "jax"]
categories = ["posts"]
description = "Turn the leapfrog integrator into a full Hamiltonian Monte Carlo sampler, collect diagnostics, and visualise the resulting chains."
+++
Welcome back to the MCMC mini-series. In [MCMC 101: Leapfrog Integrator]({{< relref "mcmc-101.md" >}}) we built a reversible, volume-preserving leapfrog core in JAX. Now we will wrap it in a full Hamiltonian Monte Carlo (HMC) transition kernel, generate draws, and inspect the resulting chains. Pick up the same Colab notebook and make sure the definitions for `potential_energy`, `grad_U`, and `leapfrog` are still in scope. If you are starting fresh, run through the import and setup cells from part 1 before continuing.

## 1. Build the HMC transition

The first step is to wrap the leapfrog integrator with a Metropolis correction so proposals respect detailed balance. We sample fresh Gaussian momentum, integrate forward with leapfrog, and then accept or reject based on the change in Hamiltonian energy. Keeping `num_steps` static during compilation lets XLA specialise each trajectory length.

```python
from functools import partial
from jax import jit, lax, random
import jax.numpy as jnp

@partial(jit, static_argnames=("num_steps",))
def hmc_step(rng_key, position, step_size, num_steps):
    """Single HMC transition using fresh momentum."""
    key_momentum, key_accept = random.split(rng_key)
    momentum = random.normal(key_momentum, shape=position.shape)

    proposal_q, proposal_p = leapfrog(position, momentum, step_size, num_steps)

    current_U = potential_energy(position)
    current_K = 0.5 * jnp.dot(momentum, momentum)
    proposed_U = potential_energy(proposal_q)
    proposed_K = 0.5 * jnp.dot(proposal_p, proposal_p)

    log_accept = current_U + current_K - proposed_U - proposed_K
    accept_prob = jnp.minimum(1.0, jnp.exp(log_accept))

    next_position = lax.cond(
        random.uniform(key_accept) < accept_prob,
        lambda _: proposal_q,
        lambda _: position,
        operand=None,
    )

    return next_position, accept_prob
```

Line-by-line rundown:

1. `random.split` keeps the momentum draw and accept/reject decision independent.
2. `random.normal` samples the kinetic energy term from a standard Gaussian with the right dimensionality.
3. `leapfrog(...)` proposes a new position–momentum pair using the integration scheme from part 1.
4. The Hamiltonian difference `current_U + current_K - proposed_U - proposed_K` drives the Metropolis acceptance ratio.
5. `lax.cond` switches between the proposal and the current position without leaving compiled XLA code.

## 2. Generate draws and diagnostics

To collect a full chain, scan over independent random keys, discard a warm-up window, and compute basic diagnostics. We will reuse the 2D correlated Gaussian from part 1 so we can compare estimates with the analytical mean and covariance.

```python
import numpy as np

@partial(jit, static_argnames=("num_samples", "num_steps"))
def run_chain(rng_key, initial_position, num_samples, step_size, num_steps):
    def transition(position, key):
        next_position, accept_prob = hmc_step(key, position, step_size, num_steps)
        return next_position, (next_position, accept_prob)

    keys = random.split(rng_key, num_samples)
    final_position, (positions, accept_probs) = lax.scan(
        transition, initial_position, keys
    )
    return positions, accept_probs

num_samples = 2000
num_warmup = 500
step_size = 0.28
num_steps = 5

rng_key = random.PRNGKey(8)
initial_position = jnp.array([3.0, 3.0], dtype=jnp.float32)

trajectory, accept_probs = run_chain(
    rng_key, initial_position, num_samples, step_size, num_steps
)

samples = np.array(trajectory[num_warmup:])
accept_rate = float(np.array(accept_probs[num_warmup:]).mean())
posterior_mean = samples.mean(axis=0)
posterior_cov = np.cov(samples, rowvar=False)

print("Acceptance rate:", round(accept_rate, 3))
print("Posterior mean:", posterior_mean)
print("Posterior covariance:\\n", posterior_cov)
```

You should see diagnostics like:

```text
Acceptance rate: 0.887
Posterior mean: [ 0.995 -1.006]
Posterior covariance:
 [[ 0.834 -0.279]
 [-0.279  0.649]]
```

The empirical mean aligns with the analytical target `[1.0, -1.0]`, and the covariance hugs the inverse precision matrix from part 1:

$$
\Sigma =
\begin{bmatrix}
0.833 & -0.278 \\
-0.278 & 0.648
\end{bmatrix}.
$$

An acceptance rate near 0.85–0.9 indicates the step size and trajectory length are well matched to the curvature of the target distribution.

## 3. Visualise the trajectory

Plot joint samples and coordinate traces to confirm the chain explores the ellipsoid without sticking. Look for dense coverage of the tilted ellipse and quickly mixing traces.

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(samples[:, 0], samples[:, 1], ".", alpha=0.25, markersize=2)
axes[0].set_xlabel("q[0]")
axes[0].set_ylabel("q[1]")
axes[0].set_title("Joint samples")

axes[1].plot(samples[:, 0], label="q[0]")
axes[1].plot(samples[:, 1], label="q[1]")
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("Value")
axes[1].set_title("Trace plot")
axes[1].legend()

fig.tight_layout()
plt.show()
```

Well-mixed traces without long flat segments signal healthy exploration. If acceptance rates collapse or one coordinate sticks, revisit `step_size` or the number of leapfrog steps before moving on.

## Where to go next

- Swap the correlated Gaussian for a banana-shaped target to see how trajectory length shapes acceptance rates.
- Feed the samples through `arviz.ess` and `arviz.rhat` to quantify mixing.
- Replace the hand-written kernel with `blackjax.hmc` or `numpyro.infer.HMC` and compare trace plots and acceptance probabilities.
- Try dual averaging or the No-U-Turn Sampler (NUTS) to automate step-size and path-length adaptation.
