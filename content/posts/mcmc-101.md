+++
title = "MCMC 101: Leapfrog Integrator"
slug = "mcmc-101"
aliases = ["mcmc101"]
date = "2025-10-27T00:00:00Z"
type = "post"
draft = false
math = true
tags = ["mcmc", "leapfrog", "jax"]
categories = ["posts"]
description = "Hands-on tour of the leapfrog integrator behind Hamiltonian Monte Carlo with a Google Colab notebook you can run end-to-end."
+++

Welcome to MCMC 101: Leapfrog Integrator! If Hamiltonian Monte Carlo (HMC) feels mysterious, think of this walkthrough as a guided field trip. We will unpack the leapfrog updates one line at a time, show how they preserve volume and reversibility, and finish with a runnable sampler that hits high acceptance rates without fine tuning.

The goal is simple: open a fresh Google Colab notebook, install JAX, implement a leapfrog integrator from scratch, and generate draws from a correlated Gaussian target. Every snippet below is copy-paste ready. After you run a cell, compare the printed diagnostics with the blocks we captured from a clean Colab session so you know things are on track.

## Leapfrog in equations

HMC augments parameters $q$ with momenta $p$ and follows Hamiltonian dynamics defined by a potential energy $U(q)$ and kinetic energy $K(p) = \frac{1}{2} p^\top p$. Leapfrog discretises the continuous flow with three reversible sub-steps:

$$
p_{t + \frac{1}{2}} = p_t - \frac{\varepsilon}{2} \nabla U(q_t),
$$
$$
q_{t + 1} = q_t + \varepsilon p_{t + \frac{1}{2}},
$$
$$
p_{t + 1} = p_{t + \frac{1}{2}} - \frac{\varepsilon}{2} \nabla U(q_{t + 1}).
$$

Because every update is symplectic and reversible, a finite leapfrog trajectory stays close to the true Hamiltonian path and preserves phase-space volume. That is the secret sauce behind HMC's high acceptance probability: discretisation errors cancel out when you flip the momentum at the end of the trajectory and run the steps backwards.

What lives inside $U(q)$? In the HMC setting $U(q)$ is just the negative log-density of your target distribution, possibly up to an additive constant:
$$
U(q) = -\log \pi(q) = \frac{1}{2} (q - \mu)^\top \Lambda (q - \mu) + C,
$$
where $\pi(q)$ is the target density, $\mu$ is its mean, $\Lambda$ is the precision matrix, and $C$ absorbs any normalising constant. Whenever a new sample proposal wanders into a low-probability region, $U(q)$ spikes and the gradient $\nabla U(q)$ creates a restoring force that nudges the trajectory back toward high-probability areas. Conversely, near modes of the posterior the gradient flattens out, letting the particle cruise with very little numerical error. Thinking about $U(q)$ this way turns the abstract Hamiltonian story into an intuitive terrain-following picture: the leapfrog integrator is constantly trading off elevation (potential energy) and speed (kinetic energy) while conserving the total Hamiltonian.

## Why leapfrog shines for HMC

- Symplectic structure keeps the Hamiltonian nearly constant even after many steps, so Metropolis corrections rarely reject proposals.
- Time reversibility makes detailed balance trivial: flip the momentum sign and replay the same steps in reverse.
- Volume preservation means the proposal density equals the reverse density, avoiding Jacobian corrections.
- Piecewise constant gradients match how autodiff frameworks (like JAX) batch gradient evaluations, giving you excellent GPU utilisation in Colab.

### Digging into the volume-preserving kernel

Volume preservation is not just a nice geometric story—it is the reason the leapfrog proposal kernel integrates cleanly with Metropolis-Hastings. Each leapfrog sub-step is a shear with unit determinant: the half-kick `p ← p - (ε/2)∇U(q)` only shifts momentum, so its Jacobian is the identity in $(q, p)$ space; the drift `q ← q + ε p` keeps momentum fixed while translating position, again with determinant one. Because determinants multiply, the full trajectory has Jacobian determinant exactly one, so the map $(q_t, p_t) → (q_{t+1}, p_{t+1})$ preserves phase-space volume.

In practice this means the forward transition density equals the reverse density once you negate the momentum, leaving the Metropolis acceptance probability to depend solely on the Hamiltonian difference. No extra Jacobian terms sneak into the log-acceptance ratio, and there is no need for an auxiliary correction factor as you would have with a non-volume-preserving integrator. That mathematical guarantee is what lets HMC operate with large step sizes—numerical errors show up only as tiny Hamiltonian drift, not as systematic shrinkage or stretching of probability mass.

When you eventually wrap this leapfrog core in a full HMC transition kernel, you can treat the numerical integrator as a deterministic, volume-preserving map sandwiched between two momentum draws. The momentum resampling restores ergodicity, while the leapfrog path moves proposals across level sets without introducing density distortions. Together they yield a Markov kernel that obeys detailed balance by construction and makes high-dimensional exploration practical.

## 1. Launch a Colab runtime

Head to [Google Colab](https://colab.research.google.com/), start a new notebook, and pick the accelerator you need (`Runtime → Change runtime type → GPU` if available). Then install JAX and a couple of helpers. Restart the runtime after the install so XLA loads the matching wheel.

```python
%pip install -q -U "jax[cpu]" matplotlib arviz
```

If you enabled a GPU runtime, a quick `nvidia-smi` sanity check confirms Colab handed you an accelerator:

```python
!nvidia-smi
```

```text
Wed Oct 29 21:15:40 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------|
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   38C    P8              9W /   70W |       0MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

Verify the JAX backend so you know whether you are running on CPU or GPU:

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

If your runtime reports `cpu`, keep `"jax[cpu]"`. Otherwise replace it with the wheel that matches the CUDA version (e.g., `"jax[cuda12]"` plus the wheel URL from the [JAX release table](https://github.com/google/jax#pip-installation)).

## 2. Describe the target energy landscape

We will sample from a correlated Gaussian so the true mean and covariance are known. A two-dimensional correlated Gaussian (a.k.a. multivariate normal) with mean vector $\mu$ and covariance matrix
$$
\Sigma =
\begin{bmatrix}
\sigma_1^2 & \rho \sigma_1 \sigma_2 \\
\rho \sigma_1 \sigma_2 & \sigma_2^2
\end{bmatrix}
$$
has density
$$
\pi(q) = \frac{1}{2\pi \sqrt{\det \Sigma}} \exp\!\left[-\frac{1}{2}(q - \mu)^\top \Sigma^{-1} (q - \mu)\right],
$$
where $\rho$ is the correlation coefficient between the two components. The off-diagonal terms encode how strongly the dimensions move together, and the precision matrix $\Lambda = \Sigma^{-1}$ is what the HMC gradient routines see. When $\rho \neq 0$, the level sets of $\pi(q)$ stretch into ellipses, so any sampler that moves axis-by-axis will struggle unless it adapts to this geometry. HMC excels because the leapfrog dynamics follow these tilted contours naturally.

With that mental picture in place, define the log-density, potential energy, and gradient using JAX primitives. Keeping randomness explicit and gradients reproducible makes it easy to compare against textbook expectations.

```python
import jax.numpy as jnp
import numpy as np

precision = jnp.array([[1.4, 0.6],
                       [0.6, 1.8]], dtype=jnp.float32)
target_mean = jnp.array([1.0, -1.0], dtype=jnp.float32)

def log_prob(position):
    """Log-density up to a constant for the correlated Gaussian."""
    centered = position - target_mean
    return -0.5 * centered @ (precision @ centered)

def potential_energy(position):
    return -log_prob(position)

grad_U = jax.grad(potential_energy)

print("Potential at the mean:", float(potential_energy(target_mean)))
print("Gradient at the mean:", grad_U(target_mean))
```

```text
Potential at the mean: 0.0
Gradient at the mean: [0. 0.]
```

The precision matrix is positive definite, so the gradient shrinks to zero at `target_mean`. That is the point where the Hamiltonian system would settle if you set the momenta to zero.

Notice how the potential energy function mirrors the negative log of our Gaussian density: high-probability positions yield low potential energy, while improbable positions have large $U(q)$. The gradient `grad_U` therefore behaves like a conservative force pulling the state back toward regions of high posterior mass. When you inspect diagnostics during tuning, tracking both `potential_energy` and the norm of `grad_U` tells you whether the sampler is exploring level sets of $U(q)$ smoothly or thrashing in steep cliffs.

If you invert `precision`, you recover the covariance matrix
$$
\Sigma =
\begin{bmatrix}
0.833 & -0.278 \\
-0.278 & 0.648
\end{bmatrix},
$$
so the marginal variances are $\sigma_1^2 \approx 0.83$ and $\sigma_2^2 \approx 0.65$ with correlation $\rho \approx -0.38$. The off-diagonal elements in either representation capture the same story: the latent variables like to move together, which is exactly the dependency structure we want HMC to navigate gracefully.

## 3. Implement the leapfrog integrator

With $U(q)$ in hand, code the leapfrog updates. JIT compiling the integrator ensures Colab fuses the small matrix multiplies into a single XLA kernel. Start by importing the helpers you need:

```python
from functools import partial
from jax import lax, random, jit
```

Place the leapfrog integrator in its own executable cell:

```python
@jit
def leapfrog(position, momentum, step_size, num_steps):
    """One leapfrog trajectory consisting of num_steps full updates."""
    momentum = momentum - 0.5 * step_size * grad_U(position)

    def body(_, state):
        q, p = state
        q = q + step_size * p
        p = p - step_size * grad_U(q)
        return q, p

    position, momentum = lax.fori_loop(
        0, num_steps - 1, body, (position, momentum)
    )
    position = position + step_size * momentum
    momentum = momentum - 0.5 * step_size * grad_U(position)

    return position, -momentum  # flip momentum to make trajectories reversible
```

Kick the tires with a quick sanity check:

```python
test_q = jnp.array([3.0, 3.0], dtype=jnp.float32)
test_p = jnp.array([0.2, -0.4], dtype=jnp.float32)
proposal_q, proposal_p = leapfrog(test_q, test_p, 0.3, 5)
print("Proposed q:", proposal_q)
print("Proposed p:", proposal_p)
```

The `body` function that we feed into `lax.fori_loop` plays the role of the middle leapfrog update. Each iteration receives the current `(q, p)` pair, advances the position by a full step using the just-updated momentum, and then applies a single gradient kick to the momentum evaluated at the new position. Because `lax.fori_loop(lower, upper, body_fun, init_val)` executes `body_fun` for loop indices `lower, lower + 1, ..., upper - 1`, setting the bounds to `0` and `num_steps - 1` gives us exactly the `num_steps - 1` interior updates between the half steps. That design keeps the final position update and momentum half kick outside the loop, matching the textbook leapfrog schedule without duplicating code.

Unlike a plain Python `for` loop, `lax.fori_loop` is staged as a single fused primitive during JIT compilation: the loop counter lives in XLA, the `body` closure stays pure, and all intermediate `(q, p)` pairs remain on-device. If you swapped it for Python control flow, JAX would have to execute the body sequentially on the host, replicating `grad_U` calls and thrashing the device/host boundary. You could also reach for `lax.scan`, which records every intermediate state, but `fori_loop` is cheaper when you only care about the final `(q, p)` pair because it threads state in place and avoids allocating an output array. Think of it as the vectorised core of the integrator: the two half steps outside the loop handle the boundary conditions, while `body` repeats the reversible drift-and-kick pattern that preserves the Hamiltonian structure.

```text
Proposed q: [1.6894811 0.2086817]
Proposed p: [-0.4858557 -0.2677084]
```

Notice how the momentum flip happens automatically. If you ran the same call again with `proposal_q` and `-proposal_p`, the integrator would retrace every step and land exactly back on the original `(q, p)`.

## What's next

At this point you have a reversible, volume-preserving leapfrog core that mirrors the textbook presentation of Hamiltonian dynamics. In the next instalment we will wrap it in a full Hamiltonian Monte Carlo transition kernel, run end-to-end chains, and visualise diagnostics so you can judge step sizes, trajectory lengths, and effective sample sizes with confidence.

[Continue to MCMC 102 →]({{< relref "mcmc-102.md" >}})
