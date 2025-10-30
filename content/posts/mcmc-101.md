+++
title = "MCMC 101: Leapfrog Integrator"
slug = "mcmc-101"
aliases = ["mcmc101"]
date = "2025-10-28T00:00:00Z"
type = "post"
draft = false
math = true
tags = ["mcmc", "leapfrog", "jax"]
categories = ["posts"]
description = "Walkthrough of the leapfrog integrator behind Hamiltonian Monte Carlo, complete with a Colab notebook you can run end-to-end."
+++

Welcome to MCMC 101: Leapfrog Integrator! This guide is written for undergraduate students who have seen multivariable calculus and linear algebra and want a friendlier explanation of Hamiltonian Monte Carlo (HMC). We will define every key term, explain why the leapfrog method is the workhorse inside HMC, and finish with JAX code you can run in Google Colab.

The learning goals are:
- understand the vocabulary that surrounds HMC,
- see the leapfrog update rules and what each symbol means,
- connect the equations to an intuitive "energy landscape" picture, and
- code the integrator from scratch so you can experiment with it yourself.

## Key vocabulary in plain language

- **Markov chain Monte Carlo (MCMC)**: a family of algorithms that generate samples from complicated probability distributions by building a chain of random steps where the next state depends only on the current state.
- **Hamiltonian Monte Carlo (HMC)**: an MCMC method that borrows ideas from physics. It treats the parameters we care about as positions and introduces imaginary momenta so that we can move through the space smoothly instead of by random jumps.
- **Hamiltonian**: the total energy of the system, written as $H(q, p) = U(q) + K(p)$, where $q$ is position and $p$ is momentum. In HMC we want to keep this total energy almost constant while exploring the distribution.
- **Potential energy $U(q)$**: measures how unlikely a position $q$ is. We define it as the negative log of the target density, so low potential energy means "high probability".
- **Kinetic energy $K(p)$**: measures how fast we pretend the particle is moving. We almost always use the simple form $K(p) = \frac{1}{2}p^\top p$, which you can recognise from classical mechanics.
- **Gradient $\nabla U(q)$**: the vector of partial derivatives of $U(q)$. It points uphill in potential energy, so $-\nabla U(q)$ points toward more likely (lower energy) regions.
- **Step size $\varepsilon$**: how far we move the position and momentum in one leapfrog update. Small $\varepsilon$ gives accurate trajectories but requires more steps; large $\varepsilon$ moves faster but risks numerical error.
- **Leapfrog integrator**: a numerical method that alternates small momentum and position updates so that energy stays nearly constant and the path can be reversed exactly.
- **Symplectic / volume-preserving map**: a transformation that keeps "phase space" (the combined position-and-momentum space) volume unchanged; in practical terms it means leapfrog neither squashes nor stretches probability mass, so the Metropolis acceptance step stays simple.
- **Precision matrix $\Lambda$**: the inverse of the covariance matrix. In Gaussian examples it controls how strongly different components move together and determines the curvature of $U(q)$.

Keep these definitions handy; we will refer back to them in each section.

## Leapfrog, one equation at a time

HMC introduces a momentum $p$ for each position $q$ and pretends both follow Hamilton's equations. The leapfrog method approximates this continuous motion using three short, reversible moves:

$$
p_{t + \frac{1}{2}} = p_t - \frac{\varepsilon}{2} \nabla U(q_t) \quad\text{(half momentum kick)},
$$
$$
q_{t + 1} = q_t + \varepsilon p_{t + \frac{1}{2}} \quad\text{(full position move)},
$$
$$
p_{t + 1} = p_{t + \frac{1}{2}} - \frac{\varepsilon}{2} \nabla U(q_{t + 1}) \quad\text{(second half momentum kick)}.
$$

Every symbol now has a job:
- $q_t$ and $p_t$ are the current position and momentum.
- $\nabla U(q_t)$ is the gradient that pushes the particle back toward high-probability regions.
- $\varepsilon$ controls the size of each move.
- The half kicks split the gradient effect evenly before and after the position update. That symmetry makes the method reversible: if you flip the final momentum and apply the same steps backward, you return exactly to where you started.

The word **symplectic** tells us that the method preserves area (in 2-D) or volume (in higher dimensions) in phase space. Because leapfrog is both symplectic and reversible, the Hamiltonian $H(q, p)$ barely changes after many steps, so the Metropolis acceptance rule in HMC almost always accepts the proposal.

### Connecting to the energy picture

To make the algebra feel less abstract, write potential energy using the common Gaussian example:
$$
U(q) = -\log \pi(q) = \frac{1}{2}(q - \mu)^\top \Lambda (q - \mu) + C.
$$
Here $\pi(q)$ is the target density, $\mu$ is the mean, $\Lambda$ is the precision matrix, and $C$ is a constant we can ignore because gradients remove it. When $q$ strays into a low-probability region, $U(q)$ increases and the gradient grows. That gradient acts like a spring force pulling the particle back toward regions where the distribution places more mass. Near the mean, the gradient is small, so the particle coasts with little change in energy.

## Why leapfrog suits HMC

- **High acceptance**: Because leapfrog keeps the Hamiltonian nearly constant, the Metropolis correction step rarely rejects proposals.
- **Easy reversibility**: Flipping the sign of the momentum and replaying the steps exactly undoes the trajectory. This property ensures detailed balance, which is the condition that keeps the Markov chain honest.
- **Volume preservation**: Each sub-step is either a shear in momentum or a shear in position, both with determinant one. Multiplying those determinants shows that the full update has determinant one, so no extra Jacobian terms appear in the acceptance probability.
- **Friendly to autodiff tools**: Autograd and JAX evaluate $\nabla U(q)$ efficiently, letting you run the method on CPU or GPU without fuss.

A good mental model is a puck sliding over a landscape of hills and valleys. The puck speeds up downhill (kinetic energy grows) and slows down uphill (potential energy grows), but the total energy stays about the same, so it glides smoothly across the surface instead of taking random jagged steps.

## 1. Launch a Colab runtime

Open [Google Colab](https://colab.research.google.com/), start a notebook, and install the packages we need. The command below grabs CPU wheels; if you switch Colab to a GPU runtime, follow the message printed by JAX to install the matching CUDA wheel.

```python
%pip install -q -U "jax[cpu]" matplotlib arviz
```

You can confirm whether JAX sees a GPU or CPU with:

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

If you do have a GPU, running `!nvidia-smi` in a code cell will show the model and memory.

## 2. Describe the target distribution

We will test the integrator on a two-dimensional correlated Gaussian so that we know the exact answer in advance. A correlated Gaussian (also called a multivariate normal) uses a covariance matrix $\Sigma$ to encode how variables move together:
$$
\Sigma =
\begin{bmatrix}
\sigma_1^2 & \rho \sigma_1 \sigma_2 \\
\rho \sigma_1 \sigma_2 & \sigma_2^2
\end{bmatrix},
$$
where $\rho$ is the correlation coefficient. The inverse matrix $\Lambda = \Sigma^{-1}$ is the precision matrix. Large off-diagonal entries mean the variables lean in the same direction; that tilt is what makes simple axis-aligned samplers fail and HMC shine.

Here is the setup code. We make $U(q)$ the negative log density and use `jax.grad` to obtain its gradient automatically.

```python
import jax
import jax.numpy as jnp
import numpy as np

precision = jnp.array([[1.4, 0.6],
                       [0.6, 1.8]], dtype=jnp.float32)
target_mean = jnp.array([1.0, -1.0], dtype=jnp.float32)

def log_prob(position):
    """Log-density (up to a constant) of the correlated Gaussian."""
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

The potential energy is zero at the mean, and the gradient vanishes there as expected. If you invert `precision` (you can use `np.linalg.inv`), you recover the covariance matrix:
$$
\Sigma =
\begin{bmatrix}
0.833 & -0.278 \\
-0.278 & 0.648
\end{bmatrix},
$$
which tells us the variables have variances about $0.83$ and $0.65$ with correlation roughly $-0.38$.

## 3. Code the leapfrog integrator

Now translate the three leapfrog equations into JAX. We JIT-compile the function so that Colab fuses the operations into a fast kernel.

```python
from jax import lax, jit

@jit
def leapfrog(position, momentum, step_size, num_steps):
    """Run num_steps leapfrog updates starting from (position, momentum)."""
    # First half-step kick updates momentum using the current gradient.
    momentum = momentum - 0.5 * step_size * grad_U(position)

    def body(_, state):
        q, p = state
        # Full position move using the up-to-date momentum.
        q = q + step_size * p
        # Full momentum kick using the gradient at the new position.
        p = p - step_size * grad_U(q)
        return q, p

    # Repeat the middle updates num_steps - 1 times.
    position, momentum = lax.fori_loop(
        0, num_steps - 1, body, (position, momentum)
    )

    # Final position move and half-step kick mirror the opening moves.
    position = position + step_size * momentum
    momentum = momentum - 0.5 * step_size * grad_U(position)

    # Return the negated momentum so the path can be retraced exactly.
    return position, -momentum
```

Test the function with a made-up starting point. We choose a moderate step size and a small number of steps so you can easily tweak the numbers and watch the effect.

```python
test_q = jnp.array([3.0, 3.0], dtype=jnp.float32)
test_p = jnp.array([0.2, -0.4], dtype=jnp.float32)
proposal_q, proposal_p = leapfrog(test_q, test_p, step_size=0.3, num_steps=5)
print("Proposed q:", proposal_q)
print("Proposed p:", proposal_p)
```

```text
Proposed q: [-0.42972934 -3.5671735 ]
Proposed p: [2.2304685 4.3425517]
```

If you run `leapfrog(proposal_q, -proposal_p, step_size=0.3, num_steps=5)` you land right back at the original `(test_q, test_p)`, demonstrating reversibility. Because every sub-step is volume-preserving, the Metropolis accept/reject rule (the standard MCMC step that decides whether to keep the proposal) only needs to compare the starting and ending Hamiltonian values, so no messy Jacobian terms (determinants that measure volume change) sneak in.

## What's next

You now have a vocabulary-first understanding of the leapfrog integrator and working code you can adapt. In the follow-up post we will wrap this integrator in a full Hamiltonian Monte Carlo sampler, tune the step size and number of steps, and study diagnostics such as acceptance rates and effective sample sizes.

[Continue to MCMC 102 ->]({{< relref "mcmc-102.md" >}})
