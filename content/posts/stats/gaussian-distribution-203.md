+++
title = "Gauss 203: Linear Combinations in 2D Gaussians"
date = "2026-01-10T00:00:00Z"
type = "post"
draft = true
math = true
tags = ["statistics", "probability", "gaussian", "linear-algebra"]
categories = ["posts"]
description = "Characterizing the distribution of any linear combination of a two-dimensional Gaussian vector."
+++

Gauss 202 derived the log probability of the full 2D sample mean. The next natural question is what happens when downstream code needs **just one scalar summary**, such as $z = x_1 + 2x_2$ or a projected contrast $w^\top \mathbf{x}$. Because estimators, detectors, and filters often score these one-dimensional projections, it helps to have the distribution at our fingertips.

## Setup: 2D Gaussian and a linear probe

Let $\mathbf{x} = (x_1, x_2)^\top \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$ with

$$
\boldsymbol{\mu} = \begin{pmatrix}\mu_1 \\ \mu_2\end{pmatrix}, \qquad
\Sigma = \begin{pmatrix}
\sigma_1^2 & \rho \sigma_1 \sigma_2 \\
\rho \sigma_1 \sigma_2 & \sigma_2^2
\end{pmatrix}, \qquad |\rho| < 1.
$$

Pick any weight vector $w = (a, b)^\top$. The scalar projection $z = w^\top \mathbf{x} = a x_1 + b x_2$ captures a linear combination of the original components.

## Distribution of the projection

Joint Gaussianity plus linearity gives an immediate result:

$$
z \sim \mathcal{N}\bigl(w^\top \boldsymbol{\mu},\; w^\top \Sigma w\bigr).
$$

Expanding the variance for the $2\times2$ case makes the dependencies explicit,

$$
\mathrm{Var}(z) =
a^2 \sigma_1^2 + b^2 \sigma_2^2 + 2ab\,\rho \sigma_1 \sigma_2.
$$

Key observations:

- The **mean** simply projects $\boldsymbol{\mu}$ onto $w$; any bias in $x_1$ and $x_2$ combines linearly.
- The **variance** tucks in the covariance term $2ab\,\rho \sigma_1 \sigma_2$, so positively correlated variables inflate projections that use same-sign weights, while opposite signs cancel.
- The quadratic form $w^\top \Sigma w$ generalizes to higher dimensions, but in 2D you can reason in head: start with marginal variances, then adjust for correlation.

## Geometric reading

Imagine the 2D Gaussian as an ellipse. The projection $z$ measures the shadow of that ellipse along direction $w$. Rotating $w$ sweeps through all possible 1D marginals:

- When $w = (1, 0)$ you recover $x_1$ alone.
- When $w = (\cos\theta, \sin\theta)$ you read the marginal on the line making angle $\theta$ with the $x_1$ axis.
- The **longest** variance occurs when $w$ aligns with the ellipse's principal axis (largest eigenvector of $\Sigma$); the **shortest** when it aligns with the smallest axis.

Understanding these projections is crucial when a downstream algorithm gates on a single statistic: thresholding $x_1 - x_2$ corresponds to $w=(1,-1)$, for instance.

## Worked example

Suppose $\sigma_1 = 2$, $\sigma_2 = 1$, $\rho = -0.3$, and we probe with $w = (0.6, 0.8)$ (a unit vector). Set $\boldsymbol{\mu} = (0.2, -0.1)$ for concreteness.

1. Mean: $w^\top \boldsymbol{\mu} = 0.6(0.2) + 0.8(-0.1) = 0.04$.
2. Variance:
   $$
   w^\top \Sigma w = 0.6^2 (4) + 0.8^2(1) + 2(0.6)(0.8)(-0.3)(2)(1) = 1.44 + 0.64 - 0.576 = 1.504.
   $$
   So $z \sim \mathcal{N}(0.04, 1.504)$.

This tells us everything about the probability of any threshold on $z$. For example, $P(z > 1)$ is just $1 - \Phi\left(\frac{1-0.04}{\sqrt{1.504}}\right)$, no 2D integration required.

## Connection to Gauss 202

Gauss 202 showed that averaging $n$ iid vectors keeps results Gaussian with shrunken covariance $\Sigma/n$. If we now project the sample mean with the same weights $w$, the distribution tightens further:

$$
w^\top \bar{\mathbf{x}} \sim \mathcal{N}\!\left(w^\top \boldsymbol{\mu}, \; \frac{1}{n} w^\top \Sigma w\right).
$$

This is exactly what Kalman filters or generalized least squares exploit: aggregate first, then score a linear functional whose variance is analytically known.

## Simulation sanity check

```python
import numpy as np

mu = np.array([0.2, -0.1])
sigma1, sigma2, rho = 2.0, 1.0, -0.3
Sigma = np.array([[sigma1**2, rho * sigma1 * sigma2],
                  [rho * sigma1 * sigma2, sigma2**2]])
w = np.array([0.6, 0.8])

samples = np.random.multivariate_normal(mu, Sigma, size=100_000)
z = samples @ w

print("Sample mean:", np.mean(z))
print("Sample var :", np.var(z))
print("Theory mean:", w @ mu)
print("Theory var :", w @ Sigma @ w)
```

The output shows the Monte Carlo mean and variance hugging their theoretical counterparts, reinforcing that no approximation was made—the linear combination is **exactly** Gaussian.

## Takeaways

- A linear combination $a x_1 + b x_2$ of a 2D Gaussian remains Gaussian.
- The mean and variance reduce to $w^\top \boldsymbol{\mu}$ and $w^\top \Sigma w$, making probability queries one-dimensional.
- Choosing $w$ selects which ellipse direction you interrogate; aligning with eigenvectors reveals extreme spreads.
- Projecting the sample mean inherits the $1/n$ variance shrinkage from Gauss 202, so linear probes become more certain with larger batches.
