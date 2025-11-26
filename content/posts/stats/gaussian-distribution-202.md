+++
title = "Gauss 202: Log Probability of 2D Sample Means"
date = "2025-11-27T00:00:00Z"
type = "post"
draft = false
math = true
tags = ["statistics", "probability", "gaussian", "sample-mean"]
categories = ["posts"]
description = "Deriving the log probability of the sample mean for a two-dimensional Gaussian population."
+++

Gauss 201 introduced the $2\times2$ Gaussian density. The natural sequel is to leave raw samples behind and reason about the **sample mean** $\bar{\mathbf{x}}$. Any inference routine that aggregates batches (MLE, Bayesian conjugacy, or even a Kalman filter update) ultimately evaluates the log probability of this mean vector. Here we derive it explicitly for dimension $d = 2$.

## Setup: iid draws and their mean

Let $\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(n)} \in \mathbb{R}^2$ be iid draws from $\mathcal{N}(\boldsymbol{\mu}, \Sigma)$ with

$$
\boldsymbol{\mu} = \begin{pmatrix}\mu_1 \\ \mu_2\end{pmatrix}, \qquad
\Sigma = \begin{pmatrix}
\sigma_1^2 & \rho \sigma_1 \sigma_2 \\
\rho \sigma_1 \sigma_2 & \sigma_2^2
\end{pmatrix}, \qquad |\rho| < 1.
$$

Define the sample mean
$$
\bar{\mathbf{x}} = \frac{1}{n} \sum_{i=1}^{n} \mathbf{x}^{(i)} = \begin{pmatrix}\bar{x}_1 \\ \bar{x}_2\end{pmatrix}.
$$

Because linear combinations of jointly Gaussian variables remain Gaussian, $\bar{\mathbf{x}}$ is also Gaussian with

$$
\bar{\mathbf{x}} \sim \mathcal{N}\left(\boldsymbol{\mu}, \; \frac{1}{n}\Sigma\right).
$$

Intuitively, averaging $n$ iid vectors divides the covariance by $n$, shrinking ellipse axes by a factor of $\sqrt{n}$.

## Log probability in the 2D case

We care about $\log p(\bar{\mathbf{x}} \mid \boldsymbol{\mu}, \Sigma, n)$. Plugging $d = 2$ into the multivariate Gaussian log-density gives

$$
\log p(\bar{\mathbf{x}}) = -\frac{2}{2} \log(2\pi) - \frac{1}{2}\log\left|\frac{1}{n} \Sigma\right|
-\frac{1}{2}(\bar{\mathbf{x}}-\boldsymbol{\mu})^\top\left(n\,\Sigma^{-1}\right)(\bar{\mathbf{x}}-\boldsymbol{\mu}).
$$

Two simplifications are worth highlighting in $2$ dimensions:

1. **Determinant term**: $\left|\frac{1}{n}\Sigma\right| = \frac{1}{n^2}|\Sigma|$, so
   $$
   -\frac{1}{2}\log\left|\frac{1}{n}\Sigma\right| = -\frac{1}{2}\log|\Sigma| + \log n.
   $$
   Averaging $n$ points adds $\log n$ to the log density, matching the intuition that the distribution tightens.
2. **Quadratic form**: $\Sigma^{-1}$ for the $2\times2$ covariance above is
   $$
   \Sigma^{-1} = \frac{1}{(1-\rho^2)}
   \begin{pmatrix}
   \frac{1}{\sigma_1^2} & -\frac{\rho}{\sigma_1 \sigma_2} \\
   -\frac{\rho}{\sigma_1 \sigma_2} & \frac{1}{\sigma_2^2}
   \end{pmatrix}.
   $$
   Multiplying by $n$ scales the precision proportionally to the number of samples, so deviations $\delta = \bar{\mathbf{x}}-\boldsymbol{\mu}$ are penalized $n$ times harder than single observations.

Combining the pieces translates into the compact 2D formula

$$
\log p(\bar{\mathbf{x}}) = -\log(2\pi) - \tfrac{1}{2}\log|\Sigma| + \log n - \tfrac{n}{2}\,\delta^\top \Sigma^{-1} \delta,
$$
where $\delta = \bar{\mathbf{x}} - \boldsymbol{\mu}$.

Everything on the right-hand side is measurable from summary statistics: $\bar{\mathbf{x}}$, the population mean hypothesis $\boldsymbol{\mu}$, the covariance parameters $(\sigma_1, \sigma_2, \rho)$, and the batch size $n$.

## Worked example

Suppose $\boldsymbol{\mu} = (0, 0)$, $\sigma_1 = 2$, $\sigma_2 = 1$, $\rho = 0.4$, and you average $n = 25$ points to get $\bar{\mathbf{x}} = (0.3, -0.1)$. Then

1. $|\Sigma| = \sigma_1^2 \sigma_2^2 (1-\rho^2) = 4 \cdot 1 \cdot (1-0.16) = 3.36$.
2. $\Sigma^{-1}$ evaluates to
   $$
   \frac{1}{0.84}
   \begin{pmatrix}
   0.25 & -0.2 \\
   -0.2 & 1
   \end{pmatrix} =
   \begin{pmatrix}
   0.2976 & -0.2381 \\
   -0.2381 & 1.1905
   \end{pmatrix}.
   $$
3. The quadratic term is $\delta^\top\Sigma^{-1}\delta = (0.3, -0.1)
   \begin{pmatrix}
   0.2976 & -0.2381 \\
   -0.2381 & 1.1905
   \end{pmatrix}
   \begin{pmatrix}0.3 \\ -0.1\end{pmatrix} \approx 0.054$.

Putting it together,
$$
\log p(\bar{\mathbf{x}}) \approx -\log(2\pi) - \tfrac{1}{2}\log 3.36 + \log 25 - \tfrac{25}{2} (0.054),
$$
which evaluates to $\log p(\bar{\mathbf{x}}) \approx 1.51$. The positive value reflects how tightly concentrated the sample mean distribution is: a modest deviation still has reasonable support when $n=25$.

## Implementation-ready snippet

```python
import numpy as np

def log_prob_sample_mean(x_bar, mu, sigma1, sigma2, rho, n):
    Sigma = np.array([[sigma1**2, rho * sigma1 * sigma2],
                      [rho * sigma1 * sigma2, sigma2**2]])
    det = sigma1**2 * sigma2**2 * (1 - rho**2)
    sigma_inv = np.array([[1/sigma1**2, -rho/(sigma1 * sigma2)],
                          [-rho/(sigma1 * sigma2), 1/sigma2**2]]) / (1 - rho**2)
    delta = x_bar - mu
    quad = delta @ (sigma_inv @ delta)
    return (-np.log(2 * np.pi)
            - 0.5 * np.log(det)
            + np.log(n)
            - 0.5 * n * quad)
```

The function mirrors the algebra above and makes it trivial to score candidate means or to embed the result inside larger likelihood calculations.

## Takeaways

- Averaging $n$ iid 2D Gaussians yields another Gaussian with covariance $\Sigma/n$.
- The determinant term gains a $+\log n$ boost, while the quadratic penalty scales with $n$.
- Evaluating $\log p(\bar{\mathbf{x}})$ only requires summary statistics—perfect for batched likelihoods, conjugate Bayesian updates, or diagnostics on Monte Carlo estimates.
- This derivation is the bridge between raw-sample reasoning (Gauss 201) and higher-level inference routines where only means need to be tracked (Gauss 203 and beyond).
