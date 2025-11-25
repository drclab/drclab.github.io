+++
title = "Gauss 201: Multivariate Gaussian in Two Dimensions"
date = "2025-11-26T00:00:00Z"
type = "post"
draft = false
math = true
tags = ["statistics", "probability", "gaussian", "multivariate"]
categories = ["posts"]
description = "A hands-on guide to the two-dimensional Gaussian, highlighting means, covariance structure, and conditioning in terms of x1 and x2."
+++

You'll meet multivariate Gaussians in every probabilistic model, but most practical intuition comes from the two-dimensional case. Limiting the dimensionality to $(x_1, x_2)$ keeps the algebra manageable while still revealing how means, variances, and correlations interact.

## The 2D density

A centered two-dimensional Gaussian with mean vector $\mu = (\mu_1, \mu_2)$ and covariance matrix

$$
\Sigma = 
\begin{pmatrix}
\sigma_{1}^2 & \rho \sigma_{1}\sigma_{2} \\
\rho \sigma_{1}\sigma_{2} & \sigma_{2}^2
\end{pmatrix}
$$

has density

$$
p(x_1, x_2) = \frac{1}{2\pi |\Sigma|^{1/2}}
\exp\Bigg(-\frac{1}{2}
\begin{pmatrix}x_1 - \mu_1 \\ x_2 - \mu_2\end{pmatrix}^\top
\Sigma^{-1}
\begin{pmatrix}x_1 - \mu_1 \\ x_2 - \mu_2\end{pmatrix}
\Bigg),
$$

where $|\Sigma| = \sigma_1^2\sigma_2^2(1-\rho^2)$ for $|\rho|<1$. Everything you need to evaluate or sample the distribution fits into that $2\times2$ covariance.

## What $\Sigma$ controls

- **Marginal spreads**: $\sigma_1^2$ and $\sigma_2^2$ give the variances of $x_1$ and $x_2$ individually.
- **Correlation**: $\rho$ determines how the coordinates co-move. $\rho = 0$ yields axis-aligned ellipses; $|\rho| \to 1$ squeezes the ellipses along a line.
- **Precision**: The inverse $\Lambda = \Sigma^{-1}$ is sometimes easier to estimate. In 2D,
  $$
  \Lambda = \frac{1}{1-\rho^2}
  \begin{pmatrix}
  \frac{1}{\sigma_1^2} & -\frac{\rho}{\sigma_1\sigma_2} \\
  -\frac{\rho}{\sigma_1\sigma_2} & \frac{1}{\sigma_2^2}
  \end{pmatrix}.
  $$
  Large diagonal entries mean the corresponding variable is well-constrained.

## Geometry of contours

Level sets of $p(x_1, x_2)$ are ellipses defined by $(\mathbf{x}-\mu)^\top \Sigma^{-1} (\mathbf{x}-\mu) = c$. The eigenvectors of $\Sigma$ give the ellipse axes; eigenvalues give squared axis lengths. Plotting a few ellipses is the fastest way to sanity-check whether your inferred covariance matches simulated data.

## Marginals and conditionals stay Gaussian

Even with only two variables, the closure properties are powerful:

- **Marginals**: Integrating out $x_2$ leaves $x_1 \sim \mathcal{N}(\mu_1, \sigma_1^2)$ (and vice versa). You can study each coordinate independently if correlation doesn't matter.
- **Conditionals**: Conditioning on $x_2$ fixes the mean of $x_1$ to a linear function of the observed value:
  $$
  x_1 \mid x_2 = a \sim \mathcal{N}\Big(\mu_1 + \rho \frac{\sigma_1}{\sigma_2}(a - \mu_2),
  (1-\rho^2)\sigma_1^2\Big).
  $$
  This linear/Gaussian structure underlies Kalman filters and Gaussian processes.

  When you observe $x_2 = a$, the conditional distribution of $x_1$ remains Gaussian but with an updated mean that shifts linearly with $a$. The slope of this shift is $\rho \frac{\sigma_1}{\sigma_2}$, reflecting how strongly $x_1$ correlates with $x_2$ and their relative scales. The conditional variance $(1-\rho^2)\sigma_1^2$ is always smaller than the marginal $\sigma_1^2$ (unless $\rho=0$), because observing $x_2$ provides information that reduces uncertainty about $x_1$.

  Intuitively, if $\rho > 0$ and $a > \mu_2$, we expect $x_1$ to be above $\mu_1$ on average. This property makes bivariate Gaussians ideal for prediction: given a value of one variable, you can linearly predict the other with quantifiable uncertainty.

  **Connections to applications:**
  - **Kalman filters**: These recursively update state estimates (e.g., position and velocity) using linear dynamics and Gaussian noise. Each update conditions on new measurements, adjusting means and variances just like the bivariate conditional formula.
  - **Gaussian processes**: For regression, predicting function values at unseen points uses the conditional distribution derived from observed data points, assuming a multivariate Gaussian prior over the function.

  For example, suppose $x_1$ is height and $x_2$ is weight, with $\rho = 0.8$. Observing a weight $a = 70$ kg (above average $\mu_2 = 65$) shifts the expected height to $\mu_1 + 0.8 \cdot (\sigma_1 / \sigma_2) \cdot 5$, with reduced variance compared to the unconditional height distribution.

## Quick sampling recipe

```python
import numpy as np

mu = np.array([mu1, mu2])
Sigma = np.array([[sigma1**2, rho * sigma1 * sigma2],
                  [rho * sigma1 * sigma2, sigma2**2]])

samples = np.random.multivariate_normal(mu, Sigma, size=5000)
x1, x2 = samples[:, 0], samples[:, 1]
```

In two dimensions it's easy to visualize `samples` with a scatter plot and verify that empirical covariances match the target $\Sigma$.

## Checklist for Gauss 201

1. Work directly with the $2 \times 2$ covariance—no need for higher-dimensional code paths.
2. Plot ellipses or scatter plots to build intuition about $(x_1, x_2)$ correlations.
3. Use conditional formulas to predict $x_1$ from $x_2$ or to impute missing values.
4. Remember that every high-dimensional Gaussian is built from these 2D blocks; mastering them pays off when you scale to Gauss 301+.
