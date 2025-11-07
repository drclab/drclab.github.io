+++
title = "Gauss 203: Laplace Approximation"
date = "2025-12-12T00:00:00Z"
type = "post"
draft = true
math = true
tags = ["statistics", "probability", "gaussian", "laplace-approximation", "bayesian"]
categories = ["posts"]
description = "Using Gaussian approximations to summarize posterior distributions: mechanics, precision matrices, and when the method breaks down."
+++

The Laplace approximation turns a peaked, differentiable posterior into a multivariate Gaussian by matching the mode and curvature. It provides a fast analytic summary when MCMC is too slow and variational inference is overkill, making it a go-to tool for prototyping Bayesian models and sanity-checking inference pipelines.

## A note on Laplace

Pierre-Simon Laplace (1749–1827) was a French mathematician and astronomer who pioneered probability theory and celestial mechanics. His work on the central limit theorem established why sums of random variables converge to the normal distribution. The Laplace approximation—using a Gaussian to approximate a distribution near its mode—exploits this convergence by treating the log-posterior as a quadratic around its peak. This method became essential in Bayesian inference before modern sampling algorithms, and it remains valuable for rapid exploratory analysis.

## The core idea

Given a posterior density $p(\theta \mid y)$ over parameters $\theta \in \mathbb{R}^d$, the Laplace approximation replaces it with a Gaussian centered at the mode:

$$
p(\theta \mid y) \approx \mathcal{N}(\theta \mid \hat{\theta}, \Sigma),
$$

where $\hat{\theta} = \arg\max_\theta \log p(\theta \mid y)$ is the maximum a posteriori (MAP) estimate, and $\Sigma$ is the inverse of the negative Hessian (the observed Fisher information):

$$
\Sigma = \Big[-\nabla^2 \log p(\theta \mid y)\Big|_{\theta=\hat{\theta}}\Big]^{-1}.
$$

This approximation is exact when the log-posterior is quadratic (i.e., when the prior and likelihood are both Gaussian), and it degrades gracefully when the posterior is approximately log-concave.

## Computing the approximation

The workflow has three steps:

1. **Find the mode**: Use Newton-Raphson, L-BFGS-B, or another optimizer to locate $\hat{\theta}$ by maximizing $\log p(\theta \mid y)$.
2. **Evaluate the Hessian**: Compute the second derivative matrix $H = \nabla^2 \log p(\theta \mid y)$ at $\hat{\theta}$, either analytically or via automatic differentiation.
3. **Invert for the covariance**: Set $\Sigma = -H^{-1}$. The diagonal entries give marginal variances; off-diagonal entries encode posterior correlations.

For numerical stability, work with the precision matrix $\Lambda = \Sigma^{-1} = -H$ directly when you can, and use Cholesky decomposition for sampling or evaluation.

## Example: Logistic regression

Consider Bayesian logistic regression with a Gaussian prior $\theta \sim \mathcal{N}(0, \tau^2 I)$ and binary observations $y_i \in \{0,1\}$ with likelihood

$$
p(y_i \mid x_i, \theta) = \sigma(x_i^\top \theta)^{y_i} \big[1 - \sigma(x_i^\top \theta)\big]^{1-y_i},
$$

where $\sigma(z) = 1/(1 + e^{-z})$ is the logistic function. The log-posterior is

$$
\log p(\theta \mid y, X) = \sum_{i=1}^{n} \Big[y_i \, x_i^\top\theta - \log(1 + e^{x_i^\top\theta})\Big] - \frac{1}{2\tau^2} \|\theta\|^2 + \text{const}.
$$

The gradient and Hessian are

$$
\nabla_\theta \log p = \sum_{i=1}^{n} x_i \big(y_i - \sigma(x_i^\top\theta)\big) - \frac{\theta}{\tau^2},
$$

$$
\nabla^2_\theta \log p = -\sum_{i=1}^{n} \sigma(x_i^\top\theta) \big[1 - \sigma(x_i^\top\theta)\big] x_i x_i^\top - \frac{I}{\tau^2}.
$$

Optimize to find $\hat{\theta}$, evaluate the Hessian, invert, and you have a Gaussian approximation for the posterior distribution of regression coefficients.

## When the approximation works

Laplace approximations excel when:

- **The posterior is unimodal and roughly symmetric** around the mode, with light tails.
- **Sample size is moderate to large**, so the likelihood dominates prior curvature and the central limit theorem kicks in.
- **You need quick uncertainty estimates** for model comparison, sensitivity analysis, or prototyping.
- **Gradient and Hessian evaluations are cheap**, making optimization faster than sampling.

In these settings, the Laplace approximation delivers posterior means, variances, and credible intervals with minimal computational overhead.

## When it breaks down

The method fails in several common scenarios:

1. **Multimodality**: If the posterior has multiple peaks, the Laplace approximation captures only one mode and underestimates uncertainty.
2. **Heavy tails**: Cauchy-like tails or mixture models produce non-Gaussian posteriors; the approximation understates extreme quantiles.
3. **Boundary constraints**: Parameters on $(0, \infty)$ or simplexes require reparameterization (e.g., log or logit transforms) before applying the approximation.
4. **Small sample sizes**: When the prior dominates, non-Gaussian prior shapes propagate into the posterior, and quadratic approximations are too crude.
5. **High correlations**: Strong posterior correlations amplify errors in the Hessian inversion, leading to overconfident intervals.

Always validate the approximation with posterior predictive checks or a few MCMC chains to catch gross mismatches.

## Implementation checklist

1. **Transform constrained parameters**: Apply log, logit, or other bijections to map constrained spaces to $\mathbb{R}^d$ before optimization.
2. **Use numerical Hessians cautiously**: Finite-difference Hessians are fragile; prefer automatic differentiation (JAX, PyTorch, Stan's `optimizing()` mode).
3. **Check the Hessian is negative definite**: If eigenvalues are positive, the "mode" is a saddle point or boundary solution—revisit the model or add regularization.
4. **Compare with MCMC on a subset**: Run a short MCMC chain to verify marginal variances and correlations roughly match the Laplace covariance.
5. **Report the approximation**: Document that intervals come from a Laplace approximation, not full posterior sampling, to manage user expectations.

## Code snippet in Python

```python
import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

def log_posterior(theta, X, y, tau):
    """Log-posterior for logistic regression with Gaussian prior."""
    logits = X @ theta
    log_lik = y * logits - np.log(1 + np.exp(logits))
    log_prior = -0.5 * np.sum(theta**2) / tau**2
    return np.sum(log_lik) + log_prior

def neg_log_posterior(theta, X, y, tau):
    return -log_posterior(theta, X, y, tau)

# Find MAP estimate
theta_init = np.zeros(X.shape[1])
result = minimize(neg_log_posterior, theta_init, args=(X, y, tau), 
                  method='L-BFGS-B', options={'disp': False})
theta_map = result.x

# Compute Hessian numerically (or use autograd)
from scipy.optimize import approx_fprime
eps = np.sqrt(np.finfo(float).eps)

def hessian_fd(theta):
    """Finite-difference Hessian."""
    d = len(theta)
    H = np.zeros((d, d))
    for i in range(d):
        def grad_i(th):
            return approx_fprime(th, lambda t: log_posterior(t, X, y, tau), eps)[i]
        H[i, :] = approx_fprime(theta, grad_i, eps)
    return H

H = hessian_fd(theta_map)
Sigma = -np.linalg.inv(H)  # Covariance matrix

# Sample from the Laplace approximation
laplace_dist = multivariate_normal(mean=theta_map, cov=Sigma)
samples = laplace_dist.rvs(size=1000)
```

## Extending to variational inference

The Laplace approximation is a fixed-point method: it finds one Gaussian and stops. Variational inference generalizes this by optimizing over a family of Gaussians (or other tractable distributions) to minimize KL divergence from the true posterior. When the variational family includes full-rank Gaussians, the solution often sits close to the Laplace approximation, but with adjustable flexibility for skewness or factorization constraints.

## Summary

The Laplace approximation converts optimization into inference by wrapping a Gaussian around the posterior mode. It is fast, interpretable, and sufficient for many exploratory analyses, but it requires validation against sampling methods when the posterior shape is unknown. Keep it in the Bayesian toolkit for prototyping and as a stepping stone to more sophisticated approximations.
