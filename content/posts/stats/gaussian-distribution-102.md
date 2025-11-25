+++
title = "Gauss 102: Distribution of the Sample Mean"
date = "2025-11-25T00:00:00Z"
type = "post"
draft = false
math = true
tags = ["statistics", "gaussian"]
categories = ["posts"]
description = "Deriving the sampling distribution of the mean from independent Gaussian observations: log-probability mechanics, variance reduction, and practical implications."
+++

When you average independent Gaussian observations, the sample mean itself follows a Gaussian distribution with reduced variance. Understanding the mechanics behind this result—and being able to write down its log-probability—is essential for uncertainty quantification, confidence intervals, and Bayesian inference. This post walks through the derivation step by step, showing how the math connects to code and experimental design.

## Setup and notation

Suppose you observe $n$ independent samples $x_1, x_2, \ldots, x_n$ from a univariate Gaussian distribution:

$$
x_i \sim \mathcal{N}(\mu, \sigma^2) \quad \text{for } i = 1, 2, \ldots, n.
$$

The sample mean is defined as

$$
\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i.
$$

Our goal is to derive the distribution of $\bar{x}$, then express its log-probability in a form that's numerically stable and easy to compute.

## Deriving the distribution via moment generating functions

The cleanest derivation uses the moment generating function (MGF). Recall from Gauss 101 that the MGF of a Gaussian random variable $X \sim \mathcal{N}(\mu, \sigma^2)$ is

$$
M_X(t) = \mathbb{E}[\exp(tX)] = \exp\big(\mu t + \tfrac{1}{2}\sigma^2 t^2\big).
$$

For the sample mean, we can write

$$
\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i.
$$

The MGF of $\bar{x}$ is

$$
M_{\bar{x}}(t) = \mathbb{E}[\exp(t\bar{x})] = \mathbb{E}\left[\exp\left(\frac{t}{n} \sum_{i=1}^{n} x_i\right)\right].
$$

Since the $x_i$ are independent, the expectation of the product is the product of expectations:

$$
M_{\bar{x}}(t) = \prod_{i=1}^{n} \mathbb{E}\left[\exp\left(\frac{t}{n} x_i\right)\right] = \prod_{i=1}^{n} M_{x_i}\left(\frac{t}{n}\right).
$$

Each $x_i$ has the same MGF, so

$$
M_{\bar{x}}(t) = \left[M_X\left(\frac{t}{n}\right)\right]^n = \left[\exp\left(\mu \cdot \frac{t}{n} + \tfrac{1}{2}\sigma^2 \cdot \frac{t^2}{n^2}\right)\right]^n.
$$

Simplifying the exponent:

$$
M_{\bar{x}}(t) = \exp\left(n \cdot \left[\mu \cdot \frac{t}{n} + \tfrac{1}{2}\sigma^2 \cdot \frac{t^2}{n^2}\right]\right) = \exp\left(\mu t + \tfrac{1}{2} \cdot \frac{\sigma^2}{n} \cdot t^2\right).
$$

This is the MGF of a Gaussian distribution with mean $\mu$ and variance $\sigma^2/n$. Therefore,

$$
\bar{x} \sim \mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right).
$$

**Key insight**: The sample mean has the same expected value as individual observations, but its variance decreases by a factor of $n$. This is why averaging reduces noise and improves estimation precision.

## Log-probability of the sample mean

The probability density function of the sample mean is

$$
p(\bar{x} \mid \mu, \sigma^2, n) = \mathcal{N}\left(\bar{x} \mid \mu, \frac{\sigma^2}{n}\right) = \frac{1}{\sqrt{2\pi \sigma^2/n}} \exp\left(-\frac{(\bar{x} - \mu)^2}{2\sigma^2/n}\right).
$$

Taking the logarithm:

$$
\log p(\bar{x} \mid \mu, \sigma^2, n) = -\tfrac{1}{2} \left[\log\left(2\pi \frac{\sigma^2}{n}\right) + \frac{(\bar{x} - \mu)^2}{\sigma^2/n}\right].
$$

Simplifying the log term:

$$
\log\left(2\pi \frac{\sigma^2}{n}\right) = \log(2\pi) + \log(\sigma^2) - \log(n).
$$

And simplifying the quadratic term:

$$
\frac{(\bar{x} - \mu)^2}{\sigma^2/n} = \frac{n (\bar{x} - \mu)^2}{\sigma^2}.
$$

Putting it together:

$$
\log p(\bar{x} \mid \mu, \sigma^2, n) = -\tfrac{1}{2} \Big[\log(2\pi) + \log(\sigma^2) - \log(n) + \frac{n(\bar{x} - \mu)^2}{\sigma^2}\Big].
$$

This form makes it clear how the sample size $n$ affects the log-probability:
- The $-\log(n)$ term increases the log-probability (reduces the normalizing constant).
- The $n(\bar{x} - \mu)^2$ term in the numerator sharpens the peak around $\mu$, reflecting increased precision.

## Alternative derivation via sufficient statistics

There's a more direct route that leverages the fact that the sum of independent Gaussians is Gaussian. If $x_i \sim \mathcal{N}(\mu, \sigma^2)$, then

$$
S = \sum_{i=1}^{n} x_i \sim \mathcal{N}(n\mu, n\sigma^2),
$$

using the linearity of expectation and variance additivity for independent variables. The sample mean is $\bar{x} = S/n$, a linear transformation of $S$:

$$
\bar{x} = \frac{1}{n} S \sim \mathcal{N}\left(\frac{1}{n} \cdot n\mu, \frac{1}{n^2} \cdot n\sigma^2\right) = \mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right).
$$

This approach emphasizes that the sample mean is a sufficient statistic for $\mu$: all information about $\mu$ in the data is captured by $\bar{x}$.

## Practical implications

### Standard error and confidence intervals

The standard deviation of the sample mean, called the **standard error** (SE), is

$$
\text{SE}(\bar{x}) = \frac{\sigma}{\sqrt{n}}.
$$

In practice, you replace $\sigma$ with the sample standard deviation $s$ to get the estimated standard error. A 95% confidence interval for $\mu$ is then

$$
\bar{x} \pm 1.96 \cdot \frac{s}{\sqrt{n}},
$$

which comes directly from the Gaussian distribution of $\bar{x}$ and the z-score for the 97.5th percentile.

### Precision scaling with sample size

The variance of $\bar{x}$ decreases as $1/n$, so the standard error decreases as $1/\sqrt{n}$. To halve the width of your confidence interval, you need **four times as many samples**. This square-root scaling is fundamental to experimental design and power analysis.

### Log-probability in likelihood functions

When estimating $\mu$ and $\sigma^2$ from data, the log-likelihood of the observations is

$$
\log p(x_1, \ldots, x_n \mid \mu, \sigma^2) = \sum_{i=1}^{n} \log p(x_i \mid \mu, \sigma^2).
$$

But if you condition on $\bar{x}$ and the sample variance, you can factor the likelihood into independent pieces. The log-probability of $\bar{x}$ derived above becomes a component of this factorization, useful in hierarchical models and meta-analysis.

## Numerical implementation

Here's how to compute the log-probability of a sample mean in Python:

```python
import numpy as np
from scipy.stats import norm

def log_prob_sample_mean(x_bar, mu, sigma, n):
    """
    Log-probability of the sample mean.
    
    Parameters:
    -----------
    x_bar : float
        Observed sample mean
    mu : float
        Population mean
    sigma : float
        Population standard deviation
    n : int
        Sample size
    
    Returns:
    --------
    log_prob : float
        Log-probability of observing x_bar
    """
    # Standard error of the mean
    se = sigma / np.sqrt(n)
    
    # Use scipy's norm.logpdf for numerical stability
    return norm.logpdf(x_bar, loc=mu, scale=se)

# Example: observed mean from 25 samples
x_bar_obs = 10.5
mu_true = 10.0
sigma_true = 2.0
n_samples = 25

log_p = log_prob_sample_mean(x_bar_obs, mu_true, sigma_true, n_samples)
print(f"Log-probability: {log_p:.4f}")
print(f"Probability: {np.exp(log_p):.6f}")

# Compare with direct calculation
se = sigma_true / np.sqrt(n_samples)
log_p_manual = -0.5 * (np.log(2 * np.pi) + np.log(se**2) + 
                       ((x_bar_obs - mu_true)**2 / se**2))
print(f"Manual log-probability: {log_p_manual:.4f}")
```

For numerical stability, always use `norm.logpdf` or equivalent library functions rather than taking `log(norm.pdf(...))`, which can underflow for extreme values.

## Connection to hypothesis testing

The t-test for a single sample tests the null hypothesis $H_0: \mu = \mu_0$ by computing the t-statistic

$$
t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}},
$$

which follows a Student-$t$ distribution with $n-1$ degrees of freedom when $\sigma$ is unknown. When $n$ is large or $\sigma$ is known, this converges to a z-test based on the Gaussian distribution of $\bar{x}$ we derived above. The log-probability framework makes it easy to construct likelihood ratios and Bayes factors for comparing hypotheses.

## Extension to Bayesian inference

In Bayesian statistics, if you observe $\bar{x}$ and know $\sigma$, you can update a prior on $\mu$ to get a posterior. If the prior is $\mu \sim \mathcal{N}(\mu_0, \tau^2)$, the posterior is also Gaussian (conjugacy):

$$
\mu \mid \bar{x} \sim \mathcal{N}\left(\mu_{\text{post}}, \sigma^2_{\text{post}}\right),
$$

where

$$
\frac{1}{\sigma^2_{\text{post}}} = \frac{1}{\tau^2} + \frac{n}{\sigma^2}, \quad\quad \mu_{\text{post}} = \sigma^2_{\text{post}} \left(\frac{\mu_0}{\tau^2} + \frac{n\bar{x}}{\sigma^2}\right).
$$

The precision (inverse variance) of the posterior is the sum of the prior precision and the data precision $n/\sigma^2$. The data precision increases linearly with $n$, which is why more samples sharpen the posterior around the observed mean.

## Computational checklist

1. **Always work in log-space**: Compute log-probabilities directly to avoid numerical underflow.
2. **Use standard library functions**: Functions like `scipy.stats.norm.logpdf` handle edge cases and numerical stability automatically.
3. **Propagate standard errors correctly**: When reporting results, include both $\bar{x}$ and $s/\sqrt{n}$ to convey precision.
4. **Check sample size assumptions**: For small $n$, use the $t$-distribution instead of the Gaussian approximation.
5. **Visualize the sampling distribution**: Plot the density of $\bar{x}$ to sanity-check that the derived variance $\sigma^2/n$ makes sense given your data.

## Summary

The sample mean of $n$ independent Gaussian observations follows a Gaussian distribution with the same mean and variance reduced by a factor of $n$. The log-probability formula

$$
\log p(\bar{x} \mid \mu, \sigma^2, n) = -\tfrac{1}{2} \Big[\log(2\pi\sigma^2) - \log(n) + \frac{n(\bar{x} - \mu)^2}{\sigma^2}\Big]
$$

encodes this variance reduction and provides a foundation for inference, hypothesis testing, and uncertainty quantification. Understanding this derivation clarifies why averaging is so powerful and how precision scales with sample size in experimental design.
