+++
title = "Gauss 101"
date = "2025-11-21T00:00:00Z"
type = "post"
draft = false
math = true
tags = ["statistics", "probability", "gaussian", "univariate"]
categories = ["posts"]
description = "Crash course on the univariate Gaussian distribution: density, moments, z-scores, and modeling tips."
+++

The univariate Gaussian (or normal) distribution, written as $\mathcal{N}(\mu, \sigma^2)$, is the workhorse for modeling single random variables dominated by additive noise. Keeping the scope to one dimension highlights the mechanics you need most often in experiments, analytics notebooks, and quick sanity checks.

## A note on Gauss

Carl Friedrich Gauss (1777–1855) was a German mathematician and physicist whose contributions span number theory, astronomy, geodesy, and statistics. Although the normal distribution appeared in earlier work by de Moivre and Laplace, Gauss popularized it through his method of least squares for orbit determination and measurement error analysis. His insight that measurement errors cluster symmetrically around the true value laid the foundation for modern regression and data fitting. The distribution bears his name in recognition of these practical applications, which remain central to scientific computing today.

## Probability density and log-density

The probability density function (PDF) is

$$
\mathcal{N}(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right),
$$

where $\mu$ is the mean (location) and $\sigma^2$ the variance (scale). In code or numerical work, evaluate the log-density to avoid underflow:

$$
\log \mathcal{N}(x \mid \mu, \sigma^2) = -\tfrac{1}{2} \Big[\log(2\pi\sigma^2) + \frac{(x-\mu)^2}{\sigma^2}\Big].
$$

## Key moments and identities

- Expectation: $\mathbb{E}[X] = \mu$.
- Variance: $\operatorname{Var}[X] = \sigma^2$.
- Moment generating function: $M_X(t) = \exp\big(\mu t + \tfrac{1}{2}\sigma^2 t^2\big)$.
- Linear transforms remain Gaussian: if $Y = aX + b$, then $Y \sim \mathcal{N}(a\mu + b, a^2\sigma^2)$.

These facts let you propagate uncertainty through measurement equations without revisiting calculus every time.

## What are moments?

Moments are summary statistics that capture the shape of a probability distribution. The $n$-th **raw moment** of a random variable $X$ is defined as

$$
\mathbb{E}[X^n] = \int_{-\infty}^{\infty} x^n \, p(x) \, dx,
$$

where $p(x)$ is the probability density function. The $n$-th **central moment** measures deviations from the mean:

$$
\mathbb{E}[(X - \mu)^n] = \int_{-\infty}^{\infty} (x - \mu)^n \, p(x) \, dx.
$$

### The first four moments

Each moment has a specific interpretation:

1. **First moment (mean)**: $\mathbb{E}[X] = \mu$ locates the center of mass of the distribution. For a Gaussian, this is the peak and the balance point.

2. **Second central moment (variance)**: $\operatorname{Var}[X] = \mathbb{E}[(X - \mu)^2] = \sigma^2$ measures spread or dispersion. Larger variance means observations scatter further from the mean.

3. **Third central moment (skewness)**: $\mathbb{E}[(X - \mu)^3]$ captures asymmetry. For Gaussians, this is zero because the distribution is symmetric. Non-zero skewness indicates a tail extending in one direction.

4. **Fourth central moment (kurtosis)**: $\mathbb{E}[(X - \mu)^4]$ measures tail heaviness. For a Gaussian, the excess kurtosis (kurtosis minus 3) is zero. Heavy-tailed distributions have positive excess kurtosis; light-tailed distributions have negative excess kurtosis.

### Computing Gaussian moments

For $X \sim \mathcal{N}(\mu, \sigma^2)$, the moments have closed forms. The mean and variance are parameters of the distribution. Higher moments follow patterns:

- All odd central moments are zero due to symmetry: $\mathbb{E}[(X - \mu)^{2k+1}] = 0$.
- Even central moments grow factorially: $\mathbb{E}[(X - \mu)^{2k}] = (2k-1)!! \, \sigma^{2k}$, where $(2k-1)!! = 1 \cdot 3 \cdot 5 \cdots (2k-1)$ is the double factorial.

For example:
- $\mathbb{E}[(X - \mu)^2] = \sigma^2$
- $\mathbb{E}[(X - \mu)^4] = 3\sigma^4$
- $\mathbb{E}[(X - \mu)^6] = 15\sigma^6$

### Practical implications

Moments are not just theoretical curiosities—they guide modeling decisions and diagnostics:

**1. Moment matching for parameter estimation**: If you have sample moments $\bar{x}$ and $s^2$, you can estimate Gaussian parameters by matching theoretical and empirical moments:
$$
\hat{\mu} = \bar{x}, \quad\quad \hat{\sigma}^2 = s^2.
$$
This method of moments is quick and intuitive, though maximum likelihood is more efficient with larger samples.

**2. Model validation via higher moments**: If your data's third and fourth sample moments deviate significantly from zero and $3\sigma^4$, respectively, the Gaussian assumption may be violated. Skewness suggests transformations (log, Box-Cox); excess kurtosis suggests heavy-tailed alternatives (Student-$t$).

**3. Error propagation in measurement**: When combining independent measurements, moments tell you how uncertainty compounds. If $Z = X + Y$ with $X$ and $Y$ independent, then
$$
\mathbb{E}[Z] = \mathbb{E}[X] + \mathbb{E}[Y], \quad\quad \operatorname{Var}[Z] = \operatorname{Var}[X] + \operatorname{Var}[Y].
$$
This variance additivity is central to experimental design and metrology.

**4. Risk and portfolio analysis**: In finance, the first moment represents expected return, and the second moment (variance) represents risk. Higher moments capture tail risk (kurtosis) and asymmetry (skewness) in asset returns, informing hedging and diversification strategies.

**5. Machine learning and feature engineering**: Moments of feature distributions inform normalization and preprocessing. Standardizing features (subtracting mean, dividing by standard deviation) ensures all inputs have comparable scales, stabilizing gradient-based optimization.

### Sample moments in code

Here's how to compute sample moments in Python:

```python
import numpy as np
from scipy import stats

# Generate sample data
np.random.seed(42)
data = np.random.normal(loc=5.0, scale=2.0, size=1000)

# First four moments
mean = np.mean(data)
variance = np.var(data, ddof=1)  # Use ddof=1 for unbiased estimate
skewness = stats.skew(data)
kurtosis = stats.kurtosis(data)  # Excess kurtosis by default

print(f"Mean: {mean:.4f}")
print(f"Variance: {variance:.4f}")
print(f"Skewness: {skewness:.4f}")
print(f"Excess Kurtosis: {kurtosis:.4f}")

# For Gaussian data, skewness and excess kurtosis should be near zero
```

For large samples from a Gaussian, skewness and excess kurtosis should hover near zero. Persistent deviations signal non-Gaussian features that may require model adjustments.

## Moment generating function in depth

The moment generating function (MGF) is a powerful tool for working with Gaussians, encoding all moments and enabling clean derivations of sampling distributions. For $X \sim \mathcal{N}(\mu, \sigma^2)$, the MGF is defined as

$$
M_X(t) = \mathbb{E}[\exp(tX)] = \int_{-\infty}^{\infty} \exp(tx) \cdot \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right) dx.
$$

### Deriving the MGF

To evaluate this integral, complete the square in the exponent. Start by combining the terms:

$$
tx - \frac{(x-\mu)^2}{2\sigma^2} = -\frac{1}{2\sigma^2}\Big[x^2 - 2\mu x - 2\sigma^2 tx\Big] = -\frac{1}{2\sigma^2}\Big[x^2 - 2x(\mu + \sigma^2 t)\Big].
$$

Complete the square by adding and subtracting $(\mu + \sigma^2 t)^2$:

$$
-\frac{1}{2\sigma^2}\Big[\big(x - (\mu + \sigma^2 t)\big)^2 - (\mu + \sigma^2 t)^2\Big] = -\frac{\big(x - (\mu + \sigma^2 t)\big)^2}{2\sigma^2} + \frac{(\mu + \sigma^2 t)^2}{2\sigma^2}.
$$

Expand the squared term:

$$
\frac{(\mu + \sigma^2 t)^2}{2\sigma^2} = \frac{\mu^2 + 2\mu\sigma^2 t + \sigma^4 t^2}{2\sigma^2} = \frac{\mu^2}{2\sigma^2} + \mu t + \tfrac{1}{2}\sigma^2 t^2.
$$

The integral becomes

$$
M_X(t) = \exp\big(\mu t + \tfrac{1}{2}\sigma^2 t^2\big) \int_{-\infty}^{\infty} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{\big(x - (\mu + \sigma^2 t)\big)^2}{2\sigma^2}\right) dx.
$$

The integral is the PDF of $\mathcal{N}(\mu + \sigma^2 t, \sigma^2)$ integrated over all $x$, which equals 1. Therefore,

$$
M_X(t) = \exp\big(\mu t + \tfrac{1}{2}\sigma^2 t^2\big).
$$

### Extracting moments

The MGF generates moments via differentiation:

$$
\mathbb{E}[X^n] = \frac{d^n M_X(t)}{dt^n}\Bigg|_{t=0}.
$$

For the Gaussian MGF, the first two derivatives are:

$$
M_X'(t) = (\mu + \sigma^2 t) \exp\big(\mu t + \tfrac{1}{2}\sigma^2 t^2\big),
$$

$$
M_X''(t) = \big[\sigma^2 + (\mu + \sigma^2 t)^2\big] \exp\big(\mu t + \tfrac{1}{2}\sigma^2 t^2\big).
$$

Evaluating at $t = 0$:

$$
\mathbb{E}[X] = M_X'(0) = \mu, \quad\quad \mathbb{E}[X^2] = M_X''(0) = \sigma^2 + \mu^2.
$$

The variance follows from $\operatorname{Var}[X] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 = \sigma^2 + \mu^2 - \mu^2 = \sigma^2$.

### Why the MGF matters

The MGF is uniquely powerful for several reasons:

1. **Uniqueness**: If two distributions have the same MGF (where it exists), they are identical. This makes MGFs ideal for proving distributional results.

2. **Sums of independent variables**: If $X$ and $Y$ are independent, then $M_{X+Y}(t) = M_X(t) \cdot M_Y(t)$. For Gaussians, this immediately shows that sums of independent Gaussians are Gaussian:
   $$
   M_{X+Y}(t) = \exp\big(\mu_X t + \tfrac{1}{2}\sigma_X^2 t^2\big) \cdot \exp\big(\mu_Y t + \tfrac{1}{2}\sigma_Y^2 t^2\big) = \exp\big((\mu_X + \mu_Y)t + \tfrac{1}{2}(\sigma_X^2 + \sigma_Y^2)t^2\big).
   $$

3. **Linear transformations**: For $Y = aX + b$, the MGF is $M_Y(t) = \exp(bt) M_X(at)$, which directly yields the transformed parameters.

4. **Sampling distributions**: The MGF derivation of the sample mean distribution (covered in Gauss 103) exploits the multiplicative property to show $\bar{X} \sim \mathcal{N}(\mu, \sigma^2/n)$ without messy convolution integrals.

In practice, the MGF reduces complex probability calculations to algebraic manipulations of exponential functions, making it indispensable for rigorous statistical work.

## Standardization and z-scores

Converting observations to z-scores removes the units and centers the distribution:

$$
Z = \frac{X - \mu}{\sigma} \sim \mathcal{N}(0, 1).
$$

Lookup tables (or `scipy.stats.norm.cdf`) for the standard normal cumulative distribution function (CDF) translate z-scores into tail probabilities, which underpins hypothesis tests and confidence intervals.

## Estimating parameters from data

Given samples $x_1,\ldots,x_n$ from a univariate Gaussian, the maximum likelihood estimates are the familiar sample statistics:

$$
\hat{\mu} = \frac{1}{n} \sum_{i=1}^{n} x_i, \quad\quad \hat{\sigma}^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \hat{\mu})^2.
$$

Use $\frac{1}{n-1}$ in the variance formula when you need the unbiased estimator (e.g., reporting sample variance). For streaming data, maintain running sums to update these quantities without storing every observation.

## Modeling checklist

1. **Check symmetry**: Histograms and quantile-quantile plots should hug the diagonal if the Gaussian assumption is reasonable.
2. **Test variance stability**: Univariate models break down when variance depends on the mean; stabilize via transformations (log, square root) first.
3. **Inspect tails**: Heavy tails inflate false positives; consider the Student-$t$ as a fallback if outliers dominate.
4. **Document units**: Because $\sigma$ carries the same units as the data, report both $\mu$ and $\sigma$ with measurement context.

Keep these diagnostics in the stats playbook so univariate Gaussian models stay interpretable and reliable.
