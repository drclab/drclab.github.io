+++
title = "TS 101: Exponential Smoothing and the LGT Model"
slug = "ts-101"
aliases = ["ts101"]
date = "2028-11-26T00:00:00Z"
type = "post"
draft = false
math = true
tags = ["time-series", "exponential-smoothing", "bayesian", "forecasting"]
categories = ["posts"]
description = "An introduction to exponential smoothing and the Local-Global Trend (LGT) model from Uber's Orbit package, bridging classical methods with modern Bayesian approaches."
+++

Welcome to TS 101: Exponential Smoothing and the Local-Global Trend Model! This guide introduces the foundations of exponential smoothing and explains how Uber's Orbit package refines these classical methods using Bayesian inference. We'll focus on the Local Global Trend (LGT) model, which elegantly handles both linear and nonlinear trends.

This post is based on the paper [*Orbit: Probabilistic Forecast with Exponential Smoothing*](https://arxiv.org/abs/2004.08492) by Edwin Ng, Zhishi Wang, Huigang Chen, Steve Yang, and Slawek Smyl (2020).

## Learning Goals

By the end of this post, you will:
- understand the evolution from classical exponential smoothing to Bayesian variants,
- see how the LGT model combines local and global trends,
- learn the mathematical formulation of the LGT model,
- appreciate why Bayesian approaches provide richer uncertainty quantification.

## Key Vocabulary

- **Exponential smoothing**: A family of forecasting methods that compute weighted averages of past observations, with weights decaying exponentially for older data.
- **Holt's linear trend method**: An extension of simple exponential smoothing that adds a trend component, allowing forecasts to capture linear growth or decline.
- **Local trend**: A trend component that evolves adaptively based on recent observations, similar to traditional exponential smoothing.
- **Global trend**: A longer-term, smoother trend component that captures the overall growth pattern across the entire time series.
- **Hybrid trend**: The combination of local and global trends in the LGT model, enabling it to handle complex trend dynamics.
- **Multiplicative vs. additive errors**: In multiplicative models, errors scale with the level of the series; in additive models, errors remain constant. Orbit uses log-transformation to handle multiplicative forms.
- **Heteroscedasticity**: When the variance of errors changes over time. LGT can model this by allowing error variance to grow with the expected value.
- **Student-t distribution**: A probability distribution with heavier tails than the normal distribution, making it more robust to outliers.
- **Probabilistic programming**: A paradigm for building statistical models using specialized languages (like Stan or Pyro) that automate inference.

## Why Exponential Smoothing Still Matters

Despite the rise of machine learning methods, exponential smoothing remains a workhorse for time series forecasting, particularly for:
- **Low granularity data**: When you have limited observations or short time series.
- **Interpretability**: The model components (level, trend, seasonality) have clear business meanings.
- **Computational efficiency**: Exponential smoothing models are fast to fit and forecast.
- **Proven track record**: They consistently perform well in forecasting competitions like the M-competitions.

However, classical exponential smoothing has limitations:
- Fixed error distributions (usually Gaussian)
- Difficulty modeling heteroscedasticity
- Point estimates only (no uncertainty quantification)
- Limited flexibility in trend specification

The Orbit package addresses these limitations using a Bayesian framework.

## From Holt's Method to LGT

### Classical Holt's Linear Trend Method

Holt's method forecasts future values using a level $\ell_t$ and a trend $b_t$:

$$
\begin{aligned}
\ell_t &= \alpha y_t + (1 - \alpha)(\ell_{t-1} + b_{t-1}), \\
b_t &= \beta^* (\ell_t - \ell_{t-1}) + (1 - \beta^*) b_{t-1}, \\
\hat{y}_{t+h} &= \ell_t + h b_t,
\end{aligned}
$$

where:
- $y_t$ is the observed value at time $t$,
- $\alpha$ and $\beta^*$ are smoothing parameters (between 0 and 1),
- $h$ is the forecast horizon.

The level equation updates based on the current observation and the previous estimate. The trend equation smooths the first difference of the level. This method assumes:
- Linear trend
- Additive errors
- Homoscedastic (constant variance) errors
- Gaussian noise

### The Local-Global Trend (LGT) Extension

The LGT model refines Holt's method by:

1. **Separating local and global trends**: Instead of a single trend $b_t$, LGT uses:
   - A **local trend** $b_t$ that evolves quickly based on recent data
   - A **global trend** $\tau$ that captures the overall growth direction

2. **Log-transformation for multiplicative form**: By applying $\log$ to the response, the model handles multiplicative trends while maintaining additive structure in the transformed space.

3. **Flexible error distributions**: Student-t errors instead of Gaussian, providing robustness to outliers.

4. **Heteroscedastic errors**: Error variance can grow with the expected value.

5. **Bayesian inference**: Full posterior distributions for all parameters, not just point estimates.

## The LGT Model Formulation

### State Space Form

The LGT model can be written in state space form. Let $\mu_t$ be the latent level at time $t$. The model equations are:

**Observation equation**:
$$
y_t \sim \text{Student-t}(\nu, \mu_t, \sigma_t),
$$

where $\nu$ is the degrees of freedom, $\mu_t$ is the location, and $\sigma_t$ is the scale (which may vary over time for heteroscedastic errors).

**State evolution equations**:
$$
\begin{aligned}
\mu_t &= \ell_{t-1} + b_{t-1} + \tau, \\
\ell_t &= \alpha y_t + (1 - \alpha) \mu_t, \\
b_t &= \beta (b_{t-1} + \tau) + (1 - \beta) (\ell_t - \ell_{t-1}),
\end{aligned}
$$

where:
- $\ell_t$ is the local level,
- $b_t$ is the local trend,
- $\tau$ is the global trend (a parameter to be estimated),
- $\alpha, \beta \in (0, 1)$ are smoothing coefficients.

### Key Components Explained

1. **$\mu_t = \ell_{t-1} + b_{t-1} + \tau$**: The expected value combines the previous level, the local trend evolution, and the global trend shift. This hybrid structure lets the model adapt locally while respecting a long-term direction.

2. **$\ell_t = \alpha y_t + (1 - \alpha) \mu_t$**: The level is a weighted average of the observed value and the predicted level. When $\alpha$ is close to 1, the model reacts quickly to new data; when $\alpha$ is near 0, it smooths heavily.

3. **$b_t = \beta (b_{t-1} + \tau) + (1 - \beta) (\ell_t - \ell_{t-1})$**: The local trend updates based on both the previous trend (adjusted by the global trend) and the recent change in level.

4. **Student-t errors**: By using a Student-t distribution with degrees of freedom $\nu$, the model is less sensitive to outliers than a Gaussian model. When $\nu$ is small (e.g., 3-5), the distribution has heavy tails; as $\nu \to \infty$, it approaches a normal distribution.

### Handling Multiplicative Trends via Log-Transformation

For series that grow multiplicatively (e.g., revenue growing at 20% per year), Orbit applies a log-transformation:
$$
\tilde{y}_t = \log(y_t).
$$

The model is then fit in the log-space, and forecasts are back-transformed:
$$
\hat{y}_{t+h} = \exp(\hat{\tilde{y}}_{t+h}).
$$

This approach requires $y_t > 0$ for all observations. The log-transformation stabilizes variance (addressing heteroscedasticity) and linearizes exponential growth.

### Heteroscedastic Errors

Classical exponential smoothing assumes constant error variance. LGT can model time-varying variance:
$$
\sigma_t = \sigma_0 \cdot f(\mu_t),
$$

where $f(\cdot)$ is a monotonically increasing function. A common choice is:
$$
\sigma_t = \sigma_0 \cdot |\mu_t|^\gamma,
$$

with $\gamma \geq 0$. When $\gamma = 0$, errors are homoscedastic; when $\gamma > 0$, larger expected values have larger variance.

## Bayesian Inference and Probabilistic Programming

Classical exponential smoothing fits parameters (like $\alpha$, $\beta$) by minimizing a loss function (e.g., mean squared error). Orbit instead uses Bayesian inference:

1. **Specify priors**: Assign prior distributions to all parameters. For example:
   $$
   \begin{aligned}
   \alpha &\sim \text{Beta}(a_\alpha, b_\alpha), \\
   \beta &\sim \text{Beta}(a_\beta, b_\beta), \\
   \tau &\sim \text{Normal}(0, \sigma_\tau^2), \\
   \sigma &\sim \text{HalfNormal}(\sigma_0), \\
   \nu &\sim \text{Gamma}(a_\nu, b_\nu).
   \end{aligned}
   $$

2. **Compute the posterior**: Given data $y_{1:T}$, we want:
   $$
   p(\alpha, \beta, \tau, \sigma, \nu \mid y_{1:T}) \propto p(y_{1:T} \mid \alpha, \beta, \tau, \sigma, \nu) \cdot p(\alpha, \beta, \tau, \sigma, \nu).
   $$

3. **Use MCMC or MAP**: Orbit supports:
   - **MCMC (via Stan)**: Full Bayesian inference with uncertainty quantification.
   - **MAP (maximum a posteriori)**: Point estimates that are faster to compute.
   - **Variational inference (via Pyro)**: Approximate posterior for faster inference.

### Why Bayesian?

- **Uncertainty quantification**: Instead of a single forecast, you get a predictive distribution.
- **Regularization**: Priors prevent overfitting, especially with limited data.
- **Flexibility**: Easy to add constraints (e.g., $\tau > 0$ for strictly positive growth).
- **Posterior inference**: Understand parameter relationships and check convergence diagnostics.

## Comparison with DLT (Damped Local Trend)

Orbit also includes a Damped Local Trend (DLT) model. Key differences:

| Feature | LGT | DLT |
|---------|-----|-----|
| Trend structure | Local + Global (hybrid) | Damped local trend |
| Global trend | Smooth, additive $\tau$ | Damping factor $\phi \in (0,1)$ |
| Transformation | Often uses log | Flexible |
| Exogenous variables | Limited | Flexible regression component |

**When to use LGT**: Series with strong, potentially non-linear trends (e.g., fast-growing startups, viral products).  
**When to use DLT**: Series with trends that dampen over time (e.g., product adoption curves), or when you need to include external predictors.

## Practical Considerations

### Data Requirements

- **Positive values**: If using log-transformation, all $y_t > 0$.
- **Sufficient history**: At least 10-20 observations to estimate smoothing parameters reliably.
- **Stationarity**: Exponential smoothing handles non-stationary series, but extreme regime shifts may require separate models.

### Hyperparameter Tuning

- **Smoothing parameters** ($\alpha$, $\beta$): Orbit estimates these via Bayesian inference, but you can set informative priors if you have domain knowledge.
- **Degrees of freedom** ($\nu$): Lower values (3-5) are robust to outliers; higher values (10+) approximate normality.
- **Prior scales**: Tight priors regularize more; diffuse priors let data dominate.

### Model Validation

- **Holdout forecasting**: Reserve the last $h$ observations and evaluate forecast accuracy.
- **Posterior predictive checks**: Simulate data from the fitted model and compare to actual data.
- **Trace plots and $\hat{R}$**: For MCMC, check that chains have converged.

## Example: Interpreting an LGT Forecast

Suppose you fit LGT to monthly revenue data and obtain:
- $\hat{\alpha} = 0.7$: The model adapts fairly quickly to new observations.
- $\hat{\beta} = 0.3$: The local trend is somewhat smoothed.
- $\hat{\tau} = 120$: Revenue has a global upward trend of \$120/month.
- $\hat{\nu} = 4$: Errors have heavy tails (robust to occasional spikes or dips).

A 12-month-ahead forecast would show:
- A **median trajectory** combining recent local dynamics and the long-term $\tau$.
- **Credible intervals** (e.g., 50% and 95%) that widen as the horizon increases, reflecting uncertainty in both parameters and future shocks.

## What's Next

You now have a vocabulary-first understanding of exponential smoothing and the LGT model. In a follow-up post, we can:
- Implement LGT in Python using the Orbit package,
- Fit the model to real data and interpret posterior samples,
- Compare LGT with DLT and classical Holt's method,
- Explore seasonal extensions (Seasonal LGT).

## References

- Ng, E., Wang, Z., Chen, H., Yang, S., & Smyl, S. (2020). *Orbit: Probabilistic Forecast with Exponential Smoothing*. [arXiv:2004.08492](https://arxiv.org/abs/2004.08492). [PDF](../../../pdf/$_Orbit_exponential_smoothing.pdf)
- [Orbit Documentation](https://uber.github.io/orbit/)
- [Orbit GitHub Repository](https://github.com/uber/orbit)

---

*This post is part of the Time Series 101 series. If you found this helpful, check out our MCMC 101 series for related Bayesian modeling topics.*
