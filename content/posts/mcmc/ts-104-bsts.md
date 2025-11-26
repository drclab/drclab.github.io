+++
title = "TS 104: Bayesian Structural Time Series (BSTS)"
slug = "ts-104"
aliases = ["ts104"]
date = "2028-11-29T00:00:00Z"
type = "post"
draft = false
math = true
tags = ["time-series", "bsts", "bayesian", "forecasting"]
categories = ["posts"]
description = "An introduction to Bayesian Structural Time Series (BSTS), a flexible modular framework for forecasting."
+++

Welcome to TS 104! We've covered [ARIMA](../ts-102) and [Exponential Smoothing](../ts-103). Now we explore **Bayesian Structural Time Series (BSTS)**, a powerful framework that combines the state-space representation with Bayesian inference to provide transparent and flexible forecasts.

This post covers material from Section 2.2.3 of the paper [*Orbit: Probabilistic Forecast with Exponential Smoothing*](https://arxiv.org/abs/2004.08492), which references the work of Scott and Varian (2014).

## Learning Goals

By the end of this post, you will:
- Understand the **Local Linear Trend** model.
- See how BSTS decomposes a series into **Level**, **Slope**, **Seasonality**, and **Regressors**.
- Recognize the modular nature of Structural Time Series.

## Section 2.2.3: Bayesian Structural Time Series

The Orbit paper highlights BSTS as a key approach in the state-space family. The core idea is to define a model where the observed value $y_t$ is the sum of several distinct components, each evolving according to its own rules.

### The Local Linear Trend Model

A common specification in BSTS is the **Local Linear Trend** model with seasonality and regressors.

#### 1. Observation Equation

$$
y_t = \mu_t + \tau_t + \beta^T x_t + \epsilon_t
$$

Where:
- $\mu_t$: The **trend** component (level).
- $\tau_t$: The **seasonal** component.
- $\beta^T x_t$: The **regression** component (effect of external predictors $x_t$).
- $\epsilon_t$: The observation error.

#### 2. State Update Equations

The components evolve as follows:

**Level and Slope ($\mu_t$ and $\delta_t$):**
$$
\begin{aligned}
\mu_{t+1} &= \mu_t + \delta_t + \eta_{0,t} \\
\delta_{t+1} &= \delta_t + \eta_{1,t}
\end{aligned}
$$
Here, $\mu_t$ represents the current level of the series, and $\delta_t$ represents the current *slope* (rate of growth). Both are allowed to drift over time due to the noise terms $\eta_{0,t}$ and $\eta_{1,t}$. This allows the model to adapt to changing trends.

**Seasonality ($\tau_t$):**
$$
\tau_{t+1} = - \sum_{s=1}^{S-1} \tau_{t} + \eta_{2,t}
$$
*(Note: The notation in the paper implies a sum-to-zero constraint, ensuring that the seasonal effects sum to approximately zero over a full cycle $S$.)*

### Why BSTS?

1.  **Modularity**: You can add or remove components (e.g., add a holiday effect, remove seasonality) without changing the rest of the model structure.
2.  **External Regressors**: The term $\beta^T x_t$ allows you to explicitly model the impact of outside factors like marketing spend, weather, or price changes.
3.  **Bayesian Inference**: By using Bayesian methods (often MCMC), BSTS provides full posterior distributions for forecasts, giving you credible intervals that naturally account for parameter uncertainty.

### Connection to Orbit

The Orbit package draws inspiration from these structural models. Orbit's **DLT (Damped Local Trend)** model, which we will discuss in the next post, can be seen as a specific variation of this structural approach, optimized for stability and performance in business forecasting contexts.

## What's Next?

In **TS 105**, we will finally bring everything together and dive deep into the **Orbit** package itself. We'll explore the **LGT** and **DLT** models in detail, seeing how they combine the best of Exponential Smoothing and Bayesian Structural Time Series.

## References

- Ng, E., Wang, Z., Chen, H., Yang, S., & Smyl, S. (2020). *Orbit: Probabilistic Forecast with Exponential Smoothing*. [arXiv:2004.08492](https://arxiv.org/abs/2004.08492).
- Scott, S. L., & Varian, H. R. (2014). *Predicting the present with Bayesian structural time series*. International Journal of Mathematical Modelling and Numerical Optimisation.
