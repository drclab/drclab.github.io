+++
title = "TS 103: Exponential Smoothing (ETS)"
slug = "ts-103"
aliases = ["ts103"]
date = "2028-11-28T00:00:00Z"
type = "post"
draft = false
math = true
tags = ["time-series", "ets", "exponential-smoothing", "forecasting"]
categories = ["posts"]
description = "Exploring the Exponential Smoothing (ETS) family of models, a state-space approach that naturally handles trend and seasonality."
+++

Welcome to TS 103! In [TS 102](../ts-102), we looked at ARIMA, which relies on differencing to handle non-stationary data. Now, we turn to **Exponential Smoothing (ETS)**, a family of models that explicitly models the components of a time series—level, trend, and seasonality—and updates them over time.

This post covers material from Section 2.2.2 of the paper [*Orbit: Probabilistic Forecast with Exponential Smoothing*](https://arxiv.org/abs/2004.08492).

## Learning Goals

By the end of this post, you will:
- Understand the **State Space** formulation of Exponential Smoothing.
- Learn the **ETS taxonomy**: Error, Trend, and Seasonality.
- See how popular methods like **Holt-Winters** fit into this framework.
- Recognize the recursive nature of state updates.

## Section 2.2.2: Exponential Smoothing

The Orbit paper presents Exponential Smoothing in a reduced state space form. Unlike ARIMA, which looks at past values, ETS looks at an evolving **latent state**.

### The General Form

The model is defined by three key equations:

1.  **Prediction Equation**:
    $$
    \hat{y}_{t|t-1} = Z^T \alpha_{t-1}
    $$
    The forecast for time $t$ (given information up to $t-1$) is a linear combination of the previous state $\alpha_{t-1}$.

2.  **Error Equation**:
    $$
    \epsilon_t = y_t - \hat{y}_{t|t-1}
    $$
    The error (innovation) is the difference between the actual observation $y_t$ and our prediction.

3.  **State Update Equation**:
    $$
    \alpha_t = T \alpha_{t-1} + k \epsilon_t
    $$
    The new state $\alpha_t$ is a transition from the old state ($T \alpha_{t-1}$) plus an adjustment based on the error ($k \epsilon_t$).

Where:
- $\alpha_t$: State vector (e.g., level, trend, seasonality).
- $Z$: Measurement vector.
- $T$: Transition matrix.
- $k$: Persistence vector (smoothing parameters).

**Intuition**: We have a mental model of the world (the state). We make a prediction. We observe reality. We calculate the error. We update our mental model based on that error.

### The ETS Taxonomy

Hyndman et al. (2008) categorized these models using the **ETS** notation: **E**rror, **T**rend, **S**easonality.

Each component can be:
- **N**: None
- **A**: Additive
- **M**: Multiplicative
- **Ad**: Additive Damped (for trend)

#### Common Examples

| Method | ETS Notation | Description |
|--------|--------------|-------------|
| **Simple Exponential Smoothing** | ETS(A, N, N) | Level only. No trend, no seasonality. |
| **Holt's Linear Trend** | ETS(A, A, N) | Level + Linear Trend. |
| **Holt-Winters (Additive)** | ETS(A, A, A) | Level + Trend + Additive Seasonality. |
| **Holt-Winters (Multiplicative)** | ETS(A, A, M) | Level + Trend + Multiplicative Seasonality. |

The Orbit paper notes that well-known methods like **Holt-Winters** are simply specific configurations of this general ETS framework.

### Why ETS?

1.  **Interpretability**: The states ($\alpha_t$) directly correspond to concepts we understand: "current level", "current growth rate", "seasonal factor".
2.  **Flexibility**: We can mix and match additive and multiplicative components.
3.  **Efficiency**: The recursive update form is computationally efficient.

## Connection to Orbit

The Orbit package builds directly on this foundation. As we'll see in the next post, Orbit's **LGT (Local Global Trend)** and **DLT (Damped Local Trend)** models are essentially Bayesian extensions of these ETS models, allowing for:
- Probabilistic inference (uncertainty quantification).
- Global trend components.
- Regression with exogenous variables.

## What's Next?

In **TS 104**, we will dive into the specific models proposed in the Orbit paper: **LGT** and **DLT**. We'll see how they refine the standard ETS approach to handle complex real-world data.

## References

- Ng, E., Wang, Z., Chen, H., Yang, S., & Smyl, S. (2020). *Orbit: Probabilistic Forecast with Exponential Smoothing*. [arXiv:2004.08492](https://arxiv.org/abs/2004.08492).
- Hyndman, R. J., Koehler, A. B., Ord, J. K., & Snyder, R. D. (2008). *Forecasting with Exponential Smoothing: The State Space Approach*.
