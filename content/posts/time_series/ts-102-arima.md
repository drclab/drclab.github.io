+++
title = "TS 102: ARIMA Models"
slug = "ts-102"
aliases = ["ts102"]
date = "2028-11-27T00:00:00Z"
type = "post"
draft = false
math = true
tags = ["time-series", "arima", "forecasting"]
categories = ["posts"]
description = "An introduction to Autoregressive Integrated Moving Average (ARIMA) models, a classical approach to time series forecasting."
+++

Welcome to TS 102! In [TS 101](../ts-101), we explored the fundamental components of time series: trend, seasonality, and error. Now, we dive into one of the most widely used classical forecasting frameworks: **ARIMA** (Autoregressive Integrated Moving Average).

This post covers material from Section 2.2.1 of the paper [*Orbit: Probabilistic Forecast with Exponential Smoothing*](https://arxiv.org/abs/2004.08492), which presents ARIMA as a standard baseline for univariate time series modeling.

## Learning Goals

By the end of this post, you will:
- Understand the building blocks of ARIMA: **AR** (Autoregressive) and **MA** (Moving Average).
- Learn how **Integration (I)** handles non-stationary data.
- Interpret the notation $ARIMA(p, d, q)$.
- See how these models express current values as linear combinations of past values and errors.

## Key Vocabulary

- **Stationarity**: A property of a time series where statistical properties (mean, variance) do not change over time.
- **Lag**: A fixed amount of time passed; $y_{t-1}$ is the first lag of $y_t$.
- **Autoregression (AR)**: Predicting the current value based on past *values*.
- **Moving Average (MA)**: Predicting the current value based on past *forecast errors*.
- **Differencing**: Subtracting the previous observation from the current one ($y_t - y_{t-1}$) to stabilize the mean.

## Section 2.2.1: ARIMA

The Orbit paper introduces ARIMA as a method to model univariate time series by combining two key components: AR and MA.

### 1. The Autoregressive (AR) Model

In an **AR(p)** model, the value of a time series $y_t$ is estimated using a linear combination of its own past $p$ observations.

$$
y_t = c + \sum_{i=1}^{p} \psi_i y_{t-i} + \epsilon_t
$$

Where:
- $c$: A constant term (intercept).
- $\psi_i$: Parameters (coefficients) for the lagged values.
- $p$: The **order** of the AR model (how far back we look).
- $\epsilon_t$: The error term (white noise).

**Intuition**: If you want to predict today's temperature, looking at yesterday's temperature ($y_{t-1}$) is a good start. If it was hot yesterday, it's likely hot today.

### 2. The Moving Average (MA) Model

In an **MA(q)** model, the value $y_t$ is modeled using past *errors* (shocks) rather than past values.

$$
y_t = \mu + \sum_{i=1}^{q} \theta_i \epsilon_{t-i} + \epsilon_t
$$

Where:
- $\mu$: The mean of the observations.
- $\theta_i$: Parameters for the lagged errors.
- $q$: The **order** of the MA model.
- $\epsilon_{t-i}$: Past error terms.

**Intuition**: Sometimes a shock to the system (like a sudden news event) has a lingering effect. An MA model captures how these random shocks dissipate over time.

### 3. ARMA(p, q)

Combining these gives us the **ARMA(p, q)** model, which uses both past values and past errors:

$$
y_t = c + \sum_{i=1}^{p} \psi_i y_{t-i} + \sum_{i=1}^{q} \theta_i \epsilon_{t-i} + \epsilon_t
$$

This model is powerful but assumes the data is **stationary**—meaning it has a constant mean and variance over time.

### 4. ARIMA(p, d, q): Handling Non-Stationarity

Real-world data is rarely stationary. It often has trends (changing mean) or seasonality.

To handle this, we use **Integration (I)**, which simply means **differencing** the data.
- First order differencing: $y'_t = y_t - y_{t-1}$
- Second order differencing: $y''_t = y'_t - y'_{t-1}$

The parameter $d$ represents the number of times we difference the data to make it stationary.

**ARIMA(p, d, q)** works by:
1. Differencing the data $d$ times to remove trends.
2. Modeling the differenced data using an $ARMA(p, q)$ model.

## Summary

| Component | Parameter | Description | Equation Concept |
|-----------|-----------|-------------|------------------|
| **AR** | $p$ | Autoregressive | Regress on past values ($y_{t-1}$) |
| **I** | $d$ | Integrated | Difference to make stationary ($\Delta y_t$) |
| **MA** | $q$ | Moving Average | Regress on past errors ($\epsilon_{t-1}$) |

While modern Bayesian methods (like Orbit) and Deep Learning (like LSTMs) are gaining popularity, ARIMA remains a robust and interpretable baseline for many forecasting problems.

## What's Next?

In **TS 103**, we will look at the **Exponential Smoothing (ETS)** family, which offers an alternative state-space perspective that naturally handles seasonality and trends without explicit differencing, and forms the core of the Orbit package.

## References

- Ng, E., Wang, Z., Chen, H., Yang, S., & Smyl, S. (2020). *Orbit: Probabilistic Forecast with Exponential Smoothing*. [arXiv:2004.08492](https://arxiv.org/abs/2004.08492).
- Box, G. E. P., Jenkins, G. M., & Reinsel, G. C. (2015). *Time Series Analysis: Forecasting and Control*.
