+++
title = "TS 101: Foundational Time Series Forecasting"
slug = "ts-101"
aliases = ["ts101"]
date = "2028-11-26T00:00:00Z"
type = "post"
draft = false
math = true
tags = ["time-series", "exponential-smoothing", "forecasting"]
categories = ["posts"]
description = "A foundational introduction to time series forecasting concepts and model components, based on Uber's Orbit framework for exponential smoothing."
+++

Welcome to TS 101: Foundational Time Series Forecasting! This guide introduces the core concepts and notation for time series forecasting, focusing on how we decompose a time series into interpretable components. We'll build understanding from the ground up, using examples from Uber's Orbit package as a modern implementation reference.

This post covers foundational material from sections 2.1 and 2.2.1 of the paper [*Orbit: Probabilistic Forecast with Exponential Smoothing*](https://arxiv.org/abs/2004.08492) by Edwin Ng, Zhishi Wang, Huigang Chen, Steve Yang, and Slawek Smyl (2020).

## Learning Goals

By the end of this post, you will:
- understand the core components of time series models (trend, seasonality, error),
- learn standard notation used in time series forecasting,
- recognize the difference between additive and multiplicative formulations,
- see how classical statistical methods remain powerful for many forecasting tasks.

## Key Vocabulary

Before diving in, let's establish our terminology:

- **Time series**: A sequence of observations recorded at successive time points, denoted $y_1, y_2, \ldots, y_T$.
- **Forecast**: A predicted value for a future time point, typically denoted $\hat{y}_{T+h}$ where $h$ is the forecast horizon.
- **Components**: The building blocks of a time series model (trend, seasonality, error).
- **Trend**: The long-term direction or pattern in the data (upward, downward, flat).
- **Seasonality**: Regular, periodic fluctuations (daily, weekly, yearly patterns).
- **Error term**: The random noise or unpredictable variation in the data.
- **Additive model**: Components are summed together: $y_t = \text{trend}_t + \text{seasonal}_t + \epsilon_t$.
- **Multiplicative model**: Components are multiplied: $y_t = \text{trend}_t \times \text{seasonal}_t \times \epsilon_t$.
- **Exponential smoothing**: A family of forecasting methods that weight recent observations more heavily than older ones.

Keep these definitions handy as we build our understanding step by step.

## Section 2.1: Why Time Series Forecasting Matters

Time series forecasting is central to countless business and scientific applications:

- **Demand forecasting**: Predicting product sales to optimize inventory
- **Financial planning**: Forecasting revenue, costs, and cash flow
- **Energy management**: Predicting electricity demand to balance supply
- **Web traffic**: Anticipating user load to scale infrastructure
- **Healthcare**: Forecasting disease spread or patient admissions

While machine learning methods (neural networks, gradient boosting) receive significant attention, **statistical methods remain powerful** for time series forecasting, especially when:

1. **Data is limited**: You have only 50-100 observations rather than millions
2. **Granularity is low**: Weekly or monthly data rather than second-by-second readings
3. **Interpretability matters**: Stakeholders need to understand *why* the forecast changed
4. **Fast iteration is critical**: You need results in seconds, not hours of GPU training

Classical methods like exponential smoothing consistently perform well in forecasting competitions (the M-competitions) and are widely used in production systems at companies like Uber, Amazon, and Facebook.

## Section 2.2.1: The Fundamental Components

### The General Additive Form

At its core, a time series model decomposes observed values into meaningful pieces. The **general additive form** is:

$$
y_t = g_t + s_t + h_t + \epsilon_t,
$$

where each symbol represents a component:

| Symbol | Component | Meaning |
|--------|-----------|---------|
| $y_t$ | **Observation** | The actual value we observe at time $t$ |
| $g_t$ | **Growth (Trend)** | Non-periodic changes—the overall direction |
| $s_t$ | **Seasonality** | Periodic patterns that repeat at regular intervals |
| $h_t$ | **Holiday/Event** | Irregular, known events (Black Friday, holidays) |
| $\epsilon_t$ | **Error** | Random noise we cannot explain |

This equation says: *the value we observe equals the trend plus seasonal effects plus special events plus random noise*.

### Unpacking Each Component

Let's examine each piece with concrete examples:

#### 1. Trend ($g_t$): Where Are We Going?

The trend captures the **long-term direction** of the series. Common trend types include:

- **Linear trend**: $g_t = \delta_{\text{intercept}} + \delta_{\text{slope}} \cdot t$
  - Example: A startup's monthly revenue grows by \$10,000/month
  
- **Log-linear trend**: $g_t = \delta_{\text{intercept}} + \ln(\delta_{\text{slope}} \cdot t)$
  - Example: User growth that accelerates then slows (concave growth)
  
- **Logistic trend**: $g_t = L + \frac{U - L}{1 + e^{-\delta_{\text{slope}} \cdot t}}$
  - Example: Product adoption that saturates at market capacity $U$
  
- **Flat trend**: $g_t = \delta_{\text{intercept}}$ (constant, no growth)
  - Example: Stable mature products with no net growth

The trend component answers: *If we ignore short-term fluctuations, where is this series heading?*

#### 2. Seasonality ($s_t$): What Repeats?

Seasonality captures **regular, predictable cycles**. Examples:

- Ice cream sales peak every summer
- Website traffic drops every weekend
- Electricity demand spikes at 6 PM every weekday

Mathematically, seasonality often uses **Fourier series** to represent periodic patterns:
$$
s_t = \sum_{n=1}^{N} \left[ a_n \cos\left(\frac{2\pi n t}{P}\right) + b_n \sin\left(\frac{2\pi n t}{P}\right) \right],
$$

where $P$ is the period (e.g., $P = 7$ for weekly seasonality, $P = 12$ for monthly).

The seasonality component answers: *What regular patterns repeat over time?*

#### 3. Holiday/Event Effects ($h_t$): What's Special About Today?

Some dates are predictably different but don't follow a regular seasonal pattern:

- Thanksgiving always causes a sales spike, but its date varies
- Product launches create one-time jumps
- Marketing campaigns have known start and end dates

We model these as:
$$
h_t = \sum_{i} \delta_i \cdot \mathbb{1}(\text{event } i \text{ occurs at } t),
$$

where $\mathbb{1}(\cdot)$ is an indicator function (1 if the event occurs, 0 otherwise).

The holiday component answers: *Are there known special events affecting this time point?*

#### 4. Error ($\epsilon_t$): What's Left Over?

After accounting for trend, seasonality, and events, the **error term** captures:

- True randomness (customer behavior is inherently variable)
- Unmeasured factors (a competitor's promotion we don't track)
- Model misspecification (our trend or seasonal form isn't perfect)

Common error distributions:

- **Gaussian (Normal)**: $\epsilon_t \sim \mathcal{N}(0, \sigma^2)$ — symmetric, light-tailed
- **Student-t**: $\epsilon_t \sim t(\nu, 0, \sigma)$ — heavier tails, robust to outliers
- **Negative Binomial**: For count data (number of purchases)

The error component answers: *What variation remains unexplained?*

### Additive vs. Multiplicative

The additive form assumes components **add together**:
$$
y_t = g_t + s_t + h_t + \epsilon_t.
$$

This works well when seasonal fluctuations are roughly constant in magnitude. For example, if coffee sales increase by 100 cups every summer regardless of overall trend.

The **multiplicative form** assumes components **multiply**:
$$
y_t = g_t \times s_t \times h_t \times \epsilon_t.
$$

This is appropriate when seasonal effects scale with the level of the series. For example, if ice cream revenue grows 20% every summer, the absolute dollar increase grows as the base revenue grows.

**Key trick**: Take the logarithm to convert multiplicative to additive:
$$
\log(y_t) = \log(g_t) + \log(s_t) + \log(h_t) + \log(\epsilon_t).
$$

Now fit an additive model to $\log(y_t)$, then exponentiate forecasts to get back to the original scale. This requires $y_t > 0$ for all observations.

## Standard Notation Reference

When reading time series papers and documentation, you'll encounter consistent notation:

| Symbol | Meaning | Common Values |
|--------|---------|---------------|
| $t$ | Time index | $1, 2, \ldots, T$ |
| $y_t$ | Observed value at time $t$ | Any real number (or positive for log models) |
| $\hat{y}_t$ | Predicted/fitted value | Model's estimate for time $t$ |
| $\hat{y}_{T+h}$ | Forecast $h$ steps ahead | Prediction beyond observed data |
| $\mu_t$ | Latent level (smoothed estimate) | Underlying signal without noise |
| $\ell_t$ | Local level | Short-term smoothed value |
| $b_t$ | Local trend | Recent rate of change |
| $\tau$ | Global trend | Long-term drift parameter |
| $s_t$ | Seasonal component | Repeating pattern |
| $\epsilon_t$ | Error term | Random shock |
| $\alpha, \beta$ | Smoothing parameters | Between 0 and 1 |

## Why These Foundations Matter

Understanding these components is critical because:

1. **Debugging**: When forecasts fail, you can isolate which component is wrong (bad trend? missed seasonality?).
2. **Communication**: You can explain "revenue increased because of both strong trend and holiday effects."
3. **Model selection**: You choose models based on which components your data exhibits.
4. **Feature engineering**: For ML models, you can create features from these components.

Modern packages like Orbit, Prophet (Facebook), and NeuralProphet build on these foundations, adding:
- Bayesian inference for uncertainty quantification
- Automatic detection of changepoints in trend
- Hierarchical models for related time series
- Exogenous regressors (external predictors)

But they all decompose series into **trend + seasonality + events + error**.

## From Components to Forecasting

Once we've decomposed the series and estimated each component, forecasting is straightforward:

1. **Extrapolate the trend**: Project $g_t$ forward using the estimated growth rate.
2. **Add future seasonality**: Use the repeating seasonal pattern for future periods.
3. **Include known future events**: If we know Black Friday is coming, add its effect.
4. **Quantify uncertainty**: The error distribution gives us prediction intervals.

For example, to forecast 3 months ahead ($h = 3$):
$$
\hat{y}_{T+3} = \hat{g}_{T+3} + \hat{s}_{T+3} + \hat{h}_{T+3},
$$

where:
- $\hat{g}_{T+3}$ continues the estimated trend
- $\hat{s}_{T+3}$ uses the seasonal pattern from 3 months ago (assuming yearly seasonality)
- $\hat{h}_{T+3}$ includes any known events 3 months out

The uncertainty around this forecast comes from:
- Parameter uncertainty (we don't know $\alpha, \beta, \tau$ exactly)
- Future error realizations (random shocks we can't predict)

Bayesian methods give us full distributions; classical methods give us standard errors.

## What You Should Remember

The core takeaways for foundational time series forecasting:

1. **Decomposition is key**: Every series is trend + seasonality + events + noise.
2. **Notation is consistent**: $y_t$ is observed, $\hat{y}_t$ is fitted, $\epsilon_t$ is error.
3. **Additive vs. multiplicative**: Choose based on whether seasonal magnitude changes with level.
4. **Statistical methods are powerful**: Especially for low granularity, interpretable forecasts.
5. **Components aid understanding**: You can explain *why* the forecast is what it is.

## What's Next

In the next posts, we'll build on these foundations:

- **TS 102**: Exponential smoothing methods (Simple, Holt's, Holt-Winters)
- **TS 103**: The Local-Global Trend (LGT) model and Bayesian inference
- **TS 104**: Seasonal models and handling multiple seasonal periods
- **TS 105**: Implementing forecasts with the Orbit package in Python

This foundational understanding will make advanced methods feel like natural extensions rather than black boxes.

## References

- Ng, E., Wang, Z., Chen, H., Yang, S., & Smyl, S. (2020). *Orbit: Probabilistic Forecast with Exponential Smoothing*. [arXiv:2004.08492](https://arxiv.org/abs/2004.08492). [PDF](../../../pdf/$_Orbit_exponential_smoothing.pdf)
- [Orbit Documentation](https://uber.github.io/orbit/)
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts. [Free online](https://otexts.com/fpp3/)

---

*This post is part of the Time Series 101 series, covering foundational concepts for forecasting. For related Bayesian modeling topics, check out our MCMC 101 series.*
