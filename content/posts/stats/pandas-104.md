+++
title = "Pandas 104: Confidence Intervals & Hypothesis Testing"
date = "2026-07-03"
tags = ["pandas", "python", "data-science", "statistics", "hypothesis-testing"]
categories = ["posts"]
series = ["Python 101"]
type = "post"
draft = false
math = true
description = "Learn how to calculate confidence intervals, perform t-tests, and build linear regression models using Pandas and SciPy."
+++

In this final installment of our exploratory data analysis series, we'll dive into statistical inference. We'll use the 2022 Chicago rideshare data to calculate confidence intervals, perform hypothesis tests, and build a linear regression model to predict fares.

## 1. Daily Rides Analysis

First, let's look at the daily ride volume throughout the year.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import statsmodels.formula.api as smf

# Load the dataset
df = pd.read_csv("./rideshare_2022_cleaned.csv", parse_dates=['trip_start_timestamp', 'date'])

# Calculate daily rides
daily_rides = df.groupby('date').size().reset_index(name='daily_rides')

# Calculate mean and standard deviation
mean_rides_per_day = daily_rides['daily_rides'].mean()
std_rides_per_day = daily_rides['daily_rides'].std()

print(f'Mean number of rides per day: {mean_rides_per_day:.2f}')
print(f'Standard deviation: {std_rides_per_day:.2f}')
```

Output:
```
Mean number of rides per day: 1893.42
Standard deviation: 404.21
```

Here is the distribution of daily rides over the year:

![Daily Rides Histogram](/images/pandas-104/daily_rides_hist.png)

## 2. Confidence Intervals

How confident can we be in our estimate of the mean daily rides? We can construct a 95% confidence interval using the t-distribution.

```python
confidence = 0.95
critical_value = scipy.stats.t.ppf(1 - (1 - confidence)/2, df=len(daily_rides)-1)
total_days = daily_rides['date'].count()
confidence_interval = critical_value * std_rides_per_day / np.sqrt(total_days)

print(f"With a {100 * confidence}% confidence, the error is no more than {confidence_interval:.4f} rides per day.")
```

Output:
```
With a 95.0% confidence, the error is no more than 41.6059 rides per day.
```

Visualizing the confidence interval:

![Daily Rides with Confidence Interval](/images/pandas-104/daily_rides_ci.png)

The narrow confidence interval suggests a precise estimate of the population mean, thanks to our large sample size (365 days).

### Holiday Season Analysis

Notice the drop in rides during the holidays? Let's analyze the last two weeks of the year separately.

```python
daily_rides_holidays = daily_rides[daily_rides["date"] > "2022-12-17"]
mean_holidays = daily_rides_holidays['daily_rides'].mean()
std_holidays = daily_rides_holidays['daily_rides'].std()

print(f'Holiday Mean: {mean_holidays:.2f} +/- {std_holidays:.2f}')
```

![Daily Rides with Holidays](/images/pandas-104/daily_rides_holidays.png)

The holiday mean is lower, and the confidence interval is wider due to the smaller sample size (14 days).

## 3. Hypothesis Testing: Two-Sample t-test

Are there significantly more rides on Fridays and Saturdays compared to the rest of the week?

```python
WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_rides['weekday'] = daily_rides['date'].apply(lambda x: WEEKDAYS[x.weekday()])

fridays_saturdays = daily_rides[daily_rides["weekday"].isin(["Friday", "Saturday"])]["daily_rides"]
other_days = daily_rides[daily_rides["weekday"].isin(["Monday", "Tuesday", "Wednesday", "Thursday", "Sunday"])]["daily_rides"]

# Perform t-test
t_stat, p_value = scipy.stats.ttest_ind(a=fridays_saturdays, b=other_days, alternative='greater')
print(f"T-statistic: {t_stat}, P-value: {p_value}")
```

Output:
```
T-statistic: 21.5689, P-value: 2.66e-67
```

The extremely small p-value allows us to reject the null hypothesis. There is strong evidence that ride volume is higher on Fridays and Saturdays.

## 4. Linear Regression: Predicting Fares

Can we predict the fare based on trip distance and duration? First, let's check the correlations.

![Fare Scatter Plots](/images/pandas-104/fare_scatter.png)

There's a clear linear relationship. We'll use `statsmodels` to fit an Ordinary Least Squares (OLS) model.

```python
model = smf.ols(formula='fare ~ trip_seconds + trip_miles', data=df)
result = model.fit()
print(result.summary())
```

The model provides coefficients for the intercept (base fare), time cost, and distance cost.

```python
starting_fare = result.params["Intercept"]
price_per_minute = result.params["trip_seconds"] * 60
price_per_mile = result.params["trip_miles"]

print(f"Base Fare: ${starting_fare:.2f}")
print(f"Cost per Minute: ${price_per_minute:.2f}")
print(f"Cost per Mile: ${price_per_mile:.2f}")
```

Output:
```
Base Fare: $6.46
Cost per Minute: $0.34
Cost per Mile: $0.87
```

Finally, let's visualize our model's predictions against the actual data.

![Prediction Plot](/images/pandas-104/prediction_plot.png)

The model captures the general trend well, though there is some variance unexplained by just time and distance (likely due to surge pricing, traffic, etc.).

## Conclusion

We've covered a lot of ground:
1.  **Descriptive Statistics**: Analyzing daily ride distributions.
2.  **Confidence Intervals**: Quantifying the uncertainty of our mean estimates.
3.  **Hypothesis Testing**: Statistically confirming that weekends are busier.
4.  **Linear Regression**: Building a model to estimate ride fares.

This concludes our Pandas series on the Chicago rideshare dataset!
