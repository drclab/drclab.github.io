
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import statsmodels.formula.api as smf
import os

# Ensure directory exists
output_dir = "static/images/pandas-104"
os.makedirs(output_dir, exist_ok=True)

# Load data
df = pd.read_csv("content/ipynb/rideshare_2022_cleaned.csv", parse_dates=['trip_start_timestamp', 'date'])

# 1. Daily Rides Histogram
daily_rides = df.groupby('date').size().reset_index(name='daily_rides')
mean_rides_per_day = daily_rides['daily_rides'].mean()

plt.figure(figsize=(18,6))
plt.bar(daily_rides['date'], daily_rides['daily_rides'], label='Rides per Day')
plt.axhline(y=mean_rides_per_day, c='r', label=f'Mean Rides per Day')
plt.ylabel('Rides per Day', fontsize=16)
plt.xlabel('Date', fontsize=16)
plt.xlim(min(daily_rides['date']), max(daily_rides['date']))
plt.legend(fontsize=14)
plt.savefig(f"{output_dir}/daily_rides_hist.png")
plt.close()

# 2. Daily Rides with CI
std_rides_per_day = daily_rides['daily_rides'].std()
confidence = 0.95
critical_value = scipy.stats.t.ppf(1 - (1 - confidence)/2, df=len(daily_rides)-1)
total_days = daily_rides['date'].count()
confidence_interval = critical_value * std_rides_per_day / np.sqrt(total_days)

plt.figure(figsize=(18,6))
plt.bar(daily_rides['date'], daily_rides['daily_rides'], label='Rides per Day')
plt.axhline(y=mean_rides_per_day, c='r', label=f'Mean Rides per Day +/- {confidence}% Confidence Interval')
plt.fill_between(daily_rides['date'], mean_rides_per_day-confidence_interval,
                 mean_rides_per_day+confidence_interval, color='r', alpha=0.2)
plt.ylabel('Rides per Day', fontsize=16)
plt.xlabel('Date', fontsize=16)
plt.xlim(min(daily_rides['date']), max(daily_rides['date']))
plt.legend(fontsize=14)
plt.savefig(f"{output_dir}/daily_rides_ci.png")
plt.close()

# 3. Daily Rides with Holidays
daily_rides_holidays = daily_rides[daily_rides["date"] > "2022-12-17"]
mean_rides_per_day_holidays = daily_rides_holidays['daily_rides'].mean()
std_rides_per_day_holidays = daily_rides_holidays['daily_rides'].std()
critical_value_holidays =  scipy.stats.t.ppf(1 - (1 - confidence)/2, df=len(daily_rides_holidays)-1)
total_days_holidays = daily_rides_holidays['date'].count()
confidence_interval_holidays = critical_value * std_rides_per_day_holidays / np.sqrt(total_days_holidays)

plt.figure(figsize=(18,6))
plt.bar(daily_rides['date'], daily_rides['daily_rides'], label='Rides per Day')
plt.axhline(y=mean_rides_per_day, color='C0', label=f'Mean Rides per Day +/- {confidence}% Confidence Interval')
plt.fill_between(daily_rides['date'], mean_rides_per_day-confidence_interval,
                 mean_rides_per_day+confidence_interval, color='C0', alpha=0.3)
plt.bar(daily_rides_holidays['date'], daily_rides_holidays['daily_rides'], label='Rides per Day (Holidays)')
plt.axhline(y=mean_rides_per_day_holidays, color='C1', label='Mean Rides per Day (Holidays) +/- {confidence}% Confidence Interval')
plt.fill_between(daily_rides_holidays['date'], mean_rides_per_day_holidays-confidence_interval_holidays,
                 mean_rides_per_day_holidays+confidence_interval_holidays, color='C1', alpha=0.5)
plt.ylabel('Rides per Day', fontsize=16)
plt.xlabel('Date', fontsize=16)
plt.xlim(min(daily_rides['date']), max(daily_rides_holidays['date']))
plt.legend(fontsize=14)
plt.savefig(f"{output_dir}/daily_rides_holidays.png")
plt.close()

# 4. Scatter Plots
fig, ax = plt.subplots(1,2, figsize=(12,4))
df.plot.scatter('fare','trip_seconds', ax=ax[0])
df.plot.scatter('fare','trip_miles', ax=ax[1])
plt.savefig(f"{output_dir}/fare_scatter.png")
plt.close()

# 5. Prediction Plot
model = smf.ols(formula='fare ~ trip_seconds + trip_miles', data=df)
result = model.fit()
x_y = df[["trip_miles", "trip_seconds", "fare"]].dropna()
x_variable = "trip_seconds"
x_plot =  x_y[x_variable]
y_plot =  x_y["fare"]
y_result = result.predict()

plt.figure()
plt.scatter(x_plot, y_plot, label="Original Data")
plt.scatter(x_plot, y_result, label="Prediction")
plt.xlabel(" ".join(x_variable.split("_")).title(), fontsize=14)
plt.ylabel("Fare", fontsize=14)
plt.legend(fontsize=14)
plt.savefig(f"{output_dir}/prediction_plot.png")
plt.close()
