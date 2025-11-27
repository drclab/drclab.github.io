#!/usr/bin/env python3
"""Execute Prophet demo and save outputs"""

from pathlib import Path

import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

ROOT_DIR = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT_DIR / "static/img/posts/ts-201-prophet"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the dataset
url = 'https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv'
df = pd.read_csv(url)

print("Data loaded:")
print(df.head())
print(f"\nDataset shape: {df.shape}")
print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")

# Fit the model
print("\nFitting Prophet model...")
m = Prophet()
m.fit(df)

# Create future dataframe
future = m.make_future_dataframe(periods=365)
print(f"\nFuture dataframe created with {len(future)} dates")
print("Last 5 dates:")
print(future.tail())

# Predict
forecast = m.predict(future)
print("\nForecast generated:")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# Plot and save the main forecast
fig1 = m.plot(forecast)
forecast_path = OUTPUT_DIR / "prophet_forecast.png"
plt.savefig(forecast_path, dpi=150, bbox_inches='tight')
print(f"\nSaved {forecast_path}")
plt.close()

# Plot and save the components
fig2 = m.plot_components(forecast)
components_path = OUTPUT_DIR / "prophet_components.png"
plt.savefig(components_path, dpi=150, bbox_inches='tight')
print(f"Saved {components_path}")
plt.close()

# Super Bowl 50 example
target_date = '2016-02-07'
row = forecast[forecast['ds'] == target_date]

print(f"\n{'='*60}")
print(f"Forecast for {target_date} (Super Bowl 50):")
print(f"{'='*60}")
print(f"Trend:         {row['trend'].values[0]:.4f}")
print(f"Weekly:        {row['weekly'].values[0]:.4f}")
print(f"Yearly:        {row['yearly'].values[0]:.4f}")
print(f"yhat (Sum):    {row['yhat'].values[0]:.4f}")
print(f"{'='*60}")

# Save forecast to CSV for reference
forecast.to_csv('prophet_forecast_full.csv', index=False)
print("\nSaved prophet_forecast_full.csv")

print("\nDone!")
