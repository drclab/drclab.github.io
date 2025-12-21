import pandas as pd
import matplotlib.pyplot as plt
import os

# Create output directory
output_dir = "/home/cjduan/drclab.github.io/static/images/pandas-102"
os.makedirs(output_dir, exist_ok=True)

# Load the dataset
dataset_path = "/home/cjduan/drclab.github.io/content/ipynb/rideshare_2022.csv"
df = pd.read_csv(dataset_path, parse_dates=['Trip Start Timestamp', 'Trip End Timestamp'])

# 1. Rides per day histogram
plt.figure(figsize=(10, 6))
df['date'] = pd.to_datetime(df['Trip Start Timestamp'].dt.date)
df.hist('date', density=True)
plt.title('Distribution of Rides Throughout the Year')
plt.xlabel('Date')
plt.ylabel('Density')
plt.savefig(os.path.join(output_dir, 'rides_per_day.png'))
plt.close()

# 2. Tip distribution (all)
plt.figure(figsize=(10, 6))
df.hist('Tip', density=True, bins=100)
plt.title('Distribution of Tips (All Rides)')
plt.xlabel('Tip ($)')
plt.ylabel('Density')
plt.savefig(os.path.join(output_dir, 'tip_distribution_all.png'))
plt.close()

# 3. Tip distribution (tippers only)
plt.figure(figsize=(10, 6))
tippers = df['Tip'] > 0
df_tippers = df[tippers]
df_tippers.hist('Tip', density=True, bins=100)
plt.title('Distribution of Tips (Conditional on Tip > 0)')
plt.xlabel('Tip ($)')
plt.ylabel('Density')
plt.savefig(os.path.join(output_dir, 'tip_distribution_tippers.png'))
plt.close()

print(f"Assets generated successfully in {output_dir}")
