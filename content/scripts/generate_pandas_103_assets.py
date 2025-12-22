import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for premium aesthetics
plt.style.use('ggplot')
sns.set_theme(style="whitegrid", palette="muted")

# Load data
# Assuming the file exists in the content/ipynb/ directory as per previous notebook analysis
data_path = '/home/cjduan/drclab.github.io/content/ipynb/rideshare_2022_cleaned.csv'
if not os.path.exists(data_path):
    # Fallback to the raw file if cleaned isn't there, or let it fail so I can debug
    data_path = '/home/cjduan/drclab.github.io/content/ipynb/rideshare_2022.csv'

df = pd.read_csv(data_path)
df['trip_start_timestamp'] = pd.to_datetime(df['trip_start_timestamp'])
df['hour'] = df['trip_start_timestamp'].dt.hour
df['weekday'] = df['trip_start_timestamp'].dt.day_name()

output_dir = '/home/cjduan/drclab.github.io/static/images/pandas-103/'
os.makedirs(output_dir, exist_ok=True)

# 1. Tip Boxplot by Weekday (All Trips)
plt.figure(figsize=(10, 6))
sns.boxplot(x='weekday', y='tip', data=df, order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.title('Tip Distribution by Weekday (All Trips)', fontsize=14)
plt.xlabel('Day of Week', fontsize=12)
plt.ylabel('Tip ($)', fontsize=12)
plt.savefig(os.path.join(output_dir, 'tip_boxplot_weekday.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. Tip Boxplot by Weekday (Tippers Only)
df_tippers = df[df['tip'] > 0]
plt.figure(figsize=(10, 6))
sns.boxplot(x='weekday', y='tip', data=df_tippers, order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.title('Tip Distribution by Weekday (Trips with Tips)', fontsize=14)
plt.xlabel('Day of Week', fontsize=12)
plt.ylabel('Tip ($)', fontsize=12)
plt.savefig(os.path.join(output_dir, 'tippers_boxplot_weekday.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. Tipper Percentage per Hour
hour_stats = df.groupby('hour')['tip'].apply(lambda x: (x > 0).mean() * 100).reset_index()
plt.figure(figsize=(12, 6))
sns.barplot(x='hour', y='tip', data=hour_stats, palette='viridis')
plt.title('Percentage of Trips with Tips by Hour', fontsize=14)
plt.xlabel('Hour of Day', fontsize=12)
plt.ylabel('% of Trips Tipped', fontsize=12)
plt.savefig(os.path.join(output_dir, 'tipper_percentage_hour.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4. Correlation: Trip Miles vs Tip
plt.figure(figsize=(10, 6))
sns.scatterplot(x='trip_miles', y='tip', data=df_tippers, alpha=0.5)
plt.title('Correlation: Trip Miles vs Tip Amount', fontsize=14)
plt.xlabel('Trip Miles', fontsize=12)
plt.ylabel('Tip ($)', fontsize=12)
plt.savefig(os.path.join(output_dir, 'miles_vs_tip_scatter.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"Successfully generated 4 plots in {output_dir}")
