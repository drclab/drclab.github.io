import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

# Create output directory
output_dir = "static/images/pandas-101"
os.makedirs(output_dir, exist_ok=True)

# Load data
csv_path = "content/ipynb/world_happiness.csv"
df = pd.read_csv(csv_path)

# Rename columns (replicate notebook steps)
df.rename(columns={c: c.lower().replace(" ", "_") for c in df.columns}, inplace=True)

# 1. Basic Plot (Line plot)
plt.figure(figsize=(10, 6))
df.plot()
plt.savefig(os.path.join(output_dir, "line_plot.png"))
plt.close()

# 2. Scatter Plot: GDP vs Life Ladder
plt.figure(figsize=(10, 6))
df.plot(kind='scatter', x='log_gdp_per_capita', y='life_ladder')
plt.savefig(os.path.join(output_dir, "scatter_plot.png"))
plt.close()

# 3. Colored Scatter Plot
cmap = {
    'Brazil': 'Green',
    'Slovenia': 'Orange',
    'India': 'purple'
}
plt.figure(figsize=(10, 6))
df.plot(
    kind='scatter',
    x='log_gdp_per_capita',
    y='life_ladder',
    c=[cmap.get(c, 'yellow') for c in df.country_name],
    s=2
)
plt.savefig(os.path.join(output_dir, "scatter_plot_colored.png"))
plt.close()

# 4. Histogram: Life Ladder
plt.figure(figsize=(10, 6))
df.hist("life_ladder")
plt.savefig(os.path.join(output_dir, "histogram.png"))
plt.close()

# 5. Pair Plot
plt.figure(figsize=(10, 10))
sns.pairplot(df)
plt.savefig(os.path.join(output_dir, "pair_plot.png"))
plt.close()

print(f"Assets generated successfully in {output_dir}")
