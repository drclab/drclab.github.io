import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import os

# Create directory for images if it doesn't exist
os.makedirs('static/images/sklearn-101', exist_ok=True)

# Load and clean data
df = pd.read_csv('content/ipynb/world_happiness.csv')
df = df.rename(columns={i: "_".join(i.split(" ")).lower() for i in df.columns})
df = df.dropna()

# 1. Pair Plot
print("Generating pair plot...")
sns.pairplot(df)
plt.savefig('static/images/sklearn-101/pair_plot.png')
plt.close()

# 2. Simple Linear Regression Plot
print("Generating simple linear regression plot...")
X = df[['log_gdp_per_capita']]
y = df['life_ladder']

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', alpha=0.5, label='Actual Data')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Log GDP per capita')
plt.ylabel('Life Ladder')
plt.title('Simple Linear Regression: Log GDP per capita vs Life Ladder')
plt.legend()
plt.savefig('static/images/sklearn-101/simple_regression.png')
plt.close()

print("Assets generated successfully.")
