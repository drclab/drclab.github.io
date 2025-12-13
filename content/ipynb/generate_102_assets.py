import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add current directory to path to find w2_tools
sys.path.append(os.getcwd())
from w2_tools import plot_f, f_example_2, dfdx_example_2

# Setup output directory
output_dir = "../../static/images/calculus-102"
os.makedirs(output_dir, exist_ok=True)

# Gradient Descent Implementation
def gradient_descent(dfdx, x, learning_rate=0.1, num_iterations=100):
    for iteration in range(num_iterations):
        x = x - learning_rate * dfdx(x)
    return x

# --- Example 1 ---
print("--- Example 1 ---")
def f_example_1(x):
    return np.exp(x) - np.log(x)

def dfdx_example_1(x):
    return np.exp(x) - 1/x

# Plot Example 1
fig, ax = plot_f([0.001, 2.5], [-0.3, 13], f_example_1, 0.0)
fig.savefig(os.path.join(output_dir, "example_1_plot.png"))
print(f"Saved example_1_plot.png")

# Run GD Example 1
num_iterations = 25
learning_rate = 0.1
x_initial = 1.6
x_min = gradient_descent(dfdx_example_1, x_initial, learning_rate, num_iterations)
print(f"Gradient descent result: x_min = {x_min}")

# --- Example 2 ---
print("\n--- Example 2 ---")
# Plot Example 2
fig, ax = plot_f([0.001, 2], [-6.3, 5], f_example_2, -6)
fig.savefig(os.path.join(output_dir, "example_2_plot.png"))
print(f"Saved example_2_plot.png")

# Run GD Example 2 - Case 1
learning_rate = 0.005
num_iterations = 35
x_initial_1 = 1.3 # Note: Notebook used 1.3, my draft said 1.5. I should align with notebook for accuracy or stick to my draft if I change the code. 
# Notebook says: x_initial = 1.3
min_1 = gradient_descent(dfdx_example_2, x=x_initial_1, learning_rate=learning_rate, num_iterations=num_iterations)
print(f"Global minimum: x_min = {min_1}")

# Run GD Example 2 - Case 2
x_initial_2 = 0.25
min_2 = gradient_descent(dfdx_example_2, x=x_initial_2, learning_rate=learning_rate, num_iterations=num_iterations)
print(f"Local minimum: x_min = {min_2}")
