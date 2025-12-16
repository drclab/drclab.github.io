
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.datasets import make_blobs
import os

# Create output directory
output_dir = "static/images/perceptron-102"
os.makedirs(output_dir, exist_ok=True)

# Set seed
np.random.seed(3)

# --- Functions ---
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def initialize_parameters(n_x, n_y):
    W = np.random.randn(n_y, n_x) * 0.01
    b = np.zeros((n_y, 1))
    return {"W": W, "b": b}

def forward_propagation(X, parameters):
    W = parameters["W"]
    b = parameters["b"]
    Z = np.matmul(W, X) + b
    A = sigmoid(Z)
    return A

def compute_cost(A, Y):
    m = Y.shape[1]
    logprobs = - np.multiply(np.log(A),Y) - np.multiply(np.log(1 - A),1 - Y)
    cost = 1/m * np.sum(logprobs)
    return cost

def backward_propagation(A, X, Y):
    m = X.shape[1]
    dZ = A - Y
    dW = 1/m * np.dot(dZ, X.T)
    db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
    return {"dW": dW, "db": db}

def update_parameters(parameters, grads, learning_rate=1.2):
    W = parameters["W"]
    b = parameters["b"]
    dW = grads["dW"]
    db = grads["db"]
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return {"W": W, "b": b}

def layer_sizes(X, Y):
    n_x = X.shape[0]
    n_y = Y.shape[0]
    return (n_x, n_y)

def nn_model(X, Y, num_iterations=10, learning_rate=1.2, print_cost=False):
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[1]
    parameters = initialize_parameters(n_x, n_y)
    
    for i in range(0, num_iterations):
        A = forward_propagation(X, parameters)
        cost = compute_cost(A, Y)
        grads = backward_propagation(A, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate)
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    return parameters

def plot_decision_boundary(X, Y, parameters, filename):
    W = parameters["W"]
    b = parameters["b"]

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.scatter(X[0, :], X[1, :], c=Y[0, :], cmap=colors.ListedColormap(['blue', 'red']), edgecolors='k')
    
    # Calculate boundary line: w1*x1 + w2*x2 + b = 0 => x2 = -(w1*x1 + b)/w2
    x_min, x_max = np.min(X[0,:]), np.max(X[0,:])
    # Add some padding
    x_range = x_max - x_min
    x_line = np.arange(x_min - 0.1*x_range, x_max + 0.1*x_range, 0.1)
    
    y_line = - W[0,0] / W[0,1] * x_line + -b[0,0] / W[0,1]
    
    ax.plot(x_line, y_line, color="black", linewidth=2)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xlim([x_min - 0.1*x_range, x_max + 0.1*x_range])
    # Adjust ylim based on data or just let matplotlib handle it? 
    # Let's let matplotlib handle it but constrained by line? 
    # Actually, let's just use data limits with some padding
    y_min, y_max = np.min(X[1,:]), np.max(X[1,:])
    y_range = y_max - y_min
    ax.set_ylim([y_min - 0.1*y_range, y_max + 0.1*y_range])
    
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# --- 1. Simple Classification Problem ---
# Recreating the exact dataset logic from notebook section 2.2
m = 30
X = np.random.randint(0, 2, (2, m))
Y = np.logical_and(X[0] == 0, X[1] == 1).astype(int).reshape((1, m))

# Plot 1: Simple Scatter (Just data)
# Note: The notebook first cell plots a manual example (4 points). 
# Section 2.2 generates random 30 points.
# The post text refers to "Simple Classification Problem" with an image.
# I will plot the 30 points dataset generated in 2.2 as the "Simple Classification" example
# or should I reproduce the 4 points manually?
# Post text says: "Imagine determining... set of sentences... 4 sentences".
# Then "Let's take a very simple set of 4 sentences".
# Let's plot the 4 points example for the first image `simple_classification.png`.

fig, ax = plt.subplots(figsize=(6, 4))
xmin, xmax = -0.2, 1.4
x_line_manual = np.arange(xmin, xmax, 0.1)
# Data points from notebook Section 1
ax.scatter([0, 1, 1], [0, 0, 1], color="b", label="Happy (0)")
ax.scatter([0], [1], color="r", label="Angry (1)")
ax.set_xlim([xmin, xmax])
ax.set_ylim([-0.1, 1.1])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
# Manual line x2 = x1 + 0.5
ax.plot(x_line_manual, x_line_manual + 0.5, color="black", linestyle="--", label="Decision Boundary")
plt.legend()
plt.savefig(os.path.join(output_dir, "simple_classification.png"))
plt.close()


# Train on the 30 points dataset
parameters = nn_model(X, Y, num_iterations=50, learning_rate=1.2, print_cost=False)

# Plot 2: Decision Boundary Simple
plot_decision_boundary(X, Y, parameters, "decision_boundary_simple.png")


# --- 2. Larger Dataset ---
n_samples = 1000
samples, labels = make_blobs(n_samples=n_samples, 
                             centers=([2.5, 3], [6.7, 7.9]), 
                             cluster_std=1.4,
                             random_state=0)

X_larger = np.transpose(samples)
Y_larger = labels.reshape((1,n_samples))

# Plot 3: Larger Dataset (Scatter only)
fig, ax = plt.subplots(figsize=(8, 6))
plt.scatter(X_larger[0, :], X_larger[1, :], c=Y_larger[0, :], cmap=colors.ListedColormap(['blue', 'red']), s=10)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
plt.savefig(os.path.join(output_dir, "larger_dataset.png"))
plt.close()

# Train larger model
parameters_larger = nn_model(X_larger, Y_larger, num_iterations=100, learning_rate=1.2, print_cost=False)

# Plot 4: Decision Boundary Large
plot_decision_boundary(X_larger, Y_larger, parameters_larger, "decision_boundary_large.png")

print("Assets generated successfully.")
