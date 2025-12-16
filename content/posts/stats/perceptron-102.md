+++
title = "Perceptron 102: Classification with Perceptron"
date = "2025-12-16"
tags = ["perceptron", "classification", "neural networks", "python", "numpy"]
categories = ["posts"]
series = ["Perceptron"]
type = "post"
draft = false
math = true
description = "Building a single perceptron neural network with sigmoid activation for binary classification."
+++

In this post, we explore how to use a single perceptron neural network model to solve a simple classification problem. This builds on the concepts of regression, but introduces the **activation function** to handle binary outputs.

## 1. Simple Classification Problem

**Classification** is the problem of identifying which of a set of categories an observation belongs to. A **binary classification problem** involves only two categories.

Imagine determining if a sentence is "happy" or "angry" based on the count of specific words. If we plot these observations, we might find that a straight line allows us to separate the two classes (blue for happy, red for angry).

![Simple Classification](/images/perceptron-102/simple_classification.png)

This is a problem with **two linearly separable classes**. We want to find a line decision boundary such that points on one side are classified as class 0 and points on the other as class 1.

## 2. Neural Network with Sigmoid Activation

To solve this with a neural network, we use a single perceptron. Unlike regression where we outputs a real number $z$, for classification we need a probability between 0 and 1. We achieve this by applying an **activation function**.

### Sigmoid Activation
We use the **sigmoid function**:

$$a = \sigma(z) = \frac{1}{1+e^{-z}}$$

This maps any real number $z$ to the interval $(0, 1)$. We can then interpret $a$ as the probability that the input belongs to class 1.

The model becomes:
$$
\begin{align}
z^{(i)} &= W x^{(i)} + b \\
a^{(i)} &= \sigma(z^{(i)})
\end{align}
$$

### Predictions
To get a final class prediction $\hat{y}$ (0 or 1), we use a threshold of 0.5:

$$\hat{y} = \begin{cases} 1 & \text{if } a > 0.5 \\ 0 & \text{otherwise} \end{cases}$$

## 3. Implementation

Let's implement this in Python using NumPy.

### Initialization
We define the structure based on input and output sizes and initialize parameters $W$ and $b$.

```python
import numpy as np

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def initialize_parameters(n_x, n_y):
    W = np.random.randn(n_y, n_x) * 0.01
    b = np.zeros((n_y, 1))
    return {"W": W, "b": b}
```

### Forward Propagation
We compute $Z = WX + b$ and then apply the sigmoid activation.

```python
def forward_propagation(X, parameters):
    W = parameters["W"]
    b = parameters["b"]
    Z = np.dot(W, X) + b
    A = sigmoid(Z)
    return A
```

### Cost Function
For classification, we use the **log loss** function:

$$\mathcal{L}(W, b) = -\frac{1}{m}\sum_{i=1}^{m} \left[ y^{(i)}\log(a^{(i)}) + (1-y^{(i)})\log(1- a^{(i)}) \right]$$

```python
def compute_cost(A, Y):
    m = Y.shape[1]
    logprobs = - np.multiply(np.log(A), Y) - np.multiply(np.log(1 - A), 1 - Y)
    cost = 1/m * np.sum(logprobs)
    return cost
```

### Backward Propagation
We calculate gradients to minimize the cost. The formulas turn out to be elegant in matrix form:

$$
\frac{\partial \mathcal{L} }{ \partial W } = \frac{1}{m}(A - Y)X^T, \quad
\frac{\partial \mathcal{L} }{ \partial b } = \frac{1}{m}(A - Y)\mathbf{1}
$$

```python
def backward_propagation(A, X, Y):
    m = X.shape[1]
    dZ = A - Y
    dW = 1/m * np.dot(dZ, X.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    return {"dW": dW, "db": db}
```

### Optimization Loop
Putting it all together to train the model:

```python
def nn_model(X, Y, num_iterations=10, learning_rate=1.2):
    n_x = X.shape[0]
    n_y = Y.shape[0]
    parameters = initialize_parameters(n_x, n_y)
    
    for i in range(num_iterations):
        A = forward_propagation(X, parameters)
        cost = compute_cost(A, Y)
        grads = backward_propagation(A, X, Y)
        
        # Update parameters
        parameters["W"] = parameters["W"] - learning_rate * grads["dW"]
        parameters["b"] = parameters["b"] - learning_rate * grads["db"]
        
    return parameters
```

## 4. Results

### Simple Dataset
Training on our simple dataset of 30 points, the model learns a decision boundary that separates the two classes effectively.

![Decision Boundary Simple](/images/perceptron-102/decision_boundary_simple.png)

### Larger Dataset
We can also apply this to a larger, more complex dataset generated with `make_blobs`.

![Larger Dataset](/images/perceptron-102/larger_dataset.png)

After training for 100 iterations, the perceptron finds a linear boundary separating the clusters.

![Decision Boundary Large](/images/perceptron-102/decision_boundary_large.png)

This demonstrates the power of a single perceptron for linearly separable data. For non-linear problems, we would need to look at multi-layer networks.
