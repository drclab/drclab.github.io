+++
title = "Linear Algebra 104: Vector Operations"
date = "2025-12-14"
type = "post"
draft = false
math = true
tags = ["linear-algebra", "numpy", "python", "vectors", "dot-product"]
categories = ["posts"]
description = "Introduction to vector operations: scalar multiplication, vector addition, norms, and dot products"
+++

# Vector Operations: Scalar Multiplication, Sum and Dot Product of Vectors

In this post, we'll explore fundamental vector operations using Python and NumPy, including scalar multiplication, vector addition, and dot products. We'll also examine the computational efficiency of vectorized operations compared to loop-based implementations.

## Scalar Multiplication and Sum of Vectors

### Visualization of a Vector

Vectors can be visualized as arrows in space. For a vector $v \in \mathbb{R}^2$, such as:

$$v = \begin{bmatrix} 1 \\ 3 \end{bmatrix}$$

The vector is defined by its **norm (length, magnitude)** and **direction**, not its actual position. For clarity and convenience, vectors are often plotted starting at the origin (in $\mathbb{R}^2$, the point $(0,0)$).

```python
import numpy as np
import matplotlib.pyplot as plt

v = np.array([[1], [3]])
```

### Scalar Multiplication

**Scalar multiplication** of a vector $v = \begin{bmatrix} v_1 & v_2 & \ldots & v_n \end{bmatrix}^T \in \mathbb{R}^n$ by a scalar $k$ produces a vector:

$$kv = \begin{bmatrix} kv_1 & kv_2 & \ldots & kv_n \end{bmatrix}^T$$

This is element-by-element multiplication where:
- If $k > 0$, then $kv$ points in the same direction as $v$ and is $k$ times as long
- If $k = 0$, then $kv$ is a zero vector
- If $k < 0$, then $kv$ points in the opposite direction

In Python, you can perform this operation with the `*` operator:

```python
v = np.array([[1], [3]])
# Examples: v, 2v, -2v
```

### Sum of Vectors

**Sum of vectors (vector addition)** is performed by adding corresponding components. If:

$$v = \begin{bmatrix} v_1 & v_2 & \ldots & v_n \end{bmatrix}^T \in \mathbb{R}^n$$

and

$$w = \begin{bmatrix} w_1 & w_2 & \ldots & w_n \end{bmatrix}^T \in \mathbb{R}^n$$

then:

$$v + w = \begin{bmatrix} v_1 + w_1 & v_2 + w_2 & \ldots & v_n + w_n \end{bmatrix}^T \in \mathbb{R}^n$$

The **parallelogram law** provides a geometric interpretation: for two vectors $u$ and $v$ represented by adjacent sides of a parallelogram, the sum $u + v$ is represented by the diagonal.

```python
v = np.array([[1], [3]])
w = np.array([[4], [-1]])

# Vector addition
result = v + w
# Or using NumPy function
result = np.add(v, w)
```

### Norm of a Vector

The norm of a vector $v$ is denoted as $|v|$. It's a nonnegative number describing the extent (length) of the vector in space.

```python
v = np.array([[1], [3]])
norm_v = np.linalg.norm(v)
print(f"Norm of vector v: {norm_v}")  # Output: 3.162...
```

## Dot Product

### Algebraic Definition

The **dot product** (or **scalar product**) is an algebraic operation that takes two vectors and returns a single scalar:

$$x \cdot y = \sum_{i=1}^{n} x_i y_i = x_1 y_1 + x_2 y_2 + \ldots + x_n y_n$$

### Dot Product in Python

Here's a simple implementation using a loop:

```python
x = [1, -2, -5]
y = [4, 3, -1]

def dot(x, y):
    s = 0
    for xi, yi in zip(x, y):
        s += xi * yi
    return s

print(f"Dot product: {dot(x, y)}")  # Output: 3
```

NumPy provides optimized functions for dot product calculation:

```python
x = np.array([1, -2, -5])
y = np.array([4, 3, -1])

# Using np.dot()
result1 = np.dot(x, y)

# Using @ operator (requires NumPy arrays)
result2 = x @ y

print(f"Dot product: {result1}")  # Output: 3
```

### Speed of Vectorized Operations

Vectorized operations are significantly faster than loop-based implementations, especially for high-dimensional vectors. This is crucial for Machine Learning applications where operations are performed on large datasets.

```python
import time

# Create large vectors
a = np.random.rand(1000000)
b = np.random.rand(1000000)

# Loop version
tic = time.time()
c = dot(a, b)
toc = time.time()
print(f"Loop version: {1000*(toc-tic)} ms")

# Vectorized version with np.dot()
tic = time.time()
c = np.dot(a, b)
toc = time.time()
print(f"Vectorized version (np.dot): {1000*(toc-tic)} ms")

# Vectorized version with @ operator
tic = time.time()
c = a @ b
toc = time.time()
print(f"Vectorized version (@): {1000*(toc-tic)} ms")
```

The vectorized versions are typically 10-100x faster than the loop version!

### Geometric Definition

In Euclidean space, the dot product of two vectors $x$ and $y$ is defined by:

$$x \cdot y = |x| |y| \cos(\theta)$$

where $\theta$ is the angle between the two vectors.

This provides an easy way to test **orthogonality**: if $x$ and $y$ are orthogonal (angle between vectors is $90°$), then $\cos(90°) = 0$, which means **the dot product of any two orthogonal vectors is 0**.

```python
# Test orthogonality
i = np.array([1, 0, 0])
j = np.array([0, 1, 0])
print(f"Dot product of i and j: {dot(i, j)}")  # Output: 0
```

### Application: Vector Similarity

The geometric definition is used to evaluate **vector similarity** in Natural Language Processing (NLP). By rearranging the equation:

$$\cos(\theta) = \frac{x \cdot y}{|x| |y|}$$

We can measure the cosine similarity between vectors:
- Value of 1: vectors point in the same direction (maximum similarity)
- Value of 0: vectors are orthogonal (no similarity)
- Value of -1: vectors point in opposite directions (maximum dissimilarity)

This is commonly used in NLP to measure similarity between word embeddings or document vectors.

## Conclusion

Vector operations are fundamental building blocks in linear algebra and machine learning. Key takeaways:

1. **Scalar multiplication** scales vectors while preserving (or reversing) direction
2. **Vector addition** follows the parallelogram law
3. **Dot product** has both algebraic and geometric interpretations
4. **Vectorized operations** are significantly faster than loops for numerical computations
5. **Vector similarity** using cosine of the angle is a practical application in NLP

Understanding these operations and their efficient implementation is essential for working with machine learning algorithms and large-scale data processing.
