+++
title = "Linear Algebra 106"
date = "2025-12-16"
type = "post"
draft = false
math = true
tags = ["linear-algebra", "numpy", "python", "transformations", "matrices"]
categories = ["posts"]
description = "Understanding linear transformations and their matrix representations"
+++

Linear transformations are fundamental operations in linear algebra that preserve vector space structure. In this post, we'll explore what makes a transformation linear, how to represent them as matrices, and see practical applications in computer graphics.

## What is a Transformation?

A **transformation** is a function from one vector space to another that respects the underlying structure of each space. We denote a transformation with a symbol like $T$, and specify the input and output spaces. For example, $T: \mathbb{R}^2 \rightarrow \mathbb{R}^3$ means $T$ takes vectors from $\mathbb{R}^2$ and produces vectors in $\mathbb{R}^3$.

When transforming vector $v$ into vector $w$ by transformation $T$, we write $T(v)=w$ and read it as "*T of v equals w*" or "*vector w is an image of vector v with transformation T*".

Here's a simple example of a transformation $T: \mathbb{R}^2 \rightarrow \mathbb{R}^3$:

$$T\begin{pmatrix}
          \begin{bmatrix}
           v_1 \\\\           
           v_2
          \end{bmatrix}\end{pmatrix}=
          \begin{bmatrix}
           3v_1 \\\\
           0 \\\\
           -2v_2
          \end{bmatrix}
          $$

In Python:

```python
import numpy as np

def T(v):
    w = np.zeros((3,1))
    w[0,0] = 3*v[0,0]
    w[2,0] = -2*v[1,0]
    return w

v = np.array([[3], [5]])
w = T(v)

print("Original vector:\n", v, "\n\n Result of the transformation:\n", w)
```

## Linear Transformations

A transformation $T$ is **linear** if it satisfies two key properties for any scalar $k$ and any vectors $u$ and $v$:

1. **Scalar multiplication**: $T(kv)=kT(v)$
2. **Additivity**: $T(u+v)=T(u)+T(v)$

Let's verify our transformation $T$ is linear:

**Property 1:**
$$T (kv) =
          T \begin{pmatrix}\begin{bmatrix}
          kv_1 \\\\
          kv_2
          \end{bmatrix}\end{pmatrix} = 
          \begin{bmatrix}
           3kv_1 \\\\
           0 \\\\
           -2kv_2
          \end{bmatrix} =
          k\begin{bmatrix}
           3v_1 \\\\
           0 \\\\
           -2v_2
          \end{bmatrix} = 
          kT(v)$$

**Property 2:**
$$T (u+v) =
          T \begin{pmatrix}\begin{bmatrix}
          u_1 + v_1 \\\\
          u_2 + v_2
          \end{bmatrix}\end{pmatrix} = 
          \begin{bmatrix}
           3(u_1+v_1) \\\\
           0 \\\\
           -2(u_2+v_2)
          \end{bmatrix} = 
          \begin{bmatrix}
           3u_1 \\\\
           0 \\\\
           -2u_2
          \end{bmatrix} +
          \begin{bmatrix}
           3v_1 \\\\
           0 \\\\
           -2v_2
          \end{bmatrix} = 
          T(u)+T(v)$$

We can verify this with code:

```python
u = np.array([[1], [-2]])
v = np.array([[2], [4]])
k = 7

print("T(k*v):\n", T(k*v), "\n k*T(v):\n", k*T(v), "\n\n")
print("T(u+v):\n", T(u+v), "\n T(u)+T(v):\n", T(u)+T(v))
```

Common examples of linear transformations include rotations, reflections, and scaling (dilations).

## Matrix Representation of Linear Transformations

Here's the key insight: **every linear transformation can be represented as matrix multiplication**.

Let $L: \mathbb{R}^m \rightarrow \mathbb{R}^n$ be defined by matrix $A$, where $L(v)=Av$ (multiplication of matrix $A$ of size $n\times m$ by vector $v$ of size $m\times 1$, resulting in vector $w$ of size $n\times 1$).

For our transformation:

$$L\begin{pmatrix}
          \begin{bmatrix}
           v_1 \\\\           
           v_2
          \end{bmatrix}\end{pmatrix}=
          \begin{bmatrix}
           3v_1 \\\\
           0 \\\\
           -2v_2
          \end{bmatrix}=
          \begin{bmatrix}
           3 &amp; 0 \\\\
           0 &amp; 0 \\\\
           0 &amp; -2
          \end{bmatrix}
          \begin{bmatrix}
           v_1 \\\\
           v_2
          \end{bmatrix}
          $$

We can derive this by writing the matrix multiplication explicitly:

$$A\begin{bmatrix}
           v_1 \\\\           
           v_2
          \end{bmatrix}=
          \begin{bmatrix}
           a_{1,1} &amp; a_{1,2} \\\\
           a_{2,1} &amp; a_{2,2} \\\\
           a_{3,1} &amp; a_{3,2}
          \end{bmatrix}
          \begin{bmatrix}
           v_1 \\\\           
           v_2
          \end{bmatrix}=
          \begin{bmatrix}
           a_{1,1}v_1+a_{1,2}v_2 \\\\
           a_{2,1}v_1+a_{2,2}v_2 \\\\
           a_{3,1}v_1+a_{3,2}v_2 \\\\
          \end{bmatrix}=
          \begin{bmatrix}
           3v_1 \\\\
           0 \\\\
           -2v_2
          \end{bmatrix}$$

Matching coefficients, we find $A = \begin{bmatrix} 3 &amp; 0 \\\\ 0 &amp; 0 \\\\ 0 &amp; -2 \end{bmatrix}$.

In code:

```python
def L(v):
    A = np.array([[3,0], [0,0], [0,-2]])
    print("Transformation matrix:\n", A, "\n")
    w = A @ v
    return w

v = np.array([[3], [5]])
w = L(v)
print("Original vector:\n", v, "\n\n Result of the transformation:\n", w)
```

This is a **fundamental connection**: every linear transformation corresponds to a matrix, and every matrix multiplication represents a linear transformation.

## Standard Transformations in a Plane

For transformations $L: \mathbb{R}^2 \rightarrow \mathbb{R}^2$, we can understand their effect by seeing what they do to the **standard basis vectors** $e_1=\begin{bmatrix}1 \\\\ 0\end{bmatrix}$ and $e_2=\begin{bmatrix}0 \\\\ 1\end{bmatrix}$.

The transformation matrix $A$ can be written as:

$$A=\begin{bmatrix}L(e_1) &amp; L(e_2)\end{bmatrix}$$

This means the columns of $A$ are the images of the standard basis vectors under the transformation.

### Example 1: Horizontal Scaling (Dilation)

Horizontal scaling by a factor of 2 transforms $e_1=\begin{bmatrix}1 \\ 0\end{bmatrix}$ to $\begin{bmatrix}2 \\ 0\end{bmatrix}$ while leaving $e_2=\begin{bmatrix}0 \\ 1\end{bmatrix}$ unchanged.

```python
def T_hscaling(v):
    A = np.array([[2,0], [0,1]])
    w = A @ v
    return w

e1 = np.array([[1], [0]])
e2 = np.array([[0], [1]])

def transform_vectors(T, v1, v2):
    V = np.hstack((v1, v2))
    W = T(V)
    return W

result = transform_vectors(T_hscaling, e1, e2)
print("Original vectors:\n e1=\n", e1, "\n e2=\n", e2, 
      "\n\n Result of transformation:\n", result)
```

When visualized, you can see that any polygon defined by these vectors is stretched in the horizontal direction by a factor of 2.

### Example 2: Reflection about y-axis

Reflection about the y-axis (vertical axis) transforms $e_1$ to $-e_1$ while leaving $e_2$ unchanged:

```python
def T_reflection_yaxis(v):
    A = np.array([[-1,0], [0,1]])
    return A @ v

e1 = np.array([1, 0])
e2 = np.array([0, 1])
```

This transformation flips everything horizontally across the vertical axis.

## Application: Computer Graphics

Linear transformations are essential in computer graphics. Shapes are defined by their vertices (corners), and transformations like scaling, rotation, reflection, and shearing efficiently manipulate these vertices.

The key advantages:
- **Efficiency**: Matrix multiplication on GPUs is extremely fast
- **Composability**: Multiple transformations can be combined by multiplying their matrices

For example, the Barnsley fern fractal uses linear transformations of a single leaf shape to create all the subleafs:

Let's see a practical example using image transformations with OpenCV:

```python
import cv2
import matplotlib.pyplot as plt

# Load a leaf image
img = cv2.imread('images/leaf_original.png', 0)
plt.imshow(img)
```

We can apply transformations in sequence. First, rotate 90 degrees clockwise:

```python
image_rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
plt.imshow(image_rotated)
```

Then apply a shear transformation:

```python
rows, cols = image_rotated.shape
# Shear transformation matrix
M = np.float32([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])
image_rotated_sheared = cv2.warpPerspective(image_rotated, M, (int(cols), int(rows)))
plt.imshow(image_rotated_sheared)
```

### Order Matters!

What happens if we apply these transformations in the opposite order?

```python
image_sheared = cv2.warpPerspective(img, M, (int(cols), int(rows)))
image_sheared_rotated = cv2.rotate(image_sheared, cv2.ROTATE_90_CLOCKWISE)
plt.imshow(image_sheared_rotated)
```

The results are **different**! This is because applying transformations with matrices $A$ and $B$ in sequence means computing $B(Av)=(BA)v$, and **matrix multiplication is not commutative**: generally $BA \neq AB$.

Let's verify this with the transformation matrices:

```python
M_rotation_90_clockwise = np.array([[0, 1], [-1, 0]])
M_shear_x = np.array([[1, 0.5], [0, 1]])

print("90 degrees clockwise rotation matrix:\n", M_rotation_90_clockwise)
print("Matrix for shear along x-axis:\n", M_shear_x)

print("\nM_rotation @ M_shear:\n", M_rotation_90_clockwise @ M_shear_x)
print("\nM_shear @ M_rotation:\n", M_shear_x @ M_rotation_90_clockwise)
```

The two products are different, confirming that the order of transformations matters!

## Summary

Linear transformations are a powerful tool in mathematics and computing:

1. **Definition**: Transformations that preserve scalar multiplication and addition
2. **Matrix representation**: Every linear transformation can be represented as matrix multiplication
3. **Standard basis**: Transformation matrices can be constructed from the images of standard basis vectors
4. **Applications**: Essential in computer graphics for manipulating shapes efficiently
5. **Composition**: Multiple transformations combine through matrix multiplication, but **order matters**

Understanding linear transformations provides the foundation for more advanced topics in machine learning, computer vision, and computational geometry.
