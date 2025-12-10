+++
title = "Linear Algebra 201"
date = "2025-12-17"
type = "post"
draft = false
math = true
tags = ["linear-algebra", "numpy", "python", "eigenvalues", "eigenvectors"]
categories = ["posts"]
description = "Understanding eigenvalues and eigenvectors through visualization and interpretation"
+++

Eigenvalues and eigenvectors are among the most important concepts in linear algebra, with applications spanning from machine learning to quantum mechanics. In this post, we'll explore what they are, how to find them, and develop intuition about their geometric meaning.

## Definition of Eigenvalues and Eigenvectors

Consider a linear transformation defined by matrix $A=\begin{bmatrix}2 & 3 \\ 2 & 1 \end{bmatrix}$. When we apply this transformation to the standard basis vectors $e_1=\begin{bmatrix}1 \\ 0\end{bmatrix}$ and $e_2=\begin{bmatrix}0 \\ 1\end{bmatrix}$, both vectors change in length **and** direction.

```python
import numpy as np

A = np.array([[2, 3], [2, 1]])
e1 = np.array([1, 0])  
e2 = np.array([0, 1])
```

But what if we could find special vectors that only change in **length**, not direction? These special vectors satisfy:

$$Av=\lambda v$$

where:
- $v$ is called an **eigenvector** 
- $\lambda$ (lambda) is called an **eigenvalue** - the scaling factor

This equation says: "When we transform eigenvector $v$ by matrix $A$, we get the same vector back, just scaled by $\lambda$."

### An Important Property

If $v$ is an eigenvector, then **any scalar multiple** of $v$ is also an eigenvector with the same eigenvalue:

$$A(kv)=k(Av)=k \lambda v = \lambda (kv)$$

This means for each eigenvalue, there are infinitely many valid eigenvectors—all pointing along the same line, just with different lengths. In practice, we typically choose the eigenvector with norm (length) equal to 1.

## Finding Eigenvalues and Eigenvectors with Python

NumPy provides `np.linalg.eig()` to find eigenvalues and eigenvectors. It returns a tuple where:
- The first element contains the eigenvalues
- The second element contains the corresponding eigenvectors (one per column, normalized to length 1)

```python
A_eig = np.linalg.eig(A)

print(f"Matrix A:\n{A} \n\nEigenvalues of matrix A:\n{A_eig[0]}\n\nEigenvectors of matrix A:\n{A_eig[1]}")
```

Output:
```
Matrix A:
[[2 3]
 [2 1]] 

Eigenvalues of matrix A:
[ 4. -1.]

Eigenvectors of matrix A:
[[ 0.83205029 -0.70710678]
 [ 0.5547002   0.70710678]]
```

To extract individual eigenvectors:
- First eigenvector: `A_eig[1][:,0]`
- Second eigenvector: `A_eig[1][:,1]`

When visualized, we can see that:
- The first eigenvector $v_1$ is stretched by a factor of **4** (eigenvalue = 4)
- The second eigenvector $v_2$ is flipped and stays the same length (eigenvalue = -1)

Both vectors remain parallel to their original directions—the defining characteristic of eigenvectors!

## Standard Transformations and Their Eigenvalues

Let's explore eigenvalues and eigenvectors for some common transformations.

### Example 1: Reflection about y-axis

Reflection about the y-axis keeps the y-component fixed while reversing the x-direction:

$$A_{\text{reflection}}= \begin{bmatrix}-1 & 0\\0 & 1\end{bmatrix}$$

```python
A_reflection_yaxis = np.array([[-1, 0], [0, 1]])
A_reflection_yaxis_eig = np.linalg.eig(A_reflection_yaxis)

print(f"Matrix A:\n {A_reflection_yaxis} \n\nEigenvalues:\n {A_reflection_yaxis_eig[0]}",
      f"\n\nEigenvectors:\n {A_reflection_yaxis_eig[1]}")
```

Output:
```
Eigenvalues: [-1.  1.]
Eigenvectors:
[[1. 0.]
 [0. 1.]]
```

**Interpretation:**
- Eigenvalue **-1** with eigenvector $\begin{bmatrix}1 \\ 0\end{bmatrix}$ (horizontal): vectors along the x-axis are flipped
- Eigenvalue **1** with eigenvector $\begin{bmatrix}0 \\ 1\end{bmatrix}$ (vertical): vectors along the y-axis stay unchanged

This makes perfect sense for a reflection!

### Example 2: Shear in x-direction

A **shear transformation** slides horizontal layers past one another:

$$A_{\text{shear}}= \begin{bmatrix}1 & 0.5\\0 & 1\end{bmatrix}$$

```python
A_shear_x = np.array([[1, 0.5], [0, 1]])
A_shear_x_eig = np.linalg.eig(A_shear_x)

print(f"Matrix A_shear_x:\n {A_shear_x}\n\nEigenvalues:\n {A_shear_x_eig[0]}",
      f"\n\nEigenvectors:\n {A_shear_x_eig[1]}")
```

Output:
```
Eigenvalues: [1. 1.]
Eigenvectors:
[[1. 0.]
 [0. 1.]]
```

Interestingly, this transformation has **only one distinct eigenvalue** ($\lambda = 1$), but not all vectors are eigenvectors. The shear has limited eigenvector structure.

### A Note on Complex Eigenvalues: Rotation

Consider a 90-degree rotation:

$$A_{\text{rotation}}= \begin{bmatrix}0 & 1\\-1 & 0\end{bmatrix}$$

```python
A_rotation = np.array([[0, 1], [-1, 0]])
A_rotation_eig = np.linalg.eig(A_rotation)

print(f"Eigenvalues:\n {A_rotation_eig[0]}")
```

Output:
```
Eigenvalues: 
[0.+1.j 0.-1.j]
```

The eigenvalues are **complex numbers** (indicated by `j` for the imaginary part in Python)! 

**Interpretation:** There are no real eigenvectors because rotation changes every vector's direction. If you rotate the entire plane by 90 degrees, no vector points in the same direction as before. This makes intuitive sense!

### Example 3: Identity Matrix and Uniform Scaling

What about the identity matrix, which doesn't change any vectors?

```python
A_identity = np.array([[1, 0], [0, 1]])
A_identity_eig = np.linalg.eig(A_identity)

print(f"Matrix A_identity:\n {A_identity}\n\nEigenvalues:\n {A_identity_eig[0]}",
      f"\n\nEigenvectors:\n {A_identity_eig[1]}")
```

Output:
```
Eigenvalues: [1. 1.]
Eigenvectors:
[[1. 0.]
 [0. 1.]]
```

For the **identity matrix**, every vector is an eigenvector (with eigenvalue 1) since nothing changes! However, NumPy only returns two representative eigenvectors. This is a limitation of the software—understanding the mathematics behind your code is crucial.

The same applies to uniform scaling by factor 2:

```python
A_scaling = np.array([[2, 0], [0, 2]])
A_scaling_eig = np.linalg.eig(A_scaling)

print(f"Eigenvalues:\n {A_scaling_eig[0]}")
```

Output:
```
Eigenvalues: [2. 2.]
```

Every vector is doubled in length but keeps its direction, so every vector is an eigenvector with eigenvalue 2.

### Example 4: Projection onto x-axis

Projection onto the x-axis keeps only the x-component and zeros out the y-component:

$$A_{\text{projection}}=\begin{bmatrix}1 & 0 \\ 0 & 0 \end{bmatrix}$$

```python
A_projection = np.array([[1, 0], [0, 0]])
A_projection_eig = np.linalg.eig(A_projection)

print(f"Matrix A_projection:\n {A_projection}\n\nEigenvalues:\n {A_projection_eig[0]}",
      f"\n\nEigenvectors:\n {A_projection_eig[1]}")
```

Output:
```
Eigenvalues: [1. 0.]
Eigenvectors:
[[1. 0.]
 [0. 1.]]
```

**Interpretation:**
- Eigenvalue **1** with eigenvector $\begin{bmatrix}1 \\ 0\end{bmatrix}$: vectors along the x-axis remain unchanged
- Eigenvalue **0** with eigenvector $\begin{bmatrix}0 \\ 1\end{bmatrix}$: vectors along the y-axis are sent to zero

Yes, eigenvalues can be zero! An eigenvalue of 0 means that eigenvector is "collapsed" or mapped to the zero vector.

## Summary: Key Insights

Understanding eigenvalues and eigenvectors provides deep geometric insight into linear transformations:

1. **Definition**: Eigenvectors are special vectors that only change in length (not direction) when transformed. Eigenvalues tell us the scaling factor.

2. **Finding them**: Use `np.linalg.eig()` in Python, which returns eigenvalues and normalized eigenvectors

3. **How many?**: A 2×2 matrix can have:
   - **Two distinct real eigenvalues** (most common case)
   - **One repeated eigenvalue** (like shear or identity)
   - **Complex eigenvalues** (like rotation—no real eigenvectors)
   - **Zero eigenvalues** (indicating dimension collapse)

4. **Geometric intuition**:
   - Reflection: eigenvectors are axes of symmetry
   - Rotation: no real eigenvectors (changes all directions)
   - Scaling: all vectors are eigenvectors
   - Projection: zero eigenvalue indicates collapsed dimension

5. **Software limitations**: Tools like NumPy may not return all eigenvectors when infinitely many exist (like for the identity matrix)

Eigenvalues and eigenvectors are foundational for understanding: matrix diagonalization, principal component analysis (PCA), stability analysis of dynamical systems, quantum mechanics, and much more. Developing this geometric intuition will serve you well as you encounter them in applications!
