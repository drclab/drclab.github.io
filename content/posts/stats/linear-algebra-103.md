+++
title = "Linear Algebra 103"
date = "2025-12-13"
type = "post"
draft = false
math = true
tags = ["linear-algebra", "numpy", "python", "matrices", "determinant"]
categories = ["posts"]
description = "Introduction to numpy.linalg and solving systems with 3 variables"
+++

Building on our understanding of 2-variable systems from Linear Algebra 102, we now extend to **3-variable systems** and introduce the powerful `numpy.linalg` sub-library—your toolkit for computational linear algebra.

## System of Linear Equations with 3 Variables

Consider this **system of linear equations** with three equations and three unknowns:

$$
\begin{cases} 
4x_1-3x_2+x_3=-10, \\\\ 2x_1+x_2+3x_3=0, \\\\ -x_1+2x_2-5x_3=17
\end{cases}
$$

**To solve** this system means to find values of $x_1$, $x_2$, $x_3$ that satisfy all three equations simultaneously.

## Solving with NumPy

First, let's import NumPy and represent our system as a matrix $A$ and vector $b$:

```python
import numpy as np

A = np.array([
        [4, -3, 1],
        [2, 1, 3],
        [-1, 2, -5]
    ], dtype=np.dtype(float))

b = np.array([-10, 0, 17], dtype=np.dtype(float))

print("Matrix A:")
print(A)
print("\nArray b:")
print(b)
```

Output:
```
Matrix A:
[[ 4. -3.  1.]
 [ 2.  1.  3.]
 [-1.  2. -5.]]

Array b:
[-10.   0.  17.]
```

Let's verify the dimensions:

```python
print(f"Shape of A: {np.shape(A)}")
print(f"Shape of b: {np.shape(b)}")
```

Output:
```
Shape of A: (3, 3)
Shape of b: (3,)
```

Now we can solve the system using `np.linalg.solve(A, b)`:

```python
x = np.linalg.solve(A, b)

print(f"Solution: {x}")
```

Output:
```
Solution: [ 1.  4. -2.]
```

The solution is $x_1 = 1$, $x_2 = 4$, $x_3 = -2$. You can verify by substituting these values back into the original equations!

## Evaluating the Determinant

Matrix $A$ is a **square matrix** (same number of rows and columns). For square matrices, we can calculate the **determinant**—a scalar that characterizes important matrix properties.

**Key insight**: A linear system has one unique solution **if and only if** the matrix has a **non-zero determinant**.

```python
d = np.linalg.det(A)

print(f"Determinant of matrix A: {d:.2f}")
```

Output:
```
Determinant of matrix A: -60.00
```

Since the determinant is non-zero ($-60$), our system has exactly one solution—as we found!

## What Happens with No Unique Solution?

Let's explore what happens when a system has no unique solution (either no solution or infinitely many).

Consider this system:

$$
\begin{cases} 
x_1+x_2+x_3=2, \\\\ x_2-3x_3=1, \\\\ 2x_1+x_2+5x_3=0
\end{cases}
$$

```python
A_2= np.array([
        [1, 1, 1],
        [0, 1, -3],
        [2, 1, 5]
    ], dtype=np.dtype(float))

b_2 = np.array([2, 1, 0], dtype=np.dtype(float))

print(np.linalg.solve(A_2, b_2))
```

This raises an error:
```
LinAlgError: Singular matrix
```

NumPy throws a `LinAlgError` because the matrix is **singular**. Let's check the determinant:

```python
d_2 = np.linalg.det(A_2)

print(f"Determinant of matrix A_2: {d_2:.2f}")
```

Output:
```
Determinant of matrix A_2: 0.00
```

The determinant is **zero**! This confirms the matrix is singular—it cannot have a unique solution.

## Understanding Singular Matrices

When a matrix is singular (determinant = 0), the system either:
- Has **no solution** (inconsistent system—like parallel lines that never meet)
- Has **infinitely many solutions** (like identical planes in 3D space)

The `np.linalg` sub-library contains many powerful linear algebra functions. As you learn more theory, these functions will become clearer and more useful.

> **Important**: Always check for singular matrices! `np.linalg.solve()` will error if there are no or infinitely many solutions. When using it in production code, wrap it in error handling to prevent crashes.

## Summary

In this post, we've explored:

1. **Extension to 3 variables**: The same matrix concepts from 2D extend naturally to higher dimensions
2. **The `numpy.linalg` sub-library**: Your computational toolkit for linear algebra
3. **Determinants indicate solution type**:
   - Non-zero determinant → unique solution
   - Zero determinant → no unique solution (singular matrix)
4. **Error handling**: `np.linalg.solve()` throws errors for singular systems

While using `np.linalg.solve()` is convenient, it gives little insight into what happens "under the hood." **Next, we'll explore Gaussian Elimination**—a fundamental method to solve linear systems that reveals the mechanics behind matrix solutions.

These tools form the computational backbone of machine learning, computer graphics, optimization, and scientific computing.
