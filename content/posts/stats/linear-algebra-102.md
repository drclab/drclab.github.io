+++
title = "Linear Algebra 102"
date = "2025-12-13"
type = "post"
draft = true
math = true
tags = ["linear-algebra", "numpy", "python", "matrices"]
categories = ["posts"]
description = "Representing systems of linear equations as matrices and understanding solution types"
+++

Building on the fundamentals from Linear Algebra 101, we now explore how to represent and solve **systems of linear equations** using matrices—a cornerstone technique in computational mathematics and machine learning.

## Systems of Linear Equations

A **system of linear equations** is a collection of linear equations involving the same variables. For example:

$$
\begin{cases} 
-x_1+3x_2=7, \\\\ 3x_1+2x_2=1
\end{cases}
$$

To **solve** this system means finding values for $x_1$ and $x_2$ that satisfy both equations simultaneously.

A system is **singular** if it doesn't have a unique solution (either no solutions or infinitely many). Otherwise, it's **non-singular**.

## Matrix Representation

We can represent this system as a matrix. The system above becomes:

$$
\begin{bmatrix}
-1 & 3 & 7 \\\\
3 & 2 & 1
\end{bmatrix}
$$

Each row represents an equation. The first two columns contain the coefficients of $x_1$ and $x_2$, while the third column contains the constants (right-hand side values).

We can separate this into:
- **Coefficient matrix** $A$: 
$$
\begin{bmatrix}
-1 & 3\\\\
3 & 2
\end{bmatrix}
$$

- **Output vector** $b$: 
$$
\begin{bmatrix}
7 \\\\
1
\end{bmatrix}
$$

In NumPy:

```python
import numpy as np

A = np.array([
    [-1, 3],
    [3, 2]
], dtype=np.dtype(float))

b = np.array([7, 1], dtype=np.dtype(float))

print("Matrix A:")
print(A)
print("\nArray b:")
print(b)
```

Output:
```
Matrix A:
[[-1.  3.]
 [ 3.  2.]]

Array b:
[7. 1.]
```

### Solving the System

NumPy provides `np.linalg.solve(A, b)` to solve systems of linear equations efficiently:

```python
x = np.linalg.solve(A, b)
print(f"Solution: {x}")  # Solution: [-1.  2.]
```

The solution is $x_1 = -1$ and $x_2 = 2$. You can verify this by substituting these values back into the original equations.

## The Determinant

For a **square matrix** (same number of rows and columns), we can calculate its **determinant**—a scalar value that characterizes important matrix properties.

A key insight: **A linear system has a unique solution if and only if the determinant of its coefficient matrix is non-zero.**

```python
d = np.linalg.det(A)
print(f"Determinant of matrix A: {d:.2f}")  # -11.00
```

Since the determinant is non-zero ($-11$), our system has exactly one solution.

## Visualizing Solutions

For 2×2 systems, each equation represents a line in the plane. The solution is where these lines intersect.

To visualize, we combine $A$ and $b$ into an augmented matrix:

```python
A_system = np.hstack((A, b.reshape((2, 1))))
print(A_system)
```

Output:
```
[[-1.  3.  7.]
 [ 3.  2.  1.]]
```

When plotted, these two lines intersect at $(-1, 2)$—our solution!

## Systems with No Solutions

Consider a slightly different system:

$$
\begin{cases} 
-x_1+3x_2=7, \\\\ 3x_1-9x_2=1
\end{cases}
$$

```python
A_2 = np.array([
    [-1, 3],
    [3, -9]
], dtype=np.dtype(float))

b_2 = np.array([7, 1], dtype=np.dtype(float))

d_2 = np.linalg.det(A_2)
print(f"Determinant of matrix A_2: {d_2:.2f}")  # 0.00
```

The determinant is **zero**, so this system cannot have a unique solution. Attempting to solve it:

```python
try:
    x_2 = np.linalg.solve(A_2, b_2)
except np.linalg.LinAlgError as err:
    print(err)  # Singular matrix
```

When we plot these equations, we see **parallel lines**—they never intersect, so there's no solution. This is an **inconsistent** system.

## Systems with Infinite Solutions

By changing the constants, we can make the system consistent:

$$
\begin{cases} 
-x_1+3x_2=7, \\\\ 3x_1-9x_2=-21
\end{cases}
$$

```python
b_3 = np.array([7, -21], dtype=np.dtype(float))

A_3_system = np.hstack((A_2, b_3.reshape((2, 1))))
print(A_3_system)
```

Output:
```
[[ -1.   3.   7.]
 [  3.  -9. -21.]]
```

Notice that the second equation is just -3 times the first equation. This simplifies to:

$$
\begin{cases} 
-x_1+3x_2=7, \\\\ 0=0
\end{cases}
$$

The second equation is always true! So we have:

$$
x_1=3x_2-7
$$

where $x_2$ can be **any real number**. When plotted, both equations represent the **same line**—infinitely many points satisfy both equations.

## Summary

Understanding systems of linear equations is crucial for linear algebra:

1. **Matrix representation** makes systems easier to work with computationally
2. **Determinants** tell us about solution uniqueness:
   - Non-zero: unique solution (intersecting lines)
   - Zero: no unique solution (parallel or identical lines)
3. **Three types of solutions**:
   - Unique (one intersection point)
   - None (parallel lines, inconsistent system)
   - Infinite (same line, consistent system)

These concepts form the foundation for solving real-world problems in machine learning, computer graphics, physics simulations, and optimization.
