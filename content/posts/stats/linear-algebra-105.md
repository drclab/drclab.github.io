+++
title = "Linear Algebra 105: Matrix Multiplication"
date = "2025-12-15"
type = "post"
draft = false
math = true
tags = ["linear-algebra", "numpy", "python", "matrices", "matrix-multiplication"]
categories = ["posts"]
description = "Understanding matrix multiplication, NumPy operations, and broadcasting"
+++

# Matrix Multiplication

In this post, we'll explore matrix multiplication using NumPy and understand how it's used in Machine Learning applications. We'll cover the mathematical definition, practical implementation, and important considerations about matrix dimensions and NumPy's broadcasting behavior.

## Definition of Matrix Multiplication

If $A$ is an $m \times n$ matrix and $B$ is an $n \times p$ matrix, the matrix product $C = AB$ (denoted without multiplication signs or dots) is defined to be the $m \times p$ matrix such that:

$$c_{ij} = a_{i1}b_{1j} + a_{i2}b_{2j} + \ldots + a_{in}b_{nj} = \sum_{k=1}^{n} a_{ik}b_{kj}$$

where $a_{ik}$ are the elements of matrix $A$, $b_{kj}$ are the elements of matrix $B$, and $i = 1, \ldots, m$, $k=1, \ldots, n$, $j = 1, \ldots, p$. 

**In other words**: $c_{ij}$ is the **dot product** of the $i$-th row of $A$ and the $j$-th column of $B$.

## Matrix Multiplication in Python

Like with the dot product, there are several ways to perform matrix multiplication in Python. As we learned in Linear Algebra 104, calculations are more efficient in vectorized form. Let's explore the most commonly used NumPy functions.

First, define two matrices:

```python
import numpy as np

A = np.array([[4, 9, 9], [9, 1, 6], [9, 2, 3]])
print("Matrix A (3 by 3):\n", A)

B = np.array([[2, 2], [5, 7], [4, 4]])
print("Matrix B (3 by 2):\n", B)
```

Output:
```
Matrix A (3 by 3):
 [[4 9 9]
  [9 1 6]
  [9 2 3]]

Matrix B (3 by 2):
 [[2 2]
  [5 7]
  [4 4]]
```

### Using `np.matmul()`

You can multiply matrices $A$ and $B$ using the NumPy function `np.matmul()`:

```python
np.matmul(A, B)
```

This will output a $3 \times 2$ matrix:
```
array([[89, 107],
       [47, 49],
       [40, 44]])
```

### Using the `@` Operator

Python's `@` operator also works for matrix multiplication, giving the same result:

```python
A @ B
```

Output:
```
array([[89, 107],
       [47, 49],
       [40, 44]])
```

## Matrix Convention and Broadcasting

### Dimension Requirements

Mathematically, matrix multiplication is **only defined if the number of columns of matrix $A$ equals the number of rows of matrix $B$**. Otherwise, the dot products between rows and columns won't be defined.

In our example above, changing the order to $BA$ will **not work** because the dimension rule doesn't hold:

```python
try:
    np.matmul(B, A)
except ValueError as err:
    print(err)
```

Output:
```
matmul: Input operand 1 has a mismatch in its core dimension 0, 
with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 3 is different from 2)
```

The same error occurs with the `@` operator:

```python
try:
    B @ A
except ValueError as err:
    print(err)
```

> **Important**: When using matrix multiplication, you must carefully consider the dimensions. The number of columns in the first matrix must match the number of rows in the second matrix. This is crucial for understanding how Neural Networks work!

### NumPy's Vector Shortcut

For vector multiplication, NumPy provides a convenient shortcut. You can define two vectors $x$ and $y$ of the same size (which can be understood as two $3 \times 1$ matrices):

```python
x = np.array([1, -2, -5])
y = np.array([4, 3, -1])

print("Shape of vector x:", x.shape)
print("Number of dimensions of vector x:", x.ndim)
print("Shape of vector x, reshaped to a matrix:", x.reshape((3, 1)).shape)
print("Number of dimensions of vector x, reshaped to a matrix:", x.reshape((3, 1)).ndim)
```

Output:
```
Shape of vector x: (3,)
Number of dimensions of vector x: 1
Shape of vector x, reshaped to a matrix: (3, 1)
Number of dimensions of vector x, reshaped to a matrix: 2
```

### Automatic Transposition

Following strict matrix convention, multiplication of matrices $3 \times 1$ and $3 \times 1$ is **not defined**. You would expect an error, but observe what happens:

```python
np.matmul(x, y)
```

Output:
```
3
```

No error! The result is actually the dot product $x \cdot y$. **Vector $x$ was automatically transposed** into a $1 \times 3$ vector, and the matrix multiplication $x^T y$ was calculated.

While this is very convenient, you need to be aware of this NumPy behavior to avoid using it incorrectly. The following **will** return an error:

```python
try:
    np.matmul(x.reshape((3, 1)), y.reshape((3, 1)))
except ValueError as err:
    print(err)
```

Output:
```
matmul: Input operand 1 has a mismatch in its core dimension 0, 
with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 3 is different from 1)
```

### Does `np.dot()` Work for Matrix Multiplication?

You might wonder: does `np.dot()` also work for matrix multiplication?

```python
np.dot(A, B)
```

Output:
```
array([[89, 107],
       [47, 49],
       [40, 44]])
```

Yes, it works! What happens is called **broadcasting** in Python: NumPy broadcasts the dot product operation to all rows and all columns, producing the resulting product matrix.

### Broadcasting with Scalars

Broadcasting also works in other cases. For example:

```python
A - 2
```

Output:
```
array([[ 2,  7,  7],
       [ 7, -1,  4],
       [ 7,  0,  1]])
```

Mathematically, subtraction of a $3 \times 3$ matrix $A$ and a scalar is not defined, but Python **broadcasts the scalar**, creating a $3 \times 3$ array and performing subtraction element by element.

## Practical Applications

A practical example of matrix multiplication can be seen in **linear regression models**. In linear regression:

- Input features are typically represented as a matrix $X$ (where each row is a sample, each column is a feature)
- Model parameters (weights) are represented as a vector $w$
- The prediction is computed as $\hat{y} = Xw$

Matrix multiplication allows us to compute predictions for all samples simultaneously in a vectorized, efficient manner.

Similarly, in **Neural Networks**:
- Each layer performs matrix multiplication between inputs and weights
- Understanding dimension compatibility is crucial for designing network architectures
- Broadcasting enables efficient batch processing

## Summary

Key takeaways about matrix multiplication:

1. **Definition**: Element $c_{ij}$ is the dot product of row $i$ of $A$ and column $j$ of $B$
2. **Dimension rule**: Number of columns in first matrix must equal number of rows in second matrix
3. **NumPy functions**: Both `np.matmul()` and `@` operator work for matrix multiplication
4. **Vector shortcut**: NumPy automatically transposes vectors for convenient multiplication
5. **Broadcasting**: Python broadcasts operations for convenience, but be careful with dimensions
6. **Applications**: Matrix multiplication is fundamental to linear regression and neural networks

Understanding these concepts is essential for working with Machine Learning algorithms, where matrix operations form the computational backbone of most models.

> **Pro tip**: Always be mindful of matrix dimensions when building ML models. Many bugs in neural network code stem from dimension mismatches in matrix operations!
