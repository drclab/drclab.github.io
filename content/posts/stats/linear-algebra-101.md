+++
title = "Linear Algebra 101"
date = "2025-12-11"
type = "post"
draft = false
math = true
tags = ["linear-algebra", "numpy"]
categories = ["posts"]
description = "An introduction to linear algebra through NumPy arrays: creating, manipulating, and understanding vectors and matrices."
+++

Linear algebra is the mathematical foundation for machine learning, data science, and scientific computing. At its core lie **vectors** and **matrices**—multi-dimensional arrays of numbers that represent data, transformations, and relationships.

In Python, **NumPy** provides the essential tools for working with these structures efficiently. This post introduces linear algebra fundamentals through hands-on NumPy examples.

## Why NumPy for Linear Algebra?

Python lists can store sequences of values, but NumPy arrays offer critical advantages:

- **Performance**: NumPy operations are implemented in C, making them orders of magnitude faster than pure Python
- **Memory efficiency**: NumPy arrays store homogeneous data types compactly
- **Rich functionality**: Built-in support for mathematical operations, linear algebra routines, and broadcasting

```python
import numpy as np

# A simple 1-D array (vector)
vector = np.array([1, 2, 3])
print(vector)  # [1 2 3]
```

## Creating Arrays

### Basic construction

The fundamental building block is `np.array()`, which creates an n-dimensional array from a Python list:

```python
# 1-D array (vector)
a = np.array([1, 2, 3])

# 2-D array (matrix)
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
```

### Sequences and ranges

Generate evenly-spaced values with `np.arange()` and `np.linspace()`:

```python
# Start from 0, count 3 integers
b = np.arange(3)  # [0 1 2]

# From 1 to 20, step by 3
c = np.arange(1, 20, 3)  # [ 1  4  7 10 13 16 19]

# 5 evenly-spaced values from 0 to 100
d = np.linspace(0, 100, 5)  # [  0.  25.  50.  75. 100.]

# Force integer type
e = np.linspace(0, 100, 5, dtype=int)  # [  0  25  50  75 100]
```

### Special arrays

NumPy provides convenient functions for common initialization patterns:

```python
# Array of ones
ones = np.ones(3)  # [1. 1. 1.]

# Array of zeros
zeros = np.zeros(3)  # [0. 0. 0.]

# Random values between 0 and 1
random_arr = np.random.rand(3)  # [0.385 0.121 0.171]
```

## Multidimensional Arrays

In linear algebra, we often work beyond simple lists:
- **Vectors**: 1-D arrays representing position, direction, or features
- **Matrices**: 2-D arrays representing linear transformations or data tables
- **Tensors**: Higher-dimensional arrays for complex data structures

```python
# Create a 2-D array (matrix)
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

# Reshape a 1-D array into 2-D
flat = np.array([1, 2, 3, 4, 5, 6])
reshaped = np.reshape(flat, (2, 3))
# [[1 2 3]
#  [4 5 6]]
```

### Array properties

Every NumPy array has three key attributes:

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

arr.ndim   # 2 (number of dimensions)
arr.shape  # (2, 3) (rows, columns)
arr.size   # 6 (total number of elements)
```

## Array Operations

### Element-wise arithmetic

NumPy performs operations element-by-element, unlike Python lists:

```python
arr_1 = np.array([2, 4, 6])
arr_2 = np.array([1, 3, 5])

arr_1 + arr_2  # [ 3  7 11]
arr_1 - arr_2  # [1 1 1]
arr_1 * arr_2  # [ 2 12 30]
```

### Broadcasting

**Broadcasting** allows operations between arrays of different shapes. When you multiply a vector by a scalar, NumPy automatically applies the operation to each element:

```python
# Convert miles to kilometers (1 mile = 1.6 km)
miles = np.array([1, 2])
kilometers = miles * 1.6  # [1.6 3.2]
```

This is extremely powerful: instead of writing loops, broadcasting handles the repetition implicitly while maintaining performance.

## Indexing and Slicing

### Basic indexing

Access individual elements using zero-based indices:

```python
a = np.array([1, 2, 3, 4, 5])

a[0]   # 1 (first element)
a[2]   # 3 (third element)
a[-1]  # 5 (last element)
```

For multidimensional arrays, provide one index per dimension:

```python
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

matrix[2][1]  # 8 (row 2, column 1)
# Equivalently: matrix[2, 1]
```

### Slicing

Extract subarrays using the `[start:end:step]` notation:

```python
a = np.array([1, 2, 3, 4, 5])

a[1:4]   # [2 3 4] (elements 1 through 3)
a[:3]    # [1 2 3] (first 3 elements)
a[2:]    # [3 4 5] (from element 2 onward)
a[::2]   # [1 3 5] (every 2nd element)
```

For matrices, you can slice rows and columns:

```python
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# First two rows
matrix[0:2]
# [[1 2 3]
#  [4 5 6]]

# All rows, second column
matrix[:, 1]  # [2 5 8]
```

## Stacking Operations

Combine multiple arrays into larger structures:

### Vertical stacking

`np.vstack()` stacks arrays row-wise (adds rows):

```python
a1 = np.array([[1, 1],
               [2, 2]])

a2 = np.array([[3, 3],
               [4, 4]])

np.vstack((a1, a2))
# [[1 1]
#  [2 2]
#  [3 3]
#  [4 4]]
```

### Horizontal stacking

`np.hstack()` stacks arrays column-wise (adds columns):

```python
np.hstack((a1, a2))
# [[1 1 3 3]
#  [2 2 4 4]]
```

## Summary

NumPy arrays are the foundation for numerical computing in Python. Key takeaways:

1. **Arrays** store homogeneous data efficiently and support vectorized operations
2. **Broadcasting** eliminates explicit loops while maintaining readability
3. **Indexing and slicing** extract subsets of data elegantly
4. **Stacking** combines arrays to build complex data structures

These fundamentals enable everything from solving systems of linear equations to training deep neural networks. Master NumPy arrays, and you unlock the full power of scientific Python.
