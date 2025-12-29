+++
title = "Optimization 101: Metric Spaces and Normed Vector Spaces"
date = "2026-01-06"
tags = ["optimization", "metric spaces", "norms", "topology", "machine learning"]
categories = ["mathematics"]
series = ["Optimization Theory"]
type = "post"
draft = false
math = true
description = "An introduction to metric spaces and normed vector spaces—the foundational mathematical structures that underpin optimization theory and machine learning."
+++

Welcome to the first post in our series on **Optimization Theory**! This series is based on the excellent textbook *"Fundamentals of Optimization Theory with Applications to Machine Learning"* by Jean Gallier and Jocelyn Quaintance (2025).

Before we can understand optimization algorithms like gradient descent, support vector machines (SVM), or ADMM, we need to build a solid foundation in the mathematical structures that make optimization rigorous. In this post, we'll explore **metric spaces** and **normed vector spaces**—the essential building blocks of analysis and optimization theory.

## 1. Why Do We Need Metric Spaces?

In optimization, we constantly ask questions like:
- *How close are we to the optimal solution?*
- *Is our sequence of iterates converging?*
- *What does it mean for a set to be "open" or "closed"?*

To answer these questions rigorously, we need a way to measure **distance** between points. This is precisely what a metric space provides.

## 2. Metric Spaces

### Definition

A **metric space** is a set $E$ together with a function $d: E \times E \to \mathbb{R}_+$ (called a **metric** or **distance**) that assigns a non-negative real number $d(x, y)$ to any two points $x, y \in E$, satisfying three axioms:

> **📐 Metric Axioms**
>
> For all $x, y, z \in E$:
>
> 1. **Symmetry:** $d(x, y) = d(y, x)$
> 2. **Positivity:** $d(x, y) \geq 0$, and $d(x, y) = 0$ if and only if $x = y$
> 3. **Triangle Inequality:** $d(x, z) \leq d(x, y) + d(y, z)$

The triangle inequality is particularly important—it captures the intuitive notion that the direct path between two points is never longer than any detour.

![Triangle Inequality](/images/opt/triangle_inequality.png)

### Examples of Metrics

Let's look at some concrete examples:

**1. The Real Line:** For $E = \mathbb{R}$, the standard metric is the absolute value:
$$d(x, y) = |x - y|$$

**2. Euclidean Space:** For $E = \mathbb{R}^n$, the Euclidean distance is:
$$d_2(x, y) = \sqrt{(x_1 - y_1)^2 + \cdots + (x_n - y_n)^2}$$

**3. The Discrete Metric:** For any set $E$, we can define:
$$d(x, y) = \begin{cases} 0 & \text{if } x = y \\ 1 & \text{if } x \neq y \end{cases}$$

This metric is somewhat unusual—as we'll see, it leads to surprising geometric properties.

## 3. Open and Closed Balls

With a distance function in hand, we can define the fundamental "building blocks" of topology: balls.

> **🔵 Balls in a Metric Space**
>
> Given a metric space $(E, d)$, for any point $a \in E$ and radius $\rho > 0$:
>
> - **Closed ball:** $B(a, \rho) = \{x \in E : d(a, x) \leq \rho\}$
> - **Open ball:** $B_0(a, \rho) = \{x \in E : d(a, x) < \rho\}$
> - **Sphere:** $S(a, \rho) = \{x \in E : d(a, x) = \rho\}$

The key difference: the closed ball includes points exactly at distance $\rho$ from the center, while the open ball excludes them.

![Open vs Closed Balls](/images/opt/open_closed_balls.png)

### The Discrete Metric: A Warning About Intuition

The discrete metric provides a fascinating counterexample to our geometric intuition:

- If $\rho < 1$: The closed ball $B(a, \rho)$ contains **only** the center point $a$
- If $\rho \geq 1$: The closed ball $B(a, \rho)$ is the **entire space** $E$

![Discrete Metric Balls](/images/opt/discrete_metric.png)

This shows that our intuition from Euclidean geometry doesn't always apply in abstract metric spaces!

## 4. Normed Vector Spaces

While metric spaces are very general, in optimization we usually work with **vector spaces** where we can add vectors and multiply by scalars. When we combine this algebraic structure with a notion of "length," we get a normed vector space.

### Definition

A **norm** on a vector space $E$ over $\mathbb{R}$ (or $\mathbb{C}$) is a function $\|\cdot\|: E \to \mathbb{R}_+$ satisfying:

> **📏 Norm Axioms**
>
> For all $x, y \in E$ and scalars $\lambda$:
>
> 1. **Positivity:** $\|x\| \geq 0$, and $\|x\| = 0$ if and only if $x = 0$
> 2. **Homogeneity:** $\|\lambda x\| = |\lambda| \|x\|$
> 3. **Triangle Inequality:** $\|x + y\| \leq \|x\| + \|y\|$

**Every normed vector space is a metric space** via the metric:
$$d(x, y) = \|x - y\|$$

This metric has a special property: it's *translation-invariant*, meaning $d(x+u, y+u) = d(x, y)$ for any vector $u$.

## 5. The $\ell_p$ Norms

The most important family of norms on $\mathbb{R}^n$ are the **$\ell_p$ norms**:

$$\|x\|_p = \left(|x_1|^p + |x_2|^p + \cdots + |x_n|^p\right)^{1/p}$$

The three most commonly used are:

| Norm | Formula | Name |
|------|---------|------|
| $\ell_1$ | $\|x\|_1 = |x_1| + |x_2| + \cdots + |x_n|$ | Manhattan/Taxicab norm |
| $\ell_2$ | $\|x\|_2 = \sqrt{x_1^2 + x_2^2 + \cdots + x_n^2}$ | Euclidean norm |
| $\ell_\infty$ | $\|x\|_\infty = \max\{|x_1|, |x_2|, \ldots, |x_n|\}$ | Max/Supremum norm |

### Visualizing Unit Balls

The **unit ball** is the set of all points with norm at most 1. The shape of the unit ball reveals the geometry induced by each norm:

![Unit Balls in 2D](/images/opt/unit_balls_2d.png)

Notice how different norms create fundamentally different shapes:
- **$\ell_1$:** Diamond (rhombus)
- **$\ell_2$:** Circle (disc)
- **$\ell_\infty$:** Square

### Comparing the Norms

When we overlay these unit balls, we can see their containment relationships:

![Unit Balls Comparison](/images/opt/unit_balls_comparison.png)

In $\mathbb{R}^n$, we have the following inequalities between norms:
$$\|x\|_\infty \leq \|x\|_2 \leq \|x\|_1 \leq n\|x\|_\infty$$

### The Family of $\ell_p$ Norms

As $p$ varies from 0 to $\infty$, the unit ball transitions smoothly from star-shaped (for $p < 1$, which isn't actually a norm) to the $\ell_\infty$ square:

![Lp Norms Family](/images/opt/lp_norms_family.png)

This visualization shows why $\ell_1$ regularization (LASSO) promotes sparsity—the sharp corners of the $\ell_1$ ball intersect the constraint set at sparse solutions!

### Unit Balls in 3D

The same patterns extend to higher dimensions:

![Unit Balls in 3D](/images/opt/unit_balls_3d.png)

- **$\ell_1$:** Octahedron
- **$\ell_2$:** Sphere
- **$\ell_\infty$:** Cube

## 6. Open Sets and Topology

Using open balls, we can define one of the most fundamental concepts in analysis:

> **📖 Open Set**
>
> A subset $U \subseteq E$ is **open** if for every point $a \in U$, there exists some $\rho > 0$ such that the open ball $B_0(a, \rho) \subseteq U$.

Intuitively, a set is open if every point has some "breathing room" around it that stays inside the set.

![Open Set Illustration](/images/opt/open_set_illustration.png)

A set $F$ is **closed** if its complement $E \setminus F$ is open. Examples:
- Open intervals $(a, b)$ are open sets
- Closed intervals $[a, b]$ are closed sets
- The sets $\emptyset$ and $E$ are both open and closed (called **clopen**)

### Properties of Open Sets

The family of open sets $\mathcal{O}$ in a metric space satisfies:

1. **Finite intersections:** If $U_1, \ldots, U_n \in \mathcal{O}$, then $U_1 \cap \cdots \cap U_n \in \mathcal{O}$
2. **Arbitrary unions:** If $\{U_i\}_{i \in I} \subseteq \mathcal{O}$, then $\bigcup_{i \in I} U_i \in \mathcal{O}$
3. **Contains $\emptyset$ and $E$:** Both are in $\mathcal{O}$

> ⚠️ **Warning:** Infinite intersections of open sets may not be open! For example, in $\mathbb{R}$, each $U_n = (-1/n, 1/n)$ is open, but $\bigcap_n U_n = \{0\}$ is not open.

## 7. The Hausdorff Separation Property

Metric spaces enjoy a crucial property that ensures they behave "nicely":

> **🔀 Hausdorff Separation Axiom**
>
> A topological space is **Hausdorff** if for any two distinct points $a \neq b$, there exist disjoint open sets $U_a$ and $U_b$ such that $a \in U_a$ and $b \in U_b$.

![Hausdorff Separation](/images/opt/hausdorff_separation.png)

Every metric space is Hausdorff (use balls of radius $\rho = d(a,b)/3$). This property guarantees that:
- Limits of sequences are unique
- Single points are closed sets
- Compact sets are closed

## 8. Why This Matters for Optimization

These foundational concepts appear throughout optimization:

| Concept | Application in Optimization |
|---------|----------------------------|
| **Metric** | Measuring convergence of iterates |
| **Norm** | Regularization ($\ell_1$, $\ell_2$, elastic net) |
| **Open/Closed sets** | Constraint set properties |
| **Triangle inequality** | Proving convergence bounds |
| **Hausdorff property** | Uniqueness of solutions |

For example:
- **Gradient descent** converges when $\|x_{k+1} - x^*\| \to 0$ in some norm
- **LASSO** uses $\ell_1$ regularization to promote sparsity
- **SVM** finds the maximum margin using $\ell_2$ norm geometry
- **Constraint sets** in convex optimization are typically closed and bounded

## 9. Summary

In this post, we've established the foundational concepts:

1. **Metric spaces** provide a rigorous notion of distance
2. **Normed vector spaces** combine algebraic structure with length measurement
3. **$\ell_p$ norms** form a family of important norms with distinct geometric properties
4. **Open and closed sets** enable us to discuss convergence and continuity
5. The **Hausdorff property** ensures uniqueness of limits

In the next post, we'll build on these foundations to explore **topological spaces**, **continuity**, and **completeness**—key concepts for understanding when optimization problems have solutions.

---

## References

1. Gallier, J., & Quaintance, J. (2025). *Fundamentals of Optimization Theory with Applications to Machine Learning*. University of Pennsylvania.

2. Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.

3. Rudin, W. (1976). *Principles of Mathematical Analysis*. McGraw-Hill.
