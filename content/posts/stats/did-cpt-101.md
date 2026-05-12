+++
title = "DiD_CPT 101: Conditional Parallel Trends in a 2x2 DiD"
date = "2026-05-14"
tags = ["difference-in-differences", "causal-inference", "panel-data", "treatment-effects"]
categories = ["posts", "stats"]
series = ["DiD_CPT"]
type = "post"
draft = false
math = true
description = "A clean 2x2 DiD setup that makes the conditional parallel trends assumption explicit."
+++

The canonical 2x2 difference-in-differences (DiD) setup is the minimal panel design: a binary treatment $D$, two time periods $t \in \{0,1\}$, and potential outcomes $Y_t(d)$ for treatment status $d \in \{0,1\}$.

The estimand of interest is the average treatment effect on the treated (ATT):

$$
\text{ATT} := \mathbb{E}[Y_1(1) - Y_1(0) \mid D = 1].
$$

Two standard design facts are baked in:

- No unit is treated in the pre-period $t = 0$.
- Some units become treated in the post-period $t = 1$.

## 1. Conditional parallel trends

The identifying assumption is that untreated outcomes for treated units would have trended like untreated outcomes for control units, after conditioning on time-invariant covariates $X$:

$$
\mathbb{E}[Y_1(0) - Y_0(0) \mid X, D = 1]
= \mathbb{E}[Y_1(0) - Y_0(0) \mid X, D = 0].
$$

Read this as: the counterfactual trend of the treated group (had they not been treated) matches the factual trend of the untreated group, once you control for $X$.

## 2. Why this identifies the ATT

The DiD logic is simple:

- The treated group shows an observed change from $t=0$ to $t=1$.
- The control group gives the counterfactual change you would expect without treatment.
- The conditional parallel trends equation lets you swap the unobserved treated trend for the observed control trend, conditional on $X$.

That substitution turns the ATT into a function of observed outcomes, which is why the 2x2 DiD is the canonical identification story.

## 3. Takeaway

If you can credibly defend the conditional parallel trends assumption and your $X$ set is fixed over time, the 2x2 DiD design delivers a clean estimate of the ATT.
