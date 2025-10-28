+++
title = "Demystifying the Positivity Assumption"
date = "2025-10-14T00:00:00Z"
type = "post"
draft = false
math = true
tags = ["causal inference", "positivity"]
categories = ["causal inference","research"]
description = "Full-spectrum look at positivity (overlap): why every covariate pattern needs support, what breaks it, diagnostics, and design tactics to rescue causal estimates."
+++

Positivity—also called overlap or common support—guarantees that for every combination of covariates we condition on, treatment assignment can take each value with non-zero probability. While exchangeability lets us treat like-with-like, positivity ensures that the data actually contain both treated and untreated units for each covariate pattern. Without it, causal estimands simply fall outside the support of the observed data.

## What Positivity Demands

- **Unconditional positivity**: Randomized experiments deliver $0 < P(D_i = 1) < 1$ outright. Every unit could in principle receive each treatment arm.
- **Conditional positivity**: In observational designs, we need $0 < P(D_i = 1 \mid X_i = x) < 1$ for all $x$ in the support of $X$. After we condition on covariates to achieve exchangeability, those strata must still contain both treatment states.
- **Practical positivity**: Even if probabilities are not literally zero, near-deterministic assignment ($P(D_i = 1 \mid X_i = x) \approx 0 \text{ or } 1$) inflates variance, destabilizes weights, and pushes estimands toward extrapolation.

Positivity, exchangeability, and SUTVA form the identification tripod: violate any leg and potential-outcomes estimands wobble or collapse.

## Why It Matters in Practice

- **Estimator stability**: Weighting, matching, and regression rely on overlapping support. Extreme estimated propensity scores yield colossal weights, causing variance blow-ups and sensitivity to modeling choices.
- **Interpretability of the estimand**: When certain covariate strata only experience one treatment level, the average treatment effect for the whole population is undefined. Analysts end up estimating effects for subsets with sufficient overlap (e.g., the overlap population).
- **Policy relevance**: Interventions targeted to specific risk profiles need evidence drawn from units resembling that profile. No positivity means no empirical basis for claims about those groups.
- **Transportability and external validity**: Reweighting results to a new population hinges on shared support. Lack of overlap between the study sample and the target population undercuts extrapolation.

## Canonical Violation Patterns

- **Deterministic assignment rules**: Eligibility criteria, clinical guidelines, or legal mandates that assign everyone with $X_i = x$ to treatment (or control) wipe out overlap in that region.
- **Structural zeros in sampling**: Survey designs that stratify by treatment or only observe treated units above a threshold produce empty strata.
- **High-dimensional covariate spaces**: As we add more covariates, the combination of values can become sparse. Even if each marginal has overlap, the joint distribution may not.
- **Time-varying treatment regimes**: In longitudinal settings, certain histories make future treatment almost certain or impossible, a classic problem for marginal structural models.
- **Measurement artifacts**: Coarsening or discretizing covariates post hoc can disguise thin support, while exact matching on many discrete variables can produce empty cells.

## Diagnosing Positivity Risks

- **Propensity score visuals**: Plot histograms or kernel densities of estimated propensity scores by treatment status; non-overlapping tails signal trouble.
- **Covariate overlap plots**: Examine marginal distributions of key covariates across treatment groups before and after weighting or matching.
- **Weight diagnostics**: Inspect inverse probability weights for maximal values, coefficient of variation, or effective sample size. Heavy-tailed weights point to near-violations.
- **Cumulative distribution checks**: Compare empirical CDFs or quantile-quantile plots for covariates to spot sections where one group disappears.
- **Leverage and influence metrics**: In outcome regressions, high leverage points tied exclusively to one treatment arm betray regions of sparse support.
- **Rule-based audits**: Scrutinize inclusion/exclusion criteria, program rules, or institutional practices that may create deterministic pockets.

## Design Remedies When Positivity Falters

1. **Redefine the estimand**: Restrict analysis to the region of common support (e.g., overlap population, trimmed population) and state the revised target estimand explicitly.
2. **Trim or truncate weights**: Cap extreme weights or drop observations with propensity scores near 0 or 1 to stabilize estimators. Report sensitivity to trimming thresholds.
3. **Coarsen or collapse covariates thoughtfully**: Aggregate categories or bin continuous variables to boost cell counts, but ensure the revised specification still secures exchangeability.
4. **Use overlap-focused weighting schemes**: Employ overlap weights, entropy balancing, or matching weights that down-weight regions with weak support.
5. **Design-based fixes**: Alter sampling or recruitment to ensure representation across covariate strata, or randomize within strata that would otherwise be deterministic.
6. **Model-based extrapolation with caveats**: If overlap cannot be engineered, rely on structural models (e.g., parametric assumptions, Bayesian priors) and clearly label the resulting estimates as extrapolations.

## Reporting Checklist for Positivity Transparency

- Document the covariate set used for adjustment and describe any trimming, pruning, or weight truncation rules.
- Provide overlap diagnostics (propensity score plots, weight distributions, effective sample sizes) in the main text or appendix.
- State whether the estimand is population-wide (ATE) or limited to a subset (e.g., overlap or treated population) and justify the choice.
- Discuss potential policy or scientific implications of excluding non-overlap regions—who is left out and why.
- If extrapolation was unavoidable, lay out the modeling assumptions and perform sensitivity checks to alternative specifications.

## For Further Study

- **Hernán & Robins (2020)**, *Causal Inference: What If*, ch. 11 for formal positivity conditions in longitudinal settings.
- **Crump et al. (2009)** on optimal trimming to maximize precision under limited overlap.
- **Li, Morgan, & Zaslavsky (2018)** introducing overlap weights for improved balance and efficiency.
- **Petersen et al. (2012)** on diagnosing and remedying positivity problems in marginal structural models.
- **Fan, Imai, & Lee (2024)** for modern overlap diagnostics in high-dimensional covariate spaces.

Positivity is the gatekeeper that keeps causal estimands anchored in observed reality. Pair it with exchangeability, SUTVA, and relentless diagnostics to ensure the data can actually speak to the questions you pose.
