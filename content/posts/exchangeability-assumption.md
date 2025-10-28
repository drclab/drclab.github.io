+++
title = "Demystifying the Exchangeability Assumption"
date = "2025-10-13T00:00:00Z"
type = "post"
draft = false
math = true
tags = ["causal inference", "exchangeability"]
categories = ["causal inference","research"]
description = "Step-by-step tour of exchangeability, how it underwrites causal contrasts, where it fails, and how to rebuild identification."
+++

Exchangeability (also called ignorability or conditional independence) is the linchpin assumption that lets us translate observed contrasts into causal claims. Formally, we demand that the potential outcomes are independent of treatment assignment once we condition on the right covariates. Intuitively, after balancing on $X$, treated and untreated units are as if randomized.

## What Exchangeability Demands

- **Unconditional exchangeability**: In a randomized trial, $D_i \perp (Y_i(0), Y_i(1))$ outright. All differences in outcomes trace back to treatment rather than pre-existing traits.
- **Conditional exchangeability**: In observational studies we settle for $D_i \perp (Y_i(0), Y_i(1)) \mid X_i$. The covariate vector $X_i$ must capture every confounder that jointly affects treatment take-up and potential outcomes.
- **Positivity riding shotgun**: Exchangeability is useless without overlap; for every $X_i = x$, there must exist both treated and untreated units so the counterfactual can be learned.

When those clauses hold, estimators such as inverse probability weighting, matching, or doubly robust scores recover $E[Y_i(1) - Y_i(0)]$ from observed data.

## Why It Matters in Practice

- **Identification pillar**: All regression adjustment, propensity scores, and doubly robust estimators rely on exchangeability. Break it and estimators converge to biased, ill-defined contrasts.
- **Trial emulation**: Pragmatic trials with noncompliance still hinge on conditional exchangeability within principal strata. Without it, late- or never-takers upend comparisons.
- **Policy transportability**: When we extend results to new populations, we implicitly assume exchangeability after reweighting to the target covariate distribution.

## Canonical Violation Patterns

- **Unmeasured confounders**: Latent health status, motivation, or institutional quality that influence both treatment and outcomes.
- **Time-varying confounding**: In longitudinal settings, past treatment affects future confounders that in turn govern subsequent treatment (think ART adherence in HIV care).
- **Measurement error in $X$**: Mismeasured covariates fail to balance the true confounders, reopening backdoor paths.
- **Selection on post-treatment variables**: Conditioning on mediators or colliders (e.g., conditioning on hospitalization when studying treatment effects) destroys exchangeability.
- **Self-selection and choice-based sampling**: Survey designs that oversample treated units or outcomes skew the assignment mechanism.

## Diagnosing Exchangeability Risks

- **Directed acyclic graphs (DAGs)**: Map hypothesized relationships to see whether all backdoor paths are blocked by observed covariates.
- **Balance diagnostics**: After adjustment, examine standardized mean differences, empirical CDFs, or prognostic score balance for each covariate and their interactions.
- **Placebo and negative controls**: Test effects on outcomes that should not respond to treatment or use exposures known to be inert; significant effects warn of hidden confounding.
- **Sensitivity analyses**: Quantify how strong an unobserved confounder must be to overturn conclusions (e.g., Rosenbaum bounds, E-values, Oster's $\delta$).
- **Transparency about data provenance**: Track how covariates were measured, imputed, or transformed; gaps signal where exchangeability might crack.

## Design Remedies When Exchangeability Is Shaky

1. **Collect richer pretreatment data**: Expand baseline surveys, link administrative records, or incorporate biomarkers to capture latent confounders.
2. **Exploit quasi-experiments**: Use designs like regression discontinuity, difference-in-differences, synthetic controls, or natural experiments that reduce reliance on covariate adjustment.
3. **Instrumental variables**: Introduce exogenous variation in treatment via encouragements or lotteries when direct exchangeability is implausible.
4. **Longitudinal g-methods**: Apply marginal structural models, g-computation, or structural nested models to handle time-varying confounding.
5. **Partial identification**: Report bounds when only weak assumptions (e.g., monotone treatment response) are defensible instead of forcing point estimates.

## Reporting Checklist for Exchangeability Transparency

- Detail the covariates used to justify exchangeability, including timing, measurement units, and any preprocessing steps.
- Document diagnostics: balance tables, overlap plots, and sensitivity metrics belong in the appendix or supplementary materials.
- Explain why omitted variables are unlikely to confound—draw on domain expertise, institutional context, or prior evidence.
- Clarify whether the estimand targets the average treatment effect (ATE), ATT, or ATC, and how exchangeability is asserted for that estimand.
- Discuss robustness across adjustment strategies (matching, weighting, outcome regression) and flag any divergences.

## For Further Study

- **Hernán & Robins (2020)**, *Causal Inference: What If*, ch. 2-7 for formal definitions and longitudinal methods.
- **Rosenbaum & Rubin (1983)** on propensity score theory and balancing properties.
- **Imbens (2003)** on sensitivity analysis frameworks for unobserved confounding.
- **Brumback et al. (2004)** for marginal structural models with time-varying confounding.
- **Ding & VanderWeele (2016)** introducing E-values as interpretable sensitivity measures.

Exchangeability is never free; it is earned through design, measurement, and transparent reporting. Treat it as a working hypothesis that needs constant stress-testing rather than a background assumption granted by default.
