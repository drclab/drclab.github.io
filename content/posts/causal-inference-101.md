+++
title = "Causal Inference 101"
date = "2025-10-26T00:00:00Z"
type = "post"
draft = false
tags = ["causal inference", "econometrics", "books"]
categories = ["posts"]
description = "Primer on counterfactual thinking, identification strategies, and practical guardrails for causal inference."
+++

Causal inference asks whether a change we observe is actually caused by the treatment we care about. This primer distills the foundational ideas from Leonard Goff’s *Causal Inference in Econometrics* (2025) into an accessible set of concepts you can carry into your own research.

## Potential Outcomes Mindset

- Each unit carries two potential outcomes, such as earnings with college `Y(1)` and earnings without `Y(0)` (ch. 1). The observed outcome is whichever aligns with the realized treatment, while the unobserved counterfactual stays hidden.
- The stable unit treatment value assumption (SUTVA) says treatments have one version and no spillovers; break it and you contaminate your counterfactual comparison.
- The fundamental problem of causal inference is that we never observe both `Y(1)` and `Y(0)` for the same unit, so individual treatment effects remain unidentifiable.
- Population targets like the average treatment effect `ATE = E[Y(1) - Y(0)]`, the average effect on the treated (ATT), or on the untreated (ATU) give us estimands we can hope to identify.

## Randomization as the Benchmark

- A naive difference in means mixes causal effect with selection bias because treated and untreated units often differ in their counterfactual outcomes (ch. 1.4).
- Random assignment guarantees potential outcomes are statistically independent of treatment, collapsing the selection-bias term and making the mean difference an unbiased estimate of the ATE (ch. 1.5).
- Regression can recover the same estimand under random assignment and serves as a bridge to the observational methods below (ch. 1.6).

## Observational Playbook

When experiments are off the table, we lean on structural assumptions that emulate the randomized ideal.

### Selection on Observables

- Assume `(Y(0), Y(1)) ⟂ D | X`, i.e. once we condition on rich covariates `X`, treatment is as good as random (ch. 2.1).
- Estimation options include:
  - Matching on `X` to pair treated and control units with similar observed characteristics;
  - Inverse propensity score weighting to balance treatment groups;
  - Flexible regressions that control for `X`, watching out for functional-form misspecification.
- The payoff is direct identification of conditional average treatment effects `m(1, x) - m(0, x)`, but the curse of dimensionality and omitted variables are ever-present risks.

### Instrumental Variables

- Use an instrument `Z` that shifts treatment but is independent of unobservables affecting the outcome (ch. 4.1).
- Valid instruments satisfy relevance (`Cov(Z, D) ≠ 0`) and the exclusion restriction (they affect `Y` only through `D`); failure on either front breaks identification.
- With a binary instrument, the Wald ratio `β = [E[Y|Z=1] - E[Y|Z=0]] / [E[D|Z=1] - E[D|Z=0]]` captures the causal effect under homogeneous responses; more generally the estimand becomes a local average treatment effect for compliers (ch. 4.2).

### Regression Discontinuity

- Leverages a deterministic rule `D = 1(X ≥ c)` so units just above and below a cutoff mimic random assignment (ch. 5.1).
- Continuity of the untreated and treated conditional expectations at the threshold ensures the jump `lim_{x↓c} E[Y|X=x] - lim_{x↑c} E[Y|X=x]` equals the local treatment effect.
- Practical work focuses on bandwidth choice, checking for manipulation of the running variable, and interpreting the inherently local estimand.

### Difference-in-Differences

- Tracks treated and control groups before and after an intervention, relying on the parallel trends assumption that the untreated outcomes would have evolved similarly absent treatment (ch. 6.1).
- The canonical estimator computes the ATT as the change in the treated group minus the change in the control group.
- With multiple periods, use event-study or two-way fixed-effects models, but scrutinize treatment-timing heterogeneity and possible violations of parallel trends.

## A Workflow for Causal Questions

1. Clarify the unit, treatment, timing, and outcome you care about.
2. Write down the relevant potential outcomes and articulate which comparability assumption you are willing to defend.
3. Choose an identification strategy (experiment, observables, IV, RDD, DiD) that best fits the data-generating process.
4. Plan estimation mechanics along with diagnostics—balance checks, first-stage strength, bandwidth sensitivity, or placebo analyses as appropriate.
5. Stress-test assumptions with robustness checks and clearly state the population to which your estimate applies.

## Common Pitfalls and Guardrails

- Violations of SUTVA or interference can bias any design; consider clustered or network-aware approaches when spillovers loom.
- Weak instruments, manipulated running variables, or diverging trends signal that your quasi-experiment is failing its core assumption.
- Remember that many methods identify local or group-specific effects (LATE, `τ(c)`, ATT). Communicate external validity limits explicitly.

## Keep Learning

Leonard Goff's full notes dive deeper into law-of-iterated-expectation proofs, matching algorithms, weak-instrument diagnostics, and modern DiD estimators. Treat this post as the map, and the PDF as the full field guide.

## Reference

Goff, Leonard. *Causal Inference in Econometrics*. 2025.

```
