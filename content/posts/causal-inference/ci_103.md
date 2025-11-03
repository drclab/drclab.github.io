+++
title = "Demystifying the Stable Unit Treatment Value Assumption"
date = "2025-10-15T00:00:00Z"
type = "post"
draft = false
math = true
tags = ["causal inference", "sutva"]
categories = ["posts"]
description = "Granular walkthrough of SUTVA's components, typical failure modes, diagnostics, and design fixes for credible causal claims."
+++

The Stable Unit Treatment Value Assumption (SUTVA) underpins the potential outcomes framework. It promises that each unit's observed outcome reflects a single, well-defined potential outcome that depends only on the unit's own treatment status. When SUTVA breaks, causal estimates target a moving object; versions of the treatment change the estimand, and spillovers smear counterfactual comparisons.

## What SUTVA Actually Says

- **No multiple versions**: If unit $i$ receives treatment $d$, the realized potential outcome is $Y_i(d)$ regardless of how that treatment was delivered. Vaccine $A$ administered in the left arm must be equivalent to the same vaccine in the right arm.
- **No interference across units**: Potential outcomes depend only on the unit's own treatment assignment: $Y_i(d) = Y_i(d, D_{-i})$ for all configurations $D_{-i}$ of other units' treatments.

Combined, these clauses ensure the mapping $D_i \to Y_i(D_i)$ is well-defined. Formally, if you attempt to write $Y_i(d, g(D_{-i}))$ where $g$ summarizes others' assignments, SUTVA insists $g$ must be irrelevant.

## Why It Matters in Practice

- **Randomized experiments**: Randomization guarantees independence between treatment and potential outcomes only if each potential outcome is uniquely tied to a treatment level. Noncompliance or protocol deviations often signal multiple versions lurking in the background.
- **Observational identification**: Selection-on-observables or IV strategies inherit the same fragility. If interference is present, conditioning or instrumenting on $D_i$ alone still leaves contamination from $D_{-i}$.
- **External validity**: When the estimand aggregates over heterogeneous versions of a treatment (e.g., different dosage schedules), reporting an "average effect" masks which version drives the result and complicates replication.

## Canonical Violation Patterns

- **Network spillovers**: Social exposure, peer effects, or infectious diseases let other units' treatments shift $Y_i$. Cluster-randomized trials often arise explicitly to handle this.
- **Implementation heterogeneity**: Education programs delivered by multiple teachers or clinics may vary subtly. Without tracking those variants, $Y_i(1)$ becomes a mixture of implementation-specific effects.
- **Supply constraints and interference through resources**: When treated units consume scarce resources (teachers' time, ICU beds), untreated units' outcomes change indirectly.
- **Measurement mismatch**: If survey instruments or timing differ across arms, you effectively measure different versions of the outcome, violating the spirit of SUTVA.

## Diagnosing SUTVA Risks

- **Map the causal graph** to identify paths where other units' treatments can reach unit $i$. Group membership, shared markets, or shared staff are danger zones.
- **Descriptive spillover checks**: Plot outcomes for untreated units against the share of treated peers (neighbors, classmates, co-workers). Monotone relationships signal potential interference.
- **Implementation logs**: Track compliance, delivery protocols, and exposure intensity. Variation beyond a tolerable threshold means you are pooling multiple versions.
- **Qualitative field work**: Interviews and process tracing often uncover informal spillovers (information sharing, material lending) invisible in administrative data.

## Design Remedies When SUTVA Fails

1. **Redefine the unit or treatment**: Aggregate to classrooms, villages, or clusters where interference is contained, or split treatments into explicit variants (e.g., "SMS reminder + phone call").
2. **Explicit exposure mapping**: Model $Y_i(d, g(D_{-i}))$ with exposure metrics such as the number of treated neighbors or total treated dosage in a network (see Aronow & Samii, 2017).
3. **Two-stage randomization**: Randomize clusters to saturation levels, then randomize individuals within clusters; estimators separate direct, spillover, and overall effects.
4. **Instrument for implementation details**: Use encouragements or randomized rollout timing to uncover effects of specific treatment versions.
5. **Report partial interference models**: Assume units interfere only within predefined groups. Variance estimators and hypothesis tests must reflect the cluster structure.

## Reporting Checklist for SUTVA Transparency

- Describe how the treatment is delivered, who delivers it, and what fidelity monitoring occurred.
- Document potential channels for spillovers and whether the study design blocked or measured them.
- Provide balance tables or diagnostics for exposure variables (e.g., average treated neighbors) if interference is plausible.
- State clearly whether estimated effects should be interpreted as direct, indirect, or total effects in the presence of interference.
- Discuss how alternative treatment versions or rollout intensities might alter the estimand.

## For Further Study

- **Imbens & Rubin (2015)**, *Causal Inference for Statistics, Social, and Biomedical Sciences*, ch. 1-2 for the formal SUTVA statement.
- **Hudgens & Halloran (2008)** on causal inference in infectious disease settings with interference.
- **Aronow & Samii (2017)** for exposure mappings and estimands under arbitrary interference.
- **Basse & Airoldi (2018)** on randomization-based inference with network interference.

Treat SUTVA as a modeling choice instead of a background assumption. By articulating where it might fail--and designing around those failures--you keep causal estimates anchored to interpretable, policy-relevant quantities.
