+++
title = "PKPD Meta 102: Semimechanistic Turnover Logic"
date = "2025-11-07T00:00:00Z"
type = "post"
draft = false
tags = ["pkpd", "pharmacometrics"]
categories = ["posts"]
description = "Explains the semimechanistic turnover framework of Jusko and Ko (1994) that underpins modern drug–disease models, including equilibrium behavior, drug modulation strategies, and guidance for Bayesian meta-analysis."
math = true
+++

[PKPD Meta 101](/posts/pkpd_meta-101) argued that translational pharmacology succeeds when we reuse a common model scaffold across disparate evidence streams. Before adapting that scaffold to specific clinical readouts, we need to understand the semimechanistic logic that keeps the differential equations biologically believable. This installment revisits the ideas set out by Jusko and Ko (1994), whose turnover logic still anchors the majority of disease progression and pharmacodynamic models used in Bayesian meta-analyses.

## Turnover as a balance of synthesis and loss

Jusko and Ko frame biomarkers as dynamic stocks $R(t)$ governed by a zero-order **production** term $k^{\text{in}}$ and a first-order **loss** term $k^{\text{out}} R(t)$:

$$
\frac{dR}{dt} = k^{\text{in}} - k^{\text{out}} R(t) + \text{drug effects}.
$$

The simplicity belies three powerful properties:

- **Natural baseline.** With drug effects off, the steady state is $R_0 = k^{\text{in}}/k^{\text{out}}$, so physiological set points emerge from measurable turnover rates rather than arbitrary intercepts.
- **Meaningful transients.** The eigenvalue $-k^{\text{out}}$ sets the recovery half-life ($t_{1/2} = \ln 2 / k^{\text{out}}$), allowing calibration to washout experiments or placebo arms.
- **Mechanistic knobs.** Drug effects enter through stimulation or inhibition of either production or loss, matching common biological mechanisms (e.g., stimulating synthesis of a hormone or blocking its degradation).

These ingredients yield models that are simple enough for inference yet faithful enough to reason about interventions.

## Four canonical drug actions

Jusko and Ko show that most indirect responses fall into four classes depending on which rate is modulated and whether the drug stimulates or inhibits. Using a generic exposure function $S(C)$ bounded between $0$ and $1$, the cases are:

1. **Inhibition of production:** $\tfrac{dR}{dt} = k^{\text{in}} \bigl(1 - S(C)\bigr) - k^{\text{out}} R$. The drug removes supply; lowering $R_0$ requires sustained presence.
2. **Stimulation of production:** $\tfrac{dR}{dt} = k^{\text{in}} \bigl(1 + S(C)\bigr) - k^{\text{out}} R$. Useful for mimicking cytokine release or compensatory synthesis.
3. **Inhibition of loss:** $\tfrac{dR}{dt} = k^{\text{in}} - k^{\text{out}} \bigl(1 - S(C)\bigr) R$. The drug slows clearance, lifting the response above baseline.
4. **Stimulation of loss:** $\tfrac{dR}{dt} = k^{\text{in}} - k^{\text{out}} \bigl(1 + S(C)\bigr) R$. Appropriate for cytotoxic therapies or rapid signal shutdown.

In each case the modulation is multiplicative, preserving the clean equilibrium $R_0$ while adjusting either the inflow or outflow channel. Because $S(C)$ is often parameterized as $E_{\max} C^\gamma / (EC_{50}^\gamma + C^\gamma)$, the model respects pharmacologic intuition: maximal effects, potency, and Hill slope map directly from in vitro or phase I data.

## Linearization and stability

The paper highlights how the turnover equation linearizes around equilibrium. Write $R(t) = R_0 + x(t)$ with $x$ representing deviations. For small excursions and smooth $S(C)$, the dynamics simplify to

$$
\frac{dx}{dt} \approx -k^{\text{out}} x(t) + k^{\text{in}} S(C) - k^{\text{out}} R_0 S(C).
$$

The homogeneous term $-k^{\text{out}} x$ is always restorative, so linear stability never depends on the drug concentration. Instability can only arise if we let drug effects multiply $x(t)$ directly (as happens when transforming the state, an issue we revisit in [PKPD Meta 103](/posts/pkpd_meta-103)). This insight motivates the design choice in many modern models to keep exposure terms additive in $x(t)$ even after reparameterizations.

## Guidance for hierarchical modeling

For Bayesian meta-analysis we typically assign patient- or study-level variation to the turnover rates. Jusko and Ko offer a clear separation between structural and variability parameters:

- $k^{\text{out}}$ governs time constants and is often shared across related populations, making it a natural target for partial pooling.
- $k^{\text{in}}$ captures baseline biology and can absorb covariate effects (e.g., disease severity) without disturbing the feedback structure.
- Drug modulation parameters ($E_{\max}$, $EC_{50}$, Hill slope) describe pharmacology and can borrow strength from nonclinical experiments.

Because the core ODE remains linear in $R$, posterior computations in tools like Stan remain stable even when we introduce random effects, measurement models, or latent concentration drives.

## Practical heuristics from Jusko & Ko

The original paper closes with pragmatic tips that still hold:

- **Validate baselines first.** Fit or simulate the placebo arm to ensure $k^{\text{in}}$ and $k^{\text{out}}$ reproduce untreated trajectories before layering on drug effects.
- **Use washout data to pin $k^{\text{out}}$.** Recovery slopes identify the loss rate; once fixed, production rates follow from baseline levels.
- **Parameterize $S(C)$ on log concentrations.** This spreads information evenly across concentration ranges and simplifies hierarchical priors.
- **Track units relentlessly.** $k^{\text{in}}$ carries the same units as $R$ per unit time; $k^{\text{out}}$ is inverse time. Keeping units straight exposes modeling mistakes early.

These heuristics provide a repeatable workflow: start with the turnover backbone, check the placebo dynamics, then explore which of the four canonical modulation schemes aligns with known biology.

## Looking ahead

With the semimechanistic foundation in place, [PKPD Meta 103](/posts/pkpd_meta-103) shows how to adapt the turnover template when the observed endpoint lives on a transformed scale, using the Gelman et al. (2018) BCVA model as a case study. That adaptation keeps the spirit of Jusko and Ko’s framework while navigating the practical constraints of modern clinical data.
