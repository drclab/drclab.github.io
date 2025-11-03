+++
title = "PKPD Meta 103: Recasting the Drug–Disease Turnover Model"
date = "2025-12-13T00:00:00Z"
type = "post"
draft = false
tags = ["pkpd", "turnover", "meta-analysis"]
categories = ["posts"]
description = "Expands the PKPD Meta series by showing how the Gelman et al. (2018) drug–disease model modifies the standard turnover form to operate on logit-transformed visual acuity data, and how that choice supports Bayesian aggregation of patient-level and averaged evidence."
math = true
+++

[PKPD Meta 101](/posts/pkpd_meta-101) framed modern drug discovery as staged learning with a single PK/PD narrative threading individual and aggregate evidence, and [PKPD Meta 102](/posts/pkpd_meta-102) distilled the semimechanistic turnover logic that makes that narrative credible. The next building block is the **drug–disease model** that carries that logic into the Gelman et al. (2018) program. Their structure is anchored on a turnover-style pharmacodynamic equation, but with a twist: the clinical endpoint, best corrected visual acuity (BCVA), is modeled on a logit scale. That change forces a careful rethink of the indirect response equations we covered in [PKPD 104: Turnover Models for Indirect Response](/posts/pkpd-104).

This post shows how the authors adapt the standard turnover template, why the logit transform makes the classic multiplicative form unusable, and how the resulting ODE in equation (1) on p. 1588 keeps the biology intact while letting BCVA live between 0 and 100 letters.

## Recap: the standard turnover scaffold

[PKPD 104](/posts/pkpd-104) introduces the canonical indirect response equation,

$$
\frac{dR}{dt} = k_\text{in}\bigl(1 + S_\text{in}(C_p)\bigr) - k_\text{out}\bigl(1 + S_\text{out}(C_p)\bigr) R,
$$

which assumes the response $R(t)$ is a **positive-valued** biomarker maintained by zero-order production ($k_\text{in}$) and first-order loss ($k_\text{out}$). The drug modulates production or loss multiplicatively through stimulation/inhibition terms $S(C_p)$. With $S = 0$, the modulation vanishes, the ODE collapses to $dR/dt = k_\text{in} - k_\text{out} R$, and the unique steady state arises by setting the derivative to zero: $R_0 = k_\text{in}/k_\text{out}$. Any perturbation or initial condition decays back to that baseline at rate $k_\text{out}$ because the homogeneous solution $e^{-k_\text{out} t}$ damps excursions, so $R_0$ represents both the physiological set point and the placebo trajectory.

This structure delivers the hallmarks of turnover dynamics—delayed response, lingering recovery, and a clear biological interpretation of rate constants.

## Why the Gelman et al. endpoint breaks the standard form

BCVA is bounded between 0 and 100 letters, so Gelman et al. logit-transform the scaled outcome $y_{jk}/100$ for patient $j$ at visit $k$:

$$
R_j(t) = \operatorname{logit}\!\left(\frac{y_{jk}}{100}\right).
$$

Once we work on the logit scale, $R_j(t)$ can take any real value—positive or negative—depending on the patient's acuity. The multiplicative loss term in the standard turnover model implicitly assumes $R(t) \ge 0$. On the logit scale that assumption fails, making the product $k_\text{out} R(t)$ a sign-sensitive term that would flip direction whenever $R(t)$ crossed zero. Worse, modulating production by $(1 + S_\text{in})$ no longer guarantees that the transformed state stays consistent with the bounded observable.

To avoid those pathologies, the BCVA model needs:

- A **linear loss term** that does not change sign with $R_j(t)$.
- A **drug effect** that can push the logit state up or down without violating the observation bounds once mapped back through the inverse logit.

## The revised turnover equation

Gelman et al. (2018) answer with a modified ODE (equation 1, p. 1588):

$$
\frac{dR_j(t)}{dt} = k_{j}^{\text{in}} - k_{j}^{\text{out}} \left[ R_j(t) - E_{\max,j}S_j(C_j(t)) \right].
$$

Key differences from the standard form:

- **Additive production and loss.** $k_{j}^{\text{in}}$ and $k_{j}^{\text{out}}$ appear linearly, so the restorative drift $-k_{j}^{\text{out}} R_j(t)$ is always proportional to the displacement from baseline and never flips direction when $R_j(t)$ crosses zero. That guarantees a single globally attracting equilibrium at $k_{j}^{\text{in}}/k_{j}^{\text{out}}$ under placebo, keeps the linear ODE stable for large positive or negative logits, and lets stochastic or hierarchical variability in $k_{j}^{\text{in}}$ simply shift the mean rather than rewriting the entire feedback structure.
- **Additive drug effect.** The exposure term $E_{\max,j} S_j C_j(t)$ is subtracted directly, reflecting that higher intraocular concentrations improve BCVA (higher $R_j$ on the logit scale). The sign convention can flip ($+/-$) depending on whether the drug stimulates or inhibits the measured response.
- **Hill-type sensitivity.** $S_j$ is a Hill function of the latent concentration, implemented as a logistic transform of $\log C_j(t)$ so that relative changes in concentration drive smooth, saturating modulations just like the $S(C)$ terms in the standard model.

The structural choices match the semimechanistic turnover logic proposed by \[Jusko and Ko (1994)\], while the ocular context fixes ancillary pharmacokinetic values such as a 4 mL vitreous volume \[Hart (1992)\] and a nine-day elimination half-life \[Xu et al. (2013)\] to reflect biologically plausible bounds.

Set $C_j(t) = 0$ (placebo) and the system collapses to $R_j^\ast = k_{j}^{\text{in}}/k_{j}^{\text{out}}$, exactly mirroring the baseline logic in classical turnover models but now stable on the logit scale.

## Linking back to the classical structure

We can view equation (1) as the **linearized sibling** of the indirect response forms from [PKPD 104](/posts/pkpd-104):

- Rewrite the standard equation by expanding the loss term: $-k_\text{out} R(t) - k_\text{out} S_\text{out}(C_p) R(t)$. On a logit scale, the second term becomes problematic because it multiplies $R(t)$.
- Gelman et al. effectively drop the multiplicative modulation and replace it with an additive driver $-E_{\max,j} S_j C_j(t)$. This is equivalent to assuming $S_\text{out}(C_p)$ acts on **rate of change** rather than directly scaling the response magnitude.
- When the observation is mapped back via $\operatorname{logit}^{-1}$, the bounded BCVA values emerge naturally, so no additional constraints are required.

The turnover intuition remains intact: $k_{j}^{\text{out}}$ controls recovery half-life, $k_{j}^{\text{in}}$ sets the untreated baseline, and the drug term captures how exposure perturbs the synthesis–loss balance. The difference lies purely in ensuring those mechanisms respect the transformed scale.

## Why the modified form matters for meta-analysis

The Bayesian aggregation strategy relies on simulating individual trajectories under varying regimens and comparing them to published averages. The revised turnover model:

- **Keeps simulations stable.** Linear additive terms avoid the numerical instabilities that a sign-changing multiplicative loss would introduce when sampling with Stan.
- **Supports partial pooling.** Patient-specific $k_{j}^{\text{in}}$ and $k_{j}^{\text{out}}$ parameters can share hierarchical priors, while the drug effect parameters ($E_{\max}$, $EC_{50}$ through $S_j$) remain interpretable across internal and external regimens.
- **Makes summaries comparable.** Because the inverse logit maps predictions back to 0–100 letters, simulated averages line up directly with published BCVA means, enabling the normal approximation used to fuse aggregate data.

## Separating drug and disease parameters

Gelman et al.'s **drug–disease model** is explicitly anchored on the placebo trajectory. Under placebo ($C_j(t) = 0$), the ODE contracts to $R_j^\ast = k_{j}^{\text{in}}/k_{j}^{\text{out}}$, the same equilibrium that describes untreated disease progression. The moment we introduce exposure, only the terms tied to the drug move: the ocular elimination half-life $t_{1/2}$ determines the PK subsystem that feeds concentration $C_j(t)$, while $E_{\max}$ and $EC_{50}$ govern how that exposure perturbs BCVA on the logit scale. Everything else—baseline acuity, recovery dynamics, measurement noise—is inherited from the placebo description and treated as nondrug-specific. That clean separation lets us reason about **treated patients relative to placebo patients** without re-estimating the entire disease model each time we change dose, formulation, or dosing interval. The drug parameters flex to capture potency and persistence, but the disease parameters continue to describe natural history that future interventions can reuse.

## Practical takeaways

- Whenever your endpoint is bounded and you transform it (e.g., logit, probit), revisit the turnover equations to ensure production and loss remain well-defined on the new scale.
- Additive exposure terms, as in Gelman et al.’s equation (1), provide a safe default when the transformed response can cross zero.
- Hierarchical priors on $k^{\text{in}}$ and $k^{\text{out}}$ let you recover the placebo natural history while still borrowing information across regimens and studies—a prerequisite for the Bayesian aggregation described in [PKPD Meta 101](/posts/pkpd_meta-101).

Next in the series we will turn those structural insights into working external likelihoods, so aggregated trial reports can update the same Stan model that fits the internal patient-level data.
