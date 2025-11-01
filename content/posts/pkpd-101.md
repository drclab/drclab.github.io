+++
title = "PKPD 101: Deriving the One-Compartment Oral Model"
date = "2025-11-01T01:00:00Z"
type = "post"
draft = true
math = true
tags = ["pkpd", "pharmacology", "modeling", "pharmacokinetics"]
categories = ["posts"]
description = "Step-by-step derivation of the concentration-time profile for oral dosing, from compartment ODEs to the closed-form solution, with visual guides to clearance and absorption ratios."
+++

This post walks through the one-compartment oral pharmacokinetic model from first principles: setting up the differential equations, solving them step-by-step using the integrating factor method, and interpreting the resulting concentration-time profile. By the end you will understand how absorption and elimination compete to shape plasma curves, recognize the key parameter ratios that control curve shape, and be ready to move on to pharmacodynamic models in [PKPD 102](/posts/pkpd-102).

## Why pharmacometrics matters

- Translate preclinical and phase I data into decision-ready dose regimens.
- Quantify uncertainty around exposure-response relationships instead of relying on point estimates.
- Simulate “what-if” scenarios (renal impairment, pediatrics, adherence gaps) before they happen in the clinic.
- Provide a common language for clinicians, statisticians, and regulators reviewing the same molecule.

## Key definitions

- **PK (pharmacokinetics)** studies how the body handles a drug: absorption, distribution, metabolism, and elimination (ADME). Units: concentration over time.
- **PD (pharmacodynamics)** describes how the drug affects the body: biochemical, physiological, or clinical responses. Units: effect over time.
- **PKPD model** links the PK time course to the PD time course, often through a dose → concentration → effect cascade.
- **Structural parameters** describe the mechanistic model (e.g., clearance, EC50); **statistical parameters** capture variability (between-subject, within-subject, residual error).

## PK building blocks

A one-compartment oral model with first-order absorption, clearance, and volume of distribution is the standard starting point. Let $A_g$ be the amount of drug in the gut and $C$ the plasma concentration. Figure&nbsp;1 sketches the canonical structure used in introductory PK discussions.

{{< figure src="/img/pkpd/one-compartment-oral.svg" alt="Oral one-compartment pharmacokinetic model with a gut compartment feeding a central compartment via first-order absorption and clearance leaving the central compartment." caption="Figure 1. Simplified oral one-compartment PK model illustrating the gut absorption rate constant $k_a$ and systemic clearance $CL$ from the central compartment." >}}

$$
\frac{dA_g}{dt} = -k_a A_g,\qquad
\frac{dC}{dt} = \frac{k_a}{V} A_g - \frac{CL}{V} C,
$$

where

- $k_a$ is the first-order absorption rate constant,
- $CL$ is systemic clearance,
- $V$ is the apparent volume of distribution.

The analytical concentration-time profile for a single oral dose $D$ given at $t=0$ is

$$
C(t) = \frac{D \, k_a}{V(k_a - k_{el})} \left(e^{-k_{el} t} - e^{-k_a t}\right),\quad k_{el} = \frac{CL}{V}.
$$

### Deriving the concentration-time profile

This closed-form solution deserves a careful walkthrough because it reveals how absorption and elimination compete to shape the plasma curve. Starting from the differential equations for the gut and central compartments:

$$
\frac{dA_g}{dt} = -k_a A_g,\qquad
\frac{dC}{dt} = \frac{k_a}{V} A_g - k_{el} C.
$$

**Step 1: Solve the gut compartment.** The first equation is a simple exponential decay with initial condition $A_g(0) = D$ (the entire dose sits in the gut at time zero):

$$
A_g(t) = D e^{-k_a t}.
$$

This tells us the amount of drug remaining in the gut falls exponentially at rate $k_a$.

**Step 2: Substitute into the plasma equation.** Now we plug $A_g(t)$ into the second differential equation:

$$
\frac{dC}{dt} = \frac{k_a D}{V} e^{-k_a t} - k_{el} C.
$$

This is a first-order linear ordinary differential equation (ODE) with a time-dependent forcing term. The general structure is $\frac{dC}{dt} + k_{el} C = \frac{k_a D}{V} e^{-k_a t}$.

**Step 3: Apply the integrating factor method.** Multiply both sides by $e^{k_{el} t}$:

$$
e^{k_{el} t} \frac{dC}{dt} + k_{el} e^{k_{el} t} C = \frac{k_a D}{V} e^{(k_{el} - k_a) t}.
$$

The left-hand side is now the derivative of $C(t) e^{k_{el} t}$, so we can write:

$$
\frac{d}{dt}\left(C(t) e^{k_{el} t}\right) = \frac{k_a D}{V} e^{(k_{el} - k_a) t}.
$$

**Step 4: Integrate both sides.** Starting from $t=0$ where $C(0)=0$ (no drug in plasma initially):

$$
C(t) e^{k_{el} t} - C(0) e^{0} = \frac{k_a D}{V} \int_0^t e^{(k_{el} - k_a) s} ds.
$$

Since $C(0)=0$, the left side simplifies to $C(t) e^{k_{el} t}$. The integral evaluates to:

$$
\int_0^t e^{(k_{el} - k_a) s} ds = \frac{e^{(k_{el} - k_a) t} - 1}{k_{el} - k_a}.
$$

Putting it together:

$$
C(t) e^{k_{el} t} = \frac{k_a D}{V(k_{el} - k_a)} \left(e^{(k_{el} - k_a) t} - 1\right).
$$

**Step 5: Solve for $C(t)$.** Multiply both sides by $e^{-k_{el} t}$:

$$
C(t) = \frac{k_a D}{V(k_{el} - k_a)} \left(e^{-k_a t} - e^{-k_{el} t}\right).
$$

Rearranging the order of terms in the parentheses and flipping the sign of the denominator gives the canonical form:

$$
C(t) = \frac{D \, k_a}{V(k_a - k_{el})} \left(e^{-k_{el} t} - e^{-k_a t}\right).
$$

**Why this form is useful:** The factor $\frac{D \, k_a}{V(k_a - k_{el})}$ scales the entire curve, while the difference of exponentials $\left(e^{-k_{el} t} - e^{-k_a t}\right)$ creates the characteristic rise-and-fall shape. At $t=0$, both exponentials equal 1 so $C(0)=0$ as expected. As $t \to \infty$, both exponentials vanish and $C(t) \to 0$. In between, the competition between absorption (driven by $k_a$) and elimination (driven by $k_{el}$) determines the peak time and height.

### Visualizing the key ratios

Two ratios govern how the profile bends: $k_{el} = CL/V$ tells you how aggressively the body clears drug relative to the "space" it has to dilute into, and the absorption-to-elimination ratio $k_a/k_{el}$ determines the curvature in the rising limb.

{{< figure src="/img/pkpd/clearance-volume-ratio.svg" alt="Three concentration-time curves showing a faster decline and lower exposure as the CL/V ratio increases." caption="Figure 2. Higher $CL/V$ (larger elimination rate constant) shortens the half-life and pulls the whole curve downward, while lower $CL/V$ stretches both $t_{1/2}$ and exposure." >}}

Figure 2 isolates the clearance-to-volume ratio. Holding absorption constant, moving from a low to a high $CL/V$ rotates the declining limb downward: the peak arrives earlier, trough concentrations shrink, and overall exposure drops. This is the geometric picture behind dose-adjustment rules of thumb such as “double the clearance, halve the steady-state trough.”

{{< figure src="/img/pkpd/absorption-elimination-ratio.svg" alt="Three concentration-time curves showing how faster, balanced, and slower absorption relative to elimination change the peak shape." caption="Figure 3. The $k_a/k_{el}$ ratio sculpts the rising limb: very fast absorption yields a sharp peak, near-balance broadens the crest, and $k_a < k_{el}$ produces flip-flop kinetics where the terminal slope tracks absorption." >}}

Figure 3 focuses on the absorption-to-elimination ratio. When $k_a \gg k_{el}$ the gut empties quickly, so concentration shoots up and the elimination phase looks like a clean mono-exponential decay. When $k_a \approx k_{el}$, absorption and clearance compete, stretching the peak and flattening $t_{\max}$. If $k_a < k_{el}$ you are in **flip-flop kinetics** territory: elimination is so quick that the terminal slope reflects the slower absorption process, which can mislead you into overestimating half-life unless you recognize the ratio is inverted. Checking these ratios early keeps you from chasing spurious covariates when the shape of the curve is rate-limited by formulation or gastrointestinal transit.

When we talk about plasma concentration in this context, we literally mean “how much drug is dissolved in a given volume of blood plasma at a specific time.” Units are often nanograms per milliliter (ng/mL) or micromoles per liter (µM), but the key idea is a ratio of amount over volume. Higher plasma concentration means more of the drug is circulating and ready to reach its targets; lower concentration means the body has absorbed less or has already cleared a portion away. Because blood sampling is minimally invasive and routinely collected in studies, plasma concentration becomes the practical window into the otherwise hidden compartments of the body.

A few gentle rules of thumb help interpret these curves:

- Right after dosing, the gut is full and the plasma is relatively empty, so concentration starts near zero and climbs as the drug is absorbed.
- The curve peaks when absorption and elimination balance each other; beyond that point elimination “wins” and concentration falls.
- Steeper declines hint at faster clearance or distribution into tissues, while flatter tails suggest the drug sticks around longer.

Even small tweaks to parameters like clearance or volume shift the whole concentration profile. For example, doubling clearance halves the steady-state plasma concentration at a fixed dose, which matters when you want to keep patients within a therapeutic window. Clinical teams lean on these intuitive pictures—rise, peak, decline—to make dose adjustments feel much less mysterious.

### Exposure metrics you will see in every report

- **$C_{\max}$**: peak concentration after a dose.
- **$t_{\max}$**: time it takes to reach $C_{\max}$.
- **AUC (area under the concentration-time curve)**: overall exposure; proportional to $D/CL$ for linear PK.
- **Half-life ($t_{1/2}$)**: time to halve the concentration; $t_{1/2} = \ln 2 / k_{el}$ for one-compartment linear elimination.
- **Accumulation ratio**: steady-state exposure versus single-dose exposure, driven by the dosing interval relative to $t_{1/2}$.

## Multiple dosing and steady state

The single-dose solution extends cleanly to repeat dosing by **superposition**: add the contribution from each prior dose shifted by its dosing interval. After roughly $4$ to $5$ half-lives the peaks and troughs stop drifting and the system reaches **steady state**. The peak-to-trough swing hinges on the relationship between the dosing interval $\tau$ and the elimination rate constant:

$$
R_{acc} = \frac{1}{1 - e^{-k_{el} \tau}}, \qquad C_{\text{trough,ss}} \approx C_{\text{max,ss}} e^{-k_{el} \tau}.
$$

Shortening $\tau$ or lengthening $t_{1/2}$ increases accumulation; the same math powers dosing adjustments such as extending the interval for renally impaired patients to keep troughs in range.

## Oral bioavailability checks

Not every milligram swallowed reaches systemic circulation. Oral bioavailability combines **fraction absorbed** from the gut and **first-pass extraction** by the liver:

$$
F = F_{\text{abs}} \times (1 - E_H), \qquad E_H = \frac{CL_H}{Q_H + CL_H}.
$$

Comparing oral and intravenous AUC values normalizes away clearance and reveals incomplete absorption or intense first-pass metabolism. Formulation scientists modulate $F_{\text{abs}}$ with solubility enhancers, while medicinal chemists target hepatic extraction by tuning lipophilicity or metabolic stability.

## Scaling PK parameters

Clearance and volume rarely stay constant across a population. Two common levers are:

- **Allometric scaling**: $CL \propto (\text{weight}/70)^{0.75}$, $V \propto (\text{weight}/70)^{1.0}$ for adult body-size adjustments.
- **Covariate models**: express clearance as $CL = CL_{\text{typ}} \exp(\theta_{eGFR} \cdot (\text{eGFR} - 90))$ or via power terms to encode organ function, formulation, or genotype effects.

Tracking how these parameters vary is the backbone of precision dosing and feeds straight into exposure simulations.

## Next steps

This primer focused on the pharmacokinetic side: how drugs move through the body and how to interpret concentration-time curves. With these PK building blocks in hand, you can now move to [PKPD 102](/posts/pkpd-102), where we explore pharmacodynamic models, link concentration to effect, and walk through the complete PKPD workflow from data exploration to simulation and dose recommendation.

## Diagnostics you cannot skip

- Overlay observed and model-predicted concentration-time curves for each subject; systematic bias in peaks or tails often signals a misspecified absorption or clearance process.
- Plot conditional weighted residuals (CWRES) versus time and predictions to catch heteroscedasticity or delayed absorption that the model fails to capture.
- Examine eta shrinkage on clearance and volume; extreme shrinkage warns that the sampling design cannot support patient-level dosing decisions.
- Run prediction-corrected visual predictive checks (pcVPCs) focused on concentration quantiles; poor agreement around peaks usually points to formulation or food effects that deserve covariate testing.

## Communicating PK findings

- Lead with exposure metrics ($C_{\max}$, trough, AUC) tied to observed safety or efficacy thresholds.
- Convert parameter changes into practical statements (e.g., “clearance is 40% lower in Child-Pugh B, suggesting a 30% dose reduction to match healthy exposure”).
- Summarize covariate impacts separately from residual variability so clinicians can distinguish controllable versus random sources of spread.

## Where to go next

- **Textbook**: “Clinical Pharmacokinetics and Pharmacodynamics” by Rowland & Tozer remains the foundational PK reference.
- **Workflows**: Explore `mrgsolve` or `nlmixr` tutorials to gain hands-on experience implementing one- and two-compartment oral models.
- **Data sets**: Recreate classic theophylline PK analyses or FDA bioequivalence case studies to practice estimation, diagnostics, and superposition.

Solid PK fluency makes the pharmacodynamic linkage in PKPD 102 far more intuitive—once you trust the exposure profiles, you can focus on how those concentrations translate into clinical effect.
