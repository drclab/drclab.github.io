+++
title = "PKPD Meta 101: Modeling the Modern Drug Discovery Process"
date = "2025-12-11T00:00:00Z"
type = "post"
draft = true
tags = ["pkpd", "meta-analysis", "drug-discovery"]
categories = ["posts"]
description = "Sets the stage for a PK/PD meta-analysis series by tracing the modern drug discovery pipeline and showing how Bayesian aggregation of individual and average data keeps decisions aligned across phases."
+++

Modern drug discovery is a relay race across preclinical models, exploratory human studies, confirmatory trials, and ultimately market commitments. Gelman et al. (2018) describe how each leg of that race depends on integrating every scrap of evidence we have—rich longitudinal measurements from our own trials and summarized outcomes reported by others—to keep dose, schedule, and safety decisions coherent as a program advances.

This opening post frames the series around a simple question: how do we keep pharmacokinetic/pharmacodynamic (PK/PD) knowledge trustworthy as we move from first-in-human data to pivotal trials when parts of the evidence arrive only as averages?

## Drug discovery as staged learning

Gelman et al. (2018) highlight the classic stages of development—tolerability, safety, and efficacy—and stress that every stage should condition on all available information when planning the next one. That implies:

- **Preclinical and translational** teams generate the first PK/PD priors, often by fitting nonlinear mixed-effects models to animal and in-vitro data.
- **Phase I–II** studies collect dense longitudinal measurements on a modest number of people. These raw trajectories let us calibrate hierarchical PK/PD models that separate drug-specific parameters (clearance, half-life, effect maxima) from placebo progression and natural history.
- **Phase III and beyond** typically add external comparators because ethics, cost, or speed rarely allow head-to-head trials against every standard of care. Published summaries, not raw patient-level data, are the norm.

The pipeline is therefore a progression from richly instrumented but small cohorts to broad evidence that is frequently summarized into mean responses, standard errors, and aggregate baselines.

## Where PK/PD data come together

The Ranibizumab and Aflibercept program discussed in the paper illustrates the data mix we now face routinely:

- Internal trials such as MARINA, ANCHOR, and EXCITE provided patient-level best corrected visual acuity (BCVA) trajectories under multiple dosing schemas.
- External evidence—from VIEW1 and VIEW2—arrived only as published averages and standard errors for the same BCVA endpoints, yet those aggregates were vital for exploring longer interdose intervals and alternative molecules.

The pharmacometric backbone was a semimechanistic turnover model linking latent ocular drug concentrations to BCVA response via Hill-type stimulation. Capturing within-patient dynamics, between-patient variability, and dosing regimen effects required the full richness of the longitudinal data. But matching those predictions to external averages meant the model also had to speak the language of summary statistics.

## Bayesian aggregation as connective tissue

Rather than simplify the model to fit analytical meta-analysis formulas, Gelman et al. (2018) simulated the external summaries directly:

- Use Stan to sample from the posterior defined by the hierarchical PK/PD model on the patient-level data.
- For each posterior draw, simulate the trial design underlying the published summaries to approximate their likelihood.
- Allow key parameters to vary by trial to reflect regimen differences, while partial pooling keeps estimates coherent with the internal evidence.

This approach respects the nonlinear structure of the pharmacometric model and keeps the connection between doses, concentrations, and responses intact. It turns average-only evidence into another constraint on the same mechanistic scaffold rather than a competing story told in a different statistical dialect.

## Why it matters for modern drug discovery

Bridging individual and aggregate data in a unified Bayesian framework pays off in three ways that every PK/PD team can recognize:

- **Designing smarter trials.** Clinical trial simulations that borrow strength from published summaries reduce the risk of underdosing or overdosing new cohorts. Gelman et al. (2018) used the aggregated data to justify exploring twelve-week dosing intervals while maintaining efficacy thresholds.
- **Communicating uncertainty.** Posterior predictive checks and credible intervals keep physicians and regulators focused on uncertainty bands instead of point-estimate comparisons, aligning with the meta-analytic standards described in the Cochrane Handbook.
- **Accelerating decision cycles.** Once the model can ingest averages, literature updates can immediately refine dose projections, even before a data-sharing agreement yields patient-level records.

## Series roadmap

To keep the focus on actionable PK/PD workflows, the next posts in this series will unpack the moving parts introduced here:

- **PKPD Meta 102: Reconstructing External Likelihoods.** Practical recipes for turning aggregated endpoint tables into Stan-friendly constraints, including variance propagation and sampling strategies.
- **PKPD Meta 103: Designing Hierarchical Priors.** How to pick hyperpriors that separate drug- and disease-level effects without over-smoothing regimen differences.
- **PKPD Meta 201: Simulation-First Trial Design.** Building decision criteria on posterior predictive distributions so schedule changes (e.g., q12w dosing) stay tethered to mechanistic insight.
- **PKPD Meta 202: Communicating Evidence to Stakeholders.** Packaging model outputs for regulators, clinicians, and program leaders who need clarity on risk, not just model fit metrics.

## Key takeaways

- Treat each stage of drug discovery as an opportunity to update a single PK/PD story that can digest both individual trajectories and published averages.
- Use simulation-based Bayesian aggregation to extend mechanistic models instead of simplifying them to match analytical shortcuts.
- Document the evidence flow—what comes from internal trials, what comes from literature—to make your program inviting for future collaborators and reviewers.

Next up we will build the technical muscle to reconstruct external likelihoods so those literature summaries slot cleanly into your Stan models.
