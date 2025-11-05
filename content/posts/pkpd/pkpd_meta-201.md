+++
title = "PKPD_META 201: Bayesian Aggregation of Average Data"
type = "post"
date = "2025-12-05"
math = true
draft = true
summary = "Distilling Section 3 of Gelman et al. (2018) into a PKPD-ready recipe: informative priors on study deltas, simulated likelihoods for average-only readouts, and convergence tuning in Stan."
tags = ["pkpd", "bayesian", "meta-analysis", "stan"]
+++

Section 3 of the Gelman et al. Bayesian aggregation paper sharpens the workflow we need when a PKPD program mixes rich internal patient-level data with literature summaries. The authors work in a hierarchical Bayesian frame, keep the core population parameters intact, and layer external evidence through a controlled deviation vector $\delta = \bar{\phi} - \phi$.

## Setup: hierarchical baseline

The internal model starts with raw observations $y_{jk}$ for individual $j$ at time $k$, individual-level parameters $\alpha_j$, and shared hyperparameters $\phi$. Standard Stan code already gives us draws from $p(\alpha, \phi \mid y) = p(\phi) \prod_j p(\alpha_j \mid \phi) \prod_j p(y_j \mid \alpha_j, \phi)$. The meta-analysis trick is to recognize that an outside trial is another draw from the same population, but its parameters can shift by $\delta$.

Rather than declare the outside study "the same" or "completely different," we put an informative prior on $\delta$. Only the components we believe can move—think exposure-response scalers or observation noise for a new assay—are allowed to vary. Everything else is pinned at zero to stabilize the joint posterior.

## Simulated Laplace approximation for averages

The practical headache is that the external dataset is summarized as visit-level means $\bar{y}$. Because the PKPD models are nonlinear, we cannot write an exact likelihood for those averages. Section 3 proposes substituting a simulation-based Laplace approximation inside the Stan loop:

- Step 1: At each MCMC draw $s$, compute the external parameter vector $\bar{\phi}_s = \phi_s + \delta_s$.
- Step 2: Simulate $\tilde{J}$ synthetic patients under $\bar{\phi}_s$ using the same visit schedule as the summary report.
- Step 3: Record the mean vector $\tilde{M}_s$ and covariance $\tilde{\Sigma}_s$ across the simulated trajectories.
- Step 4: Scale $\tilde{\Sigma}_s$ by the reported cohort size $\bar{J}$ to approximate the variance of the published averages.
- Step 5: Add a multivariate normal log-likelihood $\log \mathcal{N}(\bar{y} \mid \tilde{M}_s, \tilde{\Sigma}_s / \bar{J})$ to the target density.

This plug-in estimate replaces the unavailable exact marginalization over latent patients while reusing the validated PKPD simulator.

## Tuning guidance for Stan

Because Hamiltonian Monte Carlo cannot draw fresh random numbers during sampling, each chain precomputes the simulation seeds (the Gaussian noise and any residual draws) and passes them as fixed data. Convergence then hinges on the Monte Carlo error from the $\tilde{J}$ replicates: halve the posterior Monte Carlo error by doubling $\tilde{J}$. The authors recommend running odd chains with $\tilde{J}$ and even chains with $2\tilde{J}$; disagreements in $\hat{R}$ immediately flag that more replicas are needed. For models with Gaussian observation noise, you can skip simulating raw measurements and work directly with averaged draws, appending the residual variance analytically to $\tilde{\Sigma}_s$ to cut compute time.

## PKPD meta takeaways

For our PKPD META series, this section translates into a concrete workflow:

- Reserve informative priors on $\delta$ for pharmacologically justified shifts (e.g., different dosing routes or assay platforms); enforce zeros elsewhere to keep shrinkage tight.
- Embed the simulated-average likelihood in the Stan model, letting us recycle the internal drug-disease ODE without rewriting it for summaries.
- Size $\tilde{J}$ so that the simulation standard error sits below the sampler's Monte Carlo error; mismatched chains become a diagnostic rather than a mystery.

With these pieces in place, we can fold literature means into ongoing PKPD programs and quantify how strongly they should pull on the in-house parameter posteriors.
