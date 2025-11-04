+++ 
title = "PKPD Meta 104: Mapping the Gelman ODE into Stan"
date = "2025-12-20T00:00:00Z"
type = "post"
draft = false
tags = ["pkpd", "turnover", "meta-analysis", "stan"]
categories = ["posts"]
description = "Shows how the Gelman et al. (2018) drug-disease ODE introduced in PKPD Meta 103 is implemented in Stan, highlighting the parameterization, time-varying stimulus, and link back to bounded BCVA outcomes."
math = true
+++

[PKPD Meta 103](/posts/pkpd_meta-103) reframed the Gelman et al. (2018) drug-disease model as a turnover equation that lives on a logit scale:

$$
\frac{dR_j(t)}{dt} = k_{j}^{\text{in}} - k_{j}^{\text{out}} \left[ R_j(t) - E_{\max,j} S_j\bigl(C_j(t)\bigr) \right].
$$

This post traces that equation into the Stan program used for simulation and inference. The model lives in `content/ipynb/stand_ode.stan`, and every block mirrors one ingredient from the math: the additive drift, the exposure-driven stimulus, and the inverse-logit observation model that keeps best corrected visual acuity (BCVA) between 0 and 100 letters. Line references below point directly into the Stan file for one-to-one traceability.

## Functions block: line-by-line

{{< highlight stan "linenos=true" >}}
functions {
  vector drug_disease_stim_kinL_Et_ode(real t,
                                       vector y,
                                       array[] real theta,
                                       array[] real x_r,
                                       array[] int x_i) {
    vector[1] dydt;
    real emax_0 = theta[1];
    real lec_50 = theta[2];
    real r180 = theta[3];
    real beta = theta[4];
    real k_in = theta[5];
    real k_out = theta[6];
    real start_t = x_r[1];
    real lconc_0 = x_r[2];
    real K = x_r[3];
    real hill = x_r[4];
    real lconcentration = lconc_0 - K * (t - start_t);
    real emax = emax_0 * (r180 + (1 - r180) * exp(-beta * t / 30.0));
    real stim = emax * inv_logit(hill * (lconcentration - lec_50));
    dydt[1] = k_in - k_out * (y[1] - stim);
    return dydt;
  }
}
{{< /highlight >}}

- `stand_ode.stan:7` declares `dydt`, the vector storing the derivative `dR_j(t)/dt`.
- `stand_ode.stan:8, stand_ode.stan:9, stand_ode.stan:10, stand_ode.stan:11, stand_ode.stan:12, stand_ode.stan:13` unpack `theta[1:6]` into `Emax0`, `ell_EC50`, `r180`, `beta`, `k_in`, and `k_out`, the parameters that appear in equation (1).
- `stand_ode.stan:14, stand_ode.stan:15, stand_ode.stan:16, stand_ode.stan:17` unpack `start_t`, `lconc_0`, `K`, and `hill`, the fixed inputs that define the exposure profile.
- `stand_ode.stan:18` encodes the log-concentration path `ell_j(t) = ell_0 - K * (t - start_t)`.
- `stand_ode.stan:19` implements the time-varying gain `Emax_j(t) = Emax0 * (r180 + (1 - r180) * exp(-beta * t / 30))`, the gradual loss of drug effect discussed in PKPD Meta 103.
- `stand_ode.stan:20` calculates the drug stimulus `stim_j(t) = Emax_j(t) * inv_logit(hill * (ell_j(t) - ell_EC50))`, which corresponds to the `Emax_j * S_j(C_j(t))` term from equation (1).
- `stand_ode.stan:21` applies the turnover equation verbatim: `k_in - k_out * (y[1] - stim)`.

The helper formulas link back to the derivation:

$$
\ell_j(t) = \log C_j(t) = \ell_0 - K(t - t_{\text{start}})
$$

$$
E_{\text{max},j}(t) = E_{\text{max},0}\left(r_{180} + (1 - r_{180}) e^{-\beta t / 30}\right)
$$

$$
\text{stim}_j(t) = E_{\text{max},j}(t)\,\mathrm{logit}^{-1}\left(\text{hill} \cdot \bigl(\ell_j(t) - \ell_{\mathrm{EC}_{50}}\bigr)\right)
$$

so the Stan function faithfully reproduces every term in PKPD Meta 103.

## Data block: known quantities

- `stand_ode.stan:27` supplies `N`, the number of observation times.
- `stand_ode.stan:28, stand_ode.stan:29` pass the visit times (`time`) and observed BCVA values (`bcva_obs`).
- `stand_ode.stan:30, stand_ode.stan:31` provide `start_t` and `lconc0`, the initial log-concentration state.
- `stand_ode.stan:32` captures the washout slope `K` so that concentration decays exponentially on the natural scale.
- `stand_ode.stan:33` sets the Hill slope controlling how sharply exposure translates into effect.
- `stand_ode.stan:34, stand_ode.stan:35` give `r180` and `beta`, shaping the time-varying `Emax_j(t)`.
- `stand_ode.stan:36` introduces `sigma_prior_scale`, allowing analysts to tune how informative the noise prior should be.

## Parameters block: unknown biology

- `stand_ode.stan:40, stand_ode.stan:41` declare strictly positive `k_in` and `k_out`, anchoring the production and loss rates from equation (1).
- `stand_ode.stan:42` stores `emax0`, the peak stimulus on the logit scale.
- `stand_ode.stan:43` holds `lec50`, the log-concentration at half-maximal effect.
- `stand_ode.stan:44` keeps the residual standard deviation `sigma` positive.
- `stand_ode.stan:45` treats `R0` as an unknown initial logit acuity, aligning with the baseline state discussed in PKPD Meta 103.

## Transformed parameters: wiring the solver

- `stand_ode.stan:51` allocates `y0`, the initial condition vector used by the ODE integrator.
- `stand_ode.stan:52, stand_ode.stan:53` create `theta` and `x_r`, the parameter and data packs passed into the derivative function.
- `stand_ode.stan:55` seeds the ODE with `R0`.
- `stand_ode.stan:56` sets `theta[1] = emax0`, tying the first entry of `theta` to the peak stimulus.
- `stand_ode.stan:57` sets `theta[2] = lec50`, tying the second entry to the log EC50.
- `stand_ode.stan:58` sets `theta[3] = r180`, passing the late-time fraction.
- `stand_ode.stan:59` sets `theta[4] = beta`, passing the decay rate.
- `stand_ode.stan:60` sets `theta[5] = k_in`, carrying the production rate.
- `stand_ode.stan:61` sets `theta[6] = k_out`, carrying the loss rate.
- `stand_ode.stan:62` records `start_t`, the dosing time.
- `stand_ode.stan:63` records `lconc0`, the initial log concentration `ell_0`.
- `stand_ode.stan:64` records `K`, the washout slope.
- `stand_ode.stan:65` records `hill`, the steepness of `S_j(C_j(t))`.
- `stand_ode.stan:67` defines `t0 = start_t - 1e-6` so the solver starts infinitesimally before the first recorded visit.

The call beginning at `stand_ode.stan:68`,

{{< highlight stan "linenos=true,linenostart=68" >}}
array[N] vector[1] y_hat = ode_rk45(drug_disease_stim_kinL_Et_ode,
                                    y0,
                                    t0,
                                    to_array_1d(time),
                                    theta,
                                    x_r,
                                    x_i);
{{< /highlight >}}

requests the solution `R_j(t)` on the visit grid. The loop starting at `stand_ode.stan:75` maps the latent logits back to letters:

$$
\mu_{\mathrm{BCVA},jk} = 100 \times \operatorname{logit}^{-1}\bigl(R_j(t_{jk})\bigr).
$$

The mapping comes straight from the definition of the latent state in PKPD Meta 103: $R_j(t) = \operatorname{logit}\bigl(y_{jk} / 100\bigr)$ with $y_{jk}$ measured in ETDRS letters. Solving that relationship for $y_{jk}$ yields

$$
y_{jk} = 100 \times \frac{1}{1 + \exp\bigl(-R_j(t_{jk})\bigr)} = 100 \times \operatorname{logit}^{-1}\bigl(R_j(t_{jk})\bigr),
$$

so $\mu_{\mathrm{BCVA},jk}$ is simply the expected letter score implied by the logit-scale trajectory. The inverse-logit brings the latent process back to the unit interval, and the factor of 100 restores the original clinical scale while keeping predictions between 0 and 100.

- `stand_ode.stan:76` implements the inverse-logit followed by rescaling to the 0-100 BCVA range.

## How Stan integrates the ODE

- `stand_ode.stan:68` invokes `ode_rk45`, Stan's adaptive Runge-Kutta-Fehlberg (4,5) solver, with subsequent lines supplying the full argument list.
- Stan advances the solution by estimating

$$
R_j(t_{n+1}) = R_j(t_n) + \int_{t_n}^{t_{n+1}} f\bigl(s, R_j(s), \theta\bigr)\,ds,
$$

using embedded 4th- and 5th-order polynomials to bound the local truncation error. Step sizes shrink automatically when `stim_j(t)` changes rapidly (early after dosing) and grow when the system settles, ensuring stability without manually tuning step lengths.
- Reverse-mode automatic differentiation threads through every integration step so HMC learns `dR_j(t_jk)/dtheta` alongside the trajectory.
- The infinitesimal offset at `stand_ode.stan:67` avoids numerical issues when the first observation coincides with the initial time, a standard Stan idiom for ODE models.

## Model and generated quantities

- `stand_ode.stan:82` encodes the prior `k_in ~ Normal(0.05, 0.02)`.
- `stand_ode.stan:83` encodes the prior `k_out ~ Normal(0.04, 0.01)`.
- `stand_ode.stan:84` encodes the prior `emax0 ~ Normal(0.8, 0.3)`.
- `stand_ode.stan:85` encodes the prior `lec50 ~ Normal(log(6), 0.5)`.
- `stand_ode.stan:86` encodes the half-normal prior `sigma ~ Normal^+(0, sigma_prior_scale)`.
- `stand_ode.stan:87` encodes the prior `R0 ~ Normal(0, 1)`.
- `stand_ode.stan:89` encodes the observation model `bcva_obs ~ Normal(mu_bcva, sigma)`, matching the Gaussian noise assumption in PKPD Meta 103 once logits are mapped back to letters.
- `stand_ode.stan:92` allocates `bcva_rep`.
- `stand_ode.stan:93` opens the posterior predictive loop.
- `stand_ode.stan:94` draws posterior predictive samples via `normal_rng(mu_bcva[n], sigma)` for each visit.
- `stand_ode.stan:95` stores the replicated acuity so posterior predictive checks can compare the fitted model to observed trajectories.
- `stand_ode.stan:96` closes the loop.

`bcva_rep` mirrors the observed scale and allows replicated trajectories to be compared against data--exactly the diagnostic PKPD Meta 103 recommends before trusting the model's extrapolations.

---

Taken together, `stand_ode.stan` is a faithful computational companion to the turnover derivation in PKPD Meta 103: the ODE block encodes the linear loss and saturating drug stimulus, the transformed parameters perform the logit-to-letters conversion that keeps BCVA bounded, and the likelihood completes the Bayesian update that underpins the Gelman et al. evidence synthesis.
