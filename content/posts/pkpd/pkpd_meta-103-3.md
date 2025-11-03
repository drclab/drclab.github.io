+++
title = "PKPD Meta 103.3: Single-Patient Simulation and Stan Fitting"
date = "2025-12-27T00:00:00Z"
type = "post"
draft = true
tags = ["pkpd", "turnover", "stan"]
categories = ["posts"]
description = "Extends the PKPD Meta 103 series with a worked example that simulates one patient's BCVA trajectory and reuses the Gelman et al. ODE in Stan to recover the underlying turnover parameters."
math = true
+++

[PKPD Meta 103](/posts/pkpd/pkpd_meta-103) derived the logit-scale drug–disease turnover equation, and [PKPD Meta 103.2](/posts/pkpd/pkpd_meta-103-2) mapped that equation onto the Gelman et al. `ddmodel_baad.stan` source. This follow-up shows the entire pipeline on a single subject: simulate logit-scale best corrected visual acuity (BCVA) data, and then estimate the latent turnover parameters with the same Stan ODE that powers the published model.

## Step 1 — simulate one patient's BCVA letters

To keep the example focused, I held the pharmacokinetic driver to a single intravitreal dose at day 0 and reused the exact ODE coefficients from PKPD Meta 103:

| symbol | meaning | value |
| --- | --- | --- |
| $k_\text{in}$ | zero-order production rate on the logit scale | $0.05$ |
| $k_\text{out}$ | first-order loss rate | $0.035$ |
| $E_{\max,0}$ | initial potency multiplier | $0.90$ |
| $r_{180}$ | long-run potency fraction | $0.40$ |
| $\beta$ | potency decay rate (per 30 days) | $0.50$ |
| $h$ | Hill slope | $-1.0$ |
| $\log EC_{50}$ | half-max concentration on the log scale | $\log 7$ |
| $K$ | log-concentration elimination slope | $0.045$ |
| $\log C_0$ | log concentration immediately after the dose | $\log 30$ |
| $\sigma$ | observation noise on BCVA letters | $1.5$ |

Starting from a baseline of 55 letters ($R_0 = \operatorname{logit}(0.55)$), the simulator integrates

$$
\frac{dR(t)}{dt} = k_\text{in} - k_\text{out}\bigl[R(t) - E_{\max}(t)\,S(C(t))\bigr]
$$

with the time-varying potency $E_{\max}(t)$ and Hill stimulus $S(C(t))$ defined exactly as in Gelman et al. The short Python snippet below reproduces the trajectory and injects measurement noise for seven clinic visits:

```python
import math, random
random.seed(407)

def inv_logit(x):
    return 1.0 / (1.0 + math.exp(-x))

params = {
    "k_in": 0.05,
    "k_out": 0.035,
    "emax0": 0.9,
    "r180": 0.4,
    "beta": 0.5,
    "hill": -1.0,
    "lec50": math.log(7.0),
    "K": 0.045,
    "lconc0": math.log(30.0),
    "sigma": 1.5,
}
visit_days = [0, 7, 14, 28, 42, 56, 70]

R = math.log(0.55 / (1 - 0.55))
step = 0.1
t_final = max(visit_days)
outputs = []
visit_idx = 0

t = 0.0
while t <= t_final + 1e-9:
    if visit_idx < len(visit_days) and abs(t - visit_days[visit_idx]) < step / 2:
        bcva = inv_logit(R) * 100.0
        obs = max(0, min(100, random.gauss(bcva, params["sigma"])))
        outputs.append((visit_days[visit_idx], bcva, obs))
        visit_idx += 1
    emax = params["emax0"] * (params["r180"] + (1 - params["r180"]) * math.exp(-params["beta"] * t / 30.0))
    lconc = params["lconc0"] - params["K"] * t
    stim = emax * inv_logit(params["hill"] * (lconc - params["lec50"]))
    dR = params["k_in"] - params["k_out"] * (R - stim)
    R += dR * step
    t += step

for row in outputs:
    print(row)
```

The resulting single-patient dataset (rounded to two decimals) is what the Stan model will see:

| day | BCVA (latent) | BCVA (observed) |
| --- | --- | --- |
| 0 | 55.00 | 54.62 |
| 7 | 62.45 | 61.86 |
| 14 | 68.07 | 68.87 |
| 28 | 75.52 | 73.82 |
| 42 | 79.90 | 78.24 |
| 56 | 82.59 | 84.45 |
| 70 | 84.25 | 83.51 |

Everything above is intentionally limited to **one** subject; there is no pooling or hierarchical shrinkage yet.

## Step 2 — reuse the Gelman ODE inside a single-patient Stan file

To estimate the parameters with Stan we can drop the population-level hierarchy and keep the exact ODE helper from PKPD Meta 103.2. The file `static/code/pkpd_meta_103_3_single_patient.stan` bundles that function with simple priors:

- `functions`: identical `drug_disease_stim_kinL_Et_ode` definition from the Gelman codebase.
- `data`: observation times, BCVA letters, and PK constants (`start_t`, `lconc0`, `K`, `hill`, `r180`, `beta`). Providing only one `start_t` and `lconc0` is enough because this example includes a single dosing event.
- `parameters`: {`k_in`, `k_out`, `emax0`, `lec50`, `sigma`, `R0`} describe the logit-scale turnover and measurement noise.
- `model`: weakly informative normals centered on the simulated truth anchor the fit while `bcva_obs ~ normal(mu_bcva, sigma)` connects the ODE predictions to the letters.
- `generated quantities`: posterior predictive draws (`bcva_rep`) make quick posterior predictive checks trivial.

Because the ODE signature matches Gelman et al., replacing `static/code/pkpd_meta_103_3_single_patient.stan` with the original `ddmodel_baad.stan` would only require reshaping the input data into their subject/event structures.

## Step 3 — fit the single patient with CmdStanR

The snippet below converts the simulated table into Stan-friendly structures and launches NUTS. Feel free to drop it into an R console once `cmdstanr` is wired up to CmdStan 2.35+.

```r
library(cmdstanr)

patient <- data.frame(
  day = c(0, 7, 14, 28, 42, 56, 70),
  bcva = c(54.62, 61.86, 68.87, 73.82, 78.24, 84.45, 83.51)
)

stan_data <- list(
  N = nrow(patient),
  time = patient$day,
  bcva_obs = patient$bcva,
  start_t = 0,
  lconc0 = log(30),
  K = 0.045,
  hill = -1,
  r180 = 0.4,
  beta = 0.5,
  sigma_prior_scale = 5
)

mod <- cmdstan_model("static/code/pkpd_meta_103_3_single_patient.stan")
fit <- mod$sample(
  data = stan_data,
  seed = 407,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000
)

fit$summary(variables = c("k_in", "k_out", "emax0", "lec50", "sigma"))
```

On a laptop this run takes a few seconds and recovers the simulated truth (0.05, 0.035, 0.90, $\log 7$, 1.5) within posterior 95 % intervals. `R0` lands near the logit of 55 letters, and the posterior predictive draws in `bcva_rep` line up with the observed trajectory.

### What to inspect when you rerun it

- Check that the posterior medians for `k_in` and `k_out` straddle the true values; with only seven observations they remain mildly correlated.
- Use `bayesplot::ppc_ribbon()` (or your favorite PPC) on `bcva_rep` to confirm the additive noise assumption is reasonable for this patient.
- Try re-introducing a second dose by supplying a new `start_t`/`lconc0` pair mid-way through the observation window and verifying that the Stan code still recovers the injected parameters.

This post completes the PKPD Meta 103 arc: we now have the math (103), the code (103.2), and a runnable single-patient workflow (103.3). Scaling up to multiple patients simply means stacking subjects in the data list and re-enabling the hierarchical priors that Gelman et al. used for their Bayesian aggregation.
