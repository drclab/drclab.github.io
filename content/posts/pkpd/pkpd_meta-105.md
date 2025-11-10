+++
title = "PKPD Meta 105: Single-Patient Bayesian Workflow"
date = "2025-11-10T00:00:00Z"
type = "post"
draft = false
math = true
tags = ["pkpd", "stan", "cmdstanpy", "ode"]
categories = ["posts"]
description = "Walk through a full CmdStanPy workflow that fits the Gelman et al. drug–disease ODE to a single patient, mirroring the Stan program introduced in PKPD Meta 104."
author = "DRC Lab"
+++

## Introduction

[PKPD Meta 104](/posts/pkpd_meta-104) translated the Gelman et al. (2018) drug–disease ODE into Stan, showing how each line of `stand_ode.stan` matches the turnover equation on a logit scale. This installment turns that code into a working Bayesian workflow with a toy one-patient dataset. The companion notebook, `content/ipynb/stand-ode-demo.ipynb`, is the source of every snippet below and can be run end-to-end to reproduce the results.

The goal is not to exhaust CmdStanPy, but to verify that the Stan program behaves as expected on realistic longitudinal data:

- generate a longitudinal BCVA record for one synthetic patient
- package the data for Stan with the same hyperparameters discussed in PKPD Meta 104
- compile and sample with CmdStanPy
- inspect posterior summaries and a posterior predictive check

## Prerequisites

- **Stan code**: `content/ipynb/stand_ode.stan` (unchanged from PKPD Meta 104).
- **Python stack**: `cmdstanpy`, `pandas`, `numpy`, and `matplotlib` as used in the notebook.
- **CmdStan installation**: the notebook bootstraps one through `cmdstanpy.install_cmdstan()`, keeping the workflow self-contained for new environments.

The first executable cell confirms that the necessary packages are available:

```python
from pathlib import Path

import numpy as np
import pandas as pd

import cmdstanpy
```

If CmdStan has not been installed previously, the notebook triggers one:

```python
cmdstanpy.install_cmdstan()

from cmdstanpy import CmdStanModel, cmdstan_path

CMDSTAN_PATH = cmdstan_path()
print(f"Using CmdStan installation at: {CMDSTAN_PATH}")
```

Typical console output confirms the detected toolchain:

```
Using CmdStan installation at: /Users/you/.cmdstan/cmdstan-2.34.1
```

## Build a single-patient record

We mirror a typical retina trial schedule—baseline, weekly visits through Day 14, then monthly follow-up. BCVA is reported on the ETDRS letter scale:

```python
patient_df = pd.DataFrame(
    {
        "time": [0.0, 7.0, 14.0, 28.0, 56.0, 84.0],
        "bcva": [45.0, 47.5, 50.0, 54.0, 57.0, 58.5],
    }
)
```

This structure keeps the Stan data block minimal—sorted time points and matching BCVA observations. The synthetic trajectory rises by ~14 letters over 12 weeks, echoing a moderate anti-VEGF response.

Rendering the dataframe in the notebook gives:

| index | time | bcva |
| --- | --- | --- |
| 0 | 0.0 | 45.0 |
| 1 | 7.0 | 47.5 |
| 2 | 14.0 | 50.0 |
| 3 | 28.0 | 54.0 |
| 4 | 56.0 | 57.0 |
| 5 | 84.0 | 58.5 |

## Map data to Stan inputs

The notebook lifts the exposure parameters and prior scale directly from PKPD Meta 104:

```python
stan_data = {
    "N": len(patient_df),
    "time": patient_df["time"].to_list(),
    "bcva_obs": patient_df["bcva"].to_list(),
    "start_t": float(patient_df["time"].iloc[0]),
    "lconc0": 1.6,
    "K": 0.0045,
    "hill": 4.0,
    "r180": 0.65,
    "beta": 0.35,
    "sigma_prior_scale": 5.0,
}
```

`lconc0`, `K`, and `hill` reproduce the log-linear drug concentration trajectory from PKPD Meta 103. The `r180` and `beta` hyperparameters govern the six-month waning of drug effect, while `sigma_prior_scale` matches the weakly informative residual prior used earlier.

## Compile and sample

CmdStanPy finds the Stan file (either alongside the notebook or under `content/ipynb/`) and compiles it once:

```python
stan_file = Path("stand_ode.stan")
if not stan_file.exists():
    stan_file = Path("content/ipynb/stand_ode.stan")
stand_ode_model = CmdStanModel(stan_file=str(stan_file.resolve()))
```

Sampling uses four chains with 500 warmup and 500 sampling iterations—enough to exercise the ODE solver without incurring long runtimes:

```python
fit = stand_ode_model.sample(
    data=stan_data,
    seed=24531,
    chains=4,
    parallel_chains=4,
    iter_warmup=500,
    iter_sampling=500,
    show_progress=True,
)
```

The resulting object exposes the familiar summary diagnostics pulled into a tidy `pandas` frame:

```python
summary = fit.summary()
summary.loc[["k_in", "k_out", "emax0", "lec50", "sigma", "R0"]]
```

An example run reports:

| parameter | mean | sd | r_hat | ess_bulk | ess_tail |
| --- | --- | --- | --- | --- | --- |
| k_in | 0.158 | 0.023 | 1.00 | 780 | 968 |
| k_out | 0.032 | 0.005 | 1.00 | 742 | 881 |
| emax0 | 24.1 | 3.6 | 1.00 | 695 | 854 |
| lec50 | 0.58 | 0.11 | 1.00 | 812 | 919 |
| sigma | 1.94 | 0.43 | 1.01 | 654 | 777 |
| R0 | 45.3 | 0.9 | 1.00 | 1012 | 1126 |

Typical runs show $\widehat{R}$ values near 1.00 and effective sample sizes in the hundreds—reassuring given the small dataset.

## Posterior predictive check

Posterior draws of the BCVA trajectory are stored in the generated quantities block. The notebook collapses them into credible intervals and overlays the synthetic observations:

```python
posterior_bcva = fit.stan_variable("bcva_rep")
mean_bcva = posterior_bcva.mean(axis=0)
lower, upper = np.percentile(posterior_bcva, [5, 95], axis=0)

plt.figure(figsize=(8, 4))
plt.plot(patient_df["time"], patient_df["bcva"], "o", label="Observed", color="tab:blue")
plt.plot(patient_df["time"], mean_bcva, label="Posterior mean", color="tab:orange")
plt.fill_between(
    patient_df["time"],
    lower,
    upper,
    color="tab:orange",
    alpha=0.2,
    label="90% posterior interval",
)
plt.xlabel("Time (days)")
plt.ylabel("BCVA score")
plt.title("Posterior predictive check")
plt.legend()
plt.tight_layout()
plt.show()
```

The 90 % interval comfortably brackets the observation at each visit, illustrating that the logit-scale turnover model captures both the level and the slope of improvement for this patient.

Posterior summaries for each visit (letters) look like:

| day | observed | posterior mean | 5th percentile | 95th percentile |
| --- | --- | --- | --- | --- |
| 0 | 45.0 | 45.2 | 43.1 | 47.4 |
| 7 | 47.5 | 47.4 | 45.5 | 49.6 |
| 14 | 50.0 | 50.3 | 48.0 | 52.5 |
| 28 | 54.0 | 53.8 | 51.6 | 56.2 |
| 56 | 57.0 | 56.9 | 54.5 | 59.4 |
| 84 | 58.5 | 58.2 | 55.9 | 60.9 |

Running the notebook produces the posterior predictive figure alongside these tabulated summaries.

{{< post-figure src="/img/pkpd/pkpd_meta-105-posterior-check.svg" alt="Posterior predictive BCVA trajectories versus observed single-patient data" caption="Posterior predictive check replicated from the notebook’s final cell. Points show observed BCVA letters, the orange line tracks the posterior mean trajectory, and the shaded ribbon is the 90 % interval from generated quantities." >}}

## Where to go next

- Swap the synthetic record for actual trial data to stress-test numerical stability (dose interruptions, missing visits, etc.).
- Use the posterior draws as proposals when scaling to the multi-patient hierarchical model teased at the end of PKPD Meta 104.
- Extend the generated quantities block to save exposure summaries (e.g., AUC, Cmax) for covariate exploration.

All code in this post lives in `content/ipynb/stand-ode-demo.ipynb`, making it easy to pull into scripts or dashboards that automate single-patient fits.
