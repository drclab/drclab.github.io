+++
title = "PKPD Meta 103.2: Reading the Gelman et al. Stan Code"
date = "2025-12-20T00:00:00Z"
type = "post"
draft = true
tags = ["pkpd", "turnover", "stan"]
categories = ["posts"]
description = "Walks through the ddmodel_baad.stan implementation that powers the Gelman et al. (2018) drug–disease turnover model, highlighting how the ODE and Hill stimulus map directly onto the published specification."
math = true
+++

The PKPD Meta series now has a canonical structural description for the Gelman et al. (2018) drug–disease model, but to _reuse_ that model we need to see how it is encoded. The official Stan source lives in [wds15/baad](https://github.com/wds15/baad), a public repository that reproduces the Bayesian aggregation workflow. This companion note connects the math in PKPD Meta 103 to the concrete Stan program, `ddmodel_baad.stan`, and shows how the code implements both the modified turnover equation and the logit-based Hill function.

## Where the ODE lives

The pharmacodynamic right-hand side is implemented in the Stan function `drug_disease_stim_kinL_Et_ode`. In the repository checkout, the relevant lines are:

$$
\frac{dR_j(t)}{dt} = k_{j}^{\text{in}} - k_{j}^{\text{out}} \left[R_j(t) - E_{\max,j} S_j\!\bigl(C_j(t)\bigr)\right],
$$

which is the same drug–disease turnover equation derived in [PKPD Meta 103](/posts/pkpd/pkpd_meta-103). The Stan code below evaluates that expression numerically.

{{< highlight stan "hl_lines=8 9 10 11 18 19 20 23" >}}
real[] drug_disease_stim_kinL_Et_ode(real t,
                                     real[] y,
                                     real[] theta,
                                     real[] x_r,
                                     int[] x_i) {
  real dydt[1];
  real y_pred;

  real emax_0 = theta[1];
  real lec_50 = theta[2];
  real r180   = theta[3];
  real beta   = theta[4];
  real k_in   = theta[5];
  real k_out  = theta[6];

  real start_t = x_r[1];
  real lconc_0 = x_r[2];
  real K       = x_r[3];
  real hill    = x_r[4];

  real lconcentration = lconc_0 - K * (t - start_t);
  real emax = emax_0 * (r180 + (1 - r180) * exp(-beta * t / 30.));
real stim = emax * inv_logit(hill * (lconcentration - lec_50));

  y_pred = y[1];
  dydt[1] = k_in - k_out * (y_pred - stim);
  return dydt;
}
{{< /highlight >}}

Line by line, this matches equation (1) from the paper:

- `y[1]` corresponds to the logit-scale visual acuity state $R_j(t)$.
- `k_in` and `k_out` play the same roles as in the write-up, producing the linear turnover term $k_{j}^{\text{in}} - k_{j}^{\text{out}} R_j(t)$.
- `stim` is the additive drug driver $E_{\max,j} S_j(C_j(t))$. The code pre-computes time-varying potency (`emax`) and the Hill modulation (`inv_logit(…)`) before subtracting the product inside the parenthesis, so the derivative becomes $k_{j}^{\text{in}} - k_{j}^{\text{out}}[R_j(t) - E_{\max,j} S_j(\cdot)]$.

Switching the sign inside the parentheses is purely algebraic: the Stan code multiplies the entire bracket by $k_{j}^{\text{out}}$, whereas the paper expands to $k_{j}^{\text{in}} - k_{j}^{\text{out}}R_j(t) - k_{j}^{\text{out}}E_{\max,j}S_j(\cdot)$. Both forms are equivalent.

### Translating the scalar helper lines

The three helper lines highlighted in the Stan block map directly onto closed-form expressions:

1. **Log-concentration trajectory.**
   $$
   \ell C(t) = \ell C_0 - K\bigl(t - t_{\text{start}}\bigr),
   $$
   where $\ell C(t)$ is the current log drug concentration, $\ell C_0$ is the log concentration at the most recent dosing event, $K$ is the elimination slope (per unit time), and $t_{\text{start}}$ stores the dosing time origin. The expression simply evolves the log concentration under a first-order elimination process, so every unit of time reduces the log concentration by $K$.

2. **Time-varying potency.**
   $$
   E_{\max}(t) = E_{\max,0}\Bigl[r_{180} + \bigl(1 - r_{180}\bigr)\exp\!\Bigl(-\frac{\beta t}{30}\Bigr)\Bigr].
   $$
   Here $E_{\max,0}$ is the initial maximum effect, $r_{180}$ is the asymptotic fraction that remains after 180 days, and $\beta$ controls how quickly that floor is approached. The exponential term decays toward zero, so $E_{\max}(t)$ transitions smoothly from $E_{\max,0}$ at $t=0$ to $E_{\max,0}\,r_{180}$ as $t$ grows large, matching the waning drug potency described in Gelman et al.

3. **Hill stimulus on the logit scale.**
   $$
   \text{stim}(t) = E_{\max}(t)\,\operatorname{logit}^{-1}\!\Bigl(h_{\text{ill}}\bigl[\ell C(t) - \ell EC_{50}\bigr]\Bigr),
   $$
   where $h_{\text{ill}}$ is the Hill slope and $\ell EC_{50}$ packages the half-maximal concentration on the log scale. The inverse-logit term converts the scaled log concentration difference into the familiar $\left(0,1\right)$ saturation signal, and multiplying by $E_{\max}(t)$ re-expresses the result on the same scale as the turnover stimulus. In practice, this term grows monotonically with $\ell C(t)$ when $h_{\text{ill}} > 0$ (or decreases when $h_{\text{ill}} < 0$), capturing the sigmoidal exposure–response relationship that feeds the ODE.

## Hill function exactly as published

The Hill sensitivity is encoded on the log concentration scale:

```stan
real lconcentration = lconc_0 - K * (t - start_t);
real stim = emax * inv_logit(hill * (lconcentration - lec_50));
```

Within the Stan data pre-processing, `lconc_0` stores $\log C_j(t)$ at the last dosing event, `K` is the elimination slope, and `lec_50` contains $\log EC_{50,j}$. Plugging those definitions into the stimulus line yields

$$
S_j(C_j(t)) = \operatorname{logit}^{-1}\bigl(h_j(\log C_j(t) - \log EC_{50,j})\bigr),
$$

which is algebraically identical to the paper’s $\operatorname{logit}^{-1}(\log EC_{50,j} - \log C_j(t))$ once the user-specific Hill slope $h_j$ is allowed to take negative values. The repository sets `hill` to $-1$ in the default configuration so the argument flips sign, reproducing the exact formulation quoted on p. 1588.

## Putting the pieces together

The remainder of `ddmodel_baad.stan` wires these building blocks into the hierarchical model:

- The helper `drug_disease_stim_kinL_Et` iteratively calls the ODE to propagate each subject’s trajectory between dosing events, mirroring the simulation script described in PKPD Meta 103.
- Baseline constraints (`valogit`, `inv_valogit`) enforce that BCVA predictions map cleanly between the 0–100 letter scale and the real line.
- The Stan parameters (`Lalpha_0`, `lkout`, `lEC50`, `LEmax_0`, …) correspond exactly to the symbols defined in the article, making it straightforward to cross-reference estimates between the text and the code.

For readers only interested in the structural translation, the key takeaway is that the published model is implemented without approximations: the repository integrates the same ODE derived in PKPD Meta 103 and applies the same Hill transform the authors describe. Cloning `wds15/baad` gives you a working Stan program that can be plugged into new data or extended with alternative priors, confident that the core dynamics stay faithful to the original paper.

Next in the series we will build a minimal CmdStan script around this model so you can run the drug–disease system locally and inspect simulated trajectories straight from the Stan output.
