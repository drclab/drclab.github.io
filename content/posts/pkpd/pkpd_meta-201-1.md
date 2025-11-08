+++
title = "PKPD_META 201.1: Multivariate Normal Likelihood for Summary Means"
type = "post"
date = "2025-12-06"
math = true
draft = true
summary = "Work the PKPD_META 201 average-data likelihood into explicit multivariate normal form, including the mean vector, covariance construction, and matrix identities needed for Stan."
tags = ["pkpd", "meta"]
+++

This follow-up to [PKPD_META 201](../pkpd_meta-201/) zooms in on the multivariate normal block that carries the external visit-level averages. The simulation-based Laplace approximation still supplies the moments, but we now write down the exact matrix expressions so the likelihood becomes transparent, auditable, and easily ported to Stan or NumPyro.

## Average vector and generative map

Let the external summary report provide $\bar{y}_{t_k}$ at visit times $t_1, \ldots, t_K$ for a cohort of size $\bar{J}$. Stack the means into the column vector

$$
\bar{y} = \begin{bmatrix}
\bar{y}_{t_1} \\
\vdots \\
\bar{y}_{t_K}
\end{bmatrix} \in \mathbb{R}^K.
$$

For each patient $j$ in a hypothetical replicate cohort we compute model-implied responses $y_{jk} = g(t_k, \alpha_j, \bar{\phi})$, where $\bar{\phi} = \phi + \delta$ applies the study-specific shift from PKPD_META 201. Within the simulator, the patient averages are

$$
\tilde{y}_k = \frac{1}{\tilde{J}} \sum_{j=1}^{\tilde{J}} y_{jk}, \qquad
\tilde{M} = \begin{bmatrix}
\tilde{y}_1 \\ \vdots \\ \tilde{y}_K
\end{bmatrix}.
$$

Conditioning on $\bar{\phi}$, the expectation of the published averages equals $\operatorname{E}[\bar{y} \mid \bar{\phi}] = \mu(\bar{\phi})$, which we estimate with $\tilde{M}$. All the structure of the follow-up lies in writing down the covariance of $\bar{y}$.

## From patient heterogeneity to $\Sigma(\bar{\phi})$

Denote the patient-level random effects by $\alpha_j \sim \mathcal{N}(A \bar{\phi}, \Sigma_\alpha)$ and the residual measurement noise by $\epsilon_{jk} \sim \mathcal{N}(0, \sigma_y^2)$, independent across $j$ and $k$. Linearizing $g$ around $\mu_\alpha = A \bar{\phi}$ produces the Jacobian matrix

$$
G(\bar{\phi}) = \begin{bmatrix}
\frac{\partial g(t_1, \mu_\alpha, \bar{\phi})}{\partial \alpha^\top} \\
\vdots \\
\frac{\partial g(t_K, \mu_\alpha, \bar{\phi})}{\partial \alpha^\top}
\end{bmatrix} \in \mathbb{R}^{K \times d_\alpha}.
$$

The delta method then gives the visit-level covariance for a single patient:

$$
\Sigma_{\text{patient}}(\bar{\phi}) = G(\bar{\phi}) \Sigma_\alpha G(\bar{\phi})^\top + \sigma_y^2 I_K.
$$

Averaging $\bar{J}$ independent patients scales this covariance, producing the exact covariance of the reported averages:

$$
V(\bar{\phi}) = \frac{1}{\bar{J}} \Sigma_{\text{patient}}(\bar{\phi}).
$$

In practice we substitute the simulation-based estimate $\tilde{\Sigma} = \operatorname{Cov}(\tilde{M})$ for $\Sigma_{\text{patient}}(\bar{\phi})$ and divide by $\bar{J}$, matching the five-step recipe in PKPD_META 201. The follow-up post simply makes the mapping explicit:

$$
V(\bar{\phi}) \approx \frac{1}{\bar{J}} \tilde{\Sigma}.
$$

## Log-density in explicit matrix form

Once the Moment estimates $(\tilde{M}, \tilde{\Sigma})$ are in hand, the log-density contribution for the external averages is the multivariate normal

$$
\begin{align}
\log \mathcal{N}_K(\bar{y} \mid \tilde{M}, \tfrac{1}{\bar{J}} \tilde{\Sigma})
&= -\frac{K}{2} \log(2\pi) \\
&\quad - \frac{1}{2} \log \left| \tfrac{1}{\bar{J}} \tilde{\Sigma} \right| \\
&\quad - \frac{1}{2} (\bar{y} - \tilde{M})^\top \left( \tfrac{1}{\bar{J}} \tilde{\Sigma} \right)^{-1} (\bar{y} - \tilde{M}).
\end{align}
$$

Stan favors the Cholesky parameterization, so we compute the lower-triangular factor $L$ such that $\tilde{\Sigma} = L L^\top$. Because scaling by $\bar{J}^{-1}$ only rescales the diagonal of $L$, we can reuse the factorization:

$$
\tfrac{1}{\bar{J}} \tilde{\Sigma} = \left( \frac{1}{\sqrt{\bar{J}}} L \right) \left( \frac{1}{\sqrt{\bar{J}}} L \right)^\top.
$$

With $L_{\text{scaled}} = L / \sqrt{\bar{J}}$, the Stan increment is a one-liner:

```stan
target += multi_normal_cholesky_lpdf(bar_y | tilde_M, L_scaled);
```

This accords with Algorithm 2 in PKPD_META 201 and makes plain how the chained simulation feeds directly into the analytic form of the multivariate normal density.

## Practical checkpoints

- **Positive definiteness:** The Monte Carlo covariance $\tilde{\Sigma}$ should be symmetric positive definite; if simulations are noisy, apply a jitter $\epsilon I_K$ before the Cholesky factorization and record $\epsilon$ for reproducibility.
- **Moment diagnostics:** Compare $\tilde{M}$ against the raw visit means from a larger simulation batch (e.g., $\tilde{J} = 2,000$) to confirm the linearization and delta-method intuition hold.
- **Sensitivity to $\bar{J}$:** Because the covariance shrinks with $\bar{J}^{-1}$, the log-density sharpens rapidly as the external cohort grows. When $\bar{J}$ is uncertain, propagate it as another random variable and integrate numerically to avoid overconfident pulls.
- **Link to priors on $\delta$:** The prior variance on $\delta$ determines how far $\tilde{M}$ can drift from the internal predictions. Keeping the multivariate normal form explicit makes it easier to tune the hierarchy so that the summary data neither overwhelms nor under-utilizes the internal evidence.

These matrices are the only missing pieces from PKPD_META 201. Writing them out closes the loop between the simulator and the analytic likelihood, ensuring the meta-analysis stays interpretable even as we iterate on the external data feed.
