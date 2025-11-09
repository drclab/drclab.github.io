+++
title = "Variational Inference 101: KL Divergence"
slug = "variational-inference-101-kl"
date = "2026-11-09T16:26:32Z"
type = "post"
draft = true
math = true
tags = ["variational-inference", "kl-divergence", "mcmc"]
categories = ["posts"]
description = "A beginner-friendly tour of KL divergence, why it powers variational inference, and how the ELBO connects to Bayesian targets."
+++

Variational inference (VI) trades exact sampling for optimization: we pick a family of tractable distributions $q_\lambda(z)$ and tune the parameters $\lambda$ so that $q_\lambda$ sits close to the true posterior $p(z \mid x)$. The *distance* that tells us what "close" means is almost always the Kullback–Leibler (KL) divergence. This primer pauses the usual algebra to explain what the KL really measures, why VI prefers the *reverse* KL, and how the evidence lower bound (ELBO) falls out of one line of calculus.

## Learning goals

- Connect $KL(q \,\|\, p)$ to a simple intuition: a tax you pay whenever $q$ assigns probability to outcomes that $p$ thinks are implausible.
- Derive the ELBO from the log marginal likelihood and see how minimizing the KL is equivalent to maximizing the ELBO.
- Compare the *reverse* KL used in VI with the *forward* KL that shows up in expectation propagation and understand the consequences (mode-seeking vs. mass-covering).
- Practice estimating the KL and its gradients with Monte Carlo, the reparameterization trick, and the score-function estimator.
- Diagnose when a poor approximation stems from the KL choice versus from the variational family.

## 1. What is KL divergence?

For continuous latent variables $z$, the KL divergence from $q$ to $p$ is

$$
KL(q \,\|\, p) = \int q(z) \log \frac{q(z)}{p(z)} \, dz = \mathbb{E}_{q}[\log q(z) - \log p(z)].
$$

Key properties:

- $KL(q \,\|\, p) \ge 0$ with equality iff $q(z) = p(z)$ almost everywhere.
- It is asymmetric: swapping $p$ and $q$ changes the value, so you must state the direction.
- It ignores regions where $q(z) = 0$ even if $p(z) > 0$, making it forgiving when $q$ underestimates tails but harsh when $q$ allocates mass to impossible regions.

Think of $KL(q \,\|\, p)$ as the expected *log tax* paid when reality is governed by $q$ but you price events using $p$.

## 2. Why variational inference uses the reverse KL

In VI we minimize $KL(q_\lambda(z) \,\|\, p(z \mid x))$. Expanding the posterior with Bayes' rule:

$$
KL(q_\lambda \,\|\, p(z \mid x)) = \mathbb{E}_{q_\lambda}[\log q_\lambda(z) - \log p(z, x)] + \log p(x).
$$

The marginal likelihood $\log p(x)$ does not depend on $\lambda$, so minimizing the KL (left) is equivalent to minimizing the expectation term (right). Because the normalizing constant drops out, we can optimize even when $p(z \mid x)$ is known only up to proportionality.

The asymmetry is a feature:

- **Mode-seeking behavior**: Reverse KL pressures $q$ to avoid regions where $p(z \mid x)$ is small, but it is permissive about missing low-probability modes. This bias is acceptable when a single sharp explanation (MAP-like behavior) is enough.
- **Entropy preservation**: The $\mathbb{E}_{q}[\log q]$ term rewards higher entropy, keeping $q$ from collapsing unless the likelihood forces it.

Forward KL ($KL(p \,\|\, q)$) does the opposite: it punishes $q$ for missing modes (mass-covering) but tolerates assigning extra mass where $p$ has little. Expectation Propagation (EP) and some flow-based methods rely on this behavior instead.

## 3. Deriving the ELBO in one line

Start from the log evidence and add/subtract $\mathbb{E}_q[\log q(z)]$:

$$
\log p(x) = \log \int q(z) \frac{p(z, x)}{q(z)} \, dz \ge \int q(z) \log \frac{p(z, x)}{q(z)} \, dz = \mathcal{L}(\lambda).
$$

The inequality uses Jensen's inequality, giving the *evidence lower bound* (ELBO):

$$
\mathcal{L}(\lambda) = \mathbb{E}_{q_\lambda}[\log p(z, x)] - \mathbb{E}_{q_\lambda}[\log q_\lambda(z)].
$$

Rearranging yields

$$
KL(q_\lambda \,\|\, p(z \mid x)) = \log p(x) - \mathcal{L}(\lambda),
$$

so maximizing the ELBO is identical to minimizing the reverse KL. The decomposition also gives a sanity check: if you ever manage $\mathcal{L} = \log p(x)$, you have hit the true posterior.

## 4. Geometry: reverse vs. forward KL

Consider approximating a bimodal posterior with a single Gaussian. The two KL directions behave differently:

- Reverse KL picks one mode (mode seeking) because placing mass between the modes makes $\log p(z \mid x)$ tiny and the KL large.
- Forward KL centers between the modes (mass covering) because any missing mode contributes infinite penalty where $p>0$ but $q=0$.

This geometric view helps you decide when VI is appropriate. If your downstream decision only depends on a single plausible explanation, mode-seeking behavior is fine. If you must preserve multiple explanations (e.g., multi-modal predictive distributions), consider richer families (mixtures, normalizing flows) or alternative divergences (e.g., $\alpha$-divergences that interpolate between forward and reverse KL).

## 5. Estimating the ELBO and its gradients

In practice we approximate the ELBO with Monte Carlo samples:

$$
\widehat{\mathcal{L}}(\lambda) = \frac{1}{S} \sum_{s=1}^{S} \left[\log p(z^{(s)}, x) - \log q_\lambda(z^{(s)})\right], \quad z^{(s)} \sim q_\lambda.
$$

Two gradient estimators dominate:

- **Reparameterization trick**: Express $z^{(s)} = g_\lambda(\epsilon^{(s)})$ with $\epsilon \sim s(\epsilon)$ independent of $\lambda$. This gives low-variance gradients and enables autodiff-friendly code in JAX, PyTorch, or TensorFlow Probability.
- **Score-function (REINFORCE) estimator**: Works for discrete or implicit distributions, using $\nabla_\lambda \mathbb{E}_q[f(z)] = \mathbb{E}_q[f(z)\nabla_\lambda \log q_\lambda(z)]$. Needs variance reduction (baselines, control variates) to be practical.

```python
import jax
import jax.numpy as jnp

def elbo(lambda_params, rng_key, log_joint):
    """
    Estimate ELBO using the reparameterization trick.
    lambda_params: dict containing mean and log_std
    log_joint: function(z) -> log p(z, x)
    """
    eps = jax.random.normal(rng_key, shape=lambda_params["mean"].shape)
    z = lambda_params["mean"] + jnp.exp(lambda_params["log_std"]) * eps
    log_q = -0.5 * jnp.sum(eps**2 + jnp.log(2 * jnp.pi) + 2 * lambda_params["log_std"])
    return log_joint(z) - log_q
```

The code above sketches how a diagonal Gaussian variational family plugs into any differentiable `log_joint`. You can wrap it in `jax.vmap` to average over multiple samples per gradient step.

## 6. Diagnosing poor fits

When the ELBO stalls or predictive checks fail, ask:

- **Is the KL objective too mode-seeking?** Try a mixture of Gaussians or flows that can represent multiple spikes, or explore $\alpha$-divergences closer to forward KL.
- **Is the variational family too rigid?** Mean-field Gaussians ignore correlations; upgrading to a full-rank covariance or coupling layers often helps more than tweaking the optimizer.
- **Are gradients noisy?** Increase the number of Monte Carlo samples, add control variates, or switch to quasi-Monte Carlo draws.
- **Does the prior dominate?** Inspect $\log p(z, x)$: if the likelihood is flat relative to the prior, the KL may simply be telling you the data are not informative enough.

## 7. Where this meets MCMC

Variational approximations make excellent proposal distributions or mass matrices for Hamiltonian Monte Carlo. You can:

- Warm-start MCMC chains from $q_\lambda$ samples instead of random noise.
- Use the inverse covariance of $q_\lambda$ as a preconditioner for HMC, stabilizing step sizes.
- Alternate VI and MCMC (a "variational boost") to capture both global structure (via VI) and local refinements (via MCMC).

Framing the KL as the steering wheel of VI makes these hybrid strategies easier to reason about: once you know *what* the KL is optimizing, you can decide *when* to hand control back to sampling-based methods.

## Takeaways

- The KL divergence defines the target of VI; understanding its asymmetry explains the behavior of your approximations.
- The ELBO is just the log evidence minus the KL; maximize it and you implicitly minimize $KL(q \,\|\, p)$ without ever touching the intractable normalizer.
- Gradient estimators (reparameterization or score-function) translate the theory into runnable code; choose based on whether your latent variables are differentiable.
- Always align the divergence, variational family, and downstream task—KL minimization is powerful, but only when its mode-seeking bias matches your needs.
