+++
title = "πVAE: Encoding stochastic process priors with variational autoencoders"
description = "Hugo, the world's fastest framework for building websites"
date = "2019-02-28"
aliases = ["about-us", "about-hugo", "contact"]
author = "Hugo Authors"
math = true
+++

by Swapnil Mishra, Seth Flaxman, Samir Bhatt


___
[Abstract](https://arxiv.org/abs/2002.06873v6) 

 
Stochastic processes provide a mathematically elegant way model complex data. In theory, they provide flexible priors over function classes that can encode a wide range of interesting assumptions. In practice, however, efficient inference by optimisation or marginalisation is difficult, a problem further exacerbated with big data and high dimensional input spaces. We propose a novel variational autoencoder (VAE) called the prior encoding variational autoencoder (πVAE). The πVAE is finitely exchangeable and Kolmogorov consistent, and thus is a continuous stochastic process. We use πVAE to learn low dimensional embeddings of function classes. We show that our framework can accurately learn expressive function classes such as Gaussian processes, but also properties of functions to enable statistical inference (such as the integral of a log Gaussian process). For popular tasks, such as spatial interpolation, πVAE achieves state-of-the-art performance both in terms of accuracy and computational efficiency. Perhaps most usefully, we demonstrate that the low dimensional independently distributed latent space representation learnt provides an elegant and scalable means of performing Bayesian inference for stochastic processes within probabilistic programming languages such as Stan.
***
[GitHub](https://github.com/lukasadam/piVAE).
____
$$[z_\mu,z_\sigma]^T = e(\eta_e,x)$$