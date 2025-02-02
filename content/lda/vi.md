+++
title = " Variational Inference: A Review for Statisticians"
#slug = "contact"
date = "2017-07-13"
math = true
+++

by David M. Blei, Alp Kucukelbir & Jon D. McAuliffe
___
**Keywords**: AlgorithmsComputationally intensive methodsStatistical computing
___
**[Abstract](https://doi.org/10.1080/01621459.2017.1285773)**

One of the core problems of modern statistics is to approximate difficult-to-compute probability densities. This problem is especially important in Bayesian statistics, which frames all inference about unknown quantities as a calculation involving the posterior density. In this article, we review variational inference (VI), a method from machine learning that approximates probability densities through optimization. VI has been used in many applications and tends to be faster than classical methods, such as Markov chain Monte Carlo sampling. The idea behind VI is to first posit a family of densities and then to find a member of that family which is close to the target density. Closeness is measured by Kullback–Leibler divergence. We review the ideas behind mean-field variational inference, discuss the special case of VI applied to exponential family models, present a full example with a Bayesian mixture of Gaussians, and derive a variant that uses stochastic optimization to scale up to massive data. We discuss modern research in VI and highlight important open problems. VI is powerful, but it is not yet well understood. Our hope in writing this article is to catalyze statistical research on this class of algorithms. Supplementary materials for this article are available online.
 
 
___
