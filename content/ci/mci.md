+++
title = "The Blessings of Multiple Causes"
#slug = "contact"
date = "2019-10-03"
math = true
+++

by Yixin Wang & David M. Blei
___
**Keywords**: Causal inference, Machine learning, Probabilistic models, Unconfoundedness, Unobserved confounding
___
**[Abstract](https://doi.org/10.1080/01621459.2019.1686987)**

Causal inference from observational data is a vital problem, but it comes with strong assumptions. Most methods assume that we observe all confounders, variables that affect both the causal variables and the outcome variables. This assumption is standard but it is also untestable. In this article, we develop the deconfounder, a way to do causal inference with weaker assumptions than the traditional methods require. The deconfounder is designed for problems of multiple causal inference: scientific studies that involve multiple causes whose effects are simultaneously of interest. Specifically, the deconfounder combines unsupervised machine learning and predictive model checking to use the dependencies among multiple causes as indirect evidence for some of the unobserved confounders. We develop the deconfounder algorithm, prove that it is unbiased, and show that it requires weaker assumptions than traditional causal inference. We analyze its performance in three types of studies: semi-simulated data around smoking and lung cancer, semi-simulated data around genome-wide association studies, and a real dataset about actors and movie revenue. The deconfounder is an effective approach to estimating causal effects in problems of multiple causal inference. Supplementary materials for this article are available online.
___
[**Github**](https://github.com/blei-lab/deconfounder_tutorial)
___

[**Video**](https://www.microsoft.com/en-us/research/video/the-blessings-of-multiple-causes/)
___