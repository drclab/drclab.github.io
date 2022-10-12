+++
title = "Bayesian inference with Stan: A tutorial on adding custom distributions"
#slug = "contact"
date = "2016-06-03"
math = true
+++

by Annis, J., Miller, B.J. & Palmeri, T.J.
___
**Keywords**: Bayesian inference, Stan, Linear ballistic accumulator, Probabilistic programming
___
**[Abstract]()**

When evaluating cognitive models based on fits to observed data (or, really, any model that has free parameters), parameter estimation is critically important. Traditional techniques like hill climbing by minimizing or maximizing a fit statistic often result in point estimates. Bayesian approaches instead estimate parameters as posterior probability distributions, and thus naturally account for the uncertainty associated with parameter estimation; Bayesian approaches also offer powerful and principled methods for model comparison. Although software applications such as WinBUGS (Lunn, Thomas, Best, & Spiegelhalter, Statistics and Computing, 10, 325–337, 2000) and JAGS (Plummer, 2003) provide “turnkey”-style packages for Bayesian inference, they can be inefficient when dealing with models whose parameters are correlated, which is often the case for cognitive models, and they can impose significant technical barriers to adding custom distributions, which is often necessary when implementing cognitive models within a Bayesian framework. A recently developed software package called Stan (Stan Development Team, 2015) can solve both problems, as well as provide a turnkey solution to Bayesian inference. We present a tutorial on how to use Stan and how to add custom distributions to it, with an example using the linear ballistic accumulator model (Brown & Heathcote, Cognitive Psychology, 57, 153–178. doi:10.1016/j.cogpsych.2007.12.002, 2008).
___
[**Github**](https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-016-0746-9/MediaObjects/13428_2016_746_MOESM1_ESM.zip)
___