+++
title = "Bayesian aggregation of average data: An application in drug development"
#slug = "contact"
date = "2018-09-03"
math = true
+++

by Sebastian Weber, Andrew Gelman, Daniel Lee, Michael Betancourt, Aki Vehtari, Amy Racine-Poon
___
**Keywords**: Bayesian computation , hierarchical modeling , Meta-analysis , pharmacometrics , Stan
___
**[Abstract](https://doi.org/10.1214/17-AOAS1122)**

Throughout the different phases of a drug development program, randomized trials are used to establish the tolerability, safety and efficacy of a candidate drug. At each stage one aims to optimize the design of future studies by extrapolation from the available evidence at the time. This includes collected trial data and relevant external data. However, relevant external data are typically available as averages only, for example, from trials on alternative treatments reported in the literature. Here we report on such an example from a drug development for wet age-related macular degeneration. This disease is the leading cause of severe vision loss in the elderly. While current treatment options are efficacious, they are also a substantial burden for the patient. Hence, new treatments are under development which need to be compared against existing treatments.

The general statistical problem this leads to is meta-analysis, which addresses the question of how we can combine data sets collected under different conditions. Bayesian methods have long been used to achieve partial pooling. Here we consider the challenge when the model of interest is complex (hierarchical and nonlinear) and one data set is given as raw data while the second data set is given as averages only. In such a situation, common meta-analytic methods can only be applied when the model is sufficiently simple for analytic approaches. When the model is too complex, for example, nonlinear, an analytic approach is not possible. We provide a Bayesian solution by using simulation to approximately reconstruct the likelihood of the external summary and allowing the parameters in the model to vary under the different conditions. We first evaluate our approach using fake data simulations and then report results for the drug development program that motivated this research.
___
[**Github**](https://github.com/hublun/Bayesian_Aggregation_Average_Data)
___