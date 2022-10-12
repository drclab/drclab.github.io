+++
title = "Bayesian workflow for disease transmission modeling in Stan"
#slug = "contact"
date = "2021-07-03"
math = true
+++

by Léo Grinsztajn, Elizaveta Semenova, Charles C. Margossian, Julien Riou
___
**Keywords**: Bayesian workflow, computational models, epidemiology, infection diseases
___
**[Abstract](https://doi.org/10.1002/sim.9164)**

This tutorial shows how to build, fit, and criticize disease transmission models in Stan, and should be useful to researchers interested in modeling the COVID-19 outbreak and doing Bayesian inference. Bayesian modeling provides a principled way to quantify uncertainty and incorporate prior knowledge into the model. What is more, Stan’s main inference engine, Hamiltonian Monte Carlo sampling, is amiable to diagnostics, which means we can verify whether our inference is reliable. Stan is an expressive probabilistic programing language that abstracts the inference and allows users to focus on the modeling. The resulting code is readable and easily extensible, which makes the modeler’s work more transparent and flexible. In this tutorial, we demonstrate with a simple Susceptible-Infected-Recovered (SIR) model how to formulate, fit, and diagnose a compartmental model in Stan. We also introduce more advanced topics which can help practitioners fit sophisticated models; notably, how to use simulations to probe our model and our priors, and computational techniques to scale ODE-based models.
___
[**Github**](https://mc-stan.org/users/documentation/case-studies/boarding_school_case_study.html)
___