+++
title = "Stable Differentiable Causal Discovery"
#slug = "contact"
date = "2023-11-17"
math = true
+++

by Achille Nazaret, Justin Hong, Elham Azizi, David Blei

[Abstract](https://doi.org/10.48550/arXiv.2311.10263)

Inferring causal relationships as directed acyclic graphs (DAGs) is an important but challenging problem. Differentiable Causal Discovery (DCD) is a promising approach to this problem, framing the search as a continuous optimization. But existing DCD methods are numerically unstable, with poor performance beyond tens of variables. In this paper, we propose Stable Differentiable Causal Discovery (SDCD), a new method that improves previous DCD methods in two ways: (1) It employs an alternative constraint for acyclicity; this constraint is more stable, both theoretically and empirically, and fast to compute. (2) It uses a training procedure tailored for sparse causal graphs, which are common in real-world scenarios. We first derive SDCD and prove its stability and correctness. We then evaluate it with both observational and interventional data and on both small-scale and large-scale settings. We find that SDCD outperforms existing methods in both convergence speed and accuracy and can scale to thousands of variables.

[Github Repo](https://github.com/azizilab/sdcd)