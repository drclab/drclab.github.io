+++
title = "Identification and Inference for Synthetic Controls with Confounding"
#slug = "contact"
date = "2023-12-01"
math = true
+++

by Guido W. Imbens, Davide Viviano
____

This paper studies inference on treatment effects in panel data settings with unobserved confounding. We model outcome variables through a factor model with random factors and loadings. Such factors and loadings may act as unobserved confounders: when the treatment is implemented depends on time-varying factors, and who receives the treatment depends on unit-level confounders. We study the identification of treatment effects and illustrate the presence of a trade-off between time and unit-level confounding. We provide asymptotic results for inference for several Synthetic Control estimators and show that different sources of randomness should be considered for inference, depending on the nature of confounding. We conclude with a comparison of Synthetic Control estimators with alternatives for factor models.

____

$$
\begin{table}[H]
\centering
\caption{Summary of the identification strategies of different regressions. The “Source of randomness” denotes the residual source of variation after conditioning on the endogenous variables (and that should be considered for confidence intervals presented in Section 4). “Confounding effects” denote the components whose expectation is not zero, given the assignment mechanism.}
\begin{tabular}{|l|l|l|l|}
\hline
Regression             & Assumption                                                                 & Source of randomness & Confounding Effects \\ \hline
Horizontal             & Restricted time confounding $(\lambda_{i}, T_{0})$                        & $(\lambda_{i}, \epsilon_{i})$   & $(\lambda_{i} + \Gamma_{i})$ \\
                       & $+$ Exact matching of factors’ conditional expectations                   &                      &  \\
                       & $(\Sigma ST w_{h,s} = \Sigma T w_{h,t} for some w_h)$                      &                      &  \\ \hline
Vertical               & Restricted unit confounding $(T_i^*, N_0)$                                 &$ (\Gamma_i, \epsilon_i)$        &$ (\gamma_i, \lambda + \Lambda)$ \\
                       &$+$ Exact matching of loadings’ conditional expectations                    &&\\ 
                       &$ (\Sigma SN w_i,j = \Sigma N w_j^* for some w_j)$                          &&\\ 
                        \\ \hline
Synthetic DiD          &$Either no high dimensional unit$                                            &(r_i,\Lambda,\epsilon)& Either ($(\lambda+\Lambda,\gamma)$)\\ 
                       &$or time confounders (and matching over time or units)$                     && or ($(\lambda,r+\gamma))$\\ 
                        \\ \hline
PCA/Least Squares      &$No high rank confounders$                                                  &$\epsilon_u$          &(r+\gamma,\lambda+\Lambda)\\ 
                         \\ \hline

\end{tabular}
$$