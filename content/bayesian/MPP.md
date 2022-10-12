+++
title = "Incorporating historical controls in clinical trials with longitudinal outcomes using the modified power prior"
#slug = "contact"
date = "2022-09-03"
math = true
+++

by Hongchao Qi, Dimitris Rizopoulos, Emmanuel Lesaffre, Joost van Rosmalen 
___
**Keywords**: Bayesian statistics; clinical trials; historical borrowing; informative prior; modified power prior.
___
**[Abstract](https://doi.org/10.1002/pst.2195)**

Several dynamic borrowing methods, such as the modified power prior (MPP), the commensurate prior, have been proposed to increase statistical power and reduce the required sample size in clinical trials where comparable historical controls are available. Most methods have focused on cross-sectional endpoints, and appropriate methodology for longitudinal outcomes is lacking. In this study, we extend the MPP to the linear mixed model (LMM). An important question is whether the MPP should use the conditional version of the LMM (given the random effects) or the marginal version (averaged over the distribution of the random effects), which we refer to as the conditional MPP and the marginal MPP, respectively. We evaluated the MPP for one historical control arm via a simulation study and an analysis of the data of Alzheimer's Disease Cooperative Study (ADCS) with the commensurate prior as the comparator. The conditional MPP led to inflated type I error rate when there existed moderate or high between-study heterogeneity. The marginal MPP and the commensurate prior yielded a power gain (3.6%-10.4% vs. 0.6%-4.6%) with the type I error rates close to 5% (5.2%-6.2% vs. 3.8%-6.2%) when the between-study heterogeneity is not excessively high. For the ADCS data, all the borrowing methods improved the precision of estimates and provided the same clinical conclusions. The marginal MPP and the commensurate prior are useful for borrowing historical controls in longitudinal data analysis, while the conditional MPP is not recommended due to inflated type I error rates.
___
[**Github**](https://github.com/QiHongchao/MPP_longitudinal)
___