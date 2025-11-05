suppressPackageStartupMessages({
  library(tidyverse)
  library(mvtnorm)
  library(rstan)
  library(posterior)
  library(bayesplot)
})

rstan_options(auto_write = TRUE)
options(mc.cores = max(2L, parallel::detectCores(logical = FALSE)))
set.seed(202502)
#================================
params <- list(
  mu_alpha = c(0.5, -0.2),
  sigma_alpha = c(0.1, 0.1),
  beta = -0.1,
  sigma_y = 0.05,
  delta = c(0.1, 0.1)
)

design <- list(
  visits = seq(0, 1, length.out = 13),
  J_internal = 100L,
  J_external = 100L
)

str(params)
str(design)
#------------------------------------------

