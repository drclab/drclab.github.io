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
Sigma_alpha <- diag(params$sigma_alpha^2)
Sigma_alpha
alpha <- mvtnorm::rmvnorm(design$J_internal, params$mu_alpha, Sigma_alpha)
alpha

x <- design$visits
T <- length(x)
x
T
y <- matrix(NA_real_, nrow = design$J_internal, ncol = T)
y

beta

for (k in seq_len(T)) {
  mean_k <- alpha[, 1] + alpha[, 2] * x[k] + params$beta * x[k]^2
  y[, k] <- rnorm(design$J_internal, mean_k, params$sigma_y)
}

y



