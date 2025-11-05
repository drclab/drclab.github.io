#!/usr/bin/env Rscript
## Toy data generator + sampler for stan/linear_mvn_approx.stan

args <- commandArgs(trailingOnly = FALSE)
script_arg <- grep("^--file=", args, value = TRUE)
script_dir <- if (length(script_arg)) {
  dirname(normalizePath(sub("^--file=", "", script_arg)))
} else {
  getwd()
}
##------------------------------------------------
set.seed(20250217L)

K <- 2L
J <- 6L
T <- 5L
J_prime <- 4L
T_prime <- 4L

x <- seq(-1, 1, length.out = T)
x_prime <- seq(-1.5, 1.5, length.out = T_prime)

mu_a_true <- c(0.7, 1.0)
delta_true <- c(0.25, -0.15)
beta_true <- -0.3
sigma_a_true <- c(0.4, 0.25)
sigma_y_true <- 0.2

eta_a <- matrix(rnorm(K * J), K, J)
a <- matrix(mu_a_true, K, J) + diag(sigma_a_true) %*% eta_a

y <- matrix(NA_real_, J, T)
for (j in 1:J) {
  for (k in 1:T) {
    mu_jk <- a[1, j] + a[2, j] * x[k] + beta_true * x[k]^2
    y[j, k] <- rnorm(1, mean = mu_jk, sd = sigma_y_true)
  }
}

eta_prime_a <- matrix(rnorm(K * J_prime), K, J_prime)
a_prime <- matrix(mu_a_true + delta_true, K, J_prime) +
  diag(sigma_a_true) %*% eta_prime_a

y_prime <- matrix(NA_real_, J_prime, T_prime)
for (j in 1:J_prime) {
  for (k in 1:T_prime) {
    mu_prime_jk <- a_prime[1, j] +
      a_prime[2, j] * x_prime[k] +
      beta_true * x_prime[k]^2
    y_prime[j, k] <- rnorm(1, mean = mu_prime_jk, sd = sigma_y_true)
  }
}

K_phi <- 6L
mu_phi_p <- c(mu_a_true, beta_true, log(sigma_a_true), log(sigma_y_true))
Sigma_phi_p <- diag(c(1, 1, 1, 0.25, 0.25, 0.25))
mu_delta_p <- rep(0, K)
Sigma_delta_p <- diag(0.5, K)

J_tilde <- 10L
C <- 4L
xi <- array(rnorm(C * K * 2 * J_tilde), dim = c(C, K, 2 * J_tilde))

stan_data <- list(
  fit_all = 0L,
  fit_local = 0L,
  J = J,
  T = T,
  K = K,
  K_phi = K_phi,
  y = y,
  J_prime = J_prime,
  T_prime = T_prime,
  y_prime = y_prime,
  x = x,
  x_prime = x_prime,
  mu_phi_p = mu_phi_p,
  Sigma_phi_p = Sigma_phi_p,
  mu_delta_p = mu_delta_p,
  Sigma_delta_p = Sigma_delta_p,
  J_tilde = J_tilde,
  C = C,
  CHAIN_ID = 1L,
  xi = xi
)

str(stan_data)

if (!requireNamespace("rstan", quietly = TRUE)) {
  stop("Package 'rstan' is required. Install it with install.packages('rstan').")
}

rstan::rstan_options(auto_write = TRUE)
options(mc.cores = min(2L, parallel::detectCores()))

stan_file <- file.path(script_dir, "/content/ipynb/stan", "linear_mvn_approx.stan")

if (!file.exists(stan_file)) {
  stop("Stan model not found at ", stan_file)
}

message("Compiling Stan model at ", stan_file, " ...")
stan_model <- rstan::stan_model(file = stan_file)

message("Sampling ...")
fit <- rstan::sampling(
  stan_model,
  data = stan_data,
  chains = 1L,
  iter = 500L,
  warmup = 250L,
  seed = 20250217L,
  refresh = 100L
)

print(fit, pars = c("mu_a", "delta", "beta", "sigma_a", "sigma_y"))
