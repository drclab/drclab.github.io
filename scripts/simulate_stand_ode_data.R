#!/usr/bin/env Rscript

# Simulate a toy dataset compatible with content/ipynb/stand_ode.stan and run Stan chains.
set.seed(1234)

library(rstan)
rstan::rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

#================================================
# Configuration (edit these first when in RStudio)
#================================================
times <- seq(0, 140, by = 7)
output_path <- "content/ipynb/stand_ode_sim.rds"
stan_path <- "/Users/09344682/GitHub/drclab.github.io/content/ipynb/stand_ode.stan"
fit_output_path <- paste0(tools::file_path_sans_ext(output_path), "_fit.rds")
stan_iter <- 1000
stan_chains <- 4
stan_warmup <- stan_iter %/% 2
stan_seed <- 1234

args <- commandArgs(trailingOnly = TRUE)
if (!interactive()) {
  if (length(args) >= 1) output_path <- args[1]
  if (length(args) >= 2) stan_path <- args[2]
  if (length(args) >= 3) fit_output_path <- args[3]
}

#================================================
# Deterministic truths
#================================================
truth <- list(
  k_in = 0.05,
  k_out = 0.04,
  emax0 = 0.85,
  lec50 = log(6),
  r180 = 0.55,
  beta = 0.3,
  hill = 1.4,
  K = 0.015,
  start_t = 0,
  lconc0 = 3,
  R0 = -0.5,
  sigma = 3
)

#================================================
# Manual RK4 integration (no helper functions)
#================================================
latent <- numeric(length(times))
t_curr <- 0
y_curr <- truth$R0
dt <- 0.25
for (i in seq_along(times)) {
  target <- times[i]
  while (t_curr + 1e-9 < target) {
    step <- min(dt, target - t_curr)

    lconc <- truth$lconc0 - truth$K * (t_curr - truth$start_t)
    emax <- truth$emax0 * (truth$r180 + (1 - truth$r180) * exp(-truth$beta * t_curr / 30))
    stim <- emax * plogis(truth$hill * (lconc - truth$lec50))
    k1 <- truth$k_in - truth$k_out * (y_curr - stim)

    t_half <- t_curr + step / 2
    y_half <- y_curr + step * k1 / 2
    lconc <- truth$lconc0 - truth$K * (t_half - truth$start_t)
    emax <- truth$emax0 * (truth$r180 + (1 - truth$r180) * exp(-truth$beta * t_half / 30))
    stim <- emax * plogis(truth$hill * (lconc - truth$lec50))
    k2 <- truth$k_in - truth$k_out * (y_half - stim)

    y_half <- y_curr + step * k2 / 2
    lconc <- truth$lconc0 - truth$K * (t_half - truth$start_t)
    emax <- truth$emax0 * (truth$r180 + (1 - truth$r180) * exp(-truth$beta * t_half / 30))
    stim <- emax * plogis(truth$hill * (lconc - truth$lec50))
    k3 <- truth$k_in - truth$k_out * (y_half - stim)

    t_end <- t_curr + step
    y_end <- y_curr + step * k3
    lconc <- truth$lconc0 - truth$K * (t_end - truth$start_t)
    emax <- truth$emax0 * (truth$r180 + (1 - truth$r180) * exp(-truth$beta * t_end / 30))
    stim <- emax * plogis(truth$hill * (lconc - truth$lec50))
    k4 <- truth$k_in - truth$k_out * (y_end - stim)

    y_curr <- y_curr + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    t_curr <- t_end
  }
  latent[i] <- y_curr
}

#================================================
# Observations and Stan data
#================================================
mu_bcva <- plogis(latent) * 100
bcva_obs <- rnorm(length(times), mu_bcva, truth$sigma)

stan_data <- list(
  N = length(times),
  time = times,
  bcva_obs = bcva_obs,
  start_t = truth$start_t,
  lconc0 = truth$lconc0,
  K = truth$K,
  hill = truth$hill,
  r180 = truth$r180,
  beta = truth$beta,
  sigma_prior_scale = 5
)

#================================================
# Persist simulated data
#================================================
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
saveRDS(stan_data, output_path)
message(sprintf("Saved Stan-ready list with %d observations to %s", stan_data$N, output_path))

preview <- data.frame(
  time = times,
  mu_bcva = round(mu_bcva, 2),
  bcva_obs = round(bcva_obs, 2)
)
utils::write.csv(preview, sub("\\.rds$", ".csv", output_path, ignore.case = TRUE), row.names = FALSE)

#================================================
# Compile and sample Stan model
#================================================
if (!file.exists(stan_path)) {
  stop(sprintf("Stan file not found: %s", stan_path), call. = FALSE)
}
message(sprintf("Compiling Stan model from %s", stan_path))
message(sprintf("Sampling %d chains × %d iterations (warmup = %d)", stan_chains, stan_iter, stan_warmup))
stan_fit <- rstan::stan(file = stan_path,
                        data = stan_data,
                        chains = stan_chains,
                        iter = stan_iter,
                        warmup = stan_warmup,
                        seed = stan_seed,
                        refresh = 0)

dir.create(dirname(fit_output_path), recursive = TRUE, showWarnings = FALSE)
saveRDS(stan_fit, fit_output_path)
message(sprintf("Saved Stan fit to %s", fit_output_path))

preview_fit <- utils::capture.output(print(stan_fit))
if (length(preview_fit)) {
  message("Stan fit preview:")
  cat(paste(preview_fit[seq_len(min(5L, length(preview_fit)))], collapse = "\n"), "\n", sep = "")
}

#================================================
print(stan_fit)
traceplot(stan_fit)
