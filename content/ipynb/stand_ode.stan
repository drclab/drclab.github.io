functions {
  real[] drug_disease_stim_kinL_Et_ode(real t,
                                       real[] y,
                                       real[] theta,
                                       real[] x_r,
                                       int[] x_i) {
    real dydt[1];
    real emax_0 = theta[1];
    real lec_50 = theta[2];
    real r180 = theta[3];
    real beta = theta[4];
    real k_in = theta[5];
    real k_out = theta[6];
    real start_t = x_r[1];
    real lconc_0 = x_r[2];
    real K = x_r[3];
    real hill = x_r[4];
    real lconcentration = lconc_0 - K * (t - start_t);
    real emax = emax_0 * (r180 + (1 - r180) * exp(-beta * t / 30.0));
    real stim = emax * inv_logit(hill * (lconcentration - lec_50));
    dydt[1] = k_in - k_out * (y[1] - stim);
    return dydt;
  }
}

data {
  int<lower=1> N;
  vector[N] time;
  vector[N] bcva_obs;
  real start_t;
  real lconc0;
  real<lower=0> K;
  real hill;
  real<lower=0> r180;
  real<lower=0> beta;
  real<lower=0> sigma_prior_scale;
}

parameters {
  real<lower=0> k_in;
  real<lower=0> k_out;
  real<lower=0> emax0;
  real lec50;
  real<lower=0> sigma;
  real R0;
}

transformed parameters {
  vector[N] mu_bcva;
  {
    real y0[1];
    real theta[6];
    real x_r[4];
    int x_i[0];
    y0[1] = R0;
    theta[1] = emax0;
    theta[2] = lec50;
    theta[3] = r180;
    theta[4] = beta;
    theta[5] = k_in;
    theta[6] = k_out;
    x_r[1] = start_t;
    x_r[2] = lconc0;
    x_r[3] = K;
    x_r[4] = hill;
    {
      real y_hat[N, 1] = integrate_ode_rk45(drug_disease_stim_kinL_Et_ode,
                                            y0,
                                            0,
                                            time,
                                            theta,
                                            x_r,
                                            x_i);
      for (n in 1:N) {
        mu_bcva[n] = inv_logit(y_hat[n, 1]) * 100;
      }
    }
  }
}

model {
  k_in ~ normal(0.05, 0.02);
  k_out ~ normal(0.04, 0.01);
  emax0 ~ normal(0.8, 0.3);
  lec50 ~ normal(log(6), 0.5);
  sigma ~ normal(0, sigma_prior_scale) T[0,];
  R0 ~ normal(0, 1);
  bcva_obs ~ normal(mu_bcva, sigma);
}

generated quantities {
  vector[N] bcva_rep;
  for (n in 1:N) {
    bcva_rep[n] = normal_rng(mu_bcva[n], sigma);
  }
}
