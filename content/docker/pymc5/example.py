#!/usr/bin/env python3
"""
Minimal PyMC example to verify the Docker image works.
Samples from a simple Normal model.
"""
import pymc as pm
import arviz as az
import numpy as np

print("PyMC version:", pm.__version__)
print("Starting sampling test...")

# Generate synthetic data
np.random.seed(42)
data = np.random.randn(50) + 2.5

# Build a simple model
with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sigma=10)
    sigma = pm.Exponential("sigma", 1.0)
    obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=data)
    
    # Sample
    trace = pm.sample(draws=200, chains=2, tune=200, cores=1, progressbar=True)

# Print summary
print("\n" + "="*60)
print("Sampling completed successfully!")
print("="*60)
print(az.summary(trace, round_to=2))
print("\nDocker image test PASSED ✓")
