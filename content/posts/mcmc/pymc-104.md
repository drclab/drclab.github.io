+++
title = "PyMC 104: Change-point modeling — Coal mining disasters"
slug = "pymc-104"
date = "2025-11-19T00:00:00Z"
type = "post"
draft = false
math = true
tags = ["pymc", "bayesian", "time-series", "change-point", "discrete"]
categories = ["posts"]
description = "A worked case study: modeling change points in count data (coal-mining disasters 1851–1962) with PyMC, handling missing years and mixing samplers for discrete parameters."
+++

This note walks through a classic pedagogical example: the number of recorded coal-mining disasters in the UK between 1851 and 1962 (Jarrett, 1979). The data are counts per year and include a couple of missing entries. The scientific question is straightforward: did the disaster rate change at some point during this period, and if so when? A change-point model (two Poisson regimes with a discrete switch-point) is a simple and interpretable way to answer it.

## The data

The original example stores the counts in a pandas Series with years 1851–1962. A few entries are missing (represented as `NaN`), and PyMC will treat those as missing observations to be imputed as part of inference.

Occurrences of disasters in the time series are thought to follow a Poisson process with a large rate parameter in the early part of the time series, and one with a smaller rate in the later part. We are interested in locating the change point in the series, which is perhaps related to changes in mining safety regulations.

```python
# fml: off
disaster_data = pd.Series(
    [4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
     3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
     2, 2, 3, 4, 2, 1, 3, np.nan, 2, 1, 1, 1, 3, 0, 0,
     1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
     0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 0, 1, 0, 1]
)
# fml: on
years = np.arange(1851, 1963)

plt.plot(years, disaster_data, "o", markersize=8, alpha=0.6)
plt.ylabel("Disaster count")
plt.xlabel("Year")
plt.title("Coal mining disasters (1851–1962)")
```

The plot shows a higher frequency of disasters in the early period and fewer disasters later—suggestive of a change in the underlying Poisson rate.

## A simple change-point model

We model the yearly counts as Poisson with two rates: $\lambda_\text{early}$ before the unknown change-point $\tau$, and $\lambda_\text{late}$ after $\tau$. We place weakly informative priors on the rates and a discrete uniform prior on the switch-point.

### Mathematical formulation

In our model:

$$
D_t \sim \text{Pois}(r_t), \quad r_t = \begin{cases} e, & \text{if } t \le s \\ l, & \text{if } t > s \end{cases}
$$

$$
\begin{align}
s &\sim \text{Unif}(t_l, t_h) \\
e &\sim \exp(1) \\
l &\sim \exp(1)
\end{align}
$$

The parameters are defined as follows:

- $D_t$: The number of disasters in year $t$
- $r_t$: The rate parameter of the Poisson distribution of disasters in year $t$
- $s$: The year in which the rate parameter changes (the switchpoint)
- $e$: The rate parameter before the switchpoint $s$
- $l$: The rate parameter after the switchpoint $s$
- $t_l, t_h$: The lower and upper boundaries of year $t$

This model is built much like our previous models. The major differences are the introduction of discrete variables with the Poisson and discrete-uniform priors, and the novel form of the deterministic random variable `rate` that switches between the two rate parameters based on the position relative to the change-point.

Model sketch (implementation notation):

- $\tau \sim \mathrm{DiscreteUniform}(t_\text{min}, t_\text{max})$ (the year of the change)
- $\lambda_\text{early} \sim \mathrm{Exponential}(\alpha)$
- $\lambda_\text{late} \sim \mathrm{Exponential}(\alpha)$
- $y_t \sim \mathrm{Poisson}(\lambda_\text{early})$ for $t \le \tau$, otherwise $y_t \sim \mathrm{Poisson}(\lambda_\text{late})$

Because $\tau$ is discrete we cannot sample it with NUTS. A common and robust solution is to mix samplers: use a Metropolis (or another discrete sampler) for $\tau$ and NUTS for the continuous rate parameters.

## PyMC implementation

The model translates directly into PyMC code:

```python
import pymc as pm
import arviz as az

# prepare the observed array (numpy array with np.nan representing missing years)
obs = disaster_data.values.astype(float)

with pm.Model() as disaster_model:
    switchpoint = pm.DiscreteUniform("switchpoint", lower=years.min(), upper=years.max())
    
    # Priors for pre- and post-switch rates number of disasters
    early_rate = pm.Exponential("early_rate", 1.0)
    late_rate = pm.Exponential("late_rate", 1.0)
    
    # Allocate appropriate Poisson rates to years before and after current
    # rate = pm.math.switch(switchpoint >= years, early_rate, late_rate)
    rate = pm.math.switch(switchpoint >= years, early_rate, late_rate)
    
    disasters = pm.Poisson("disasters", rate, observed=disaster_data)
```

### The `switch` function

The logic for the rate random variable,

```python
rate = switch(switchpoint >= year, early_rate, late_rate)
```

is implemented using `switch`, a function that works like an if statement. It uses the first argument to switch between the next two arguments. In this case:
- If `switchpoint >= year` evaluates to `True`, the rate is set to `early_rate`
- Otherwise, the rate is set to `late_rate`

This creates a vector of rates (one per year) that automatically switches at the change-point, and the expression remains differentiable with respect to the continuous rate parameters.

### Handling missing data

Missing values are handled transparently by passing a NumPy `MaskedArray` or a `DataFrame` with NaN values to the `observed` argument when creating an observed stochastic random variable. Behind the scenes, another random variable, `disasters.missing_values`, is created to model the missing values.

Because we pass `observed=disaster_data` with `np.nan` values, PyMC will automatically:
- Create latent variables for the missing observations
- Sample those missing values as part of the inference
- Include them in posterior predictive checks

No special handling is required—PyMC detects the missing values and treats them as parameters to be estimated.

### Sampling strategy: automatic sampler selection

Unfortunately, because they are discrete variables and thus have no meaningful gradient, we cannot use NUTS for sampling `switchpoint` or the missing disaster observations. Instead, we will sample using a `Metropolis` step method, which implements adaptive Metropolis-Hastings, because it is designed to handle discrete values. PyMC automatically assigns the correct sampling algorithms.

When we call `pm.sample()`, PyMC inspects the model and builds a compound sampler:

```python
with disaster_model:
    idata = pm.sample(10000)
```

This produces output showing the automatic sampler assignment:

```
Multiprocess sampling (4 chains in 4 jobs)
CompoundStep
>>Metropolis: [switchpoint]
>>Metropolis: [disasters_unobserved]
>>NUTS: [early_rate, late_rate]
```

PyMC automatically:
- Uses **Metropolis** for the discrete `switchpoint` parameter
- Uses **Metropolis** for the discrete missing disaster values (`disasters_unobserved`)
- Uses **NUTS** for the continuous rate parameters (`early_rate`, `late_rate`)

This compound step approach combines the strengths of each algorithm: gradient-based exploration for continuous parameters and discrete proposal steps for categorical/count variables.

### Alternative: explicit sampler specification

If you prefer to specify samplers explicitly (for customization or educational purposes), you can do so:

### Alternative: explicit sampler specification

If you prefer to specify samplers explicitly (for customization or educational purposes), you can do so:

```python
import pymc as pm
import arviz as az

# prepare the observed array (numpy array with np.nan representing missing years)
obs = disaster_data.values.astype(float)

with pm.Model() as disasters_model:
    # Discrete change point: note we use integer indices into the years array
    tau = pm.DiscreteUniform("tau", lower=0, upper=len(years) - 1)

    # Rate priors (weakly informative)
    lambda_early = pm.Exponential("lambda_early", 1.0)
    lambda_late = pm.Exponential("lambda_late", 1.0)

    # a vector of rates switching at tau
    rate = pm.math.switch(pm.arange(len(years)) <= tau, lambda_early, lambda_late)

    # Observations: pass the array with NaNs — PyMC will create latent missing entries
    obs_node = pm.Poisson("obs", mu=rate, observed=obs)

    # samplers: Metropolis for discrete tau, NUTS for continuous
    step1 = pm.Metropolis(vars=[tau])
    step2 = pm.NUTS(vars=[lambda_early, lambda_late])

    trace = pm.sample(2000, tune=1000, step=[step1, step2], random_seed=42)
```

Notes:
- Passing `observed=obs` with `np.nan` values instructs PyMC to treat those entries as missing: it will create latent variables for the missing observations which are sampled along with model parameters.
- The `pm.math.switch` expression builds a vector of rates (one rate per year) in a way that is differentiable for the continuous parameters — the switch index `tau` itself remains discrete.
- We explicitly instantiate a `Metropolis` step for `tau` and `NUTS` for the continuous rates and pass both to `pm.sample()`.

However, in most cases, letting PyMC auto-assign samplers (as shown above) is simpler and equally effective.

## Posterior checks and missing data imputation

After sampling you can inspect the posterior distribution of the switch-point and the two rates, impute missing years, and perform posterior predictive checks:

### Visualizing the posterior

```python
# Trace plot for all parameters
axes_arr = az.plot_trace(idata)
plt.draw()

# Customize switchpoint axis for better readability
for ax in axes_arr.flatten():
    if ax.get_title() == "switchpoint":
        labels = [label.get_text() for label in ax.get_xticklabels()]
        ax.set_xticklabels(labels, rotation=45, ha="right")
        break
plt.draw()
```

In the trace plot we can see that there's about a 10-year span that's plausible for a significant change in safety, but a 5-year span that contains most of the probability mass. The distribution is jagged because of the jumpy relationship between the year switchpoint and the likelihood; the jaggedness is not due to sampling error.

The trace plot reveals:
- **Left panels** show the posterior distributions (marginal densities)
- **Right panels** show the sampling chains over iterations
- The **switchpoint** posterior is discrete and concentrated around 1890–1895
- The **early_rate** and **late_rate** parameters show clear separation, confirming a genuine rate change

### Posterior distribution of the change-point

```python
# posterior distribution for tau (convert index to year)
tau_post = trace.posterior["tau"].values.flatten()
plt.hist(tau_post, bins=np.arange(len(years)+1)-0.5)
plt.xticks(np.arange(0, len(years), 10), years[::10], rotation=45)
plt.xlabel("Estimated change-point (year index)")
plt.title("Posterior of the switch-point")
```

Or if using actual years instead of indices:

```python
switchpoint_post = idata.posterior["switchpoint"].values.flatten()
plt.hist(switchpoint_post, bins=range(years.min(), years.max()+2), alpha=0.7, edgecolor='black')
plt.xlabel("Year")
plt.ylabel("Posterior frequency")
plt.title("Posterior distribution of the switchpoint")
```

### Rate parameters
### Rate parameters

```python
# rates
az.plot_posterior(trace, var_names=["lambda_early", "lambda_late"]) 
```

Or with the alternative naming:

```python
az.plot_posterior(idata, var_names=["early_rate", "late_rate"])
```

This shows the clear difference between the early period (higher disaster rate) and the late period (lower disaster rate), with minimal overlap in the posterior distributions.

### Understanding the `rate` variable

Note that the `rate` random variable does not appear in the trace. That is fine in this case, because it is not of interest in itself. Remember from the previous example, we would trace the variable by wrapping it in a `Deterministic` class, and giving it a name.

### Visualizing the change-point with data

The following plot shows the switch point as an orange vertical line, together with its highest posterior density (HPD) as a semitransparent band. The dashed black line shows the accident rate.

```python
plt.figure(figsize=(10, 8))
plt.plot(years, disaster_data, ".", alpha=0.6)
plt.ylabel("Number of accidents", fontsize=16)
plt.xlabel("Year", fontsize=16)

# Stack posterior draws for easy access
trace = idata.posterior.stack(draws=("chain", "draw"))

# Plot switchpoint mean as vertical line
plt.vlines(trace["switchpoint"].mean(), disaster_data.min(), disaster_data.max(), 
           color="C1", alpha=0.7, label="Mean switchpoint")

# Calculate average disaster rate for each year based on switchpoint posterior
average_disasters = np.zeros_like(disaster_data, dtype="float")
for i, year in enumerate(years):
    idx = year < trace["switchpoint"]
    average_disasters[i] = np.mean(np.where(idx, trace["early_rate"], trace["late_rate"]))

# Plot HPD band for switchpoint
sp_hpd = az.hdi(idata, var_names=["switchpoint"])["switchpoint"].values
plt.fill_betweenx(
    y=[disaster_data.min(), disaster_data.max()],
    x1=sp_hpd[0],
    x2=sp_hpd[1],
    alpha=0.5,
    color="C1",
    label="Switchpoint HPD"
)

# Plot the average disaster rate over time
plt.plot(years, average_disasters, "k--", lw=2, label="Expected rate")
plt.legend()
plt.title("Coal mining disasters with estimated change-point")
```

This comprehensive plot shows:
- The observed disaster counts (blue dots)
- The mean switchpoint estimate (orange vertical line)
- The 94% HPD interval for the switchpoint (orange shaded band)
- The expected disaster rate that adapts to the switchpoint (dashed black line)

The expected rate line clearly shows the transition from higher disaster frequency in the early period to lower frequency in the later period.

### Imputed missing values

```python
# Imputed missing values (posterior predictive)
with disasters_model:
    ppc = pm.sample_posterior_predictive(trace, var_names=["obs"], random_seed=42)

# ppc["obs"] has shape (draws, years). For missing-year index i, summarize ppc["obs"][ :, i]
```

Because PyMC created latent variables for missing entries, the posterior predictive draws include draws for those years and give an imputed distribution.

## Practical tips

- Discrete parameters cannot be handled by gradient-based samplers. Use a mixture of samplers (Metropolis / CompoundStep + NUTS) or marginalize discrete parameters analytically if possible.
- For short time series with small counts, the Poisson model is appropriate. For over-dispersed data consider a Negative Binomial likelihood instead.
- Visualize the posterior on the switch-point as a histogram (or rug plot over years) to see the probable change interval, not a single point estimate.
- Use posterior predictive checks to confirm the model reproduces the shape and variability of the data.

## Extensions

- Allow multiple change-points (e.g., two or more): discrete combinatorics grow quickly; hierarchical priors and reversible-jump approaches are possible but more advanced.
- Model seasonality or covariates: include time-varying covariates or a latent Gaussian process for a smoothly varying rate.
- Marginalize discrete parameters: in some conjugate models you can sum/integrate out the discrete variable analytically to keep everything continuous.

## Arbitrary deterministics

Due to its reliance on PyTensor, PyMC provides many mathematical functions and operators for transforming random variables into new random variables. However, the library of functions in PyTensor is not exhaustive, therefore PyTensor and PyMC provide functionality for creating arbitrary functions in pure Python, and including these functions in PyMC models. This is supported with the `as_op` function decorator.

PyTensor needs to know the types of the inputs and outputs of a function, which are specified for `as_op` by `itypes` for inputs and `otypes` for outputs.

```python
from pytensor.compile.ops import as_op

@as_op(itypes=[pt.lscalar], otypes=[pt.lscalar])
def crazy_modulo3(value):
    if value > 0:
        return value % 3
    else:
        return (-value + 1) % 3

with pm.Model() as model_deterministic:
    a = pm.Poisson("a", 1)
    b = crazy_modulo3(a)
```

### Important caveat: gradient limitations

An important drawback of this approach is that it is not possible for `pytensor` to inspect these functions in order to compute the gradient required for the Hamiltonian-based samplers. Therefore, it is not possible to use the HMC or NUTS samplers for a model that uses such an operator. However, it is possible to add a gradient if we inherit from `Op` instead of using `as_op`. The PyMC example set includes a more elaborate example of the usage of `as_op`.

This limitation is why our coal-mining disasters model uses the simpler `pm.math.switch` function (which is differentiable) rather than custom Python functions for the rate switching logic.

### When to use `as_op`

Use `as_op` when:
- You need a custom transformation not available in PyTensor
- You're using discrete-only models (where gradients aren't needed)
- You're willing to use Metropolis or other gradient-free samplers

Avoid `as_op` when:
- You want to use NUTS or HMC samplers
- The transformation can be expressed using built-in PyTensor operations
- Performance is critical (native PyTensor operations are faster)

## Arbitrary distributions

Similarly, the library of statistical distributions in PyMC is not exhaustive, but PyMC allows for the creation of user-defined functions for an arbitrary probability distribution. For simple statistical distributions, the `CustomDist` class takes as an argument any function that calculates a log-probability $\log(p(x))$. This function may employ other random variables in its calculation. Here is an example inspired by a blog post by Jake Vanderplas on which priors to use for a linear regression (Vanderplas, 2014).

```python
import pytensor.tensor as pt

with pm.Model() as model:
    alpha = pm.Uniform('intercept', -100, 100)
    
    # Create variables with custom log-densities
    beta = pm.CustomDist('beta', logp=lambda value: -1.5 * pt.log(1 + value**2))
    eps = pm.CustomDist('eps', logp=lambda value: -pt.log(pt.abs_(value)))
    
    # Create likelihood
    like = pm.Normal('y_est', mu=alpha + beta * X, sigma=eps, observed=Y)
```

### Advanced custom distributions

For more complex distributions, one can create a subclass of `Continuous` or `Discrete` and provide the custom `logp` function, as required. This is how the built-in distributions in PyMC are specified. As an example, fields like psychology and astrophysics have complex likelihood functions for particular processes that may require numerical approximation.

#### Example: Custom distribution with `RandomVariable`

Implementing the `beta` variable above as a `Continuous` subclass is shown below, along with an associated `RandomVariable` object, an instance of which becomes an attribute of the distribution.

```python
class BetaRV(pt.random.op.RandomVariable):
    name = "beta"
    ndim_supp = 0
    ndims_params = []
    dtype = "floatX"
    
    @classmethod
    def rng_fn(cls, rng, size):
        raise NotImplementedError("Cannot sample from beta variable")

beta = BetaRV()
```

#### Example: Full custom distribution subclass

```python
class Beta(pm.Continuous):
    rv_op = beta
    
    @classmethod
    def dist(cls, mu=0, **kwargs):
        mu = pt.as_tensor_variable(mu)
        return super().dist([mu], **kwargs)
    
    def logp(self, value):
        mu = self.mu
        return beta_logp(value - mu)

def beta_logp(value):
    return -1.5 * pt.log(1 + (value) ** 2)

with pm.Model() as model:
    beta = Beta("beta", mu=0)
```

This example shows:
- Creating a custom `RandomVariable` class (`BetaRV`)
- Defining the distribution's properties (name, dimensionality, data type)
- Implementing a `Continuous` distribution subclass (`Beta`)
- Providing the `dist` class method for parameter handling
- Implementing the `logp` method for log-probability calculation
- Using the custom distribution in a model

#### Using `as_op` for log-probability functions

If your logp cannot be expressed in PyTensor, you can decorate the function with `as_op` as follows:

```python
@as_op(itypes=[pt.dscalar], otypes=[pt.dscalar])
```

Note that this will create a blackbox Python function that will be much slower and not provide the gradients necessary for e.g. NUTS. This should only be used as a last resort when the log-probability cannot be expressed using PyTensor operations.

Creating a custom distribution subclass allows you to:
- Define complex probability models not available in the standard library
- Implement domain-specific likelihoods
- Add custom validation and parameterization logic
- Provide custom sampling methods if needed

This flexibility makes PyMC extensible to virtually any probabilistic model, from standard textbook examples to cutting-edge research applications.

## Summary

Change-point models are an excellent first application of Bayesian techniques for count time series. This case study demonstrates how PyMC handles missing observations and how to combine samplers to infer discrete parameters. The coal-mining disasters example remains a compact and instructive problem for learning these techniques.
