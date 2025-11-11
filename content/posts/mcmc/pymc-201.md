+++
title = "PyMC 201: Advanced Topics — Custom Operations and Distributions"
slug = "pymc-201"
date = "2025-11-19T01:00:00Z"
type = "post"
draft = false
math = true
tags = ["pymc", "advanced", "custom-distributions"]
categories = ["posts"]
description = "Advanced PyMC techniques: creating arbitrary deterministic transformations and custom probability distributions for specialized modeling needs."
+++

This note covers advanced PyMC techniques for extending the framework beyond its built-in capabilities. You'll learn how to create custom deterministic operations and define custom probability distributions when the standard library doesn't meet your modeling needs.

## Prerequisites

This guide assumes familiarity with:
- Basic PyMC modeling (see [PyMC 102](/posts/pymc-102/) and [PyMC 104](/posts/pymc-104/))
- PyTensor tensor operations
- Probability distributions and log-probabilities
- Python class inheritance

## Arbitrary deterministics

Due to its reliance on PyTensor, PyMC provides many mathematical functions and operators for transforming random variables into new random variables. However, the library of functions in PyTensor is not exhaustive, therefore PyTensor and PyMC provide functionality for creating arbitrary functions in pure Python, and including these functions in PyMC models. This is supported with the `as_op` function decorator.

PyTensor needs to know the types of the inputs and outputs of a function, which are specified for `as_op` by `itypes` for inputs and `otypes` for outputs.

```python
from pytensor.compile.ops import as_op
import pytensor.tensor as pt

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

This limitation is why models that need gradient-based sampling typically use built-in PyTensor operations. For example, the coal-mining disasters change-point model uses the simpler `pm.math.switch` function (which is differentiable) rather than custom Python functions for the rate switching logic.

### When to use `as_op`

Use `as_op` when:
- You need a custom transformation not available in PyTensor
- You're using discrete-only models (where gradients aren't needed)
- You're willing to use Metropolis or other gradient-free samplers
- The transformation is complex and would be difficult to express in PyTensor

Avoid `as_op` when:
- You want to use NUTS or HMC samplers
- The transformation can be expressed using built-in PyTensor operations
- Performance is critical (native PyTensor operations are faster)
- You need automatic differentiation

### Modern alternative: Creating custom Ops

For more control and better integration with PyTensor, you can create custom operations by subclassing `Op`:

```python
import pytensor.tensor as pt
from pytensor.graph import Op
from pytensor.graph.basic import Apply

class CrazyModulo3(Op):
    def make_node(self, x):
        x = pt.as_tensor_variable(x)
        return Apply(self, [x], [x.type()])
    
    def perform(self, node, inputs, outputs):
        value = inputs[0]
        if value > 0:
            outputs[0][0] = value % 3
        else:
            outputs[0][0] = (-value + 1) % 3

crazy_modulo3 = CrazyModulo3()

with pm.Model() as model_deterministic:
    a = pm.Poisson("a", 1)
    b = crazy_modulo3(a)
```

This approach:
- Avoids deprecation warnings from `as_op`
- Provides better integration with PyTensor's graph system
- Allows for more sophisticated behavior and optimization
- Still has the same gradient limitations unless you implement `grad` method

## Arbitrary distributions

Similarly, the library of statistical distributions in PyMC is not exhaustive, but PyMC allows for the creation of user-defined functions for an arbitrary probability distribution. For simple statistical distributions, the `CustomDist` class takes as an argument any function that calculates a log-probability $\log(p(x))$. This function may employ other random variables in its calculation.

### Basic custom distributions with `CustomDist`

Here is an example inspired by a blog post by Jake Vanderplas on which priors to use for a linear regression (Vanderplas, 2014).

```python
import pymc as pm
import pytensor.tensor as pt
import numpy as np

# Generate sample data for demonstration
np.random.seed(42)
X = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
Y = np.array([2.5, 4.1, 5.8, 8.2, 10.5, 12.1, 14.3, 16.8, 18.9, 21.2])

with pm.Model() as model:
    alpha = pm.Uniform('intercept', -100, 100)
    
    # Create variables with custom log-densities
    beta = pm.CustomDist('beta', logp=lambda value: -1.5 * pt.log(1 + value**2))
    eps = pm.CustomDist('eps', logp=lambda value: -pt.log(pt.abs_(value)))
    
    # Create likelihood
    like = pm.Normal('y_est', mu=alpha + beta * X, sigma=eps, observed=Y)
```

The custom log-probability functions can use any PyTensor operations, including:
- Mathematical operations: `pt.log`, `pt.exp`, `pt.sqrt`, etc.
- Conditional logic: `pt.switch`, `pt.where`
- Array operations: `pt.sum`, `pt.prod`, `pt.mean`
- Other random variables (for hierarchical models)

### Advanced custom distributions

For more complex distributions, one can create a subclass of `Continuous` or `Discrete` and provide the custom `logp` function, as required. This is how the built-in distributions in PyMC are specified. As an example, fields like psychology and astrophysics have complex likelihood functions for particular processes that may require numerical approximation.

#### Example: Custom distribution with `RandomVariable`

Implementing the `beta` variable above as a `Continuous` subclass is shown below, along with an associated `RandomVariable` object, an instance of which becomes an attribute of the distribution.

```python
import pytensor.tensor as pt
import pymc as pm

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

### Key components of custom distributions

When creating a custom distribution subclass, you typically need to implement:

1. **`rv_op`**: The RandomVariable operation associated with the distribution
2. **`dist()` classmethod**: Constructs the distribution with given parameters
3. **`logp()` method**: Computes the log-probability density/mass
4. **`logcdf()` method** (optional): Computes the log cumulative distribution function
5. **`rng_fn()` classmethod**: Defines how to sample from the distribution (in the RV class)

### Using `as_op` for log-probability functions

If your logp cannot be expressed in PyTensor, you can decorate the function with `as_op` as follows:

```python
from pytensor.compile.ops import as_op

@as_op(itypes=[pt.dscalar], otypes=[pt.dscalar])
def custom_logp(value):
    # Pure Python computation
    return some_complex_calculation(value)

with pm.Model() as model:
    custom_var = pm.CustomDist('custom', logp=custom_logp)
```

Note that this will create a blackbox Python function that will be much slower and not provide the gradients necessary for e.g. NUTS. This should only be used as a last resort when the log-probability cannot be expressed using PyTensor operations.

### Benefits of custom distributions

Creating a custom distribution subclass allows you to:
- **Define complex probability models** not available in the standard library
- **Implement domain-specific likelihoods** (e.g., for specialized scientific applications)
- **Add custom validation and parameterization logic**
- **Provide custom sampling methods** if needed
- **Share and reuse distributions** across multiple models
- **Integrate with PyMC's automatic inference machinery**

This flexibility makes PyMC extensible to virtually any probabilistic model, from standard textbook examples to cutting-edge research applications.

## Use cases and examples

### Use case 1: Non-standard likelihood functions

In fields like psychology or neuroscience, you might have complex likelihood functions based on cognitive models:

```python
def drift_diffusion_logp(rt, choice, drift, threshold, non_decision):
    """Log-probability for drift diffusion model"""
    # Complex calculation involving numerical integration
    # or lookup tables
    return logp_value

with pm.Model() as ddm_model:
    drift = pm.Normal('drift', 0, 1)
    threshold = pm.Gamma('threshold', 1, 1)
    non_decision = pm.Uniform('non_decision', 0, 1)
    
    likelihood = pm.CustomDist('rt', 
                               drift, threshold, non_decision,
                               logp=drift_diffusion_logp,
                               observed={'rt': rt_data, 'choice': choice_data})
```

### Use case 2: Astrophysical models

In astrophysics, you might need custom distributions for modeling astronomical phenomena:

```python
def lognormal_mixture_logp(x, mu1, sigma1, mu2, sigma2, weight):
    """Mixture of two lognormal distributions"""
    logp1 = -pt.log(x * sigma1 * pt.sqrt(2 * np.pi)) - 0.5 * ((pt.log(x) - mu1) / sigma1)**2
    logp2 = -pt.log(x * sigma2 * pt.sqrt(2 * np.pi)) - 0.5 * ((pt.log(x) - mu2) / sigma2)**2
    return pt.log(weight * pt.exp(logp1) + (1 - weight) * pt.exp(logp2))
```

### Use case 3: Truncated and censored distributions

When standard truncation isn't sufficient:

```python
def doubly_truncated_normal_logp(x, mu, sigma, lower, upper):
    """Normal distribution truncated on both sides"""
    z = (x - mu) / sigma
    z_lower = (lower - mu) / sigma
    z_upper = (upper - mu) / sigma
    
    normalizing_constant = pt.log(
        0.5 * (pt.erf(z_upper / pt.sqrt(2)) - pt.erf(z_lower / pt.sqrt(2)))
    )
    
    return -0.5 * z**2 - pt.log(sigma * pt.sqrt(2 * np.pi)) - normalizing_constant
```

## Best practices

1. **Start simple**: Try `CustomDist` with a lambda function before creating full subclasses
2. **Test thoroughly**: Verify your log-probability is correct with known cases
3. **Document assumptions**: Clearly state what your custom distribution represents
4. **Check gradients**: If using NUTS/HMC, ensure gradients are available and correct
5. **Consider performance**: Native PyTensor operations are much faster than Python loops
6. **Validate numerically**: Check for numerical stability (overflow, underflow, NaN)
7. **Provide examples**: Include usage examples when sharing custom distributions

## Common pitfalls

- **Forgetting normalization constants**: Your logp must include all terms, including constants
- **Mixing Python and PyTensor**: Stick to PyTensor operations inside logp functions
- **Not handling edge cases**: Check for division by zero, log of negative numbers, etc.
- **Incorrect broadcasting**: Ensure your logp handles array inputs correctly
- **Performance issues**: Avoid Python loops; use PyTensor's vectorized operations

## Testing custom distributions

Always test your custom distributions before using them in production:

```python
# Test 1: Check that logp evaluates
test_value = 1.0
logp_result = my_custom_dist.logp(test_value)
print(f"logp({test_value}) = {logp_result.eval()}")

# Test 2: Compare with known distribution (if possible)
from scipy import stats
scipy_logp = stats.norm.logpdf(test_value, loc=0, scale=1)
pymc_logp = pm.Normal.dist(0, 1).logp(test_value).eval()
assert np.isclose(scipy_logp, pymc_logp)

# Test 3: Check gradients (for continuous distributions)
import pytensor
x = pt.scalar('x')
logp_expr = my_custom_logp(x)
grad_expr = pytensor.gradient.grad(logp_expr, x)
grad_fn = pytensor.function([x], grad_expr)
print(f"Gradient at {test_value}: {grad_fn(test_value)}")
```

## Resources

- [PyMC Examples Gallery](https://www.pymc.io/projects/examples/en/latest/)
- [PyTensor Documentation](https://pytensor.readthedocs.io/)
- [CustomDist API Reference](https://www.pymc.io/projects/docs/en/latest/api/distributions.html)
- [Creating Custom Ops](https://pytensor.readthedocs.io/en/latest/extending/creating_an_op.html)

## Summary

Custom operations and distributions are powerful tools for extending PyMC to handle specialized modeling needs. While the built-in library covers most common use cases, understanding how to create custom components allows you to:

- Implement domain-specific models from scientific literature
- Prototype new statistical methods
- Handle unusual data structures or likelihood functions
- Push the boundaries of probabilistic programming

Remember to start with the simplest approach (`CustomDist` with lambda functions) and only move to more complex implementations (subclassing distributions or creating custom Ops) when necessary. Always validate your implementations thoroughly before using them in production analyses.
