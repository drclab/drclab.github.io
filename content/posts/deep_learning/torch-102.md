+++
title = "Torch 102: Activation Functions Meet Delivery Reality"
date = "2025-12-02T00:00:18Z"
type = "post"
draft = false
tags = [
  "deep learning",
  "torch",
  "relu",
  "tutorial"
]
categories = ["deep_learning"]
description = "Follow-up to Torch 101 that adds normalization, hidden layers, and ReLU to solve the curved delivery-time dataset from Lab 2."
+++

Torch 101 showed that a single linear neuron can set the baseline for bike-only deliveries, but the combined bike + car dataset instantly exposed its limits. Lab 2 in `content/ipynb/C1_M1_Lab_2_activation_functions.ipynb` picks up right there: same courier story, brand new wrinkles. This post distills the notebook so you can keep iterating inside Hugo without reopening Jupyter.

## The New Failure Mode: Curves

Once car routes join the mix, delivery distance no longer scales linearly with time. The plot bends upward twice—traffic and stoplights stretch long trips disproportionately. Add as many linear neurons as you want and they will still collapse into a single straight line. We need two upgrades:

1. **Normalized inputs/targets** so training stays numerically stable as miles and minutes span different ranges.
2. **Non-linear activation** between layers so the network can learn piecewise functions instead of one slope.

## Step 1: Normalize Before You Train

Standardization gives every feature zero mean and unit variance. That keeps gradients from exploding while still letting PyTorch learn the relationships.

```python
# Bike-only routes (shorter, linear-ish) + car routes (longer, curved by traffic)
distances = torch.tensor([
    1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,  # bike deliveries
    5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0  # car deliveries
], dtype=torch.float32)

times = torch.tensor([
    7.0, 10.0, 13.0, 16.0, 19.0, 22.0, 25.0, 28.0,  # bike: ~6.5 min/mile
    32.0, 38.0, 45.0, 52.0, 66.0, 78.0, 87.0, 91.0, 93.0  # car: curves up
], dtype=torch.float32)

dist_mean, dist_std = distances.mean(), distances.std()
time_mean, time_std = times.mean(), times.std()

distances_norm = (distances - dist_mean) / dist_std
times_norm = (times - time_mean) / time_std
```

Save the means and standard deviations—you will need them at inference time.

## Step 2: Hidden Layer + ReLU

One hidden layer with three neurons is enough to mimic the notebook’s curved fit when you add ReLU.

```python
model = nn.Sequential(
    nn.Linear(1, 3),
    nn.ReLU(),
    nn.Linear(3, 1)
)
```

`nn.Linear(1, 3)` projects the single normalized distance into three features, `nn.ReLU()` keeps only the positive halves to introduce bends, and `nn.Linear(3, 1)` collapses everything back to a single normalized travel time.

## Step 3: Train Longer, Watch the Fit

Curved targets take more iterations than the four-point straight line from Torch 101. Stick with MSE loss and SGD, but give the loop breathing room:

```python
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(3000):
    optimizer.zero_grad()
    preds = model(distances_norm)
    loss = loss_fn(preds, times_norm)
    loss.backward()
    optimizer.step()
```

In the lab notebook you can watch the plot update live—the line gradually gains the bends needed to hug the car-heavy points. Even without the plot, logging the loss every few hundred epochs confirms you’re still descending.

## Step 4: Predict in Real Units

Because the network only understands normalized tensors, you have to transform inputs and outputs at the edges:

```python
def predict_minutes(distance_miles: float) -> float:
    distance_norm = (torch.tensor([[distance_miles]]) - dist_mean) / dist_std
    with torch.no_grad():
        time_norm = model(distance_norm)
    return (time_norm * time_std + time_mean).item()

eta = predict_minutes(7.0)
late = eta > 45  # company now promises <45 minutes
vehicle = "car" if late else "bike"
```

On the mixed dataset, the ReLU model lands around 44–45 minutes for a 7-mile route—close enough that switching vehicles now matters. This is the moment where activation functions pay off: the business decision changes because the curve finally fits.

## What to Explore Next

- Sweep the number of hidden neurons to see how model capacity trades off with overfitting the sparse dataset.
- Swap `ReLU` for `LeakyReLU` or `SiLU` and compare the resulting delivery-time curves.
- Package the normalization constants alongside the saved model weights so production callers never forget to scale inputs.

Torch 101 gave you the vocabulary. Torch 102 gives you the bends—keep layering these ideas and you will be ready for multi-feature route prediction and even classification in the next module.
