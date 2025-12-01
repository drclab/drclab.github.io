+++
title = "Torch 101: One Neuron, Real Insights"
date = "2025-12-01T01:53:45Z"
type = "post"
draft = true
tags = ["deep learning", "torch", "tutorial"]
categories = ["deep_learning"]
description = "Step-by-step PyTorch primer based on the C1_M1 simple neural network lab notebook."
+++

PyTorch feels intimidating until you build something tangible. The `content/ipynb/C1_M1_Lab_1_simple_nn.ipynb` notebook walks through a courier scenario—can a rider finish a 7-mile order before the 30-minute promise? This post distills that worksheet into a reference you can revisit whenever you need a quick reset on tensors, modules, and the ML pipeline.

## Why Start With One Neuron?

The lab follows a slimmed-down Machine Learning Pipeline:

1. **Prepare** the data you already trust.
2. **Build** the right-sized network (a single linear neuron).
3. **Train** it with an optimizer + loss pair you understand.
4. **Predict** and sanity-check against reality.

Working inside this scaffold keeps the focus on PyTorch primitives rather than tooling quirks.

## Stage 1 & 2: Data Ingestion + Preparation

You only need two tensors: delivery distances (miles) and total times (minutes). Keeping them as 32-bit floats matches CUDA defaults later, and adding a column dimension lines up with `nn.Linear` expectations.

```python
import torch

distances = torch.tensor([[1.0], [3.0], [5.0], [7.0]], dtype=torch.float32)
times = torch.tensor([[15.0], [27.0], [39.0], [45.0]], dtype=torch.float32)
```

Even in a toy lab, it helps to remind yourself what each tensor represents and the measurement units. That discipline pays off once the dataset is streaming from a feature store instead of a notebook cell.

## Stage 3: Model Building

A single neuron with one input and one output implements the familiar line `time = w * distance + b`. PyTorch makes that explicit with a Linear layer wrapped in `nn.Sequential` (handy when you later stack more layers).

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(1, 1)  # one distance in, one predicted time out
)
```

`nn.Linear` registers both the weight and bias parameters, so you can inspect or log them during training.

## Stage 4: Training Loop Essentials

Loss + optimizer define how learning progresses. The notebook sticks with Mean Squared Error plus vanilla Stochastic Gradient Descent—perfect for reasoning about what each line does.

```python
import torch.optim as optim

loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(500):
    optimizer.zero_grad()
    preds = model(distances)
    loss = loss_fn(preds, times)
    loss.backward()
    optimizer.step()
```

A few habits to copy from the lab:

- `zero_grad()` before every backward pass to avoid gradient accumulation.
- Keep the loop small enough to print intermediate losses; 500 epochs on four samples finishes instantly.
- Track both the loss and the learned `weight`/`bias` so you can compare against intuition (roughly six minutes per mile in the courier data).

## Predict, Interpret, Repeat

Once the neuron settles, plug in the 7-mile request:

```python
with torch.no_grad():
    eta_minutes = model(torch.tensor([[7.0]])).item()
```

In the curated bike-only dataset, the model predicts just under 42 minutes—late for the service-level promise. When the notebook swaps in a mixed bike + car dataset, the linear neuron struggles, showing why capacity matters. Treat that failure as a signal to evolve the architecture, not as a reason to abandon PyTorch.

## Where to Go Next

- Add polynomial or ReLU-activated hidden layers to capture the curved relationship in the second dataset.
- Wrap the tensors with `DataLoader` once you have more than a handful of points.
- Keep the ML pipeline checklist handy: every new dataset still flows through ingestion, preparation, modeling, training, and validation.

Bookmark this Torch 101 primer whenever you need to re-ground yourself before reaching for fancier architectures.
