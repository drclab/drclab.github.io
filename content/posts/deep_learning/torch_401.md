+++
title = "Torch 401: Visualizing and Interpreting CNNs"
date = "2025-12-09T00:00:00Z"
type = "post"
draft = false
tags = ["deep learning", "torch", "cnn", "visualization", "interpretability"]
categories = ["deep_learning"]
description = "Peeking inside the black box: Visualizing feature maps, using hook functions, and understanding receptive fields."
+++

Understanding the inner workings of Convolutional Neural Networks (CNNs) is crucial for developing intuition about how these models interpret images. In this post, we will explore how to visualize internal layers, use hook functions to extract intermediate outputs, and compute the receptive field of neurons.

## Visualizing Layers

To understand what a CNN detects, we can visualize the feature maps of its layers.

### Convolutional Layer

When an image passes through a convolutional layer, it is transformed into a set of feature maps. These maps highlight specific features like edges, textures, or shapes.

```python
import torch.nn as nn

# Define a simple Conv2d layer
conv_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
output = conv_layer(input_tensor)
```

By visualizing the output channels, we can see which regions of the input activate specific filters. Early layers often detect simple edges, while deeper layers capture more complex patterns.

### Pooling Layer

Pooling layers reduce the spatial dimensions of feature maps, balancing local details with global context.

**Max Pooling** takes the maximum value in a window:
$$y_{i,j} = \max_{m,n \in R_{i,j}} x_{m,n}$$

**Average Pooling** takes the average value:
$$y_{i,j} = \frac{1}{k^2} \sum_{m,n \in R_{i,j}} x_{m,n}$$

Pooling helps the network become invariant to small translations and reduces computational cost.

## Hook Functions

Hooks are powerful tools in PyTorch that allow you to intercept and inspect the flow of data inside a model. **Forward hooks** are particularly useful for capturing intermediate activations without modifying the model's architecture.

To use a hook, you define a function that receives the layer, input, and output, and then register it to a specific layer.

```python
# Dictionary to store activations
activations = {}

def grab(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# Register the hook
model.layer1.register_forward_hook(grab('layer1'))

# Run forward pass
output = model(input_image)

# Access captured activations
print(activations['layer1'].shape)
```

This technique is essential for debugging and for visualizing what features each layer is detecting.

## Receptive Field

The **receptive field** of a neuron is the specific region of the input image that influences its value. It tells us how much "context" a neuron can see.

*   **Small receptive field**: The neuron sees a small, local part of the input.
*   **Large receptive field**: The neuron sees a larger, more global part of the input.

### Calculation

The receptive field grows as we stack layers. For a layer with kernel size $k$ and stride $s$, the receptive field $r$ and jump $j$ (effective stride) update as follows:

$$r_{new} = r_{prev} + (k - 1) \times j_{prev}$$
$$j_{new} = j_{prev} \times s$$

Understanding the receptive field helps ensure your network is capturing enough context to make accurate predictions.

## Conclusion

Visualizing CNNs transforms them from "black boxes" into understandable systems. By examining feature maps, we see the progression from simple edges to complex objects. Hook functions give us the surgical ability to inspect any part of the network, and understanding receptive fields helps us design architectures with the right amount of context. These tools are fundamental for debugging, interpreting, and improving deep learning models.
