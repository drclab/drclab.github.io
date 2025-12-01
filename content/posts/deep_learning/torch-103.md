+++
title = "Torch 103: Debugging, Modularizing, and Inspecting Models"
date = "2025-12-03T02:34:04-05:00"
type = "post"
draft = false
tags = ["deep learning", "torch", "debugging", "inspection"]
categories = ["deep_learning"]
description = "Learn how to debug shape mismatches, refactor for modularity, and inspect complex models in PyTorch."
+++

In the real world, your first attempt at a model rarely works perfectly. You'll often encounter cryptic error messages about mismatched tensor shapes, or worse, your model will run without errors but fail to produce meaningful results.

This post, based on the `content/ipynb/deepL/C1_M4_Lab_2_debugging.ipynb` lab, covers three essential skills for a model investigator: **debugging**, **modularization**, and **inspection**.

## Debugging: The Forward Pass

When you encounter a dimension mismatch error (e.g., `mat1 and mat2 shapes cannot be multiplied`), the error message often doesn't specify *where* the issue is. The dynamic nature of PyTorch allows you to insert print statements directly into the `forward` method to trace tensor shapes.

Instead of guessing, define a debug version of your model:

```python
class SimpleCNNDebug(SimpleCNN):
    def forward(self, x):
        print("Input shape:", x.shape)
        
        # Check conv layer
        x = self.pool(self.relu(self.conv(x)))
        print("(Activation) After pooling:", x.shape)

        # Check linear layer inputs
        # This is where you often find you forgot to flatten!
        print("(Layer components) fc1 weights:", self.fc1.weight.shape)
        
        x = self.relu_fc(self.fc1(x))
        x = self.fc2(x)
        return x
```

Running a single batch through this debug model will reveal exactly where the input shape doesn't match the layer's expectation (often a missing `torch.flatten`).

## Modularization with `nn.Sequential`

Once your model is working, the `forward` method might look verbose. `nn.Sequential` allows you to group layers into logical blocks, making your code cleaner, more modular, and less error-prone.

```python
class SimpleCNN2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional Block
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Fully Connected Block
        self.fc_block = nn.Sequential(
            nn.Linear(32 * 14 * 14, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_block(x)
        return x
```

## Model Inspection

### Activation Statistics
A model that runs doesn't always learn. A common sanity check is to inspect activation statistics to ensure they aren't exploding or vanishing.

```python
def get_statistics(activation):
    print(f" Mean: {activation.mean().item()}")
    print(f" Std: {activation.std().item()}")
```

### Exploring Complex Architectures
For large, pre-trained models like SqueezeNet, printing the entire object is overwhelming. Use `.named_children()` to iterate through top-level blocks:

```python
from torchvision.models import SqueezeNet
model = SqueezeNet()

for name, block in model.named_children():
    print(f"Block {name} has {len(list(block.children()))} layers")
```

You can also programmatically count parameters to understand model complexity:

```python
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
```

## Conclusion

Debugging is not just about fixing errors; it's about understanding your model's flow. By inspecting shapes during the forward pass, structuring code with `nn.Sequential`, and verifying statistics, you build a robust foundation for tackling more complex deep learning challenges.
