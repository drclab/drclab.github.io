+++
title = "Torch 402: Saliency Maps and Grad-CAM"
date = "2025-12-10T00:00:00Z"
type = "post"
draft = false
tags = ["deep learning", "torch", "cnn", "visualization", "interpretability", "grad-cam"]
categories = ["deep_learning"]
description = "Understanding CNN decisions through saliency maps and class activation mapping."
+++

Deep neural networks are powerful, but they often behave like "black boxes": you input an image and get a prediction—but how does the model decide what it sees? **Saliency maps** and **Grad-CAM** are two powerful visualization techniques that help you peek inside a model's decision process, revealing which areas of an image matter most for a particular prediction.

## Saliency Maps

A **saliency map** visualizes which parts of an image are most important for the model's decision. Think of it as a heatmap where "hot" pixels strongly influence the prediction.

### Theory

For an input image $x$ and model output $f(x)$, if you're interested in class $c$, the saliency is:

$$S_{i, j} = \left| \frac{\partial f_c(x)}{\partial x_{i, j}} \right|$$

This measures how much changing each pixel would change the score for class $c$. In practice:
1. Compute the gradient of the output for class $c$ with respect to each input pixel
2. Take the absolute value and sum across color channels
3. This gives a heatmap showing which pixels most affect the prediction

### Implementation

```python
def compute_saliency_map(model, input_image, target_class=None):
    """Compute saliency map using gradients."""
    input_image = input_image.clone().detach()
    input_image.requires_grad_()
    
    output = model(input_image)
    
    # Determine target class
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    
    model.zero_grad()
    output[0, target_class].backward()
    
    # Get gradients and create saliency map
    gradients = input_image.grad.data[0]
    saliency_map = torch.abs(gradients).sum(dim=0).cpu().numpy()
    
    # Normalize to [0, 1]
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
    
    return saliency_map
```

## Grad-CAM: Class Activation Mapping

**Grad-CAM** (Gradient-weighted Class Activation Mapping) shows which spatial regions influenced the model's decision. Unlike saliency maps which highlight individual pixels, Grad-CAM reveals broader regions of interest.

### Theory

Grad-CAM combines the model's "attention" from the last convolutional layer with gradients of the output:

1. Pass the image through the CNN and save activations from the last conv layer
2. Compute gradients of the class score with respect to these feature maps
3. Average the gradients to get importance weights for each channel
4. Weight the feature maps and sum them
5. Apply ReLU (keep only positive influences)
6. Upsample and overlay on the input image

If $A^k$ is the $k$-th feature map and $y^c$ is the output for class $c$:

- Importance weight: $\alpha_k^c = \frac{1}{Z} \sum_i\sum_j \frac{\partial y^c}{\partial A^k_{ij}}$
- Grad-CAM: $L^{c}_{\text{Grad-CAM}} = \mathrm{ReLU}\left(\sum_k \alpha_k^c A^k\right)$

### Implementation

```python
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.target_layer.register_forward_hook(self._on_forward)
    
    def _on_forward(self, module, inputs, output):
        self.activations = output.detach()
        def _on_backward(grad):
            self.gradients = grad.detach()
        output.register_hook(_on_backward)
    
    def __call__(self, x, class_idx=None):
        self.model.zero_grad(set_to_none=True)
        output = self.model(x)
        
        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())
        
        score = output[:, class_idx].sum()
        score.backward()
        
        # Channel-wise mean of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=False)
        cam = cam.relu()[0]
        
        # Normalize to [0, 1]
        cam -= cam.min()
        cam /= cam.max().clamp_min(1e-8)
        
        return cam.detach().cpu().numpy(), class_idx

# Usage with ResNet
grad_cam = GradCAM(model, model.layer4[-1].conv3)
heatmap, class_idx = grad_cam(input_tensor, target_class)
```

## Comparison

*   **Saliency Maps**: Pixel-level detail, shows exactly which pixels matter. Can be noisy.
*   **Grad-CAM**: Region-level understanding, shows which areas matter. Smoother and easier to interpret.

Both techniques are complementary—saliency gives precision, Grad-CAM gives context.

## Conclusion

Saliency maps and Grad-CAM transform CNNs from black boxes into interpretable systems. By visualizing what the model "sees," you can debug failures, build trust, and gain insights into how deep learning works. These tools are essential for production systems where understanding model decisions is critical.
