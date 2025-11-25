+++
title = "Torch 201: Pre-trained Models and Visual Inference"
date = "2025-11-25T18:16:00Z"
type = "post"
draft = false
tags = ["deep learning", "torch", "computer vision", "tutorial"]
categories = ["deep_learning"]
description = "Leveraging torchvision's pre-trained models for immediate computer vision results—from bounding boxes to pixel-perfect segmentation masks."
+++

Building a major model from scratch takes massive datasets and weeks of GPU time. TorchVision offers a professional shortcut: state-of-the-art pre-trained models trained on ImageNet, COCO, and other benchmark datasets. The `content/ipynb/C2_M2_Lab_3_torchvision_3.ipynb` notebook demonstrates how to use these expert models for immediate inference on classification, segmentation, and object detection tasks. This post distills that workflow into a practical reference.

## Why Pre-trained Models Matter

Training ResNet-50 or Faster R-CNN from scratch requires:
- Millions of labeled images
- Days of distributed GPU compute
- Careful hyperparameter tuning

Pre-trained models give you that learned knowledge instantly. You download weights, run inference, and get production-grade results in minutes—not months.

## The Two-Step Investigation: Know Your Model

Before running inference, answer two critical questions:

1. **How many classes** can this model predict?
2. **What are those class names?**

Modern TorchVision models embed this information in `.meta` attributes. For example, checking `DeepLabV3_ResNet50_Weights.DEFAULT`:

```python
import torchvision.models as tv_models

# Load model and weights object
seg_model_weights = tv_models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT

# Access metadata
if hasattr(seg_model_weights, 'meta') and "categories" in seg_model_weights.meta:
    class_names = seg_model_weights.meta["categories"]
    print(f"Model recognizes {len(class_names)} classes")
    print(class_names)  # ['background', 'aeroplane', 'bicycle', 'bird', ...]
```

This confirms that `'dog'` is in the class list before you waste time on inference.

### Legacy Models: Manual Detective Work

Older models loaded via `pretrained=True` lack the `.meta` attribute. You inspect the architecture directly:

```python
resnet50_model = tv_models.resnet50(pretrained=True)
print(resnet50_model)  # Look for the final layer

# Output includes: (fc): Linear(in_features=2048, out_features=1000, bias=True)
num_classes = resnet50_model.fc.out_features  # 1000
```

Then find the class names in external files like `imagenet_class_index.json`. This manual step is only necessary for legacy loading methods—modern weights objects handle it automatically.

## Visualizing Predictions: Draw What You Detect

Raw tensor outputs are abstract until you draw them. TorchVision provides two essential utilities:

### 1. Bounding Boxes

Object detection models return coordinates. `draw_bounding_boxes` turns them into visual frames:

```python
from torchvision.utils import draw_bounding_boxes
from torchvision.io import decode_image

image = decode_image('./dog1.jpg')
boxes = torch.tensor([[140, 30, 375, 315], [200, 70, 230, 110]], dtype=torch.float)
labels = ["dog", "eye"]

result = draw_bounding_boxes(
    image=image,
    boxes=boxes,  # Shape: (N, 4) for (xmin, ymin, xmax, ymax)
    labels=labels,
    colors=["red", "blue"],
    width=3
)
```

This is the exact technique used in self-driving car dashboards and automated checkout systems.

### 2. Segmentation Masks

For pixel-perfect object boundaries, `draw_segmentation_masks` overlays boolean masks:

```python
from torchvision.utils import draw_segmentation_masks

# object_mask: boolean tensor of shape (1, H, W)
result = draw_segmentation_masks(
    image=image,
    masks=object_mask,
    alpha=0.5,  # Transparency
    colors=["blue"]
)
```

Medical imaging and autonomous driving rely on this precision to outline tumors or road boundaries.

## Inference Workflow 1: Image Segmentation

Using `DeepLabV3_ResNet50` to find and mask a dog:

```python
from PIL import Image
from torchvision import transforms

# 1. Load model
seg_model = tv_models.segmentation.deeplabv3_resnet50(
    weights=tv_models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
).eval()

# 2. Prepare tensors
img = Image.open('./dog2.jpg')
original_image_tensor = transforms.ToTensor()(img)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
input_tensor = normalize(original_image_tensor).unsqueeze(0)

# 3. Inference
with torch.no_grad():
    output = seg_model(input_tensor)['out'][0]

# 4. Generate mask
output_predictions = output.argmax(0)  # Best class per pixel
dog_class_idx = class_names.index('dog')
dog_mask = (output_predictions == dog_class_idx).unsqueeze(0)

# 5. Visualize
result = draw_segmentation_masks(
    image=(original_image_tensor * 255).byte(),
    masks=dog_mask,
    alpha=0.5,
    colors=["blue"]
)
```

The model outputs a tensor of shape `(num_classes, H, W)` with scores per pixel. You take the `argmax` to get the winning class, then filter for your target.

## Inference Workflow 2: Image Classification

Using legacy `ResNet50` to predict the main subject:

```python
# 1. Load model and class names
resnet50_model = tv_models.resnet50(pretrained=True).eval()
imagenet_classes = load_imagenet_classes('./imagenet_class_index.json')

# 2. Transform image to model's expected format
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_batch = transform(img).unsqueeze(0)

# 3. Inference
with torch.no_grad():
    output = resnet50_model(input_batch)

# 4. Convert to probabilities and get top predictions
probabilities = torch.nn.functional.softmax(output[0], dim=0)
top_prob, top_catid = torch.topk(probabilities, 5)

# 5. Display results
for i in range(5):
    class_id_str = str(top_catid[i].item())
    class_name = imagenet_classes[class_id_str][1]
    confidence = top_prob[i].item() * 100
    print(f"Top-{i+1}: {class_name} ({confidence:.2f}%)")
```

The model correctly identifies `'golden_retriever'` with high confidence.

## Object Detection: Faster R-CNN in Action

For detecting multiple objects with bounding boxes:

```python
# Load Faster R-CNN model
bb_model_weights = tv_models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
bb_model = tv_models.detection.fasterrcnn_resnet50_fpn(weights=bb_model_weights).eval()

# Define targets
target_class_names = ['car', 'traffic light']
bbox_colors = ['red', 'blue']

# Inference
tensor_image_batch = transforms.ToTensor()(pil_image).unsqueeze(0)
with torch.no_grad():
    prediction = bb_model(tensor_image_batch)[0]

# Filter by class and confidence threshold
for index, label, color in zip(object_indices, target_class_names, bbox_colors):
    class_mask = (prediction['labels'] == index) & (prediction['scores'] > 0.7)
    boxes = prediction['boxes'][class_mask]
    # Draw boxes...
```

The model returns dictionaries with `'boxes'`, `'labels'`, and `'scores'` keys—filter and visualize as needed.

## Key Architecture Catalog

TorchVision organizes models by task:

- **Classification**: ResNet, VGG, MobileNetV3, DenseNet
- **Segmentation**: FCN, DeepLabV3
- **Object Detection**: Faster R-CNN, RetinaNet, SSD
- **Video**: R(2+1)D, MC3, Video MViT

Always use `.DEFAULT` weights for the current best practice, and check the `.meta` attribute first.

## Critical Habits for Production Inference

1. **Match preprocessing to training data**: Models trained on ImageNet expect 224×224 inputs with specific normalization. Mismatches degrade performance silently.

2. **Check class lists before inference**: Avoid wasting compute on models that don't recognize your target objects.

3. **Use `.eval()` mode**: Disables dropout and batch normalization updates during inference.

4. **Wrap inference in `torch.no_grad()`**: Prevents gradient computation, saving memory and speeding up predictions.

5. **Validate with visualization**: Always draw bounding boxes or masks to verify model behavior before trusting predictions.

## Where to Go Next

- **Transfer Learning**: Fine-tune these pre-trained models on custom datasets (coming in the next lab).
- **Custom Datasets**: Wrap your data in `DataLoader` and adapt model heads for specialized classes.
- **Model Export**: Convert to ONNX or TorchScript for deployment to edge devices.

Pre-trained models aren't a shortcut—they're the standard, efficient path to production vision systems. Master inference first, then customize through fine-tuning.
