+++
title = "Torch 201: TorchVision for Pre-Processing"
date = "2025-12-04T00:20:05"
type = "post"
draft = false
tags = ["deep learning", "torch", "computer vision", "tutorial"]
categories = ["deep_learning"]
description = "Mastering image preparation with TorchVision: from basic tensor conversion to building complex data augmentation pipelines."
+++

Before a deep learning model can learn to "see", the visual data you feed it must be carefully prepared. Raw images come in various sizes and formats, but neural networks require a standardized input: a **tensor**.

This is where **TorchVision** comes in. As PyTorch’s standard toolkit for computer vision, it provides powerful and efficient tools designed to handle the common, often tedious, components of a vision workload. Instead of reinventing the wheel, you can use TorchVision’s battle-tested data pipelines and functions to focus on building innovative models.

In this post, we will:

*   Practice converting images between the common **Pillow (PIL)** image format and **PyTorch Tensors**.
*   Explore handy **TorchVision utilities** like `make_grid` and `save_image` to simplify debugging and visualize batches of images.
*   Apply individual transformations to resize, crop, and augment images.
*   Define and implement a **custom transformation** from scratch.
*   Chain transforms together using `transforms.Compose` to build powerful and reusable **preprocessing and data augmentation pipelines**.

## Imports

```python
import os

from IPython.display import Image as DisplayImage
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
from torchvision import datasets
from torchvision.io import decode_image
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm.auto import tqdm

import helper_utils
```

## Image Conversion (PIL and Tensor)

Your first practical step in any computer vision workflow is to bridge the gap between how humans see images and how machines process them. An image file is a grid of pixels, often handled by a library like Pillow (PIL). A neural network, however, requires a numerical format it can perform calculations on: a **Tensor**.

Gaining fluency in converting between these two formats is a foundational skill. It's the mechanism that lets you load, process, and inspect your data at every stage of a project.

To begin, let's perform a "roundtrip" conversion. By taking a PIL image, changing it to a PyTorch Tensor, and then converting it back, you'll confirm that this core operation is seamless and preserves your data perfectly.

### Loading an Image

First, load an image with Pillow. Note that Pillow reports dimensions in `(Width, Height)` format.

```python
# Load an image
image = Image.open('./images/mangoes.jpg')

# Dimensions of the original PIL image
print("Original PIL Image Dimensions:", image.size)
print(f"The maximum pixel value is: {image.getextrema()[0][1]}, and the minimum is: {image.getextrema()[0][0]}")
```

### PIL to Tensor

`transforms.ToTensor()` converts a PIL image object into a PyTorch Tensor.

*   **Dimension Change**: This transform rearranges the image data from Pillow's `(Width, Height)` format to PyTorch's `(Channels, Height, Width)` format.
*   **Scaling**: It scales the image's pixel values from the `[0, 255]` range to a floating point `[0.0, 1.0]` range.

```python
# Convert the PIL image to a PyTorch Tensor
img_tensor = transforms.ToTensor()(image)

# Dimensions (shape) of the tensor
# [C, H, W] format
print(f"Dimensions After Converting to a Tensor: {img_tensor.shape}")
print(f"The maximum pixel value is: {img_tensor.max()}, and the minimum is: {img_tensor.min()}")
```

### Tensor to PIL

`transforms.ToPILImage()` converts a PyTorch Tensor back into a PIL image object.

*   **Dimension Change**: It performs the reverse, converting a `(Channels, Height, Width)` tensor back into a PIL image that reports its size as `(Width, Height)`.

```python
# Convert the tensor back to a PIL image
img_pil = transforms.ToPILImage()(img_tensor)

# Dimensions of the converted back PIL image
print("Dimensions After Converting Back to PIL:", img_pil.size)
```

## TorchVision Utilities for Image Handling

TorchVision equips you with a powerful toolkit for the practical logistics of a computer vision project. These utilities are designed to manage the entire lifecycle of your image data, from initial loading to final output.

We will explore three indispensable functions:

*   `decode_image`: Instantly converts compressed image files like JPEGs or PNGs directly into tensors.
*   `make_grid`: Arranges a batch of images into a clean, single grid.
*   `save_image`: Saves your tensor-based images back to a standard file format.

### Decoding Images into Tensors with `decode_image`

The `decode_image` function efficiently converts an image file directly into a **numerical PyTorch tensor**. Unlike `Image.open`, which returns a visual PIL object, the tensor from `decode_image` is a purely numerical object.

*   **Dimension Ordering**: The output tensor follows the standard PyTorch **channel-first convention** (`[C, H, W]`).

```python
# Define the path to the image file.
image_path = './images/apples.jpg'

# Load the image
image = decode_image(image_path)

print(f"Image tensor dimensions: {image.shape}")
print(f"Image tensor dtype: {image.dtype}")
print(f"The maximum pixel value is: {image.max()}, and the minimum is: {image.min()}\n")
```

### Creating Image Grids with `make_grid`

In deep learning, you almost always process data in **batches**. The `make_grid` function takes a batch of image tensors and arranges them into a single, clean grid for easy inspection.

```python
# Create a batch of images (./images/ contains only 6 images). The images are loaded as 300x300 pixels
images_tensor = helper_utils.load_images("./images/")

# The size is 6 images x 3 color channels x 300 pixels height x 300 pixels width
print(f"Image tensor dimensions: {images_tensor.shape}")

# Make a grid from the loaded images (2 rows of 3 for 6 images)
grid = vutils.make_grid(tensor=images_tensor, nrow=3, padding=5, normalize=True)

# Display the grid of images using a helper function
helper_utils.display_grid(grid)
```

### Saving Tensors as Images with `save_image`

To make your work tangible and shareable, you need to export tensors back into standard image files. `save_image` takes your tensor and efficiently saves it as a high-quality image file.

```python
# Define the path to save the image file.
image_path = "./fruits_grid.png"

# Save the grid as a PNG image
vutils.save_image(tensor=grid, fp=image_path)
```

## Image Transformations and Data Augmentation

Image transformations are used to standardize the size and format of images and to perform **data augmentation**. Augmentation artificially increases the diversity of your training data by creating modified versions of existing images.

The **order in which you apply these transformations matters**. A common practice is:
1.  Geometric transformations (resizing, cropping)
2.  Color and other augmentations
3.  Convert to tensor and normalize

### Common Transformations

Let's look at some individual transformations.

#### Resize

`transforms.Resize` rescales an input PIL image to a desired size.

```python
original_image = Image.open('./images/strawberries.jpg')

# Define the resize transformation (50x50 square)
resize_transform = transforms.Resize(size=50)

# Apply the transformation
resized_image = resize_transform(original_image)
```

#### CenterCrop

`transforms.CenterCrop` extracts a square patch from the center of the image.

```python
# Define the center crop transformation (256x256)
center_crop_transform = transforms.CenterCrop(size=256)

# Apply the transformation
cropped_image = center_crop_transform(original_image)
```

#### RandomResizedCrop

`transforms.RandomResizedCrop` randomly crops a portion of the image and then resizes it. This helps the model become robust to variations in object scale and position.

```python
# Define the RandomResizedCrop transformation (224x224)
random_resized_crop_transform = transforms.RandomResizedCrop(size=224)

# Apply the transformation
cropped_resized_image = random_resized_crop_transform(original_image)
```

#### RandomHorizontalFlip

`transforms.RandomHorizontalFlip` randomly flips the image horizontally.

```python
# Define the horizontal flip transformation
flip_transform = transforms.RandomHorizontalFlip(p=1.0) # p=1.0 guarantees flip for demo

# Apply the transformation
flipped_image = flip_transform(original_image)
```

#### ColorJitter

`transforms.ColorJitter` randomly alters the image's brightness, contrast, and saturation.

```python
# Define the ColorJitter transformation
jitter_transform = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)

# Apply the transformation
jittered_image = jitter_transform(original_image)
```

### Custom Transformations

Sometimes you need a specific operation not built-in. You can create a custom transform by defining a Python class with a `__call__` method.

Here is a custom transform to apply salt and pepper noise:

```python
class SaltAndPepperNoise:
    """
    A custom transform to add salt and pepper noise to a PIL image.
    """
    def __init__(self, salt_vs_pepper=0.5, amount=0.04):
        self.s_vs_p = salt_vs_pepper
        self.amount = amount

    def __call__(self, image):
        # Make a copy of the image
        output = np.copy(np.array(image))

        # Add Salt Noise
        num_salt = np.ceil(self.amount * image.size[0] * image.size[1] * self.s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.size]
        output[coords[1], coords[0]] = 255  

        # Add Pepper Noise
        num_pepper = np.ceil(self.amount * image.size[0] * image.size[1] * (1.0 - self.s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.size]
        output[coords[1], coords[0]] = 0

        # Convert the NumPy array back to a PIL image
        return Image.fromarray(output)

    def __repr__(self):
        return self.__class__.__name__ + f'(salt_vs_pepper={self.s_vs_p}, amount={self.amount})'

# Instantiate and apply
sp_transform = SaltAndPepperNoise(salt_vs_pepper=0.5, amount=0.5)
sp_image = sp_transform(original_image)
```

### Normalize

`transforms.Normalize` standardizes the pixel values of an image tensor by subtracting the mean and dividing by the standard deviation for each channel. This helps the model converge faster.

**Note**: `transforms.ToTensor()` must always be applied before this transformation.

```python
# Convert to tensor (scales to [0, 1])
tensor_image = transforms.ToTensor()(original_image)

# Define the normalization transform using ImageNet stats
normalize_transform = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# Apply the transformation
normalized_tensor = normalize_transform(tensor_image)
```

#### Calculating Dataset Mean and Standard Deviation

While ImageNet stats are a good default, calculating your dataset's specific mean and standard deviation can improve performance, especially when training from scratch.

```python
def calculate_mean_std(dataset):
    """
    Calculates the mean and standard deviation of a PyTorch dataset.
    """
    loader = data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    channel_sum = torch.zeros(3)
    channel_sum_sq = torch.zeros(3)
    num_pixels = 0

    for images, _ in tqdm(loader, desc="Calculating Dataset Stats"):
        num_pixels += images.size(0) * images.size(2) * images.size(3)
        channel_sum += images.sum(dim=[0, 2, 3])
        channel_sum_sq += (images ** 2).sum(dim=[0, 2, 3])

    mean = channel_sum / num_pixels
    std = (channel_sum_sq / num_pixels - mean ** 2) ** 0.5

    return mean, std
```

## Composing Transformations for Data Augmentation

`transforms.Compose` creates a single pipeline that applies a sequence of transformations to an image in order.

Here we create a full augmentation pipeline:

```python
# The full augmentation pipeline with all random transformations
full_augmentation_pipeline = transforms.Compose([
    transforms.RandomResizedCrop(size=224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.05, contrast=0.05),
    SaltAndPepperNoise(amount=0.001),
    transforms.ToTensor(),
    # Using `mean` and `std` values as calculated on the 100x100 images
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # Example values
                         std=[0.229, 0.224, 0.225])
])
```

When you pass this pipeline to a `DataLoader`, the transforms are applied on the fly. With each epoch, your model is fed a unique version of the same original image, virtually increasing the size and diversity of your training data.

## Conclusion

We've covered the core components of image preprocessing and augmentation with TorchVision:

1.  **Conversion**: `Pillow (PIL)` <-> `PyTorch Tensors`.
2.  **Utilities**: `make_grid` and `save_image`.
3.  **Transformations**: Individual transforms like `RandomResizedCrop` and `ColorJitter`.
4.  **Custom Transforms**: Creating your own classes.
5.  **Normalization**: Calculating and applying dataset statistics.
6.  **Pipelines**: Using `transforms.Compose`.

These skills are foundational to building robust and efficient computer vision models.
