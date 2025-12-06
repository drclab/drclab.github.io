+++
title = "Torch 202: TorchVision Datasets"
date = "2025-12-05T00:01:00"
type = "post"
draft = false
tags = ["deep learning", "torch", "computer vision", "tutorial"]
categories = ["deep_learning"]
description = "A guide to using TorchVision's built-in datasets, handling custom data with ImageFolder, and generating synthetic data."
+++

Before building any computer vision model, and after defining the problem you are trying to solve, you must answer one important question: **what data will you train it on?** In deep learning, the quality and structure of your dataset are fundamental to your model's performance.

Fortunately, **TorchVision** provides access to a rich collection of well-known, pre-formatted datasets, saving you the effort of writing data loading and preprocessing code from scratch. These datasets are designed to integrate seamlessly into a PyTorch training pipeline.

In this post, we will:

*   Load and inspect a standard built-in dataset like `CIFAR-10`.
*   Load datasets that have unique loading requirements, such as `EMNIST`.
*   Load your own images using the generic `ImageFolder` data loader.
*   Generate placeholder data for testing and debugging using `FakeData`.

## Imports

```python
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import helper_utils

# Set dataset directory
root_dir = './pytorch_datasets'
```

## Using Pre-built Datasets

TorchVision offers a variety of popular, ready-to-use datasets. Here are some common ones:

*   **Image Classification**: [MNIST](https://docs.pytorch.org/vision/0.21/generated/torchvision.datasets.MNIST.html), [Fashion-MNIST](https://docs.pytorch.org/vision/0.21/generated/torchvision.datasets.FashionMNIST.html), [CIFAR-10](https://docs.pytorch.org/vision/0.21/generated/torchvision.datasets.CIFAR10.html), [ImageNet](https://docs.pytorch.org/vision/0.21/generated/torchvision.datasets.ImageNet.html).
*   **Object Detection & Segmentation**: [VOC](https://docs.pytorch.org/vision/0.21/generated/torchvision.datasets.VOCDetection.html), [COCO](https://docs.pytorch.org/vision/0.21/generated/torchvision.datasets.CocoDetection.html).
*   **Video Classification**: [UCF-101](https://docs.pytorch.org/vision/0.21/generated/torchvision.datasets.UCF101.html), [Kinetics](https://docs.pytorch.org/vision/0.21/generated/torchvision.datasets.Kinetics.html).

### Standard Dataset Example: CIFAR10

Let's break down how to initialize a standard dataset like CIFAR-10.

```python
# Initialize the CIFAR-10 training dataset
cifar_dataset = datasets.CIFAR10(
    root=root_dir,      # Path to the directory where the data is/will be stored
    train=True,         # Specify that you want the training split of the dataset
    download=True       # Download the data if it's not found in the root directory
)
```

*   `root`: Where the dataset will be stored.
*   `train`: Selects the split (`True` for training, `False` for test).
*   `download`: Downloads the dataset if missing.

Each item in the dataset is a tuple: `(image, label)`. The image is a PIL Image, and the label is an integer.

```python
# Get the first sample
image, label = cifar_dataset[0]
print(f"Image Type: {type(image)}")
print(f"Image Dimensions: {image.size}")
print(f"Label Type: {type(label)}")
```

#### Preparing the Data with Transformations

PyTorch models require input data to be **tensors**. We can pass a transformation pipeline directly to the dataset's initializer using the `transform` parameter.

```python
# Define a transformations pipeline
cifar_transformation = transforms.Compose([
    transforms.ToTensor(),
    # The mean and std values are standard for the CIFAR-10 dataset
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2023, 0.1994, 0.2010))
])

# Initialize with transform
cifar_dataset = datasets.CIFAR10(root=root_dir, 
                                 train=False, 
                                 download=True,
                                 transform=cifar_transformation)
```

Now, when we access an item, the image is automatically transformed into a tensor.

```python
image, label = cifar_dataset[0]
print(f"Image Shape After Transform: {image.shape}")
```

We can use a `DataLoader` to batch and shuffle the data for training.

```python
cifar_dataloader = data.DataLoader(cifar_dataset, batch_size=8, shuffle=True)
helper_utils.display_images(cifar_dataloader)
```

### Dataset with Special Parameters Example: EMNIST

Some datasets have unique requirements. `EMNIST` (Extended MNIST) is a collection of six different datasets, so it requires a `split` parameter.

```python
# Define the transformation pipeline
emnist_transformation = transforms.Compose([
    transforms.RandomRotation(degrees=(90, 90)),
    transforms.RandomVerticalFlip(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Initialize EMNIST with 'digits' split
emnist_digits_dataset = datasets.EMNIST(root=root_dir,
                                        split='digits',  # Specify the 'digits' split
                                        train=False,
                                        download=True,
                                        transform=emnist_transformation)

emnist_digits_dataloader = data.DataLoader(emnist_digits_dataset, batch_size=8, shuffle=True)
helper_utils.display_images(emnist_digits_dataloader)
```

## Custom and Specialized Datasets

### Loading Custom Images: ImageFolder

`ImageFolder` is a generic dataset loader for your own images. It requires a specific directory structure where each sub-directory corresponds to a class.

Structure:
```
./tiny_fruit_and_vegetable/
├── Apple__Healthy/
│   ├── image1.jpg
│   └── ...
├── Guava__Healthy/
│   ├── image2.jpg
│   └── ...
└── ...
```

`ImageFolder` automatically assigns labels based on folder names.

```python
root_dir = './tiny_fruit_and_vegetable'

# Define a transformation pipeline
image_transformation = transforms.Compose([
    transforms.Resize((100, 100)), # Resize is essential for batching
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Initialize ImageFolder
fruit_dataset = datasets.ImageFolder(root=root_dir,
                                     transform=image_transformation)

fruit_dataloader = data.DataLoader(fruit_dataset, batch_size=8, shuffle=True)
helper_utils.display_images(fruit_dataloader)
```

**Note on Splits**: `ImageFolder` doesn't have a `train` argument. You can split it using `torch.utils.data.random_split` or by organizing your folders into `train` and `test` directories beforehand.

### Generating Synthetic Data: FakeData

`FakeData` generates random images and labels on the fly. It's perfect for testing pipelines without downloading real data.

```python
# Define a transformation pipeline
fake_data_transform = transforms.Compose([
    transforms.ToTensor()
])

# Initialize the FakeData dataset
fake_dataset = datasets.FakeData(
    size=1000,                    # Total number of fake images
    image_size=(3, 32, 32),       # (Channels, Height, Width)
    num_classes=10,               # Number of possible classes
    transform=fake_data_transform # Apply the transformation
)

fake_dataloader = data.DataLoader(fake_dataset, batch_size=8, shuffle=True)
helper_utils.display_images(fake_dataloader)
```

## Conclusion

TorchVision datasets are organized, standardized, and designed to integrate with the entire PyTorch ecosystem. You now have the foundational skills to load:

1.  **Standard Benchmarks**: Like CIFAR-10.
2.  **Specialized Datasets**: Like EMNIST with unique splits.
3.  **Custom Data**: Using `ImageFolder`.
4.  **Synthetic Data**: Using `FakeData` for debugging.

By mastering `torchvision.datasets`, you have a solid foundation to build, test, and scale your computer vision models.
