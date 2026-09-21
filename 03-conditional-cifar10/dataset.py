"""Load CIFAR-10 RGB images and their class labels."""

import torch
from torchvision.datasets import CIFAR10


CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")


def load_cifar10():
    dataset = CIFAR10(root="data", train=True, download=True)
    images = torch.from_numpy(dataset.data).permute(0, 3, 1, 2).contiguous()  # [50000, 3, 32, 32], uint8
    labels = torch.tensor(dataset.targets)  # [50000], int64 classes 0–9
    return images, labels  # Convert each batch to [-1, 1] during training.
