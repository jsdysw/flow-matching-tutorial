"""Load MNIST images and their digit labels."""

from torchvision.datasets import MNIST


def load_mnist():
    dataset = MNIST(root="data", train=True, download=True)
    images = dataset.data.float() / 127.5 - 1  # [60000, 28, 28], pixels in [-1, 1]
    images = images.unsqueeze(1)  # [60000, 1, 28, 28], grayscale channel first
    labels = dataset.targets  # [60000], int64 digits from 0 to 9
    return images, labels
