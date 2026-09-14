"""2D target distributions."""

import math

import torch


DATASETS = ("moons", "8gaussians", "checkerboard")


def sample_target(n, dataset="moons", *, generator=None):
    """Draw fresh CPU samples; training needs no downloaded dataset."""
    # Generate n points from a chosen target distribution, not a stored dataset.
    # Repeated calls draw new points from the same distribution as the RNG advances.
    # A new generator with the same seed reproduces the same sequence of draws.
    # The * requires generator to be passed by name: generator=g.
    if dataset == "moons":
        theta = torch.rand(n, generator=generator) * math.pi  # [n]
        side = torch.randint(2, (n,), generator=generator).bool()  # [n], bool
        x = torch.where(side, 1 - theta.cos(), theta.cos())  # [n]
        y = torch.where(side, 0.5 - theta.sin(), theta.sin())  # [n]
        points = torch.stack((x - 0.5, y - 0.25), dim=1) * 1.8  # [n], [n] → [n, 2]
        return points + 0.08 * torch.randn(n, 2, generator=generator)  # [n, 2] + [n, 2] → [n, 2]
    if dataset == "8gaussians":
        theta = torch.randint(8, (n,), generator=generator) * (math.pi / 4)  # [n]
        centers = 2.5 * torch.stack((theta.cos(), theta.sin()), dim=1)  # [n], [n] → [n, 2]
        return centers + 0.15 * torch.randn(n, 2, generator=generator)  # [n, 2] + [n, 2] → [n, 2]
    if dataset == "checkerboard":
        col = torch.randint(4, (n,), generator=generator)  # [n]
        row = 2 * torch.randint(2, (n,), generator=generator) + col % 2  # [n]
        return torch.stack((col, row), dim=1) + torch.rand(n, 2, generator=generator) - 2  # [n, 2]
