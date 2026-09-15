"""A U-Net conditioned on time and digit labels."""

import torch
from torch import nn
from torch.nn import functional as F


class ConditionalBlock(nn.Module):
    """Two convolutions, time/class conditioning, and a residual connection."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.condition = nn.Linear(128, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, condition):
        h = self.conv1(x)
        h = h + self.condition(condition)[:, :, None, None]  # [B, C, 1, 1], broadcast over pixels
        h = F.silu(self.norm1(h))
        h = self.norm2(self.conv2(h))
        return F.silu(h + self.skip(x))


class VelocityField(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.time = nn.Sequential(nn.Linear(1, 128), nn.SiLU(), nn.Linear(128, 128))
        self.label = nn.Embedding(10, 128)
        self.down1 = ConditionalBlock(1, channels)
        self.down2 = ConditionalBlock(channels, channels * 2)
        self.middle = ConditionalBlock(channels * 2, channels * 4)
        self.up2 = ConditionalBlock(channels * 6, channels * 2)
        self.up1 = ConditionalBlock(channels * 3, channels)
        self.output = nn.Conv2d(channels, 1, 1)

    def forward(self, x, t, labels):
        # t: [B, 1], labels: [B] integer digits → condition: [B, 128]
        condition = self.time(t) + self.label(labels)
        h1 = self.down1(x, condition)  # [B, 1, 28, 28] → [B, C, 28, 28]
        h2 = self.down2(F.avg_pool2d(h1, 2), condition)  # [B, 2C, 14, 14]
        h = self.middle(F.avg_pool2d(h2, 2), condition)  # [B, 4C, 7, 7]
        h = F.interpolate(h, scale_factor=2, mode="nearest")  # [B, 4C, 14, 14]
        h = self.up2(torch.cat((h, h2), dim=1), condition)  # [B, 2C, 14, 14]
        h = F.interpolate(h, scale_factor=2, mode="nearest")  # [B, 2C, 28, 28]
        h = self.up1(torch.cat((h, h1), dim=1), condition)  # [B, C, 28, 28]
        return self.output(h)  # [B, 1, 28, 28], velocity rather than pixel intensity
