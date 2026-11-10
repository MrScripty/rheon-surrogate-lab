"""A 3D U-Net trunk for grids of any size, including odd and non-cubic ones.

Each coarser level halves the grid, so with enough levels one pass carries
information across the whole domain. There are no normalisation layers: residual
blocks start as the identity instead, which keeps field amplitudes intact.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.first = nn.Conv3d(channels, channels, 3, padding=1)
        self.second = nn.Conv3d(channels, channels, 3, padding=1)
        nn.init.zeros_(self.second.weight)
        nn.init.zeros_(self.second.bias)

    def forward(self, x):
        return x + self.second(F.silu(self.first(F.silu(x))))


def _blocks(channels, count):
    return nn.Sequential(*(ResidualBlock(channels) for _ in range(count)))


def _pad_to_even(x):
    """Repeat the last cell along any odd axis so a stride-2 layer covers every cell."""
    z, y, x_cells = x.shape[-3:]
    padding = (0, x_cells % 2, 0, y % 2, 0, z % 2)
    return F.pad(x, padding, mode='replicate') if any(padding) else x


class UNet3d(nn.Module):
    """Maps [B, in_channels, Z, Y, X] to features [B, widths[0], Z, Y, X]."""

    def __init__(self, in_channels, widths=(32, 64, 128), blocks_per_level=2):
        super().__init__()
        if len(widths) < 2:
            raise ValueError('A U-Net needs at least two levels.')
        self.widths = tuple(widths)
        self.stem = nn.Conv3d(in_channels, widths[0], 3, padding=1)
        self.encoders = nn.ModuleList(_blocks(width, blocks_per_level) for width in widths[:-1])
        self.downsamples = nn.ModuleList(
            nn.Conv3d(fine, coarse, 2, stride=2) for fine, coarse in zip(widths[:-1], widths[1:]))
        self.bottleneck = _blocks(widths[-1], blocks_per_level)
        # One vector summarising the whole coarsest grid, added back everywhere: a
        # direct domain-wide path that does not depend on how many levels the grid allows.
        self.global_context = nn.Sequential(
            nn.Linear(widths[-1], widths[-1]), nn.SiLU(), nn.Linear(widths[-1], widths[-1]))
        self.reductions = nn.ModuleList(
            nn.Conv3d(coarse, fine, 1) for fine, coarse in zip(widths[:-1], widths[1:]))
        self.fusions = nn.ModuleList(nn.Conv3d(2 * width, width, 3, padding=1) for width in widths[:-1])
        self.decoders = nn.ModuleList(_blocks(width, blocks_per_level) for width in widths[:-1])

    def forward(self, x):
        x = self.stem(x)
        skips = []
        for encoder, downsample in zip(self.encoders, self.downsamples):
            x = encoder(x)
            skips.append(x)
            x = downsample(_pad_to_even(x))
        x = self.bottleneck(x)
        x = x + self.global_context(x.mean(dim=(-3, -2, -1)))[..., None, None, None]
        for reduce, fuse, decoder, skip in zip(reversed(self.reductions), reversed(self.fusions),
                                               reversed(self.decoders), reversed(skips)):
            x = F.interpolate(reduce(x), size=skip.shape[-3:], mode='trilinear', align_corners=False)
            x = decoder(fuse(torch.cat((x, skip), dim=1)))
        return x
