"""Learned local point-velocity projection; no numerical sampling in inference."""
import torch
from torch import nn


class PointSamplingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv3d(3, 12, 3, padding=1, padding_mode='replicate', bias=False)
        nn.init.zeros_(self.projection.weight)

    def forward(self, velocity):
        return self.projection(velocity)


def objective(network, velocity, target, mask):
    error = network(velocity) - target
    return (error.square() * mask).sum() / (velocity.shape[0] * mask.sum())
