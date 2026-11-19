"""Learned 243-coefficient viscosity layer; numerical teachers are training-only."""
import torch
from torch import nn


class ViscosityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection=nn.Conv3d(3,3,3,padding=1,padding_mode='replicate',bias=False)
        nn.init.zeros_(self.projection.weight)

    def forward(self,velocity):return self.projection(velocity)


def objective(network,velocity,target,mask):
    return ((network(velocity)-target).square()*mask).sum()/(velocity.shape[0]*mask.sum())
