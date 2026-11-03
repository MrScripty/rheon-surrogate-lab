"""Explicit CPU layout candidate; unchanged mathematical network and float32 weights."""
import copy
import torch

class LargeChannelsLast(torch.nn.Module):
    def __init__(self,wrapper):
        super().__init__()
        self.eager=wrapper
        self.large=copy.deepcopy(wrapper).eval().to(memory_format=torch.channels_last_3d)
    def forward(self,values):
        if max(values.shape[2:])>=64:
            return self.large(values.contiguous(memory_format=torch.channels_last_3d))
        return self.eager(values)
