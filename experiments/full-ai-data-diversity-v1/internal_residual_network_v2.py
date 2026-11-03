"""Same learned parameters with an exact zero correction for quiescent inputs.

Only current-state, coarse-change and known-force channels control activity.
Tracer content and geometry cannot create a correction in an otherwise still
scene. All decisions remain CUDA tensor operations, including in CUDA graphs.
"""
import torch
from internal_residual_network_v1 import InternalGroupResidual as BaseInternalGroupResidual
from internal_residual_network_v1 import CHANNELS,HIDDEN,PARAMETERS


def active_input(features):
    return torch.count_nonzero(features[:,:7]).ne(0) | torch.count_nonzero(features[:,8:39]).ne(0)


class InternalGroupResidual(BaseInternalGroupResidual):
    def forward(self,features):
        values=super().forward(features)
        active=active_input(features)
        return [torch.where(active,value,value.new_zeros(())) for value in values]
