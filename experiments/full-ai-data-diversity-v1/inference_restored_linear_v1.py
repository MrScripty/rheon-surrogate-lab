"""Use existing ordered learned CUDA gathers when autograd is disabled.

Training retains the qualified differentiable CSR implementation. Both paths
consume the same restored learned coefficients, channels and offsets. No new
physical stencil or inverse is constructed here.
"""
from types import SimpleNamespace

import torch

from coupled_macrostep_v1 import RestoredLinear
from fused_cuda_generator_v1 import FusedCudaGenerator


class InferenceRestoredLinear(RestoredLinear):
    def __init__(self, model):
        super().__init__(model)
        self.execution = SimpleNamespace(fusion_mode='transport_and_linear',
            device=model.device, sparse=model.sparse,
            kernel_sparse=model.kernel_sparse, kernels=model.kernels)

    def __call__(self, fields, shape, name, head, selected_channel=None, sign=1):
        if torch.is_grad_enabled() or any(v.ndim != 3 for v in fields):
            return super().__call__(fields, shape, name, head, selected_channel, sign)
        return FusedCudaGenerator.linear(self.execution, fields, shape, name,
            head, selected_channel, sign)
