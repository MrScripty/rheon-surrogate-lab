"""Explicit Torch-owned CUDA context before standalone driver module loading."""
import torch
from cuda_known_controls_v1 import CudaKnownControls as CudaKnownControlsV1

class CudaKnownControls(CudaKnownControlsV1):
    def __init__(self,build):
        # Device-property queries alone need not create a current CUDA context.
        # An actual Torch allocation binds the driver module to Torch's context.
        self._context_marker=torch.empty(1,dtype=torch.float64,device='cuda')
        super().__init__(build)
