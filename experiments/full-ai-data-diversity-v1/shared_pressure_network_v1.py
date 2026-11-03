"""Shared temporal trunk with independent, zero-initialized pressure RHS heads.

CPU construction supports typed restoration only; all learned forwards remain
CUDA-only. Existing trunk and velocity/tracer tensors retain their old seed.
"""
import torch
from torch import nn
from torch.nn import functional as F

from shared_temporal_network_v1 import SharedTemporalResidual, heads


class SharedPressureResidual(SharedTemporalResidual):
    def __init__(self, hidden=4, device='cuda', seed=27183):
        super().__init__(hidden=hidden, device=device, seed=seed)
        device = torch.device(device)
        devices = [torch.cuda.current_device()] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed + 1)
            self.endpoint_pressure = nn.Conv3d(hidden, 1, 1, bias=False, device=device, dtype=torch.float32)
            self.interior_pressure = nn.Conv3d(hidden, 1, 1, bias=False, device=device, dtype=torch.float32)
            nn.init.zeros_(self.endpoint_pressure.weight)
            nn.init.zeros_(self.interior_pressure.weight)

    @staticmethod
    def pressure(hidden, layer, active):
        value = layer(hidden)[0, 0]
        return torch.where(active, value, value.new_zeros(()))

    def endpoint(self, hidden, active):
        result = super().endpoint(hidden, active)
        return [*result, self.pressure(hidden, self.endpoint_pressure, active)]

    def interior(self, hidden, conditioning, active):
        if (hidden.device.type != 'cuda' or conditioning.device.type != 'cuda'
                or conditioning.dtype != torch.float32 or conditioning.shape[1] != 8
                or conditioning.shape[-3:] != hidden.shape[-3:] or active.device.type != 'cuda'):
            raise ValueError('CUDA-only predicted-endpoint interior conditioning')
        value = F.silu(self.interior_mix(torch.cat((hidden, conditioning), 1)))
        return [*heads(value, self.interior_velocity, self.interior_tracer, active),
                self.pressure(value, self.interior_pressure, active)]
