"""Exact ordered sum of prescribed forces without dense per-fine force fields."""
import ctypes as C

import torch

from cuda_kernel_runtime_v1 import CudaKernels, check
from fused_cuda_generator_v1 import pointer
from recurrent_sampler_native import dimensions


class CudaKnownControlSum:
    def __init__(self, build):
        self.kernels = CudaKernels(build)
        for name in ('known_axes_batched', 'known_control_ordered_sum'):
            entry = C.c_void_p()
            check(self.kernels.driver.cuModuleGetFunction(C.byref(entry), self.kernels.module, name.encode()), name)
            self.kernels.functions[name] = entry
        self.identity = self.kernels.identity

    def sum_from_packet(self, counts, packet):
        if (len(counts) != 3 or any(type(n) is not int or n < 1 for n in counts)
                or packet.device.type != 'cuda' or packet.dtype != torch.float64
                or packet.ndim != 2 or packet.shape[1] != 18 or not packet.is_contiguous()
                or not 1 <= packet.shape[0] <= 32):
            raise ValueError('Explicit CUDA F64 whole-block control packet and grid counts required')
        length, width = max(counts) + 1, packet.shape[0]
        tables = torch.empty((width, 3, 3, length, 3), dtype=torch.float64, device='cuda')
        args = [*(C.c_int(n) for n in counts), C.c_int(length), C.c_int(width), pointer(packet), pointer(tables)]
        self.kernels.launch('known_axes_batched', width * 9 * length, args)
        outputs = [torch.empty(s, dtype=torch.float64, device='cuda') for s in dimensions(counts)[:3]]
        args = [*(C.c_int(n) for n in counts), C.c_int(length), C.c_int(width),
                pointer(tables), pointer(packet), *(pointer(v) for v in outputs)]
        self.kernels.launch('known_control_ordered_sum', sum(v.numel() for v in outputs), args)
        return outputs

    def close(self):
        self.kernels.close()
