"""Generic fused pointwise operations around unchanged learned layer results."""
import ctypes as C
import math
import torch
from cuda_kernel_runtime_v1 import CudaKernels, check
from fused_cuda_generator_v1 import pointer
from recurrent_sampler_native import dimensions


class GenericAffineOps:
    def __init__(self, build):
        self.kernels = CudaKernels(build)
        for name in ('affine_three', 'combine_ordered_three', 'difference_scale', 'sum_gauge'):
            entry = C.c_void_p()
            check(self.kernels.driver.cuModuleGetFunction(C.byref(entry), self.kernels.module, name.encode()), 'generic affine entry')
            self.kernels.functions[name] = entry

    def three(self, base, delta, counts, scale=1., multiply_only=False, as_double=False, wall=False):
        sizes = dimensions(counts)[:3]
        if len(delta) != 3 or any(v.device.type != 'cuda' or v.dtype != torch.float64 or tuple(v.shape) != s or not v.is_contiguous() for v, s in zip(delta, sizes)):
            raise ValueError('Complete CUDA double learned deltas required')
        if multiply_only:
            base = [None]*3
        elif len(base) != 3 or any(v.device.type != 'cuda' or v.dtype != torch.float32 or tuple(v.shape) != s or not v.is_contiguous() for v, s in zip(base, sizes)):
            raise ValueError('Complete CUDA float base fields required')
        output = [torch.empty(s, device='cuda', dtype=torch.float64 if as_double else torch.float32) for s in sizes]
        args = [*(C.c_int(n) for n in counts), *(pointer(v) for v in base), *(pointer(v) for v in delta),
                *(pointer(v) for v in output), C.c_double(scale), C.c_int(multiply_only), C.c_int(as_double), C.c_int(wall)]
        self.kernels.launch('affine_three', sum(math.prod(s) for s in sizes), args)
        return output

    def ordered_three(self, fields, scale, denominator):
        self.validate(fields)
        out = torch.empty_like(fields[0])
        args = [C.c_longlong(out.numel()), *(pointer(v) for v in fields), C.c_double(scale), C.c_double(denominator), pointer(out)]
        self.kernels.launch('combine_ordered_three', out.numel(), args)
        return out

    def binary(self, name, a, b, scale=None):
        self.validate([a,b])
        out = torch.empty_like(a)
        args = [C.c_longlong(out.numel()), pointer(a), pointer(b)]
        if scale is not None: args.append(C.c_double(scale))
        args.append(pointer(out))
        self.kernels.launch(name, out.numel(), args)
        return out

    @staticmethod
    def validate(fields):
        if not fields or any(v.device.type != 'cuda' or v.dtype != torch.float64 or v.shape != fields[0].shape or not v.is_contiguous() for v in fields):
            raise ValueError('Matching contiguous double CUDA arrays required')

    def close(self):
        self.kernels.close()
