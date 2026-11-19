"""Generic CSR and affine composition around restored learned weights."""
import ctypes as C
import math
import torch
from cuda_kernel_runtime_v1 import CudaKernels, check
from fused_cuda_generator_v1 import pointer
from recurrent_sampler_native import dimensions


class LearnedGatherAffine:
    def __init__(self, build):
        self.kernels=CudaKernels(build)
        for name in ('learned_gather_three_affine','learned_gather_ordered_rhs','learned_gather_difference_scale'):
            entry=C.c_void_p()
            check(self.kernels.driver.cuModuleGetFunction(C.byref(entry),self.kernels.module,name.encode()),'get generic gather readout')
            self.kernels.functions[name]=entry

    @staticmethod
    def inputs(fields):
        if not 1<=len(fields)<=3 or any(v.device.type!='cuda' or v.ndim!=3 or v.dtype!=fields[0].dtype or not v.is_contiguous() for v in fields) or fields[0].dtype not in (torch.float32,torch.float64):
            raise ValueError('Matching native contiguous learned CUDA inputs required')
        padded=list(fields)+[None]*(3-len(fields))
        sizes=[tuple(reversed(v.shape)) if v is not None else (1,1,1) for v in padded]
        return [*(pointer(v) for v in padded),*(C.c_int(n) for size in sizes for n in size)]

    def three(self, core, fields, base, counts, name, scale, multiply_only=False, as_double=False):
        inputs=self.inputs(fields);sizes=dimensions(counts)[:3]
        if not multiply_only and (len(base)!=3 or any(v.dtype!=torch.float32 or v.device.type!='cuda' or tuple(v.shape)!=s or not v.is_contiguous() for v,s in zip(base,sizes))):
            raise ValueError('Complete native float base fields required')
        if multiply_only:base=[None]*3
        out=[torch.empty(s,device='cuda',dtype=torch.float64 if as_double else torch.float32) for s in sizes]
        args=[*(C.c_int(n) for n in counts),*inputs,C.c_int(fields[0].dtype==torch.float64),
              *(pointer(v) for v in base),*(pointer(v) for v in out),*(pointer(v) for v in core.kernel_sparse[name]),
              C.c_double(1.),C.c_double(scale),C.c_int(multiply_only),C.c_int(as_double)]
        self.kernels.launch('learned_gather_three_affine',sum(math.prod(s) for s in sizes),args)
        return out

    def ordered_rhs(self,core,fields,shape,name,head,scale,denominator):
        inputs=self.inputs(fields)
        if len(fields)!=3 or fields[0].dtype!=torch.float32:raise ValueError('Three float learned inputs required')
        out=torch.empty(shape,device='cuda',dtype=torch.float64)
        args=[*(C.c_int(n) for n in reversed(shape)),*inputs,C.c_int(head),*(pointer(v) for v in core.kernel_sparse[name]),
              C.c_double(1.),C.c_double(scale),C.c_double(denominator),pointer(out)]
        self.kernels.launch('learned_gather_ordered_rhs',out.numel(),args)
        return out

    def difference(self,core,fields,base,shape,name,head,sign,scale):
        inputs=self.inputs(fields)
        if base.device.type!='cuda' or base.dtype!=torch.float64 or tuple(base.shape)!=shape or not base.is_contiguous():raise ValueError('Matching double readout base required')
        out=torch.empty(shape,device='cuda',dtype=torch.float64)
        args=[*(C.c_int(n) for n in reversed(shape)),*inputs,C.c_int(fields[0].dtype==torch.float64),C.c_int(head),
              *(pointer(v) for v in core.kernel_sparse[name]),C.c_double(sign),pointer(base),C.c_double(scale),pointer(out)]
        self.kernels.launch('learned_gather_difference_scale',out.numel(),args)
        return out

    def close(self):self.kernels.close()
