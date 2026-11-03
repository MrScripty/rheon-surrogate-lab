"""Generic CUDA execution fusion of unchanged restored learned stages."""
import ctypes as C
import math
import torch
from support import ROOT
from expanded_cuda_generator_v2 import ExpandedCudaGeneratorV2
from recurrent_sampler_native import dimensions
from cuda_kernel_runtime_v1 import CudaKernels


def pointer(value):return C.c_void_p(value.data_ptr() if value is not None else 0)


class FusedCudaGenerator(ExpandedCudaGeneratorV2):
    def __init__(self,pair,shared_pair,build,fusion_mode='transport'):
        super().__init__(pair,shared_pair)
        if fusion_mode not in ('transport','transport_and_linear'):raise ValueError('Registered fusion variant required')
        self.fusion_mode=fusion_mode;self.kernels=CudaKernels(build);self.kernel_sparse={}
        for name,(rows,channels,offsets,weights) in self.sparse.items():
            metadata=[]
            for key,value in zip(('rows','channels','offsets'),(rows,channels,offsets)):
                tensor=torch.tensor(value,dtype=torch.int32,device=self.device).contiguous()
                self.register_buffer('_kernel_'+name+'_'+key,tensor);metadata.append(tensor)
            self.kernel_sparse[name]=(*metadata,weights)
        self.asset_identity.update(execution_backend='fused_cuda_v1',fusion_mode=fusion_mode,
            kernel_build=self.kernels.identity,learned_parameter_change=False,field_dtype_change=False,
            fast_math=False,fma_contraction=False)

    def transport(self,fields,counts,alpha):
        shapes=dimensions(counts)
        if len(fields)!=4 or any(value.device!=self.device or value.dtype!=torch.float32 or tuple(value.shape)!=shape for value,shape in zip(fields,shapes)):
            raise ValueError('Complete native CUDA float carriers required')
        inputs=[value.contiguous() for value in fields]
        outputs=[torch.empty(shape,dtype=torch.float32,device=self.device) for shape in shapes]
        arguments=[*(C.c_int(n) for n in counts),*(pointer(v) for v in inputs),*(pointer(v) for v in outputs),
            *(pointer(v) for v in self.kernel_sparse['point']),pointer(self.mid_gate),pointer(self.mix),C.c_double(alpha)]
        self.kernels.launch('learned_transport',sum(math.prod(shape) for shape in shapes),arguments)
        return outputs

    def linear(self,fields,target_shape,name,head,selected_channel=None,sign=1):
        if self.fusion_mode=='transport':return super().linear(fields,target_shape,name,head,selected_channel,sign)
        if (not 1<=len(fields)<=3 or not 0<=head<len(self.sparse[name][0])-1
                or any(v.device!=self.device or v.ndim!=3 or v.dtype!=fields[0].dtype for v in fields)
                or fields[0].dtype not in (torch.float32,torch.float64)):
            raise ValueError('Exact CUDA learned gather layout required')
        inputs=[value.contiguous() for value in fields]+[None]*(3-len(fields))
        shapes=[tuple(reversed(value.shape)) if value is not None else (1,1,1) for value in inputs]
        output=torch.empty(target_shape,dtype=torch.float64,device=self.device)
        arguments=[*(C.c_int(n) for n in reversed(target_shape)),*(pointer(v) for v in inputs),
            *(C.c_int(n) for shape in shapes for n in shape),C.c_int(int(fields[0].dtype==torch.float64)),C.c_int(head),
            C.c_int(-1 if selected_channel is None else selected_channel),C.c_double(sign),
            *(pointer(v) for v in self.kernel_sparse[name]),pointer(output)]
        self.kernels.launch('learned_linear',math.prod(target_shape),arguments)
        return output

    def close(self):
        super().close()
        if hasattr(self,'kernels'):self.kernels.close()
