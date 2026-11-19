"""Original N1 and direct-pressure N8 with one changed F64 interpolation module."""
import ctypes as C
import math
import types
import torch
from absolute_pressure_graph_v3 import AbsolutePressureGraph as ParentGraph, predict as parent_predict
from complete_native_entry_v1 import CompleteNativeEntry
from cuda_kernel_runtime_v1 import CudaKernels, check
from debug.grid_descriptor_graph_v1 import SourceTable, Shape
from recurrent_sampler_native import dimensions
from fused_cuda_generator_v1 import pointer


def transport(core, fields, counts, alpha):
    if not core._nested_enabled:
        return core._nested_original_transport(fields, counts, alpha)
    counts = tuple(counts)
    if core.transport_geometry is None: raise ValueError('Explicit physical sampler geometry required')
    h, dt = core.transport_geometry
    if alpha != dt/h: raise ValueError('Physical step/spacing and proposal scale disagree')
    shapes = dimensions(counts)
    if len(fields) != 4 or any(v.device.type != 'cuda' or v.dtype != torch.float32 or tuple(v.shape) != s for v, s in zip(fields, shapes)):
        raise ValueError('Complete native CUDA carriers required')
    inputs = [v.contiguous() for v in fields]
    outputs = [torch.empty(s, device='cuda', dtype=torch.float32) for s in shapes]
    lengths = [math.prod(s) for s in shapes]
    kernels = core._nested_kernels
    if counts in core._nested_geometry_functions:
        descriptor = SourceTable((Shape*4)(*(Shape(*reversed(s)) for s in shapes)),
                                 (C.c_void_p*4)(*(v.data_ptr() for v in inputs)),
                                 (C.c_void_p*4)(*(v.data_ptr() for v in outputs)),
                                 (C.c_longlong*4)(*lengths), sum(lengths))
        arguments = [descriptor, *(pointer(v) for v in core.kernel_sparse['point']), pointer(core.mid_gate), pointer(core.mix), C.c_double(dt), C.c_double(h)]
        pointers = (C.c_void_p*len(arguments))(*(C.addressof(v) for v in arguments))
        check(kernels.driver.cuLaunchKernel(core._nested_geometry_functions[counts], (max(lengths)+127)//128, 4, 1, 128, 1, 1, 0,
                                            C.c_void_p(torch.cuda.current_stream().cuda_stream), pointers, None), 'nestedF64geometrytransport')
    else:
        arguments = [*(C.c_int(n) for n in counts), *(pointer(v) for v in inputs), *(pointer(v) for v in outputs),
                     *(pointer(v) for v in core.kernel_sparse['point']), pointer(core.mid_gate), pointer(core.mix), C.c_double(dt), C.c_double(h)]
        kernels.launch('learned_transport', sum(lengths), arguments)
    return outputs


@torch.inference_mode()
def predict(core, fields, scene, step, horizons, external, substeps=1):
    if substeps not in (1, 8): raise ValueError('Known original N1/N8 integration metadata required')
    core._nested_enabled = substeps == 8
    return parent_predict(core, fields, scene, step, horizons, external, substeps)


class NestedF64PressureGraph(ParentGraph):
    def __init__(self, pair, shared_pair, build, control_build, sampler_build='runs/full-ai-data-diversity-v1/cuda-fusion-build-19'):
        super().__init__(pair, shared_pair, build, control_build)
        self.nested_kernels = CudaKernels(sampler_build)
        if not self.nested_kernels.identity.get('explicit_fma_intrinsics'):
            raise ValueError('Source-bound explicit nestedF64interpolation build required')
        self.model._nested_kernels = self.nested_kernels
        self.model._nested_geometry_functions = {}
        for counts in self.nested_kernels.identity['geometry_counts_xyz']:
            name = 'geometry_transport_'+'_'.join(map(str, counts)); function = C.c_void_p()
            check(self.nested_kernels.driver.cuModuleGetFunction(C.byref(function), self.nested_kernels.module, name.encode()), name)
            self.model._nested_geometry_functions[tuple(counts)] = function
        self.model._nested_original_transport = self.model.transport
        self.model._nested_enabled = False
        self.model.transport = types.MethodType(transport, self.model)
        self.model.predict = types.MethodType(predict, self.model)
        self.asset_identity.update(nested_F64_interpolation_build=self.nested_kernels.identity,
                                   sampler_precision='F64 nested endpoint-aware explicitRNFMAlerps for N8; original N1unchanged',
                                   learned_coefficients_unchanged=True, implicit_fma_contraction=False, explicit_lerp_FMA=True,
                                   native_numerical_oracle_in_candidate=False, whole_model_accuracy_requires_qualification=True)

    def close(self):
        try: super().close()
        finally:
            if hasattr(self, 'nested_kernels'): self.nested_kernels.close()


class NestedF64PressureNativeEntry(CompleteNativeEntry):
    generator_class = NestedF64PressureGraph

    def __init__(self, *args):
        super().__init__(*args)
        self.asset_identity.update(learned_arithmetic_unchanged=False, learned_coefficients_unchanged=True,
                                   explicit_lerp_FMA=True, original_N1_arithmetic_unchanged=True)
