"""N8 exact scalar reuse; no CPU learned forward or precision change."""
import ctypes as C
import math
import types
import torch
from cell_coordinate_pressure_graph_v1 import CellCoordinatePressureGraph
from complete_native_entry_v1 import CompleteNativeEntry
from cuda_kernel_runtime_v1 import check
from debug.grid_descriptor_graph_v1 import SourceTable, Shape
from fused_cuda_generator_v1 import pointer
from recurrent_sampler_native import dimensions

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
        arguments = [descriptor, *(pointer(v) for v in core.kernel_sparse['point']), pointer(core.mid_gate), pointer(core.mix), C.c_double(alpha), C.c_double(h)]
        pointers = (C.c_void_p*len(arguments))(*(C.addressof(v) for v in arguments))
        check(kernels.driver.cuLaunchKernel(core._nested_geometry_functions[counts], (max(lengths)+127)//128, 4, 1, 128, 1, 1, 0,
                                            C.c_void_p(torch.cuda.current_stream().cuda_stream), pointers, None), 'nestedF64geometrytransport')
    else:
        arguments = [*(C.c_int(n) for n in counts), *(pointer(v) for v in inputs), *(pointer(v) for v in outputs),
                     *(pointer(v) for v in core.kernel_sparse['point']), pointer(core.mid_gate), pointer(core.mix), C.c_double(alpha), C.c_double(h)]
        kernels.launch('learned_transport', sum(lengths), arguments)
    return outputs


class AlphaArgumentPressureGraph(CellCoordinatePressureGraph):
    def __init__(self,pair,shared_pair,build,control_build):
        super().__init__(pair,shared_pair,build,control_build,'runs/full-ai-data-diversity-v1/cuda-fusion-build-24')
        if self.nested_kernels.identity.get('sampler_alpha_argument')!='validated_existing_RN64_dt_over_h':
            raise ValueError('Exact alpha-argument ABI build required')
        self.model.transport=types.MethodType(transport,self.model)
        self.asset_identity.update(sampler_alpha_argument='existing validated binary64 alpha',
            learned_coefficients_unchanged=True, original_N1_arithmetic_unchanged=True,
            interpolation_and_all_remaining_F64_arithmetic_unchanged=True,
            ratio_recomputation_removed=True, new_qualification_required=True)


class AlphaArgumentPressureNativeEntry(CompleteNativeEntry):
    generator_class=AlphaArgumentPressureGraph
    def __init__(self,*args):
        super().__init__(*args)
        self.asset_identity.update(learned_coefficients_unchanged=True,original_N1_arithmetic_unchanged=True,
            scalar_encoding_reuse_only=True,new_qualification_required=True)
