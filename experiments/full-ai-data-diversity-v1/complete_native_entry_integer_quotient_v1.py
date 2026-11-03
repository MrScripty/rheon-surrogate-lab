"""Exact integer-midpoint quotient transport over supplied integer grid dimensions."""
import ctypes as C
import math
import types
import torch
from debug.gather_fused_graph_v1 import GatherFusedGraph
from debug.grid_descriptor_graph_v1 import SourceTable, Shape
from cuda_kernel_runtime_v1 import check
from fused_cuda_generator_v1 import pointer
from recurrent_sampler_native import dimensions


def transport(self, fields, counts, alpha):
    counts = tuple(counts)
    if self.transport_geometry is None or counts not in self.geometry_functions:
        raise ValueError('Prepared registered integer geometry required')
    h, dt = self.transport_geometry
    if alpha != dt / h:
        raise ValueError('Physical step and proposal scale disagree')
    shapes = dimensions(counts)
    if len(fields) != 4 or any(v.device != self.device or v.dtype != torch.float32
                              or tuple(v.shape) != s for v, s in zip(fields, shapes)):
        raise ValueError('Complete native CUDA carriers required')
    inputs = [v.contiguous() for v in fields]
    outputs = [torch.empty(s, dtype=torch.float32, device=self.device) for s in shapes]
    lengths = [math.prod(s) for s in shapes]
    descriptor = SourceTable((Shape * 4)(*(Shape(*reversed(s)) for s in shapes)),
                             (C.c_void_p * 4)(*(v.data_ptr() for v in inputs)),
                             (C.c_void_p * 4)(*(v.data_ptr() for v in outputs)),
                             (C.c_longlong * 4)(*lengths), sum(lengths))
    arguments = [descriptor, *(pointer(v) for v in self.kernel_sparse['point']),
                 pointer(self.mid_gate), pointer(self.mix), C.c_double(dt), C.c_double(h), C.c_double(1. / h)]
    pointers = (C.c_void_p * len(arguments))(*(C.addressof(v) for v in arguments))
    stream = C.c_void_p(torch.cuda.current_stream().cuda_stream)
    check(self.kernels.driver.cuLaunchKernel(self.geometry_functions[counts],
          (max(lengths) + 127) // 128, 4, 1, 128, 1, 1, 0, stream, pointers, None),
          'cuLaunchKernel generic geometry transport')
    return outputs


from debug.geometry_specialized_graph_v1 import GeometrySpecializedGraph
from complete_native_entry_v1 import CompleteNativeEntry


class IntegerQuotientGraph(GeometrySpecializedGraph):
    def __init__(self, pair, shared_pair, build, control_build):
        super().__init__(pair, shared_pair, build, control_build)
        if self.model.kernels.identity.get('quotient_certification') != 'integer_midpoint_RN64_with_original_division_fallback':
            raise ValueError('Registered integer-midpoint division build required')
        self.model.transport = types.MethodType(transport, self.model)
        self.asset_identity.update(
            sampler_arithmetic_unchanged=False,
            sampler_internal_precision='Original F64 interpolation and samples; exact integer-midpoint RN64 quotient certificates or original division',
            quotient_certificate=self.model.kernels.identity['enclosure_derivation'],
            known_reciprocal_setup='CPU arithmetic on supplied positive physical spacing only; no learned values or future fields',
            native_field_dtypes_unchanged=True, learned_coefficients_unchanged=True,
            pressure_transform_arithmetic_unchanged=True)


class IntegerQuotientEntry(CompleteNativeEntry):
    generator_class = IntegerQuotientGraph
