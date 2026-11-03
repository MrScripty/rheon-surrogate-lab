"""Read-only kernel parameter tables for unchanged learned transport arithmetic."""
import ctypes as C
import math
import types
import torch
from debug.gather_fused_graph_v1 import GatherFusedGraph
from fused_cuda_generator_v1 import pointer
from recurrent_sampler_native import dimensions


class Shape(C.Structure):
    _fields_ = [('x', C.c_int), ('y', C.c_int), ('z', C.c_int)]


class SourceTable(C.Structure):
    _fields_ = [('shapes', Shape * 4), ('input', C.c_void_p * 4),
                ('output', C.c_void_p * 4), ('length', C.c_longlong * 4),
                ('total', C.c_longlong)]


if (C.sizeof(Shape), C.sizeof(SourceTable), C.alignment(SourceTable)) != (12, 152, 8):
    raise RuntimeError('CUDA descriptor ABI mismatch')
if [getattr(SourceTable, n).offset for n in ('shapes', 'input', 'output', 'length', 'total')] != [0, 48, 80, 112, 144]:
    raise RuntimeError('CUDA descriptor member layout mismatch')


def transport(self, fields, counts, alpha):
    if self.transport_geometry is None:
        raise ValueError('Explicit physical transport geometry required')
    h, dt = self.transport_geometry
    if alpha != dt / h:
        raise ValueError('Physical step and normalized proposal scale disagree')
    shapes = dimensions(counts)
    if len(fields) != 4 or any(v.device != self.device or v.dtype != torch.float32
                              or tuple(v.shape) != s for v, s in zip(fields, shapes)):
        raise ValueError('Complete native CUDA carriers required')
    inputs = [v.contiguous() for v in fields]
    outputs = [torch.empty(s, dtype=torch.float32, device=self.device) for s in shapes]
    sizes = [math.prod(s) for s in shapes]
    table = SourceTable((Shape * 4)(*(Shape(*reversed(s)) for s in shapes)),
                        (C.c_void_p * 4)(*(v.data_ptr() for v in inputs)),
                        (C.c_void_p * 4)(*(v.data_ptr() for v in outputs)),
                        (C.c_longlong * 4)(*sizes), sum(sizes))
    # cuLaunchKernel copies each by-value argument before returning. Captured
    # graphs own argument values; tensor storage lifetime follows the existing
    # graph input/output pool. The stack descriptor is never a device pointer.
    arguments = [*(C.c_int(n) for n in counts), table,
                 *(pointer(v) for v in self.kernel_sparse['point']),
                 pointer(self.mid_gate), pointer(self.mix), C.c_double(dt), C.c_double(h)]
    self.kernels.launch('learned_transport', sum(sizes), arguments)
    return outputs


class GridDescriptorGraph(GatherFusedGraph):
    def __init__(self, pair, shared_pair, build, control_build):
        super().__init__(pair, shared_pair, build, control_build)
        self.model.transport = types.MethodType(transport, self.model)
        self.asset_identity.update(
            sampler_metadata='152-byte by-value read-only CUDA grid-constant descriptor',
            learned_coefficients_unchanged=True, sampler_arithmetic_unchanged=True,
            native_numerical_oracle_in_candidate=False)
