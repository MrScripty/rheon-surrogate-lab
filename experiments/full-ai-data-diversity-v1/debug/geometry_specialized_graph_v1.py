"""Exact generic transport specialized for supplied integer grid dimensions."""
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
                 pointer(self.mid_gate), pointer(self.mix), C.c_double(dt), C.c_double(h)]
    pointers = (C.c_void_p * len(arguments))(*(C.addressof(v) for v in arguments))
    stream = C.c_void_p(torch.cuda.current_stream().cuda_stream)
    check(self.kernels.driver.cuLaunchKernel(self.geometry_functions[counts],
          (max(lengths) + 127) // 128, 4, 1, 128, 1, 1, 0, stream, pointers, None),
          'cuLaunchKernel generic geometry transport')
    return outputs


class GeometrySpecializedGraph(GatherFusedGraph):
    def __init__(self, pair, shared_pair, build, control_build):
        super().__init__(pair, shared_pair, build, control_build)
        kernels = self.model.kernels
        geometry = kernels.identity.get('geometry_counts_xyz')
        if not geometry or not kernels.identity.get('learned_coefficients_dynamic'):
            raise ValueError('Precise geometry-only build required')
        self.model.geometry_functions = {}
        for counts in geometry:
            name = 'geometry_transport_' + '_'.join(map(str, counts))
            function = C.c_void_p()
            check(kernels.driver.cuModuleGetFunction(C.byref(function), kernels.module, name.encode()),
                  'cuModuleGetFunction ' + name)
            self.model.geometry_functions[tuple(counts)] = function
        self.model.transport = types.MethodType(transport, self.model)
        self.asset_identity.update(
            sampler_specialization='Supplied integer grid dimensions and carrier kinds; one four-row CUDA grid',
            geometry_counts_xyz=geometry, learned_coefficients_unchanged=True,
            physical_spacing_and_timestep_dynamic=True, sampler_arithmetic_unchanged=True,
            native_numerical_oracle_in_candidate=False)
