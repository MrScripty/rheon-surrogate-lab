"""Exact external scalar partial evaluation around unchanged learned transport."""
import ctypes as C
import math
import types
import torch
from debug.geometry_specialized_graph_v1 import GeometrySpecializedGraph
from debug.grid_descriptor_graph_v1 import SourceTable, Shape
from cuda_kernel_runtime_v1 import check
from fused_cuda_generator_v1 import pointer
from recurrent_sampler_native import dimensions


def transport(self, fields, counts, alpha):
    counts = tuple(counts)
    if self.transport_geometry is None:
        raise ValueError('Prepared registered integer geometry required')
    h, dt = self.transport_geometry
    key = counts, h.hex(), dt.hex()
    if key not in self.scalar_geometry_functions:
        raise ValueError('Registered exact external scalar configuration required')
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
                 pointer(self.mid_gate), pointer(self.mix)]
    pointers = (C.c_void_p * len(arguments))(*(C.addressof(v) for v in arguments))
    stream = C.c_void_p(torch.cuda.current_stream().cuda_stream)
    check(self.kernels.driver.cuLaunchKernel(self.scalar_geometry_functions[key],
          (max(lengths) + 127) // 128, 4, 1, 128, 1, 1, 0, stream, pointers, None),
          'cuLaunchKernel generic geometry transport')
    return outputs


class ScalarGeometryGraph(GeometrySpecializedGraph):
    def __init__(self, pair, shared_pair, build, control_build):
        super().__init__(pair, shared_pair, build, control_build)
        kernels = self.model.kernels
        registry = kernels.identity.get('geometry_scalar_registry')
        if not registry or kernels.identity.get('physical_spacing_and_timestep_dynamic'):
            raise ValueError('Exact external-scalar build required')
        self.model.scalar_geometry_functions = {}
        for row in registry:
            counts, cpm, n = tuple(row['counts_xyz']), row['cells_per_metre'], row['internal_substeps']
            h, dt = 1 / cpm, 1 / (60 * n)
            if h.hex() != row['spacing_hex'] or dt.hex() != row['timestep_hex']:
                raise ValueError('Exact supplied scalar bits differ from compiled literals')
            key = counts, h.hex(), dt.hex()
            if key in self.model.scalar_geometry_functions:
                raise ValueError('Unique compiled external metadata required')
            function = C.c_void_p()
            check(kernels.driver.cuModuleGetFunction(C.byref(function), kernels.module, row['function'].encode()),
                  'cuModuleGetFunction ' + row['function'])
            self.model.scalar_geometry_functions[key] = function
        self.model.transport = types.MethodType(transport, self.model)
        self.asset_identity.update(
            sampler_specialization='Integer geometry/carrier kinds and exact binary external spacing/timestep',
            geometry_scalar_registry=registry, learned_coefficients_unchanged=True,
            physical_spacing_and_timestep_dynamic=False, scalar_configuration_validated_each_call=True,
            sampler_arithmetic_unchanged=True, native_numerical_oracle_in_candidate=False)
