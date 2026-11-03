"""Guarded scalar-division execution with unchanged restored learned tensors."""
import ctypes as C
import math
import types
import torch
from debug.cube64_layout_transform_candidate_v1 import Cube64LayoutTransformCandidate
from fused_cuda_generator_v1 import pointer
from recurrent_sampler_native import dimensions


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
    arguments = [*(C.c_int(n) for n in counts), *(pointer(v) for v in inputs),
                 *(pointer(v) for v in outputs), *(pointer(v) for v in self.kernel_sparse['point']),
                 pointer(self.mid_gate), pointer(self.mix), C.c_double(dt), C.c_double(h), C.c_double(1. / h)]
    self.kernels.launch('learned_transport', sum(math.prod(s) for s in shapes), arguments)
    return outputs


class SharedReciprocalGraph(Cube64LayoutTransformCandidate):
    def __init__(self, pair, shared_pair, build, control_build):
        super().__init__(pair, shared_pair, build, control_build)
        self.model.transport = types.MethodType(transport, self.model)
        self.asset_identity.update(
            sampler_scalar_division='Known spacing RNreciprocal reused with two residual/correction pairs in conservative normal range; CUDAstandard division otherwise',
            scalar_primitive_qualification='runs/full-ai-data-diversity-v1/shared-reciprocal-primitive-01/results.json',
            learned_coefficients_unchanged=True, native_numerical_oracle_in_candidate=False,
            all_internal_advances_and_outputs_preserved=True)
