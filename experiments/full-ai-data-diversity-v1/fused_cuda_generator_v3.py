"""Generic learned sampler with explicit physical-coordinate interpolation."""
import ctypes as C
import math
import torch
from fused_cuda_generator_v1 import pointer
from fused_cuda_generator_v2 import FusedCudaGeneratorV2
from recurrent_sampler_native import dimensions


class FusedCudaGeneratorV3(FusedCudaGeneratorV2):
    def __init__(self, pair, shared_pair, build):
        super().__init__(pair, shared_pair, build)
        self.transport_geometry = None
        self.asset_identity.update(
            fixed_interpolation_coordinates='Physical position and spacing, with domain clamping before source-lattice phase subtraction',
            learned_sampler_parameters_changed=False,
            native_solver_oracles_in_candidate=False)

    def configure_transport(self, h, dt):
        if not math.isfinite(h) or not math.isfinite(dt) or h <= 0 or dt <= 0:
            raise ValueError('Finite positive physical spacing and timestep required')
        self.transport_geometry = h, dt

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
                     pointer(self.mid_gate), pointer(self.mix), C.c_double(dt), C.c_double(h)]
        self.kernels.launch('learned_transport', sum(math.prod(s) for s in shapes), arguments)
        return outputs

    @torch.inference_mode()
    def predict(self, fields, scene, step, horizons, external, substeps=1):
        self.configure_transport(1 / scene['cells_per_metre'], 1 / (60 * substeps))
        return super().predict(fields, scene, step, horizons, external, substeps)
