"""The lab's earlier GPU solver composition ("F64 cell composition") as a benchmark subject.

It is Rheon's step written as tensor operations: midpoint semi-Lagrangian
transport, forcing, viscosity, and a pressure solve by cosine transform. Its
coefficients were fitted to data and converged to the solver's own stencil values,
so it reproduces the reference to rounding. The benchmark lists it as a solver:
it is the speed and accuracy a learned model has to justify itself against on
these sealed-box scenes.

The implementation lives in experiments/full-ai-data-diversity-v1 and needs CUDA
plus kernel builds and fitted coefficients under runs/, which are not in Git.
"""
import os
import sys

import torch

from ..data import REPOSITORY
from ..fields import FluidState
from .runner import Subject, Unsupported

_SOURCE = 'experiments/full-ai-data-diversity-v1'
_ASSETS = ('runs/full-ai-tight-targets-v1/sampler-head-pair-01',        # fitted transport coefficients
           'runs/full-ai-data-diversity-v1/shared-spectral-pair-01',    # fitted pressure response
           'runs/full-ai-data-diversity-v1/cuda-fusion-build-12',       # compiled kernels
           'runs/full-ai-data-diversity-v1/cuda-fusion-build-04')       # compiled forcing controls
MAX_OUTPUTS_PER_PASS = 4


class SpectralComposition(Subject):
    attributes = {'kind': 'solver', 'device': 'cuda'}

    def __init__(self, outputs_per_pass=1):
        if not 1 <= outputs_per_pass <= MAX_OUTPUTS_PER_PASS:
            raise ValueError(f'The composition produces 1 to {MAX_OUTPUTS_PER_PASS} outputs per pass.')
        if not torch.cuda.is_available():
            raise RuntimeError('The solver composition needs CUDA.')
        missing = [asset for asset in _ASSETS if not (REPOSITORY / asset).is_dir()]
        if missing:
            raise FileNotFoundError(f'The solver composition needs local builds that are absent: {missing}')
        self.outputs_per_pass = outputs_per_pass
        self.id = f'spectral-composition-k{outputs_per_pass}'
        self.label = f'GPU solver composition · spectral pressure · {outputs_per_pass} per pass'
        # The implementation resolves its own modules and assets relative to its folder.
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        source = str(REPOSITORY / _SOURCE)
        if source not in sys.path:
            sys.path.insert(0, source)
        import support  # noqa: F401  (puts the folders its modules import from on sys.path)
        from cell_coordinate_pressure_graph_v1 import CellCoordinatePressureNativeEntry
        self._entry = CellCoordinatePressureNativeEntry(*_ASSETS)

    def begin(self, case):
        scene = case.parameters
        self._scene = scene
        self._substeps = scene.get('reference_substeps', 1)
        if case.start + case.intervals > scene['steps']:
            raise Unsupported('The case runs past the steps this scene defines forcing for.')
        start = case.reference.map(lambda tensor: tensor[:1])
        self._entry.prepare_native(self._flat(start), scene, case.start, self.outputs_per_pass, self._substeps)

    @torch.inference_mode()
    def advance(self, state, forcing, scene, step):
        outputs = self._entry.predict_native(self._flat(state), self._scene, step, self.outputs_per_pass,
                                             self._substeps)
        nx, ny, nz = self._scene['counts']
        shapes = {'pressure': (nz, ny, nx), 'vx': (nz, ny, nx + 1), 'vy': (nz, ny + 1, nx),
                  'vz': (nz + 1, ny, nx), 'tracer': (nz, ny, nx)}
        states = []
        for output in outputs:
            fields = {name: output[name].reshape(1, *shape).clone() for name, shape in shapes.items()}
            states.append(FluidState(u=fields['vx'], v=fields['vy'], w=fields['vz'], tracer=fields['tracer'],
                                     pressure=fields['pressure']))
        return states

    def close(self):
        self._entry.close()

    @staticmethod
    def _flat(state):
        """The implementation's input layout: flat host arrays, pressure float64 (first cell zero), the rest float32."""
        pressure = state.pressure[0].double()
        fields = {'pressure': pressure - pressure.flatten()[0], 'vx': state.u[0].float(), 'vy': state.v[0].float(),
                  'vz': state.w[0].float(), 'tracer': state.tracer[0].float()}
        return {name: tensor.contiguous().reshape(-1) for name, tensor in fields.items()}
