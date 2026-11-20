"""The GPU solver composition (the "F64 cell composition") as a playground backend.

Rheon's step as fixed GPU operations, replayed as one CUDA graph per grid: semi-Lagrangian transport, the forcing
kernel, viscosity, a pressure solve by cosine transform. Its coefficients converged to the solver's own stencil, so it
is listed as an exact solver. The implementation and its builds are the lab's earlier ones
(surrogate/bench/composition.py says where they live); this adapter only feeds it.

Forcing: the graph's forcing kernel takes one blob per step (the control packet row, applied after transport like
the solver's own force). Every further blob is added to the state's velocity before the step. The state stays on the
GPU between steps; a step is one graph replay plus the small copies around it.
"""
import os
import sys

import torch

from .. import blobs as blob_forces
from ...data import REPOSITORY
from .base import DT, Backend, Unsupported, zero_state

SOURCE = 'experiments/full-ai-data-diversity-v1'
ASSETS = ('runs/full-ai-tight-targets-v1/sampler-head-pair-01',
          'runs/full-ai-data-diversity-v1/shared-spectral-pair-01',
          'runs/full-ai-data-diversity-v1/cuda-fusion-build-12',
          'runs/full-ai-data-diversity-v1/cuda-fusion-build-04')
EXTRA_ASSETS = ('runs/full-ai-data-diversity-v1/cuda-fusion-build-21',)
GRIDS = ((9, 11, 11), (15, 6, 8), (16, 16, 16), (16, 24, 40), (32, 24, 48), (32, 32, 32), (64, 64, 64),
         (104, 112, 120), (128, 128, 128))
_FIELDS = ('pressure', 'vx', 'vy', 'vz', 'tracer')


class CompositionBackend(Backend):
    id = 'composition'
    label = 'GPU solver composition (exact solver)'
    kind = 'solver'
    device = 'cuda'
    forcing = 'first force inside the step, after transport (the solver\'s place); others added before the step'

    def __init__(self):
        super().__init__()
        self._entry = None
        self._key = None

    @classmethod
    def grids(cls):
        return GRIDS

    @classmethod
    def available(cls):
        if not torch.cuda.is_available():
            return False, 'needs CUDA (start the playground with the Pumas PyTorch, see the README)'
        missing = [a for a in ASSETS + EXTRA_ASSETS if not (REPOSITORY / a).is_dir()]
        if missing:
            return False, f'its kernel builds and coefficients are not on this machine: {", ".join(missing)}'
        return True, ''

    def _load(self):
        ok, reason = self.available()
        if not ok:
            raise Unsupported(f'The solver composition {reason}.')
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        source = str(REPOSITORY / SOURCE)
        if source not in sys.path:
            sys.path.insert(0, source)
        import support  # noqa: F401  (puts the folders its modules import from on sys.path)
        from cell_coordinate_pressure_graph_v1 import CellCoordinatePressureNativeEntry
        entry = CellCoordinatePressureNativeEntry(*ASSETS)
        # The graph reads an 18-column known-control row per substep; it is written here, one blob per step.
        entry._generator.controls.parameters = self._parameters
        self._entry = entry

    def _parameters(self, scene, step, horizons, substeps):
        if horizons != 1 or substeps != 1:
            raise ValueError('The playground runs one output step of one substep per call.')
        return torch.tensor([self._row], dtype=torch.float64)

    def _scene(self, spec):
        return dict(counts=list(spec.counts), cells_per_metre=spec.cells_per_metre, sgs=False, steps=1 << 40)

    def _open(self, spec):
        if self._entry is None:
            self._load()
        key = (tuple(spec.counts), spec.cells_per_metre)
        if self._key is not None and self._key != key:
            self._entry.release_graphs()
            torch.cuda.empty_cache()
        self._row = self._packet_row(None, spec, 0.0)
        start = {name: value.contiguous().reshape(-1) for name, value in
                 _native(zero_state(spec.counts, tracer=spec.tracer)).items()}
        self._entry.prepare_native(start, self._scene(spec), 0, 1, 1)
        self._key = key
        self.state = {name: value.to('cuda') for name, value in start.items()}

    @staticmethod
    def _packet_row(blob, spec, time_s):
        blob = blob or blob_forces.Blob((0.5, 0.5, 0.5), spec.spacing, (0.0, 0.0, 0.0), 0.0)
        return blob_forces.packet_row(blob, spec.counts, spec.cells_per_metre, time_s, DT)

    def state_device(self):
        return 'cuda'

    def fields(self):
        nx, ny, nz = self.spec.counts
        s = self.state
        return dict(u=s['vx'].view(nz, ny, nx + 1), v=s['vy'].view(nz, ny + 1, nx), w=s['vz'].view(nz + 1, ny, nx),
                    pressure=s['pressure'].view(nz, ny, nx), tracer=s['tracer'].view(nz, ny, nx))

    def _set_tracer(self, tracer):
        self.state['tracer'] = tracer.to('cuda', torch.float32).contiguous().reshape(-1)

    @torch.inference_mode()
    def _advance(self):
        spec, time_s = self.spec, self.time_s()
        first, rest = (self.forces[0], self.forces[1:]) if self.forces else (None, [])
        self._row = self._packet_row(first, spec, time_s)
        fields = dict(self.state)
        if rest:
            for name, part in zip(('vx', 'vy', 'vz'), self.force_increment(rest, device='cuda')):
                fields[name] = (fields[name] + part.reshape(-1).float()).contiguous()
        known = torch.tensor([self._row], dtype=torch.float64).pin_memory()
        scene = self._scene(spec)
        with self._entry._exclusive():
            outputs = self._entry._replay_borrowed(fields, scene, self.steps, 1, 1, known)
            self.state = {name: outputs[0][name].clone() for name in _FIELDS}
            torch.cuda.current_stream().synchronize()

    def close(self):
        if self._entry is not None:
            self._entry.close()
            self._entry = None
            torch.cuda.empty_cache()


def _native(state):
    """The composition's field names and dtypes: pressure float64 with the first cell zero, the rest float32."""
    pressure = state['pressure'].double()
    return dict(pressure=pressure - pressure.flatten()[0], vx=state['u'].float(), vy=state['v'].float(),
                vz=state['w'].float(), tracer=state['tracer'].float())

