"""A stand-in backend for tests and machines without CUDA. Not fluid physics.

The forces add their velocity to the faces each step, the velocity decays a little, and the tracer is carried by the
cell-centred velocity (semi-Lagrangian, trilinear). There is no pressure and nothing keeps the flow divergence
free; it only exercises the page and the server with a field that answers to the forces.
"""
import torch
from torch.nn import functional as F

from .base import DT, Backend, limit_cpu_threads, zero_state

GRIDS = ((9, 11, 11), (15, 6, 8), (16, 16, 16), (16, 24, 40), (32, 24, 48), (32, 32, 32), (64, 64, 64))
DECAY = 0.98


class FakeBackend(Backend):
    id = 'fake'
    label = 'Test backend (not physics) · CPU'
    kind = 'test'
    device = 'cpu'
    forcing = 'every force before the step'

    @classmethod
    def grids(cls):
        return GRIDS

    def _open(self, spec):
        limit_cpu_threads()
        self.state = zero_state(spec.counts, tracer=spec.tracer)
        nx, ny, nz = spec.counts
        z, y, x = torch.meshgrid(*((torch.arange(n, dtype=torch.float32) * 2 + 1) / n - 1 for n in (nz, ny, nx)),
                                 indexing='ij')
        self._cells = torch.stack((x, y, z), dim=-1)[None]          # grid_sample order: x, y, z

    def fields(self):
        return self.state

    def _set_tracer(self, tracer):
        self.state['tracer'] = tracer

    def _advance(self):
        state, spec = self.state, self.spec
        for name, part in zip('uvw', self.force_increment()):
            state[name] = (state[name] + part.float()) * DECAY
        u, v, w = state['u'], state['v'], state['w']
        centred = torch.stack((0.5 * (u[..., 1:] + u[..., :-1]), 0.5 * (v[:, 1:] + v[:, :-1]),
                               0.5 * (w[1:] + w[:-1])), dim=-1)
        scale = torch.tensor([2 * DT / (spec.spacing * n) for n in spec.counts])
        grid = self._cells - centred[None] * scale
        tracer = F.grid_sample(state['tracer'][None, None], grid, mode='bilinear', padding_mode='border',
                               align_corners=False)
        state['tracer'] = tracer[0, 0].contiguous()
