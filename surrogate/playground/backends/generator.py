"""The learned full-state generator b1-fixed-viscosity as a playground backend.

A network that takes the whole state and the prescribed force and returns the next state (one 1/60 s output per
call), trained in the lab's earlier study (experiments/full-ai-next-priorities-v1) on sealed boxes forced by one
sinusoidally moving blob (0.08-0.16 m/s^2, radius 15-24 % of the box). Its force input is the velocity each face gains this step divided by the velocity scale;
the playground writes its own blobs there. Anything other than one blob of the training's size and strength is
outside what it was trained on.
"""
import math
import sys

import torch
from torch.nn import functional as F

from ...data import REPOSITORY
from .base import Backend, Unsupported, limit_cpu_threads, zero_state

SOURCE = 'experiments/full-ai-next-priorities-v1'
PAIR = 'runs/full-ai-next-priorities-v1/b1-fixed-viscosity-pair-01'
GRIDS = ((9, 11, 11), (15, 6, 8), (16, 16, 16), (32, 32, 32))


class GeneratorBackend(Backend):
    id = 'generator'
    label = 'Learned generator b1-fixed-viscosity (learned model)'
    kind = 'learned'
    device = 'cuda'
    forcing = 'every force inside the step, after transport (as in its training)'

    def __init__(self):
        super().__init__()
        self._network = None

    @classmethod
    def grids(cls):
        return GRIDS

    @classmethod
    def available(cls):
        if not torch.cuda.is_available():
            return False, 'needs CUDA (start the playground with the Pumas PyTorch, see the README)'
        if not (REPOSITORY / PAIR / 'restore/weights.npz').is_file():
            return False, f'its weights are not on this machine ({PAIR})'
        return True, ''

    def _load(self):
        ok, reason = self.available()
        if not ok:
            raise Unsupported(f'The learned generator {reason}.')
        limit_cpu_threads()     # its construction runs small CPU convolutions; on many threads it takes minutes
        source = str(REPOSITORY / SOURCE)
        if source not in sys.path:
            sys.path.insert(0, source)
        from fixed_viscosity_evaluation_support import load_model
        import temporal_features_v2
        network, normalization, identity = load_model(REPOSITORY / PAIR)
        if network.horizons != 1:
            raise Unsupported('The playground needs a generator with one output per call.')
        self._network = network.to('cuda').eval()
        self._scales = normalization['scales']
        self._features = temporal_features_v2
        self.identity = identity

    def _open(self, spec):
        if self._network is None:
            self._load()
        state = zero_state(spec.counts, tracer=spec.tracer)
        self.state = dict(pressure=state['pressure'], vx=state['u'], vy=state['v'], vz=state['w'],
                          tracer=state['tracer'])
        self.state = {name: value.to('cuda').contiguous().reshape(-1) for name, value in self.state.items()}
        coordinates, _ = self._features.geometry(tuple(spec.counts), 'cuda')
        spatial = tuple(coordinates.shape[1:])
        constants = torch.tensor([math.log(n / 16) for n in spec.counts] + [math.log(16 / spec.cells_per_metre)],
                                 device='cuda', dtype=torch.float32).reshape(1, 4, 1, 1, 1).expand(1, 4, *spatial)
        self._fixed = torch.cat((coordinates.unsqueeze(0), constants), dim=1)

    def state_device(self):
        return 'cuda'

    def fields(self):
        nx, ny, nz = self.spec.counts
        s = self.state
        return dict(u=s['vx'].view(nz, ny, nx + 1), v=s['vy'].view(nz, ny + 1, nx), w=s['vz'].view(nz + 1, ny, nx),
                    pressure=s['pressure'].view(nz, ny, nx), tracer=s['tracer'].view(nz, ny, nx))

    def _set_tracer(self, tracer):
        self.state['tracer'] = tracer.to('cuda', torch.float32).contiguous().reshape(-1)

    def force_channels(self):
        """The network's force input: each face's velocity gain this step over the velocity scale, padded to the
        (Z+1, Y+1, X+1) node grid by repeating the last cell along the axes the face is not normal to."""
        channels = []
        for d, part in enumerate(self.force_increment(device='cuda')):
            pad = [0, 1, 0, 1, 0, 1]                    # x, y, z ends
            pad[2 * d + 1] = 0
            channels.append(F.pad(part[None, None], pad, mode='replicate')[0, 0])
        return (torch.stack(channels)[None] / self._scales['velocity']).float()

    @torch.inference_mode()
    def _advance(self):
        counts = tuple(self.spec.counts)
        packed = self._features.packed_state(self.state, counts, self._scales, 'cuda')
        features = torch.cat((self._fixed, self.force_channels()), dim=1)
        output = self._network(packed, features)
        self.state = self._features.unpack_outputs(output, counts, self._scales)[0]
        torch.cuda.synchronize()

    def close(self):
        self._network = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
