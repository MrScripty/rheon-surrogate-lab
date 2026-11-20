"""The interface every playground backend has, and what they share.

A backend holds one sealed box of water and advances it one whole step (1/60 s) at a time. It is used from one
thread only (the simulation thread); nothing here locks.
"""
import math
from dataclasses import dataclass

import torch

from .. import blobs as blob_forces
from .. import views

DT = 1.0 / 60.0
TRACERS = ('layer', 'stripes', 'ball', 'none')


def limit_cpu_threads(count=4):
    """Small CPU tensors (a 32^3 force field) run far slower on many threads than on a few; keep a few."""
    if torch.get_num_threads() > count:
        torch.set_num_threads(count)


class Unsupported(ValueError):
    """The backend cannot run what was asked (grid, cells per metre, number of forces); the message says why."""


@dataclass(frozen=True)
class SceneSpec:
    counts: tuple = (32, 32, 32)          # cells along x, y, z
    cells_per_metre: float = 16.0
    tracer: str = 'layer'
    dt: float = DT

    @classmethod
    def from_json(cls, value, default=None):
        default = default or cls()
        value = value or {}
        try:
            counts = tuple(int(n) for n in value.get('counts', default.counts))
            cells_per_metre = float(value.get('cells_per_metre', default.cells_per_metre))
        except (TypeError, ValueError):
            raise ValueError('Grid counts are three whole numbers and cells per metre a number.') from None
        tracer = value.get('tracer', default.tracer)
        if len(counts) != 3 or not all(2 <= n <= 512 for n in counts):
            raise ValueError('A grid has 2 to 512 cells along each of x, y and z.')
        if not (math.isfinite(cells_per_metre) and 0.5 <= cells_per_metre <= 1024):
            raise ValueError('Cells per metre is between 0.5 and 1024.')
        if tracer not in TRACERS:
            raise ValueError(f'The tracer pattern is one of {", ".join(TRACERS)}.')
        return cls(counts, cells_per_metre, tracer)

    @property
    def spacing(self):
        return 1.0 / self.cells_per_metre

    @property
    def extents(self):
        return tuple(n * self.spacing for n in self.counts)

    def to_json(self):
        return dict(counts=list(self.counts), cells_per_metre=self.cells_per_metre, tracer=self.tracer, dt=self.dt,
                    extents_m=list(self.extents))


def tracer_pattern(counts, pattern, dtype=torch.float32, device='cpu'):
    """The initial tracer [Z, Y, X]: dye 1 where the pattern is, 0 elsewhere."""
    nx, ny, nz = counts
    z, y, x = torch.meshgrid(*((torch.arange(n, dtype=torch.float64, device=device) + 0.5) / n for n in (nz, ny, nx)),
                             indexing='ij')
    if pattern == 'layer':          # the lower third along y
        dye = y < 1 / 3
    elif pattern == 'stripes':      # eight bands along x
        dye = (x * 8).floor().remainder(2) == 0
    elif pattern == 'ball':
        dye = (x - .5) ** 2 + (y - .5) ** 2 + (z - .5) ** 2 < .2 ** 2
    else:
        dye = torch.zeros_like(x, dtype=torch.bool)
    return dye.to(dtype)


def zero_state(counts, device='cpu', tracer='layer'):
    """A fluid at rest: staggered velocities float32, pressure float64, tracer float32, [Z, Y, X]-ordered."""
    nx, ny, nz = counts
    return dict(u=torch.zeros((nz, ny, nx + 1), device=device), v=torch.zeros((nz, ny + 1, nx), device=device),
                w=torch.zeros((nz + 1, ny, nx), device=device),
                pressure=torch.zeros((nz, ny, nx), dtype=torch.float64, device=device),
                tracer=tracer_pattern(counts, tracer, device=device))


class Backend:
    """Base of every adapter. Subclasses set the class attributes and implement `_open`, `_advance`, `fields`
    and `_set_tracer`."""
    id = 'base'
    label = 'Backend'
    kind = 'solver'              # 'solver' (exact arithmetic of a solver) or 'learned' or 'test'
    device = 'cpu'
    max_blobs = 8
    forcing = 'before the step'  # where each blob's velocity enters

    def __init__(self):
        self.spec = None
        self.forces = []
        self.steps = 0

    # ---- what a backend can do ------------------------------------------------------------------------------
    @classmethod
    def grids(cls):
        """Grid counts (x, y, z) this backend runs, or None when any grid within `limits()` works."""
        return None

    @classmethod
    def cells_per_metre_options(cls):
        return (8.0, 16.0)

    @classmethod
    def available(cls):
        """(True, '') or (False, the reason) on this machine."""
        return True, ''

    def capabilities(self):
        return dict(id=self.id, label=self.label, kind=self.kind, device=self.device,
                    grids=[list(g) for g in self.grids()] if self.grids() is not None else None,
                    cells_per_metre=list(self.cells_per_metre_options()), max_blobs=self.max_blobs,
                    forcing=self.forcing, fields=list(views.FIELDS), tracers=list(TRACERS), dt=DT)

    def info(self):
        return dict(id=self.id, label=self.label, kind=self.kind, exact=self.kind == 'solver', device=self.device,
                    scene=self.spec.to_json() if self.spec else None, steps=self.steps, time_s=self.steps * DT)

    # ---- the interface --------------------------------------------------------------------------------------
    def check(self, spec):
        """Raise Unsupported when this backend cannot run `spec`."""
        grids = self.grids()
        if grids is not None and tuple(spec.counts) not in {tuple(g) for g in grids}:
            listed = ', '.join('x'.join(map(str, g)) for g in grids)
            raise Unsupported(f'{self.label} runs only the grids it has: {listed}; not '
                              f'{"x".join(map(str, spec.counts))}.')
        if spec.cells_per_metre not in self.cells_per_metre_options():
            raise Unsupported(f'{self.label} runs at {" or ".join(f"{c:g}" for c in self.cells_per_metre_options())} '
                              f'cells per metre, not {spec.cells_per_metre:g}.')

    def reset(self, spec):
        self.check(spec)
        self._open(spec)
        self.spec, self.steps = spec, 0

    def reset_tracer(self, pattern):
        if pattern not in TRACERS:
            raise ValueError(f'The tracer pattern is one of {", ".join(TRACERS)}.')
        self._set_tracer(tracer_pattern(self.spec.counts, pattern, device=self.state_device()))

    def set_forces(self, forces):
        forces = [f if isinstance(f, blob_forces.Blob) else blob_forces.Blob.from_json(f) for f in forces]
        if len(forces) > self.max_blobs:
            raise Unsupported(f'{self.label} takes at most {self.max_blobs} forces at once.')
        self.forces = forces

    def step(self):
        """Advance one whole step. The caller times it; `synchronize` is part of the step."""
        self._advance()
        self.steps += 1

    def slice(self, axis, index, field):
        return views.plane(self.fields(), self.spec.counts, self.spec.spacing, axis, index, field)

    def in_plane(self, axis, index):
        return views.in_plane_velocity(self.fields(), self.spec.counts, axis, index)

    def close(self):
        pass

    # ---- for subclasses -------------------------------------------------------------------------------------
    def time_s(self):
        return self.steps * DT

    def force_increment(self, forces=None, device=None):
        """The velocity (x, y, z faces, float64) the given forces (default all) add over this step."""
        return blob_forces.increment(self.forces if forces is None else forces, self.spec.counts,
                                     self.spec.cells_per_metre, self.time_s(), DT,
                                     device=device or self.state_device())

    def state_device(self):
        return 'cpu'

    def fields(self):
        """u, v, w, pressure, tracer as [Z, Y, X] tensors (any device)."""
        raise NotImplementedError

    def _set_tracer(self, tracer):
        raise NotImplementedError

    def _open(self, spec):
        raise NotImplementedError

    def _advance(self):
        raise NotImplementedError
