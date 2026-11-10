"""Run the Rheon build behind the lab's reference data, one output step at a time.

The "probe" is a small Rust program around a Rheon build. It keeps one scene's
state and speaks newline-delimited JSON with raw little-endian arrays on its
standard streams. This module is the only place that protocol is written down.

A step can use either pressure solve:

    advance_reference   the probe asks for the pressure and gets an exact cosine-transform
                        solve from this module. Every recorded reference state was made this way.
    advance_native      Rheon's own conjugate-gradient solve, used as a benchmark baseline.

The probe binaries are local builds under runs/ and are not in Git.
"""
import json
import math
import os
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .fields import FluidState

REPOSITORY = Path(__file__).resolve().parents[1]
# One build per forcing family: the scheduled build reads a centre and pulse per step from the scene.
PROBES = {'sinusoidal': 'runs/full-ai-next-priorities-v1/temporal-build-02/rheon-temporal-probe',
          'scheduled': 'runs/full-ai-data-diversity-v1/native-forcing-build-02/rheon-temporal-probe'}
# Keys this package adds to a scene for its own use; the probe does not read them.
_LOCAL_KEYS = ('reference_substeps', 'forcing_control_substeps')
_MAX_LINE = 65536


def probe_for(scene, repository=REPOSITORY):
    """The probe binary that runs this scene; raises if it has not been built here."""
    if scene.get('reference_substeps', 1) != 1:
        raise NotImplementedError('Scenes that need several solver substeps per output are not wired to a probe yet.')
    path = Path(repository) / PROBES['scheduled' if 'forcing_schedule' in scene else 'sinusoidal']
    if not path.is_file():
        raise FileNotFoundError(f'The Rheon probe is not built at {path}.')
    return path


@lru_cache(maxsize=16)
def _cosine_modes(grid):
    """Per-axis orthonormal cosine bases and the eigenvalues of the sealed-box Laplacian (grid units)."""
    axes, eigenvalues = [], []
    for n in grid:
        modes = torch.arange(n, dtype=torch.float64)
        cells = torch.arange(n, dtype=torch.float64)
        basis = torch.cos(math.pi / n * modes[:, None] * (cells[None] + .5))
        basis *= math.sqrt(2. / n)
        basis[0] /= math.sqrt(2.)
        axes.append(basis)
        eigenvalues.append(4 * torch.sin(math.pi * modes / (2 * n)).square())
    z, y, x = eigenvalues
    values = z[:, None, None] + y[None, :, None] + x[None, None, :]
    values[0, 0, 0] = 1.   # the constant mode is set to zero after the division
    return tuple(axes), values


def pressure_flux(pressure, spacing, density):
    """The sealed-box pressure operator applied to `pressure` [Z, Y, X]: net flux of its gradient."""
    result = torch.zeros_like(pressure)
    for axis in (0, 1, 2):
        difference = pressure.narrow(axis, 0, pressure.shape[axis] - 1) - pressure.narrow(axis, 1, pressure.shape[axis] - 1)
        result.narrow(axis, 0, pressure.shape[axis] - 1).add_(difference)
        result.narrow(axis, 1, pressure.shape[axis] - 1).sub_(difference)
    return result * (spacing / density)


def spectral_correction(residual, spacing, density):
    """Solve the sealed-box pressure equation exactly for `residual` [Z, Y, X], float64; first cell zero."""
    size = residual.numel()
    norm = torch.linalg.vector_norm(residual)
    if residual.mean().abs() * math.sqrt(size) > max(norm.item() * 1e-8, 1e-12):
        raise ValueError('The pressure equation has no solution: its right-hand side does not sum to zero.')
    (z, y, x), eigenvalues = _cosine_modes(tuple(residual.shape))
    modes = torch.einsum('mx,klx->klm', x, torch.einsum('ly,kyx->klx', y, torch.einsum('kz,zyx->kyx', z,
                                                                                    residual * (density / spacing))))
    modes = modes / eigenvalues
    modes[0, 0, 0] = 0.
    result = torch.einsum('kz,kyx->zyx', z, torch.einsum('ly,klx->kyx', y, torch.einsum('mx,klm->klx', x, modes)))
    return result - result[0, 0, 0]


class RheonProbe:
    """One running probe holding one scene. Use as a context manager."""

    def __init__(self, scene, binary=None, density=1000.0):
        self.scene = scene
        self.density = density
        self.spacing = 1.0 / scene['cells_per_metre']
        nx, ny, nz = scene['counts']
        self._shapes = {'pressure': (nz, ny, nx), 'vx': (nz, ny, nx + 1), 'vy': (nz, ny + 1, nx),
                        'vz': (nz + 1, ny, nx), 'tracer': (nz, ny, nx)}
        self._errors = tempfile.TemporaryFile()
        self._child = subprocess.Popen([str(binary or probe_for(scene))], stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=self._errors,
                                       env={**os.environ, 'RAYON_NUM_THREADS': '1'})
        try:
            self._send({key: value for key, value in scene.items() if key not in _LOCAL_KEYS})
            self._expect('ready')
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def restore(self, step, state):
        """Replace the probe's state with `state` (no batch dimension), declared to be at output `step`."""
        self._send({'kind': 'restore', 'step': step})
        # The probe requires the solver's gauge: pressure of the first cell exactly zero.
        pressure = state.pressure.double()
        for tensor, kind in ((pressure - pressure.flatten()[0], '<f8'), (state.u, '<f4'), (state.v, '<f4'),
                             (state.w, '<f4'), (state.tracer, '<f4')):
            self._child.stdin.write(np.ascontiguousarray(tensor.numpy(), dtype=kind).tobytes())
        self._child.stdin.flush()
        self._expect('restored')

    def advance_native(self):
        """One output step with Rheon's own pressure solve. Returns the probe's report of the step."""
        self._send({'kind': 'advance', 'native': True})
        return self._expect('frame')['frame']

    def advance_reference(self):
        """One output step with the exact pressure solve.

        Returns (report, labels): the probe's report of the step, and the pressure
        equation it posed as float64 [Z, Y, X] tensors `rhs`, `native_input_residual`
        and `pressure_base`, which reference files keep beside the state.
        """
        self._send({'kind': 'advance', 'native': False})
        self._expect('predict')
        grid = self._shapes['pressure']
        base = torch.from_numpy(self._array(grid, '<f8'))
        residual = torch.from_numpy(self._array(grid, '<f8'))
        pressure = base + spectral_correction(residual, self.spacing, self.density)
        pressure = pressure - pressure[0, 0, 0]
        self._child.stdin.write(np.ascontiguousarray(pressure.numpy(), dtype='<f8').tobytes())
        self._child.stdin.flush()
        labels = {'rhs': residual + pressure_flux(base, self.spacing, self.density),
                  'native_input_residual': residual, 'pressure_base': base}
        return self._expect('frame')['frame'], labels

    def state(self):
        """The probe's current (output step, state)."""
        self._send({'kind': 'inspect'})
        header = self._expect('state')
        fields = {name: torch.from_numpy(self._array(shape, '<f8' if name == 'pressure' else '<f4'))
                  for name, shape in self._shapes.items()}
        return header['step'], FluidState(u=fields['vx'], v=fields['vy'], w=fields['vz'], tracer=fields['tracer'],
                                          pressure=fields['pressure'])

    def close(self):
        if self._child is None:
            return
        try:
            if self._child.poll() is None:
                self._send({'kind': 'stop'})
                self._child.stdin.close()
                self._child.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            pass
        finally:
            if self._child.poll() is None:
                self._child.kill()
                self._child.wait()
            self._errors.close()
            self._child = None

    def _send(self, message):
        self._child.stdin.write((json.dumps(message, separators=(',', ':')) + '\n').encode())
        self._child.stdin.flush()

    def _expect(self, kind):
        line = self._child.stdout.readline(_MAX_LINE + 1)
        if not line.endswith(b'\n'):
            self._errors.seek(0)
            raise RuntimeError(f'The Rheon probe stopped answering: {self._errors.read()[-400:].decode(errors="replace")}')
        message = json.loads(line)
        if message.get('kind') != kind:
            raise RuntimeError(f"The Rheon probe answered {message.get('kind')!r} where {kind!r} was expected.")
        return message

    def _array(self, shape, kind):
        size = math.prod(shape) * np.dtype(kind).itemsize
        payload = self._child.stdout.read(size)
        if len(payload) != size:
            raise RuntimeError('The Rheon probe sent a truncated field.')
        return np.frombuffer(payload, dtype=kind).reshape(shape).copy()
