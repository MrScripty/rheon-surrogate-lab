"""Read the lab's recorded reference trajectories.

A "bank" is an index file listing trajectories; each trajectory is a scene with
saved states at output steps (1/60 s apart). Two index layouts exist and both are
read here:

    dense   "states": 129         every step 0..128, files state-NNN.npz beside the receipt
    sparse  "states": [{step, path}, ...]   only some steps were kept

State files hold flat arrays with x fastest. This module reshapes them to the
layout in fields.py and never changes values: velocity and tracer stay float32,
pressure stays float64, pressure keeps the solver's gauge (cell 0 pinned to zero).
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .fields import FluidState, Scene
from .forcing import OUTPUT_RATE_HZ, velocity_increment

REPOSITORY = Path(__file__).resolve().parents[1]
# Fixed in the Rheon build that produced every bank so far (its diagnostic/scene.rs); receipts do not
# record them. A bank from a build with other constants must carry them itself.
WATER_DENSITY = 1000.0      # kg/m^3
WATER_VISCOSITY = 1.0e-3    # Pa s


@dataclass(frozen=True)
class Trajectory:
    id: str
    split: str
    scene: dict                 # forcing parameters; see forcing.py
    paths: dict                 # output step -> state file
    attributes: dict = field(default_factory=dict)

    @property
    def grid(self):
        """Cell counts (Z, Y, X)."""
        nx, ny, nz = self.scene['counts']
        return nz, ny, nx

    @property
    def cells(self):
        z, y, x = self.grid
        return z * y * x

    @property
    def spacing(self):
        return 1.0 / self.scene['cells_per_metre']

    def starts(self, intervals):
        """Steps from which `intervals` consecutive later states are all saved."""
        return [step for step in sorted(self.paths)
                if all(step + lead in self.paths for lead in range(1, intervals + 1))]

    def state(self, step):
        z, y, x = self.grid
        shapes = {'pressure': (z, y, x), 'vx': (z, y, x + 1), 'vy': (z, y + 1, x), 'vz': (z + 1, y, x),
                  'tracer': (z, y, x)}
        with np.load(self.paths[step], allow_pickle=False) as archive:
            arrays = {name: torch.from_numpy(archive[name].reshape(shape)) for name, shape in shapes.items()}
        return FluidState(u=arrays['vx'], v=arrays['vy'], w=arrays['vz'], tracer=arrays['tracer'],
                          pressure=arrays['pressure'])

    def states(self, start, count):
        """`count` consecutive states from `start`, stacked along a leading time dimension."""
        loaded = [self.state(step) for step in range(start, start + count)]
        return FluidState(**{name: torch.stack([getattr(state, name) for state in loaded])
                             for name in ('u', 'v', 'w', 'tracer', 'pressure')})

    def right_hand_sides(self, start, count):
        """The pressure equation's right-hand side saved with each of `count` states from `start`: [count, Z, Y, X]."""
        loaded = []
        for step in range(start, start + count):
            with np.load(self.paths[step], allow_pickle=False) as archive:
                loaded.append(torch.from_numpy(archive['rhs'].reshape(self.grid)))
        return torch.stack(loaded)

    def forcing(self, start, intervals):
        """Velocity added during each of `intervals` output intervals from `start`: three [intervals, ...] float32."""
        increments = [velocity_increment(self.scene, step) for step in range(start, start + intervals)]
        return tuple(torch.stack([increment[axis] for increment in increments]).float() for axis in range(3))

    def scene_constants(self, dtype=torch.float32):
        one = lambda value: torch.tensor([value], dtype=dtype)
        return Scene(spacing=one(self.spacing), timestep=one(1.0 / OUTPUT_RATE_HZ),
                     density=one(WATER_DENSITY), viscosity=one(WATER_VISCOSITY))


def load_bank(index, repository=REPOSITORY):
    """Trajectories listed by a bank index file.

    Paths inside the index are relative to `repository`, or to the index's own
    folder when it says `"paths_relative_to": "bank"` (generated datasets do).
    """
    repository = Path(repository)
    index = Path(index)
    index = index if index.is_absolute() else repository / index
    document = json.loads(index.read_text())
    base = index.parent if document.get('paths_relative_to') == 'bank' else repository
    trajectories = []
    for record in document['trajectories']:
        receipt = base / record['receipt']
        scene = json.loads(receipt.read_text())['scene']
        if list(record['counts_xyz']) != list(scene['counts']):
            raise ValueError(f"{record['identifier']}: the index and the receipt disagree on the grid.")
        for key in ('reference_substeps', 'forcing_control_substeps'):
            if key in record:
                scene[key] = record[key]
        if isinstance(record['states'], int):
            paths = {step: receipt.parent / f'state-{step:03d}.npz' for step in range(record['states'])}
        else:
            paths = {state['step']: base / state['path'] for state in record['states']}
        attributes = {name: record[name] for name in ('group', 'profile', 'motion', 'reference_substeps')
                      if name in record}
        trajectories.append(Trajectory(id=record['identifier'], split=record['split'], scene=scene, paths=paths,
                                       attributes=attributes))
    return trajectories


def load_banks(indexes, repository=REPOSITORY):
    trajectories = [trajectory for index in indexes for trajectory in load_bank(index, repository)]
    identities = [trajectory.id for trajectory in trajectories]
    if len(set(identities)) != len(identities):
        raise ValueError('The same trajectory is listed more than once across these banks.')
    return trajectories


def windows(trajectories, intervals):
    """Every (trajectory, start) with `intervals` consecutive later states saved."""
    return [(trajectory, start) for trajectory in trajectories for start in trajectory.starts(intervals)]


def batch(selection, intervals):
    """Stack windows of one grid shape for training.

    Returns (start state, forcing, future states, scene) with fields
    [B, ...], [B, intervals, ...], [B, intervals, ...] and [B]; everything float32.
    """
    grids = {trajectory.grid for trajectory, _ in selection}
    if len(grids) != 1:
        raise ValueError('A batch must hold windows of one grid shape.')
    histories = [trajectory.states(start, intervals + 1) for trajectory, start in selection]
    stacked = FluidState(**{name: torch.stack([getattr(history, name) for history in histories]).float()
                            for name in ('u', 'v', 'w', 'tracer', 'pressure')})
    forcings = [trajectory.forcing(start, intervals) for trajectory, start in selection]
    forcing = tuple(torch.stack([f[axis] for f in forcings]) for axis in range(3))
    constants = [trajectory.scene_constants() for trajectory, _ in selection]
    scene = Scene(**{name: torch.cat([getattr(c, name) for c in constants])
                     for name in ('spacing', 'timestep', 'density', 'viscosity')})
    return stacked.map(lambda t: t[:, 0]), forcing, stacked.map(lambda t: t[:, 1:]), scene
