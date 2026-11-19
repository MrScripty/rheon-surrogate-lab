"""Strict prescribed legacy inputs for the existing full-AI prototype.

Only known scene parameters and physical time enter this construction. No
physical state, pressure solve, transport or viscosity operator is evaluated.
"""
from contextlib import contextmanager
from functools import lru_cache
import json
import math
from support import ROOT
from force_direction import direction_phase
from scheduled_force_fine_controls import force


@lru_cache(maxsize=128)
def schedule(serialized):
    scene = json.loads(serialized)
    if scene.get('forcing_schedule') is not None or scene.get('forcing_program') is not None:
        raise ValueError('This control applies only to the original analytic legacy inputs')
    if not 1 <= scene['steps'] <= 600:
        raise ValueError('Bounded legacy external input horizon required')
    _, phase = direction_phase(scene['seed'])
    omega = math.tau*scene['frequency']
    frames = []
    for step in range(scene['steps']):
        time = step*(1/60)
        frames.append(dict(
            center=[scene['center'][a]+.05*math.sin(omega*time+phase+a) for a in range(3)],
            pulse=.65+.35*math.sin(omega*time+phase)))
    return dict(scene, forcing_schedule=frames, reference_substeps=1, forcing_control_substeps=1)


def forcing(scene, step, horizons):
    if horizons not in (1, 2, 3, 4) or not 0 <= step < step+horizons <= scene['steps']:
        raise ValueError('Complete bounded external input block required')
    prescribed = schedule(json.dumps(scene, sort_keys=True, separators=(',', ':')))
    return [force(prescribed, step+lead)[0] for lead in range(horizons)]


@contextmanager
def bind_for_diagnostic():
    import learned_pressure_history
    previous = learned_pressure_history.forcing
    learned_pressure_history.forcing = forcing
    try:
        yield
    finally:
        learned_pressure_history.forcing = previous
