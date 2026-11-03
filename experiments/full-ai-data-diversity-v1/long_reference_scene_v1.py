"""Independent 20-second prescribed programs and varied complete initial states."""
import math
import random

import numpy as np

from continuous_force_program import evaluate
from force_program import make_scene, PROFILES, MOTIONS


def make_long_scene(seed, counts, density, profile, motion, *, duration_s=20,
                    reference_substeps=4, force_frequency_hz=None):
    if (profile not in PROFILES or motion not in MOTIONS
            or duration_s not in (1, 20) or reference_substeps not in (4, 8)):
        raise ValueError('Registered long-reference scene contract required')
    scene = make_scene(seed, counts, density, profile, motion, steps=1)
    rng = random.Random(seed ^ 0x4D9F718B)
    program = scene['forcing_program']
    frequency = force_frequency_hz if force_frequency_hz is not None else rng.uniform(.25, 20.)
    if not math.isfinite(frequency) or frequency <= 0:
        raise ValueError('Positive physical forcing frequency required')
    program.update(
        version=3, duration_s=duration_s, evaluation='actual fine physical time',
        dt=1 / (60 * reference_substeps), pulse_frequency_hz=frequency,
        variance_weights=[.2, .3, .5],
        variance_frequencies_hz=[frequency * v for v in (.25, .5, 1.)],
        variance_phases=[rng.uniform(0., math.tau) for _ in range(3)],
        sharp_pulse_edges=profile in ('regular_pulse', 'irregular_pulse'),
        temporal_convergence_qualified=False,
    )
    if profile == 'irregular_pulse':
        events = []
        at = -rng.random() / frequency
        while at < duration_s:
            end = at + rng.uniform(.15, .85) / frequency
            events.append(dict(start=at, end=end, amplitude=rng.uniform(.2, 1.6)))
            at = end + rng.uniform(.15, .85) / frequency
        program['irregular_pulse_events'] = events
    scene.update(steps=duration_s * 60, reference_substeps=reference_substeps,
                 forcing_control_substeps=reference_substeps)
    dt = 1 / (60 * reference_substeps)
    scene['forcing_schedule'] = [evaluate(program, i * dt)
                                 for i in range(scene['steps'] * reference_substeps)]
    scene['center'] = scene['forcing_schedule'][0]['center']
    return scene


def initial_state(scene, recipe):
    """Seeded tracer mixtures plus an optional discrete curl with zero wall flux."""
    nx, ny, nz = scene['counts']
    h = 1 / scene['cells_per_metre']
    rng = random.Random(recipe['seed'])
    z, y, x = np.meshgrid((np.arange(nz) + .5) / nz,
                          (np.arange(ny) + .5) / ny,
                          (np.arange(nx) + .5) / nx, indexing='ij')
    tracer = np.zeros((nz, ny, nx), dtype='<f8')
    for _ in range(recipe['tracer_blobs']):
        center = [rng.uniform(.1, .9) for _ in range(3)]
        widths = [rng.uniform(.04, .16) for _ in range(3)]
        r2 = sum(((a - c) / w) ** 2 for a, c, w in zip((x, y, z), center, widths))
        tracer += rng.uniform(.1, .4) * np.exp(-.5 * r2)
    vx = np.zeros((nz, ny, nx + 1), dtype='<f8')
    vy = np.zeros((nz, ny + 1, nx), dtype='<f8')
    vz = np.zeros((nz + 1, ny, nx), dtype='<f8')
    speed = recipe['initial_velocity_max_m_per_s']
    if speed:
        psi = (np.cos(math.tau * z[:, :1, :1] * recipe['curl_z_modes'])
               * np.sin(math.pi * np.arange(ny + 1)[None, :, None] / ny) ** 2
               * np.sin(math.pi * np.arange(nx + 1)[None, None, :] / nx) ** 2)
        psi[:, [0, -1], :] = 0.
        psi[:, :, [0, -1]] = 0.
        vx = np.diff(psi, axis=1) / h
        vy = -np.diff(psi, axis=2) / h
        scale = speed / max(float(np.abs(vx).max()), float(np.abs(vy).max()), 1e-30)
        vx *= scale
        vy *= scale
    return dict(pressure=np.zeros(nx * ny * nz, dtype='<f8'),
                vx=vx.astype('<f4').reshape(-1), vy=vy.astype('<f4').reshape(-1),
                vz=vz.astype('<f4').reshape(-1),
                tracer=np.minimum(tracer, .9).astype('<f4').reshape(-1))
