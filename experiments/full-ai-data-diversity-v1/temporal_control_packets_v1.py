"""Deployment-available force packets at explicit integer fine-clock indices."""
import math

import numpy as np


def control_packet(scene, start_fine, fine_steps):
    factor = scene['reference_substeps']
    rate = 60 * factor
    if (factor not in (4, 8) or scene['forcing_control_substeps'] != factor
            or type(start_fine) is not int or type(fine_steps) is not int
            or not 0 <= start_fine < start_fine + fine_steps <= scene['steps'] * factor
            or len(scene['forcing_schedule']) != scene['steps'] * factor):
        raise ValueError('Explicit bounded fine-clock control interval required')
    seed = scene['seed']
    mask = (1 << 64) - 1

    def next_value():
        nonlocal seed
        seed = (seed + 0x9e3779b97f4a7c15) & mask
        value = seed
        value = ((value ^ (value >> 30)) * 0xbf58476d1ce4e5b9) & mask
        value = ((value ^ (value >> 27)) * 0x94d049bb133111eb) & mask
        return float((value ^ (value >> 31)) >> 11) / float(1 << 53)

    direction = []
    norm = 0.
    for _ in range(3):
        value = 2. * next_value() - 1.
        direction.append(value)
        norm = norm + value * value
    norm = max(math.sqrt(norm), 1e-12)
    direction = [value / norm for value in direction]
    tau = 6.283185307179586476925286766559
    phase = next_value() * tau
    h = 1. / scene['cells_per_metre']
    extents = [float(n) * h for n in scene['counts']]
    radius = scene['radius'] * min(extents)
    omega = tau * scene['frequency']
    dt = 1. / rate
    rows = np.empty((fine_steps, 18), dtype='<f8')
    for offset in range(fine_steps):
        step = start_fine + offset
        frame = scene['forcing_schedule'][step]
        center = [extents[a] * frame['center'][a] for a in range(3)]
        rows[offset] = [h, *extents, *center, radius, omega, float(step) * dt,
                        phase, *direction, scene['acceleration'] * frame['pulse'],
                        scene['swirl'], scene['fluctuation'], dt]
    if not np.isfinite(rows).all():
        raise ValueError('Nonfinite known external controls')
    return rows
