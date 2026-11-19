"""The prescribed forcing of a recorded scene, as the velocity it adds per output interval.

Rheon's reference scenes push the fluid with a soft-edged blob of acceleration:
a direction, a swirl around it and a travelling wave, scaled by a pulse. Scenes
store the parameters, never the field, so this module rebuilds the field. It
reproduces the lab's two existing implementations (the scheduled-force kernel
and the older parametric one) in plain torch on the CPU.

A scene is the `scene` dictionary of a capture receipt. Newer scenes carry a
`forcing_schedule` (centre and pulse per control step); older ones move the
centre and pulse sinusoidally.
"""
import math

import torch

OUTPUT_RATE_HZ = 60
_MASK = (1 << 64) - 1


def _direction_and_phase(seed):
    """The scene's fixed push direction and wave phase, from its seed (splitmix64)."""
    state = seed

    def next_value():
        nonlocal state
        state = (state + 0x9E3779B97F4A7C15) & _MASK
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
        return ((z ^ (z >> 31)) >> 11) / (1 << 53)

    direction = [2 * next_value() - 1 for _ in range(3)]
    norm = max(math.sqrt(sum(value * value for value in direction)), 1e-12)
    return [value / norm for value in direction], next_value() * math.tau


def _acceleration(scene, centre, pulse, time_s):
    """Acceleration on the faces of each axis, m/s^2, as (ax [Z,Y,X+1], ay [Z,Y+1,X], az [Z+1,Y,X])."""
    counts = scene['counts']                      # cells along x, y, z
    spacing = 1.0 / scene['cells_per_metre']
    extents = [count * spacing for count in counts]
    radius = scene['radius'] * min(extents)
    direction, phase = _direction_and_phase(scene['seed'])
    omega = math.tau * scene['frequency']
    fields = []
    for d in range(3):
        positions = []
        for axis, count in enumerate(counts):
            faces = axis == d
            coordinate = (torch.arange(count + faces, dtype=torch.float64) + (0.0 if faces else 0.5)) * spacing
            shape = [1, 1, 1]
            shape[2 - axis] = count + faces       # tensors are ordered z, y, x
            positions.append(coordinate.reshape(shape))
        offset = [(positions[axis] - extents[axis] * centre[axis]) / radius for axis in range(3)]
        r2 = sum(component.square() for component in offset)
        envelope = torch.where(r2 < 16, torch.exp(-0.5 * r2) * (1 - r2 / 16).square(), torch.zeros_like(r2))
        a, b = (d + 1) % 3, (d + 2) % 3
        rotation = direction[a] * offset[b] - direction[b] * offset[a]
        wave = sum(torch.sin(math.tau * positions[k] / extents[k] + omega * time_s * (k + 1) + phase + d)
                   for k in range(3)) / 3
        field = scene['acceleration'] * pulse * envelope * (
            direction[d] + scene['swirl'] * rotation + scene['fluctuation'] * wave)
        wall = [slice(None)] * 3
        for end in (0, -1):                       # no forcing through the walls
            wall[2 - d] = end
            field[tuple(wall)] = 0.0
        fields.append(field)
    return tuple(fields)


def velocity_increment(scene, step):
    """Velocity the forcing adds during output interval `step` -> `step + 1`, m/s, float64.

    Scenes integrated with several solver substeps per output get the sum of the
    substep increments. The solver applies each between its own advection and
    projection, so for those scenes the sum describes the forcing, not exactly
    the velocity change it caused.
    """
    if not 0 <= step < scene['steps']:
        raise ValueError(f"Step {step} is outside the scene's {scene['steps']} recorded intervals.")
    substeps = scene.get('reference_substeps', 1)
    fine_dt = 1.0 / (OUTPUT_RATE_HZ * substeps)
    schedule = scene.get('forcing_schedule')
    total = None
    for fine in range(step * substeps, (step + 1) * substeps):
        time_s = fine * fine_dt
        if schedule is not None:
            controls_per_output = scene.get('forcing_control_substeps', 1)
            frame = schedule[fine // (substeps // controls_per_output)]
            centre, pulse = frame['center'], frame['pulse']
        else:
            _, phase = _direction_and_phase(scene['seed'])
            swing = math.tau * scene['frequency'] * time_s + phase
            centre = [scene['center'][axis] + 0.05 * math.sin(swing + axis) for axis in range(3)]
            pulse = 0.65 + 0.35 * math.sin(swing)
        increment = [fine_dt * field for field in _acceleration(scene, centre, pulse, time_s)]
        total = increment if total is None else [t + i for t, i in zip(total, increment)]
    return tuple(total)
