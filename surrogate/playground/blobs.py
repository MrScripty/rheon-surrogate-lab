"""Forces as soft blobs of acceleration, and the velocity they add per step.

A blob is the forcing shape of the lab's scenes, with its parameters set by hand instead of by a seed:

    acceleration = magnitude * envelope(r) * (direction + swirl * rotation + shake * wave)

where r is the distance from the centre in radii, envelope = exp(-r^2/2)(1 - r^2/16)^2 inside 4 radii and 0
outside, rotation is the turn about `direction` (offset x direction, in radii), and wave is a travelling sinusoid
that changes sign SHAKE_HZ times a second. No forcing goes through a wall.

`packet_row` writes a blob as one row of the solver composition's 18-column known-control packet, and
`kernel_increment` evaluates such a row exactly as the composition's CUDA kernel `known_control_dynamic` does
(runs/full-ai-data-diversity-v1/cuda-fusion-build-04/source.cu), in float64 on any torch device. Every backend uses
`increment` for the blobs it does not hand to that kernel, so a blob means the same force everywhere.
"""
import math
from dataclasses import dataclass

import torch

SHAKE_HZ = 2.0
MAX_MAGNITUDE = 1000.0      # m/s^2
MAX_COEFFICIENT = 20.0      # swirl and shake
PACKET_COLUMNS = ('h', 'extent_x', 'extent_y', 'extent_z', 'centre_x', 'centre_y', 'centre_z', 'radius', 'omega',
                  'time', 'phase', 'direction_x', 'direction_y', 'direction_z', 'amplitude', 'swirl', 'fluctuation',
                  'dt')


@dataclass(frozen=True)
class Blob:
    centre: tuple        # fractions of the box along x, y, z
    radius: float        # m (the Gaussian's width; the force reaches 4 radii)
    direction: tuple     # unit vector, or zero for a blob that only shakes
    magnitude: float     # m/s^2
    swirl: float = 0.0
    shake: float = 0.0

    @classmethod
    def from_json(cls, value):
        """A blob from the page's JSON. Raises ValueError with the reason when a value is unusable."""
        if not isinstance(value, dict):
            raise ValueError('A force is an object with centre, radius, direction and magnitude.')
        try:
            centre = [float(v) for v in value['centre']]
            direction = [float(v) for v in value.get('direction', (0, 0, 0))]
            radius, magnitude = float(value['radius']), float(value.get('magnitude', 0))
            swirl, shake = float(value.get('swirl', 0)), float(value.get('shake', 0))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f'A force needs numeric centre, radius, direction and magnitude ({error}).') from None
        numbers = [*centre, *direction, radius, magnitude, swirl, shake]
        if len(centre) != 3 or len(direction) != 3 or not all(math.isfinite(v) for v in numbers):
            raise ValueError('A force has three centre and three direction components, all finite.')
        if not 0 < radius <= 100:
            raise ValueError('A force radius is between 0 and 100 m.')
        if abs(magnitude) > MAX_MAGNITUDE or abs(swirl) > MAX_COEFFICIENT or abs(shake) > MAX_COEFFICIENT:
            raise ValueError(f'Magnitude is at most {MAX_MAGNITUDE:g} m/s² and swirl and shake at most '
                             f'{MAX_COEFFICIENT:g}.')
        norm = math.sqrt(sum(v * v for v in direction))
        direction = [v / norm for v in direction] if norm > 1e-12 else [0.0, 0.0, 0.0]
        centre = [min(max(v, 0.0), 1.0) for v in centre]
        return cls(tuple(centre), radius, tuple(direction), magnitude, swirl, shake)

    def to_json(self):
        return dict(centre=list(self.centre), radius=self.radius, direction=list(self.direction),
                    magnitude=self.magnitude, swirl=self.swirl, shake=self.shake)


def packet_row(blob, counts, cells_per_metre, time_s, dt, omega=None, phase=0.0):
    """The composition's control-packet row for `blob` at simulated time `time_s`. `counts` are cells along x, y, z."""
    h = 1.0 / cells_per_metre
    extents = [n * h for n in counts]
    omega = math.tau * SHAKE_HZ if omega is None else omega
    centre = [extents[a] * blob.centre[a] for a in range(3)]
    return [h, *extents, *centre, blob.radius, omega, time_s, phase, *blob.direction, blob.magnitude, blob.swirl,
            blob.shake, dt]


def face_shape(d, counts):
    """Tensor shape (z, y, x) of the faces normal to axis d."""
    nx, ny, nz = counts
    return ((nz, ny, nx + 1), (nz, ny + 1, nx), (nz + 1, ny, nx))[d]


def kernel_increment(row, counts, device='cpu'):
    """Velocity increments (x, y, z faces), float64, of one packet row: `known_control_dynamic` in torch."""
    h, extents, centre, radius = row[0], row[1:4], row[4:7], row[7]
    omega, time_s, phase = row[8], row[9], row[10]
    direction, amplitude, swirl, fluctuation, dt = row[11:14], row[14], row[15], row[16], row[17]
    result = []
    for d in range(3):
        shape = face_shape(d, counts)
        offsets, r2, wave = [], 0.0, 0.0
        for a in range(3):
            n = counts[a] + (a == d)
            pos = (torch.arange(n, dtype=torch.float64, device=device) + (0.0 if a == d else 0.5)) * h
            view = [1, 1, 1]
            view[2 - a] = n
            offset = ((pos - centre[a]) / radius).reshape(view)
            offsets.append(offset)
            r2 = r2 + offset * offset
            wave = wave + torch.sin(math.tau * pos / extents[a] + omega * time_s * (a + 1) + phase + d).reshape(view)
        r2 = r2.expand(shape)
        cutoff = 1.0 - r2 / 16.0
        envelope = torch.where(r2 < 16.0, torch.exp(-0.5 * r2) * (cutoff * cutoff), torch.zeros_like(r2))
        a, b = (d + 1) % 3, (d + 2) % 3
        rotation = direction[a] * offsets[b] - direction[b] * offsets[a]
        acceleration = (amplitude * envelope) * ((direction[d] + swirl * rotation) + fluctuation * (wave / 3.0))
        field = dt * acceleration
        index = [slice(None)] * 3
        for end in (0, -1):
            index[2 - d] = end
            field[tuple(index)] = 0.0
        result.append(field)
    return tuple(result)


def increment(blobs, counts, cells_per_metre, time_s, dt, device='cpu'):
    """Summed velocity increments of several blobs over one step of length dt, float64 (x, y, z faces)."""
    total = [torch.zeros(face_shape(d, counts), dtype=torch.float64, device=device) for d in range(3)]
    for blob in blobs:
        for d, part in enumerate(kernel_increment(packet_row(blob, counts, cells_per_metre, time_s, dt), counts,
                                                  device)):
            total[d] += part
    return tuple(total)
