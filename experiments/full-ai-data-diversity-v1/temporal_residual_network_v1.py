"""CUDA residual prototype with a learned encoder for variable force paths.

Current fields and coarse predicted fields are deployment-available inputs.
The force encoder compresses the ordered path; it is a finite hypothesis, not
a proof that every possible control history remains distinguishable.
"""
import torch
from torch import nn
from torch.nn import functional as F

from internal_residual_network_v1 import _ClampedPad, wall
from recurrent_sampler_native import dimensions

SPATIAL_CHANNELS = 34
FORCE_CHANNELS = 6
FORCE_HIDDEN = 4
HIDDEN = 8
PARAMETERS = 12732


def face_pairs(value, component):
    axis = 2 - component
    return value.narrow(axis, 0, value.shape[axis] - 1), value.narrow(axis, 1, value.shape[axis] - 1)


@torch.no_grad()
def features(phase):
    """Nondimensional current/coarse fields, explicit timing and ordered forces."""
    counts, shape = phase['counts'], phase['shape']
    sizes = dimensions(counts)
    h, dt, rho = phase['h'], phase['dt'], phase['rho']
    width = phase['width']
    if len(phase['forcing']) != width or not 1 <= width <= 64:
        raise ValueError('All explicitly bounded force-path rows required')
    current = phase['current']
    velocity = [current[n].reshape(s) for n, s in zip(('vx', 'vy', 'vz'), sizes[:3])]
    scale = dt / h
    values = []
    for c, v in enumerate(velocity):
        values.extend(a.double() * scale for a in face_pairs(v, c))
    pressure = current['pressure'].reshape(shape).double()
    values.extend(((pressure - pressure.mean()) * (dt * dt / (rho * h * h)),
                   current['tracer'].reshape(shape).double()))
    for c, (advected, v) in enumerate(zip(phase['advected'][:3], velocity)):
        values.extend(a * scale for a in face_pairs(advected.double() - v.double(), c))
    for c, (provisional, advected) in enumerate(zip(phase['provisional'], phase['advected'][:3])):
        values.extend(a * scale for a in face_pairs(provisional.double() - advected.double(), c))
    values.append(phase['base_tracer'].double() - current['tracer'].reshape(shape).double())
    coarse_p = phase['base_pressure'].reshape(shape)
    values.append((coarse_p - coarse_p.mean()) * (dt * dt / (rho * h * h)))
    for c, v in enumerate(phase['base_velocity']):
        values.extend(a.double() * scale for a in face_pairs(v, c))
    template = values[0]
    values.extend(torch.full_like(template, v) for v in
                  (240 * dt, 240 * phase['reference_dt_s'], 100 * h,
                   counts[0] * h, counts[1] * h, counts[2] * h))
    if len(values) != SPATIAL_CHANNELS or any(tuple(v.shape) != shape for v in values):
        raise ValueError('Frozen physical feature schema differs')
    force_rows = []
    active = torch.zeros((), dtype=torch.bool, device='cuda')
    for v in velocity:
        active = active | torch.count_nonzero(v).ne(0)
    for row in phase['forcing']:
        components = []
        for c, value in enumerate(row):
            active = active | torch.count_nonzero(value).ne(0)
            components.extend(a * scale for a in face_pairs(value, c))
        force_rows.append(torch.stack(components))
    # Companding is fixed physical preprocessing, not a fitted normalizer.
    spatial = torch.asinh(torch.stack(values)).unsqueeze(0).float()
    ordered = torch.asinh(torch.stack(force_rows)).permute(1, 0, 2, 3, 4).unsqueeze(0).float()
    return spatial, ordered.contiguous(), active


class TemporalResidual(nn.Module):
    def __init__(self, device='cuda', seed=1729):
        super().__init__()
        device = torch.device(device)
        devices = [torch.cuda.current_device()] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            self.force1 = nn.Conv1d(FORCE_CHANNELS, FORCE_HIDDEN, 3, padding=1, device=device, dtype=torch.float32)
            self.force2 = nn.Conv1d(FORCE_HIDDEN, FORCE_HIDDEN, 3, padding=1, device=device, dtype=torch.float32)
            self.conv1 = nn.Conv3d(SPATIAL_CHANNELS + 4 * FORCE_HIDDEN, HIDDEN, 3, device=device, dtype=torch.float32)
            self.conv2 = nn.Conv3d(HIDDEN, HIDDEN, 3, device=device, dtype=torch.float32)
            self.velocity = nn.ModuleList(nn.ConvTranspose3d(HIDDEN, 1, k, device=device, dtype=torch.float32)
                for k in ((1, 1, 2), (1, 2, 1), (2, 1, 1)))
            self.tracer = nn.Conv3d(HIDDEN, 1, 1, device=device, dtype=torch.float32)
            for layer in (*self.velocity, self.tracer):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
        if sum(p.numel() for p in self.parameters()) != PARAMETERS:
            raise ValueError('Temporal residual parameter count changed')

    def encode_forces(self, ordered):
        if ordered.device.type != 'cuda' or ordered.dtype != torch.float32 or ordered.ndim != 6:
            raise ValueError('CUDA F32 ordered force path required')
        b, channels, width, nz, ny, nx = ordered.shape
        if b != 1 or channels != FORCE_CHANNELS or not 1 <= width <= 64:
            raise ValueError('Explicit one-to64-row force path required')
        sequence = ordered.permute(0, 3, 4, 5, 1, 2).reshape(-1, channels, width)
        hidden = F.silu(self.force1(sequence))
        hidden = F.silu(self.force2(hidden))
        position = (torch.arange(width, device='cuda', dtype=torch.float32) + .5) * (2 / width) - 1
        pooled = torch.cat((hidden[:, :, 0], hidden[:, :, -1], hidden.mean(-1),
                            (hidden * position).mean(-1)), dim=1)
        return pooled.reshape(1, nz, ny, nx, 4 * FORCE_HIDDEN).permute(0, 4, 1, 2, 3).contiguous()

    def forward(self, spatial, ordered, active):
        if (spatial.device.type != 'cuda' or spatial.dtype != torch.float32 or spatial.ndim != 5
                or spatial.shape[:2] != (1, SPATIAL_CHANNELS)
                or active.device.type != 'cuda' or active.dtype != torch.bool or active.numel() != 1):
            raise ValueError('CUDA physical feature field and activity scalar required')
        forcing = self.encode_forces(ordered)
        hidden = F.silu(self.conv1(_ClampedPad.apply(torch.cat((spatial, forcing), dim=1))))
        hidden = F.silu(self.conv2(_ClampedPad.apply(hidden)))
        velocity = [wall(layer(hidden)[0, 0], c) for c, layer in enumerate(self.velocity)]
        outputs = [*velocity, self.tracer(hidden)[0, 0]]
        return [torch.where(active, v, v.new_zeros(())) for v in outputs]


def physical_deltas(values, phase):
    return [v.double() * (phase['h'] / phase['dt']) for v in values[:3]], values[3].double()
