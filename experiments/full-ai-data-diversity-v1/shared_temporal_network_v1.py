"""Small shared spatial trunk and explicit prescribed-control encoders.

Force-path convolutions run once on scalar controls, rather than separately
at every voxel. Exact event offsets are an additional finite conditioning
hypothesis. Neither compressed encoder is claimed to preserve every possible
force history. All learned forwards require CUDA; CPU construction is for
typed parameter restoration only.
"""
import torch
from torch import nn
from torch.nn import functional as F

from internal_residual_network_v1 import _ClampedPad, wall
from temporal_residual_network_v1 import face_pairs
from recurrent_sampler_native import dimensions

SPATIAL_CHANNELS = 26
CONTROL_CHANNELS = 24
EVENT_CHANNELS = 5
CONTROL_HIDDEN = 8


def heads(hidden, velocity, tracer, active):
    values = [wall(layer(hidden)[0, 0], c) for c, layer in enumerate(velocity)]
    values.append(tracer(hidden)[0, 0])
    return [torch.where(active, value, value.new_zeros(())) for value in values]


class SharedTemporalResidual(nn.Module):
    def __init__(self, hidden=8, device='cuda', seed=27183):
        super().__init__()
        if hidden not in (4, 8):
            raise ValueError('Registered full and reduced width variants required')
        self.hidden = hidden
        device = torch.device(device)
        devices = [torch.cuda.current_device()] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            self.force1 = nn.Conv1d(CONTROL_CHANNELS, CONTROL_HIDDEN, 3, padding=1, device=device, dtype=torch.float32)
            self.force2 = nn.Conv1d(CONTROL_HIDDEN, CONTROL_HIDDEN, 3, padding=1, device=device, dtype=torch.float32)
            self.event1 = nn.Conv1d(EVENT_CHANNELS, CONTROL_HIDDEN, 1, device=device, dtype=torch.float32)
            self.event2 = nn.Conv1d(CONTROL_HIDDEN, CONTROL_HIDDEN, 1, device=device, dtype=torch.float32)
            self.mix = nn.Conv3d(SPATIAL_CHANNELS + 5 * CONTROL_HIDDEN, hidden, 1, device=device, dtype=torch.float32)
            self.local = nn.Conv3d(hidden, hidden, 3, device=device, dtype=torch.float32)
            self.endpoint_velocity = nn.ModuleList(nn.ConvTranspose3d(hidden, 1, k, device=device, dtype=torch.float32)
                for k in ((1, 1, 2), (1, 2, 1), (2, 1, 1)))
            self.endpoint_tracer = nn.Conv3d(hidden, 1, 1, device=device, dtype=torch.float32)
            self.interior_mix = nn.Conv3d(hidden + 8, hidden, 1, device=device, dtype=torch.float32)
            self.interior_velocity = nn.ModuleList(nn.ConvTranspose3d(hidden, 1, k, device=device, dtype=torch.float32)
                for k in ((1, 1, 2), (1, 2, 1), (2, 1, 1)))
            self.interior_tracer = nn.Conv3d(hidden, 1, 1, device=device, dtype=torch.float32)
            for layer in (*self.endpoint_velocity, self.endpoint_tracer,
                          *self.interior_velocity, self.interior_tracer):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

    def encode(self, spatial, controls, events, event_mask):
        if (spatial.device.type != 'cuda' or spatial.dtype != torch.float32
                or spatial.ndim != 5 or spatial.shape[:2] != (1, SPATIAL_CHANNELS)
                or controls.device.type != 'cuda' or controls.dtype != torch.float32
                or controls.ndim != 3 or controls.shape[:2] != (1, CONTROL_CHANNELS)
                or not 1 <= controls.shape[-1] <= 32
                or events.device.type != 'cuda' or events.dtype != torch.float32
                or events.shape != (1, EVENT_CHANNELS, 32)
                or event_mask.device.type != 'cuda' or event_mask.dtype != torch.float32
                or event_mask.shape != (1, 1, 32)):
            raise ValueError('Exact CUDA spatial, ordered-control and padded event schemas required')
        temporal = F.silu(self.force2(F.silu(self.force1(controls))))
        width = temporal.shape[-1]
        position = (torch.arange(width, device='cuda', dtype=torch.float32) + .5) * (2 / width) - 1
        force_code = torch.cat((temporal[:, :, 0], temporal[:, :, -1], temporal.mean(-1),
                               (temporal * position).mean(-1)), 1)
        event_values = F.silu(self.event2(F.silu(self.event1(events)))) * event_mask
        event_code = event_values.sum(-1) / event_mask.sum(-1).clamp_min(1.)
        code = torch.cat((force_code, event_code), 1)[:, :, None, None, None]
        code = code.expand(-1, -1, *spatial.shape[-3:])
        hidden = F.silu(self.mix(torch.cat((spatial, code), 1)))
        return F.silu(self.local(_ClampedPad.apply(hidden)))

    def endpoint(self, hidden, active):
        if hidden.device.type != 'cuda' or active.device.type != 'cuda':
            raise ValueError('CUDA-only endpoint decoder')
        return heads(hidden, self.endpoint_velocity, self.endpoint_tracer, active)

    def interior(self, hidden, conditioning, active):
        if (hidden.device.type != 'cuda' or conditioning.device.type != 'cuda'
                or conditioning.dtype != torch.float32 or conditioning.shape[1] != 8
                or conditioning.shape[-3:] != hidden.shape[-3:] or active.device.type != 'cuda'):
            raise ValueError('CUDA-only predicted-endpoint interior conditioning')
        value = F.silu(self.interior_mix(torch.cat((hidden, conditioning), 1)))
        return heads(value, self.interior_velocity, self.interior_tracer, active)


@torch.no_grad()
def spatial_context(phase):
    shape, counts = phase['shape'], phase['counts']
    scale = phase['dt'] / phase['h']
    sizes = dimensions(counts)
    values = []
    for fields in ([phase['current'][n].reshape(s) for n, s in zip(('vx', 'vy', 'vz'), sizes[:3])], phase['base_velocity']):
        for c, value in enumerate(fields):
            values.extend(a.float() * scale for a in face_pairs(value, c))
    for pressure, tracer in [(phase['current']['pressure'].reshape(shape), phase['current']['tracer'].reshape(shape)),
                             (phase['base_pressure'].reshape(shape), phase['base_tracer'])]:
        values.extend((((pressure - pressure.mean()) * (phase['dt'] ** 2 / (phase['rho'] * phase['h'] ** 2))).float(), tracer.float()))
    template = values[0]
    values.extend(torch.full_like(template, v) for v in (240 * phase['dt'], 240 * phase['reference_dt_s'],
        100 * phase['h'], *(n * phase['h'] for n in counts), phase['width'] / 32))
    for axis, n in enumerate(shape):
        dims = [1, 1, 1]
        dims[axis] = n
        values.append(((torch.arange(n, device='cuda', dtype=torch.float32) + .5) / n).reshape(dims).expand(shape))
    if len(values) != SPATIAL_CHANNELS:
        raise ValueError('Physical shared spatial feature schema changed')
    return torch.stack(values).unsqueeze(0)


@torch.no_grad()
def control_context(packet, event_values, event_mask, phase):
    # All inputs here are prescribed controls, never future fluid fields.
    value = packet.clone()
    value[:, 0] *= 100
    value[:, 4:7] /= packet[:, 1:4]
    value[:, 7] /= packet[:, 1:4].amin(1)
    value[:, 8] *= phase['dt']
    value[:, 9] = (packet[:, 9] - packet[0, 9]) / phase['dt']
    value[:, 10] /= 6.2831853071795864769
    value[:, 14] *= phase['dt'] ** 2 / phase['h']
    value[:, 17] /= phase['dt']
    angles = torch.stack([packet[:, 8] * packet[:, 9] * a + packet[:, 10] for a in (1, 2, 3)], 1)
    waves = torch.stack((angles.sin(), angles.cos()), 2).flatten(1)
    controls = torch.cat((value, waves), 1).T.contiguous().float().unsqueeze(0)
    events = event_values.T.contiguous().float().unsqueeze(0)
    # Offset and pulse duration are normalized by the whole physical horizon.
    events[:, 0] /= phase['dt']
    events[:, 4] /= phase['dt']
    events[:, 2:4] *= phase['acceleration_m_per_s2'] * phase['dt'] ** 2 / phase['h']
    active = (torch.count_nonzero(packet[:, 14]).ne(0)
              | torch.count_nonzero(event_values[:, 3] * phase['acceleration_m_per_s2']).ne(0))
    for name in ('vx', 'vy', 'vz'):
        active = active | torch.count_nonzero(phase['current'][name]).ne(0)
    return controls, events, event_mask.float().reshape(1, 1, 32), active


def interior_conditioning(current, endpoint, phase, lead_dt, alpha):
    shape, counts = phase['shape'], phase['counts']
    sizes = dimensions(counts)
    values = []
    for c, (name, size) in enumerate(zip(('vx', 'vy', 'vz'), sizes[:3])):
        delta = (endpoint[name].reshape(size).double() - current[name].reshape(size).double())
        left, right = face_pairs(delta, c)
        values.append(((left + right) * (.5 * phase['dt'] / phase['h'])).float())
    dp = endpoint['pressure'].reshape(shape) - current['pressure'].reshape(shape)
    values.append(((dp - dp.mean()) * (phase['dt'] ** 2 / (phase['rho'] * phase['h'] ** 2))).float())
    values.append((endpoint['tracer'].reshape(shape).double() - current['tracer'].reshape(shape).double()).float())
    values.extend(torch.full_like(values[0], v) for v in (alpha, 240 * lead_dt, 240 * phase['dt']))
    return torch.stack(values).unsqueeze(0)
