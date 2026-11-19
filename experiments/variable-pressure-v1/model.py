"""Variable-shape GroupNorm pressure correction with cubic physical inputs."""
import copy
import math

import torch
from torch.nn import functional as F

import policy

frozen = policy.frozen_model()
require = frozen.require
configure_cpu = frozen.configure_cpu
parameter_count = frozen.parameter_count
CHANNELS = frozen.CHANNELS
ARCHITECTURE = copy.deepcopy(frozen.ARCHITECTURE)
ARCHITECTURE.update(schema='rheon-cubic-groupnorm-model-v2', counts_supported='positive per-axis integers within cell budget',
                    max_cells=policy.MAX_CELLS, pool='per-axis average, kernel=min(2,count), ceil, exclude padding',
                    physical_cells='one shared spacing on X/Y/Z', coordinates='cell centres in metres; unit scale 1 m')


def spacing_contract(spacing):
    require(spacing.ndim == 2 and spacing.shape[1] == 3 and spacing.dtype == torch.float64
            and spacing.device.type == 'cpu', 'spacing N,3 f64 CPU')
    require(bool(torch.isfinite(spacing).all() and (spacing > 0).all()), 'positive finite spacing')
    require(bool((spacing == spacing[:, :1]).all()), 'cubic cells require equal spacing on all axes')


def field_shape_contract(field):
    require(field.ndim == 5 and field.shape[1] == 1 and 1 <= field.shape[0] <= 4
            and all(n >= 1 for n in field.shape[2:]) and math.prod(field.shape[2:]) <= policy.MAX_CELLS,
            'positive N,1,Z,Y,X grid within CPU cell budget')


class PressureCorrectionNet(frozen.PressureCorrectionNet):
    def forward(self, values):
        require(values.ndim == 5 and values.shape[1] == 10 and 1 <= values.shape[0] <= 4, 'input N,10,Z,Y,X; batch 1–4')
        require(values.dtype == torch.float32 and values.device.type == 'cpu', 'eager f32 CPU inputs')
        require(all(n >= 1 for n in values.shape[2:]) and math.prod(values.shape[2:]) <= policy.MAX_CELLS,
                'positive per-axis dimensions within CPU cell budget')
        require(bool(torch.isfinite(values).all()), 'finite model input')
        skips = []
        for encoder in self.encoders:
            values = encoder(values)
            skips.append(values)
            kernel = tuple(min(2, n) for n in values.shape[2:])
            values = F.avg_pool3d(values, kernel, stride=kernel, ceil_mode=True, count_include_pad=False)
        values = self.bottleneck(values)
        values = values + self.global_pool(values).expand_as(values)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            values = F.interpolate(values, size=skip.shape[2:], mode='nearest')
            values = decoder(torch.cat((values, skip), dim=1))
        return self.head(values)


def full_operator(pressure, spacing, rho):
    field_shape_contract(pressure)
    spacing_contract(spacing)
    return frozen.full_operator(pressure, spacing, rho)


def build_inputs(residual, base, spacing, rho, dt, normalization):
    field_shape_contract(base)
    spacing_contract(spacing)
    scales = frozen._scale_values(normalization)
    require(scales['hx'] == scales['hy'] == scales['hz'], 'cubic cells require one shared spacing normalization scale')
    # Retain the original residual/pressure/density/time preprocessing, replacing
    # only its unit-box coordinates with physical cell centres on the actual grid.
    result = frozen.build_inputs(residual, base, spacing, rho, dt, normalization)
    for channel, tensor_axis, physical_axis in ((2, 4, 0), (3, 3, 1), (4, 2, 2)):
        count = base.shape[tensor_axis]
        shape = [1] * 5; shape[tensor_axis] = count
        centres = (torch.arange(count, dtype=torch.float64) + .5).reshape(shape)
        physical = centres * spacing[:, physical_axis].reshape(-1, 1, 1, 1, 1)
        result[:, channel:channel+1] = physical.expand_as(base).to(torch.float32)
    require(bool(torch.isfinite(result).all()), 'finite physical coordinates')
    return result


def predict_pressure(network, sample, normalization):
    values = build_inputs(sample['native_input_residual'], sample['base'], sample['spacing'], sample['rho'], sample['dt'], normalization)
    active = (sample['native_input_residual'].flatten(1) != 0).any(dim=1)
    if not bool(active.any()):
        return sample['base'].clone()
    selected = torch.nonzero(active, as_tuple=False).flatten()
    raw = network(values.index_select(0, selected))
    require(raw.shape == (selected.numel(), 1, *values.shape[2:]) and bool(torch.isfinite(raw).all()), 'same-grid finite per-cell output')
    correction = frozen.gauge_correction(raw, frozen._scale_values(normalization)['pressure'])
    return sample['base'].clone().index_copy(0, selected, sample['base'].index_select(0, selected) + correction)


def pressure_loss(prediction, sample, normalization):
    field_shape_contract(prediction)
    spacing_contract(sample['spacing'])
    return frozen.pressure_loss(prediction, sample['reference'], sample['rhs'], sample['spacing'], sample['rho'], normalization)
