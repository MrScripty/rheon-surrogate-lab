"""Seven spatial input channels; cubic cell size is scalar physical metadata."""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

import policy

frozen = policy.frozen_model()
require = frozen.require
configure_cpu = frozen.configure_cpu
parameter_count = frozen.parameter_count
CHANNELS = ['native_residual', 'previous_pressure', 'x', 'y', 'z', 'rho', 'dt']
RETAINED_CHANNELS = [0, 1, 2, 3, 4, 8, 9]
ARCHITECTURE = copy.deepcopy(frozen.ARCHITECTURE)
ARCHITECTURE.update(schema='rheon-cubic-groupnorm-model-v3', input_channels=7,
                    counts_supported='positive per-axis integers within cell budget', max_cells=policy.MAX_CELLS,
                    pool='per-axis average, kernel=min(2,count), ceil, exclude padding',
                    physical_cells='one scalar h per sample; no cell-size network channels',
                    coordinates='cell centres in metres; unit scale 1 m',
                    initialization='seeded original blocks; first convolution retains columns 0,1,2,3,4,8,9')


def cell_size_contract(cell_size):
    require(cell_size.ndim == 1 and cell_size.dtype == torch.float64 and cell_size.device.type == 'cpu',
            'cell_size N f64 CPU scalar per sample')
    require(bool(torch.isfinite(cell_size).all() and (cell_size > 0).all()), 'positive finite cubic cell size')


def field_shape_contract(field):
    require(field.ndim == 5 and field.shape[1] == 1 and 1 <= field.shape[0] <= 4
            and all(n >= 1 for n in field.shape[2:]) and math.prod(field.shape[2:]) <= policy.MAX_CELLS,
            'positive N,1,Z,Y,X grid within CPU cell budget')


def scales(normalization):
    require(normalization.get('schema') == 'rheon-cubic-pressure-normalization-v2'
            and normalization.get('channels') == CHANNELS, 'seven-channel normalization schema')
    result = normalization['scales']
    require(set(result) == {'residual', 'pressure', 'rhs', 'rho', 'dt'}, 'normalization scale inventory; no voxel-size scales')
    require(all(type(spec['value']) in (int, float) and math.isfinite(spec['value']) and spec['value'] > 0
                for spec in result.values()), 'positive finite scales')
    return {name: spec['value'] for name, spec in result.items()}


class PressureCorrectionNet(frozen.PressureCorrectionNet):
    def __init__(self):
        super().__init__()
        original = self.encoders[0].layers[0]
        # Remove spacing columns from the seeded initialization, not trained weights.
        # Preserve every surviving coefficient and the rest of the architecture.
        with torch.random.fork_rng(devices=[]):
            first = nn.Conv3d(7, original.out_channels, original.kernel_size, padding=original.padding,
                              padding_mode=original.padding_mode)
        with torch.no_grad():
            first.weight.copy_(original.weight[:, RETAINED_CHANNELS])
            first.bias.copy_(original.bias)
        self.encoders[0].layers[0] = first
        require(parameter_count(self) == 371729, 'seven-channel parameter inventory')

    def forward(self, values):
        require(values.ndim == 5 and values.shape[1] == 7 and 1 <= values.shape[0] <= 4, 'input N,7,Z,Y,X; batch 1–4')
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


def full_operator(pressure, cell_size, rho):
    field_shape_contract(pressure)
    cell_size_contract(cell_size)
    require(cell_size.shape == (pressure.shape[0],), 'one cell size per sample')
    # The unchanged FV implementation takes three scalar edge lengths, not maps.
    return frozen.full_operator(pressure, cell_size[:, None].expand(-1, 3), rho)


def build_inputs(residual, base, cell_size, rho, dt, normalization):
    field_shape_contract(base)
    cell_size_contract(cell_size)
    require(cell_size.shape == (base.shape[0],), 'one cell size per sample')
    require(base.dtype == residual.dtype == torch.float64 and base.device.type == residual.device.type == 'cpu'
            and residual.shape == base.shape, 'base and residual N,1,Z,Y,X f64 CPU')
    require(bool(torch.isfinite(base).all() and torch.isfinite(residual).all())
            and bool((base[:, :, 0, 0, 0] == 0).all()), 'finite gauge-fixed pressure and residual')
    for scalar in (rho, dt):
        require(scalar.shape == cell_size.shape and scalar.dtype == torch.float64 and scalar.device.type == 'cpu'
                and bool(torch.isfinite(scalar).all() and (scalar > 0).all()), 'positive finite rho/dt scalar per sample')
    scale = scales(normalization)
    channels = [residual / scale['residual'], base / scale['pressure']]
    for tensor_axis in (4, 3, 2):
        shape = [1]*5; shape[tensor_axis] = base.shape[tensor_axis]
        centres = (torch.arange(base.shape[tensor_axis], dtype=torch.float64)+.5).reshape(shape)
        channels.append((centres*cell_size.reshape(-1,1,1,1,1)).expand_as(base))
    for scalar, name in ((rho, 'rho'), (dt, 'dt')):
        channels.append((scalar/scale[name]).reshape(-1,1,1,1,1).expand_as(base))
    result = torch.cat(channels, dim=1).to(torch.float32)
    require(result.shape[1] == 7 and bool(torch.isfinite(result).all()), 'seven finite channels; no spacing maps')
    return result


def predict_pressure(network, sample, normalization):
    values = build_inputs(sample['native_input_residual'], sample['base'], sample['cell_size_m'], sample['rho'], sample['dt'], normalization)
    active = (sample['native_input_residual'].flatten(1) != 0).any(dim=1)
    if not bool(active.any()):
        return sample['base'].clone()
    selected = torch.nonzero(active, as_tuple=False).flatten()
    raw = network(values.index_select(0, selected))
    require(raw.shape == (selected.numel(), 1, *values.shape[2:]) and bool(torch.isfinite(raw).all()), 'same-grid finite per-cell output')
    correction = frozen.gauge_correction(raw, scales(normalization)['pressure'])
    return sample['base'].clone().index_copy(0, selected, sample['base'].index_select(0, selected) + correction)


def pressure_loss(prediction, sample, normalization):
    field_shape_contract(prediction)
    reference, rhs = sample['reference'], sample['rhs']
    require(reference.shape == rhs.shape == prediction.shape and reference.dtype == rhs.dtype == torch.float64
            and reference.device.type == rhs.device.type == 'cpu'
            and bool(torch.isfinite(reference).all() and torch.isfinite(rhs).all()), 'finite f64 labels/RHS')
    scale = scales(normalization); cells = prediction[0].numel()
    denominator_p = reference.flatten(1).square().sum(dim=1).clamp_min((1e-6*math.sqrt(cells)*scale['pressure'])**2)
    denominator_b = rhs.flatten(1).square().sum(dim=1).clamp_min((1e-6*math.sqrt(cells)*scale['rhs'])**2)
    residual = full_operator(prediction, sample['cell_size_m'], sample['rho']) - rhs
    return .5*(prediction-reference).flatten(1).square().sum(dim=1)/denominator_p + .5*residual.flatten(1).square().sum(dim=1)/denominator_b
