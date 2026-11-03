"""Explicit-time CUDA baseline and untrained grouped-step controls.

Uses restored learned spatial operators and absolute learned pressure with the
bound F64 sampler. K outputs are sequential here; this is a baseline for shared
temporal models, not a claim of joint generation or reduced work by packing.
"""
import math

import torch

from bound_sampler_pressure_graph_v1 import BoundSamplerPressureGraph
from recurrent_sampler_native import dimensions
from temporal_control_packets_v1 import control_packet

FIELDS = ('pressure', 'vx', 'vy', 'vz', 'tracer')


class TemporalFrozenBackbone:
    def __init__(self, pair, shared_pair, build, control_build):
        self.backend = BoundSamplerPressureGraph(pair, shared_pair, build, control_build)
        self.model, self.controls = self.backend.model, self.backend.controls
        self.asset_identity = dict(self.backend.asset_identity,
            temporal_contract='Explicit outputdt and K; each output advances a whole physical interval.',
            reference_force_clock='Explicit240or480/s', sequential_baseline=True,
            output_count_is_not_internal_advance_count=True,
            learned_parameter_updates=0, cpu_model_forward=False,
            timestep_and_accuracy_require_new_qualification=True)

    def state_dict(self):
        return self.backend.state_dict()

    @torch.inference_mode()
    def predict(self, fields, scene, start_fine, output_rate, k, groups=None):
        rate = 60 * scene['reference_substeps']
        if (rate not in (240, 480) or output_rate not in (30, 60, 120)
                or type(k) is not int or k not in (1, 2, 3, 4)
                or scene.get('sgs', False)):
            raise ValueError('Explicit registered physical cadence andKrequired')
        stride = rate // output_rate
        groups = (1,) * stride if groups is None else tuple(groups)
        if (not groups or any(type(n) is not int or not 1 <= n <= stride for n in groups)
                or sum(groups) != stride):
            raise ValueError('Internal groups must partition one complete output interval')
        counts = tuple(scene['counts'])
        shape, sizes = tuple(reversed(counts)), dimensions(counts)
        if set(fields) != set(FIELDS):
            raise ValueError('Complete current native fields required')
        for name, size in [('pressure', shape), *zip(FIELDS[1:], sizes)]:
            value = fields[name]
            if (value.device.type != 'cuda' or not value.is_contiguous()
                    or value.dtype != (torch.float64 if name == 'pressure' else torch.float32)
                    or value.numel() != math.prod(size)):
                raise ValueError('Exact CUDA native field layout required')
        packet = torch.from_numpy(control_packet(scene, start_fine, k * stride)).to('cuda')
        external = self.controls.generate_from_packet(counts, packet)
        core = self.model
        core.prepare_geometry(counts)
        core._nested_enabled = True
        current = [fields[name].reshape(size) for name, size in zip(FIELDS[1:], sizes)]
        h, rho = 1. / scene['cells_per_metre'], 1000.
        offset, outputs = 0, []
        for _ in range(k):
            for width in groups:
                dt = width / rate
                core.configure_transport(h, dt)
                addition = external[offset]
                for j in range(1, width):
                    addition = [a + b for a, b in zip(addition, external[offset + j])]
                offset += width
                advected = core.transport(current, counts, dt / h)
                entering = core.affine_ops.three(advected[:3], addition, counts)
                provisional = core.gather_ops.three(core, entering, entering, counts,
                    'viscosity', (.001 / rho * dt / h ** 2))
                rhs = core.gather_ops.ordered_rhs(core, provisional, shape, 'flux', 0, h * h, dt)
                pressure = core.pressure((rhs * (rho / h)).reshape(1, 1, *shape), counts)
                velocity = core.gather_ops.three(core, [pressure[0, 0]], provisional,
                    counts, 'momentum', dt * (1 / (rho * h)))
                current = [*velocity, advected[3]]
            outputs.append(dict(pressure=pressure.contiguous().flatten(),
                **{name: value.contiguous().flatten() for name, value in zip(FIELDS[1:], current)}))
        return outputs

    def close(self):
        self.backend.close()
