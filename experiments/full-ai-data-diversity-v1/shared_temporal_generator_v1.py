"""One endpoint advance and one shared trunk for K whole physical outputs.

Interiors use a small decoder conditioned on the predicted endpoint and shared
hidden field, followed by the same learned global correction projection.
No true future state, native pressure solve, or K full advances enter a pass.
This is an untrained research prototype, not a qualified accuracy successor.
"""
import math

import torch

from temporal_group_projector_v2 import TemporalGroupProjector
from inference_restored_linear_v1 import InferenceRestoredLinear
from temporal_control_packets_v1 import control_packet
from pulse_event_controls_v1 import event_packet
from shared_temporal_network_v1 import (SharedTemporalResidual, spatial_context,
    control_context, interior_conditioning)
from recurrent_sampler_native import dimensions
from cuda_native_transfer_v1 import FIELDS


class SharedTemporalGenerator(TemporalGroupProjector):
    def __init__(self, *args, hidden=8):
        super().__init__(*args)
        self.linear = InferenceRestoredLinear(self.model)
        self.network = SharedTemporalResidual(hidden=hidden).eval().requires_grad_(False)
        self.asset_identity.update(shared_temporal_prototype=True, spatial_hidden_width=hidden,
            new_neural_parameters=sum(p.numel() for p in self.network.parameters()),
            residual_optimizer_updates=0, true_future_fields_in_inputs=False,
            endpoint_transport_passes_per_block=1, endpoint_pressure_projections=2,
            interior_pressure_projections_per_output=1,
            shared_spatial_encoder_calls_per_block=1,
            force_encoder_scope='one scalar24channel path, not a path per voxel',
            exact_prescribed_event_offsets=True, maximum_event_edges_per_block=32,
            reference='selected240leftsampled numerical labels; no event-exact truth claim',
            original_fine_baseline_retained_elsewhere=True,
            accuracy_and_complete_GPU_cost_unqualified=True)

    def state_dict(self):
        result = dict(super().state_dict())
        result.update({'shared_temporal.' + n: v for n, v in self.network.state_dict().items()})
        return result

    def packets(self, scene, start_fine, output_rate, k):
        width = k * (240 // output_rate)
        known = torch.from_numpy(control_packet(scene, start_fine, width)).to('cuda')
        events = event_packet(scene['forcing_program'], start_fine / 240,
            k / output_rate, [j / output_rate for j in range(1, k + 1)])
        if len(events['events']) > 32:
            raise ValueError('Prototype event capacity exceeded; never truncate event input')
        values = torch.zeros((32, 5), dtype=torch.float64)
        mask = torch.zeros(32, dtype=torch.float32)
        if events['events']:
            values[:len(events['events'])] = torch.tensor(events['events'], dtype=torch.float64)
            mask[:len(events['events'])] = 1.
        return known, values.to('cuda'), mask.to('cuda')

    @torch.inference_mode()
    def predict(self, fields, scene, start_fine, output_rate, k):
        if (scene['reference_substeps'] != 4 or output_rate not in (30, 60, 120)
                or type(k) is not int or k not in (1, 2, 3, 4) or scene.get('sgs', False)):
            raise ValueError('Selected240reference and explicit physical cadence/K required')
        counts = tuple(scene['counts'])
        if set(fields) != set(FIELDS):
            raise ValueError('Complete current native CUDA state required')
        for name, size in [('pressure', tuple(reversed(counts))), *zip(FIELDS[1:], dimensions(counts))]:
            value = fields[name]
            if (value.device.type != 'cuda' or not value.is_contiguous()
                    or value.dtype != (torch.float64 if name == 'pressure' else torch.float32)
                    or value.numel() != math.prod(size)):
                raise ValueError('Exact contiguous native CUDA field layout required')
        return self.predict_packet(fields, scene, output_rate, k, *self.packets(scene, start_fine, output_rate, k))

    def predict_packet(self, fields, scene, output_rate, k, packet, events, event_mask):
        width = k * (240 // output_rate)
        if packet.shape != (width, 18) or packet.device.type != 'cuda':
            raise ValueError('Complete whole-block prescribed force path required')
        phase = self.phase_packet(fields, scene, width, packet)
        phase['acceleration_m_per_s2'] = scene['acceleration']
        spatial = spatial_context(phase)
        controls, event_features, mask, active = control_context(packet, events, event_mask, phase)
        hidden = self.network.encode(spatial, controls, event_features, mask)
        endpoint_delta = self.network.endpoint(hidden, active)
        endpoint = self.emit(phase,
            [v.double() * (phase['h'] / phase['dt']) for v in endpoint_delta[:3]],
            endpoint_delta[3].double())
        outputs = []
        for lead in range(1, k):
            alpha, dt = lead / k, lead / output_rate
            base = dict(counts=phase['counts'], shape=phase['shape'], h=phase['h'],
                rho=phase['rho'], dt=dt,
                base_pressure=(fields['pressure'].double() + alpha * (endpoint['pressure'] - fields['pressure'].double())).reshape_as(phase['base_pressure']),
                base_velocity=[(fields[name].double() + alpha * (endpoint[name].double() - fields[name].double())).reshape(size)
                    for name, size in zip(('vx', 'vy', 'vz'), dimensions(phase['counts'])[:3])],
                base_tracer=(fields['tracer'].double() + alpha * (endpoint['tracer'].double() - fields['tracer'].double())).reshape(phase['shape']))
            condition = interior_conditioning(fields, endpoint, phase, dt, alpha)
            delta = self.network.interior(hidden, condition, active)
            outputs.append(self.emit(base, [v.double() * (phase['h'] / dt) for v in delta[:3]], delta[3].double()))
        outputs.append(endpoint)
        return outputs
