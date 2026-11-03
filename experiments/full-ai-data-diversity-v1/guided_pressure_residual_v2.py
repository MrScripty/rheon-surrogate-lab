"""Independent pressure and solenoidal velocity corrections to a frozen guide.

The pressure head predicts a dimensionless learned-RHS correction, scaled only
by the restored learned action on the predicted guide pressure. Velocity
projection pressure is internal and never added to the reported pressure.
Future native fields/RHS are loss targets only. All original loss floors stay
unchanged. Removing overwritten guide readouts has separately proved byte and
encoder-context parity across all 240 selected training windows.
"""
import torch

from coupled_macrostep_v1 import _AnchoredAdd, _RestoredPressure
from guided_shared_residual_v3 import guide_correction_scales
from minimal_fine_readout_v1 import MinimalFineReadoutGenerator
from shared_temporal_generator_v1 import SharedTemporalGenerator as Entry
from shared_temporal_network_v1 import spatial_context, control_context, interior_conditioning
from shared_pressure_network_v2 import SharedPressureResidual
from temporal_group_projector_v1 import TemporalGroupProjector
from shared_stationary_prefix_v1 import stationary_prefix_outputs
from recurrent_sampler_native import dimensions


class GuidedPressureResidual:
    predict = Entry.predict
    emit = TemporalGroupProjector.emit

    def __init__(self, *args, hidden=4, **kwargs):
        self.guide = MinimalFineReadoutGenerator(*args, hidden=4,
            fine_mode='preoutput_fine_fullstate_moment_tracer', **kwargs)
        self.guide_certificate = self.guide.certify_zero_heads()
        self.model, self.linear = self.guide.model, self.guide.linear
        self.network = SharedPressureResidual(hidden=hidden).eval().requires_grad_(False)
        self.asset_identity = dict(self.guide.asset_identity,
            model_family='guided_pressure_residual_v2', pressure_head_arithmetic_dtype='float64', pressure_head_parameter_dtype='float32', independent_pressure_RHS_head=True,
            velocity_projection_pressure_reported=False, guide_autograd=False,
            residual_hidden_width=hidden, frozen_guide_hidden_width=4,
            new_neural_parameters=sum(p.numel() for p in self.network.parameters()),
            learned_shared_trunks_per_block=1, pressure_response_dt_s=1 / 240,
            new_correction_pressure_responses_per_block='2*K',
            pressure_RHS_units='RMS restored learned flux(momentum(predicted guide pressure))',
            velocity_tracer_units='Unchanged guide-derived RMS/displacement units from guided_shared_residual_v3',
            true_future_fields_in_inputs=False, optimizer_updates=0,
            zero_head_output_parity_unqualified=True,
            accuracy_and_complete_GPU_cost_unqualified=True)

    def packets(self, *args):
        return self.guide.packets(*args)

    def state_dict(self):
        result = {'guide.' + n: v for n, v in self.guide.state_dict().items()}
        result.update({'residual.' + n: v for n, v in self.network.state_dict().items()})
        return result

    @torch.no_grad()
    def pressure_scale(self, guide, phase):
        core, shape, counts = self.model, phase['shape'], phase['counts']
        pressure = guide['pressure'].reshape(shape)
        gradient = core.gather_ops.three(core, [pressure], None, counts,
            'momentum', 1., multiply_only=True, as_double=True)
        action = self.linear(gradient, shape, 'flux', 0, sign=-1.)
        return action.square().mean().sqrt()

    def predict_packet(self, fields, scene, output_rate, k, packet, events, event_mask):
        with torch.no_grad():
            guides = self.guide.predict_packet(fields, scene, output_rate, k, packet, events, event_mask)
            self.last_guide_outputs = guides
            phase = self.guide._guide_phase
            sizes, endpoint = dimensions(phase['counts']), guides[-1]
            encoding_phase = dict(phase,
                base_pressure=endpoint['pressure'].reshape(1, 1, *phase['shape']),
                base_velocity=[endpoint[n].reshape(s).double()
                    for n, s in zip(('vx', 'vy', 'vz'), sizes[:3])],
                base_tracer=endpoint['tracer'].reshape(phase['shape']))
            spatial = spatial_context(encoding_phase)
            if spatial.shape[1] != 26:
                raise ValueError('Frozen spatial context schema changed')
            spatial[:, 22] = phase['base_tracer'].float()
            controls, event_features, mask, active = control_context(packet, events, event_mask, phase)
        hidden = self.network.encode(spatial, controls, event_features, mask)
        outputs = []
        for lead, guide in enumerate(guides, 1):
            lead_dt = lead / output_rate
            if lead == k:
                delta = self.network.endpoint(hidden, active)
            else:
                conditioning = interior_conditioning(fields, guide, phase, lead_dt, lead / k)
                delta = self.network.interior(hidden, conditioning, active)
            base = dict(counts=phase['counts'], shape=phase['shape'], h=phase['h'], rho=phase['rho'],
                dt=phase['reference_dt_s'], base_pressure=guide['pressure'].reshape(1, 1, *phase['shape']),
                base_velocity=[guide[n].reshape(s).double() for n, s in zip(('vx', 'vy', 'vz'), sizes[:3])],
                base_tracer=guide['tracer'].reshape(phase['shape']).double())
            velocity_scale, tracer_scale = guide_correction_scales(fields, guide, phase, lead_dt)
            output = self.emit(base, [v.double() * velocity_scale for v in delta[:3]],
                delta[3].double() * tracer_scale)
            response = _RestoredPressure.apply(
                (delta[4].double() * self.pressure_scale(guide, phase)).reshape(1, 1, *phase['shape']),
                self.model, phase['counts'])
            output['pressure'] = _AnchoredAdd.apply(base['base_pressure'], response).contiguous().flatten()
            outputs.append(output)
        return stationary_prefix_outputs(fields, packet, output_rate, k, outputs)

    def close(self):
        self.guide.close()
