"""One shared learned residual trunk around frozen time-aware field guides.

Guide generation consumes current fields and known controls only. Future native
fields remain external loss targets. Corrections use the restored learned
projection with the final fine pressure interval. Guide field magnitudes set
the correction units; lead time sets the tracer displacement scale. The guide
graph is deliberately not differentiated.
"""
import torch

from folded_fine_readout_v1 import FoldedFineReadoutGenerator
from shared_temporal_generator_v1 import SharedTemporalGenerator as Entry
from shared_temporal_network_v1 import (SharedTemporalResidual, spatial_context,
    control_context, interior_conditioning)
from temporal_group_projector_v1 import TemporalGroupProjector
from shared_stationary_prefix_v1 import stationary_prefix_outputs
from recurrent_sampler_native import dimensions


@torch.no_grad()
def guide_correction_scales(current, predicted, phase, lead_dt):
    velocity, tracer = [], []
    for state in (current, predicted):
        carriers = [state[name].double() for name in ('vx', 'vy', 'vz')]
        velocity.append((sum(v.square().sum() for v in carriers) / sum(v.numel() for v in carriers)).sqrt())
        pressure = state['pressure'].double()
        pressure_rms = (pressure - pressure.mean()).square().mean().sqrt()
        velocity.append(pressure_rms * (phase['reference_dt_s'] / (phase['rho'] * phase['h'])))
        tracer.append(state['tracer'].double().square().mean().sqrt())
    velocity_scale = torch.stack(velocity).amax()
    tracer_scale = torch.stack(tracer).amax() * (velocity_scale * (lead_dt / phase['h'])).clamp_max(1.)
    return velocity_scale, tracer_scale


class GuidedSharedResidual:

    predict = Entry.predict
    emit = TemporalGroupProjector.emit

    def __init__(self, *args, hidden=4, **kwargs):
        # The frozen width4 forecast is the same for both learned widths. Its
        # output heads are certified zero; only the new network below is fitted.
        self.guide = FoldedFineReadoutGenerator(*args, hidden=4,
            fine_mode='preoutput_fine_fullstate_moment_tracer', **kwargs)
        self.guide_certificate = self.guide.certify_zero_heads()
        self.model, self.linear = self.guide.model, self.guide.linear
        self.network = SharedTemporalResidual(hidden=hidden).eval().requires_grad_(False)
        self.asset_identity = dict(self.guide.asset_identity,
            guided_shared_residual=True, new_neural_parameters=sum(p.numel() for p in self.network.parameters()),
            residual_hidden_width=hidden, frozen_guide_hidden_width=4,
            guide_autograd=False, learned_shared_trunks_per_block=1,
            new_correction_pressure_responses_per_block='K',
            pressure_response_dt_s=1 / 240,
            velocity_correction_units='Maximum RMS current/predicted velocity and fine pressure-impulse equivalent; guide-derived F64 scalar',
            tracer_correction_units='Maximum RMS current/predicted tracer times min(1, velocity_scale*lead_dt/h)',
            spatial_context='Current and predictedfineendpointPVT; channel22 is coarseforecasttracer in place of redundant width/32.',
            true_future_fields_in_inputs=False, optimizer_updates=0,
            zero_head_output_parity_unqualified=True,
            accuracy_and_complete_GPU_cost_unqualified=True)

    def packets(self, *args):
        return self.guide.packets(*args)

    def state_dict(self):
        values = {'guide.' + n: v for n, v in self.guide.state_dict().items()}
        values.update({'residual.' + n: v for n, v in self.network.state_dict().items()})
        return values

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
                raise ValueError('Frozen shared spatial context schema changed')
            # Wholeblock duration already carries width through240*dt. Retain
            # the coarse tracer as an additional teacher-free fallback feature.
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
            outputs.append(self.emit(base, [v.double() * velocity_scale for v in delta[:3]], delta[3].double() * tracer_scale))
        return stationary_prefix_outputs(fields, packet, output_rate, k, outputs)

    def close(self):
        self.guide.close()


class GuidedSharedResidualGraph(GuidedSharedResidual):
    from shared_temporal_graph_v1 import SharedTemporalGraph as _Graph
    predict_packet = _Graph.predict_packet
    release_graphs = _Graph.release_graphs

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graphs = {}

    @torch.inference_mode()
    def prepare_graph(self, key, fields, scene, output_rate, k, packet, events, event_mask):
        if any(v.device.type != 'cuda' for v in [*fields.values(), packet, events, event_mask]):
            raise ValueError('CUDA-only graph inputs required')
        state = {n: v.clone() for n, v in fields.items()}
        controls, edge_values, mask = packet.clone(), events.clone(), event_mask.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                GuidedSharedResidual.predict_packet(self, state, scene, output_rate, k, controls, edge_values, mask)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = GuidedSharedResidual.predict_packet(self, state, scene, output_rate, k, controls, edge_values, mask)
        torch.cuda.synchronize()
        self.graphs[key] = graph, state, controls, edge_values, mask, outputs

    def close(self):
        self.release_graphs()
        super().close()
