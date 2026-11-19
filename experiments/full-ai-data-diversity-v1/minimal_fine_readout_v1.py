"""Full-state-only guide: remove controls whose outputs are overwritten.

The final fine-state readout uses only the coarse endpoint velocity/tracer and
coarse phase, then replaces every output. Earlier per-output pressure timing
and interior prefix-velocity readouts are unused in this particular mode.
Removing them saves2*K-1 learned pressure calls. History/actual fine advances
and moment transport remain unchanged. Byte and context parity is unproven.
"""
import torch
from folded_fine_readout_v1 import FoldedFineReadoutGenerator as Parent, folded_coarse_packet
from temporal_group_sum_projector_v1 import phase_packet
from recurrent_sampler_native import dimensions
from shared_stationary_prefix_v1 import stationary_prefix_outputs


def minimal_fine_packet(self, fields, scene, output_rate, k, packet, events, event_mask):
    outputs = folded_coarse_packet(self, fields, scene, output_rate, k, packet, events, event_mask)
    if self.fine_mode == 'prefix_guide':
        return outputs
    phase, endpoint = self._guide_phase, outputs[-1]
    core, counts = self.model, phase['counts']
    total, width = phase['known_force_sum'], packet.shape[0]
    stride, sizes = 240 // output_rate, dimensions(counts)
    result = []
    for lead, original in enumerate(outputs, 1):
        end = lead * stride
        pre_end, alpha = end - 1, (end - 1) / width
        pre_dt = pre_end / 240
        prefix = self.control_sum.sum_from_packet(counts, packet[:pre_end].contiguous())
        delta = [(a - alpha * b).float().contiguous() for a, b in zip(prefix, total)]
        delta_phase = dict(phase, dt=pre_dt)
        pressure = self.pressure_response(delta, delta_phase)
        projected = core.gather_ops.three(core, [pressure[0, 0]], delta, counts,
            'momentum', pre_dt / (phase['rho'] * phase['h']), as_double=True)
        predicted = {}
        for name, size, addition in zip(('vx', 'vy', 'vz'), sizes[:3], projected):
            value = fields[name].reshape(size).double()
            predicted[name] = (value + alpha * (endpoint[name].reshape(size).double() - value)
                + addition).float().contiguous().flatten()
        tracer = fields['tracer'].double()
        predicted['tracer'] = (tracer + alpha * (endpoint['tracer'].double() - tracer)).float().contiguous()

        # Approximate pressure immediately before the final fine interval;
        # retain history composition, rather than silently dropping it.
        last = self.control_sum.sum_from_packet(counts, packet[pre_end - 1:pre_end].contiguous())
        delta = [(width * a - b).float().contiguous() for a, b in zip(last, total)]
        delta = core.gather_ops.three(core, delta, delta, counts, 'viscosity',
            .001 / phase['rho'] * phase['dt'] / phase['h'] ** 2)
        predicted['pressure'] = (phase['base_pressure'] + self.pressure_response(delta, phase)).contiguous().flatten()
        fine = phase_packet(self, predicted, scene, 1, packet[end - 1:end].contiguous())
        updated = dict(pressure=fine['base_pressure'].contiguous().flatten(),
            **{n: v.float().contiguous().flatten() for n, v in zip(
                ('vx', 'vy', 'vz', 'tracer'), [*fine['base_velocity'], fine['base_tracer']])})
        if self.fine_mode == 'preoutput_fine_pressure':
            updated = dict(original, pressure=updated['pressure'])
        result.append(updated)
    return stationary_prefix_outputs(fields, packet, output_rate, k, result)

def minimal_composite_packet(self, fields, scene, output_rate, k, packet, events, event_mask):
    outputs = minimal_fine_packet(self, fields, scene, output_rate, k, packet, events, event_mask)
    if not self.moment_tracer:
        return outputs
    phase = self._guide_phase
    core, counts = self.model, phase['counts']
    sizes = dimensions(counts)
    for lead, output in enumerate(outputs, 1):
        end, dt = lead * (240 // output_rate), lead / output_rate
        controls = packet[:end].clone()
        weights = torch.arange(end - 1, -1, -1, device='cuda', dtype=torch.float64) / end
        controls[:, 14] *= weights
        delta = [v.float().contiguous() for v in self.control_sum.sum_from_packet(counts, controls)]
        response = self.pressure_response(delta, dict(phase, dt=dt))
        projected = core.gather_ops.three(core, [response[0, 0]], delta, counts,
            'momentum', dt / (phase['rho'] * phase['h']), as_double=True)
        velocity = [(fields[n].reshape(s).double() + dv).float().contiguous()
            for n, s, dv in zip(('vx', 'vy', 'vz'), sizes[:3], projected)]
        current = [*velocity, fields['tracer'].reshape(sizes[3])]
        core.configure_transport(phase['h'], dt)
        output['tracer'] = core.transport(current, counts, dt / phase['h'])[3].contiguous().flatten()
    return stationary_prefix_outputs(fields, packet, output_rate, k, outputs)

class MinimalFineReadoutGenerator(Parent):
    predict_packet = minimal_composite_packet

    def __init__(self, *args, **kwargs):
        if kwargs.get('fine_mode') != 'preoutput_fine_fullstate_moment_tracer':
            raise ValueError('Dead-control removal applies only to full-state-plus-moment mode')
        super().__init__(*args, **kwargs)
        self.asset_identity.update(full_state_only_dead_control_removal=True,
            additional_unused_learned_pressure_calls_removed='2*K-1',
            exact_byte_and_coarse_context_parity_unqualified=True,
            original_N1_history_and_actual_fine_advances_unchanged=True)


class MinimalFineReadoutGraph(MinimalFineReadoutGenerator):
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
                MinimalFineReadoutGenerator.predict_packet(self, state, scene, output_rate, k, controls, edge_values, mask)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = MinimalFineReadoutGenerator.predict_packet(self, state, scene, output_rate, k, controls, edge_values, mask)
        torch.cuda.synchronize()
        self.graphs[key] = graph, state, controls, edge_values, mask, outputs

    def close(self):
        self.release_graphs()
        super().close()
