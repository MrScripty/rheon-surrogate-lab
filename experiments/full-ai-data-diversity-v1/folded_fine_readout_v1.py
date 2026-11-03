"""Fold only certified zero correction heads in the frozen shared forecast.

All control, fine-readout, moment and stationary-prefix arithmetic below is
copied unchanged from the bound candidate. Only the zero shared CNN and zero
emit responses are removed. This is not qualified until complete byte parity
is measured. Changing forecast correction parameters requires recertification
and new graphs. No trained nonzero correction head may use this folding.
"""
import torch
from predicted_fine_readout_v2 import FineReadoutGenerator as Parent
from temporal_group_sum_projector_v1 import phase_packet
from recurrent_sampler_native import dimensions
from shared_stationary_prefix_v1 import stationary_prefix_outputs
from coupled_macrostep_v1 import wall


def folded_coarse_packet(self, fields, scene, output_rate, k, packet, events, event_mask):
    if not self.zero_heads_certified:
        raise ValueError('Exact zero forecast correction heads must be certified before graph capture')
    width = k * (240 // output_rate)
    if packet.shape != (width, 18) or packet.device.type != 'cuda':
        raise ValueError('Complete whole-block prescribed force path required')
    phase = self.phase_packet(fields, scene, width, packet)
    phase['acceleration_m_per_s2'] = scene['acceleration']
    endpoint = dict(pressure=phase['base_pressure'].contiguous().flatten(),
        **{name: wall(value.double(), c).float().contiguous().flatten()
           for c, (name, value) in enumerate(zip(('vx','vy','vz'),phase['base_velocity']))},
        tracer=phase['base_tracer'].double().float().contiguous().flatten())
    outputs = []
    for lead in range(1, k):
        alpha = lead / k
        pressure = fields['pressure'].double() + alpha * (endpoint['pressure'] - fields['pressure'].double())
        velocity = [wall((fields[name].double() + alpha * (endpoint[name].double() - fields[name].double())).reshape(size), c)
            for c, (name, size) in enumerate(zip(('vx','vy','vz'), dimensions(phase['counts'])[:3]))]
        tracer = fields['tracer'].double() + alpha * (endpoint['tracer'].double() - fields['tracer'].double())
        outputs.append(dict(pressure=pressure.contiguous().flatten(),
            **{name: value.float().contiguous().flatten() for name, value in zip(('vx','vy','vz'), velocity)},
            tracer=tracer.float().contiguous().flatten()))
    outputs.append(endpoint)
    return stationary_prefix_outputs(fields, packet, output_rate, k, outputs)


def folded_control_packet(self, fields, scene, output_rate, k, packet, events, event_mask):
    outputs = folded_coarse_packet(self, fields, scene, output_rate, k, packet, events, event_mask)
    if self.guide_mode == 'original':
        return outputs
    phase = self._guide_phase
    total = phase['known_force_sum']
    stride, width = 240 // output_rate, packet.shape[0]
    core = self.model
    for lead, output in enumerate(outputs, 1):
        end, alpha = lead * stride, lead / k
        last = self.control_sum.sum_from_packet(phase['counts'], packet[end - 1:end].contiguous())
        delta = [(width * a - b).float().contiguous() for a, b in zip(last, total)]
        # Match the endpoint phase's linear viscosity on its force addition.
        delta = core.gather_ops.three(core, delta, delta, phase['counts'],
            'viscosity', .001 / phase['rho'] * phase['dt'] / phase['h'] ** 2)
        response = self.pressure_response(delta, phase)
        output['pressure'] = (phase['base_pressure'] + response).contiguous().flatten()
        if self.guide_mode == 'last_force_pressure_prefix_impulse' and lead < k:
            prefix = self.control_sum.sum_from_packet(phase['counts'], packet[:end].contiguous())
            impulse_delta = [(a - alpha * b).float().contiguous() for a, b in zip(prefix, total)]
            head_phase = dict(phase, dt=lead / output_rate)
            pressure = self.pressure_response(impulse_delta, head_phase)
            velocity = core.gather_ops.three(core, [pressure[0, 0]], impulse_delta,
                phase['counts'], 'momentum', head_phase['dt'] / (phase['rho'] * phase['h']),
                as_double=True)
            for name, size, correction in zip(('vx', 'vy', 'vz'), dimensions(phase['counts'])[:3], velocity):
                output[name] = (output[name].reshape(size).double() + correction).float().contiguous().flatten()
    return stationary_prefix_outputs(fields, packet, output_rate, k, outputs)

def folded_fine_packet(self, fields, scene, output_rate, k, packet, events, event_mask):
    outputs = folded_control_packet(self, fields, scene, output_rate, k, packet, events, event_mask)
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

def folded_composite_packet(self, fields, scene, output_rate, k, packet, events, event_mask):
    outputs = folded_fine_packet(self, fields, scene, output_rate, k, packet, events, event_mask)
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

class FoldedFineReadoutGenerator(Parent):
    predict_packet = folded_composite_packet

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zero_heads_certified = False
        self.asset_identity.update(zero_head_forecast_fold=True,
            shared_forecast_encoder_calls_per_block=0,
            zero_emit_pressure_projections_removed_per_block='K',
            actual_fine_advances_unchanged=True, pressure_history_unchanged=True,
            learned_parameters_unchanged=True, byte_parity_unqualified=True)

    def certify_zero_heads(self):
        names = ('endpoint_velocity.', 'endpoint_tracer.', 'interior_velocity.', 'interior_tracer.')
        selected = {n: p for n, p in self.network.state_dict().items() if n.startswith(names)}
        if len(selected) != 16 or any(torch.count_nonzero(p).item() for p in selected.values()):
            raise ValueError('Zero-head folding cannot execute a nonzero forecast correction')
        self.zero_heads_certified = True
        return dict(parameters_checked=sorted(selected), all_exact_zero=True,
            construction_only=True, learned_CPU_forward=False)


class FoldedFineReadoutGraph(FoldedFineReadoutGenerator):
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
                FoldedFineReadoutGenerator.predict_packet(self, state, scene, output_rate, k, controls, edge_values, mask)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = FoldedFineReadoutGenerator.predict_packet(self, state, scene, output_rate, k, controls, edge_values, mask)
        torch.cuda.synchronize()
        self.graphs[key] = graph, state, controls, edge_values, mask, outputs

    def close(self):
        self.release_graphs()
        super().close()
