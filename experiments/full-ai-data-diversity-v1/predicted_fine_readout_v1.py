"""One fine readout from each teacher-free predicted pre-output state.

This zero-fit experiment isolates nonlinear pressure missing from the force
timing guide. One shared whole-block forecast supplies every approximate
pre-output state, with known prefix impulse and previous-force pressure. Each
readout then advances one 1/240s interval using restored learned operators and
the original N1 pressure-history branch. All added work must be timed.
"""
import torch

from shared_temporal_control_guides_v2 import ControlGuideGenerator as Parent
from temporal_group_sum_projector_v1 import phase_packet
from recurrent_sampler_native import dimensions
from shared_stationary_prefix_v1 import stationary_prefix_outputs


class FineReadoutGenerator(Parent):
    def __init__(self, *args, fine_mode, **kwargs):
        if fine_mode not in ('prefix_guide', 'preoutput_fine_pressure', 'preoutput_fine_fullstate'):
            raise ValueError('Registered predicted fine readout required')
        super().__init__(*args, guide_mode='last_force_pressure_prefix_impulse', **kwargs)
        self.fine_mode = fine_mode
        self.asset_identity.update(predicted_fine_readout_mode=fine_mode,
            native_future_state_or_RHS_inputs=False,
            original_N1_history_composition=True,
            pre_output_time='start + j/output_rate - 1/240',
            additional_actual_learned_advances_per_block='K, unless prefix_guide control',
            parameter_updates=0, accuracy_unqualified=True)

    def predict_packet(self, fields, scene, output_rate, k, packet, events, event_mask):
        outputs = Parent.predict_packet(self, fields, scene, output_rate, k, packet, events, event_mask)
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


class FineReadoutGraph(FineReadoutGenerator):
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
                FineReadoutGenerator.predict_packet(self, state, scene, output_rate, k, controls, edge_values, mask)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = FineReadoutGenerator.predict_packet(self, state, scene, output_rate, k, controls, edge_values, mask)
        torch.cuda.synchronize()
        self.graphs[key] = graph, state, controls, edge_values, mask, outputs

    def close(self):
        self.release_graphs()
        super().close()
