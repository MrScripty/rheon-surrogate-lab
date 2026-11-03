"""Zero-fit ablation of pressure timing and prescribed-force prefix impulses.

The whole-block transport and N1 pressure-history branch remain unchanged.
Only restored learned maps supply pressure responses. Prescribed forces are
deployment inputs; no native future fluid field or RHS enters this generator.
This is a diagnostic candidate, not a qualified successor.
"""
import torch

from shared_temporal_generator_v3 import SharedTemporalGenerator as Parent
from temporal_group_sum_projector_v1 import phase_packet
from shared_stationary_prefix_v1 import stationary_prefix_outputs
from recurrent_sampler_native import dimensions


class ControlGuideGenerator(Parent):
    def __init__(self, *args, guide_mode, **kwargs):
        if guide_mode not in ('original', 'last_force_pressure', 'last_force_pressure_prefix_impulse'):
            raise ValueError('Registered control-guide ablation required')
        super().__init__(*args, **kwargs)
        self.guide_mode = guide_mode
        self.asset_identity.update(control_guide_mode=guide_mode,
            parameter_updates=0, reference_labels_changed=False,
            numerical_native_pressure_solves=0,
            additional_work_included_in_complete_GPU_cost=True)

    def phase_packet(self, *args):
        result = phase_packet(self, *args)
        self._guide_phase = result
        return result

    def pressure_response(self, delta, phase):
        core = self.model
        rhs = core.gather_ops.ordered_rhs(core, delta, phase['shape'], 'flux',
            0, phase['h'] ** 2, phase['dt'])
        return core.pressure((rhs * (phase['rho'] / phase['h'])).reshape(
            1, 1, *phase['shape']), phase['counts'])

    def predict_packet(self, fields, scene, output_rate, k, packet, events, event_mask):
        outputs = Parent.predict_packet(self, fields, scene, output_rate, k,
            packet, events, event_mask)
        if self.guide_mode == 'original':
            return outputs
        phase = self._guide_phase
        total = phase['known_force_sum']
        stride, width = 240 // output_rate, packet.shape[0]
        core = self.model
        for lead, output in enumerate(outputs, 1):
            end, alpha = lead * stride, lead / k
            last = self.control_sum.sum_from_packet(phase['counts'], packet[end - 1:end].contiguous())
            delta = [width * a - b for a, b in zip(last, total)]
            # Match the endpoint phase's linear viscosity on its force addition.
            delta = core.gather_ops.three(core, delta, delta, phase['counts'],
                'viscosity', .001 / phase['rho'] * phase['dt'] / phase['h'] ** 2)
            response = self.pressure_response(delta, phase)
            output['pressure'] = (phase['base_pressure'] + response).contiguous().flatten()
            if self.guide_mode == 'last_force_pressure_prefix_impulse' and lead < k:
                prefix = self.control_sum.sum_from_packet(phase['counts'], packet[:end].contiguous())
                impulse_delta = [a - alpha * b for a, b in zip(prefix, total)]
                head_phase = dict(phase, dt=lead / output_rate)
                pressure = self.pressure_response(impulse_delta, head_phase)
                velocity = core.gather_ops.three(core, [pressure[0, 0]], impulse_delta,
                    phase['counts'], 'momentum', head_phase['dt'] / (phase['rho'] * phase['h']),
                    as_double=True)
                for name, size, correction in zip(('vx', 'vy', 'vz'), dimensions(phase['counts'])[:3], velocity):
                    output[name] = (output[name].reshape(size).double() + correction).float().contiguous().flatten()
        return stationary_prefix_outputs(fields, packet, output_rate, k, outputs)


class ControlGuideGraph(ControlGuideGenerator):
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
                ControlGuideGenerator.predict_packet(self, state, scene, output_rate, k, controls, edge_values, mask)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = ControlGuideGenerator.predict_packet(self, state, scene, output_rate, k, controls, edge_values, mask)
        torch.cuda.synchronize()
        self.graphs[key] = graph, state, controls, edge_values, mask, outputs

    def close(self):
        self.release_graphs()
        super().close()
