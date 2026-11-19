"""Optional known-control displacement moment for the tracer readout.

The reference advects before adding each fine interval's force. Force sample r
therefore contributes to m-1-r subsequent advections, not to all m intervals.
This guide weights the prescribed spatial force fields accordingly, projects
the accumulated velocity using restored learned maps, and transports current
tracer once over each full output horizon. It remains an approximation: moving
fluid characteristics, nonlinear acceleration and viscosity are not exact.
"""
import torch

from predicted_fine_readout_v1 import FineReadoutGenerator as Parent
from shared_stationary_prefix_v1 import stationary_prefix_outputs
from recurrent_sampler_native import dimensions


class FineReadoutGenerator(Parent):
    def __init__(self, *args, fine_mode, **kwargs):
        self.moment_tracer = fine_mode == 'preoutput_fine_fullstate_moment_tracer'
        super().__init__(*args, fine_mode='preoutput_fine_fullstate' if self.moment_tracer else fine_mode, **kwargs)
        self.asset_identity.update(predicted_fine_readout_mode=fine_mode,
            known_control_displacement_moment=self.moment_tracer,
            moment_velocity_exact_fluid_trajectory=False,
            extra_full_transport_calls_per_block='K when moment guide enabled; zero otherwise')

    def predict_packet(self, fields, scene, output_rate, k, packet, events, event_mask):
        outputs = Parent.predict_packet(self, fields, scene, output_rate, k, packet, events, event_mask)
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
