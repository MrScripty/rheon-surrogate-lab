"""Whole captured CUDA graph for independent pressure-head generation."""
import torch
from guided_pressure_residual_v1 import GuidedPressureResidual

class GuidedPressureGraph(GuidedPressureResidual):
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
                GuidedPressureResidual.predict_packet(self, state, scene, output_rate, k, controls, edge_values, mask)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = GuidedPressureResidual.predict_packet(self, state, scene, output_rate, k, controls, edge_values, mask)
        torch.cuda.synchronize()
        self.graphs[key] = graph, state, controls, edge_values, mask, outputs

    def close(self):
        self.release_graphs()
        super().close()
