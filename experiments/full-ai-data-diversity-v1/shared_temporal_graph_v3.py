"""Complete shared graph including exact stationary-prefix output handling."""
import torch

from shared_temporal_graph_v2 import SharedTemporalGraph as Parent
from shared_temporal_generator_v3 import SharedTemporalGenerator


class SharedTemporalGraph(Parent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.asset_identity.update(exact_stationary_prefix=True,
            frozen_accuracy_and_loss_floors_unchanged=True)

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
                SharedTemporalGenerator.predict_packet(self, state, scene, output_rate, k, controls, edge_values, mask)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = SharedTemporalGenerator.predict_packet(self, state, scene, output_rate, k, controls, edge_values, mask)
        torch.cuda.synchronize()
        self.graphs[key] = graph, state, controls, edge_values, mask, outputs
