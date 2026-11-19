"""Whole shared endpoint/interior CUDA graph; owned output clones on replay."""
import torch

from shared_temporal_generator_v1 import SharedTemporalGenerator


class SharedTemporalGraph(SharedTemporalGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graphs = {}
        self.asset_identity.update(complete_shared_temporal_graph=True,
            graph_replay_includes='Currentstate/control/eventcopies,oneendpoint/sharedtrunk,allinteriorlearnedprojections,andownedoutputclones',
            setup_and_warmups_separate=True)

    @torch.inference_mode()
    def predict_packet(self, fields, scene, output_rate, k, packet, events, event_mask):
        key = (tuple(scene['counts']), scene['cells_per_metre'], output_rate, k, scene['acceleration'])
        if key not in self.graphs:
            self.release_graphs()
            self.prepare_graph(key, fields, scene, output_rate, k, packet, events, event_mask)
        graph, state, controls, edge_values, mask, outputs = self.graphs[key]
        for name, value in state.items():
            source = fields[name]
            if source.device != value.device or source.dtype != value.dtype or source.shape != value.shape:
                raise ValueError('Graph requires exact typed current field layout')
            value.copy_(source)
        for value, source in [(controls, packet), (edge_values, events), (mask, event_mask)]:
            if source.device != value.device or source.dtype != value.dtype or source.shape != value.shape:
                raise ValueError('Graph requires exact known control/event layout')
            value.copy_(source)
        graph.replay()
        return [{n: v.clone() for n, v in output.items()} for output in outputs]

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

    def release_graphs(self):
        torch.cuda.synchronize()
        self.graphs.clear()

    def close(self):
        self.release_graphs()
        super().close()
