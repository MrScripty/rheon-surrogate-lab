"""CUDA graph of explicit-time learned pressure-history stages and controls."""
import math
import torch

from temporal_frozen_history_v1 import TemporalFrozenBackbone
from temporal_control_packets_v1 import control_packet
from recurrent_sampler_native import dimensions
from temporal_frozen_history_v1 import FIELDS


class TemporalFrozenGraph(TemporalFrozenBackbone):
    def __init__(self, *args):
        super().__init__(*args)
        self.graphs = {}
        self.asset_identity.update(temporal_graph=True,
            replay_scope='All current fields and known control packet D2D copies, GPU force evaluation, all grouped learned advances, every output and output ownership clones.',
            setup_separate=True)

    @torch.inference_mode()
    def predict(self, fields, scene, start_fine, output_rate, k, groups=None):
        rate = 60 * scene['reference_substeps']
        if (rate not in (240, 480) or output_rate not in (30, 60, 120)
                or type(k) is not int or k not in (1, 2, 3, 4) or scene.get('sgs', False)):
            raise ValueError('Registered physical cadence/K required')
        stride = rate // output_rate
        groups = (1,) * stride if groups is None else tuple(groups)
        if not groups or any(type(n) is not int or n < 1 for n in groups) or sum(groups) != stride:
            raise ValueError('Group widths must partition the whole physical interval')
        key = (tuple(scene['counts']), scene['cells_per_metre'], rate, output_rate, k, groups)
        if set(fields) != set(FIELDS):
            raise ValueError('Complete current native state required')
        for name, size in [('pressure', tuple(reversed(key[0]))), *zip(FIELDS[1:], dimensions(key[0]))]:
            value = fields[name]
            if (value.device.type != 'cuda' or not value.is_contiguous()
                    or value.dtype != (torch.float64 if name == 'pressure' else torch.float32)
                    or value.numel() != math.prod(size)):
                raise ValueError('Exact CUDA native field layout required')
        packet = torch.from_numpy(control_packet(scene, start_fine, k * stride)).to('cuda')
        if key not in self.graphs:
            if self.graphs and (next(iter(self.graphs))[:5] != key[:5] or len(self.graphs) >= 3):
                self.release_graphs()
            self.prepare_graph(key, fields, scene, packet, k, groups)
        graph, state, controls, outputs = self.graphs[key]
        if set(state) != set(fields):
            raise ValueError('Complete current state required')
        for name, target in state.items():
            source = fields[name]
            if source.device != target.device or source.dtype != target.dtype or source.shape != target.shape:
                raise ValueError('Graph requires exact CUDA native input layout')
            target.copy_(source)
        controls.copy_(packet)
        graph.replay()
        return [{n: v.clone() for n, v in output.items()} for output in outputs]

    @torch.inference_mode()
    def prepare_graph(self, key, fields, scene, packet, k, groups):
        if any(v.device.type != 'cuda' for v in fields.values()):
            raise ValueError('GPU-only graph preparation required')
        state = {n: v.clone() for n, v in fields.items()}
        controls = packet.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.predict_packet(state, scene, k, groups, controls)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = self.predict_packet(state, scene, k, groups, controls)
        torch.cuda.synchronize()
        self.graphs[key] = (graph, state, controls, outputs)

    def release_graphs(self):
        torch.cuda.synchronize()
        self.graphs.clear()

    def close(self):
        self.release_graphs()
        super().close()
