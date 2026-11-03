"""CUDA graph replay of the unchanged, source-bound learned full-state stages.

Graph setup is explicit and separately timed. Replay includes device-to-device
copies of all current state and prescribed inputs. No CPU model fallback.
"""
import time
import torch
from cuda_divisor_generator_v1 import FIELDS
from cuda_divisor_generator_v2 import CudaDivisorGeneratorV2


class CudaGraphDivisorGeneratorV2:
    def __init__(self, pair, pressure_pair):
        self.model = CudaDivisorGeneratorV2(pair, pressure_pair)
        self.asset_identity = dict(self.model.asset_identity, execution='CUDA graph of complete generator',
            graph_replay_includes='All current-state/control D2D input copies and complete K output stages',
            graph_setup_excluded_from_steady_state=True)
        self.graphs = {}
        self.setup_observations = []

    def state_dict(self):
        return self.model.state_dict()

    def prepare_geometry(self, counts):
        self.model.prepare_geometry(counts)

    def prepare_block(self, fields, scene, step, horizons, external):
        key = (tuple(scene['counts']), scene['cells_per_metre'], horizons)
        if key in self.graphs:
            return
        torch.cuda.synchronize()
        tick = time.perf_counter()
        state = {name: value.clone() for name, value in fields.items()}
        controls = [[value.clone() for value in row] for row in external]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        # Warm up all shape-dependent indices and allocations before capture.
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.model.predict(state, scene, step, horizons, controls)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = self.model.predict(state, scene, step, horizons, controls)
        torch.cuda.synchronize()
        self.graphs[key] = (graph, state, controls, outputs)
        self.setup_observations.append(dict(counts_xyz=list(key[0]), cells_per_metre=key[1], K=horizons,
            graph_setup_ms=1000*(time.perf_counter()-tick)))

    @torch.inference_mode()
    def predict(self, fields, scene, step, horizons, external):
        key = (tuple(scene['counts']), scene['cells_per_metre'], horizons)
        if key not in self.graphs:
            raise ValueError('Explicit graph setup required before timed replay')
        if set(fields) != set(FIELDS) or len(external) != horizons or scene.get('sgs', False):
            raise ValueError('Complete current CUDA state and all prescribed controls required')
        graph, state, controls, outputs = self.graphs[key]
        for name, target in state.items():
            source = fields[name]
            if source.device != target.device or source.dtype != target.dtype or source.shape != target.shape:
                raise ValueError('Exact CUDA graph current-state layout required')
            target.copy_(source)
        for expected, actual in zip(controls, external):
            if len(actual) != 3:
                raise ValueError('Complete prescribed vector required')
            for target, source in zip(expected, actual):
                if source.device != target.device or source.dtype != target.dtype or source.shape != target.shape:
                    raise ValueError('Exact CUDA graph control layout required')
                target.copy_(source)
        graph.replay()
        # Independent output ownership across subsequent calls is part of the
        # generator contract, and these clones are included in every replay time.
        return [{name: value.clone() for name, value in row.items()} for row in outputs]

    def close(self):
        self.graphs.clear()
        self.model.close()
