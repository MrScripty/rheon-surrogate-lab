"""CUDA graph execution of source-bound fused learned primitives."""
from expanded_cuda_graph_v2 import ExpandedCudaGraphGeneratorV2
from fused_cuda_generator_v1 import FusedCudaGenerator


class FusedCudaGraphGenerator(ExpandedCudaGraphGeneratorV2):
    def __init__(self, pair, shared_pair, build, fusion_mode):
        self.model = FusedCudaGenerator(pair, shared_pair, build, fusion_mode)
        self.asset_identity = dict(self.model.asset_identity, execution='CUDA graph of complete fused generator',
            graph_replay_includes='All current-state/control D2D input copies and complete K output stages',
            graph_setup_excluded_from_steady_state=True)
        self.graphs = {}
        self.setup_observations = []
