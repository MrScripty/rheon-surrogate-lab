"""Complete graph using unchanged learned heads and physical sampler coordinates."""
from complete_cuda_graph_v1 import CompleteCudaGraphGenerator
from fused_cuda_generator_v3 import FusedCudaGeneratorV3
from dynamic_cuda_controls_v1 import CudaKnownControls


class CompleteCudaGraphGeneratorV3(CompleteCudaGraphGenerator):
    def __init__(self, pair, shared_pair, build, control_build):
        self.model = FusedCudaGeneratorV3(pair, shared_pair, build)
        self.controls = CudaKnownControls(control_build)
        self.asset_identity = dict(self.model.asset_identity,
            execution='Complete dynamic-control CUDA graph with physical sampler coordinates',
            known_controls=self.controls.identity, known_input_packet_columns=18,
            graph_includes='All known GPU force generation and all learned stages for every output',
            replay_scope='Complete current-field copies, compact known-input upload, graph replay and independently owned outputs')
        self.graphs = {}
        self.setup_observations = []
        self.pending_packets = []
