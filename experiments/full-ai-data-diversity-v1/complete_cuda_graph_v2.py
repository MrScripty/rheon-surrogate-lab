"""Complete dynamic control graph with the verified learned-evaluation repair."""
from complete_cuda_graph_v1 import CompleteCudaGraphGenerator
from fused_cuda_generator_v2 import FusedCudaGeneratorV2
from dynamic_cuda_controls_v1 import CudaKnownControls

class CompleteCudaGraphGeneratorV2(CompleteCudaGraphGenerator):
    def __init__(self,pair,shared_pair,build,control_build):
        self.model=FusedCudaGeneratorV2(pair,shared_pair,build)
        self.controls=CudaKnownControls(control_build)
        self.asset_identity=dict(self.model.asset_identity,execution='complete dynamic-control CUDA graph with stable learned numerical evaluation',
            known_controls=self.controls.identity,known_input_packet_columns=18,
            graph_includes='AllGPUcoordinate/phase/force generation and every learned internal advance',
            replay_scope='Complete current-fieldD2Dcopies, compact known-inputH2Dpacket, graph replay and independentKcomplete output clones',
            cpu_packet_lifetime='Owned until actual recordedH2Devent completes; maximum64inflight')
        self.graphs={};self.setup_observations=[];self.pending_packets=[]
