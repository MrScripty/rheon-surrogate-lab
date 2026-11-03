"""Qualified graph arithmetic with registered full-objective residual weights."""
from internal_residual_graph_v1 import InternalResidualGraphGenerator
from internal_residual_generator_v3 import InternalResidualGenerator

class InternalResidualFusedGraphGenerator(InternalResidualGraphGenerator):
    def __init__(self,*args):
        InternalResidualGenerator.__init__(self,*args)
        self.graphs={};self.geometry=None
        core=self.backbone.model;core.fusion_mode='transport_and_linear';self.backbone.linear=core.linear
        self.asset_identity.update(execution='CUDAgraph_internal_residual_fused_linear_v3_direct_delta_fit',
            precise_existing_learned_linear_build=core.kernels.identity,learned_parameters_unchanged=True,
            complete_bitwise_equivalence_requires_qualification=True)
