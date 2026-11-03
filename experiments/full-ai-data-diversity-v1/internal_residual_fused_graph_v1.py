"""Same restored residual graph using the existing precise learned-linear kernel."""
from internal_residual_graph_v1 import InternalResidualGraphGenerator


class InternalResidualFusedGraphGenerator(InternalResidualGraphGenerator):
    def __init__(self, *args):
        super().__init__(*args)
        core = self.backbone.model
        core.fusion_mode = 'transport_and_linear'
        self.backbone.linear = core.linear
        self.asset_identity.update(execution='CUDAgraph_internal_residual_fused_linear_v1',
            precise_existing_learned_linear_build=core.kernels.identity,
            learned_parameters_unchanged=True,
            complete_bitwise_equivalence_requires_qualification=True)
