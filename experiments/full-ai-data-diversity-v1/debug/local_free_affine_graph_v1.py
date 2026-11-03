"""Affine-fused generator with generic sampler pointer/shape table elimination."""
from debug.affine_generator_graph_v2 import AffineGeneratorGraph


class LocalFreeAffineGraph(AffineGeneratorGraph):
    def __init__(self, pair, shared_pair, build, control_build):
        super().__init__(pair, shared_pair, build, control_build)
        self.asset_identity.update(sampler_address_execution='Direct generic field-pointer selection and shape construction; dynamic per-thread pointer/shape tables removed',
            learned_coefficients_and_floating_operation_order_unchanged=True)
