"""Generic64matrix-layout accumulation repair around unchanged learned maps."""
from debug.affine_generator_graph_v2 import AffineGeneratorGraph
from debug.cube128_exact_transform_candidate_v2 import FixedAxisTransform


class Cube64LayoutTransformCandidate(AffineGeneratorGraph):
    def __init__(self, pair, shared_pair, build, control_build):
        super().__init__(pair, shared_pair, build, control_build)
        replacement = FixedAxisTransform('runs/full-ai-data-diversity-v1/debug/cube64-density100-failure-01/dct-build-01')
        self.fixed_transform.close()
        self.fixed_transform = replacement
        self.asset_identity.update(fixed_transform_build=replacement.kernels.identity,
            generic64matrix_accumulation='SequentialFMAexcept forwardXlast4output coordinates use two lanes; arbitrary-matrix qualified',
            other_shape_accumulation_unchanged=True, learned_coefficients_unchanged=True,
            native_numerical_oracle_in_candidate=False)
