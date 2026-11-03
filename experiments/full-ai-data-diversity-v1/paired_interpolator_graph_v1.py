"""N8-only compensated pair interpolation; original N1 and learned values stay."""
from cell_coordinate_pressure_graph_v1 import CellCoordinatePressureGraph
from complete_native_entry_v1 import CompleteNativeEntry


class PairedInterpolatorGraph(CellCoordinatePressureGraph):
    def __init__(self, pair, shared_pair, build, control_build,
                 sampler_build='runs/full-ai-data-diversity-v1/cuda-fusion-build-23'):
        super().__init__(pair, shared_pair, build, control_build, sampler_build)
        if self.nested_kernels.identity.get('sampler_interpolation') != 'compensated_FP32_pairs_F64_coordinates':
            raise ValueError('Source-bound compensated interpolation build required')
        self.asset_identity.update(sampler_precision='N8 compensated FP32 pairs; F64 coordinates and exact learned parameters; original N1',
                                   donor_rounding_changed=True, learned_coefficients_unchanged=True,
                                   whole_model_accuracy_requires_qualification=True)


class PairedInterpolatorNativeEntry(CompleteNativeEntry):
    generator_class = PairedInterpolatorGraph

    def __init__(self, *args):
        super().__init__(*args)
        self.asset_identity.update(learned_arithmetic_unchanged=False, learned_coefficients_unchanged=True,
                                   original_N1_arithmetic_unchanged=True, donor_rounding_changed=True)
