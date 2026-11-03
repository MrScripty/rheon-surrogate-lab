"""One N8 FP32 donor trial with exact managed parameters and original N1."""
from cell_coordinate_pressure_graph_v1 import CellCoordinatePressureGraph
from complete_native_entry_v1 import CompleteNativeEntry


class FP32InterpolatorGraph(CellCoordinatePressureGraph):
    def __init__(self, pair, shared_pair, build, control_build,
                 sampler_build='runs/full-ai-data-diversity-v1/cuda-fusion-build-22'):
        super().__init__(pair, shared_pair, build, control_build, sampler_build)
        if self.nested_kernels.identity.get('sampler_interpolation') != 'guarded_FP32_donors_F64_coordinates':
            raise ValueError('Source-bound FP32 interpolation trial required')
        self.asset_identity.update(sampler_precision='N8 guarded FP32 donors; F64 coordinates, exact learned parameters, original N1',
                                   donor_rounding_changed=True, learned_coefficients_unchanged=True,
                                   learned_coefficient_arithmetic_order_preserved=True,
                                   whole_model_accuracy_requires_qualification=True)


class FP32InterpolatorNativeEntry(CompleteNativeEntry):
    generator_class = FP32InterpolatorGraph

    def __init__(self, *args):
        super().__init__(*args)
        self.asset_identity.update(learned_arithmetic_unchanged=False, learned_coefficients_unchanged=True,
                                   original_N1_arithmetic_unchanged=True, donor_rounding_changed=True)
