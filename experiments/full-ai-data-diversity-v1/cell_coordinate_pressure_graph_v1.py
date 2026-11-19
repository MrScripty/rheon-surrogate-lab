"""N8-only cell-coordinate trial; original N1 transition is retained."""
from bound_sampler_pressure_graph_v1 import BoundSamplerPressureGraph
from complete_native_entry_v1 import CompleteNativeEntry


class CellCoordinatePressureGraph(BoundSamplerPressureGraph):
    def __init__(self, pair, shared_pair, build, control_build,
                 sampler_build='runs/full-ai-data-diversity-v1/cuda-fusion-build-21'):
        super().__init__(pair, shared_pair, build, control_build, sampler_build)
        if self.nested_kernels.identity.get('sampler_coordinate_system') != 'cell_coordinates_N8_only':
            raise ValueError('Exact cell-coordinate trial build required')
        self.asset_identity.update(sampler_coordinate_system='cell_coordinates_N8_only',
                                   ordered_arithmetic_preserved_in_generated_source=False,
                                   learned_coefficient_arithmetic_order_preserved=True,
                                   coordinate_rounding_changed=True,
                                   original_N1_arithmetic_unchanged=True,
                                   whole_model_accuracy_requires_qualification=True)


class CellCoordinatePressureNativeEntry(CompleteNativeEntry):
    generator_class = CellCoordinatePressureGraph

    def __init__(self, *args):
        super().__init__(*args)
        self.asset_identity.update(learned_arithmetic_unchanged=False,
                                   learned_coefficients_unchanged=True,
                                   coordinate_rounding_changed=True,
                                   original_N1_arithmetic_unchanged=True)
