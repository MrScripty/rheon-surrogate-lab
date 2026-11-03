"""N8-only 3D FFT execution; unchanged learned response and original N1."""
import types
from cell_coordinate_pressure_graph_v1 import CellCoordinatePressureGraph
from complete_native_entry_v1 import CompleteNativeEntry
from fft3d_dct_v3 import FFT3DDCT


class FFT3DPressureGraph(CellCoordinatePressureGraph):
    def __init__(self, pair, shared_pair, build, control_build):
        super().__init__(pair, shared_pair, build, control_build)
        self.fft_transform = FFT3DDCT()
        original = self.model.pressure

        def pressure(core, value, counts):
            if not core._nested_enabled:
                return original(value, counts)
            return self.fft_transform.pressure(core, value, counts)

        self.model.pressure = types.MethodType(pressure, self.model)
        self.asset_identity.update(pressure_transform_N8=self.fft_transform.identity,
            pressure_transform_N1='original unchanged fixed-axis transform',
            learned_response_and_all_parameters_unchanged=True,
            accumulation_order_changed_N8=True, native_numerical_oracle_in_candidate=False,
            whole_model_accuracy_requires_qualification=True)

    def prepare_geometry(self, counts):
        super().prepare_geometry(counts)
        self.fft_transform.prepare(tuple(reversed(counts)))

    def close(self):
        super().close()
        self.fft_transform.close()


class FFT3DPressureNativeEntry(CompleteNativeEntry):
    generator_class = FFT3DPressureGraph

    def __init__(self, *args):
        super().__init__(*args)
        self.asset_identity.update(learned_arithmetic_unchanged=False, learned_coefficients_unchanged=True,
            original_N1_arithmetic_unchanged=True, new_qualification_required=True)
