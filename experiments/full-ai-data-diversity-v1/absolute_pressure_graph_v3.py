"""Fixed integration-regime branch: original N1, absolute learned N8 pressure."""
import types
import torch
from absolute_pressure_graph_v2 import AbsolutePressureGraph as ParentGraph
from absolute_pressure_graph_v2 import predict as absolute_predict
from debug.gather_fused_graph_v1 import predict as original_predict
from complete_native_entry_v1 import CompleteNativeEntry


@torch.inference_mode()
def predict(core, fields, scene, step, horizons, external, substeps=1):
    if substeps == 1:
        return original_predict(core, fields, scene, step, horizons, external, substeps)
    if substeps == 8:
        return absolute_predict(core, fields, scene, step, horizons, external, substeps)
    raise ValueError('Original known N1/N8 integration metadata required')


class AbsolutePressureGraph(ParentGraph):
    def __init__(self, *args):
        super().__init__(*args)
        self.model.predict = types.MethodType(predict, self.model)
        self.asset_identity.update(pressure_composition_by_regime={'1': 'original_restored_learned_history', '8': 'absolute_restored_learned_pressure'},
                                   regime_selection='Known original integration metadata only; no per-scene IDs or accuracy-triggered fallback')


class AbsolutePressureNativeEntry(CompleteNativeEntry):
    generator_class = AbsolutePressureGraph

    def __init__(self, *args):
        super().__init__(*args)
        self.asset_identity.update(learned_arithmetic_unchanged=False, learned_coefficients_unchanged=True,
                                   pressure_composition_by_regime={'1': 'original', '8': 'absolute'},
                                   accuracy_and_latency_require_new_qualification=True)
