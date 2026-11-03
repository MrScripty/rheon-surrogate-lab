"""Full native entry with paired middle samples and RN64 interval guards."""
from debug.geometry_specialized_graph_v1 import GeometrySpecializedGraph
from complete_native_entry_v1 import CompleteNativeEntry


class GuardedMiddleGraph(GeometrySpecializedGraph):
    def __init__(self, pair, shared_pair, build, control_build):
        super().__init__(pair, shared_pair, build, control_build)
        identity = self.model.kernels.identity
        if identity.get('guarded_middle_internal_precision') != 'pairedFP32middle_terminal_with_F64intervals_and_fullfallback':
            raise ValueError('Registered guarded middle interval build required')
        self.asset_identity.update(
            sampler_arithmetic_unchanged=False,
            sampler_internal_precision='Paired FP32 middle/terminal with F64 interval propagation and complete original F64 transport fallback',
            enclosure_derivation=identity['enclosure_derivation'],
            diagnostic_counter_instrumentation=identity['diagnostic_counter_instrumentation'],
            native_field_dtypes_unchanged=True,
            learned_coefficients_unchanged=True,
            pressure_transform_arithmetic_unchanged=True)


class GuardedMiddleEntry(CompleteNativeEntry):
    generator_class = GuardedMiddleGraph
