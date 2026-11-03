"""Native entry with terminal-only guarded paired-FP32 interpolation."""
from debug.geometry_specialized_graph_v1 import GeometrySpecializedGraph
from complete_native_entry_v1 import CompleteNativeEntry


class GuardedTerminalGraph(GeometrySpecializedGraph):
    def __init__(self, pair, shared_pair, build, control_build):
        super().__init__(pair, shared_pair, build, control_build)
        if self.model.kernels.identity.get('guarded_terminal_internal_precision') != 'pairedFP32withF64fallback':
            raise ValueError('Registered guarded terminal build required')
        self.asset_identity.update(
            sampler_arithmetic_unchanged=False,
            sampler_internal_precision='Middle samples/coordinates remain F64; terminal paired FP32 with certified rounding or original F64 fallback',
            terminal_enclosure_derivation=self.model.kernels.identity['enclosure_derivation'],
            native_field_dtypes_unchanged=True, learned_coefficients_unchanged=True,
            pressure_transform_arithmetic_unchanged=True)


class GuardedTerminalEntry(CompleteNativeEntry):
    generator_class = GuardedTerminalGraph
