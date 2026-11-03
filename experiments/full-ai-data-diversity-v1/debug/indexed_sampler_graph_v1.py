"""Integer-only sampler addressing under an explicit supported geometry bound."""
from debug.no_scratch_fill_graph_v1 import NoScratchFillGraph


class IndexedSamplerGraph(NoScratchFillGraph):
    def __init__(self, pair, shared_pair, build, control_build):
        super().__init__(pair, shared_pair, build, control_build)
        self.asset_identity.update(
            sampler_address_execution='Unsigned32 local decode and already-bounded donor addresses; all F64 math/order unchanged',
            geometry_axis_cap=1024, learned_parameters_changed=False)

    def prepare_geometry(self, counts):
        if len(counts) != 3 or any(type(n) is not int or not 1 <= n <= 1024 for n in counts):
            raise ValueError('Integer sampler requires three base grid dimensions in [1,1024]')
        return super().prepare_geometry(counts)

    def prepare_block(self, fields, scene, step, horizons, substeps=1):
        counts = scene['counts']
        if len(counts) != 3 or any(type(n) is not int or not 1 <= n <= 1024 for n in counts):
            raise ValueError('Integer sampler geometry outside registered bound')
        return super().prepare_block(fields, scene, step, horizons, substeps)
