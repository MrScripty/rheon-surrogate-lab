"""GPU graph setup without redundant poisoning of fully overwritten scratch.

All deterministic algorithms and explicit zeros/full/boundary fills remain.
Only allocation poisoning is disabled during serialized setup/capture, then
the process setting is restored. No learned coefficient or arithmetic changes.
"""
from torch.utils import deterministic
from debug.cube128_exact_transform_candidate_v2 import Cube128ExactTransformCandidate


class NoScratchFillGraph(Cube128ExactTransformCandidate):
    def __init__(self, pair, shared_pair, build, control_build):
        super().__init__(pair, shared_pair, build, control_build)
        self.asset_identity.update(
            graph_scratch_allocation='Allocation poisoning disabled only during serialized graph setup; restored before returning',
            overwrite_contract='Learned transport/linear/fixed-transform/control kernels write every live output element before use; state/packet clones copy complete inputs',
            explicit_value_initialization='All zeros/full/constant-mode/boundary fills unchanged',
            deterministic_algorithms_unchanged=True, learned_arithmetic_unchanged=True)

    def prepare_block(self, fields, scene, step, horizons, substeps=1):
        original = deterministic.fill_uninitialized_memory
        deterministic.fill_uninitialized_memory = False
        try:
            return super().prepare_block(fields, scene, step, horizons, substeps)
        finally:
            deterministic.fill_uninitialized_memory = original
