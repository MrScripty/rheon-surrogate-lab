"""Same shared learned model with exact batched prescribed-force summation."""
from shared_temporal_generator_v1 import SharedTemporalGenerator as Parent
from batched_known_controls_v1 import CudaKnownControlSum
from temporal_group_sum_projector_v1 import phase_packet


class SharedTemporalGenerator(Parent):
    phase_packet = phase_packet

    def __init__(self, *args, force_sum_build, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_sum = CudaKnownControlSum(force_sum_build)
        self.asset_identity.update(batched_known_force_sum=self.control_sum.identity,
            dense_per_fine_force_fields_materialized=False,
            learned_weights_unchanged=True, ordered_force_arithmetic_unchanged=True)

    def close(self):
        self.control_sum.close()
        super().close()
