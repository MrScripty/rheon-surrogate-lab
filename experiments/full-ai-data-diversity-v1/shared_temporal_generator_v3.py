"""Shared generator preserving exact stationary sampled-force prefixes."""
from shared_temporal_generator_v2 import SharedTemporalGenerator as Parent
from shared_stationary_prefix_v1 import stationary_prefix_outputs


class SharedTemporalGenerator(Parent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.asset_identity.update(exact_stationary_prefix=True,
            stationary_rule='Current pressure/velocity zero and no sampled prescribed force before this output',
            frozen_accuracy_and_loss_floors_unchanged=True)

    def predict_packet(self, fields, scene, output_rate, k, packet, events, event_mask):
        outputs = Parent.predict_packet(self, fields, scene, output_rate, k, packet, events, event_mask)
        return stationary_prefix_outputs(fields, packet, output_rate, k, outputs)
