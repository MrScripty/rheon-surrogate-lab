"""Add exact known pulse events without changing sampled240 reference labels."""
from long_temporal_dataset_v1 import LongTemporalDataset as Parent
from pulse_event_controls_v1 import event_packet


class LongTemporalDataset(Parent):
    def event_inputs(self, example):
        _, receipt, _ = self.receipts[example['identifier']]
        rate, k = example['output_steps_per_s'], example['K']
        if rate not in (30, 60, 120) or k not in (1, 2, 3, 4):
            raise ValueError('Registered output cadence and full physical output count required')
        return event_packet(receipt['scene']['forcing_program'],
            example['start_fine_step'] / 240, k / rate, [j / rate for j in range(1, k + 1)])

    def inputs(self, example):
        result = super().inputs(example)
        result['known_pulse_events'] = self.event_inputs(example)
        result['reference_event_sampling'] = 'left_sampled240_controls; exact event metadata does not replace discretized labels'
        return result
