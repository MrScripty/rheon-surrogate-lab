"""A recent population regression cannot disappear behind a better aggregate."""
import json
from pathlib import Path
import unittest
from analyze import gates


class PopulationGates(unittest.TestCase):
    def test_retained_failed_final_is_rejected_with_actual_observed_metrics(self):
        lab = Path(__file__).resolve().parents[2]
        old = json.loads((lab/'docs/plans/water-stopping-floor-v1/reports/validation-summary.json').read_text())
        models = {}
        for name, record in old['models'].items():
            models[name] = {**record, 'scenes': record['episodes'], 'complete_scenes': record['episodes'],
                            'qualified_scenes': record['qualified_episodes']}
        result = gates(models['reference-current'], models['late-minimum8'], 8)
        self.assertTrue(result['all_declared_scenes_qualified'])
        self.assertTrue(result['mean_calls_not_regressed'])
        self.assertFalse(result['worst_pressure_not_regressed'])
        self.assertFalse(result['maximum_calls_not_regressed'])

    def test_missing_scene_rejects_apparently_better_partial_population(self):
        reference = {'scenes': 8, 'complete_scenes': 8, 'qualified_scenes': 8,
            'worst_paired_pressure_relative_l2': 1e-5, 'mean_network_evaluations': 10, 'maximum_network_evaluations': 12}
        partial = {**reference, 'scenes': 7, 'complete_scenes': 7, 'qualified_scenes': 7,
                   'worst_paired_pressure_relative_l2': 1e-8, 'mean_network_evaluations': 6, 'maximum_network_evaluations': 8}
        self.assertFalse(any(gates(reference, partial, 8).values()))


if __name__ == '__main__':
    unittest.main()
