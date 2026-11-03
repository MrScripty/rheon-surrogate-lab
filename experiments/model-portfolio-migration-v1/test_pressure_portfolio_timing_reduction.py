"""Adversarial joins on retained real pressure results; no neural runtime."""
import copy
import math
from pathlib import Path
import unittest

import summarize_pressure_portfolio_timing as reduction


class PressureTimingReductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = reduction.read(reduction.BASE / 'pressure-portfolio-GPU-timing-protocol-02.json')
        cls.case = reduction.read(reduction.ROOT / cls.protocol['frozen_cases']['path'])['cases'][0]
        state = reduction.read(reduction.BASE / 'pressure-portfolio-GPU-timing-01/status.json')
        cls.actual = reduction.read(reduction.ROOT / state['completed'][0]['result'])['rows'][0]
        cls.reference = reduction.read(reduction.ROOT / cls.protocol['accuracy']['C03']['path'])['rows'][0]
        score = reduction.read(reduction.ROOT / cls.protocol['score'])
        cls.scored = reduction.read(reduction.ROOT / score['details']['path'])['records']['C03']['rows'][0]

    def check(self, actual=None, case=None, reference=None, scored=None):
        return reduction.verify_row(self.actual if actual is None else actual,
            self.case if case is None else case, self.reference if reference is None else reference,
            self.scored if scored is None else scored, 10)

    def test_original_native_case_output_and_raw_statistics_join(self):
        self.assertEqual(self.check(), self.actual['raw_wall_nanoseconds'])

    def test_domain_density_cadence_input_or_output_mismatch_rejects(self):
        changes = {'case_id': 'other-case', 'trajectory_id': 'other-scene', 'parent': 99,
                   'native_frame': 14, 'counts_xyz': [16, 16, 15], 'cell_size_m': .1,
                   'cells_per_metre': 16, 'input_dt_s': 1/30,
                   'input_sha256': {}, 'output_sha256': 'a'*64,
                   'output_independent_of_inputs': False, 'median_ms': 0}
        for key, value in changes.items():
            changed = copy.deepcopy(self.actual)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check(actual=changed)

    def test_common_accuracy_and_old_reference_cannot_be_substituted(self):
        for key in ('reference_pressure_sha256', 'reference_rhs_sha256'):
            changed = copy.deepcopy(self.reference)
            changed[key] = 'a'*64
            with self.subTest(reference=key), self.assertRaises(ValueError):
                self.check(reference=changed)
        for metric in ('unknown_metric', 'common_equation_relative_l2'):
            changed = copy.deepcopy(self.scored)
            changed['metrics'][metric] = math.nan
            with self.subTest(metric=metric), self.assertRaises(ValueError):
                self.check(scored=changed)

    def test_incomplete_or_invalid_raw_samples_reject(self):
        for samples in ([1]*9, [1]*11, [True]*10, [1.0]*10, [0]*10, [-1]*10):
            changed = copy.deepcopy(self.actual)
            changed['raw_wall_nanoseconds'] = samples
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                self.check(actual=changed)

    def test_percentile_and_pooled_median_are_sample_reductions(self):
        samples = [v*1_000_000 for v in range(1, 21)]
        summary = reduction.summarize(list(reversed(samples)))
        self.assertEqual(summary, dict(samples=20, minimum_ms=1, median_ms=10.5,
                                      p95_ms=19.05, maximum_ms=20))

    def test_model_frontier_preserves_equal_points_and_rejects_dominated(self):
        models = {name: dict(summary=dict(worst_case_median_ms=latency,
                    worst_accuracy={'error': error}))
                  for name, latency, error in [('fast', 1, .1), ('accurate', 2, .01),
                                                ('dominated', 3, .2), ('equal', 1, .1)]}
        self.assertEqual(reduction.frontier(models, 'error'), ['fast', 'accurate', 'equal'])


if __name__ == '__main__':
    unittest.main()
