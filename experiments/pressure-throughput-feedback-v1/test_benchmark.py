"""Fresh request identity must not masquerade as a physical regression."""
import copy
import unittest

from benchmark import numerical_frames


class NumericalComparison(unittest.TestCase):
    def test_identity_and_clocks_are_excluded_but_pressure_and_calls_remain(self):
        frame = {'step': 16, 'ai_seconds': 1., 'reference_seconds': .1,
            'relative_pressure_difference': 1e-6,
            'diagnostic_attempt': {'request_sha256': [1, 2], 'outcome': {'qualified': True, 'residual_l2': 1e-8}}}
        other = copy.deepcopy(frame)
        other.update(ai_seconds=2., reference_seconds=.2)
        other['diagnostic_attempt']['request_sha256'] = [3, 4]
        self.assertEqual(numerical_frames([frame]), numerical_frames([other]))
        other['relative_pressure_difference'] *= 2
        self.assertNotEqual(numerical_frames([frame]), numerical_frames([other]))
