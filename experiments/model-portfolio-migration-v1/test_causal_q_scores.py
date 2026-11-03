"""Distinguish raw q errors, centered errors, unit conversion and zero targets."""
import math
import unittest

import numpy as np

from reduce_causal_request_component_benchmark import score_q


class CausalQScoreTests(unittest.TestCase):
    def test_constant_bias_remains_in_raw_and_physical_errors(self):
        target = np.array([0.,1.],dtype='f4').reshape(1,1,1,1,1,2)
        predicted = np.array([2.,3.],dtype='f4').reshape(target.shape)
        result = score_q(predicted,target,.25)
        self.assertAlmostEqual(result['q_relative_l2'],math.sqrt(8))
        self.assertEqual(result['q_mean_free_relative_l2'],0)
        self.assertEqual(result['q_absolute_rms_normalized'],2)
        self.assertEqual(result['q_absolute_rms_physical'],.5)
        self.assertFalse(result['q_target_floor_active'])

    def test_zero_reference_exposes_both_floors(self):
        target = np.zeros((1,1,1,1,1,2),dtype='f4')
        result = score_q(np.ones_like(target),target,.25)
        self.assertTrue(result['q_target_floor_active'])
        self.assertTrue(result['q_reference_centered_floor_active'])
        self.assertEqual(result['q_absolute_max_physical'],.25)
        self.assertGreater(result['q_relative_l2'],1e30)

    def test_changed_dtype_is_rejected(self):
        data = np.zeros((1,1,1,1,1,2),dtype='f8')
        with self.assertRaises(ValueError):
            score_q(data,data,1.)


if __name__ == '__main__':
    unittest.main()
