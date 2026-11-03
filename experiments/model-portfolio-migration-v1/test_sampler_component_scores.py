"""Literal vector-error checks for zero-reference points inside nonzero groups."""
import math
import unittest

import numpy as np

from reduce_sampler_component_benchmark import score_stage


class SamplerScoreTests(unittest.TestCase):
    def test_zero_point_error_is_absolute_and_separate_from_nonzero_relative_error(self):
        target = np.array([[3.,4.,0.],[0.,0.,0.]])
        prediction = np.array([[4.,3.,0.],[1.,0.,0.]])
        score = score_stage(prediction,target)
        self.assertAlmostEqual(score['relative_l2'],math.sqrt(3)/5)
        self.assertAlmostEqual(score['absolute_rms'],math.sqrt(.5))
        self.assertEqual(score['absolute_max'],1.)
        self.assertAlmostEqual(score['worst_point_relative_l2_nonzero_targets'],math.sqrt(2)/5)
        self.assertEqual(score['zero_target_worst_absolute_l2'],1.)
        self.assertEqual(score['zero_target_points'],1)
        self.assertFalse(score['target_floor_active'])

    def test_zero_group_exposes_floor_and_no_nonzero_point_statistic(self):
        score = score_stage(np.zeros((2,3)),np.zeros((2,3)))
        self.assertTrue(score['target_floor_active'])
        self.assertIsNone(score['worst_point_relative_l2_nonzero_targets'])
        self.assertEqual(score['zero_target_points'],2)
        self.assertEqual(score['zero_target_worst_absolute_l2'],0.)

    def test_wrong_vector_dimension_is_rejected(self):
        with self.assertRaises(ValueError):
            score_stage(np.zeros((2,2)),np.zeros((2,2)))


if __name__ == '__main__':
    unittest.main()
