"""Literal scoring examples that distinguish destination groups from joint error."""
import math
import unittest

import numpy as np

from reduce_point_component_benchmark import independent_metrics, independent_destination_metrics


class PointMetricsTests(unittest.TestCase):
    def test_destination_error_is_not_the_joint_error_or_group_average(self):
        target = np.zeros((12,1,1,2),dtype='f8')
        mask = np.zeros_like(target); mask[:,:,:,0] = 1
        target[[0,3,6,9],0,0,0] = [1,2,4,8]
        prediction = target.copy(); prediction[[0,3,6,9],0,0,0] += 1
        prediction[:,:,:,1] = 1e20
        joint = independent_metrics(prediction,target,mask)
        by_kind = independent_destination_metrics(prediction,target,mask)
        self.assertAlmostEqual(joint['masked_relative_l2'],2/math.sqrt(85))
        for name,expected in zip(('vx_face','vy_face','vz_face','tracer_cell'),(1,.5,.25,.125)):
            self.assertEqual(by_kind[name]['masked_relative_l2'],expected)
            self.assertEqual(by_kind[name]['masked_values_count'],3)
        self.assertEqual(joint['masked_values_count'],12)

    def test_zero_target_exposes_floor_and_absolute_error(self):
        target = np.zeros((3,1,1,1)); prediction = np.ones_like(target)
        score = independent_metrics(prediction,target,np.ones_like(target))
        self.assertTrue(score['target_denominator_floor_active'])
        self.assertEqual(score['target_masked_l2'],0)
        self.assertEqual(score['masked_absolute_rms'],1)
        self.assertEqual(score['masked_absolute_max'],1)

    def test_missing_group_and_weighted_mask_are_rejected(self):
        data = np.zeros((3,1,1,1))
        with self.assertRaises(ValueError):
            independent_destination_metrics(data,data,np.ones_like(data))
        with self.assertRaises(ValueError):
            independent_metrics(data,data,np.full_like(data,.5))


if __name__ == '__main__':
    unittest.main()
