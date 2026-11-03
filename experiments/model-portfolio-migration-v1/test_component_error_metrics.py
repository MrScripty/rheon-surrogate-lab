"""Literal component-score examples independent of a learned implementation."""
import unittest
import numpy as np

from component_error_metrics import masked_errors


class ErrorTests(unittest.TestCase):
    def test_masked_vector_and_absolute_errors(self):
        value = masked_errors(np.array([4.,3.,999.]),np.array([3.,4.,-999.]),np.array([1.,1.,0.]))
        self.assertAlmostEqual(value['masked_relative_l2'],np.sqrt(2)/5)
        self.assertAlmostEqual(value['masked_absolute_rms'],1.)
        self.assertEqual(value['masked_absolute_max'],1.)
        self.assertEqual(value['free_face_values'],2)
        self.assertFalse(value['target_denominator_floor_active'])

    def test_zero_target_retains_absolute_error_and_floor_flag(self):
        value = masked_errors(np.array([1.]),np.array([0.]),np.array([1.]))
        self.assertTrue(value['target_denominator_floor_active'])
        self.assertEqual(value['target_masked_l2'],0.)
        self.assertEqual(value['masked_absolute_max'],1.)

    def test_empty_or_weighted_mask_rejected(self):
        for mask in (np.array([0.]),np.array([.5])):
            with self.assertRaisesRegex(ValueError,'binary face mask'):
                masked_errors(np.array([1.]),np.array([1.]),mask)


if __name__ == '__main__':
    unittest.main()
