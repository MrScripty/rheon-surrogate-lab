"""Literal vector/gauge/zero-denominator oracles for diagnostic pressure errors."""
import unittest

import numpy as np

from diagnose_fixed_viscosity_pressure_error import decomposition


class PressureErrorPartsTests(unittest.TestCase):
    def test_three_known_errors_sum_to_the_original_error(self):
        result = decomposition(np,[-.4,.4],[-.7,.7],[-.9,.9],[-1.,1.])
        for name,expected in [('total',.6),('output_quantization',.3),('request_propagation',.2),('pressure_map',.1)]:
            self.assertAlmostEqual(result[name]['relative_l2'],expected,places=12)
        self.assertLess(result['decomposition_residual_l2'],1e-14)

    def test_independent_pressure_gauges_do_not_change_the_parts(self):
        result = decomposition(np,[99.6,100.4],[6.3,7.7],[-13.9,-12.1],[999.,1001.])
        for name,expected in [('total',.6),('output_quantization',.3),('request_propagation',.2),('pressure_map',.1)]:
            self.assertAlmostEqual(result[name]['relative_l2'],expected,places=12)

    def test_nonzero_error_against_zero_mean_free_truth_is_undefined(self):
        result = decomposition(np,[-1.,1.],[0.,0.],[0.,0.],[5.,5.])
        self.assertIsNone(result['total']['relative_l2'])
        self.assertIsNone(result['output_quantization']['relative_l2'])
        self.assertEqual(result['total']['absolute_rms'],1.)


if __name__ == '__main__':
    unittest.main()
