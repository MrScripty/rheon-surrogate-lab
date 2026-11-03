"""Literal numerical oracles for the scalar pressure retention boundary."""
import math
import unittest

import numpy as np

from pressure_scalar_diagnostics import equation, error_summary, measure


class PressureStatisticsOracles(unittest.TestCase):
    def sample(self):
        return dict(base=np.array([1.,3.]),reference=np.array([1.,3.]),rhs=np.zeros(2),
            native_input_residual=np.zeros(2),cell_size_m=1.,rho=1.)

    def test_sealed_flux_literal(self):
        # The captured operator is -Laplacian: outward face differences
        # yield a positive diagonal, not the opposite Laplacian convention.
        np.testing.assert_array_equal(equation(np.array([[[0.,1.,4.]]]),[1.,1.,1.],1.),[[[-1.,-2.,3.]]])

    def test_anisotropic_flux_literal(self):
        value=np.arange(8.,dtype=np.float64).reshape(2,2,2)
        expected=np.array([[[-3.5,.5],[-1.5,2.5]],[[-2.5,1.5],[-.5,3.5]]])
        np.testing.assert_array_equal(equation(value,[.5,1.,2.],2.),expected)

    def test_pressure_gauge_is_separate_from_error(self):
        result=measure(np.array([5.,7.]),self.sample(),[2,1,1])
        self.assertEqual(result['pressure_mean_free_error']['rms'],0.)
        self.assertEqual(result['pressure_gauge_offset_vs_reference'],4.)
        self.assertEqual(result['prediction_minus_supplied_base']['rms'],0.)

    def test_nonuniform_error_and_zero_baseline(self):
        result=measure(np.array([1.,5.]),self.sample(),[2,1,1])
        error=result['pressure_mean_free_error']
        self.assertEqual(error['rms'],1.)
        self.assertEqual(error['mae'],1.)
        self.assertEqual(error['maximum_absolute'],1.)
        self.assertEqual(error['relative_l2'],1.)
        self.assertEqual(result['absolute_pressure_RMS_improvement_vs_supplied_base'],-1.)
        self.assertIsNone(result['pressure_RMS_ratio_vs_supplied_base'])
        self.assertFalse(result['pressure_better_than_supplied_base'])

    def test_zero_target_floor_and_percentile(self):
        result=error_summary(np.array([1.,2.,3.,10.]),np.zeros(4))
        self.assertTrue(result['denominator_floor_active'])
        self.assertAlmostEqual(result['absolute_p95'],8.95)
        self.assertTrue(math.isfinite(result['relative_l2']))

    def test_reject_nonfinite_and_wrong_size(self):
        with self.assertRaises(ValueError):
            measure(np.array([np.nan,1.]),self.sample(),[2,1,1])
        with self.assertRaises(ValueError):
            measure(np.array([1.,2.,3.]),self.sample(),[2,1,1])


if __name__=='__main__':
    unittest.main()
