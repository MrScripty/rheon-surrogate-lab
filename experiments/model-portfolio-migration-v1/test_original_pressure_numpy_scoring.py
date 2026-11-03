"""Literal closed-boundary stencils and gauge invariance, independent of Torch."""
import unittest
import numpy as np

from reduce_original_train_pressure_accuracy import equation,metrics


class NumPyPressureScoring(unittest.TestCase):
    def test_constant_pressure_has_no_closed_boundary_flux(self):
        pressure = np.full((1,1,2,3,4),7.,dtype='<f8')
        np.testing.assert_array_equal(equation(pressure,.125,2.),np.zeros_like(pressure))

    def test_three_cell_line_has_literal_boundary_stencil(self):
        pressure = np.asarray([2.,3.,5.],dtype='<f8').reshape(1,1,1,1,3)
        expected = np.asarray([-.0625,-.0625,.125],dtype='<f8').reshape(1,1,1,1,3)
        np.testing.assert_array_equal(equation(pressure,.125,2.),expected)

    def test_corner_impulse_has_three_positive_diagonal_faces(self):
        pressure = np.zeros((1,1,2,2,2),dtype='<f8');pressure[0,0,0,0,0] = 1.
        expected = np.zeros_like(pressure);expected[0,0,0,0,0] = .1875
        expected[0,0,1,0,0] = expected[0,0,0,1,0] = expected[0,0,0,0,1] = -.0625
        np.testing.assert_array_equal(equation(pressure,.125,2.),expected)

    def test_uniform_gauge_shift_does_not_change_equation_or_pressure_error(self):
        reference = np.arange(8,dtype='<f8').reshape(1,1,2,2,2)
        rhs = equation(reference,.125,2.)
        self.assertEqual(metrics(reference+16.,reference,rhs,.125,2.),dict(
            common_equation_relative_l2=0.,pressure_mean_free_relative_l2=0.,pressure_absolute_rms=0.,pressure_absolute_max=0.))


if __name__ == '__main__':
    unittest.main()
