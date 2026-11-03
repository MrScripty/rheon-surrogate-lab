"""Analytic metric oracles; no neural runtime or measured model execution."""
import math
import unittest

import numpy as np

from joint_generator_gpu_benchmark import FIELDS, score


class FullStateScore(unittest.TestCase):
    def setUp(self):
        # The nonconstant two-cell x mode has Neumann graph eigenvalue2.
        pressure = np.tile(np.array([-1.,1.],dtype='<f8'),4)
        self.scene = dict(counts=[2,2,2],cells_per_metre=2)
        self.reference = dict(pressure=pressure,rhs=.001*pressure,
            vx=np.ones(12,dtype='<f4'),vy=np.ones(12,dtype='<f4'),vz=np.ones(12,dtype='<f4'),
            tracer=np.ones(8,dtype='<f4'))
        self.output = {n:self.reference[n].copy() for n in FIELDS}

    def test_exact_reference_and_constant_pressure_gauge(self):
        self.output['pressure'] += 7.
        result = score(np,self.output,self.reference,self.scene)
        self.assertEqual(result['pressure_relative_l2'],0.)
        self.assertEqual(result['equation_relative_l2'],0.)
        self.assertEqual(result['velocity_relative_l2'],0.)
        self.assertEqual(result['tracer_relative_l2'],0.)

    def test_known_independent_velocity_and_tracer_errors(self):
        self.output['vx'][0] += 1.
        self.output['tracer'][0] = 0.
        result = score(np,self.output,self.reference,self.scene)
        self.assertAlmostEqual(result['velocity_relative_l2'],1/6)
        self.assertAlmostEqual(result['vx_relative_l2'],1/math.sqrt(12))
        self.assertAlmostEqual(result['tracer_relative_l2'],1/math.sqrt(8))
        self.assertEqual(result['pressure_relative_l2'],0.)

    def test_zero_reference_with_nonzero_error_is_explicitly_undefined(self):
        self.reference['tracer'][:] = 0.
        result = score(np,self.output,self.reference,self.scene)
        self.assertIsNone(result['tracer_relative_l2'])
        self.assertEqual(result['tracer_absolute_rms'],1.)


if __name__ == '__main__':
    unittest.main()
