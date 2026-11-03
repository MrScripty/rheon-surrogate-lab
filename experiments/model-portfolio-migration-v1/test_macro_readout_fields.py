"""Literal physical score oracles; no learned tensors or model execution."""
import math
import unittest

import numpy as np
from macro_readout_fields import physical_scores


class Scores(unittest.TestCase):
    def setUp(self):
        self.scene = dict(counts=[2,1,1],cells_per_metre=1.)
        self.norm = dict(relative_denominator_rms_floor=dict(pressure=.25,velocity=.25,tracer=.25),equation_rhs_floor=1e-30)
        self.truth = dict(pressure=np.array([0.,2.]),rhs=np.array([-.002,.002]),
            vx=np.full(3,2.,dtype=np.float32),vy=np.full(4,2.,dtype=np.float32),vz=np.full(4,2.,dtype=np.float32),
            tracer=np.array([0.,2.],dtype=np.float32))

    def test_half_fields_have_half_relative_error_in_each_physical_metric(self):
        output = {name:value*.5 for name,value in self.truth.items() if name != 'rhs'}
        scores = physical_scores(output,self.truth,self.scene,self.norm)
        for name in ('pressure','velocity','tracer','equation'):
            self.assertAlmostEqual(scores['relative'][name],.5)
            self.assertFalse(scores['target_floor_active'][name])
        self.assertEqual(scores['absolute_rms']['velocity'],1.)
        self.assertAlmostEqual(scores['absolute_rms']['tracer'],math.sqrt(.5))

    def test_pressure_gauge_offset_does_not_hide_its_absolute_diagnostic(self):
        output = {name:value.copy() for name,value in self.truth.items() if name != 'rhs'}
        output['pressure'] += 3.
        scores = physical_scores(output,self.truth,self.scene,self.norm)
        self.assertEqual(scores['relative']['pressure'],0.)
        self.assertEqual(scores['relative']['equation'],0.)
        self.assertEqual(scores['pressure_cell_zero_absolute_error'],3.)

    def test_zero_target_floors_preserve_absolute_error_and_unknown_raw_relative(self):
        truth = {name:np.zeros_like(value) for name,value in self.truth.items()}
        output = {name:np.ones_like(value) for name,value in truth.items() if name != 'rhs'}
        scores = physical_scores(output,truth,self.scene,self.norm)
        self.assertEqual(scores['absolute_rms']['velocity'],1.)
        self.assertEqual(scores['relative']['velocity'],4.)
        self.assertTrue(scores['target_floor_active']['velocity'])
        self.assertIsNone(scores['raw_relative']['velocity'])
        self.assertEqual(scores['relative']['pressure'],0.)
        self.assertTrue(scores['target_floor_active']['pressure'])


if __name__ == '__main__':
    unittest.main()
