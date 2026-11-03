"""Literal scalar oracles;no inference,field arrays or model runtime."""
import unittest
from portfolio_statistics import distribution, grouped


class ScalarStatistics(unittest.TestCase):
    def test_even_distribution_and_tail_interpolation(self):
        result=distribution([10.,1.,3.,2.])
        self.assertEqual([result[k] for k in ('samples','best','worst','mean','median')],[4,1.,10.,4.,2.5])
        self.assertAlmostEqual(result['p95'],8.95)

    def test_singleton_and_nonfinite_rejection(self):
        self.assertEqual(distribution([7.])['median'],7.)
        for values in ([],[float('nan')],[float('inf')]):
            with self.assertRaisesRegex(ValueError,'Nonempty finite scalar observations required'):distribution(values)

    def test_cadence_and_forcing_frequency_are_separate_dimensions(self):
        rows=[dict(identity=dict(identifier='a',cadence=60,force_hz=2),values={'error':1.}),
              dict(identity=dict(identifier='b',cadence=60,force_hz=4),values={'error':3.}),
              dict(identity=dict(identifier='a',cadence=30,force_hz=2),values={'error':5.})]
        result=grouped(rows,dict(output_cadence=lambda r:r['identity']['cadence'],forcing_hz=lambda r:r['identity']['force_hz']))
        self.assertEqual(result['output_cadence']['60']['metrics']['error']['mean'],2.)
        self.assertEqual(result['forcing_hz']['2']['metrics']['error']['median'],3.)
        self.assertEqual(result['forcing_hz']['2']['independent_scene_ids'],1)

    def test_literal_staggered_divergence_pressure_gauge_and_tracer_integral(self):
        import math
        import numpy as np
        from spatial_field_statistics import measure
        truth=dict(pressure=np.array([0.,2.]),tracer=np.zeros(2),vx=np.zeros(3),vy=np.zeros(4),vz=np.zeros(4))
        output=dict(pressure=np.array([0.,4.]),tracer=np.array([-1.,3.]),vx=np.array([0.,1.,2.]),vy=np.zeros(4),vz=np.zeros(4))
        result=measure(output,truth,[2,1,1],[1.,1.,1.])
        self.assertEqual(result['pressure_maximum_cell_absolute_error'],1.)
        self.assertAlmostEqual(result['vx_absolute_rms_error'],math.sqrt(5/3))
        self.assertEqual(result['vx_maximum_cell_absolute_error'],2.)
        self.assertEqual(result['divergence_rms_per_s'],1.)
        self.assertEqual(result['divergence_error_maximum_absolute_per_s'],1.)
        self.assertEqual(result['boundary_normal_velocity_maximum_absolute'],2.)
        self.assertEqual(result['tracer_volume_integral'],2.)
        self.assertEqual(result['tracer_negative_cells'],1)
        output['pressure']+=7.
        self.assertEqual(measure(output,truth,[2,1,1],[1.,1.,1.])['pressure_maximum_cell_absolute_error'],1.)


if __name__=='__main__':unittest.main()
