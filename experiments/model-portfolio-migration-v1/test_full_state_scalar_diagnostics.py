"""Physical and temporal scalar oracles without any learned execution."""
import unittest
import numpy as np

from full_state_scalar_diagnostics import measure, vorticity, shaped


def zero():
    return dict(pressure=np.zeros((2,2,2)),tracer=np.zeros((2,2,2)),
        vx=np.zeros((2,2,3)),vy=np.zeros((2,3,2)),vz=np.zeros((3,2,2)),rhs=np.zeros((2,2,2)))


class FullStateOracles(unittest.TestCase):
    def test_equal_states_and_zero_denominators(self):
        result=measure(zero(),zero(),zero(),[2,2,2],[1.,1.,1.],2.)
        for value in result['fields'].values():
            self.assertEqual(value['error']['rms'],0.)
            self.assertTrue(value['error']['denominator_floor_active'])
        self.assertEqual(result['state_quantities']['kinetic_energy_error_joule'],0.)

    def test_divergence_boundary_and_integral(self):
        output=zero();output['vx'][:]=np.arange(3)[None,None,:];output['tracer'][:]=2.
        result=measure(output,zero(),zero(),[2,2,2],[1.,1.,1.],2.)
        self.assertEqual(result['spatial']['divergence_rms_per_s'],1.)
        self.assertEqual(result['spatial']['boundary_normal_velocity_maximum_absolute'],2.)
        self.assertEqual(result['spatial']['tracer_volume_integral'],16.)
        self.assertEqual(result['state_quantities']['prediction']['kinetic_energy_joule'],10.)

    def test_solid_rotation_vorticity(self):
        output=zero();output['vx'][:]=-np.array([.5,1.5])[None,:,None]
        output['vy'][:]=np.array([.5,1.5])[None,None,:]
        vort=vorticity(shaped(output,[2,2,2]),[1.,1.,1.])
        np.testing.assert_array_equal(vort['z'],np.full((2,2,2),2.))
        np.testing.assert_array_equal(vort['x'],np.zeros((2,2,2)))
        np.testing.assert_array_equal(vort['y'],np.zeros((2,2,2)))

    def test_lead_change_uses_preceding_output_and_initial_baseline(self):
        current,prior,output,truth=zero(),zero(),zero(),zero()
        prior['tracer'][:]=1.;output['tracer'][:]=3.;truth['tracer'][:]=4.
        result=measure(output,truth,current,[2,2,2],[1.,1.,1.],2.,prior,prior)
        tracer=result['fields']['tracer']
        self.assertEqual(tracer['prediction_minus_input']['rms'],3.)
        self.assertEqual(tracer['prediction_minus_preceding_output']['rms'],2.)
        self.assertEqual(tracer['reference_minus_preceding_reference']['rms'],3.)
        self.assertEqual(tracer['RMS_improvement_vs_unchanged_input'],3.)

    def test_pressure_gauge_and_negative_tracer(self):
        output=zero();output['pressure'][:]=5.;output['tracer'].flat[0]=-1.
        result=measure(output,zero(),zero(),[2,2,2],[1.,1.,1.],2.)
        self.assertEqual(result['fields']['pressure']['error']['rms'],0.)
        self.assertEqual(result['pressure_gauge_offset_vs_reference'],5.)
        self.assertEqual(result['spatial']['tracer_negative_cells'],1)

    def test_direction_is_distinct_from_speed(self):
        output,truth=zero(),zero();output['vx'][:]=1.;truth['vy'][:]=1.
        result=measure(output,truth,zero(),[2,2,2],[1.,1.,1.],2.)
        self.assertEqual(result['cell_centered_speed_error']['rms'],0.)
        self.assertEqual(result['cell_centered_velocity_direction']['defined_cells'],8)
        self.assertEqual(result['cell_centered_velocity_direction']['angle_degrees']['mean'],90.)

    def test_nonfinite_output_rejected(self):
        output=zero();output['vy'].flat[0]=np.inf
        with self.assertRaises(ValueError):measure(output,zero(),zero(),[2,2,2],[1.,1.,1.],2.)


if __name__=='__main__':
    unittest.main()
