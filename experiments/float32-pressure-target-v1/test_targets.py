import copy
from pathlib import Path
import tempfile
import unittest

from targets import assess, calibrated_relative_target, load


class Float32TargetTests(unittest.TestCase):
    def setUp(self):
        self.profile = load(Path(__file__).with_name('configs')/'development.json')

    def test_rounding_margin_is_selected_from_training_peak(self):
        self.assertEqual(calibrated_relative_target(4.7665255694152723e-7, self.profile['calibration']),1e-6)
        for invalid in (float('nan'),float('inf'),-1,True):
            with self.assertRaises(ValueError):
                calibrated_relative_target(invalid,self.profile['calibration'])

    def test_only_residual_flag_changes_and_measured_values_are_preserved(self):
        quality = {'qualified':False,'failures':['original_rhs_full_residual'],'residual_l2':4e-9,'rhs_l2':.01}
        previous = copy.deepcopy(quality)
        self.assertTrue(assess(quality,self.profile)['qualified'])
        self.assertEqual(quality,previous)
        quality['residual_l2']=2e-8
        self.assertFalse(assess(quality,self.profile)['qualified'])

    def test_physical_and_invalid_prediction_failures_never_become_passes(self):
        for reason in ('predicted_divergence','actual_f32_divergence','net_volume_flux','wall_normal_flux','courant'):
            quality={'qualified':False,'failures':['original_rhs_full_residual',reason],'residual_l2':4e-9,'rhs_l2':.01}
            self.assertEqual(assess(quality,self.profile)['failures'],[reason])
        self.assertFalse(assess({'qualified':False,'failures':['prediction_nonfinite']},self.profile)['qualified'])

    def test_zero_rhs_keeps_original_absolute_floor(self):
        quality={'qualified':True,'failures':[],'residual_l2':1e-12,'rhs_l2':0.0}
        self.assertTrue(assess(quality,self.profile)['qualified'])
        quality.update(qualified=False,failures=['original_rhs_full_residual'],residual_l2=2e-12)
        self.assertFalse(assess(quality,self.profile)['qualified'])

    def test_profile_rejects_nonfinite_limits_and_changed_absolute_floor(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'profile.json'
            for value in (float('nan'),float('inf'),-1,True):
                path.write_text(json.dumps({**self.profile,'relative_residual':value}))
                with self.assertRaises(ValueError):load(path)
            path.write_text(json.dumps({**self.profile,'absolute_residual':1e-3}))
            with self.assertRaises(ValueError):load(path)


if __name__ == '__main__':
    unittest.main()
