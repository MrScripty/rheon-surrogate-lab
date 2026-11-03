"""Real retained sampler fixtures and single-condition negative evidence."""
import copy
from pathlib import Path
import unittest

from review_pressure_capacity_fits import audit_curriculum, audit_fundamental, read

ROOT = Path(__file__).resolve().parents[2]


class CapacityReviewTest(unittest.TestCase):
    def fundamental(self,folder):
        project = ROOT/'runs'/folder
        material = read(ROOT/'runs/model-portfolio-migration-v1/discovery/pressure-material-01.json')
        entry = next(m for m in material['models'] if ROOT/m['project'] == project)
        stats = read((ROOT/entry['restore']).parent/'train/receipt.json')['fine_tuning_statistics']['fundamental_bank']
        return read(project/'fine-tuning.json'),stats

    def test_actual_eight_shape_private_order(self):
        fine,stats = self.fundamental('pressure-fundamental-bank-fit-v1/shape-isolation-project-01')
        result = audit_fundamental(fine,stats,1600,aggregate=False)
        self.assertEqual(result['sampling_order_sha256'],'d0ea221a7e03a83673ece7fc9cdf4f5212bb54f3023511f93d95c84e45647899')
        self.assertEqual(result['student_network_forwards'],6400)
        self.assertEqual(result['exposures_per_case'],[200]*8)

    def test_changed_exposure_reaches_own_diagnostic(self):
        fine,stats = self.fundamental('pressure-fundamental-bank-fit-v1/shape-isolation-project-01')
        audit_fundamental(fine,stats,1600,aggregate=False)
        changed = copy.deepcopy(stats)
        changed['optimizer_exposures'][0] += 1
        with self.assertRaisesRegex(ValueError,'^Fundamental optimizer exposures differ$'):
            audit_fundamental(fine,changed,1600,aggregate=False)

    def test_aggregate_updates_are_not_example_count(self):
        fine,stats = self.fundamental('pressure-balanced-shapes-v1/balanced-shapes-project-01')
        result = audit_fundamental(fine,stats,200,aggregate=True)
        self.assertEqual((result['optimizer_updates'],result['optimizer_examples']),(200,1600))
        changed = copy.deepcopy(stats)
        changed['optimizer_examples'] = 200
        with self.assertRaisesRegex(ValueError,'^Fundamental optimizer example budget differs$'):
            audit_fundamental(fine,changed,200,aggregate=True)

    def test_changed_curriculum_category_exposure(self):
        project = ROOT/'runs/pressure-single-mode-fit-v1/frequency-curriculum-project-01'
        history = read(ROOT/'runs/pressure-single-mode-fit-v1/frequency-curriculum-study-01/train/history.json')
        config = read(project/'config.json')
        result = audit_curriculum(config,history)
        self.assertEqual(result['mode_category_exposures'],dict(fundamental=579,other_low=542,high=379))
        changed = copy.deepcopy(history)
        changed['supplemental_feedback']['actual_mode_category_exposures']['fundamental'] += 1
        with self.assertRaisesRegex(ValueError,'^Supplemental static frequency category exposures differ$'):
            audit_curriculum(config,changed)


if __name__ == '__main__':
    unittest.main()
