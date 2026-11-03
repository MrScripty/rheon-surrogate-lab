"""Original sampler digests and isolated train/validation rejection."""
import copy
import hashlib
from pathlib import Path
import unittest

from captured_methods_pressure_review import read,sample_order

ROOT = Path(__file__).resolve().parents[2]


class MethodsSamplerTest(unittest.TestCase):
    def fixture(self,project):
        cfg = read(ROOT/'runs'/project/'config.json')
        material = read(ROOT/'runs/model-portfolio-migration-v1/discovery/pressure-material-01.json')['models']
        selected = next(m for m in material if ROOT/m['project'] == ROOT/'runs'/project)
        training = (ROOT/selected['restore']).parent/'train'
        split = read(training/'splits.json')
        parents = [dict(p,entry=p.get('entry',dict(trajectory_id=p.get('trajectory_id'),split=p.get('split'),dt_s=p.get('dt_s'))))
                   for p in split['parents']]
        receipt = read(training/'receipt.json')
        return cfg,parents,receipt

    def test_original_equal_tier_order(self):
        cfg,parents,receipt = self.fixture('pressure-methods-v1/domain-expanded-project-01')
        order,_ = sample_order(parents,cfg,'equal_tiers')
        self.assertEqual(hashlib.sha256(order.tobytes()).hexdigest(),receipt['closure']['sample_order_sha256'])

    def test_original_retained_small_stream(self):
        cfg,parents,receipt = self.fixture('pressure-methods-v2/v2-retain24-project-02')
        order,small = sample_order(parents,cfg,'small_replay')
        self.assertEqual(order.shape,(2400,1,2))
        self.assertEqual(small.shape,(1600,2))
        self.assertEqual(hashlib.sha256(order.tobytes()).hexdigest(),receipt['closure']['sample_order_sha256'])
        saved = read(ROOT/'runs/pressure-methods-v2/v2-retain24-study-02/train/history.json')['current_feedback']['sampling']
        self.assertEqual(hashlib.sha256(small.tobytes()).hexdigest(),saved['small_subsequence_sha256'])

    def test_validation_parent_reaches_intended_diagnostic(self):
        cfg,parents,_ = self.fixture('pressure-methods-v1/domain-expanded-project-01')
        sample_order(parents,cfg,'equal_tiers')
        changed = copy.deepcopy(parents)
        changed[0]['entry']['split'] = 'validation'
        with self.assertRaisesRegex(ValueError,'^Methods sampler requires serial train-only parents$'):
            sample_order(changed,cfg,'equal_tiers')


if __name__ == '__main__':
    unittest.main()
