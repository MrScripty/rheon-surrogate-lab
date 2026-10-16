"""Frozen final profiles reject drift before any worker or native dispatch."""
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

LAB = Path(__file__).resolve().parents[2]
spec = spec_from_file_location('stopping_study_owner', Path(__file__).with_name('run.py'))
owner = module_from_spec(spec)
spec.loader.exec_module(owner)


class FrozenSelection(unittest.TestCase):
    def test_changed_model_or_profile_digest_rejected_before_dispatch(self):
        config_path = LAB/'experiments/water-stopping-floor-v1/configs/validation.json'
        if not config_path.exists():
            self.skipTest('requires the retained real frozen validation inputs')
        config = json.loads(config_path.read_text())
        actual_digest = owner.digest
        targets = [LAB/config['models'][0]['artifacts']/'weights.npz', LAB/config['profile']]
        for target in targets:
            with self.subTest(target=target.name):
                def changed(path):
                    return '0'*64 if Path(path) == target else actual_digest(path)
                with patch.object(owner, 'digest', side_effect=changed), \
                     patch.object(owner, 'command') as command, \
                     patch.object(sys, 'argv', ['run.py', '--config', str(config_path), '--phase', 'validation']):
                    with self.assertRaises(AssertionError):
                        owner.main()
                    command.assert_not_called()


if __name__ == '__main__':
    unittest.main()
