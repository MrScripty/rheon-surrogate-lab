"""Local real-record recovery guards; no optimizer, HTTP or native simulations."""
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

LAB = Path(__file__).resolve().parents[2]
spec = spec_from_file_location('direction_recovery_owner', LAB/'experiments/water-coverage-feedback-v1/run.py')
owner = module_from_spec(spec)
spec.loader.exec_module(owner)


class Recovery(unittest.TestCase):
    def setUp(self):
        self.sequence_path = LAB/'experiments/water-late-residual-v1/configs/sequence.json'
        self.sequence = owner.read(self.sequence_path)
        self.root = LAB/self.sequence['root']
        self.reports = LAB/'docs/plans/water-late-residual-v1/reports'
        if not (self.reports/'comparison.json').exists():
            self.skipTest('requires retained completed local comparison records')
        self.bindings = owner.references(self.sequence)

    def test_missing_capture_duration_refuses_before_any_managed_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence = {**self.sequence, 'root': str(root)}
            path = root/'sequence.json'
            owner.write(path, sequence)
            owner.write(root/'capture.json', {'source_commit': 'fixture'})
            argv = ['run.py', '--sequence', str(path), '--reports', str(root),
                '--managed-controller', 'must-not-open', '--torch-python', sys.executable]
            with patch.object(sys, 'argv', argv), patch.object(owner, 'Observer'), \
                    patch.object(owner, 'Watch', return_value=MagicMock()), \
                    patch.object(owner, 'command', side_effect=AssertionError('managed work must not start')):
                with self.assertRaisesRegex(ValueError, 'capture duration'):
                    owner.main()
            self.assertFalse((root/'study-status.json').exists())

    def test_real_completed_comparison_reuses_six_drained_runs(self):
        receipt = owner.recovery_evidence(self.root, self.reports, self.sequence, self.bindings, self.sequence_path)
        self.assertEqual(len(receipt['managed_run_ids_reused']), 6)
        self.assertFalse(receipt['managed_work_repeated'])

    def test_changed_source_and_attempted_final_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            reports = Path(directory)
            original = owner.read(self.reports/'comparison.json')
            changed = json.loads(json.dumps(original))
            changed['sources'][next(iter(changed['sources']))] = '0'*64
            owner.write(reports/'comparison.json', changed)
            with self.assertRaisesRegex(ValueError, 'source changed'):
                owner.recovery_evidence(self.root, reports, self.sequence, self.bindings, self.sequence_path)
            owner.write(reports/'comparison.json', original)
            owner.write(reports/'final-selection.json', {})
            with self.assertRaisesRegex(ValueError, 'cannot repeat'):
                owner.recovery_evidence(self.root, reports, self.sequence, self.bindings, self.sequence_path)


if __name__ == '__main__':
    unittest.main()
