"""Retain zero-frame native setup rejection instead of hiding it with report IO."""
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

LAB = Path(__file__).resolve().parents[2]
spec = spec_from_file_location('native_scaling_summary', LAB/'experiments/water-inference-budget-v1/scaling.py')
owner = module_from_spec(spec)
spec.loader.exec_module(owner)


class FailedSummary(unittest.TestCase):
    def test_failed_native_without_frames_retains_failure_and_reaping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = {'status': 'failed', 'result': None,
                'native_process': {'returncode': 1, 'reaped': True}, 'worst_ai_relative_residual': 0}
            (root/'receipt.json').write_text(json.dumps(receipt))
            result = owner.summarize(root)
            self.assertEqual(result['status'], 'failed')
            self.assertTrue(result['native_process']['reaped'])
            self.assertFalse(result['all_ai_steps_qualified'])
            self.assertEqual(result['timing_samples'], 0)

    def test_completed_native_without_frame_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'receipt.json').write_text(json.dumps({'status': 'completed'}))
            with self.assertRaisesRegex(ValueError, 'completed.*frames'):
                owner.summarize(root)

    def test_incomplete_trajectory_requests_reconciliation(self):
        spec = spec_from_file_location('complete_trajectory_summary', LAB/'experiments/water-multigrid-learning-v1/trajectory_summary.py')
        trajectory = module_from_spec(spec)
        spec.loader.exec_module(trajectory)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'scene-00/reference-0'
            source.mkdir(parents=True)
            receipt = {'native_process': {'reaped': True}, 'result': None,
                'scene': {'steps': 2, 'warmup_steps': 0}}
            (source/'receipt.json').write_text(json.dumps(receipt))
            (source/'frames.jsonl').write_text('')
            report = {'configuration': {'scenes': [receipt['scene']]}, 'binary_sha256': 'b'*64,
                'models': [{'name': 'reference'}], 'records': [{'model': 'reference', 'scene': 0,
                    'repetition': 0, 'receipt_sha256': trajectory.digest(source/'receipt.json')}]}
            (root/'report.json').write_text(json.dumps(report))
            with patch.object(sys, 'argv', ['summary.py', '--studies', str(root), '--output', str(root/'summary.json')]):
                with self.assertRaisesRegex(ValueError, 'incomplete.*reconcil'):
                    trajectory.main()
            self.assertFalse((root/'summary.json').exists())


if __name__ == '__main__':
    unittest.main()
