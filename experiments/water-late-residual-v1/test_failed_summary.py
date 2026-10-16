"""Retain zero-frame native setup rejection instead of hiding it with report IO."""
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import tempfile
import unittest

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


if __name__ == '__main__':
    unittest.main()
