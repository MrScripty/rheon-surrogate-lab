"""A rejected native child must still leave its failure and reaping receipt."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB/'experiments/water-residual-learning-v1'))
import rollout


class RejectedPipe:
    closed = False

    def write(self, value):
        raise BrokenPipeError('native rejected invocation')

    def close(self):
        self.closed = True
        raise BrokenPipeError('buffered stdin close')


class ExitedChild:
    pid = 12345
    returncode = 2

    def __init__(self):
        self.stdin = RejectedPipe()
        self.stdout = open('/dev/null', 'rb')

    def poll(self):
        return self.returncode


class EarlyExit(unittest.TestCase):
    def test_broken_stdin_cleanup_preserves_original_failure_and_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root/'binary'
            binary.write_bytes(b'rejected native test fixture')
            child = ExitedChild()
            with patch.object(rollout.subprocess, 'Popen', return_value=child):
                with self.assertRaisesRegex(BrokenPipeError, 'native rejected invocation'):
                    rollout.run(binary, {}, root/'output', control=True)
            receipt = json.loads((root/'output/receipt.json').read_text())
            self.assertEqual(receipt['status'], 'failed')
            self.assertEqual(receipt['error'], 'BrokenPipeError: native rejected invocation')
            self.assertTrue(receipt['native_process']['reaped'])
            self.assertEqual(receipt['native_process']['returncode'], 2)
            self.assertTrue(child.stdout.closed)


if __name__ == '__main__':
    unittest.main()
