"""Real process ownership on failed exit and deadline, isolated temporary roots."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import rollout


class NativeOwnership(unittest.TestCase):
    def helper(self,root,body):
        path=root/'native';path.write_text('#!'+sys.executable+'\n'+body);path.chmod(0o700);return path

    def test_failed_native_exit_retains_failure_and_reaps(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);binary=self.helper(root,'import sys\nsys.exit(3)\n')
            with self.assertRaisesRegex(RuntimeError,'did not complete'):
                rollout.run(binary,{},root/'record',control=True)
            result=json.loads((root/'record/receipt.json').read_text())
            self.assertEqual(result['status'],'failed');self.assertTrue(result['native_process']['reaped'])
            self.assertEqual(result['native_process']['returncode'],3)

    def test_deadline_terminates_and_reaps_owned_native(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);binary=self.helper(root,'import time\ntime.sleep(30)\n')
            with self.assertRaisesRegex(TimeoutError,'deadline'):
                rollout.run(binary,{},root/'record',control=True,timeout=.05)
            result=json.loads((root/'record/receipt.json').read_text())
            self.assertEqual(result['status'],'failed');self.assertTrue(result['native_process']['terminated'])
            self.assertTrue(result['native_process']['reaped'])
            self.assertIsNotNone(result['native_process']['returncode'])


if __name__=='__main__':unittest.main()
