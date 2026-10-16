"""Unknown or unrecovered failures cannot be presented as successful execution."""
import copy
import tempfile
from pathlib import Path
import unittest

from finalize import digest,verify_commands


class Lifecycle(unittest.TestCase):
    def test_failure_requires_exact_receipt_and_successful_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'failed-timing.json';path.write_text('retained failure')
            failed=str(path);recovery=str(Path(directory)/'recovery-timing.json')
            commands={failed:{'status':'failed','returncode':1,'child_reaped':True},
                recovery:{'status':'completed','returncode':0,'child_reaped':True}}
            exception={'failed_timing':failed,'timing_sha256':digest(path),'recovery_timing':recovery}
            self.assertEqual(verify_commands(commands,[exception]),{failed})
            with self.assertRaises(AssertionError):verify_commands(commands,[])
            bad=copy.deepcopy(commands);bad[recovery]['returncode']=1
            with self.assertRaises(AssertionError):verify_commands(bad,[exception])
            bad=copy.deepcopy(commands);bad[failed]['child_reaped']=False
            with self.assertRaises(AssertionError):verify_commands(bad,[exception])
            path.write_text('changed receipt')
            with self.assertRaises(AssertionError):verify_commands(commands,[exception])

    def test_terminal_success_requires_zero_exit_and_reaping(self):
        command={'status':'completed','returncode':0,'child_reaped':True}
        self.assertEqual(verify_commands({'success':command},[]),set())
        for key,value in [('returncode',1),('child_reaped',False),('status','running')]:
            with self.assertRaises(AssertionError):verify_commands({'success':{**command,key:value}},[])


if __name__=='__main__':unittest.main()
