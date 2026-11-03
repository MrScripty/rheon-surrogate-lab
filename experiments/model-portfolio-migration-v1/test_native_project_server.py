"""Reboot recovery admits only demonstrably stopped prior-boot owners."""
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import native_project_server as subject


class RebootRecoveryTests(unittest.TestCase):
    def check(self, *, ready=100, pid_exists=False, pg_returncode=3):
        def read(path):
            return 'cpu 1\nbtime 200\n' if str(path) == '/proc/stat' else 'new-boot-id\n'
        previous=dict(ready_unix=ready,launcher_pid=999991,app_pid=999992)
        with patch.object(Path,'read_text',read), patch.object(Path,'exists',return_value=pid_exists), \
                patch.object(subject.subprocess,'run',return_value=subprocess.CompletedProcess([],pg_returncode)):
            return subject.require_prior_boot(previous,Path('/owned/cluster'))

    def test_prior_boot_stopped_owner_is_explicit_not_mutation_replay(self):
        result=self.check()
        self.assertEqual(result['host_boot_unix'],200)
        self.assertFalse(result['native_mutation_replayed'])

    def test_same_boot_cannot_be_mistaken_for_reboot(self):
        with self.assertRaisesRegex(ValueError,'earlier host boot'):self.check(ready=200)

    def test_reused_pid_requires_reconciliation(self):
        with self.assertRaisesRegex(ValueError,'identifier exists'):self.check(pid_exists=True)

    def test_live_or_unknown_cluster_is_refused(self):
        for code in (0,1,4):
            with self.subTest(code=code),self.assertRaisesRegex(ValueError,'cluster to be stopped'):
                self.check(pg_returncode=code)


if __name__ == '__main__':
    unittest.main()
