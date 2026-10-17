"""Real subprocesses prove SIGTERM drains the supervisor's owned workers.

The producer fixture only waits; this test claims lifecycle behavior, not model
or solver correctness. Numerical behavior is covered by the real CPU campaign.
"""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest


class SupervisorTests(unittest.TestCase):
    def test_sigterm_waits_for_both_workers_and_records_cancelled(self):
        with tempfile.TemporaryDirectory(prefix='pressure-sweep-lifecycle-')as temporary:
            root=Path(temporary)
            for name in ['run_all.py','material.py']:shutil.copyfile(Path(__file__).with_name(name),root/name)
            (root/'worker.py').write_text("import argparse,os,time\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument('--material');p.add_argument('--id');p.add_argument('--cpu');p.add_argument('--resume',action='store_true');a=p.parse_args();Path(a.material).with_name(a.id+'.pid').write_text(str(os.getpid()))\nwhile True:time.sleep(.1)\n")
            (root/'runtime.json').write_text(json.dumps(dict(python=sys.executable,site_packages='')))
            (root/'material.json').write_text(json.dumps(dict(root=str(root),models=[{'id':'A'},{'id':'B'}],config=dict(runtime='runtime.json',reserve_memory_bytes=0,worker_memory_bytes=1,maximum_workers=2))))
            child=subprocess.Popen([sys.executable,str(root/'run_all.py'),'--material',str(root/'material.json')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            try:
                deadline=time.monotonic()+10
                while len(list(root.glob('*.pid')))<2 and time.monotonic()<deadline:time.sleep(.05)
                self.assertEqual(len(list(root.glob('*.pid'))),2,'Both workers must be active before cancellation')
                pids=[int(p.read_text())for p in root.glob('*.pid')]
                child.send_signal(signal.SIGTERM);output,_=child.communicate(timeout=15)
                receipt=json.loads((root/'execution.json').read_text())
                self.assertEqual(receipt['status'],'cancelled',output)
                self.assertTrue(receipt['all_owned_children_reaped'],output)
                for pid in pids:
                    with self.assertRaises(ProcessLookupError):os.kill(pid,0)
            finally:
                if child.poll()is None:
                    child.terminate()
                    try:child.communicate(timeout=5)
                    except subprocess.TimeoutExpired:child.kill();child.communicate(timeout=5)


if __name__=='__main__':unittest.main()
