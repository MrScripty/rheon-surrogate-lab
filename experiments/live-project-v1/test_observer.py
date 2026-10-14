import json
from pathlib import Path
import tempfile
import threading
import unittest
import subprocess
import sys

from observer import Observer, Watch, atomic


class ObserverTests(unittest.TestCase):
    def test_atomic_failure_preserves_previous_publication(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'activity.json'; atomic(path,{'sequence':1})
            with self.assertRaises(ValueError): atomic(path,{'bad':float('nan')})
            self.assertEqual(json.loads(path.read_text()),{'sequence':1})
            self.assertEqual([p.name for p in Path(root).iterdir()],['activity.json'])

    def test_latest_preview_carries_terminal_state_and_session(self):
        with tempfile.TemporaryDirectory() as root:
            owner=Observer(root)
            owner.publish('Generation','Frame ready',1,2,preview={'tracer':[0.0],'label':'actual fixture'})
            owner.publish('Generation','Failed qualification',1,2,state='failed')
            activity=json.loads((Path(root)/'activity.json').read_text())
            frame=json.loads((Path(root)/'frame.json').read_text())
            self.assertEqual(activity['sequence'],frame['sequence'])
            self.assertEqual(frame['state'],'failed')
            self.assertEqual(frame['session'],owner.session)
            self.assertEqual(frame['tracer'],[0.0])

    def test_scientific_worker_remains_single_threaded_with_parent_observation(self):
        with tempfile.TemporaryDirectory() as root:
            fixture=Path(root)/'identity-fixture'
            fixture.write_text("#!/usr/bin/python3.12\nimport json\nprint(json.dumps(dict(identity=True)))\n")
            fixture.chmod(0o755)
            profile=Path(__file__).resolve().parents[1]/'local-pressure-v1'
            code="import json,sys,time;from pathlib import Path;sys.path.insert(0,sys.argv[1]);import process; result,evidence=process.native_call(Path(sys.argv[2]),'--identity',None,time.monotonic()+4,3);assert evidence['drained'] and evidence['echild'];print(json.dumps(result))"
            with Watch(lambda:None):
                result=subprocess.run([sys.executable,'-c',code,str(profile),str(fixture)],check=True,capture_output=True,timeout=6)
            self.assertEqual(json.loads(result.stdout),{'identity':True})

    def test_observer_thread_joins_and_does_not_hide_producer_failure(self):
        observed=threading.Event()
        with self.assertRaisesRegex(RuntimeError,'producer'):
            with Watch(observed.set) as watch:
                self.assertTrue(observed.wait(2))
                raise RuntimeError('producer')
        self.assertFalse(watch.thread.is_alive())


if __name__=='__main__':unittest.main()
