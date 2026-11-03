"""Real delegated child success and cancellation through the existing owner."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from run import execute_tasks


class Lifecycle(unittest.TestCase):
    def test_success_records_every_child(self):
        with tempfile.TemporaryDirectory() as folder:
            cpus = sorted(os.sched_getaffinity(0))[:2]
            tasks = [(f'child-{i}', [sys.executable, '-c', 'import os; print(sorted(os.sched_getaffinity(0)))']) for i in range(4)]
            results, failures, interruption = execute_tasks(tasks, folder, cpus, 2)
            self.assertEqual(len(results), 4)
            self.assertEqual(failures, [])
            self.assertIsNone(interruption)
            self.assertTrue(all(r['timing']['child_reaped'] and r['timing']['returncode'] == 0 for r in results))
            self.assertTrue(all(len(json.loads((Path(folder)/(name+'.log')).read_text())) == 1 for name, _ in tasks))

    def test_interrupt_reaps_active_children_and_cancels_pending_work(self):
        with tempfile.TemporaryDirectory() as folder:
            driver = Path(folder)/'driver.py'
            source = str(Path(__file__).resolve().parent)
            driver.write_text('import sys,os,json\n'
                f'sys.path.insert(0,{source!r})\nfrom run import execute_tasks\n'
                'tasks=[(f"child-{i}",[sys.executable,"-c","import time; time.sleep(30)"]) for i in range(6)]\n'
                f'results,failures,interruption=execute_tasks(tasks,{folder!r},sorted(os.sched_getaffinity(0))[:2],2)\n'
                'assert isinstance(interruption,KeyboardInterrupt)\nassert len(failures)==6\n'
                'print(json.dumps({"interrupted":True,"failures":failures}))\n')
            child = subprocess.Popen([sys.executable, str(driver)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                deadline = time.monotonic()+10
                while time.monotonic() < deadline:
                    timings = list(Path(folder).glob('*-timing.json'))
                    try:
                        if len(timings) == 2 and all('pid' in json.loads(p.read_text()) for p in timings):
                            break
                    except json.JSONDecodeError:
                        pass
                    time.sleep(.02)
                else:
                    self.fail('two active delegated children did not start')
                child.send_signal(signal.SIGINT)
                stdout, stderr = child.communicate(timeout=10)
                self.assertEqual(child.returncode, 0, stderr.decode())
                self.assertTrue(json.loads(stdout)['interrupted'])
                timings = [json.loads(p.read_text()) for p in Path(folder).glob('*-timing.json')]
                self.assertEqual(len(timings), 2)
                self.assertTrue(all(t['status'] == 'interrupted' and t['child_reaped'] and t['returncode'] != 0 for t in timings))
            finally:
                if child.poll() is None:
                    child.kill()
                child.communicate(timeout=5)


if __name__ == '__main__':
    unittest.main()
