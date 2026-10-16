"""Independent resource-lease and completed-observation scheduling checks."""
import threading
import unittest
from study import execute_partitioned


class StudyScheduling(unittest.TestCase):
    def test_uneven_work_streams_without_overlapping_a_core_lease(self):
        release=threading.Event();lock=threading.Lock();active=set();finished=[];observed=[]
        tasks=[(index,index%2) for index in range(6)]
        def execute(task):
            index,core=task
            with lock:
                self.assertNotIn(core,active);active.add(core)
            try:
                if index==0:self.assertTrue(release.wait(2),'faster partition must progress independently')
                if index==3:release.set()
                return index
            finally:
                with lock:active.remove(core);finished.append(index)
        execute_partitioned(tasks,2,execute,observed.append)
        self.assertEqual(sorted(observed),list(range(6)))
        self.assertLess(observed.index(1),observed.index(0))
        self.assertFalse(active);self.assertEqual(sorted(finished),list(range(6)))

    def test_worker_failure_does_not_launch_its_next_task(self):
        started=[];finished=[];lock=threading.Lock()
        def execute(index):
            with lock:started.append(index)
            try:
                if index==0:raise ValueError('owned worker failure')
                return index
            finally:
                with lock:finished.append(index)
        with self.assertRaisesRegex(ValueError,'owned worker failure'):
            execute_partitioned(list(range(6)),2,execute,lambda _value:None)
        self.assertNotIn(2,started);self.assertEqual(sorted(started),sorted(finished))


if __name__=='__main__':unittest.main()
