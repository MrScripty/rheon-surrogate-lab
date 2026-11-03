"""Wait for the verified owned controller, then run the finite full-width series."""
import os
from pathlib import Path
import sys
import time
from v4_support import ROOT,BASE,read,write,sha

def main():
    identity=read(BASE/'student-controller-identity.json');process=Path('/proc')/str(identity['pid'])
    deadline=time.monotonic()+4*3600;observations=0
    while True:
        try:
            stat=(process/'stat').read_text().split()
            live=stat[21]==identity['start_ticks']and stat[2]!='Z'
            if live:
                args=(process/'cmdline').read_bytes().split(b'\0')
                if b'experiments/pressure-methods-v2/run_batch.py'not in args or b'runs/pressure-methods-v4/student-training-plan-01.json'not in args:
                    raise ValueError('Dependency command identity changed')
        except FileNotFoundError:live=False
        if not live:break
        observations+=1
        write(BASE/'full-training-dependency-01.json',dict(status='verified_wait',pid=identity['pid'],start_ticks=identity['start_ticks'],
            observations=observations,checked_unix=time.time(),source_sha256=sha(__file__)))
        if time.monotonic()>deadline:raise TimeoutError('Finite student dependency observation deadline exceeded; no fit submitted')
        time.sleep(20)
    state=read(BASE/'student-training-batch-01/execution.json')
    if state['status']!='completed'or not state['owned_child_reaped']:raise ValueError('Student series did not complete; no full-width series submitted')
    write(BASE/'full-training-dependency-01.json',dict(status='dependency_completed',student_execution_sha256=sha(BASE/'student-training-batch-01/execution.json'),
        verified_process_absent=True,observations=observations,source_sha256=sha(__file__)))
    print('Student controller completed and reaped; starting full-width series',flush=True)
    os.execv(sys.executable,[sys.executable,'-B',str(ROOT/'experiments/pressure-methods-v2/run_batch.py'),
        '--specification',str(BASE/'full-training-plan-01.json'),'--output',str(BASE/'full-training-batch-01')])

if __name__=='__main__':main()
