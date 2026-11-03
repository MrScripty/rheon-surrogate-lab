"""Wait for the already owned replica timing sequence, never replay its work."""
import json
import time
from study_support import BASE,read,write,sha


if __name__=='__main__':
    path=BASE/'fixed-viscosity-replication-postprocessing-batch-01/execution.json'
    deadline=time.monotonic()+60000
    while True:
        try:state=read(path)
        except json.JSONDecodeError:
            if time.monotonic()>=deadline:raise TimeoutError('Same finalist owner observation deadline')
            time.sleep(1);continue
        if state['status']!='running':break
        if time.monotonic()>=deadline:raise TimeoutError('Same finalist owner remains active; no resubmission')
        time.sleep(5)
    if state['status']!='completed'or not state['owned_child_reaped']:
        raise RuntimeError('Existing finalist owner stopped; inspect its actual retained failure')
    timing_path=BASE/'fixed-viscosity-replication-timing-01/execution.json';timing=read(timing_path)
    if timing['status']!='completed'or not all(w['reaped']and w['returncode']==0 for w in timing['workers']):
        raise ValueError('Every actual finalist timing worker must complete and reap')
    write(BASE/'temporal-final-confirmation-admission-01.json',dict(status='admitted',
        finalist_owner_sha256=sha(path),timing_workers_sha256=sha(timing_path),source_sha256=sha(__file__),
        new_training_runs_submitted=0,numerical_measurements_replayed=False))
    print('All already-owned finalist costs complete/reaped; frozen reserved/application confirmation admitted',flush=True)
