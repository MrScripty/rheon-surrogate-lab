"""Observe the existing six-fit owner without admitting or replaying any Run."""
import json
import time
from study_support import BASE,read,write,sha


if __name__=='__main__':
    path=BASE/'fixed-viscosity-replication-admission-batch-02/execution.json'
    deadline=time.monotonic()+36000
    while True:
        try:state=read(path)
        except json.JSONDecodeError:
            if time.monotonic()>=deadline:raise TimeoutError('Same replica owner observation deadline')
            time.sleep(1);continue
        if state['status']!='running':break
        if time.monotonic()>=deadline:raise TimeoutError('Same replica owner remains active; do not resubmit')
        time.sleep(5)
    training_path=BASE/'fixed-viscosity-replication-training-batch-01/execution.json';training=read(training_path)
    if state['status']!='completed'or not state['owned_child_reaped']or training['status']!='completed'or not training['owned_child_reaped']:
        raise RuntimeError('Existing replica owner did not complete/reap; inspect the retained failure, no retry')
    write(BASE/'fixed-viscosity-replication-postprocessing-admission-01.json',dict(status='admitted',
        parent_owner_sha256=sha(path),training_owner_sha256=sha(training_path),source_sha256=sha(__file__),
        new_training_runs_submitted=0,numerical_work_replayed=False))
    print('Six managed replica fits/restores/evaluations and audit complete/reaped; serial execution work admitted',flush=True)
