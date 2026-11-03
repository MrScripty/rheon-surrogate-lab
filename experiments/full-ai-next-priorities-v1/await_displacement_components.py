"""Observe the specific component batch and require terminal child reaping."""
import time
from study_support import BASE,read,write,sha


if __name__=='__main__':
    path=BASE/'displacement-component-diagnostic-batch-02/execution.json';deadline=time.monotonic()+1800
    while True:
        state=read(path)
        if state['status']!='running':break
        if time.monotonic()>deadline:raise TimeoutError('Same component batch still active; no resubmission')
        time.sleep(5)
    if not state['owned_child_reaped']:raise ValueError('Prior numerical child must be reaped before profiling')
    write(BASE/'displacement-CPU-stage-profile-admission-01.json',dict(status='admitted',prior_batch_status=state['status'],
        prior_batch_sha256=sha(path),source_sha256=sha(__file__),new_training_runs_submitted=0))
    print('Existing component workers are terminal and reaped; isolated profile admitted',flush=True)
