"""Observe the existing four-fit controller without submitting or replaying work."""
import time
from study_support import BASE,read,sha,write


if __name__=='__main__':
    path=BASE/'fixed-viscosity-admission-batch-01/execution.json'
    deadline=time.monotonic()+23500
    while True:
        state=read(path)
        if state['status']!='running':break
        if time.monotonic()>=deadline:raise TimeoutError('Same family controller still active; do not resubmit')
        time.sleep(5)
    training_path=BASE/'fixed-viscosity-training-batch-01/execution.json';training=read(training_path)
    if state['status']!='completed'or not state['owned_child_reaped']or training['status']!='completed'or not training['owned_child_reaped']:
        raise RuntimeError('Existing family sequence did not complete and reap; inspect its retained failure')
    write(BASE/'fixed-viscosity-postprocessing-admission-01.json',dict(status='admitted',
        parent_admission_batch_sha256=sha(path),training_batch_sha256=sha(training_path),source_sha256=sha(__file__),new_training_runs_submitted=0))
    print('Four managed fits/restores/evaluations complete and reaped; serial postprocessing admitted',flush=True)
