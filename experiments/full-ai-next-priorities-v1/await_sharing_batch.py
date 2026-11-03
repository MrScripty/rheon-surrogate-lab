"""Observe the existing owned batch to terminal state; never resubmit any work."""
import time
from study_support import BASE,read,sha,write


if __name__=='__main__':
    path=BASE/'temporal-sharing-batch-01/execution.json'
    deadline=time.monotonic()+21600
    while True:
        state=read(path)
        if state['status']!='running':break
        if time.monotonic()>=deadline:raise TimeoutError('Existing sharing batch still active; no retry')
        time.sleep(5)
    if state['status']!='completed'or not state['owned_child_reaped']:
        raise RuntimeError('Existing sharing batch did not complete and reap; inspect its retained failure')
    write(BASE/'temporal-sharing-postprocessing-admission-01.json',dict(status='admitted',
        training_batch_sha256=sha(path),source_sha256=sha(__file__),new_training_runs_submitted=0))
    print('Existing sharing batch completed and reaped; admitted serial postprocessing',flush=True)
