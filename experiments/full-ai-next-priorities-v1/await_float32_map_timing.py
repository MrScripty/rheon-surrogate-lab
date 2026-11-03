"""Wait for existing numerical campaigns to reap before isolated CPU timing."""
import json
import time
from study_support import BASE,read,write,sha


if __name__=='__main__':
    paths=[BASE/n/'execution.json'for n in ('float32-map-evaluation-batch-01','fixed-viscosity-postprocessing-batch-01')]
    deadline=time.monotonic()+36500
    for path in paths:
        while True:
            if time.monotonic()>=deadline:raise TimeoutError('Same numerical campaigns still active; no resubmission')
            try:state=read(path)
            except (FileNotFoundError,json.JSONDecodeError):time.sleep(1);continue
            if state['status']!='running':break
            time.sleep(5)
        if state['status']!='completed'or not state['owned_child_reaped']:
            raise RuntimeError('Existing numerical campaign did not complete and reap; inspect retained failure')
    write(BASE/'float32-map-timing-admission-01.json',dict(status='admitted',
        dependency_execution_sha256={str(p):sha(p)for p in paths},source_sha256=sha(__file__),new_training_runs_submitted=0))
    print('Original family and float32 evaluation campaigns reaped; isolated CPU timing admitted',flush=True)
