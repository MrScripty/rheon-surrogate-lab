"""Observe the existing final application owner before supplemental boundary work."""
import json
import time
from study_support import BASE,read,write,sha


if __name__=='__main__':
    path=BASE/'temporal-final-confirmation-batch-01/execution.json';deadline=time.monotonic()+90000
    while True:
        try:value=read(path)
        except json.JSONDecodeError:
            if time.monotonic()>=deadline:raise TimeoutError('Same application owner observation deadline')
            time.sleep(1);continue
        if value['status']!='running':break
        if time.monotonic()>=deadline:raise TimeoutError('Same application owner still active; no resubmission')
        time.sleep(5)
    if value['status']!='completed'or not value['owned_child_reaped']:
        raise RuntimeError('Existing application owner stopped; inspect the retained failure without retry')
    execution=read(BASE/'temporal-final-application-01/execution.json')
    if execution['status']!='completed'or not all(w['reaped']and w['returncode']==0 for w in execution['workers']):
        raise ValueError('All existing application workers must complete/reap before new numerical work')
    write(BASE/'application-domain-change-admission-01.json',dict(status='admitted',
        original_owner_sha256=sha(path),source_sha256=sha(__file__),new_training_runs_submitted=0,numerical_work_replayed=False))
    print('Existing final application campaign complete/reaped; supported boundary-change qualification admitted',flush=True)
