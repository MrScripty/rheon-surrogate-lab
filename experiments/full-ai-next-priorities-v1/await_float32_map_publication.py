"""Admit read-only publication after the original timing owner has reaped."""
import json
import time
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


if __name__=='__main__':
    registration=DOCS/'float32-map-publication-repair-registration-01.json';reg=read(registration)
    for name,digest in reg['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Registered read-only publication repair changed')
    path=BASE/'float32-map-timing-batch-01/execution.json';deadline=time.monotonic()+40000
    while True:
        if time.monotonic()>=deadline:raise TimeoutError('Same timing owner still active; no resubmission')
        try:state=read(path)
        except (FileNotFoundError,json.JSONDecodeError):time.sleep(1);continue
        if state['status']!='running':break
        time.sleep(5)
    commands=state['completed']
    if not state['owned_child_reaped']or state.get('active')is not None or len(commands)!=3 or not all(c['reaped']for c in commands):
        raise RuntimeError('All original command attempts must be terminal and reaped')
    if any(c['returncode']for c in commands[:2]):raise RuntimeError('Numerical campaign failed; publication-only repair cannot recover it')
    if state['status']=='stopped':
        failed=commands[2];log=ROOT/failed['log']
        if failed['id']!='publish-actual-F32-map-cost-and-accuracy'or failed['returncode']!=1 or sha(log)!=failed['log_sha256']:
            raise RuntimeError('Unexpected original failure')
        if 'ValueError: Complete finite serial CPU campaign required'not in log.read_text():
            raise RuntimeError('Only the known parent-completion check is recoverable here')
    elif state['status']!='completed':raise RuntimeError('Unrecognized original terminal state')
    write(BASE/'float32-map-publication-admission-01.json',dict(status='admitted',original_batch_sha256=sha(path),
        original_status=state['status'],source_sha256=sha(__file__),registration_sha256=sha(registration),numerical_work_replayed=False))
    print('Original CPU timing owner reaped; admitted dependent read-only publication without replay',flush=True)
