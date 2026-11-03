"""Evaluate each fresh CPU restore once; reconcile the live training owner while waiting."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,write,sha,environment

def run(output):
    if output.exists():raise ValueError('Fresh evaluation controller required')
    runtime=read(ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json')
    base=ROOT/'runs/pressure-methods-v2';training=base/'training-controller-02/execution.json'
    arms=['v2-small16','v2-small24','v2-mixed16','v2-retain24','v2-headbase','v2-headshape']
    output.mkdir();(output/'controller-source.py').write_bytes(Path(__file__).read_bytes())
    record=dict(status='running',active=None,completed=[],source_sha256=sha(__file__))
    child=None;deadline=time.monotonic()+14400
    def stop(*_):raise InterruptedError('Evaluation controller stopped; reaping owned child')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for arm in arms:
            receipt=base/(arm+'-study-02/receipt.json')
            while not receipt.exists():
                owner=read(training)
                if owner['status']!='running' or not owner['active']:
                    raise RuntimeError('Training owner is terminal without the required restore; no retry')
                os.kill(owner['active']['pid'],0)
                if time.monotonic()>deadline:raise TimeoutError('Bounded wait for exact live training owner expired')
                record.update(active=dict(arm=arm,phase='waiting_for_live_training_owner',pid=owner['active']['pid']))
                write(output/'execution.json',record);time.sleep(2)
            restored=read(receipt)
            if restored['status']!='completed' or not all(restored['artifact_parity'].values()):
                raise ValueError('Verified completed fit and fresh CPU parity required')
            command=[runtime['python'],'-B',str(ROOT/'experiments/pressure-methods-v2/evaluate.py'),
                '--name',arm,'--project',str(base/(arm+'-project-02')),'--study',str(base/(arm+'-study-02')),
                '--output',str(base/'evaluation-01'/arm),'--cpu','22']
            with (output/(arm+'.log')).open('xb') as log:
                child=subprocess.Popen(command,env=environment(runtime),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                record.update(active=dict(arm=arm,phase='evaluating',pid=child.pid));write(output/'execution.json',record)
                if child.wait(timeout=1800):raise RuntimeError('Evaluation failed; inspect retained log')
            result=base/'evaluation-01'/arm/'results.json'
            record['completed'].append(dict(arm=arm,result=str(result.relative_to(ROOT)),sha256=sha(result),
                training_receipt_sha256=sha(receipt),evaluation_pid=child.pid,reaped=child.poll() is not None))
            print('Evaluated',arm,flush=True)
        record.update(status='completed',active=None)
    except BaseException as error:
        record.update(status='stopped',error=repr(error));raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        record['owned_child_reaped']=child is None or child.poll() is not None
        write(output/'execution.json',record)

if __name__=='__main__':run(ROOT/'runs/pressure-methods-v2/evaluation-controller-01')
