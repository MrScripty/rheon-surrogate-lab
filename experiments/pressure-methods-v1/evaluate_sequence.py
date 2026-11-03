"""Evaluate each exact CPU fit once as its finite training sequence publishes it."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read, write, sha, environment


def run():
    base = ROOT/'runs/pressure-methods-v1'
    output = base/'fit-evaluation-sequence-01'
    arms = ['teacher-coarse','teacher-half','teacher-quarter','spacing-data-control',
            'numerical-coarse','sparse-anchors','learned-coarse','spacing-zero-feature','spacing-conditioned']
    if output.exists():
        raise ValueError('Fresh evaluation controller required; completed evidence is never overwritten')
    output.mkdir()
    source = ROOT/'experiments/pressure-methods-v1/evaluate_fit.py'
    (output/'evaluate-fit-source.py').write_bytes(source.read_bytes())
    runtime = read(ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json')
    os.sched_setaffinity(0,{sorted(os.sched_getaffinity(0))[-1]})
    record = dict(status='waiting',arms=arms,completed=[],active=None,source_sha256=sha(__file__),
                  evaluator_sha256=sha(source),timing_repetitions=1,automatic_retry=False)
    process = None
    def stop(*_):
        raise KeyboardInterrupt('Evaluation controller stopped; reaping owned local worker')
    signal.signal(signal.SIGTERM,stop)
    try:
        for arm in arms:
            study = base/(arm+'-study-01')
            destination = base/'evaluation'/(arm+'-01')
            if destination.exists():
                raise ValueError('Evaluation already attempted: '+arm)
            deadline = time.monotonic()+7200
            record.update(status='waiting',active=arm)
            write(output/'execution.json',record)
            while not (study/'receipt.json').exists():
                sequence = base/('teacher-spacing-sequence-01' if arm in arms[:4] else 'hybrid-spacing-sequence-01')
                status = read(sequence/'execution.json')
                if status['status']=='stopped' or time.monotonic()>deadline:
                    raise RuntimeError('Managed fit unavailable; stop without retry: '+arm)
                time.sleep(2)
            if sha(source) != record['evaluator_sha256']:
                raise ValueError('Frozen evaluator changed')
            record.update(status='running')
            write(output/'execution.json',record)
            command = [runtime['python'],'-B',str(source),'--project',str(base/(arm+'-project-01')),
                       '--study',str(study),'--name',arm,'--output',str(destination),'--repetitions','1']
            with (output/(arm+'.log')).open('xb') as log:
                process = subprocess.Popen(command,env=environment(runtime),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                print('Evaluating',arm,flush=True)
                process.wait(timeout=900)
            if process.returncode != 0:
                raise RuntimeError('Evaluation failed; inspect retained output: '+arm)
            result = read(destination/'results.json')
            record['completed'].append(dict(arm=arm,results=str(destination/'results.json'),
                results_sha256=sha(destination/'results.json'),summary=result['summary']))
            print('Evaluated',arm,flush=True)
        record.update(status='completed',active=None)
    except BaseException as error:
        record.update(status='stopped',error=type(error).__name__+': '+str(error))
        raise
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait()
        record['owned_worker_reaped'] = process is None or process.poll() is not None
        write(output/'execution.json',record)


if __name__ == '__main__':
    run()
