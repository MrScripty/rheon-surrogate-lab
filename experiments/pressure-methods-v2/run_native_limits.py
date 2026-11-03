"""Measure native physical horizons; preserve verified physical failures as outcomes."""
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,write,sha,environment

def run():
    base=ROOT/'runs/pressure-methods-v2';output=base/'native-long-controller-01'
    if output.exists():raise ValueError('Fresh native horizon controller required')
    runtime=read(ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json')
    output.mkdir();(output/'controller-source.py').write_bytes(Path(__file__).read_bytes())
    record=dict(status='running',active=None,completed=[],source_sha256=sha(__file__))
    child=None
    def stop(*_):raise InterruptedError('Native horizon controller interrupted; reaping owned child')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for scene in [0,1,4]:
            destination=base/'native-long-01'/f'scene{scene}'
            command=[runtime['python'],'-B',str(ROOT/'experiments/pressure-methods-v2/drift_episode.py'),
                '--build',str(base/'harder-drift-build-02'),'--scene',str(scene),'--kind','native',
                '--cpu','17','--output',str(destination)]
            if scene==4:command+=['--reservation',str(ROOT/'docs/plans/pressure-methods-v2/drift-reservation-03.json')]
            log=output/f'scene{scene}.log';tick=time.perf_counter()
            with log.open('xb') as stream:
                child=subprocess.Popen(command,env=environment(runtime),cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                record['active']=dict(scene=scene,pid=child.pid);write(output/'execution.json',record)
                code=child.wait(timeout=1900)
            if code:
                receipt=read(destination/'native/receipt.json')
                stderr=destination/'native/native-stderr.log';text=stderr.read_text()
                if (not receipt['native_process']['reaped'] or 'Native reference rejected: Quality' not in text
                        or 'qualified: false' not in text):
                    raise RuntimeError('Non-physical native failure; inspect without retry')
                import json
                frames=[json.loads(line) for line in (destination/'native/frames.jsonl').read_text().splitlines()]
                limit=dict(status='native_physical_gate_failure',scene=scene,
                    last_accepted_step=frames[-1]['step'] if frames else 0,
                    last_accepted_time_s=frames[-1]['time_s'] if frames else 0.,
                    native_process=receipt['native_process'],stderr_sha256=sha(stderr),
                    quality=text.split('Native reference rejected: ',1)[1].strip())
                write(destination/'native-limit.json',limit)
                status=limit['status']
            else:
                result=read(destination/'results.json')
                if result['status']!='completed':raise RuntimeError('Native trajectory rejected an unexpected control candidate')
                status=result['status']
            record['completed'].append(dict(scene=scene,status=status,returncode=code,
                pid=child.pid,reaped=child.poll() is not None,elapsed_s=time.perf_counter()-tick,
                directory=str(destination.relative_to(ROOT)),log_sha256=sha(log)))
            record['active']=None;write(output/'execution.json',record)
            print('Native horizon',scene,status,flush=True)
        record['status']='completed'
    except BaseException as error:
        record.update(status='stopped',error=repr(error));raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        record['owned_child_reaped']=child is None or child.poll() is not None
        write(output/'execution.json',record)

if __name__=='__main__':run()
