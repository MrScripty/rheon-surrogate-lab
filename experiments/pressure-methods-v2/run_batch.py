"""Own a declared finite command list, retain terminal process evidence, never retry."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,write,sha,environment

def run(specification,output):
    if output.exists():raise ValueError('Fresh batch required; inspect prior custody before resuming')
    spec=read(specification)
    if not 1<=len(spec['commands'])<=128:raise ValueError('Finite batch required')
    runtime=read(ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json')
    output.mkdir(parents=True)
    (output/'controller-source.py').write_bytes(Path(__file__).read_bytes())
    record=dict(status='running',specification_sha256=sha(specification),source_sha256=sha(__file__),
        completed=[],active=None,automatic_retry=False)
    child=None
    def stop(*_):raise InterruptedError('Batch interrupted; reaping owned command')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for index,row in enumerate(spec['commands']):
            if row.get('runtime','torch')=='control':
                interpreter=str(ROOT/'runs/dependency-migration-v1/app-env/bin/python3.12')
                child_environment={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'}
                child_environment.pop('PYTHONPATH',None)
            else:
                interpreter=runtime['python'];child_environment=environment(runtime)
            command=[interpreter,'-B',*row['arguments']]
            log=output/f'{index:03d}-{row["id"]}.log'
            tick=time.perf_counter()
            with log.open('xb') as stream:
                child=subprocess.Popen(command,env=child_environment,cwd=ROOT,
                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                record['active']=dict(id=row['id'],pid=child.pid,command=command)
                write(output/'execution.json',record)
                try:code=child.wait(timeout=row.get('timeout',900))
                finally:
                    if child.poll() is None:
                        os.killpg(child.pid,signal.SIGTERM)
                        try:child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid,signal.SIGKILL);child.wait()
                record['completed'].append(dict(id=row['id'],pid=child.pid,returncode=child.returncode,
                    reaped=child.poll() is not None,elapsed_s=time.perf_counter()-tick,
                    log=str(log.relative_to(ROOT)),log_sha256=sha(log)))
                record['active']=None;write(output/'execution.json',record)
                if code:raise RuntimeError('Command failed: '+row['id'])
            print('Completed',row['id'],flush=True)
        record['status']='completed'
    except BaseException as error:
        record.update(status='stopped',error=repr(error));raise
    finally:
        record['owned_child_reaped']=child is None or child.poll() is not None
        write(output/'execution.json',record)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--specification',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.specification.resolve(),args.output.resolve())
