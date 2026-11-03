"""Finite serial experiment controller with retained child custody, no retries."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time
from v4_support import ROOT,read,write,sha
from common import environment

def run(args):
    if args.output.exists():raise ValueError('Fresh batch controller required')
    args.output.mkdir(parents=True)
    plan=read(args.plan);runtime=read(ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json')
    record=dict(status='running',plan_sha256=sha(args.plan),source_sha256=sha(__file__),completed=[],active=None,automatic_retry=False)
    child=None
    def stop(*_):raise InterruptedError('Owned batch stopped')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for i,item in enumerate(plan['commands']):
            log=args.output/f'{i:02d}-{item["id"]}.log';tick=time.perf_counter()
            with log.open('xb')as stream:
                child=subprocess.Popen([runtime['python'],'-B',*item['arguments']],cwd=ROOT,env=environment(runtime),
                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                record['active']=dict(index=i,id=item['id'],pid=child.pid);write(args.output/'execution.json',record)
                try:code=child.wait(timeout=item['timeout'])
                finally:
                    if child.poll()is None:
                        os.killpg(child.pid,signal.SIGTERM)
                        try:child.wait(timeout=15)
                        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            record['completed'].append(dict(id=item['id'],returncode=code,reaped=child.poll()is not None,seconds=time.perf_counter()-tick,log_sha256=sha(log)))
            record['active']=None;write(args.output/'execution.json',record)
            if code:raise RuntimeError('Experiment failed without automatic retry: '+item['id'])
            print('Completed',item['id'],flush=True)
        record['status']='completed'
    except BaseException as error:record.update(status='stopped',error=repr(error));raise
    finally:
        record['owned_child_reaped']=child is None or child.poll()is not None;write(args.output/'execution.json',record)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
