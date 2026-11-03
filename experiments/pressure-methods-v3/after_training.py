"""Bounded dependency gate: evaluate only complete drained fits and CPU restores."""
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

def run(args):
    if args.output.exists():raise ValueError('Fresh dependent evaluator required')
    args.output.mkdir(parents=True);base=ROOT/'runs/pressure-methods-v3'
    runtime=read(ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json');deadline=time.monotonic()+7200
    record=dict(status='waiting',source_sha256=sha(__file__),completed=[],active=None,automatic_retry=False);child=None
    def stop(*_):raise InterruptedError('Owned evaluator interrupted')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for arm in ['v3-replay','v3-native','v3-c37half','v3-c44half']:
            receipt=base/(arm+'-study-03/receipt.json')
            while not receipt.exists():
                if time.monotonic()>deadline:raise TimeoutError('Finite managed-fit dependency deadline exceeded')
                record.update(status='waiting',dependency=arm);write(args.output/'execution.json',record);time.sleep(10)
            for label,command in [('entry',[str(ROOT/'experiments/pressure-methods-v3/restore_entry.py'),'--arm',arm,'--output',str(base/'entries'/f'{arm}.json')]),
                ('evaluation',[str(ROOT/'experiments/pressure-methods-v3/evaluate.py'),'--entry',str(base/'entries'/f'{arm}.json'),
                    '--kind','numerical'if'half'in arm else'learned','--old','--cpu',str(args.cpu),'--output',str(base/'evaluation'/f'{arm}.json')])]:
                tick=time.perf_counter();log=args.output/(arm+'-'+label+'.log')
                with log.open('xb')as stream:
                    child=subprocess.Popen([runtime['python'],'-B',*command],env=environment(runtime),cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    record.update(status='running',active=dict(arm=arm,label=label,pid=child.pid));write(args.output/'execution.json',record)
                    try:code=child.wait(timeout=1800)
                    finally:
                        if child.poll()is None:
                            os.killpg(child.pid,signal.SIGTERM)
                            try:child.wait(timeout=10)
                            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                record['completed'].append(dict(arm=arm,label=label,returncode=child.returncode,reaped=child.poll()is not None,seconds=time.perf_counter()-tick,log_sha256=sha(log)))
                record['active']=None;write(args.output/'execution.json',record)
                if code:raise RuntimeError('Dependent '+label+' failed: '+arm)
                print('Completed',arm,label,flush=True)
        record['status']='completed'
    except BaseException as error:record.update(status='stopped',error=repr(error));raise
    finally:
        record['owned_child_reaped']=child is None or child.poll()is not None;write(args.output/'execution.json',record)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cpu',type=int,default=22)
    run(parser.parse_args())
