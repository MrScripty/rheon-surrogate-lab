"""Own the matched trained numerical pair after the existing six-fit owner drains."""
import os
from pathlib import Path
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,write,sha
from run_batch import run as batch

def run():
    base=ROOT/'runs/pressure-methods-v2';output=base/'numerical-followup-01'
    if output.exists():raise ValueError('Fresh follow-up owner required')
    output.mkdir();(output/'controller-source.py').write_bytes(Path(__file__).read_bytes())
    state=dict(status='waiting_for_live_training_owner',source_sha256=sha(__file__),active_pid=None)
    def stop(*_):raise InterruptedError('Follow-up interrupted before a new dispatch')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    deadline=time.monotonic()+14400
    try:
        while True:
            owner=read(base/'training-controller-02/execution.json')
            if owner['status']=='completed':
                if len(owner['completed'])!=6 or any(r['returncode'] or not r['reaped'] for r in owner['completed']):
                    raise ValueError('Six complete drained managed commands required')
                break
            if owner['status']!='running':raise RuntimeError('Prior training owner stopped; do not dispatch new fits')
            if owner.get('active'):
                os.kill(owner['active']['pid'],0);state['active_pid']=owner['active']['pid']
            if time.monotonic()>deadline:raise TimeoutError('Bounded wait for existing owner expired')
            write(output/'execution.json',state);time.sleep(2)
        commands=[]
        for arm in ['v2-num4','v2-num6']:
            qualification=ROOT/'docs/plans/pressure-methods-v2'/(arm+'-qualification-02.json')
            q=read(qualification)
            if q['status']!='passed' or not q['entrypoint_configuration_validated']:
                raise ValueError('Actual numerical training entrypoint must qualify before dispatch')
            arguments=[str(ROOT/'experiments/pressure-methods-v1/manage.py')]
            for key,path in [('config',base/(arm+'-experiment-02.json')),('assembly',base/(arm+'-project-02/assembly.json')),
                ('qualification',qualification),('output',base/(arm+'-study-02')),
                ('tuldok',ROOT/'runs/dependency-migration-v1/Tuldok'),
                ('session-file',ROOT/'runs/tuldok-performance-v1/private/owner-session.txt')]:
                arguments+=['--'+key,str(path)]
            arguments+=['--endpoint','http://127.0.0.1:8091','--catalog-id','a7da4a8de53541a2a0e2aad08005bb71','--project-alias','current-lab']
            commands.append(dict(id=arm,runtime='control',arguments=arguments,timeout=4200))
        specification=output/'training-specification.json';write(specification,dict(commands=commands))
        state.update(status='training',active_pid=None);write(output/'execution.json',state)
        batch(specification,base/'numerical-training-controller-01')
        # The first evaluator may still be finishing the preceding architecture.
        while True:
            owner=read(base/'evaluation-controller-01/execution.json')
            if owner['status']=='completed':break
            if owner['status']!='running':raise RuntimeError('Prior evaluator stopped; preserve completed fits')
            if owner.get('active') and owner['active']['phase']=='evaluating':os.kill(owner['active']['pid'],0)
            if time.monotonic()>deadline:raise TimeoutError('Prior evaluator wait expired')
            time.sleep(2)
        commands=[]
        for arm in ['v2-num4','v2-num6']:
            commands.append(dict(id=arm,arguments=[str(ROOT/'experiments/pressure-methods-v2/evaluate.py'),
                '--name',arm,'--project',str(base/(arm+'-project-02')),'--study',str(base/(arm+'-study-02')),
                '--output',str(base/'evaluation-01'/arm),'--cpu','22'],timeout=1800))
        specification=output/'evaluation-specification.json';write(specification,dict(commands=commands))
        state.update(status='evaluating');write(output/'execution.json',state)
        batch(specification,base/'numerical-evaluation-controller-01')
        state.update(status='completed')
    except BaseException as error:
        state.update(status='stopped',error=repr(error));raise
    finally:write(output/'execution.json',state)

if __name__=='__main__':run()
