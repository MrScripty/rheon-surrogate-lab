"""Validate restored training arms and promising intermediate-pass policies natively."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read, write, sha, environment


def run():
    base = ROOT/'runs/pressure-methods-v1'
    output = base/'trained-native-sequence-01'
    if output.exists():
        raise ValueError('Fresh native fit controller required')
    output.mkdir()
    cpu = sorted(os.sched_getaffinity(0))[-5]
    runtime = read(ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json')
    source = HERE/'drift_episode_extended.py'
    arms = ['domain-small-control','domain-expanded','teacher-coarse','teacher-half','teacher-quarter',
            'spacing-data-control','numerical-coarse','sparse-anchors','learned-coarse','spacing-zero-feature','spacing-conditioned']
    plan = dict(arms=arms,steps=512,passes=8,seed=20261103,cpu=cpu,
        extra_policy_rule='For each frozen checkpoint/order and trained numerical-coarse, test the fewest passes below eight qualifying all consumed128 requests, if present.',
        source_sha256=sha(__file__),episode_sha256=sha(source),
        qualification_sha256=sha(ROOT/'docs/plans/pressure-methods-v1/drift-observation-qualification.json'),
        accepted_drifts='Independent committed pressure/velocity/tracer; same declared 1% relative L2 diagnostic limits',automatic_retry=False)
    write(output/'plan.json',plan)
    state = dict(status='running',active=None,workers=[],skipped=[],plan_sha256=sha(output/'plan.json'))
    write(output/'execution.json',state)
    process = None
    def stop(*_):
        raise KeyboardInterrupt('Native fit sequence interrupted; reaping its episode owner')
    signal.signal(signal.SIGTERM,stop)
    def wait(path):
        deadline = time.monotonic()+7200
        while not path.exists():
            if time.monotonic()>deadline:
                raise TimeoutError('Prior evaluation unavailable; no retry')
            for name in ['hybrid-spacing-sequence-01','fit-evaluation-sequence-01','extra-evaluation-sequence-01']:
                p=base/name/'execution.json'
                if p.exists() and read(p)['status']=='stopped':
                    raise RuntimeError('Preceding experiment stopped; inspect custody')
            time.sleep(2)
        return read(path)
    def worker(name,arguments):
        nonlocal process
        if sha(source)!=plan['episode_sha256']:
            raise ValueError('Frozen extended episode changed')
        state.update(active=name)
        write(output/'execution.json',state)
        command = [runtime['python'],'-B',str(source),'--build',str(base/'drift-observation-02'),
            '--output',str(output/name),'--cpu',str(cpu),'--steps','512','--seed','20261103',*arguments]
        tick = time.perf_counter()
        with (output/(name+'.log')).open('xb') as log:
            process = subprocess.Popen(command,env=environment(runtime),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            print('Testing native fit policy',name,flush=True)
            process.wait(timeout=660)
        if process.returncode:
            raise RuntimeError('Native episode failed: '+name)
        result = read(output/name/'results.json')
        if result['status'] not in ['completed','pressure_rejection'] or not result['native_process']['reaped']:
            raise ValueError('Drained native result required')
        state['workers'].append(dict(name=name,status=result['status'],post_warmup_steps=result['completed_post_warmup_steps'],
            results_sha256=sha(output/name/'results.json'),pid=process.pid,reaped=True,elapsed_s=time.perf_counter()-tick))
        write(output/'execution.json',state)
        print(name,result['status'],result['completed_post_warmup_steps'],flush=True)
    try:
        for arm in arms:
            evaluation = base/'evaluation'/(arm+'-01/results.json')
            wait(evaluation)
            worker(arm+'-eight',['--kind','neural','--passes','8','--fit-evaluation',str(evaluation)])
        extra = base/'extra-evaluation-sequence-01'
        for name in ['C53','C44','C37','numerical-coarse']:
            extension = wait(extra/(name+'-pass-extension.json'))
            for ordering in (['none','before','after'] if name!='numerical-coarse' else ['none']):
                options = [r for r in extension['summary']if r['ordering']==ordering and r['passes']<8 and r['qualified']==r['cases']]
                if not options:
                    state['skipped'].append(dict(model=name,ordering=ordering,reason='No below-eight pass budget qualifies all consumed128 requests'))
                    continue
                selected = min(options,key=lambda r:r['passes'])
                arguments = ['--kind','neural'if ordering=='none'else ordering,'--passes',str(selected['passes'])]
                if name=='numerical-coarse':
                    arguments += ['--fit-evaluation',str(base/'evaluation/numerical-coarse-01/results.json')]
                else:
                    arguments += ['--checkpoint',name]
                worker(name+'-'+ordering+'-'+str(selected['passes']),arguments)
        state.update(status='completed',active=None)
    except BaseException as error:
        state.update(status='stopped',error=type(error).__name__+': '+str(error))
        raise
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait()
        state['owned_episode_reaped'] = process is None or process.poll() is not None
        write(output/'execution.json',state)


if __name__ == '__main__':
    run()
