"""Own bounded spacing and promising pass-count follow-ups without fitting."""
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
    output = base/'extra-evaluation-sequence-01'
    if output.exists():
        raise ValueError('Fresh extra-evaluation controller required')
    output.mkdir()
    runtime = read(ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json')
    source = HERE/'evaluate_extra.py'
    (output/'evaluator-source.py').write_bytes(source.read_bytes())
    cpu = sorted(os.sched_getaffinity(0))[-4]
    sequence = ['domain-small-control','domain-expanded','teacher-coarse','teacher-half','teacher-quarter',
                'spacing-data-control','numerical-coarse','sparse-anchors','learned-coarse',
                'spacing-zero-feature','spacing-conditioned']
    plan = dict(source_sha256=sha(__file__),evaluator_sha256=sha(source),cpu=cpu,
        frozen_pass_followups=['C53','C44','C37'],passes=[4,5,6,8,10,12,16],orderings=['none','before','after'],
        trained_coarse_followup='Numerical-coarse passes 4/5/6/8/10/12/16 on consumed128',
        spacing_models=['C53',*sequence],independent_spacing_scenes=12,
        conditional_followup='Sparse-anchor and learned-coarse pass extension if four-pass acceptance exceeds 20/128 or eight-pass worst residual is below 1e-7',
        optimizer_updates=0,automatic_retry=False)
    write(output/'plan.json',plan)
    state = dict(status='running',active=None,workers=[],plan_sha256=sha(output/'plan.json'))
    write(output/'execution.json',state)
    process = None
    def stop(*_):
        raise KeyboardInterrupt('Extra-evaluation controller interrupted; reaping its worker')
    signal.signal(signal.SIGTERM,stop)
    def wait(path):
        deadline = time.monotonic()+7200
        while not path.exists():
            if time.monotonic()>deadline:
                raise TimeoutError('Required managed result unavailable; no new fit or retry')
            for name in ['teacher-spacing-sequence-01','hybrid-spacing-sequence-01','fit-evaluation-sequence-01']:
                p = base/name/'execution.json'
                if p.exists() and read(p)['status']=='stopped':
                    raise RuntimeError('Preceding controller stopped; inspect retained custody')
            time.sleep(2)
        return read(path)
    def worker(name,arguments):
        nonlocal process
        if sha(source)!=plan['evaluator_sha256']:
            raise ValueError('Frozen follow-up evaluator changed')
        command = [runtime['python'],'-B',str(source),'--cpu',str(cpu),'--output',str(output/(name+'.json')),*arguments]
        state.update(active=name)
        write(output/'execution.json',state)
        tick = time.perf_counter()
        with (output/(name+'.log')).open('xb') as log:
            process = subprocess.Popen(command,env=environment(runtime),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            print('Started follow-up',name,flush=True)
            process.wait(timeout=900)
        if process.returncode:
            raise RuntimeError('Follow-up failed: '+name)
        state['workers'].append(dict(name=name,pid=process.pid,reaped=True,returncode=0,
            elapsed_s=time.perf_counter()-tick,results_sha256=sha(output/(name+'.json'))))
        write(output/'execution.json',state)
        print('Completed follow-up',name,flush=True)
    try:
        for checkpoint in plan['frozen_pass_followups']:
            worker(checkpoint+'-pass-extension',['--checkpoint',checkpoint,'--population','consumed128',
                '--passes',*map(str,plan['passes']),'--orderings',*plan['orderings']])
        evaluation = base/'evaluation/numerical-coarse-01/results.json'
        wait(evaluation)
        worker('numerical-coarse-pass-extension',['--evaluation',str(evaluation),'--population','consumed128','--passes',*map(str,plan['passes'])])
        capture_execution = ROOT/'docs/plans/pressure-methods-v1/native-training-spacing-development-execution.json'
        if wait(capture_execution)['status']!='completed':
            raise ValueError('Independent spacing captures must finish before evaluation')
        for name in ['C53',*sequence]:
            selected = ['--checkpoint','C53']
            if name!='C53':
                evaluation = base/'evaluation'/(name+'-01/results.json')
                wait(evaluation)
                selected = ['--evaluation',str(evaluation)]
            worker(name+'-spacing',[*selected,'--population','spacing-development'])
        for name in ['sparse-anchors','learned-coarse']:
            evaluation = base/'evaluation'/(name+'-01/results.json')
            result = wait(evaluation)
            four = next(r for r in result['summary']if r['population']=='consumed128'and r['policy']=='neural-4')
            eight = next(r for r in result['summary']if r['population']=='consumed128'and r['policy']=='neural-8')
            if four['qualified']>20 or eight['worst_relative_equation_residual']<1e-7:
                worker(name+'-pass-extension',['--evaluation',str(evaluation),'--population','consumed128','--passes',*map(str,plan['passes'])])
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
        state['owned_worker_reaped'] = process is None or process.poll() is not None
        write(output/'execution.json',state)


if __name__ == '__main__':
    run()
