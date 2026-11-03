"""Repeat shortlisted prepared-pressure comparisons after campaign workers cease."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read, write, sha, environment


def run():
    base=ROOT/'runs/pressure-methods-v1'
    output=base/'quiet-timing-01'
    source=HERE/'quiet_evaluate.py'
    if output.exists() or source.exists():
        raise ValueError('Fresh quiet timing command required')
    output.mkdir()
    original=HERE/'evaluate_extra.py'
    text=original.read_text()
    edits=[("policies = [(ordering,passes) for ordering in args.orderings for passes in args.passes]",
            "policies = [(ordering,passes) for ordering in args.orderings for passes in args.passes]+[('full-numerical',0)]"),
           ("                        pressure = sample['base']\n                        if ordering=='before':",
            "                        pressure = sample['base']\n                        if ordering=='full-numerical':\n                            pressure = pressure+numerics.correction(sample['native_input_residual'],sample['cell_size_m'],sample['rho'],None)\n                        if ordering=='before':")]
    for before,after in edits:
        if text.count(before)!=1:raise ValueError('Quiet evaluator seam changed')
        text=text.replace(before,after)
    source.write_text(text)
    (output/'original-evaluator.py').write_bytes(original.read_bytes())
    pending=[base/name/'execution.json'for name in ['hybrid-spacing-sequence-01','fit-evaluation-sequence-01',
        'extra-evaluation-sequence-01','trained-native-sequence-01','coarse-head-rate-controller-01']]
    pending.append(ROOT/'docs/plans/pressure-methods-v1/drift-execution.json')
    long=base/'before-six-long-followup-01/results.json'
    cpu=sorted(os.sched_getaffinity(0))[-1]
    plan=dict(controllers=[str(p)for p in pending],additional_episode=str(long),
        source_sha256=sha(__file__),original_evaluator_sha256=sha(original),evaluator_sha256=sha(source),
        derivation=edits,cpu=cpu,timing_repetitions=3,waiting_deadline_s=14400,
        frozen_models=['C53','C44','C37'],trained_models=['numerical-coarse','coarse-head-rate'],
        scope='Same CPU for every worker; all owned fitting/evaluation/drift work ceased before measurement. Prepared physical tensors through final residual; full numerical control included.')
    write(output/'plan.json',plan)
    state=dict(status='waiting_for_campaign_workers',active=None,workers=[],plan_sha256=sha(output/'plan.json'))
    write(output/'execution.json',state)
    runtime=read(ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json')
    process=None
    def stop(*_):raise KeyboardInterrupt('Quiet timing interrupted; reaping owned worker')
    signal.signal(signal.SIGTERM,stop)
    def worker(name,arguments):
        nonlocal process
        if sha(source)!=plan['evaluator_sha256']:raise ValueError('Frozen quiet evaluator changed')
        state.update(status='running',active=name)
        write(output/'execution.json',state)
        command=[runtime['python'],'-B',str(source),'--population','consumed128','--cpu',str(cpu),
                 '--repetitions','3','--output',str(output/(name+'.json')),*arguments]
        tick=time.perf_counter()
        with (output/(name+'.log')).open('xb')as log:
            process=subprocess.Popen(command,env=environment(runtime),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            print('Quiet matched timing',name,flush=True)
            process.wait(timeout=900)
        if process.returncode:raise RuntimeError('Quiet worker failed: '+name)
        state['workers'].append(dict(name=name,pid=process.pid,reaped=True,returncode=0,
            elapsed_s=time.perf_counter()-tick,results_sha256=sha(output/(name+'.json'))))
        write(output/'execution.json',state)
    try:
        deadline=time.monotonic()+14400
        while True:
            values=[read(p)if p.exists()else{}for p in pending]
            if any(v.get('status')=='stopped'for v in values):raise RuntimeError('Prior campaign work stopped; inspect before benchmarking')
            if all(v.get('status')=='completed'for v in values)and long.exists():break
            failed=long.parent/'native/receipt.json'
            if failed.exists()and read(failed).get('status')=='failed':raise RuntimeError('Long follow-up failed; inspect before benchmarking')
            if time.monotonic()>deadline:raise TimeoutError('Campaign workers did not finish within quiet-wait budget')
            time.sleep(5)
        write(output/'quiet-start.json',dict(controllers={str(p):sha(p)for p in pending},
            long_followup_sha256=sha(long),host_load_1_5_15=list(os.getloadavg()),unix_time=time.time()))
        for name in plan['frozen_models']:
            passes=['5','6','7','8']if name!='C37'else['7','8','10','12']
            worker(name,['--checkpoint',name,'--passes',*passes,'--orderings','none','before','after'])
        for name in plan['trained_models']:
            evaluation=base/'evaluation'/(name+'-01/results.json')
            passes=['4','5','6','7','8']if name=='numerical-coarse'else['1','2','3','4','5','6','7','8']
            worker(name,['--evaluation',str(evaluation),'--passes',*passes])
        state.update(status='completed',active=None)
    except BaseException as error:
        state.update(status='stopped',error=type(error).__name__+': '+str(error))
        raise
    finally:
        if process is not None and process.poll()is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait()
        state['owned_worker_reaped']=process is None or process.poll()is not None
        write(output/'execution.json',state)


if __name__=='__main__':run()
