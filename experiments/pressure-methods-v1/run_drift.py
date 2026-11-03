"""Finite two-worker independent drift matrix with prespecified long follow-ups."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
from queue import Queue
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read, write, sha, environment


def run():
    base = ROOT/'runs/pressure-methods-v1'
    output = base/'drift-matrix-01'
    report = ROOT/'docs/plans/pressure-methods-v1/drift-execution.json'
    qualification_path = report.with_name('drift-observation-qualification.json')
    qualification = read(qualification_path)
    if output.exists() or report.exists() or qualification['status'] != 'passed':
        raise ValueError('Fresh qualified drift matrix required')
    cpus = sorted(os.sched_getaffinity(0))[-3:-1]
    if len(cpus) != 2:
        raise ValueError('Two CPUs separate from the evaluation CPU required')
    build = base/'drift-observation-02'
    if sha(build/'build-receipt.json') != qualification['build_receipt_sha256']:
        raise ValueError('Qualified drift binary changed')
    runtime = read(ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json')
    rows = []
    for seed in [20261031,20261101,20261102]:
        for kind,interval in [('native-control',16),('full-numerical',16),('neural',16),
                              ('periodic',4),('periodic',16),('periodic',64)]:
            name = f'{seed}-{kind}'+(f'-{interval}' if kind=='periodic' else '')
            rows.append(dict(name=name,kind=kind,interval=interval,passes=8,steps=512,seed=seed,checkpoint='C53'))
    for checkpoint in ['C44','C37']:
        rows.append(dict(name=checkpoint+'-neural',kind='neural',interval=16,passes=8,steps=512,seed=20261031,checkpoint=checkpoint))
        for kind in ['before','after']:
            rows.append(dict(name=checkpoint+'-'+kind+'-4',kind=kind,interval=16,passes=4,steps=16,seed=20261031,checkpoint=checkpoint))
    rows.append(dict(name='periodic-four-pass-rejection',kind='periodic',interval=16,passes=4,steps=128,seed=20261031,checkpoint='C53'))
    for arm in ['domain-small-control','domain-expanded']:
        evaluation = base/'evaluation'/(arm+'-01/results.json')
        if read(evaluation)['status'] != 'completed':
            raise ValueError('Completed domain evaluation required')
        rows.append(dict(name=arm+'-native32',kind='neural',interval=16,passes=8,steps=128,seed=20261031,
                         checkpoint='C53',counts=[32]*3,evaluation=str(evaluation),evaluation_sha256=sha(evaluation)))
    output.mkdir()
    source = HERE/'drift_episode.py'
    (output/'drift-episode-source.py').write_bytes(source.read_bytes())
    plan = dict(schema='rheon-pressure-methods-drift-matrix-v1',rows=rows,cpus=cpus,
        qualification_sha256=sha(qualification_path),episode_sha256=sha(source),controller_sha256=sha(__file__),
        diagnostic_relative_l2_limits=dict(pressure=.01,velocity=.01,tracer=.01),
        long_followup='For each first-scene six-policy control that completes 512 steps with no drift crossing, repeat at 3600 steps.',
        native_child_deadline_s=600,owned_worker_deadline_s=660,reserve_host_bytes=12*1024**3,per_worker_budget_bytes=2*1024**3,
        timing_scope='Two disjoint single-CPU diagnostic workers; times are instrumented dual-track costs, not production timing.')
    write(output/'plan.json',plan)
    state = dict(status='running',phase='initial',active={},workers=[],plan_sha256=sha(output/'plan.json'),automatic_retry=False)
    write(report,state)
    lock = threading.Lock()
    slots = Queue()
    for cpu in cpus:
        slots.put(cpu)
    children = {}
    stopping = threading.Event()
    def stop(*_):
        stopping.set()
        with lock:
            for child in children.values():
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
        raise InterruptedError('Drift matrix interrupted; workers reap owned native children')
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)

    def worker(row):
        if stopping.is_set():
            raise InterruptedError('Pending drift worker cancelled before launch')
        available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
        if available < 16*1024**3 or sha(source) != plan['episode_sha256']:
            raise ValueError('Memory reserve or frozen episode identity unavailable')
        if 'evaluation' in row and sha(row['evaluation']) != row['evaluation_sha256']:
            raise ValueError('Frozen evaluated fit changed')
        cpu = slots.get()
        destination = output/row['name']
        command = [runtime['python'],'-B',str(source),'--build',str(build),'--output',str(destination),'--cpu',str(cpu)]
        for name in ['kind','interval','passes','steps','seed','checkpoint']:
            command += ['--'+name,str(row[name])]
        if 'counts' in row:
            command += ['--counts',*map(str,row['counts'])]
        if 'evaluation' in row:
            command += ['--fit-evaluation',row['evaluation']]
        process = None
        tick = time.perf_counter()
        try:
            if stopping.is_set():
                raise InterruptedError('Drift worker cancelled before launch')
            with (output/(row['name']+'.log')).open('xb') as log:
                process = subprocess.Popen(command,env=environment(runtime),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                with lock:
                    children[row['name']] = process
                    state['active'][row['name']] = dict(pid=process.pid,cpu=cpu)
                    write(report,state)
                process.wait(timeout=660)
            if process.returncode != 0:
                raise RuntimeError('Drift worker failed; inspect '+row['name'])
            result = read(destination/'results.json')
            if result['status'] not in ['completed','pressure_rejection'] or not result['native_process']['reaped']:
                raise ValueError('Terminal drained drift evidence required')
            return row,result
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL)
                    process.wait()
            with lock:
                children.pop(row['name'],None)
                state['active'].pop(row['name'],None)
                state['workers'].append(dict(name=row['name'],cpu=cpu,pid=process.pid if process else None,
                    returncode=process.returncode if process else None,reaped=process is not None and process.poll() is not None,
                    elapsed_s=time.perf_counter()-tick,results_sha256=sha(destination/'results.json') if (destination/'results.json').exists() else None))
                write(report,state)
            slots.put(cpu)

    completed = []
    def stage(selected):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker,row) for row in selected]
            try:
                for future in as_completed(futures):
                    row,result = future.result()
                    completed.append((row,result))
                    print(row['name'],result['status'],result['completed_post_warmup_steps'],flush=True)
            except BaseException:
                stopping.set()
                for future in futures:
                    future.cancel()
                raise
    try:
        stage(rows)
        long = []
        for row,result in completed:
            if (row['seed']==20261031 and row['checkpoint']=='C53' and row['steps']==512
                    and result['status']=='completed' and all(v is None for v in result['first_crossings'].values())):
                long.append(dict(row,name=row['name']+'-long',steps=3600))
        with lock:
            state.update(phase='prespecified_long_followups')
            write(report,state)
        write(output/'long-followup-selection.json',dict(rows=long,rule=plan['long_followup'],
            source_results={row['name']:sha(output/row['name']/'results.json') for row,result in completed}))
        stage(long)
        state.update(status='completed',phase='complete')
    except BaseException as error:
        state.update(status='stopped',error=type(error).__name__+': '+str(error))
        raise
    finally:
        write(report,state)


if __name__ == '__main__':
    run()
