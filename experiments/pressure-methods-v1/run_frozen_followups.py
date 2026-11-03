"""Native checks for the separate seven-pass and C37 ten-pass screen results."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read, write, sha, environment


def run():
    base=ROOT/'runs/pressure-methods-v1'
    output=base/'frozen-native-followups-01'
    if output.exists():raise ValueError('Fresh finite frozen-policy followup required')
    rows=[('C53','neural',7),('C53','before',7),('C44','before',7),('C44','after',7),
          ('C37','before',10),('C37','after',10)]
    for checkpoint,kind,passes in rows:
        path=base/'seven-pass-followups'/(checkpoint+'.json')if passes==7 else base/'extra-evaluation-sequence-01'/(checkpoint+'-pass-extension.json')
        found=[r for r in read(path)['summary']if r['passes']==passes and r['ordering']==('none'if kind=='neural'else kind)]
        if len(found)!=1 or found[0]['qualified']!=128:raise ValueError('Prescreened all-128 qualification required')
    output.mkdir()
    source=HERE/'drift_episode_extended.py'
    runtime=read(ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json')
    cpus=sorted(os.sched_getaffinity(0))[-7:-5]
    plan=dict(rows=[dict(checkpoint=c,kind=k,passes=p)for c,k,p in rows],cpus=cpus,steps=512,seed=20261103,
        source_sha256=sha(__file__),episode_sha256=sha(source),optimizer_updates=0,automatic_retry=False)
    write(output/'plan.json',plan)
    state=dict(status='running',active={},workers=[])
    write(output/'execution.json',state)
    lock=threading.Lock()
    children={}
    stopping=threading.Event()
    def stop(*_):
        stopping.set()
        with lock:
            for child in children.values():
                if child.poll()is None:os.killpg(child.pid,signal.SIGTERM)
        raise InterruptedError('Frozen followups stopped; native episode owners reap children')
    signal.signal(signal.SIGTERM,stop)
    def worker(index):
        for checkpoint,kind,passes in rows[index::2]:
            if stopping.is_set():return
            if sha(source)!=plan['episode_sha256']:raise ValueError('Frozen episode changed')
            name=f'{checkpoint}-{kind}-{passes}'
            destination=output/name
            command=[runtime['python'],'-B',str(source),'--build',str(base/'drift-observation-02'),'--output',str(destination),
                '--cpu',str(cpus[index]),'--steps','512','--seed','20261103','--checkpoint',checkpoint,'--kind',kind,'--passes',str(passes)]
            process=None
            tick=time.perf_counter()
            try:
                with (output/(name+'.log')).open('xb')as log:
                    process=subprocess.Popen(command,env=environment(runtime),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    with lock:
                        children[name]=process
                        state['active'][name]=process.pid
                        write(output/'execution.json',state)
                    process.wait(timeout=660)
                if process.returncode:raise RuntimeError('Native frozen followup failed: '+name)
                result=read(destination/'results.json')
                print(name,result['status'],result['completed_post_warmup_steps'],flush=True)
            finally:
                if process is not None and process.poll()is None:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL)
                        process.wait()
                with lock:
                    children.pop(name,None)
                    state['active'].pop(name,None)
                    state['workers'].append(dict(name=name,pid=process.pid if process else None,reaped=process is not None and process.poll()is not None,
                        returncode=process.returncode if process else None,elapsed_s=time.perf_counter()-tick,
                        results_sha256=sha(destination/'results.json')if(destination/'results.json').exists()else None))
                    write(output/'execution.json',state)
    try:
        with ThreadPoolExecutor(max_workers=2)as pool:
            futures=[pool.submit(worker,i)for i in range(2)]
            for future in as_completed(futures):future.result()
        state.update(status='completed')
    except BaseException as error:
        stopping.set()
        state.update(status='stopped',error=type(error).__name__+': '+str(error))
        raise
    finally:write(output/'execution.json',state)


if __name__=='__main__':run()
