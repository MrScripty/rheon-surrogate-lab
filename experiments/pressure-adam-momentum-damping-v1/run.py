"""Bound a queue of CPU snapshots by concurrency, wall time and host memory."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import environment,read,sha,write


def available_memory():
    return int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024


def rss(pid):
    try:
        rows=Path('/proc',str(pid),'status').read_text().splitlines()
        return int(next(l.split()[1] for l in rows if l.startswith('VmRSS:')))*1024
    except (FileNotFoundError,StopIteration):return 0


def run(config_path,output,results):
    config=read(config_path);runtime=read(ROOT/config['runtime'])
    if (config['workers']!=3 or config['updates']!=[1,10,100,150,200]
            or not 0<config['wall_seconds']<=120 or config['retention_factors']!=[0,.25,.5,.75,1]):
        raise ValueError('Bounded predeclared CPU momentum screen required')
    cpus=sorted(os.sched_getaffinity(0))[-config['workers']:]
    if len(cpus)!=3:raise ValueError('Three distinct CPU cores required')
    available=available_memory()
    if available<config['host_reserve_bytes']+3*config['worker_rss_limit_bytes']:
        raise MemoryError('Host reserve plus three worker budgets required')
    if output.exists() or any((results/('update-%03d.json'%u)).exists() for u in config['updates']) or (results/'qualification.json').exists():
        raise ValueError('Fresh process and result destinations required')
    output.mkdir(parents=True);results.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();jobs=[];status='failed';maximum_active=0;minimum_available=available
    cancel_requested=False
    def stop(*_):
        nonlocal cancel_requested
        cancel_requested=True
    def check_cancel():
        if cancel_requested:raise KeyboardInterrupt('CPU momentum screen cancelled')
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    def launch(name,command,cpu,update=None):
        log=(output/(name+'.log')).open('xb')
        try:process=subprocess.Popen(command,env=environment(runtime),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        except BaseException:log.close();raise
        job=dict(name=name,command=command,process=process,log=log,cpu=cpu,update=update,peak_rss_bytes=0)
        jobs.append(job)
        write(output/'launched.json',dict(workers=[dict(name=j['name'],pid=j['process'].pid,cpu=j['cpu'])for j in jobs]))
        return job
    try:
        qualification=launch('qualification',[runtime['python'],'-B',str(HERE/'check.py'),
            '--output',str(results/'qualification.json'),'--cpu',str(cpus[-1])],cpus[-1])
        qualification_deadline=time.monotonic()+min(20,config['wall_seconds'])
        while qualification['process'].poll()is None:
            check_cancel()
            qualification['peak_rss_bytes']=max(qualification['peak_rss_bytes'],rss(qualification['process'].pid))
            if qualification['peak_rss_bytes']>config['worker_rss_limit_bytes']:raise MemoryError('Qualifier RSS budget breached')
            if time.monotonic()>qualification_deadline:raise TimeoutError('CPU oracle deadline elapsed')
            time.sleep(.05)
        if qualification['process'].wait():
            raise RuntimeError('CPU oracles failed; no measurement admitted')
        check_cancel()
        pending=list(config['updates']);active=[]
        while pending or active:
            check_cancel()
            if time.perf_counter()-started>config['wall_seconds']:
                raise TimeoutError('Global CPU momentum screen deadline elapsed')
            minimum_available=min(minimum_available,available_memory())
            if minimum_available<config['host_reserve_bytes']:raise MemoryError('Host memory reserve breached')
            for job in list(active):
                process=job['process'];job['peak_rss_bytes']=max(job['peak_rss_bytes'],rss(process.pid))
                if job['peak_rss_bytes']>config['worker_rss_limit_bytes']:raise MemoryError('Worker RSS budget breached')
                if process.poll()is not None:
                    if process.wait()!=0:raise RuntimeError('CPU snapshot failed; retained log identifies the boundary')
                    active.remove(job)
            used={j['cpu'] for j in active};spawned=False
            for cpu in cpus:
                check_cancel()
                if cpu in used or not pending:continue
                update=pending.pop(0)
                active.append(launch('update-%03d'%update,[runtime['python'],'-B',str(HERE/'measure.py'),
                    '--config',str(config_path),'--update',str(update),'--cpu',str(cpu),
                    '--output',str(results/('update-%03d.json'%update))],cpu,update))
                spawned=True
            maximum_active=max(maximum_active,len(active))
            if spawned:write(output/'started.json',dict(workers=[dict(update=j['update'],pid=j['process'].pid,cpu=j['cpu']) for j in jobs if j['update']is not None]))
            time.sleep(.05)
        status='completed'
    except KeyboardInterrupt:
        status='cancelled';raise
    finally:
        for job in jobs:
            if job['process'].poll()is None:
                try:os.killpg(job['process'].pid,signal.SIGTERM)
                except ProcessLookupError:pass
        deadline=time.monotonic()+10
        for job in jobs:
            process=job['process']
            if process.poll()is None:
                try:process.wait(timeout=max(.1,deadline-time.monotonic()))
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            job['log'].close()
        write(output/'execution.json',dict(status=status,elapsed_s=time.perf_counter()-started,
            maximum_active_workers=maximum_active,minimum_host_available_bytes=minimum_available,
            available_host_bytes_before=available,all_owned_children_reaped=all(j['process'].poll()is not None for j in jobs),
            workers=[dict(name=j['name'],update=j['update'],cpu=j['cpu'],command=j['command'],pid=j['process'].pid,
                returncode=j['process'].returncode,reaped=j['process'].poll()is not None,peak_rss_bytes=j['peak_rss_bytes']) for j in jobs],
            optimizer_updates=0,automatic_retry=False,config_sha256=sha(config_path),runtime_sha256=sha(ROOT/config['runtime']),source_sha256=sha(__file__)))
    print(status,'CPU momentum screen',time.perf_counter()-started,'s',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','output','results'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.config.resolve(),a.output.resolve(),a.results.resolve())
