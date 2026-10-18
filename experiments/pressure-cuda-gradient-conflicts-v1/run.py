"""Own and reap one bounded CUDA diagnostic using the selected Pumas runtime."""
import argparse
from datetime import datetime,timezone
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import environment,read,sha,write


def available():
    return int(next(l.split()[1]for l in Path('/proc/meminfo').read_text().splitlines()if l.startswith('MemAvailable:')))*1024


def rss(pid):
    try:
        return int(next(l.split()[1]for l in Path('/proc',str(pid),'status').read_text().splitlines()if l.startswith('VmRSS:')))*1024
    except (FileNotFoundError,StopIteration):
        return 0


def activity(phase,message,progress):
    path = ROOT/'runs/dependency-migration-v1/live-migration/activity.json'
    with (path.parent/'.publisher.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        value=read(path)
        value.update(sequence=value['sequence']+1,updated_at=datetime.now(timezone.utc).isoformat(),
                     state='running',phase=phase,message=message,progress=progress)
        write(path,value)


def run(config_path,output,publish=True):
    config=read(config_path);runtime=read(ROOT/config['runtime'])
    if (config['updates']!=[100,150,200]or config['passes']!=[1,2,3,4]or
            config['target_relative_residual']!=1e-6 or not 0<config['wall_seconds']<=180):
        raise ValueError('Predeclared bounded CUDA gradient study required')
    if output.exists():
        raise ValueError('Fresh CUDA execution directory required')
    memory=available()
    if memory<config['host_reserve_bytes']+config['worker_rss_limit_bytes']:
        raise MemoryError('Host reserve plus diagnostic worker budget required')
    output.mkdir(parents=True)
    started=time.perf_counter();jobs=[];status='failed';minimum=memory;cancelled=False
    def stop(*_):
        nonlocal cancelled
        cancelled=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def cancel_check():
        if cancelled:
            raise KeyboardInterrupt('CUDA diagnostic cancelled')
    def launch(name,command,device):
        log=(output/(name+'.log')).open('xb')
        try:
            process=subprocess.Popen(command,env=environment(runtime,device),stdout=log,
                                     stderr=subprocess.STDOUT,start_new_session=True)
        except BaseException:
            log.close();raise
        job=dict(name=name,command=command,process=process,log=log,peak_rss_bytes=0,device=device)
        jobs.append(job)
        write(output/(name+'-started.json'),dict(pid=process.pid,command=command))
        return job
    def wait(job):
        nonlocal minimum
        while job['process'].poll()is None:
            cancel_check()
            job['peak_rss_bytes']=max(job['peak_rss_bytes'],rss(job['process'].pid))
            minimum=min(minimum,available())
            if job['peak_rss_bytes']>config['worker_rss_limit_bytes']:
                raise MemoryError('CUDA diagnostic RSS cap exceeded')
            if minimum<config['host_reserve_bytes']:
                raise MemoryError('Host available memory reserve breached')
            if time.perf_counter()-started>config['wall_seconds']:
                raise TimeoutError('CUDA diagnostic wall cap elapsed')
            time.sleep(.05)
        cancel_check()
        if job['process'].wait()!=0:
            raise RuntimeError('Diagnostic worker failed; inspect retained boundary log')
    try:
        if publish:
            activity('CUDA gradient-conflict diagnosis','Qualifying geometry then diagnostic backward passes; C53 remains deployed.',dict(updates=config['updates'],training_started=False,model_promoted=False))
        job=launch('qualification',[runtime['python'],'-B',str(HERE/'check.py'),'--output',str(output/'qualification.json')],'cpu')
        wait(job)
        cancel_check()
        job=launch('cuda',[runtime['python'],'-B',str(HERE/'measure.py'),'--config',str(config_path),
                           '--vectors',str(output/'vectors'),'--output',str(output/'measurement.json')],'cuda')
        wait(job)
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
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
            job['log'].close()
        write(output/'execution.json',dict(status=status,elapsed_s=time.perf_counter()-started,
            minimum_host_available_bytes=minimum,available_host_bytes_before=memory,
            all_owned_children_reaped=all(j['process'].poll()is not None for j in jobs),
            workers=[dict(name=j['name'],command=j['command'],pid=j['process'].pid,device=j['device'],
                          peak_rss_bytes=j['peak_rss_bytes'],returncode=j['process'].returncode,
                          reaped=j['process'].poll()is not None)for j in jobs],
            optimizer_updates=0,automatic_retry=False,config_sha256=sha(config_path),
            runtime_sha256=sha(ROOT/config['runtime']),source_sha256=sha(__file__)))
    print('CUDA diagnostic',status,time.perf_counter()-started,'s',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--no-publish',action='store_true')
    args=parser.parse_args()
    run(args.config.resolve(),args.output.resolve(),not args.no_publish)
