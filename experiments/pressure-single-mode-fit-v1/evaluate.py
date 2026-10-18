"""Own isolated CPU timings and bounded independent fitting/regression workers."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
PREVIOUS=ROOT/'experiments/pressure-four-pass-distillation-v1'
sys.path.insert(0,str(PREVIOUS))
from common import environment,read,sha,write
from cpu import material


def evaluate(config_path,projects,controller_path,output, *, fitted_script=None):
    config=read(config_path)
    controller=read(controller_path)
    if controller['status']!='completed' or not controller['owned_controller_reaped']:
        raise ValueError('Complete managed readback and controller reaping required')
    available=int(next(line.split()[1]for line in Path('/proc/meminfo').read_text().splitlines()if line.startswith('MemAvailable:')))*1024
    if available<18*1024**3:
        raise MemoryError('12 GiB host reserve plus three 2 GiB workers required')
    output.mkdir(exist_ok=False)
    frozen=output/'material.json'
    material(config_path,projects,frozen,True)
    if len(config['arms'])!=1:
        raise ValueError('One completed candidate required')
    candidate=config['arms'][0]['model_id']
    diagnostic=read(ROOT/'experiments/pressure-low-mode-tail-v1/config.json')
    diagnostic['baseline_material']=str(frozen.relative_to(ROOT))
    write(output/'diagnostic-config.json',diagnostic)
    runtime=read(ROOT/config['runtime'])
    env=environment(runtime)
    jobs=[];records=[]
    started=time.perf_counter();status='failed'
    def stop(*_):
        raise KeyboardInterrupt('CPU workers cancelled')
    signal.signal(signal.SIGTERM,stop)
    def start(command,label):
        log=(output/(label+'.log')).open('xb')
        process=subprocess.Popen([runtime['python'],'-B',*map(str,command)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        job=dict(process=process,log=log,label=label,command=command,started=time.perf_counter())
        jobs.append(job)
        return job
    def finish(job):
        code=job['process'].wait(timeout=180)
        records.append(dict(label=job['label'],command=list(map(str,job['command'])),returncode=code,
            elapsed_s=time.perf_counter()-job['started'],reaped=True))
        if code:
            raise RuntimeError('CPU evaluation failed: '+job['label'])
    try:
        cpus=sorted(os.sched_getaffinity(0))[-3:]
        if len(cpus)!=3:
            raise ValueError('Three distinct CPU cores required')
        for identifier in ['C53',candidate]:
            finish(start([PREVIOUS/'cpu.py','snapshots','--material',frozen,'--id',identifier,'--cpu',cpus[-1],
                '--output',output/identifier],identifier+'-snapshots'))
            print('Isolated snapshots complete',identifier,flush=True)
        parallel=[]
        for identifier,cpu in zip(['C53',candidate],cpus):
            parallel.append(start([fitted_script or HERE/'fitted.py','--config',config_path,'--material',frozen,'--model',identifier,
                '--cpu',cpu,'--output',output/(identifier+'-fitted.json')],identifier+'-fitted'))
        parallel.append(start([ROOT/'experiments/pressure-low-mode-tail-v1/diagnose.py','--config',output/'diagnostic-config.json',
            '--model',candidate,'--cpu',cpus[-1],'--output',output/(candidate+'-diagnosis.json')],candidate+'-diagnosis'))
        for job in parallel:
            finish(job)
        status='completed'
    finally:
        for job in jobs:
            process=job['process']
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
            job['log'].close()
        write(output/'execution.json',dict(status=status,elapsed_s=time.perf_counter()-started,rows=records,
            available_host_bytes_before=available,all_owned_children_reaped=all(j['process'].poll()is not None for j in jobs),
            timing_policy='128-case timings serial, one CPU thread/core; then three independent one-core diagnosis workers.',
            environment={k:env[k]for k in ['PYTHONPATH','CUDA_VISIBLE_DEVICES','TULDOK_TORCH_DEVICE','OMP_NUM_THREADS',
                'OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS']},source_sha256=sha(__file__)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['config','projects','controller','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    evaluate(args.config.resolve(),args.projects.resolve(),args.controller.resolve(),args.output.resolve())
