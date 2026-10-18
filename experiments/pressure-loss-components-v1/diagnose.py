"""Own three bounded independent CPU gradient workers and their lifecycle receipt."""
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


def run(config_path,output,results):
    output.mkdir(exist_ok=False);results.mkdir(exist_ok=True)
    config=read(config_path);runtime=read(ROOT/config['runtime']);env=environment(runtime)
    available=int(next(l.split()[1]for l in Path('/proc/meminfo').read_text().splitlines()if l.startswith('MemAvailable:')))*1024
    if available<18*1024**3:raise MemoryError('12 GiB reserve plus three 2 GiB workers required')
    started=time.perf_counter();jobs=[];status='failed'
    def stop(*_):raise KeyboardInterrupt('CPU diagnostic cancelled')
    signal.signal(signal.SIGTERM,stop)
    try:
        for identifier,cpu in zip(config['models'],[21,22,23],strict=True):
            command=[runtime['python'],'-B',str(HERE/'measure.py'),'--config',str(config_path),
                '--model',identifier,'--cpu',str(cpu),'--output',str(results/(identifier+'-components.json')),'--vectors',str(output/(identifier+'-vectors'))]
            log=(output/(identifier+'.log')).open('xb')
            try:p=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            except BaseException:log.close();raise
            jobs.append(dict(identifier=identifier,process=p,log=log,command=command))
        codes=[j['process'].wait(timeout=max(.1,120-(time.perf_counter()-started)))for j in jobs]
        if any(codes):raise RuntimeError('Loss-component diagnostic failed; logs retained')
        status='completed'
    finally:
        for job in jobs:
            p=job['process']
            if p.poll()is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            job['log'].close()
        write(output/'execution.json',dict(status=status,elapsed_s=time.perf_counter()-started,
            workers=[dict(model=j['identifier'],command=j['command'],pid=j['process'].pid,returncode=j['process'].returncode,reaped=j['process'].poll()is not None)for j in jobs],
            available_host_bytes_before=available,optimizer_updates=0,config_sha256=sha(config_path),source_sha256=sha(__file__)))
    print(status,'CPU loss components',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['config','output','results']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();run(args.config.resolve(),args.output.resolve(),args.results.resolve())
