"""Measure checkpoint latency serially after the throughput campaign drains."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from material import read,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--material',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--id')
    p.add_argument('--wait',action='store_true')
    args=p.parse_args();material=read(args.material);config=read(args.config);root=Path(material['root']);run=args.material.parent
    if args.id:
        import inference
        import worker
        os.sched_setaffinity(0,{config['cpu']});start=time.perf_counter()
        state=inference.load(material,args.id)
        local={**material,'config':{**material['config'],'timing_repetitions':config['repetitions_per_case'],'warmup_each_case':True}}
        output=run/'isolated'/args.id;output.mkdir(parents=True,exist_ok=False)
        worker.snapshots(state,local,output)
        (output/'receipt.json').write_text(json.dumps(dict(status='passed',id=args.id,elapsed_s=time.perf_counter()-start,
            cpu=config['cpu'],cuda_initialized=state.torch.cuda.is_initialized(),torch_threads=state.torch.get_num_threads(),
            material_sha256=sha(args.material),timing_config_sha256=sha(args.config),source_sha256=sha(__file__)))+'\n')
        return
    def stop(*_):raise KeyboardInterrupt('Isolated timing cancelled')
    signal.signal(signal.SIGTERM,stop)
    (run/'isolated').mkdir(exist_ok=True)
    (run/'isolated/owner.json').write_text(json.dumps(dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],source_sha256=sha(__file__))))
    deadline=time.monotonic()+material['config']['worker_timeout_seconds']
    execution=read(run/'execution.json')
    while args.wait and execution['status']=='running':
        if time.monotonic()>deadline:raise TimeoutError('Parallel campaign did not drain before timing deadline')
        time.sleep(2)
        execution=read(run/'execution.json')
    if execution['status']!='completed' or not execution['all_owned_children_reaped']:
        raise ValueError('Parallel workers must complete and drain before isolated latency measurements')
    runtime=read(root/material['config']['runtime']);env={**os.environ,'PYTHONPATH':runtime['site_packages'],
        'CUDA_VISIBLE_DEVICES':'','TULDOK_TORCH_DEVICE':'cpu','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
        'OPENBLAS_NUM_THREADS':'1','RAYON_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
    start=time.perf_counter();rows=[]
    (run/'isolated/execution.json').write_text(json.dumps(dict(status='running',completed=0,total=len(material['models']))))
    for model in material['models']:
        child=None
        try:
            child=subprocess.Popen([runtime['python'],str(Path(__file__).resolve()),'--material',str(args.material.resolve()),
                '--config',str(args.config.resolve()),'--id',model['id']],env=env,start_new_session=True)
            code=child.wait(timeout=config['worker_deadline_seconds'])
            if code:raise RuntimeError('Isolated timing failed '+model['id'])
            rows.append(dict(id=model['id'],pid=child.pid,returncode=code,reaped=True))
        finally:
            if child is not None and child.poll()is None:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
        print(json.dumps(dict(completed=len(rows),total=len(material['models']),elapsed_s=time.perf_counter()-start)),flush=True)
        (run/'isolated/execution.json').write_text(json.dumps(dict(status='running',completed=len(rows),total=len(material['models']),elapsed_s=time.perf_counter()-start)))
    (run/'isolated/execution.json').write_text(json.dumps(dict(status='completed',elapsed_s=time.perf_counter()-start,
        workers=rows,cpu=config['cpu'],source_sha256=sha(__file__),config_sha256=sha(args.config)),indent=2)+'\n')


if __name__=='__main__':main()
