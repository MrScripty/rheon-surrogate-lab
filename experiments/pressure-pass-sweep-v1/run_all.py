"""Own bounded CPU workers through completion, cancellation and process reaping."""
import argparse
import asyncio
import fcntl
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import time
from material import read,sha


async def run(material_path):
    material=read(material_path);config=material['config'];root=Path(material['root'])
    runtime=read(root/config['runtime'])
    cpus=sorted(os.sched_getaffinity(0))
    available=int(next(line.split()[1]for line in Path('/proc/meminfo').read_text().splitlines()if line.startswith('MemAvailable:')))*1024
    memory_workers=(available-config['reserve_memory_bytes'])//config['worker_memory_bytes']
    retained=[];pending=[]
    for model in material['models']:
        receipt=material_path.parent/model['id']/'receipt.json'
        if receipt.exists() and read(receipt)['status']=='passed':retained.append(model['id'])
        else:pending.append(model)
    maximum=min(len(cpus),config['maximum_workers'],memory_workers,max(1,len(pending)))
    if maximum<1:raise RuntimeError('RAM reserve leaves no admissible CPU worker')
    environment={**os.environ,'PYTHONPATH':runtime['site_packages'],'CUDA_VISIBLE_DEVICES':'','TULDOK_TORCH_DEVICE':'cpu',
        'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','RAYON_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
    workers=asyncio.Queue()
    for cpu in cpus[:maximum]:workers.put_nowait(cpu)
    started=time.perf_counter();children=set();rows=[]
    loop=asyncio.get_running_loop();owner=asyncio.current_task()
    loop.add_signal_handler(signal.SIGTERM,owner.cancel)
    (material_path.parent/'owner.json').write_text(json.dumps(dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),material_sha256=sha(material_path))))
    (material_path.parent/'execution.json').write_text(json.dumps(dict(status='running',maximum_workers=maximum,retained_checkpoints=retained)))
    async def one(model):
        cpu=await workers.get();child=None
        try:
            log=material_path.parent/(model['id']+'.log')
            if log.exists():
                archive=material_path.parent/'interrupted-logs';archive.mkdir(exist_ok=True)
                log.rename(archive/(log.name+'-'+str(time.time_ns())))
            with log.open('xb')as stream:
                child=await asyncio.create_subprocess_exec(runtime['python'],str(Path(__file__).with_name('worker.py')),
                    '--material',str(material_path),'--id',model['id'],'--cpu',str(cpu),'--resume',
                    stdout=stream,stderr=asyncio.subprocess.STDOUT,env=environment,start_new_session=True)
                children.add(child)
                code=await child.wait()
            row=dict(id=model['id'],pid=child.pid,cpu=cpu,returncode=code,reaped=True)
            rows.append(row);print(json.dumps(row),flush=True)
            if code:raise RuntimeError('Checkpoint worker failed: '+model['id'])
        finally:
            if child is not None and child.returncode is None:
                try:child.terminate()
                except ProcessLookupError:pass
                try:await asyncio.wait_for(child.wait(),15)
                except asyncio.TimeoutError:
                    try:child.kill()
                    except ProcessLookupError:pass
                    await child.wait()
            if child is not None:children.discard(child)
            workers.put_nowait(cpu)
    tasks=[asyncio.create_task(one(model))for model in pending]
    status='failed'
    try:
        await asyncio.gather(*tasks);status='completed'
    except asyncio.CancelledError:
        status='cancelled'
        raise
    finally:
        for task in tasks:
            # gather already propagates owner cancellation to its children.
            # Cancelling again can interrupt their termination/reaping finally.
            if not task.done() and not task.cancelling():task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
        loop.remove_signal_handler(signal.SIGTERM)
        receipt=dict(status=status,elapsed_s=time.perf_counter()-started,maximum_workers=maximum,available_memory_at_start=available,retained_checkpoints=retained,
            cpu_pool=cpus[:maximum],workers=rows,all_owned_children_reaped=not children,
            material_sha256=sha(material_path),runtime_sha256=sha(root/config['runtime']),environment={k:v for k,v in environment.items()if k in ['PYTHONPATH','CUDA_VISIBLE_DEVICES','TULDOK_TORCH_DEVICE','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','RAYON_NUM_THREADS']},
            sources={p.name:sha(p)for p in Path(__file__).parent.glob('*.py')},completed_utc=datetime.now(timezone.utc).isoformat())
        (material_path.parent/'execution.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--material',type=Path,required=True)
    args=parser.parse_args()
    with (args.material.parent/'.owner.lock').open('a')as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        asyncio.run(run(args.material.resolve()))
