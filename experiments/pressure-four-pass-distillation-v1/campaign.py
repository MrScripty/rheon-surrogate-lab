"""Bound independent CPU episodes and own cancellation through native drainage."""
import argparse
import asyncio
import os
from pathlib import Path
import signal
import time

from common import ROOT, HERE, environment, read, write


async def campaign(args):
    config=read(args.config);runtime=read(ROOT/config['runtime'])
    if args.output.exists():
        raise ValueError('Fresh episode campaign required')
    available=int(next(l.split()[1]for l in Path('/proc/meminfo').read_text().splitlines()if l.startswith('MemAvailable:')))*1024
    workers=min(config['maximum_cpu_workers'],(available-config['reserve_memory_bytes'])//config['cpu_worker_memory_bytes'])
    if workers<1:
        raise MemoryError('Episode memory reserve unavailable')
    cpus=sorted(os.sched_getaffinity(0))[-workers:]
    pool=asyncio.Queue()
    for cpu in cpus:
        pool.put_nowait(cpu)
    scenes=sorted(args.scenes.glob('scene-*.json'))
    if len(scenes)!=8:
        raise ValueError('Exact eight predeclared independent setups required')
    args.output.mkdir();rows=[];tasks=[];children=set();started=time.perf_counter()
    owner=asyncio.current_task();loop=asyncio.get_running_loop();loop.add_signal_handler(signal.SIGTERM,owner.cancel)
    async def one(index,scene):
        cpu=await pool.get();child=None
        try:
            folder=args.output/f'scene-{index:02d}'
            command=[runtime['python'],'-B',str(HERE/'episode.py'),'--material',str(args.material.resolve()),
                     '--scene',str(scene.resolve()),'--output',str(folder.resolve()),'--cpu',str(cpu)]
            if args.id:
                command+=['--id',args.id,'--passes',str(args.passes)]
            with(args.output/f'scene-{index:02d}.log').open('xb')as log:
                child=await asyncio.create_subprocess_exec(*command,env=environment(runtime),stdout=log,
                        stderr=asyncio.subprocess.STDOUT,start_new_session=True)
                children.add(child)
                code=await asyncio.wait_for(child.wait(),660)
            if code:
                raise RuntimeError('Episode failed; retained scene log '+str(folder))
            record=read(folder/'episode.json')
            rows.append(dict(scene=index,status=record['status'],cpu=cpu,command=command,
                             elapsed_s=record['elapsed_s'],native_reaped=record['native_reaped']))
            write(args.output/'progress.json',dict(completed=len(rows),total=len(scenes),rows=rows))
            print('CPU episode',index,record['status'],flush=True)
        finally:
            if child is not None and child.returncode is None:
                try:
                    os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(child.wait(),15)
                except asyncio.TimeoutError:
                    try:
                        os.killpg(child.pid,signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    await child.wait()
            if child is not None:
                children.discard(child)
            pool.put_nowait(cpu)
    status='failed'
    try:
        tasks=[asyncio.create_task(one(i,scene))for i,scene in enumerate(scenes)]
        await asyncio.gather(*tasks);status='completed'
    finally:
        for task in tasks:
            if not task.done()and not task.cancelling():
                task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
        loop.remove_signal_handler(signal.SIGTERM)
        write(args.output/'execution.json',dict(status=status,elapsed_s=time.perf_counter()-started,
             maximum_workers=workers,cpus=cpus,rows=rows,all_owned_children_reaped=not children,
             model=args.id or'native-control',passes=args.passes))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['config','material','scenes','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--id');parser.add_argument('--passes',type=int)
    asyncio.run(campaign(parser.parse_args()))
