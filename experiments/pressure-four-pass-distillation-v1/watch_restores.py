"""Qualify each completed CPU restore while subsequent CUDA jobs continue."""
import argparse
import asyncio
import os
from pathlib import Path
import signal
import time

from common import ROOT, HERE, environment, read, write
from cpu import material


async def watch(config_path,projects,output):
    config=read(config_path);runtime=read(ROOT/config['runtime'])
    output.mkdir(exist_ok=False);pending={a['id']:i+57 for i,a in enumerate(config['arms'])}
    cpus=asyncio.Queue()
    for cpu in sorted(os.sched_getaffinity(0))[-config['maximum_cpu_workers']:]:
        cpus.put_nowait(cpu)
    tasks=[];children=set();rows=[];started=time.perf_counter()
    owner=asyncio.current_task();loop=asyncio.get_running_loop();loop.add_signal_handler(signal.SIGTERM,owner.cancel)
    async def one(arm,number):
        cpu=await cpus.get();child=None
        try:
            selected=projects/(arm+'-cpu-material.json')
            material(config_path,projects,selected,True)
            identifier='C'+str(number)
            command=[runtime['python'],'-B',str(HERE/'cpu.py'),'snapshots','--material',str(selected),
                     '--id',identifier,'--cpu',str(cpu),'--output',str(output/identifier)]
            with(output/(identifier+'.log')).open('xb')as log:
                child=await asyncio.create_subprocess_exec(*command,env=environment(runtime),stdout=log,
                           stderr=asyncio.subprocess.STDOUT,start_new_session=True)
                children.add(child)
                code=await asyncio.wait_for(child.wait(),180)
            if code:
                raise RuntimeError('CPU qualification failed: '+identifier)
            result=read(output/identifier/'snapshots.json')
            rows.append(dict(arm=arm,id=identifier,status='qualified',cpu=cpu,
                  elapsed_s=result['elapsed_s'],summary=result['summary'],command=command,reaped=True))
            write(output/'progress.json',dict(rows=rows,pending=list(pending)))
            print('Restored model qualified',identifier,flush=True)
        finally:
            if child is not None and child.returncode is None:
                try:
                    os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(child.wait(),10)
                except asyncio.TimeoutError:
                    child.kill();await child.wait()
            if child is not None:
                children.discard(child)
            cpus.put_nowait(cpu)
    status='failed'
    try:
        while pending:
            for arm,number in list(pending.items()):
                directory=projects/(arm+'-study-01')
                if (directory/'receipt.json').exists():
                    tasks.append(asyncio.create_task(one(arm,number)));pending.pop(arm)
                elif (directory/'failure.json').exists() or (projects/'controller-execution.json').exists():
                    rows.append(dict(arm=arm,id='C'+str(number),status='no_successful_restore'));pending.pop(arm)
            if time.perf_counter()-started>4000:
                raise TimeoutError('Restore observation expired; do not resubmit managed work')
            if pending:
                await asyncio.sleep(2)
        await asyncio.gather(*tasks);status='completed'
    finally:
        for task in tasks:
            if not task.done()and not task.cancelling():
                task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
        loop.remove_signal_handler(signal.SIGTERM)
        write(output/'execution.json',dict(status=status,elapsed_s=time.perf_counter()-started,rows=rows,
                       scope='Includes waiting for managed CPU restore; individual qualification timings are separate',
                       all_owned_children_reaped=not children))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['config','projects','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    asyncio.run(watch(args.config.resolve(),args.projects.resolve(),args.output.resolve()))
