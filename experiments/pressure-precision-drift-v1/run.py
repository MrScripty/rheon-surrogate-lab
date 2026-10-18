"""Qualify the fixture, then own bounded matched-precision native processes."""
import argparse
import asyncio
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import signal
import time
from build import read,sha


async def run(build,output):
    material=read(build/'material.json');root=Path(material['root']);config=material['config'];receipt=read(build/'build-receipt.json')
    binary=root/receipt['binary']
    if sha(binary)!=receipt['sha256']:raise ValueError('Precision binary changed')
    for path,expected in material['generated_sources'].items():
        if sha(build/'source'/path)!=expected:raise ValueError('Closed research source changed: '+path)
    taskset=shutil.which('taskset')
    if not taskset:raise ValueError('Explicit native process CPU affinity requires taskset')
    output.mkdir(parents=True,exist_ok=False)
    (output/'material.json').write_text(json.dumps(dict(build=str(build.relative_to(root)),build_sha256=sha(build/'build-receipt.json'),material_sha256=sha(build/'material.json')),indent=2)+'\n')
    cpus=sorted(os.sched_getaffinity(0));available=int(next(l.split()[1]for l in Path('/proc/meminfo').read_text().splitlines()if l.startswith('MemAvailable:')))*1024
    maximum=min(config['maximum_workers'],len(cpus),max(0,(available-12*1024**3)//(512*1024**2)))
    if maximum<1:raise ValueError('RAM reserve does not admit a precision worker')
    environment={**os.environ,'RAYON_NUM_THREADS':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''}
    pool=asyncio.Queue()
    for cpu in cpus[:maximum]:pool.put_nowait(cpu)
    children=set();rows=[];tasks=[];start=time.perf_counter();owner=asyncio.current_task();loop=asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM,owner.cancel)
    (output/'owner.json').write_text(json.dumps(dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],binary_sha256=receipt['sha256'])))
    (output/'execution.json').write_text(json.dumps(dict(status='running',completed=0,total=len(material['scenes']),phase='fixture')))
    async def one(identifier,scene,cpu):
        folder=output/identifier;folder.mkdir();child=None;started=time.perf_counter()
        command=[taskset,'-c',str(cpu),str(binary),str(scene),str(folder/'frames.jsonl')]
        try:
            with(folder/'stderr.log').open('wb')as log:
                child=await asyncio.create_subprocess_exec(*command,env=environment,stdout=log,stderr=asyncio.subprocess.STDOUT,start_new_session=True)
                children.add(child)
                code=await asyncio.wait_for(child.wait(),config['worker_timeout_s'])
            if code:raise RuntimeError('Precision case failed '+identifier+'; inspect '+str(folder/'stderr.log'))
            records=[json.loads(line)for line in(folder/'frames.jsonl').read_bytes().splitlines()]
            last=records[-1];expected=read(scene)['steps']
            if (last['kind']!='complete' or last['steps']!=expected or len(records)!=expected+1
                or last['canonical_f32_exact_steps']!=expected or last['f64_repeat_exact_steps']!=expected):
                raise ValueError('Precision case did not complete and qualify '+identifier)
            row=dict(id=identifier,status='passed',elapsed_s=time.perf_counter()-started,cpu=cpu,command=command,
                process=dict(pid=child.pid,returncode=code,reaped=True),native=last,frames_sha256=sha(folder/'frames.jsonl'),
                scene_sha256=sha(scene),binary_sha256=receipt['sha256'])
            (folder/'receipt.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
            print(json.dumps(dict(id=identifier,status='passed',elapsed_s=row['elapsed_s'],verified_cast_values=last['verified_cast_values'])),flush=True)
            return row
        finally:
            if child is not None and child.returncode is None:
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:await asyncio.wait_for(child.wait(),5)
                except asyncio.TimeoutError:
                    try:os.killpg(child.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    await child.wait()
            if child is not None:children.discard(child)
    async def case(row):
        cpu=await pool.get()
        try:
            await one(row['id'],root/row['path'],cpu)
            (output/'execution.json').write_text(json.dumps(dict(status='running',completed=len(rows)-1,total=len(material['scenes']),phase='scenes')))
        finally:pool.put_nowait(cpu)
    status='failed'
    try:
        await one('fixture',build/'fixture-scene.json',cpus[0])
        tasks=[asyncio.create_task(case(row))for row in material['scenes']]
        await asyncio.gather(*tasks);status='completed'
    except asyncio.CancelledError:
        status='cancelled';raise
    finally:
        for task in tasks:
            if not task.done()and not task.cancelling():task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
        loop.remove_signal_handler(signal.SIGTERM)
        result=dict(status=status,elapsed_s=time.perf_counter()-start,maximum_workers=maximum,cpu_pool=cpus[:maximum],
            available_memory_at_start=available,workers=rows,all_owned_children_reaped=not children,
            binary_sha256=receipt['sha256'],material_sha256=sha(build/'material.json'),
            environment={k:environment[k]for k in ['RAYON_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES']},
            source_sha256=sha(__file__),completed_utc=datetime.now(timezone.utc).isoformat())
        (output/'execution.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=status,elapsed_s=result['elapsed_s'],all_owned_children_reaped=result['all_owned_children_reaped'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();asyncio.run(run(args.build.resolve(),args.output.resolve()))
