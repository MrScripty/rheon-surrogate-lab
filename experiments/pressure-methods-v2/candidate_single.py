"""Frozen candidate policies in one native fluid track, all IPC and frame work counted."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time

START=time.perf_counter()
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'experiments/pressure-four-pass-distillation-v1'),
             str(ROOT/'experiments/pressure-pass-sweep-v1'),
             str(ROOT/'experiments/pressure-methods-v1')]
from common import read,write,sha

def run(args):
    if args.output.exists():raise ValueError('Fresh benchmark output required')
    os.sched_setaffinity(0,{args.cpu})
    args.output.mkdir(parents=True)
    scene=read(args.scene)
    build=read(args.build/'build-receipt.json')
    binary=args.build/'rheon-single-track'
    if build['status']!='qualified_build' or sha(binary)!=build['binary_sha256']:
        raise ValueError('Qualified single-track binary required')
    load_start=time.perf_counter()
    state=None
    if args.policy not in ('native','native-loose'):
        import torch
        torch.set_num_threads(1)
        import numerics
        if args.policy!='spectral':
            import inference
            material=read(ROOT/'runs/pressure-pass-sweep-v1/cpu-01/material.json')
            if args.policy in ('candidate','candidate-adaptive'):
                evaluated=read(args.prior)
                if evaluated['status']!='completed':raise ValueError('Completed candidate required')
                state=inference.load(dict(material,models=[evaluated['entry']]),evaluated['entry']['id'])
                state.project=ROOT/state.entry['project']
            elif args.policy=='coarse':
                evaluated=read(ROOT/'runs/pressure-methods-v1/evaluation/numerical-coarse-01/results.json')
                material=dict(material,models=[evaluated['entry']])
                state=inference.load(material,'numerical-coarse')
            else:state=inference.load(material,'C53')
        h=torch.tensor([1/scene['cells_per_metre']],dtype=torch.float64)
        rho=torch.tensor([1000.],dtype=torch.float64)
        dt=torch.tensor([1/60],dtype=torch.float64)
        shape=(1,1,*scene['counts'][::-1])
    load_seconds=time.perf_counter()-load_start
    sys.path.insert(0,str(ROOT/'runs/dependency-migration-v1/lab/experiments/global-pressure-v1'))
    spec=importlib.util.spec_from_file_location('bounded_native',ROOT/'runs/dependency-migration-v1/lab/experiments/global-pressure-v1/capture.py')
    owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
    command=[str(binary)]
    if args.policy.startswith('native'):command+=['--native']
    if args.policy=='native-loose':command+=['--native-1e-6']
    frames=[];callbacks=[];network_calls=[];ready=None;complete=False
    hook=state.network.register_forward_hook(lambda *_:network_calls.append(1)) if state else None
    tick=time.perf_counter()
    child=None
    try:
        with (args.output/'stderr.log').open('xb') as stderr:
            with owner.bounded_process(command,args.timeout,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                    stderr=stderr,env={**os.environ,'RAYON_NUM_THREADS':'1'}) as child:
                child.stdin.write(json.dumps(scene).encode()+b'\n');child.stdin.flush()
                while True:
                    raw=child.stdout.readline(32*1024*1024+1)
                    if not raw or len(raw)>32*1024*1024:raise ValueError('Complete bounded native packet required')
                    packet=json.loads(raw)
                    if packet['kind']=='ready':ready=packet
                    elif packet['kind']=='request':
                        callback_start=time.perf_counter();network_calls.clear()
                        base=torch.tensor(packet['base'],dtype=torch.float64).reshape(shape)
                        residual=torch.tensor(packet['residual'],dtype=torch.float64).reshape(shape)
                        with torch.no_grad():
                            if args.policy=='spectral':
                                pressure=base+numerics.correction(residual,h,rho,None)
                            else:
                                rhs=residual+state.model.full_operator(base,h,rho)
                                sample=dict(base=base,rhs=rhs,native_input_residual=residual,cell_size_m=h,rho=rho,dt=dt)
                                current=base
                                if args.policy=='before':
                                    current=base+numerics.correction(residual,h,rho,3)
                                pressure=current
                                if args.policy=='candidate-adaptive':
                                    import policies_physical
                                    pressure,_=policies_physical.run(state,sample,nodes=6,placement='each',max_passes=args.passes,adaptive=True)
                                else:
                                    for _ in range(args.passes):pressure=inference.step(state,sample,pressure)
                        payload=json.dumps(pressure.flatten().tolist(),allow_nan=False).encode()+b'\n'
                        child.stdin.write(payload);child.stdin.flush()
                        callbacks.append(dict(seconds=time.perf_counter()-callback_start,
                            network_calls=len(network_calls),reply_bytes=len(payload),request_bytes=len(raw)))
                    elif packet['kind']=='frame':frames.append(packet['frame'])
                    elif packet['kind']=='complete':complete=True;break
                    else:raise ValueError('Unknown packet kind')
                child.stdin.close()
                if child.stdout.read(1)!=b'' or child.wait(timeout=10)!=0:
                    raise ValueError('Native worker failed to drain')
    finally:
        if hook:hook.remove()
    wall=time.perf_counter()-tick
    active=[f for f in frames if f['step']>scene['warmup_steps'] and f['advanced']]
    successful=complete and len(frames)==scene['steps'] and all(f['advanced'] for f in frames)
    result=dict(status='completed' if successful else 'pressure_rejection',policy=args.policy,passes=args.passes,
        scene=scene,cpu=args.cpu,threads=1,model_load_seconds=load_seconds,native_setup=ready,
        simulation_wall_seconds=wall,worker_total_seconds=time.perf_counter()-START,
        frames=frames,callbacks=callbacks,completed_steps=sum(f['advanced'] for f in frames),
        post_warmup_steps=len(active),source_sha256=sha(__file__),scene_sha256=sha(args.scene),
        build_receipt_sha256=sha(args.build/'build-receipt.json'),
        native_process=dict(pid=child.pid,returncode=child.returncode,reaped=child.poll() is not None),
        pressure_gate=1e-6,native_solver_tolerance=1e-6 if args.policy=='native-loose' else 1e-8,
        timing_scope='One fluid track; frame includes preparation, pressure request/reply transfers and host projection/gates. Wall includes native startup/setup and output IO; worker total additionally includes Python/model loading.',
        peak_worker_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        peak_native_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    result['post_warmup_median_ms']={key:1000*statistics.median(f[key] for f in active)
        for key in ['preparation_seconds','pressure_seconds','projection_seconds','frame_seconds']} if active else None
    result['state_identity']=state.entry if state else None
    if any(not math.isfinite(v) for f in frames for k,v in f.items() if k.endswith('_seconds')):
        raise ValueError('Nonfinite timings')
    write(args.output/'results.json',result)
    print(json.dumps({k:result[k] for k in ['status','policy','completed_steps','post_warmup_median_ms']}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',required=True,type=Path)
    parser.add_argument('--scene',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--policy',choices=['native','native-loose','spectral','neural','before','coarse','candidate','candidate-adaptive'],required=True)
    parser.add_argument('--prior',type=Path)
    parser.add_argument('--passes',type=int,choices=range(1,17),default=8)
    parser.add_argument('--cpu',type=int,default=23)
    parser.add_argument('--timeout',type=int,default=600)
    args=parser.parse_args()
    if args.policy in ('candidate','candidate-adaptive') and not args.prior:parser.error('Candidate prior required')
    run(args)
