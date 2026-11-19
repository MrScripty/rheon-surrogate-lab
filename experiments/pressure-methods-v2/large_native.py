"""Measure bounded native fluid steps with binary pressure transport and large grids."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'experiments/pressure-four-pass-distillation-v1'),
             str(ROOT/'experiments/pressure-pass-sweep-v1'),str(ROOT/'experiments/pressure-methods-v1')]
from common import read,write,sha

def read_field(stream,count):
    result=bytearray(count*8);view=memoryview(result);offset=0
    while offset<len(result):
        length=stream.readinto(view[offset:])
        if not length:raise ValueError('Incomplete binary pressure request')
        offset+=length
    return result

def available():
    return int(next(row.split()[1] for row in Path('/proc/meminfo').read_text().splitlines() if row.startswith('MemAvailable:')))*1024

def run(args):
    if args.output.exists():raise ValueError('Fresh large native measurement required')
    os.sched_setaffinity(0,{args.cpu})
    scene=read(args.scene);cells=math.prod(scene['counts'])
    if not 1<=scene['steps']<=4 or cells>256**3:raise ValueError('Bounded coupling scene required')
    if available()<12*1024**3+cells*512:raise MemoryError('Declared host reserve unavailable')
    build=read(args.build/'build-receipt.json');binary=args.build/'rheon-large-single-track'
    if build['status']!='qualified_build' or sha(binary)!=build['binary_sha256']:
        raise ValueError('Qualified binary protocol and native physical implementation required')
    for path,expected in build['sources'].items():
        if sha(ROOT/path)!=expected:raise ValueError('Bound large-grid source changed')
    args.output.mkdir(parents=True)
    state=None;wrapper=None;qualification=[];recurrence=[];entry=None
    start=time.perf_counter();calls=[]
    if args.kind!='native':
        import numpy as np
        import torch
        import numerics,numerical
        torch.set_num_threads(1)
        if args.kind not in ['spectral','pcg']:
            import inference,scalable
            material=read(ROOT/'runs/pressure-pass-sweep-v1/cpu-01/material.json')
            if args.prior:
                prior=read(args.prior)
                if prior['status']!='completed':raise ValueError('Completed evaluated checkpoint required')
                entry=prior['entry'];material=dict(material,models=[entry]);identifier=entry['id']
            else:identifier='C53'
            state=inference.load(material,identifier);entry=state.entry
            wrapper,qualification=scalable.qualify(state)
            recurrence=scalable.qualify_recurrence(state,wrapper,args.kind,nodes=args.nodes,passes=args.passes)
        h=torch.tensor([1/scene['cells_per_metre']],dtype=torch.float64)
        rho=torch.tensor([1000.],dtype=torch.float64);dt=torch.tensor([1/60],dtype=torch.float64)
    loading=time.perf_counter()-start
    sys.path.insert(0,str(ROOT/'runs/dependency-migration-v1/lab/experiments/global-pressure-v1'))
    spec=importlib.util.spec_from_file_location('large_native_process_owner',ROOT/'runs/dependency-migration-v1/lab/experiments/global-pressure-v1/capture.py')
    owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
    ready=None;frames=[];callbacks=[];complete=False;child=None;last_pressure=None;last_fields=None
    hook=wrapper.register_forward_hook(lambda *_:calls.append(1)) if wrapper else None
    command=[str(binary),*(['--native'] if args.kind=='native' else [])]
    start=time.perf_counter()
    try:
        with (args.output/'stderr.log').open('xb') as stderr:
            with owner.bounded_process(command,args.timeout,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                    stderr=stderr,env={**os.environ,'RAYON_NUM_THREADS':'1'}) as child:
                child.stdin.write(json.dumps(scene).encode()+b'\n');child.stdin.flush()
                while True:
                    raw=child.stdout.readline(65537)
                    if not raw or len(raw)>65536:raise ValueError('Complete bounded binary-wire header required')
                    packet=json.loads(raw)
                    if packet['kind']=='ready':ready=packet
                    elif packet['kind']=='predict':
                        callback_start=time.perf_counter();calls.clear()
                        if (packet['cells']!=cells or packet['counts_xyz']!=scene['counts'] or packet['dtype']!='<f8'
                            or packet['fields']!=['base','native_input_residual'] or packet['cell_size_m']!=1/scene['cells_per_metre']
                            or packet['rho']!=1000. or packet['dt']!=1/60):
                            raise ValueError('Binary pressure equation metadata differs')
                        base_raw=read_field(child.stdout,cells);residual_raw=read_field(child.stdout,cells)
                        shape=(1,1,*scene['counts'][::-1])
                        base=torch.from_numpy(np.frombuffer(base_raw,dtype='<f8').reshape(shape))
                        residual=torch.from_numpy(np.frombuffer(residual_raw,dtype='<f8').reshape(shape))
                        rhs=residual+numerics.fine_action(base,h,rho)
                        sample=dict(base=base,native_input_residual=residual,rhs=rhs,cell_size_m=h,rho=rho,dt=dt)
                        method_start=time.perf_counter()
                        with torch.no_grad():
                            if args.kind=='spectral':
                                pressure=base+numerics.correction(residual,h,rho,None);trace=[]
                            elif args.kind=='pcg':
                                pressure,iterations=numerical.pcg(sample,nodes=6,max_iterations=256)
                                trace=[dict(iterations=iterations)]
                            else:
                                pressure,trace=scalable.predict(wrapper,sample,state.normalization,passes=args.passes,
                                    kind=args.kind,nodes=args.nodes)
                        defect=float(torch.linalg.vector_norm(rhs-numerics.fine_action(pressure,h,rho)))/max(float(torch.linalg.vector_norm(rhs)),1e-300)
                        method_seconds=time.perf_counter()-method_start
                        payload=np.ascontiguousarray(pressure.numpy(),dtype='<f8')
                        child.stdin.write(memoryview(payload).cast('B'));child.stdin.flush()
                        callbacks.append(dict(seconds=time.perf_counter()-callback_start,method_seconds=method_seconds,
                            network_calls=len(calls),trace=trace,relative_residual=defect,
                            request_field_bytes=2*cells*8,reply_bytes=cells*8))
                        last_pressure=pressure;last_fields=(base_raw,residual_raw)
                    elif packet['kind']=='frame':frames.append(packet['frame'])
                    elif packet['kind']=='complete':complete=True;break
                    else:raise ValueError('Unknown binary-wire packet')
                child.stdin.close()
                if child.stdout.read(1)!=b'' or child.wait(timeout=10)!=0:
                    raise ValueError('Native binary worker failed to drain')
    finally:
        if hook:hook.remove()
    wall=time.perf_counter()-start
    peak_worker=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    completed=complete and len(frames)==scene['steps'] and all(frame['advanced'] for frame in frames)
    request_identity=None;prediction=None
    if last_fields:
        request_identity=dict(base_sha256=hashlib.sha256(last_fields[0]).hexdigest(),
            residual_sha256=hashlib.sha256(last_fields[1]).hexdigest(),counts_xyz=scene['counts'],cell_size_m=1/scene['cells_per_metre'],rho=1000.,dt=1/60)
    if args.save_pressure and last_pressure is not None:
        path=args.output/'prediction.npy';np.save(path,last_pressure.numpy())
        prediction=dict(path=path.name,sha256=sha(path),shape=list(last_pressure.shape),dtype='<f8')
    result=dict(status='completed' if completed else 'pressure_rejection',kind=args.kind,passes=args.passes,nodes=args.nodes,
        scene=scene,grid_dimensions=scene['counts'],cells_per_metre=scene['cells_per_metre'],
        physical_extent_m=[n/scene['cells_per_metre'] for n in scene['counts']],
        native_setup=ready,model_loading_seconds=loading,native_wall_seconds=wall,frames=frames,callbacks=callbacks,
        request_identity=request_identity,prediction=prediction,entry=entry,wrapper_qualification=qualification,recurrence_qualification=recurrence,
        peak_worker_rss_kib=peak_worker,peak_native_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        peak_worker_with_export_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        native_process=dict(pid=child.pid,returncode=child.returncode,reaped=child.poll() is not None),
        source_sha256=sha(__file__),scalable_source_sha256=sha(HERE/'scalable.py'),build_receipt_sha256=sha(args.build/'build-receipt.json'),
        scene_sha256=sha(args.scene),cpu=args.cpu,threads=1,optimizer_updates=0,
        scope='Native fluid preparation, pressure prediction, velocity projection and host physical gates for at most four steps; not a long-horizon qualification.',
        timing_scope='Native frame includes preparation, binary transfer/conversion, pressure method and projection/gates. Wall includes setup and stream IO. Pressure artifact export and hashing occur after measured wall; model loading separate.',
        concurrency_scope='Other owned campaign workers may be active; final isolated timing still required for a deployment claim')
    write(args.output/'results.json',result)
    print('Native scale probe',scene['counts'],scene['cells_per_metre'],args.kind,result['status'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--kind',choices=['native','spectral','pcg','fine','numerical-coarse','learned-coarse'],required=True)
    parser.add_argument('--prior',type=Path)
    parser.add_argument('--passes',type=int,choices=range(1,17),default=8)
    parser.add_argument('--nodes',type=int,choices=[4,6],default=6)
    parser.add_argument('--cpu',type=int,default=21)
    parser.add_argument('--timeout',type=int,default=1800)
    parser.add_argument('--save-pressure',action='store_true')
    args=parser.parse_args()
    for name in ['build','scene','output','prior']:
        if getattr(args,name):setattr(args,name,getattr(args,name).resolve())
    run(args)
