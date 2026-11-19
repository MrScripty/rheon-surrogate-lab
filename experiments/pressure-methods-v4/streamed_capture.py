"""Native full-state rollouts and exact same-request training labels, no fallback."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
from v4_support import ROOT,BASE,DOCS,read,write,sha,load

def run(args):
    if args.output.exists():raise ValueError('Fresh native episode required')
    def stop(*_):raise InterruptedError('Owned native episode stopped')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    os.sched_setaffinity(0,{args.cpu})
    reservation=read(DOCS/'scene-reservation.json')
    row=next(r for r in reservation['rows']if r['trajectory_id']==args.trajectory)
    if row['population']=='blind'and not (DOCS/'selection.json').exists():raise ValueError('Blind capture requires frozen selection')
    scene=row['scene'];cells=math.prod(scene['counts'])
    if scene['warmup_steps']!=0 or not 1<=scene['steps']<=600 or cells>256**3:raise ValueError('Bounded cold-start streamed scene required')
    import numpy as np,torch
    torch.set_num_threads(1)
    if args.policy!='spectral':
        state,wrapper,parity=load();inventory=state.data.helpers().state_inventory(state.network)
        branch={k:v.detach()for k,v in state.network.state_dict().items()if k=='coarse_logits'or k.startswith('shape_head.')}
    import numerics,multilevel,preconditioned
    from frozen_matrix import run_policy
    from large_native import available,read_field
    if available()<12*1024**3+cells*512:raise MemoryError('Host memory reserve unavailable')
    build=read(args.build/'build-receipt.json');binary=args.build/'rheon-large-single-track'
    if build['status']!='qualified_build'or sha(binary)!=build['binary_sha256']:raise ValueError('Qualified streamed build required')
    for name,expected in build['sources'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Native source changed')
    sys.path.insert(0,str(ROOT/'runs/dependency-migration-v1/lab/experiments/global-pressure-v1'))
    spec=importlib.util.spec_from_file_location('v4_native_process_owner',ROOT/'runs/dependency-migration-v1/lab/experiments/global-pressure-v1/capture.py')
    owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
    h=torch.tensor([1/scene['cells_per_metre']],dtype=torch.float64);rho=torch.tensor([1000.],dtype=torch.float64)
    args.output.mkdir(parents=True);frames=[];callbacks=[];snapshots=[];states=[];complete=False;pending=None;started=time.perf_counter()
    command=[str(binary)]+(['--state-fields']if args.states else[])
    with (args.output/'stderr.log').open('xb')as stderr:
        with owner.bounded_process(command,args.timeout,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,
                env={**os.environ,'RAYON_NUM_THREADS':'1'})as child:
            child.stdin.write(json.dumps(scene).encode()+b'\n');child.stdin.flush()
            while True:
                raw=child.stdout.readline(65537)
                if not raw or len(raw)>65536:raise ValueError('Complete bounded native header required')
                packet=json.loads(raw)
                if packet['kind']=='ready':setup=packet
                elif packet['kind']=='predict':
                    if packet['cells']!=cells or packet['counts_xyz']!=scene['counts']or packet['dtype']!='<f8'or packet['rho']!=1000.0 or packet['dt']!=1/60 or packet['cell_size_m']!=float(h[0]):
                        raise ValueError('Native equation metadata differs')
                    shape=(1,1,*scene['counts'][::-1]);base=torch.from_numpy(np.frombuffer(read_field(child.stdout,cells),dtype='<f8').reshape(shape))
                    residual=torch.from_numpy(np.frombuffer(read_field(child.stdout,cells),dtype='<f8').reshape(shape))
                    rhs=residual+numerics.fine_action(base,h,rho)
                    sample=dict(base=base,native_input_residual=residual,rhs=rhs,cell_size_m=h,rho=rho,dt=torch.tensor([1/60],dtype=torch.float64))
                    tick=time.perf_counter()
                    with torch.inference_mode():
                        if args.policy=='spectral':
                            pressure=base+numerics.correction(residual,h,rho,None);work=dict(network_calls=0)
                        elif args.policy=='learned':pressure,work=run_policy(wrapper,sample,state.normalization,branch,'learned',max_passes=args.passes)
                        elif args.policy.startswith('gcr-'):pressure,work=preconditioned.solve(wrapper,sample,state.normalization,branch,kind=args.policy[4:],max_iterations=args.passes)
                        else:pressure,work=multilevel.solve(wrapper,sample,state.normalization,max_passes=args.passes,neural=args.policy=='vcycle-neural')
                        pressure=pressure-pressure[:,:,:1,:1,:1].clone()
                    method_seconds=time.perf_counter()-tick
                    child.stdin.write(memoryview(np.ascontiguousarray(pressure.numpy(),dtype='<f8')).cast('B'));child.stdin.flush()
                    r=residual-numerics.fine_action(pressure-base,h,rho)
                    callbacks.append(dict(index=len(frames),method_seconds=method_seconds,work=work,
                        relative_residual=float(torch.linalg.vector_norm(r))/max(float(torch.linalg.vector_norm(rhs)),1e-300)))
                    pending=dict(base=base,rhs=rhs,native_input_residual=residual)
                    if len(frames)in row['snapshots']:
                        with torch.inference_mode():reference=base+numerics.correction(residual,h,rho,None)
                        pending['reference']=reference-reference[:,:,:1,:1,:1].clone()
                elif packet['kind']=='frame':
                    frame=packet['frame'];index=len(frames);frames.append(dict(**frame,preparation_profile=packet['preparation_profile']))
                    if index in row['snapshots']:
                        if pending is None or 'reference'not in pending:raise ValueError('Same-request labels missing')
                        arrays={}
                        for name,value in pending.items():
                            path=args.output/f'request-{index:03d}-{name}.f64';value.numpy().astype('<f8',copy=False).tofile(path)
                            arrays[name]=dict(path=path.name,bytes=path.stat().st_size,sha256=sha(path),shape=list(value.shape),dtype='<f8')
                        snapshots.append(dict(index=index,time_s=(index+1)/60,arrays=arrays,candidate_advanced=frame['advanced']))
                    pending=None
                elif packet['kind']=='state':
                    expected_vel=[(scene['counts'][axis]+1)*math.prod(scene['counts'][j]for j in range(3)if j!=axis)for axis in range(3)]
                    if packet['pressure_cells']!=cells or packet['tracer_cells']!=cells or packet['velocity_lengths']!=expected_vel:
                        raise ValueError('Full native state shape differs')
                    values={}
                    for name,size,dtype in [('pressure',cells,'<f8'),*[(name,n,'<f4')for name,n in zip(['vx','vy','vz'],expected_vel)],('tracer',cells,'<f4')]:
                        # Existing read_field reads size * 8 bytes; f32 sizes are not necessarily even.
                        count=size*np.dtype(dtype).itemsize;blocks=[]
                        while count:
                            block=child.stdout.read(min(count,65536))
                            if not block:raise EOFError('Truncated full native state')
                            blocks.append(block);count-=len(block)
                        values[name]=np.frombuffer(b''.join(blocks),dtype=dtype).copy()
                    path=args.output/f'state-{len(frames)-1:03d}.npz';np.savez_compressed(path,**values)
                    states.append(dict(index=len(frames)-1,path=path.name,sha256=sha(path),bytes=path.stat().st_size,
                        arrays={k:dict(dtype=v.dtype.str,shape=list(v.shape),sha256=hashlib.sha256(v.tobytes()).hexdigest())for k,v in values.items()}))
                elif packet['kind']=='complete':complete=True;break
                else:raise ValueError('Unknown streamed native packet')
            child.stdin.close()
            if child.stdout.read(1)!=b''or child.wait(timeout=10)!=0:raise ValueError('Native child did not drain')
    completed=complete and len(frames)==scene['steps']and all(f['advanced']for f in frames)
    if args.policy!='spectral'and inventory!=state.data.helpers().state_inventory(state.network):raise ValueError('Episode changed original weights')
    write(args.output/'receipt.json',dict(status='completed'if completed else'pressure_rejection',row=row,scene=scene,policy=args.policy,max_passes=args.passes,
        entry=state.entry if args.policy!='spectral'else None,frames=frames,callbacks=callbacks,snapshots=snapshots,states=states,setup=setup,
        native_process=dict(pid=child.pid,returncode=child.returncode,reaped=child.poll()is not None),cpu=args.cpu,threads=1,
        source_sha256=sha(__file__),build_receipt_sha256=sha(args.build/'build-receipt.json'),reservation_sha256=sha(DOCS/'scene-reservation.json'),
        elapsed_s=time.perf_counter()-started,peak_worker_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        peak_native_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,optimizer_updates=0,
        label_semantics='Full spectral solve of each actual input equation, including model-generated states; no reference-trajectory endpoint substitution.',
        timing_scope='Headless native fluid step; selected snapshot oracle labeling and full-state exports are diagnostics outside policy time. No GUI/rendering.'))
    print(args.trajectory,args.policy,'completed'if completed else'pressure_rejection',len(frames),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectory',required=True);parser.add_argument('--policy',choices=['spectral','learned','vcycle-neural','vcycle-only','gcr-learned','gcr-multilevel','gcr-multilevel-neural'],required=True)
    parser.add_argument('--build',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cpu',type=int,default=23);parser.add_argument('--passes',type=int,choices=[8,16],default=8)
    parser.add_argument('--timeout',type=int,default=1800);parser.add_argument('--states',action='store_true')
    run(parser.parse_args())
