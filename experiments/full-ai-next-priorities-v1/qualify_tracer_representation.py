"""Native transport and displacement-readout controls on consumed training scenes."""
import argparse
import os
from pathlib import Path
import statistics
import numpy as np
import torch
from offline_tracer_readout import native,readout,factorized_weights
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run(args):
    if args.output.exists():raise ValueError('Fresh tracer representation control required')
    os.sched_setaffinity(0,{22});torch.set_num_threads(1)
    protocol=DOCS/'tracer-representation-protocol-01.json';reg=read(protocol)
    for name,digest in reg['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Registered readout/teacher source changed')
    rng=np.random.Generator(np.random.PCG64(2026102023));pending=[];sources={}
    for row in read(BASE/'temporal-data-01/bank.json')['trajectories']:
        if row['split']!='train':continue
        receipt_path=ROOT/row['receipt']
        if sha(receipt_path)!=row['receipt_sha256']:raise ValueError('Training receipt changed')
        captured=read(receipt_path);scene=captured['scene'];counts=scene['counts'];cells=int(np.prod(counts));h=1/scene['cells_per_metre']
        sources[str(receipt_path.relative_to(ROOT))]=sha(receipt_path)
        for start in reg['starts']:
            states=[]
            for step in (start,start+1):
                spec=captured['states'][step];path=receipt_path.parent/spec['path']
                if sha(path)!=spec['sha256']:raise ValueError('Captured tracer input/target changed')
                sources[str(path.relative_to(ROOT))]=sha(path)
                with np.load(path,allow_pickle=False)as arrays:states.append({n:arrays[n].copy()for n in ('tracer','vx','vy','vz')})
            current,target=states
            cfl=max(float(np.abs(current[n]).max())for n in ('vx','vy','vz'))/(60*h)
            if cfl>reg['maximum_CFL']:raise ValueError('Subcell diagnostic scope exceeded')
            tracers=[current['tracer'],np.zeros(cells,dtype='f4'),np.full(cells,.25,dtype='f4'),rng.random(cells).astype('f4')]
            packet=dict(counts=counts,cell_size_m=h,dt_s=1/60,
                velocity=[current[n].astype('f8').tolist()for n in ('vx','vy','vz')],
                tracers=[v.astype('f8').tolist()for v in tracers])
            pending.append((packet,scene,current,target,tracers,dict(identifier=row['identifier'],start=start,CFL=cfl)))
    rows=[]
    with torch.inference_mode():
        for start in range(0,len(pending),4):
            batch=pending[start:start+4];results=native([p[0]for p in batch])
            for (_,scene,current,target,tracers,identity),result in zip(batch,results):
                counts=scene['counts'];shape=(1,1,*counts[::-1])
                if not np.array_equal(np.asarray(result['tracers'][0],dtype='f4').view('u4'),target['tracer'].view('u4')):
                    raise ValueError('Native tracer primitive does not reproduce captured next tracer')
                shift=torch.tensor(result['offsets_cells'],dtype=torch.float64).reshape(1,3,*counts[::-1])
                weights=factorized_weights(shift)
                if float((weights.sum(1)-1).abs().max())>1e-12 or float(weights.min())<0 or int((weights>0).sum(1).max())>8:
                    raise ValueError('Convex eight-donor support failed')
                euler=-torch.tensor(result['center_velocity'],dtype=torch.float64).reshape_as(shift)*(scene['cells_per_metre']/60)
                for pattern,values,reference in zip(('captured','zero','constant','random'),tracers,result['tracers']):
                    truth=torch.tensor(reference,dtype=torch.float32).reshape(shape);t=torch.from_numpy(values).reshape(shape)
                    prediction=readout(t,shift);approx=readout(t,euler)
                    norm=max(float(truth.double().norm()),1e-30)
                    exact_error=float((prediction-truth).double().norm())/norm
                    euler_error=float((approx-truth).double().norm())/norm
                    if exact_error>reg['native_readout_relative_tolerance']:raise ValueError('Factorized native-displacement readout failed')
                    if pattern=='zero'and bool((prediction!=0).any()):raise ValueError('Zero tracer changed')
                    if pattern=='constant'and not torch.equal(prediction,t):raise ValueError('Constant tracer changed')
                    rows.append(dict(**identity,counts_xyz=counts,pattern=pattern,
                        native_displacement_readout_relative_error=exact_error,offline_Euler_readout_relative_error=euler_error,
                        RK2_minus_Euler_maximum_cell_displacement=float((shift-euler).abs().max()),native_reference_displacement=True))
    summaries=[]
    for pattern in ('captured','zero','constant','random'):
        selected=[r for r in rows if r['pattern']==pattern]
        summaries.append(dict(pattern=pattern,queries=len(selected),
            native_displacement_worst=max(r['native_displacement_readout_relative_error']for r in selected),
            offline_Euler_median=statistics.median(r['offline_Euler_readout_relative_error']for r in selected),
            offline_Euler_worst=max(r['offline_Euler_readout_relative_error']for r in selected)))
    write(args.output,dict(status='passed_offline_representation_qualification',rows=rows,summaries=summaries,
        native_primitive_matches_captured_next_tracer_bitwise=True,at_most_eight_active_donors=True,
        maximum_CFL=max(r['CFL']for r in rows),source_evidence=sources,source_sha256=reg['source_sha256'],
        protocol_sha256=sha(protocol),native_build_receipt_sha256=sha(BASE/'tracer-teacher-build-01/build-receipt.json'),
        native_tracer_primitive_calls=len(rows),native_pressure_solves=0,native_steps_executed=0,new_optimizer_updates=0,
        candidate_inference_implemented=False,candidate_frontier_point=False,CPU_latency_claim=False,
        training_queries_only=True,reference_displacements_only_offline=True,reserved_scenes_consumed=False))
    print('Qualified offline native-displacement tracer readout:',len(rows),'queries; not a learned predictor',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
