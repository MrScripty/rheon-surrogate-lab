"""Profile the actual C107 path and freeze train-only request priorities."""
import math
import os
from pathlib import Path
import statistics
import time
from collections import defaultdict
from study_support import ROOT,BASE,DOCS,read,write,sha,load_checkpoint,historical_cases,equation_error

def tier(parent):
    n=max(parent['counts_xyz'])
    return 16 if n<=16 else 32 if n<=32 else 64 if n<=64 else 128

def run():
    output=BASE/'pressure-profile-03'
    if output.exists():raise ValueError('Fresh profile/ranking output required')
    output.mkdir(parents=True)
    os.sched_setaffinity(0,{22})
    state=load_checkpoint('C107');torch=state.torch
    import train
    project=ROOT/state.entry['project']
    slots={r['id']:ROOT/r['path']for r in read(project/'project.json')['data']}
    _,parents=state.data.load(tuple(slots[k].read_bytes()for k in ['dataset','dataset_extra',*train.VIEW_SLOTS]),read(slots['splits']))
    if any(p['entry']['split']!='train'for p in parents):raise ValueError('Train-only ranking required')
    inventory=state.data.helpers().state_inventory(state.network)
    records=[]
    with torch.inference_mode():
        for i,parent in enumerate(parents):
            for frame in range(len(parent['frames'])):
                sample=state.data.sample(parent,frame)
                prediction=state.model.predict_pressure(state.network,sample,state.normalization)
                records.append(dict(parent=i,frame=frame,trajectory_id=parent['trajectory_id'],
                    native_frame=parent['frames'][frame]['index'],split='train',tier=tier(parent),
                    counts_xyz=parent['counts_xyz'],cell_size_m=parent['cell_size_m'],
                    equation_relative_residual=equation_error(state,prediction,sample)))
                print('Ranked train request',len(records),'tier',tier(parent),flush=True)
    for band in sorted(set(r['tier']for r in records)):
        group=sorted((r for r in records if r['tier']==band),key=lambda r:(-r['equation_relative_residual'],r['parent'],r['frame']))
        for rank,row in enumerate(group):row['multiplier']=4. if rank<math.ceil(len(group)/4)else 1.
    ranking=dict(status='frozen_from_training_only',source_entry=state.entry,records=records,
        original_splits_sha256=sha(slots['splits']),multiplier_cap=4.,
        algorithm='Top quarter within each size tier receives multiplier4; others1. Base mass is uniform parent then uniform frame.',
        development_or_reserved_requests_used=False,optimizer_updates=0,source_sha256=sha(__file__))
    write(output/'train-ranking.json',ranking)
    cases=historical_cases(state)
    with torch.inference_mode():
        difficult=sorted(cases,key=lambda pair:equation_error(state,state.model.predict_pressure(state.network,pair[1],state.normalization),pair[1]),reverse=True)[:4]
    chosen=[cases[0],*difficult]
    import physical_residual as physical
    measurements=[]
    for identity,sample in chosen:
        counters=defaultdict(lambda:dict(seconds=0.,calls=0));handles=[];originals=[]
        def wrap(module,name,label):
            original=getattr(module,name)
            def timed(*args,**kwargs):
                tick=time.perf_counter()
                try:return original(*args,**kwargs)
                finally:counters[label]['seconds']+=time.perf_counter()-tick;counters[label]['calls']+=1
            originals.append((module,name,original));setattr(module,name,timed)
        wrap(physical.coarse,'restrict','low_mode_restriction')
        wrap(physical.coarse,'prolong','low_mode_reconstruction')
        wrap(state.model.owner,'full_operator','pressure_operator')
        wrap(physical.historical_physical,'predict_pressure','fine_stage_including_preprocessing_and_reconstruction')
        for name,module in state.network.named_modules():
            if isinstance(module,torch.nn.Conv3d)or module is state.network or name=='shape_head':
                label='network'if module is state.network else'global_shape_head'if name=='shape_head'else'convolutions'
                def pre(mod,args,label=label):mod._study_profile_tick=time.perf_counter()
                def post(mod,args,out,label=label):
                    counters[label]['seconds']+=time.perf_counter()-mod._study_profile_tick;counters[label]['calls']+=1
                handles.extend([module.register_forward_pre_hook(pre),module.register_forward_hook(post)])
        try:
            with torch.inference_mode():
                tick=time.perf_counter();pressure=state.model.predict_pressure(state.network,sample,state.normalization);total=time.perf_counter()-tick
        finally:
            for handle in handles:handle.remove()
            for module,name,original in reversed(originals):setattr(module,name,original)
        with torch.inference_mode():
            residual=state.model.full_operator(pressure,sample['cell_size_m'],sample['rho'])-sample['rhs']
            boundary=torch.zeros_like(residual,dtype=torch.bool)
            for axis in (2,3,4):
                a=[slice(None)]*5;a[axis]=0;boundary[tuple(a)]=True;a[axis]=-1;boundary[tuple(a)]=True
            import numerics
            axes,_=numerics.geometry(tuple(residual.shape[2:]),3)
            coefficients=numerics.coefficients(residual,axes)
            norm=float(torch.linalg.vector_norm(residual));low=float(torch.linalg.vector_norm(coefficients))
            reference=sample['reference'];difference=pressure-reference;difference=difference-difference.mean()
            measurements.append(dict(**identity,total_instrumented_seconds=total,components=dict(counters),
                boundary_residual_l2=float(torch.linalg.vector_norm(residual[boundary])),
                interior_residual_l2=float(torch.linalg.vector_norm(residual[~boundary])),
                low_mode_residual_norm_fraction=low/max(norm,1e-30),
                mean_free_pressure_error_l2=float(torch.linalg.vector_norm(difference)),
                equation_relative_residual=equation_error(state,pressure,sample)))
    with torch.inference_mode(),torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],profile_memory=True,record_shapes=True)as profiler:
        state.model.predict_pressure(state.network,chosen[0][1],state.normalization)
    operators=[dict(name=e.key,calls=e.count,self_cpu_time_us=e.self_cpu_time_total,
                    total_cpu_time_us=e.cpu_time_total,self_cpu_memory_bytes=e.self_cpu_memory_usage)
               for e in profiler.key_averages()]
    if inventory!=state.data.helpers().state_inventory(state.network):raise ValueError('Profile changed model weights')
    write(output/'profile.json',dict(status='completed',source_entry=state.entry,rows=measurements,
        operators=sorted(operators,key=lambda e:-e['self_cpu_time_us']),
        caveat='Instrumented component times include profiling overhead and nested totals; use uninstrumented balanced timings for speed decisions.',
        cpu=22,threads=1,source_sha256=sha(__file__),optimizer_updates=0,unchanged_inventory=True))
    print('Completed actual prediction profile and train-only priorities',flush=True)

if __name__=='__main__':run()
