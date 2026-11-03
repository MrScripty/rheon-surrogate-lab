"""Profile the complete qualified prototype without changing emitted outputs."""
import argparse
import os
from pathlib import Path
import statistics
import time
from profile_frozen_components import ROOT,read,write,digest
from recurrent_sampler_constructor import construct
import recurrent_sampler_model as model_module
from diagnose_causal_composition import inventory
import numpy as np
import torch


def run(args):
    if args.output.exists():raise ValueError('Fresh failed-cost prototype diagnosis required')
    reg=read(args.protocol)
    for path,expected in reg['source_sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError('Registered stage-profile source/evidence changed: '+path)
    os.sched_setaffinity(0,{reg['cpu']});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    model=construct(reg['library_path'],'seeded',reg['seed']);before=inventory(model);rows=[]
    original={name:getattr(model_module,name)for name in ('transport','spatial','forcing')}
    elapsed={};counts={};pressure_clock=[]
    def observe(name,fn):
        def wrapped(*a,**kw):
            key=name+str(a[0])if name=='spatial'else name;tick=time.perf_counter()
            try:return fn(*a,**kw)
            finally:elapsed[key]=elapsed.get(key,0.)+time.perf_counter()-tick;counts[key]=counts.get(key,0)+1
        return wrapped
    def pressure_begin(*unused):pressure_clock.append(time.perf_counter())
    def pressure_end(*unused):
        elapsed['pressure']=elapsed.get('pressure',0.)+time.perf_counter()-pressure_clock.pop();counts['pressure']=counts.get('pressure',0)+1
    with torch.inference_mode():
        try:
            for trajectory in read(ROOT/reg['bank'])['trajectories']:
                path=ROOT/trajectory['receipt'];receipt=read(path)
                if digest(path)!=trajectory['receipt_sha256']:raise ValueError('Consumed receipt changed')
                for start in reg['starts']:
                    entry=receipt['states'][start];source=path.parent/entry['path']
                    if digest(source)!=entry['sha256']:raise ValueError('Current state changed')
                    with np.load(source,allow_pickle=False)as archive:fields={name:archive[name].copy()for name in ('pressure','vx','vy','vz','tracer')}
                    expected=model.predict(fields,receipt['scene'],start,1)
                    handles=[model.pressure_core.register_forward_pre_hook(pressure_begin),model.pressure_core.register_forward_hook(pressure_end)]
                    for name,fn in original.items():setattr(model_module,name,observe(name,fn))
                    try:
                        for repeat in range(reg['repeats']):
                            elapsed.clear();counts.clear();pressure_clock.clear();tick=time.perf_counter()
                            actual=model.predict(fields,receipt['scene'],start,1);whole=time.perf_counter()-tick
                            if any(not torch.equal(a[name],b[name])for a,b in zip(actual,expected)for name in a):raise ValueError('Instrumentation changed physical outputs')
                            if any(counts.get(name)!=1 for name in ('transport','spatial0','spatial1','spatial2','forcing','pressure')):raise ValueError('Actual one-step graph call counts changed')
                            rows.append(dict(identifier=trajectory['identifier'],start=start,counts_xyz=receipt['scene']['counts'],repeat=repeat,
                                whole_ms=whole*1000,exclusive_stage_ms={name:value*1000 for name,value in elapsed.items()},
                                remaining_input_tensor_and_output_work_ms=(whole-sum(elapsed.values()))*1000))
                    finally:
                        for name,fn in original.items():setattr(model_module,name,fn)
                        for handle in handles:handle.remove()
        finally:model.close()
    if inventory(model)!=before or torch.cuda.is_initialized():raise ValueError('Profile changed parameters or initialized CUDA')
    summaries=[]
    for regime in ('small','32cubed'):
        selected=[row for row in rows if(max(row['counts_xyz'])<=16)==(regime=='small')]
        summaries.append(dict(regime=regime,observations=len(selected),whole_median_ms=statistics.median(row['whole_ms']for row in selected),
            stage_median_ms={name:statistics.median(row['exclusive_stage_ms'][name]for row in selected)for name in ('transport','spatial0','spatial1','spatial2','forcing','pressure')},
            remaining_median_ms=statistics.median(row['remaining_input_tensor_and_output_work_ms']for row in selected)))
    write(args.output,dict(status='completed_qualified_prototype_stage_diagnostic',rows=rows,summaries=summaries,source_sha256=reg['source_sha256'],
        protocol_sha256=digest(args.protocol),optimizer_updates=0,weights_unchanged=True,predictions_bitwise_unchanged=True,
        profile_overhead_is_not_benchmark_latency=True,candidate_frontier_point=False,model_accuracy_qualified=False,
        all_plan_owners_closed=True,cuda_initialized=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
