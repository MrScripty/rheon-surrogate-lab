"""Full restored trained K-block cost after complete neural graph qualification."""
import argparse
import hashlib
import math
import os
from pathlib import Path
import statistics
import time
from profile_frozen_components import ROOT,read,write,digest
from restored_recurrent_sampler import load_model
from diagnose_causal_composition import inventory
import numpy as np
import torch


def output_identity(outputs):
    return [{name:dict(dtype=value.numpy().dtype.str,shape=list(value.shape),
                       sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest())for name,value in output.items()}for output in outputs]


def percentile(values,fraction):
    ordered=sorted(values);return ordered[min(len(ordered)-1,math.ceil(len(ordered)*fraction)-1)]


def run(args):
    if args.output.exists():raise ValueError('Fresh complete prototype cost gate required')
    reg=read(args.protocol)
    for path,expected in reg['source_sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError('Registered complete prototype source changed: '+path)
    if args.cpu not in reg['cpus']:raise ValueError('Registered CPU affinity required')
    qualification=read(ROOT/reg['qualification'])
    if qualification['status']!='passed_trained_complete_graph_execution'or qualification['optimizer_updates']!=0 or len(qualification['rows'])!=16:
        raise ValueError('Actual qualified untrained graph required before timing')
    os.sched_setaffinity(0,{args.cpu});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    cases=[];sources={}
    for row in read(ROOT/reg['bank'])['trajectories']:
        path=ROOT/row['receipt']
        if digest(path)!=row['receipt_sha256']:raise ValueError('Consumed current-state receipt changed')
        receipt=read(path);sources[str(path.relative_to(ROOT))]=digest(path)
        for start in reg['starts']:
            spec=receipt['states'][start];state=path.parent/spec['path']
            if digest(state)!=spec['sha256']:raise ValueError('Current physical input changed')
            sources[str(state.relative_to(ROOT))]=digest(state)
            with np.load(state,allow_pickle=False)as archive:fields={name:archive[name].copy()for name in ('pressure','vx','vy','vz','tracer')}
            cases.append(dict(identifier=row['identifier'],scene=receipt['scene'],start=start,fields=fields))
    rows=[];models=[]
    with torch.inference_mode():
        for initializer in reg['initializers']:
            tick=time.perf_counter();model=load_model(reg['pair'],reg['library_path']);setup=time.perf_counter()-tick
            before=inventory(model);models.append(dict(initializer=initializer,construction_seconds=setup,inventory=before,assets=model.asset_identity))
            try:
                for case_index,case in enumerate(cases):
                    order=reg['horizons'][case_index%4:]+reg['horizons'][:case_index%4]
                    for horizons in order:
                        def predict():return model.predict(case['fields'],case['scene'],case['start'],horizons)
                        tick=time.perf_counter();expected=predict();first_call=time.perf_counter()-tick
                        times=[]
                        for repeat in range(reg['repeats']):
                            tick=time.perf_counter();outputs=predict();elapsed=(time.perf_counter()-tick)*1000
                            if len(outputs)!=horizons or any(not torch.equal(a[name],b[name])for a,b in zip(outputs,expected)for name in a):
                                raise ValueError('Complete deterministic neural output changed while timing')
                            times.append(elapsed)
                        finite=all(bool(torch.isfinite(value).all())for output in outputs for value in output.values())
                        if not finite:raise ValueError('Nonfinite actual complete prototype state')
                        rows.append(dict(initializer=initializer,identifier=case['identifier'],counts_xyz=case['scene']['counts'],
                            cells_per_metre=case['scene']['cells_per_metre'],start=case['start'],horizons=horizons,
                            warmup_first_call_seconds=first_call,raw_complete_block_ms=times,
                            median_block_ms=statistics.median(times),p95_block_ms=percentile(times,.95),maximum_block_ms=max(times),
                            maximum_ms_per_generated_state=max(times)/horizons,complete_output_identity=output_identity(outputs),
                            output_shape_dtype_and_finiteness_verified=True))
                    print('CPU',args.cpu,initializer,case['identifier'],case['start'],'all K cost observations complete',flush=True)
                if inventory(model)!=before:raise ValueError('Prototype timing changed learned tensors')
            finally:model.close()
    summaries=[]
    for initializer in reg['initializers']:
        for regime in ('small','32cubed'):
            for horizons in reg['horizons']:
                selected=[row for row in rows if row['initializer']==initializer and row['horizons']==horizons and(max(row['counts_xyz'])<=16)==(regime=='small')]
                times=[t for row in selected for t in row['raw_complete_block_ms']]
                summaries.append(dict(initializer=initializer,regime=regime,horizons=horizons,complete_blocks=len(times),
                    median_block_ms=statistics.median(times),median_ms_per_generated_state=statistics.median(times)/horizons,
                    p95_block_ms=percentile(times,.95),maximum_block_ms=max(times),maximum_ms_per_generated_state=max(times)/horizons,
                    prototype_cost_under_budget=max(times)<=reg['maximum_ms_per_generated_step']*horizons))
    if len(rows)!=reg['expected_configurations_per_cpu']or torch.cuda.is_initialized():raise ValueError('Finite cost population/device changed')
    write(args.output,dict(status='completed_trained_complete_block_cost_gate',cpu=args.cpu,threads=1,batch_size=1,
        rows=rows,summaries=summaries,models=models,source_evidence=sources,source_sha256=reg['source_sha256'],
        protocol_sha256=digest(args.protocol),optimizer_updates=0,new_parameters_fitted=True,all_plan_owners_closed=True,cuda_initialized=False,
        clock_scope='Current full input validation/encoding, all prescribed forcing, every neural stage and learned pressure, all intermediate physical outputs and internal history updates',
        setup_scope='Construction and first warm calls separately recorded; learned weights and geometry/FFTW caches reused within warm costs',
        fresh_holdout_consumed=False,model_accuracy_qualified=False,full_generator_target_pass=False,candidate_frontier_point=False,
        prototype_costs_are_not_trained_model_results=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--cpu',type=int,choices=(6,23),required=True)
    run(parser.parse_args())
