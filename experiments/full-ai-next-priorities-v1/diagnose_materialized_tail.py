"""Qualify whole-graph execution against historical and consumed fresh pressure outputs."""
import argparse
import os
from pathlib import Path
import torch
from study_support import ROOT,HERE,BASE,read,write,sha,load_checkpoint,historical_cases,equation_error
from fresh_pressure_data import cases as fresh_cases
from materialized_tail_pressure import MaterializedTailPrediction


def run(args):
    if args.output.exists():
        raise ValueError('Fresh whole-graph qualification required')
    os.sched_setaffinity(0,{22})
    state=load_checkpoint(args.identifier)
    before=state.data.helpers().state_inventory(state.network)
    original=lambda sample:state.model.predict_pressure(state.network,sample,state.normalization)
    predict=MaterializedTailPrediction(state)
    cases=[(dict(identity,population='consumed_historical'),sample)for identity,sample in historical_cases(state)]
    if not args.historical_only:
        cases.extend((dict(identity,population='consumed_fresh_'+identity['population']),sample)
                     for identity,sample in fresh_cases(BASE/'pressure-fresh-data-01/bank.json',torch))
    if not args.historical_only:
        cases.extend((dict(identity,population='consumed_confirmation_'+identity['population']),sample)
                     for identity,sample in fresh_cases(BASE/'pressure-confirmation-data-01/bank.json',torch))
    rows=[]
    with torch.inference_mode():
        def check(sample):
            expected=original(sample)
            actual=predict(sample)
            relative=float(torch.linalg.vector_norm(actual-expected))/max(float(torch.linalg.vector_norm(expected)),1e-30)
            old=equation_error(state,expected,sample)
            new=equation_error(state,actual,sample)
            parity=relative<=1e-6 and abs(new-old)<=max(1e-7,old*1e-4)and (new<=7e-4)==(old<=7e-4)
            return dict(parity_pass=parity,pressure_relative_difference=relative,original_equation_error=old,compiled_equation_error=new,
                        pressure_bitwise_equal=torch.equal(actual,expected))
        for identity,sample in cases:
            rows.append(dict(identity,**check(sample)))
            if len(rows)%16==0:print("Diagnosed materialized requests",len(rows),flush=True)
        sample=cases[0][1]
        first=predict(sample)
        saved=first.clone()
        predict(cases[1][1])
        if not torch.equal(first,saved):
            raise ValueError('Whole pressure result aliases temporary storage')
        poisoned=dict(sample,rhs=sample['rhs']+123,reference=sample['reference']-99)
        if not torch.equal(saved,predict(poisoned)):
            raise ValueError('Whole pressure execution consumed labels')
        edges=[]
        for scale in(0.,-.3,10.,1e-20):
            changed=dict(sample,native_input_residual=sample['native_input_residual']*scale)
            edges.append(dict(kind='residual_scale',value=scale,**check(changed)))
        edges.append(dict(kind='different_physical_scalars',**check(dict(sample,
            cell_size_m=sample['cell_size_m']*1.5,rho=sample['rho']*2,dt=sample['dt']*2))))
        misses=predict.traces
        state.network.tail_logits.add_(0.)
        check(sample)
        if predict.traces!=misses+1:
            raise ValueError('Complete-graph weight invalidation failed')
        scale_name=state.normalization.get('output_scale','pressure')
        value=state.normalization['scales'][scale_name]['value']
        misses=predict.traces
        try:
            state.normalization['scales'][scale_name]['value']=value*1.001
            edges.append(dict(kind='different_normalization',**check(sample)))
        finally:
            state.normalization['scales'][scale_name]['value']=value
        if predict.traces!=misses+1:
            raise ValueError('Complete-graph normalization invalidation failed')
    if before!=state.data.helpers().state_inventory(state.network):
        raise ValueError('Whole-graph qualification changed weights')
    write(args.output,dict(status='diagnostic_completed',identifier=args.identifier,rows=rows,edges=edges,
        tested_requests=len(rows),optimizer_updates=0,weights_unchanged=True,materialization_sha256=sha(HERE/'materialized_fine.py'),pressure_relative_tolerance=1e-6,equation_absolute_tolerance=1e-7,equation_relative_tolerance=1e-4,owned_outputs=True,label_independent=True,
        weight_and_normalization_invalidation=True,larger_grids_use_original_eager_prediction=True,
        trace_preparation=predict.preparation,dense_cache_bytes=predict.cache_bytes,dense_cache_limit_bytes=predict.cache_limit_bytes,cache_evictions=predict.evictions,source_entry=state.entry,source_sha256=sha(__file__),
        execution_sha256=sha(HERE/'materialized_tail_pressure.py'),base_execution_sha256=sha(HERE/'whole_pressure_trace.py'),CUDA_initialized=torch.cuda.is_initialized(),
        candidate_numerical_pressure_solve=False,new_blind_evidence=False,
        scientific_scope='Execution equivalence on already consumed cases; no new fit, seed selection or reserved accuracy claim'))
    print('Diagnosed materialized learned-tail graph',args.identifier,'requests',len(rows),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--identifier',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--historical-only',action='store_true')
    run(parser.parse_args())
