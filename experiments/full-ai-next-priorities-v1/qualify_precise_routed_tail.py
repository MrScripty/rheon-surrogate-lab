"""Full consumed-bank qualification, retaining detailed failures and progress."""
import argparse
import os
from pathlib import Path
import math
import torch
from study_support import ROOT,HERE,BASE,read,write,sha,load_checkpoint,historical_cases,equation_error
from fresh_pressure_data import cases as fresh_cases
from precise_routed_tail_pressure import PreciseRoutedTailPrediction


def run(args):
    args.output=args.output.resolve()
    if args.output.parent.exists():raise ValueError('Fresh precise qualification directory required')
    args.output.parent.mkdir(parents=True);os.sched_setaffinity(0,{22})
    state=load_checkpoint('a-tail8');before=state.data.helpers().state_inventory(state.network)
    original=lambda sample:state.model.predict_pressure(state.network,sample,state.normalization)
    predict=PreciseRoutedTailPrediction(state)
    cases=[(dict(identity,population='consumed_historical'),sample)for identity,sample in historical_cases(state)]
    for population,path in (('consumed_fresh_',BASE/'pressure-fresh-data-01/bank.json'),
                            ('consumed_confirmation_',BASE/'pressure-confirmation-data-01/bank.json')):
        cases.extend((dict(identity,population=population+identity['population']),sample)
                     for identity,sample in fresh_cases(path,torch))
    rows=[];edges=[];active=None
    def progress(status):
        write(args.output.parent/'progress.json',dict(status=status,completed_requests=len(rows),required_requests=len(cases),active=active))
    try:
        with torch.inference_mode():
            def check(sample):
                expected=original(sample);actual=predict(sample)
                relative=float((actual-expected).norm())/max(float(expected.norm()),1e-30)
                old=equation_error(state,expected,sample);new=equation_error(state,actual,sample)
                result=dict(pressure_relative_difference=relative,original_equation_error=old,
                    compiled_equation_error=new,pressure_bitwise_equal=torch.equal(actual,expected))
                if relative>1e-6 or abs(new-old)>max(1e-7,old*1e-4)or (new<=7e-4)!=(old<=7e-4):
                    write(args.output.parent/'failure.json',dict(status='failed',active=active,result=result))
                    raise ValueError('Precise routed numerical qualification failed; detailed failure retained')
                return result
            for identity,sample in cases:
                active=identity
                rows.append(dict(identity,**check(sample)))
                if len(rows)%128==0:
                    progress('running');write(args.output.parent/'rows.partial.json',rows)
                    print('Qualified precise routed requests',len(rows),flush=True)
            for route in ('dense','CNN'):
                sample=next(sample for _,sample in cases if max(sample['base'].shape[2:])<=16 and
                    (math.prod(sample['base'].shape[2:])<=2048)==(route=='dense'))
                active=dict(edge_route=route)
                first=predict(sample);saved=first.clone();predict(cases[1][1])
                if not torch.equal(first,saved):raise ValueError('Output aliases temporary storage')
                if not torch.equal(saved,predict(dict(sample,rhs=sample['rhs']+123,reference=sample['reference']-99))):
                    raise ValueError('Reference labels consumed at inference')
                for scale in (0.,-.3,10.,1e-20):
                    edges.append(dict(route=route,kind='residual_scale',scale=scale,**check(dict(sample,
                        native_input_residual=sample['native_input_residual']*scale))))
                edges.append(dict(route=route,kind='physical_scalars',**check(dict(sample,
                    cell_size_m=sample['cell_size_m']*1.5,rho=sample['rho']*2,dt=sample['dt']*2))))
                misses=predict.traces;state.network.tail_logits.add_(0.);check(sample)
                if predict.traces!=misses+1:raise ValueError('Weight invalidation failed')
                key=state.normalization.get('output_scale','pressure');value=state.normalization['scales'][key]['value']
                misses=predict.traces
                try:
                    state.normalization['scales'][key]['value']=value*1.001
                    edges.append(dict(route=route,kind='normalization',**check(sample)))
                finally:state.normalization['scales'][key]['value']=value
                if predict.traces!=misses+1:raise ValueError('Normalization invalidation failed')
        if before!=state.data.helpers().state_inventory(state.network):raise ValueError('Qualification changed weights')
        write(args.output,dict(status='passed',identifier='a-tail8',rows=rows,edges=edges,tested_requests=len(rows),
            optimizer_updates=0,weights_unchanged=True,owned_outputs=True,label_independent=True,
            weight_and_normalization_invalidation=True,larger_grids_use_original_eager_prediction=True,
            trace_preparation=predict.preparation,dense_cache_bytes=predict.cache_bytes,
            dense_cache_limit_bytes=predict.cache_limit_bytes,cache_evictions=predict.evictions,
            source_entry=state.entry,source_sha256=sha(__file__),
            execution_sha256=sha(HERE/'precise_routed_tail_pressure.py'),
            materialization_sha256=sha(HERE/'materialized_fine_precise.py'),
            fallback_sha256=sha(HERE/'tail_whole_pressure.py'),
            pressure_relative_tolerance=1e-6,equation_absolute_tolerance=1e-7,equation_relative_tolerance=1e-4,
            candidate_numerical_pressure_solve=False,new_independent_accuracy_claim=False,
            CUDA_initialized=torch.cuda.is_initialized()))
        active=None;progress('completed')
        print('Qualified precise routed learned pressure',len(rows),'requests',flush=True)
    except BaseException as error:
        write(args.output.parent/'terminal.json',dict(status='failed',error=repr(error),active=active,completed_requests=len(rows)))
        progress('failed');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
