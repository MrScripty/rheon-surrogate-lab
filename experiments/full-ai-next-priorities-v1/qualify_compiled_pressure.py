"""Check every historical output, equation residual, owned output and trace invalidation."""
import argparse
import os
from pathlib import Path
from study_support import ROOT,HERE,read,write,sha,load_checkpoint,historical_cases,equation_error
from compiled_pressure import CompiledPrediction,fine_name

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--identifier',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise ValueError('Fresh compiled inference qualification required')
    os.sched_setaffinity(0,{23});state=load_checkpoint(args.identifier);torch=state.torch
    before=state.data.helpers().state_inventory(state.network);cases=historical_cases(state);predict=CompiledPrediction(state);rows=[]
    with torch.inference_mode():
        for identity,sample in cases:
            predict(sample);predict(sample)
            reference=state.model.predict_pressure(state.network,sample,state.normalization);actual=predict(sample)
            relative=float(torch.linalg.vector_norm(actual-reference))/max(float(torch.linalg.vector_norm(reference)),1e-30)
            old=equation_error(state,reference,sample);new=equation_error(state,actual,sample)
            if relative>1e-6 or float(actual[:,:,:1,:1,:1].abs().max())!=0 or abs(new-old)>max(1e-9,old*1e-5):
                raise ValueError('Compiled pressure, gauge or equation parity failed')
            rows.append(dict(identity,pressure_relative_difference=relative,original_equation_error=old,compiled_equation_error=new))
        sample=cases[0][1];first=predict(sample);saved=first.clone();predict(cases[1][1])
        if not torch.equal(first,saved):raise ValueError('Compiled prediction aliases temporary output storage')
        changed=dict(sample,rhs=sample['rhs']+123,reference=sample['reference']-99)
        if not torch.equal(saved,predict(changed)):raise ValueError('Compiled prediction used labels')
        misses=predict.fine.traces
        next(p for name,p in state.network.named_parameters()if fine_name(name)).add_(0.)
        predict(sample)
        if predict.fine.traces!=misses+1:raise ValueError('Fine weight-version trace invalidation failed')
    if state.data.helpers().state_inventory(state.network)!=before:raise ValueError('Compilation qualification changed weights')
    write(args.output,dict(status='passed',identifier=args.identifier,rows=rows,pressure_relative_tolerance=1e-6,
        equation_absolute_tolerance=1e-9,equation_relative_tolerance=1e-5,unchanged_inventory=True,
        owned_outputs=True,label_independent=True,fine_weight_version_invalidation=True,
        trace_preparation=predict.fine.preparation,source_sha256=sha(__file__),execution_sha256=sha(HERE/'compiled_pressure.py'),
        source_entry=state.entry,optimizer_updates=0,cuda_initialized=torch.cuda.is_initialized(),
        setup_and_cache_misses_must_be_reported_separately=True))
    print('Qualified compiled fine inference on all128requests',args.identifier,flush=True)
