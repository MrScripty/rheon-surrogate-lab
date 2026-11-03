"""Evaluate the actual managed tail prediction path, never the parent-only helper."""
import argparse
import os
from pathlib import Path
import statistics
from study_support import ROOT,HERE,BASE,read,write,sha,load_checkpoint,historical_cases,equation_error
from fresh_pressure_data import cases as fresh_cases


def run(args):
    args.output=args.output.resolve()
    if args.output.exists():raise ValueError('Fresh actual-tail evaluation required')
    args.output.mkdir(parents=True);os.sched_setaffinity(0,{23})
    state=load_checkpoint(args.identifier);torch=state.torch
    before=state.data.helpers().state_inventory(state.network)
    sources=[];summaries=[]
    with torch.inference_mode():
        for population,bank in(('historical',None),('initial_fresh',BASE/'pressure-fresh-data-01/bank.json'),
                                ('confirmation',BASE/'pressure-confirmation-data-01/bank.json')):
            cases=historical_cases(state)if bank is None else fresh_cases(bank,torch)
            rows=[]
            for identity,sample in cases:
                actual=state.model.predict_pressure(state.network,sample,state.normalization)
                parent=state.model.parent_model.predict_pressure(state.network,sample,state.normalization)
                if not bool(torch.isfinite(actual).all())or float(actual[0,0,0,0,0])!=0:
                    raise ValueError('Actual tail output contract failed')
                if max(identity['counts_xyz'])>16 and not torch.equal(actual,parent):
                    raise ValueError('Larger-grid behavior must retain parent bitwise')
                centered=sample['reference']-sample['reference'].mean()
                rows.append(dict(identity,equation_error=equation_error(state,actual,sample),
                    parent_equation_error=equation_error(state,parent,sample),
                    pressure_relative_l2=float((actual-actual.mean()-centered).norm())/max(float(centered.norm()),1e-30),
                    larger_grid_parent_bitwise_equal=torch.equal(actual,parent)if max(identity['counts_xyz'])>16 else None))
            groups=[]
            for regime in('small','larger'):
                chosen=[row for row in rows if(max(row['counts_xyz'])<=16)==(regime=='small')]
                if not chosen:continue
                errors=sorted(row['equation_error']for row in chosen)
                groups.append(dict(regime=regime,requests=len(chosen),median_error=statistics.median(errors),
                    p95_error=errors[min(len(errors)-1,int(.95*len(errors)))],worst_error=max(errors),
                    worst_pressure_error=max(row['pressure_relative_l2']for row in chosen),
                    passing_7e_4=sum(value<=.0007 for value in errors),
                    pressure_and_equation_4e_3_passes=sum(max(row['pressure_relative_l2'],row['equation_error'])<=.004 for row in chosen)))
            path=args.output/(population+'.json')
            write(path,dict(status='completed',identifier=args.identifier,population=population,rows=rows,summaries=groups,
                source_entry=state.entry,actual_prediction='state.model.predict_pressure including final learned8modeblock',
                source_sha256=sha(__file__),bank_sha256=sha(bank)if bank else None,optimizer_updates=0,
                classification='Consumed development/regression populations; no new independent claim'))
            sources.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
            summaries.append(dict(population=population,groups=groups))
            print('Actual tail evaluation',args.identifier,population,groups,flush=True)
    if before!=state.data.helpers().state_inventory(state.network):raise ValueError('Evaluation changed trained weights')
    write(args.output/'summary.json',dict(status='completed',identifier=args.identifier,populations=summaries,sources=sources,
        source_entry=state.entry,source_sha256=sha(__file__),optimizer_updates=0,cuda_initialized=torch.cuda.is_initialized(),
        actual_tail_prediction_path=True,model_promotion=False,new_independent_accuracy_claim=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--identifier',required=True);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
