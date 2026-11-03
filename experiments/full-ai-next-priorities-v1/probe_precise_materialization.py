"""Isolate matrix construction/storage roundoff on the failed pressure examples."""
import argparse
import os
from pathlib import Path
import torch
from study_support import HERE,write,sha,load_checkpoint,historical_cases,equation_error
from materialized_fine_precise import materialize,PreciseFine
from materialized_tail_pressure import MaterializedTailGraph


def run(args):
    if args.output.exists():raise ValueError('Fresh precision pilot required')
    os.sched_setaffinity(0,{23})
    state=load_checkpoint('a-tail8');cases=historical_cases(state)
    before=state.data.helpers().state_inventory(state.network);rows=[]
    with torch.inference_mode():
        for index in (42,83,0):
            identity,sample=cases[index];counts=tuple(sample['base'].shape[2:])
            expected=state.model.predict_pressure(state.network,sample,state.normalization)
            old=equation_error(state,expected,sample)
            matrix,preparation=materialize(state.network,counts)
            for dtype in (torch.float32,torch.float64):
                graph=MaterializedTailGraph(state,counts,matrix.to(dtype))
                graph.network=PreciseFine(state.network,matrix.to(dtype))
                actual=graph(*(sample[name]for name in ('base','native_input_residual','cell_size_m','rho','dt')))
                relative=float((actual-expected).norm())/max(float(expected.norm()),1e-30)
                new=equation_error(state,actual,sample)
                result=dict(identity,matrix_dtype=str(dtype),preparation=preparation,
                    original_equation_error=old,new_equation_error=new,pressure_relative_difference=relative,
                    equation_absolute_difference=abs(new-old),
                    passes_registered_tolerance=relative<=1e-6 and abs(new-old)<=max(1e-7,old*1e-4)and (new<=7e-4)==(old<=7e-4))
                rows.append(result);print('Precise materialization',index,str(dtype),'pressure difference',relative,'equation delta',abs(new-old),'pass',result['passes_registered_tolerance'],flush=True)
    if before!=state.data.helpers().state_inventory(state.network):raise ValueError('Precision pilot mutated weights')
    write(args.output,dict(status='completed',rows=rows,source_entry=state.entry,optimizer_updates=0,
        source_sha256=sha(__file__),materialization_sha256=sha(HERE/'materialized_fine_precise.py'),
        candidate_numerical_pressure_solve=False,classification='Consumed-case precision diagnosis, not full qualification or new accuracy claim'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
