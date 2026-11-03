"""Offline oracle ceiling for using the learned pressure component in temporal models."""
import argparse
import os
from pathlib import Path
import statistics
import numpy as np
import torch
from study_support import ROOT,HERE,BASE,read,write,sha,load_checkpoint,equation_error
from tail_whole_pressure import TailWholePrediction


def run(args):
    if args.output.exists():
        raise ValueError('Fresh component ceiling diagnostic required')
    os.sched_setaffinity(0,{23})
    state=load_checkpoint('a-tail8')
    predict=TailWholePrediction(state)
    bank=read(BASE/'temporal-data-01/bank.json')
    rows=[]
    with torch.inference_mode():
        for row in bank['trajectories']:
            if row['split']not in('train','development'):
                raise ValueError('No reserved scene in component diagnosis')
            path=ROOT/row['receipt']
            if sha(path)!=row['receipt_sha256']:
                raise ValueError('Consumed scene receipt changed')
            receipt=read(path)
            scene=receipt['scene']
            shape=(1,1,*scene['counts'][::-1])
            for start in(0,8,63,120):
                for horizon in(1,2,3,4):
                    step=start+horizon
                    record=receipt['states'][step]
                    source=path.parent/record['path']
                    if sha(source)!=record['sha256']:
                        raise ValueError('Consumed reference changed')
                    with np.load(source,allow_pickle=False)as arrays:
                        rhs=torch.from_numpy(arrays['rhs'].copy().reshape(shape))
                        reference=torch.from_numpy(arrays['pressure'].copy().reshape(shape))
                    sample=dict(base=torch.zeros_like(rhs),native_input_residual=rhs,rhs=rhs,reference=reference,
                        cell_size_m=torch.tensor([1/scene['cells_per_metre']],dtype=torch.float64),
                        rho=torch.tensor([1000.],dtype=torch.float64),dt=torch.tensor([1/60],dtype=torch.float64))
                    predicted=predict(sample)
                    reference=reference-reference.mean()
                    difference=predicted-predicted.mean()-reference
                    rows.append(dict(identifier=row['identifier'],split=row['split'],counts_xyz=scene['counts'],start=start,
                        horizon=horizon,step=step,pressure_relative_l2=float(difference.norm())/max(float(reference.norm()),1e-30),
                        equation_relative_residual=equation_error(state,predicted,sample)))
    summaries=[]
    for split in('train','development'):
        for regime in('small','32cubed'):
            selected=[row for row in rows if row['split']==split and(max(row['counts_xyz'])<=16)==(regime=='small')]
            summaries.append(dict(split=split,regime=regime,queries=len(selected),
                median_pressure_error=statistics.median(row['pressure_relative_l2']for row in selected),
                worst_pressure_error=max(row['pressure_relative_l2']for row in selected),
                worst_equation_error=max(row['equation_relative_residual']for row in selected),
                joint_pressure_equation_4e_3_pass=all(max(row['pressure_relative_l2'],row['equation_relative_residual'])<=.004 for row in selected)))
    write(args.output,dict(status='completed',rows=rows,summaries=summaries,model_entry=state.entry,
        source_sha256=sha(__file__),execution_sha256=sha(HERE/'tail_whole_pressure.py'),optimizer_updates=0,
        classification='Offline oracle with true future RHS; explicitly not a temporal candidate or frontier point',
        reason='Test the pressure component ceiling before training a causal neural pressure-request predictor',
        candidate_future_RHS_input_permitted=False,previous_pressure_input=False,reserved_scenes_consumed=False))
    print('Measured offline neural-pressure ceiling on384future-reference queries; not a temporal predictor',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
