"""Screen a cheap learned tail and separately labeled offline analytical ceilings."""
import argparse
import math
import os
from pathlib import Path
import numpy as np
from study_support import ROOT,HERE,BASE,read,write,sha,load_checkpoint,equation_error
from fresh_pressure_data import cases
from learned_pressure_tail import append,apply_gains


def run(args):
    if args.output.exists():raise ValueError('Fresh tail-policy screen required')
    os.sched_setaffinity(0,{23})
    state=load_checkpoint('a-head-calibration-s2-routed')
    torch=state.torch
    before=state.data.helpers().state_inventory(state.network)
    rows=[]
    with torch.inference_mode():
        for identity,sample in cases(BASE/'pressure-confirmation-data-01/bank.json',torch):
            if max(identity['counts_xyz'])>16:continue
            ordinary=state.model.predict_pressure(state.network,sample,state.normalization)
            predictions={'ordinary':ordinary,'learned_tail4':append(state,sample,ordinary)}
            counts=tuple(ordinary.shape[2:])
            for modes in(4,8):
                axes=[];eigenvalues=None
                for dimension,n in enumerate(counts):
                    mode=torch.arange(min(modes,n),dtype=torch.float64)
                    axis=torch.cos(math.pi/n*mode[:,None]*(torch.arange(n,dtype=torch.float64)[None,:]+.5))*math.sqrt(2/n)
                    axis[0]/=math.sqrt(2)
                    axes.append(axis)
                    shape=[1,1,1];shape[dimension]=len(mode)
                    value=(2*(1-torch.cos(math.pi*mode/n))).reshape(shape)
                    eigenvalues=value if eigenvalues is None else eigenvalues+value
                gains=torch.zeros_like(eigenvalues)
                nonzero=eigenvalues>0
                gains[nonzero]=eigenvalues[nonzero].reciprocal()
                predictions['offline_analytical_tail'+str(modes)]=apply_gains(state,sample,ordinary,tuple(axes),gains)
            reference=sample['reference']-sample['reference'].mean()
            for variant,pressure in predictions.items():
                if not bool(torch.isfinite(pressure).all())or float(pressure[0,0,0,0,0])!=0:
                    raise ValueError('Tail output finite/gauge contract failed')
                rows.append(dict(identity,variant=variant,equation_error=equation_error(state,pressure,sample),
                    pressure_error=float((pressure-pressure.mean()-reference).norm())/max(float(reference.norm()),1e-30)))
    if before!=state.data.helpers().state_inventory(state.network):raise ValueError('Tail screen changed weights')
    summaries=[]
    for variant in('ordinary','learned_tail4','offline_analytical_tail4','offline_analytical_tail8'):
        group=[row for row in rows if row['variant']==variant]
        summaries.append(dict(variant=variant,requests=len(group),worst_equation_error=max(row['equation_error']for row in group),
            worst_pressure_error=max(row['pressure_error']for row in group),equation_7e_4_passes=sum(row['equation_error']<=.0007 for row in group),
            pressure_and_equation_4e_3_passes=sum(max(row['equation_error'],row['pressure_error'])<=.004 for row in group),
            full_AI_candidate=variant in('ordinary','learned_tail4')))
    write(args.output,dict(status='completed',summaries=summaries,rows=rows,source_sha256=sha(__file__),
        tail_source_sha256=sha(HERE/'learned_pressure_tail.py'),model_entry=state.entry,optimizer_updates=0,
        learned_candidate='Two fine CNN calls and three learned global blocks; no numerical inverse, no extra fine call',
        analytical_controls='Inverse eigenvalues used offline only; excluded from candidate frontiers',
        population='Consumed confirmation bank, development screen only; needs freezing and new independent confirmation',
        measured_latency=False,model_promotion=False))
    print('Screened learned-only tail and separate offline analytical ceilings',summaries,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
