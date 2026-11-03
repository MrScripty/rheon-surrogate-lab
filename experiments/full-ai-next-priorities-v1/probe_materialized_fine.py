"""Bounded frozen-network linearity, execution accuracy and latency pilot."""
import argparse
import os
from pathlib import Path
import statistics
import time
import torch
from study_support import HERE,write,sha,load_checkpoint
from materialized_fine import materialize,MaterializedFine,require_linear_spatial_network


def run(args):
    if args.output.exists():raise ValueError('Fresh materialization pilot required')
    os.sched_setaffinity(0,{23})
    state=load_checkpoint('a-tail8');network=state.network
    before=state.data.helpers().state_inventory(network)
    modules=require_linear_spatial_network(network);rows=[]
    generator=torch.Generator().manual_seed(2026101991)
    with torch.inference_mode():
        for counts in ((5,5,5),(9,11,11),(16,16,16)):
            matrix,preparation=materialize(network,counts)
            predict=MaterializedFine(network,matrix)
            examples=[];worst=0.;linearity=0.
            for scale in (1e-6,1.,1e6):
                x=torch.randn((1,1,*counts),generator=generator)*scale
                y=torch.randn((1,1,*counts),generator=generator)*scale
                expected=network(x)
                relative=float((predict(x)-expected).norm())/max(float(expected.norm()),1e-30)
                additive=float((network(x+y)-expected-network(y)).norm())/max(float(expected.norm()),1e-30)
                worst=max(worst,relative);linearity=max(linearity,additive)
                examples.append(dict(scale=scale,relative_difference=relative,relative_additivity_error=additive))
            values=torch.randn((1,1,*counts),generator=generator)
            timings={}
            for name,method in (('original',network),('matrix',predict)):
                for _ in range(10):method(values)
                elapsed=[]
                for _ in range(50):
                    tick=time.perf_counter();method(values);elapsed.append(time.perf_counter()-tick)
                timings[name]=1000*statistics.median(elapsed)
            rows.append(dict(counts_zyx=list(counts),preparation=preparation,examples=examples,
                worst_output_relative_difference=worst,worst_linearity_relative_error=linearity,
                median_raw_fine_ms=timings))
            print('Materialized fine pilot',counts,'difference',worst,'raw CNN/matrix ms',timings,flush=True)
    if before!=state.data.helpers().state_inventory(network):raise ValueError('Pilot changed checkpoint weights')
    write(args.output,dict(status='completed',rows=rows,spatial_modules=modules,source_entry=state.entry,
        source_sha256=sha(__file__),execution_sha256=sha(HERE/'materialized_fine.py'),optimizer_updates=0,
        candidate_numerical_pressure_solve=False,new_independent_accuracy_claim=False,
        scope='Raw learned CNN pilot only; full pressure output and total CPU cost not yet qualified'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
