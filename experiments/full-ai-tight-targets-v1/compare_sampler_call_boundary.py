"""Alternating complete-block costs with one restored model and byte-exact outputs."""
import argparse
import os
from pathlib import Path
import statistics
import time
from profile_frozen_components import ROOT,read,write,digest
from restored_recurrent_sampler import load_model
from recurrent_sampler_contiguous_v3 import library as baseline
from recurrent_sampler_noinline_v5 import library as contiguous
import recurrent_sampler_native as native
from benchmark_recurrent_sampler import output_identity
from diagnose_causal_composition import inventory,FIELDS
import numpy as np
import torch


def run(args):
    if args.output.exists():raise ValueError('Fresh paired complete-clock comparison required')
    reg=read(args.protocol)
    for path,expected in reg['source_sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError('Paired execution evidence changed: '+path)
    os.sched_setaffinity(0,{reg['cpu']});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    model=load_model(reg['pair'],reg['library_path']);before=inventory(model);rows=[];functions={'baseline_v3':baseline,'noinline_v5':contiguous}
    try:
        with torch.inference_mode():
            for case_index,trajectory in enumerate(read(ROOT/reg['bank'])['trajectories']):
                path=ROOT/trajectory['receipt'];receipt=read(path)
                if digest(path)!=trajectory['receipt_sha256']:raise ValueError('Paired development receipt changed')
                for start in reg['starts']:
                    spec=receipt['states'][start];state=path.parent/spec['path']
                    if digest(state)!=spec['sha256']:raise ValueError('Paired native input changed')
                    with np.load(state,allow_pickle=False)as archive:fields={name:archive[name].copy()for name in FIELDS}
                    for horizons in reg['horizons']:
                        native.library=baseline;expected=output_identity(model.predict(fields,receipt['scene'],start,horizons))
                        native.library=contiguous
                        if output_identity(model.predict(fields,receipt['scene'],start,horizons))!=expected:raise ValueError('Paired warm complete outputs differ')
                        times={key:[]for key in functions}
                        for repeat in range(reg['repeats']):
                            order=list(functions)
                            if(case_index+start+repeat+horizons)%2:order.reverse()
                            for variant in order:
                                native.library=functions[variant];tick=time.perf_counter()
                                outputs=model.predict(fields,receipt['scene'],start,horizons);elapsed=(time.perf_counter()-tick)*1000
                                if output_identity(outputs)!=expected:raise ValueError('Paired complete physical bytes changed')
                                times[variant].append(elapsed)
                        rows.append(dict(identifier=trajectory['identifier'],counts_xyz=receipt['scene']['counts'],start=start,horizons=horizons,
                            raw_complete_block_ms=times,all_intermediate_physical_outputs_bitwise_equal=True))
                print(trajectory['identifier'],'all paired K-block costs complete',flush=True)
        if inventory(model)!=before or torch.cuda.is_initialized():raise ValueError('Paired costs changed parameters/device')
        summaries=[]
        for regime in ('small','32cubed'):
            for horizons in reg['horizons']:
                chosen=[row for row in rows if row['horizons']==horizons and(max(row['counts_xyz'])<=16)==(regime=='small')]
                values={name:[t/horizons for row in chosen for t in row['raw_complete_block_ms'][name]]for name in functions}
                medians={name:statistics.median(v)for name,v in values.items()}
                summaries.append(dict(regime=regime,horizons=horizons,median_ms_per_state=medians,maximum_ms_per_state={name:max(v)for name,v in values.items()},
                    median_speedup=medians['baseline_v3']/medians['noinline_v5']))
        write(args.output,dict(status='completed_paired_complete_sampler_comparison',rows=rows,summaries=summaries,
            optimizer_updates=0,learned_tensors_unchanged=True,all_intermediate_outputs_bitwise_equal=True,source_sha256=reg['source_sha256'],
            protocol_sha256=digest(args.protocol),model_identity=model.asset_identity,all_plan_owners_closed=True,cuda_initialized=False,
            clock_scope='Full input checks, forcing, all neural stages and every intermediate physical output; alternating variants share the same learned tensors and FFTW owners',
            fresh_holdout_consumed=False,target_pass=False))
    finally:native.library=baseline;model.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
