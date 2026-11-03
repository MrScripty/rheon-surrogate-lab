"""Untrained full-output CPU feasibility only; not a candidate frontier measurement."""
import argparse
import os
from pathlib import Path
import statistics
import time
import numpy as np
import torch
from study_support import ROOT,HERE,BASE,read,write,sha
from bundle_model_global import BundleNet


def run(args):
    if args.output.exists():
        raise ValueError('Fresh untrained feasibility profile required')
    os.sched_setaffinity(0,{23})
    torch.set_num_threads(1)
    norm=read(BASE/'temporal-training-view-02/normalization.json')
    bank=read(BASE/'temporal-data-01/bank.json')
    cases=[]
    for counts in([16,16,16],[9,11,11],[15,6,8],[32,32,32]):
        row=next(row for row in bank['trajectories']if row['split']=='train'and row['counts_xyz']==counts)
        path=ROOT/row['receipt']
        if sha(path)!=row['receipt_sha256']:
            raise ValueError('Training receipt changed')
        receipt=read(path)
        for step in(0,8,63,120):
            record=receipt['states'][step]
            source=path.parent/record['path']
            if sha(source)!=record['sha256']:
                raise ValueError('Training current state changed')
            with np.load(source,allow_pickle=False)as arrays:
                fields={name:arrays[name].copy()for name in ('pressure','vx','vy','vz','tracer')}
            cases.append((dict(counts_xyz=counts,step=step,scene=row['identifier']),fields,receipt['scene']))
    torch.manual_seed(2026101921)
    networks={k:BundleNet(k).eval()for k in(1,2,3,4)}
    rows=[]
    with torch.inference_mode():
        for k,network in networks.items():
            for _,fields,scene in cases:
                network.predict(fields,scene,8,norm)
        for repeat in range(5):
            order=list(networks)
            order=order[repeat%4:]+order[:repeat%4]
            if repeat%2:order.reverse()
            for k in order:
                network=networks[k]
                for identity,fields,scene in cases:
                    tick=time.perf_counter()
                    outputs=network.predict(fields,scene,identity['step'],norm)
                    elapsed=time.perf_counter()-tick
                    if len(outputs)!=k or not all(bool(torch.isfinite(value).all())for state in outputs for value in state.values()):
                        raise ValueError('Full joint physical output contract failed')
                    rows.append(dict(identity,horizons=k,repeat=repeat,full_block_ms=elapsed*1000,
                        per_step_ms=elapsed*1000/k))
    summaries=[]
    for k in networks:
        for counts in([16,16,16],[9,11,11],[15,6,8],[32,32,32]):
            group=[r for r in rows if r['horizons']==k and r['counts_xyz']==counts]
            summaries.append(dict(horizons=k,counts_xyz=counts,parameters=sum(p.numel()for p in networks[k].parameters()),
                median_full_block_ms=statistics.median(r['full_block_ms']for r in group),
                median_per_step_ms=statistics.median(r['per_step_ms']for r in group)))
    write(args.output,dict(status='completed',classification='Untrained CPU feasibility; no accuracy or frontier claim',
        summaries=summaries,rows=rows,cpu=23,threads=1,batch_size=1,
        scope='Full causal encoding, known forcing, shared network and physical reconstruction of every field/horizon',
        source_sha256=sha(__file__),model_sha256=sha(HERE/'bundle_model_global.py'),global_operator_sha256=sha(HERE/'learned_global.py'),
        CUDA_initialized=torch.cuda.is_initialized(),optimizer_updates=0))
    print('Profiled untrained shared-global CPU full-block feasibility for allK/allgrids',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
