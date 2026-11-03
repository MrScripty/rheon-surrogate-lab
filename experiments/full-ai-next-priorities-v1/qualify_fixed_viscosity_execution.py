"""Require exact physical outputs for fast execution of each new managed model."""
import argparse
import os
from pathlib import Path
import numpy as np
import torch
import temporal_displacement_objective as teaching
from fixed_viscosity_evaluation_support import load_model
from fast_preparation_model_support import wrap
from temporal_train_fixed_viscosity import inventory
from evaluate_coupled import references
from temporal_metrics import FIELDS
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run(args):
    if args.output.exists():raise ValueError('Fresh exact execution qualification required')
    protocol_path=DOCS/'fixed-viscosity-execution-protocol-01.json';reg=read(protocol_path)
    for name,digest in reg['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Registered execution source changed')
    for relative,digest in reg['execution_assets_sha256'].items():
        if sha(ROOT/relative)!=digest:raise ValueError('Qualified execution asset changed')
    if sha(DOCS/'fixed-viscosity-fit-registration-01.json')!=reg['fit_registration_sha256']:
        raise ValueError('Registered fitting process changed')
    os.sched_setaffinity(0,{22});torch.set_num_threads(1)
    k=args.horizons;identifier=f'b{k}-fixed-viscosity';pair=BASE/(identifier+'-pair-01')
    original,norm,identity=load_model(pair);project=ROOT/read(pair/'receipt.json')['source_project']
    candidate,_,variant=wrap(original,norm,identity,project/'expert-assets');before=inventory(original)
    evaluation_path=BASE/'fixed-viscosity-evaluation-01'/identifier/'summary.json';evaluation=read(evaluation_path)
    if evaluation['model']['weights_sha256']!=identity['weights_sha256']or evaluation['status']!='completed':
        raise ValueError('Exact complete managed-model accuracy evaluation required')
    bank_path=BASE/'temporal-development-240-01/bank.json'
    if sha(bank_path)!=reg['bank_sha256']:raise ValueError('Registered scene bank changed')
    def forbidden(*args,**kwargs):raise ValueError('Training labels reached inference')
    teaching.displacement_labels=forbidden
    rows=[];outputs=0
    def compare(a,b):
        nonlocal outputs
        if len(a)!=k or len(b)!=k:raise ValueError('Every physical horizon required')
        for expected,actual in zip(a,b):
            if set(expected)!=set(FIELDS)or set(actual)!=set(FIELDS):raise ValueError('All physical fields required')
            for name in FIELDS:
                if not np.array_equal(expected[name].numpy().view('u1'),actual[name].numpy().view('u1')):
                    raise ValueError('Fast execution changed physical bytes: '+name)
            outputs+=1
    with torch.inference_mode():
        for row in read(bank_path)['trajectories']:
            if row['split']!='development':raise ValueError('Consumed development scenes only')
            scene,states=references(row)
            for start in range(121):
                current={n:states[start][n]for n in FIELDS}
                compare(original.predict(current,scene,start,norm),candidate.predict(current,scene,start,norm))
            left={n:states[0][n]for n in FIELDS};right=left
            for start in range(0,240,k):
                a=original.predict(left,scene,start,norm);b=candidate.predict(right,scene,start,norm);compare(a,b)
                left=a[-1];right=b[-1]
            rows.append(dict(identifier=row['identifier'],single_block_outputs=121*k,own_history_outputs=240,all_physical_bytes_exact=True))
            print('Qualified exact fast execution:',identifier,row['identifier'],flush=True)
    if inventory(original)!=before or inventory(candidate)!=before or torch.cuda.is_initialized():
        raise ValueError('Qualification changed weights or initialized CUDA')
    if outputs!=(121*k+240)*8 or len(rows)!=8:raise ValueError('Complete original evaluation scope required')
    write(args.output,dict(status='passed',identifier=identifier,horizons=k,model=variant,rows=rows,outputs=outputs,
        all_physical_bytes_exact=True,all_accuracy_classifications_preserved=True,optimizer_updates=0,
        source_sha256=reg['source_sha256'],protocol_sha256=sha(protocol_path),CPU_latency_claim=False,
        training_teacher_disabled=True,source_evidence={str(p.relative_to(ROOT)):sha(p)for p in
            (pair/'receipt.json',pair/'restore/weights.npz',evaluation_path,bank_path,project/'assembly.json')},
        reserved_scenes_consumed=False))
    print('Qualified exact fast execution:',identifier,outputs,'physical outputs',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--horizons',type=int,choices=(1,2,3,4),required=True)
    parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
