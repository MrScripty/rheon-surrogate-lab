"""Require exact physical fields for every block and own240 execution output."""
import argparse
import os
from pathlib import Path
import numpy as np
import torch
from viscosity_composition_support import load_model
from fast_preparation_model_support import wrap
from temporal_train_displacement import inventory
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run(args):
    if args.output.exists():raise ValueError('Fresh full-model execution qualification required')
    protocol_path=DOCS/'fast-preparation-model-protocol-01.json';reg=read(protocol_path)
    for name,digest in reg['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Registered full-model execution source changed')
    component_path=BASE/'fused-force-construction-qualification-01.json';component=read(component_path)
    if component['status']!='passed'or not component['all_float32_bytes_exact']or sha(component_path)!=reg['force_qualification_sha256']:
        raise ValueError('Exact component execution qualification required')
    os.sched_setaffinity(0,{22});torch.set_num_threads(1)
    directory=BASE/'b2-displacement-viscosity-assembly-01';original,norm,identity=load_model(directory)
    assembly=read(directory/'receipt.json');parent=ROOT/assembly['parent_pair'];project=ROOT/read(parent/'receipt.json')['source_project']
    candidate,_,variant_identity=wrap(original,norm,identity,project/'expert-assets');before=inventory(original)
    bank_path=BASE/'temporal-development-240-01/bank.json'
    if sha(bank_path)!=reg['bank_sha256']:raise ValueError('Registered development bank changed')
    rows=[];sources={};total=0;byte_exact=True
    def compare(a,b):
        nonlocal total,byte_exact
        if len(a)!=len(b)or len(a)!=2:raise ValueError('Every physical horizon required')
        for expected,actual in zip(a,b):
            if set(expected)!=set(actual):raise ValueError('Full physical fields required')
            for name in expected:
                if not torch.equal(expected[name],actual[name]):raise ValueError('Fused execution changed a physical output: '+name)
                byte_exact &= np.array_equal(expected[name].numpy().view('u1'),actual[name].numpy().view('u1'))
            total+=1
    with torch.inference_mode():
        for row in read(bank_path)['trajectories']:
            if row['split']!='development':raise ValueError('Only registered development scenes required')
            path=ROOT/row['receipt']
            if sha(path)!=row['receipt_sha256']:raise ValueError('Reference scene changed')
            receipt=read(path);scene=receipt['scene'];sources[str(path.relative_to(ROOT))]=sha(path);states=[]
            for spec in receipt['states']:
                source=path.parent/spec['path']
                if sha(source)!=spec['sha256']:raise ValueError('Current native state changed')
                sources[str(source.relative_to(ROOT))]=sha(source)
                with np.load(source,allow_pickle=False)as a:states.append({n:a[n].copy()for n in ('pressure','vx','vy','vz','tracer')})
            for start in range(121):compare(original.predict(states[start],scene,start,norm),candidate.predict(states[start],scene,start,norm))
            left=states[0];right=states[0]
            for start in range(0,240,2):
                a=original.predict(left,scene,start,norm);b=candidate.predict(right,scene,start,norm);compare(a,b)
                left=a[-1];right=b[-1]
            rows.append(dict(identifier=row['identifier'],single_block_outputs=242,own_history_outputs=240,float32_and_float64_physical_values_exact=True))
            print('Qualified fast preparation model every horizon/own240:',row['identifier'],flush=True)
    if inventory(original)!=before or inventory(candidate)!=before or torch.cuda.is_initialized():raise ValueError('Execution qualification changed weights or initialized CUDA')
    if total!=3856 or len(rows)!=8:raise ValueError('Complete original evaluation scope required')
    if not byte_exact:raise ValueError('Byte-exact physical outputs required')
    write(args.output,dict(status='passed',model=variant_identity,rows=rows,outputs=total,all_physical_values_exact=True,
        all_physical_bytes_exact=byte_exact,all_existing_accuracy_classifications_preserved=True,
        source_sha256=reg['source_sha256'],source_evidence=sources,protocol_sha256=sha(protocol_path),
        optimizer_updates=0,CPU_latency_claim=False,reserved_scenes_consumed=False))
    print('Qualified fast preparation model:',total,'exact physical outputs; no weights changed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
