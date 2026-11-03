"""Check the actual restored complete four-step graph against its tensor reference."""
import argparse
import os
from pathlib import Path
from profile_frozen_components import ROOT,read,write,digest
from restored_recurrent_sampler import load_model
from qualify_recurrent_sampler import reference_predict,relative
from diagnose_causal_composition import inventory,FIELDS
import numpy as np
import torch


def run(args):
    if args.output.exists():raise ValueError('Fresh trained graph qualification required')
    reg=read(args.protocol)
    for path,expected in reg['source_sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError('Restored trained graph/source changed: '+path)
    os.sched_setaffinity(0,{reg['cpu']});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    model=load_model(reg['pair'],reg['library_path']);before=inventory(model);rows=[]
    try:
        with torch.inference_mode():
            for trajectory in read(ROOT/reg['bank'])['trajectories']:
                path=ROOT/trajectory['receipt'];receipt=read(path)
                if digest(path)!=trajectory['receipt_sha256']:raise ValueError('Qualified current-state source changed')
                for start in reg['starts']:
                    spec=receipt['states'][start];state=path.parent/spec['path']
                    if digest(state)!=spec['sha256']:raise ValueError('Qualified current-state archive changed')
                    with np.load(state,allow_pickle=False)as archive:fields={key:archive[key].copy()for key in FIELDS}
                    actual=model.predict(fields,receipt['scene'],start,4);expected=reference_predict(model,fields,receipt['scene'],start,4)
                    errors={name:max(relative(a[name],b[name])for a,b in zip(actual,expected))for name in FIELDS}
                    if max(errors.values())>reg['maximum_execution_relative_error']:raise ValueError('Actual trained complete graph differs: '+str(errors))
                    rows.append(dict(identifier=trajectory['identifier'],start=start,counts_xyz=receipt['scene']['counts'],relative_l2=errors,complete_states=4,
                        carrier_bytes_exact=all(torch.equal(a[k],b[k])for a,b in zip(actual,expected)for k in ('vx','vy','vz','tracer'))))
                print(trajectory['identifier'],'restored four-step neural graph qualified',flush=True)
        if inventory(model)!=before or torch.cuda.is_initialized():raise ValueError('Qualification changed model/device')
        write(args.output,dict(status='passed_trained_complete_graph_execution',rows=rows,optimizer_updates=0,complete_states=4*len(rows),
            model_identity=model.asset_identity,source_sha256=reg['source_sha256'],protocol_sha256=digest(args.protocol),
            learned_tensors_unchanged=True,cuda_initialized=False,all_plan_owners_closed=True,
            accuracy_qualified=False,latency_claim=False))
    finally:model.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
