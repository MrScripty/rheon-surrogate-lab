"""Exercise the actual temporal objective through two predicted blocks without fitting."""
import argparse
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from study_support import ROOT,HERE,BASE,read,write,sha
from temporal_data import load,make_features,feature_at
from temporal_objective import loss,errors
from bundle_model_v5 import BundleNet

def run(args):
    if args.output.exists():raise ValueError('Fresh temporal objective qualification required')
    os.sched_setaffinity(0,{23});torch.set_num_threads(1)
    view=BASE/'temporal-training-view-02';manifest=read(view/'manifest.json')
    _,norm,parents=load(view/'manifest.json',view/'normalization.json',[view/r['path']for r in manifest['shards']],args.device)
    rows=[]
    for horizons in (1,2,3,4):
        torch.manual_seed(2026101921);network=BundleNet(horizons).to(args.device)
        before={name:p.detach().cpu().numpy().tobytes()for name,p in network.named_parameters()}
        for counts in ([16,16,16],[32,32,32]):
            parent=next(p for p in parents if p['scene']['counts']==counts)
            features=make_features(parent,norm['scales'],horizons,args.device)
            state=parent['states'][8:9]
            tick=time.perf_counter();first=network(state,feature_at(features,8,horizons))
            second=network(first[:,-1],feature_at(features,8+horizons,horizons))
            target1=parent['states'][9:9+horizons].unsqueeze(0)
            target2=parent['states'][9+horizons:9+2*horizons].unsqueeze(0)
            objective=(loss(first,target1,counts,norm['scales']['tracer'])+loss(second,target2,counts,norm['scales']['tracer']))/2
            network.zero_grad(set_to_none=True);objective.backward()
            if not bool(torch.isfinite(objective))or any(p.grad is None or not bool(torch.isfinite(p.grad).all())for p in network.parameters()):
                raise ValueError('Actual two-block objective or gradients nonfinite')
            if any(p.detach().cpu().numpy().tobytes()!=before[name]for name,p in network.named_parameters()):
                raise ValueError('Zero-update objective qualification changed weights')
            with torch.no_grad():
                equal=loss(target1,target1,counts,norm['scales']['tracer'])
                if float(equal)>1e-12:raise ValueError('Identical physically valid states must have zero objective')
            rows.append(dict(horizons=horizons,counts_xyz=counts,two_blocks=True,second_block_uses_predicted_history=True,
                loss=float(objective.detach()),finite_complete_backward=True,identical_state_loss=float(equal),
                optimizer_updates=0,seconds=time.perf_counter()-tick))
    write(args.output,dict(status='passed',rows=rows,device=args.device,optimizer_updates=0,
        model_sha256=sha(HERE/'bundle_model_v5.py'),objective_sha256=sha(HERE/'temporal_objective.py'),
        fitting_view_sha256=sha(view/'manifest.json'),normalization_sha256=sha(view/'normalization.json'),
        source_sha256=sha(__file__),cuda_initialized=torch.cuda.is_initialized()))
    print('Qualified real two-block temporal losses and all finite gradients for K=1,2,3,4 at16³/32³',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    run(parser.parse_args())
