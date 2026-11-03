"""Qualify complete prediction transfer before using an extended shape ceiling."""
import argparse
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'experiments/pressure-four-pass-distillation-v1'),
             str(ROOT/'experiments/pressure-pass-sweep-v1'),str(ROOT/'experiments/pressure-methods-v1')]
from common import read,write,sha

def qualify(args):
    if args.output.exists():raise ValueError('Fresh predictor-transfer qualification required')
    os.sched_setaffinity(0,{21})
    import inference,scalable
    prior=read(args.prior);entry=prior['entry']
    if prior['status']!='completed':raise ValueError('Completed evaluated checkpoint required')
    material=read(ROOT/'runs/pressure-pass-sweep-v1/cpu-01/material.json')
    state=inference.load(dict(material,models=[entry]),entry['id'])
    inventory=state.data.helpers().state_inventory(state.network)
    wrapper,raw=scalable.qualify(state)
    recurrence=scalable.qualify_recurrence(state,wrapper,args.kind,nodes=args.nodes,passes=8)
    if state.data.helpers().state_inventory(state.network)!=inventory or state.torch.cuda.is_initialized():
        raise ValueError('Transfer qualification changed tensors or initialized CUDA')
    write(args.output,dict(status='passed',entry=entry,raw_forward_parity=raw,recurrence_parity=recurrence,
        optimizer_updates=0,unchanged_tensor_inventory=True,source_sha256=sha(__file__),
        scalable_source_sha256=sha(Path(__file__).with_name('scalable.py')),prior_evaluation_sha256=sha(args.prior)))
    print('Qualified full extended predictor',entry['id'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior',type=Path,required=True)
    parser.add_argument('--kind',choices=['fine','numerical-coarse','learned-coarse'],required=True)
    parser.add_argument('--nodes',type=int,choices=[4,6],default=6)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();qualify(args)
