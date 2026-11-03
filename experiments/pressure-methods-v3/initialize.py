"""Freeze exact continuation or deterministic sliced checkpoint initializations."""
import argparse
import hashlib
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'experiments/pressure-four-pass-distillation-v1'),str(ROOT/'experiments/pressure-pass-sweep-v1')]
from common import read,write,sha

def prepare(identifier,output):
    if output.exists(): raise ValueError('Fresh initialization required')
    material = read(ROOT/'runs/pressure-pass-sweep-v1/cpu-01/material.json')
    if identifier=='v2-headshape':
        entry = read(ROOT/'runs/pressure-methods-v2/evaluation-01/v2-headshape/results.json')['entry']
        for path,expected in entry['identities'].items():
            if sha(ROOT/path)!=expected: raise ValueError('Continuation source changed')
        artifacts=ROOT/entry['restore']; history=read(artifacts/'history.json')
        inventory=history['checkpoints']['checkpoint-2400.npz']['inventory']
        output.mkdir(parents=True);shutil.copyfile(artifacts/'weights.npz',output/'weights.npz')
        count=sum(s['bytes']//4 for s in inventory.values()); slices=[]
    else:
        import inference
        from frozen_matrix import half_width
        state=inference.load(material,identifier);original=state.data.helpers().state_inventory(state.network)
        network,slices=half_width(state.network,state.torch)
        output.mkdir(parents=True);inventory=state.data.helpers().save_weights(network,output/'weights.npz')
        if original!=state.data.helpers().state_inventory(state.network): raise ValueError('Original checkpoint changed')
        entry=state.entry;count=sum(p.numel()for p in network.parameters())
    write(output/'initialization.json',dict(status='qualified_initialization',source_entry=entry,
        weights_sha256=sha(output/'weights.npz'),inventory=inventory,parameters=count,slices=slices,
        surviving_source_tensor_bytes_preserved=True,source_sha256=sha(__file__),optimizer_updates=0))
    print('Frozen initialization',identifier,count,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',choices=['C37','C44','v2-headshape'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();prepare(args.checkpoint,args.output.resolve())
