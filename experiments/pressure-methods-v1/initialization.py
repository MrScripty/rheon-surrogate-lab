"""Create declared new-parameter inventories while preserving surviving C53 bytes."""
import argparse
import hashlib
import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read, write, sha


def prepare(kind,output):
    if output.exists() or kind not in ['learned-coarse','spacing-two-channel']:
        raise ValueError('Fresh declared initialization required')
    material = read(ROOT/'runs/pressure-pass-sweep-v1/cpu-01/material.json')
    selected = next(row for row in material['models'] if row['id']=='C53')
    source = ROOT/selected['restore']/'weights.npz'
    if sha(source) != selected['weights_sha256']:
        raise ValueError('C53 changed')
    with np.load(source,allow_pickle=False) as archive:
        original = {k:archive[k].copy() for k in archive.files}
    values = {k:v.copy() for k,v in original.items()}
    if kind == 'learned-coarse':
        values['coarse_logits'] = np.full((63,),math.log(1/127),dtype='<f4')
        preserved = all(np.array_equal(values[k],v) and values[k].tobytes()==v.tobytes() for k,v in original.items())
    else:
        key = 'encoders.0.layers.0.weight'
        old = values[key]
        if old.shape[1] != 1:
            raise ValueError('Expected one-channel C53 initialization')
        values[key] = np.concatenate((old,np.zeros_like(old)),axis=1)
        preserved = values[key][:,:1].tobytes()==old.tobytes() and all(values[k].tobytes()==v.tobytes() for k,v in original.items() if k!=key)
    if not preserved:
        raise ValueError('Surviving C53 tensors changed')
    inventory = {k:dict(dtype=v.dtype.str,shape=list(v.shape),bytes=v.nbytes,sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in values.items()}
    output.mkdir()
    with (output/'weights.npz').open('xb') as stream:
        np.savez(stream,**values)
    write(output/'initialization.json',dict(status='qualified_initialization',kind=kind,
        source_weights_sha256=sha(source),weights_sha256=sha(output/'weights.npz'),inventory=inventory,
        surviving_c53_tensor_bytes_preserved=True,source_sha256=sha(__file__),optimizer_updates=0))
    print('Qualified initialization',kind,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',required=True)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    prepare(args.kind,args.output.resolve())
