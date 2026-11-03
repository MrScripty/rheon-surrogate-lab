"""Preserve every retained tensor and add a zero-output shape conditioner."""
import hashlib
import math
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,write,sha

def prepare():
    source=ROOT/'runs/pressure-methods-v1/learned-coarse-initialization-01'
    output=ROOT/'runs/pressure-methods-v2/shape-head-initialization-01'
    if output.exists():raise ValueError('Initialization already exists')
    prior=read(source/'initialization.json')
    if sha(source/'weights.npz')!=prior['weights_sha256']:raise ValueError('Retained initialization changed')
    with np.load(source/'weights.npz',allow_pickle=False) as archive:
        values={k:archive[k].copy() for k in archive.files}
    original={k:v.tobytes() for k,v in values.items()}
    rng=np.random.Generator(np.random.PCG64(2026101820))
    bound=1/math.sqrt(3)
    values.update({'shape_head.0.weight':rng.uniform(-bound,bound,(16,3)).astype('<f4'),
        'shape_head.0.bias':rng.uniform(-bound,bound,(16,)).astype('<f4'),
        'shape_head.2.weight':np.zeros((63,16),dtype='<f4'),
        'shape_head.2.bias':np.zeros((63,),dtype='<f4')})
    if any(values[k].tobytes()!=v for k,v in original.items()):raise ValueError('Surviving weights changed')
    inventory={k:dict(dtype=v.dtype.str,shape=list(v.shape),bytes=v.nbytes,
        sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in values.items()}
    output.mkdir();np.savez(output/'weights.npz',**values)
    write(output/'initialization.json',dict(status='qualified_initialization',
        kind='shape-informed-coarse',source_weights_sha256=sha(source/'weights.npz'),
        weights_sha256=sha(output/'weights.npz'),inventory=inventory,
        surviving_c53_tensor_bytes_preserved=True,initial_shape_branch_output_zero=True,
        source_sha256=sha(__file__),optimizer_updates=0))
    print('Preserved all C53/coarse tensors; added zero-output shape branch',flush=True)

if __name__=='__main__':prepare()
