"""Extend the immutable mixed-domain view with qualified training rectangles."""
import hashlib
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,write,sha

def pack(output):
    if output.exists():raise ValueError('Fresh data view required')
    original=ROOT/'runs/pressure-methods-v1/domain-training-view-01'
    manifest=read(original/'manifest.json');parents=list(manifest['parents']);values={}
    for shard in manifest['shards']:
        path=original/shard['path']
        if sha(path)!=shard['sha256']:raise ValueError('Retained view changed')
        with np.load(path,allow_pickle=False) as arrays:
            values.update({k:arrays[k].copy() for k in arrays.files})
    retained={k:v.tobytes() for k,v in values.items()}
    reservation=read(ROOT/'docs/plans/pressure-methods-v2/reservation.json')
    evidence={str((original/'manifest.json').relative_to(ROOT)):sha(original/'manifest.json')}
    for row in reservation['scenes']:
        if row['population']!='train':continue
        directory=ROOT/'runs/pressure-methods-v2/native-data/capture'/row['trajectory_id']
        receipt=read(directory/'receipt.json')
        if receipt['status']!='qualified_pilot' or not receipt['native_process']['reaped']:
            raise ValueError('Qualified, reaped native training capture required')
        index=len(parents)
        parents.append(dict(trajectory_id=row['trajectory_id'],split='train',counts_xyz=row['counts_xyz'],
            cell_size_m=1/row['cells_per_metre'],density_kg_m3=1000.,dt_s=1/60,
            frames=[dict(index=f['index'],time_s=f['time_s']) for f in receipt['frames']],
            source='v2-native-rectangular-training'))
        for field in ['base','rhs','reference','native_input_residual']:
            captured='pressure' if field=='reference' else field
            frames=[]
            for frame in receipt['frames']:
                descriptor=frame['arrays'][captured];path=directory/descriptor['path']
                if sha(path)!=descriptor['sha256']:raise ValueError('Native field changed')
                frames.append(np.fromfile(path,dtype='<f8').reshape(row['counts_xyz'][::-1]))
            values[f'p{index:04d}_{field}']=np.stack(frames)
        evidence[str((directory/'receipt.json').relative_to(ROOT))]=sha(directory/'receipt.json')
    if len(parents)!=26 or any(values[k].tobytes()!=v for k,v in retained.items()):
        raise ValueError('Exact original fields and all six rectangular training parents required')
    output.mkdir(parents=True)
    shards=[]
    for number,fields in enumerate([('base','rhs'),('reference','native_input_residual')]):
        selected={k:v for k,v in values.items() if any(k.endswith('_'+f) for f in fields)}
        path=output/f'dataset-{number:02d}.npz';np.savez(path,**selected)
        if path.stat().st_size>32*1024**2:raise ValueError('Declared shard budget exceeded')
        shards.append(dict(path=path.name,bytes=path.stat().st_size,sha256=sha(path),keys=sorted(selected)))
    inventory={k:dict(shape=list(v.shape),dtype=v.dtype.str,bytes=v.nbytes,
        sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in values.items()}
    write(output/'manifest.json',dict(manifest,parents=parents,arrays=inventory,shards=shards,
        source_evidence=evidence,source_sha256=sha(__file__),retained_small_parents=16,
        added_larger_parents=10,original_field_bytes_preserved=True))
    print('Packed 26 parents; original small/cube fields preserved exactly',flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    pack(parser.parse_args().output.resolve())
