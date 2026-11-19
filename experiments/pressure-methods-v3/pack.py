"""Lossless bounded training view, retaining all original shape-training fields."""
import argparse
import hashlib
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,write,sha

def pack(output):
    if output.exists(): raise ValueError('Fresh view required')
    original = ROOT/'runs/pressure-methods-v2/shape-training-view-01'
    manifest = read(original/'manifest.json'); parents = list(manifest['parents']); values = {}
    for shard in manifest['shards']:
        path = original/shard['path']
        if sha(path) != shard['sha256']: raise ValueError('Retained view changed')
        with np.load(path,allow_pickle=False) as archive:
            values.update({k:archive[k].copy() for k in archive.files})
    retained = {k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in values.items()}
    reservation = ROOT/'docs/plans/pressure-methods-v3/reservation.json'
    evidence = {str((original/'manifest.json').relative_to(ROOT)):sha(original/'manifest.json'),str(reservation.relative_to(ROOT)):sha(reservation)}
    for row in read(reservation)['rows']:
        if row['population'] != 'train': continue
        directory = ROOT/'runs/pressure-methods-v3/native-data'/row['trajectory_id']
        receipt = read(directory/'receipt.json'); scene = row['scene']
        if (receipt['status']!='completed' or receipt['scene']!=scene or not receipt['native_process']['reaped']
                or [f['index'] for f in receipt['snapshots']]!=row['snapshots']):
            raise ValueError('Complete qualified native training trajectory required')
        index = len(parents)
        parents.append(dict(trajectory_id=row['trajectory_id'],split='train',counts_xyz=scene['counts'],
            cell_size_m=1/scene['cells_per_metre'],density_kg_m3=1000.,dt_s=1/60,
            frames=[dict(index=f['index'],time_s=f['time_s']) for f in receipt['snapshots']],
            source='native-generated-spectral-reference-cold-warm'))
        for field in ['base','rhs','reference','native_input_residual']:
            frames = []
            for frame in receipt['snapshots']:
                descriptor = frame['arrays'][field]; path = directory/descriptor['path']
                if sha(path)!=descriptor['sha256']: raise ValueError('Native training field changed')
                frames.append(np.fromfile(path,dtype='<f8').reshape(scene['counts'][::-1]))
            values[f'p{index:04d}_{field}'] = np.stack(frames)
        evidence[str((directory/'receipt.json').relative_to(ROOT))] = sha(directory/'receipt.json')
    if len(parents)!=34 or any(hashlib.sha256(values[k].tobytes()).hexdigest()!=v for k,v in retained.items()):
        raise ValueError('All 26 prior parents and eight new parents required')
    if sum(v.nbytes for v in values.values())>768*1024**2: raise ValueError('768 MiB decoded budget exceeded')
    output.mkdir(parents=True); shards=[]; groups=[]; sizes=[]
    # Keep every ordinary managed input below its 40 MiB contract. Large arrays
    # remain intact; smaller arrays fill the remaining space deterministically.
    for key in sorted(values,key=lambda k:(-values[k].nbytes,k)):
        count=values[key].nbytes+512
        options=[i for i,n in enumerate(sizes) if n+count<=39*1024**2]
        if options: index=min(options,key=lambda i:39*1024**2-sizes[i]-count)
        else: index=len(groups);groups.append({});sizes.append(0)
        groups[index][key]=values[key];sizes[index]+=count
    if not 2<=len(groups)<=18: raise ValueError('Declared managed shard/argument budget exceeded')
    for number,selected in enumerate(groups):
        path = output/f'dataset-{number:02d}.npz'; np.savez(path,**selected)
        if path.stat().st_size>40*1024**2: raise ValueError('40 MiB managed input budget exceeded')
        shards.append(dict(path=path.name,bytes=path.stat().st_size,sha256=sha(path),keys=sorted(selected)))
    inventory = {k:dict(shape=list(v.shape),dtype=v.dtype.str,bytes=v.nbytes,sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in values.items()}
    write(output/'manifest.json',dict(manifest,schema='rheon-pressure-methods-learning-view-v3',max_cells=128**3,
        parents=parents,arrays=inventory,shards=shards,source_evidence=evidence,source_sha256=sha(__file__),
        retained_small_parents=16,retained_prior_parents=26,added_larger_parents=8,original_field_bytes_preserved=True))
    print('Packed 34 parents with original fields preserved',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    pack(parser.parse_args().output.resolve())
