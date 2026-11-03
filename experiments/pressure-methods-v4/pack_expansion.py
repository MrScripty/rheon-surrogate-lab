"""Losslessly retain all 34 prior parents and add one separately reserved data track."""
import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
from v4_support import ROOT,BASE,DOCS,read,write,sha

def pack(group,output):
    if output.exists():raise ValueError('Fresh expanded view required')
    os.sched_setaffinity(0,{21})
    import numpy as np,torch
    torch.set_num_threads(1)
    spectral=ROOT/'experiments/pressure-methods-v1/numerics.py'
    spec=importlib.util.spec_from_file_location('reference_spectral',spectral);module=importlib.util.module_from_spec(spec);sys.modules['reference_spectral']=module;spec.loader.exec_module(module)
    import reconstruct
    original=ROOT/'runs/pressure-methods-v3/derived-training-view-01';manifest=read(original/'manifest.json');values={}
    for shard in manifest['shards']:
        path=original/shard['path']
        if sha(path)!=shard['sha256']:raise ValueError('Original encoded bank changed')
        with np.load(path,allow_pickle=False)as archive:values.update({k:archive[k].copy()for k in archive.files})
    parents=list(manifest['parents']);inventory=dict(manifest['arrays']);derived=dict(manifest['derived_fields']);evidence=dict(manifest['source_evidence'])
    for row in read(DOCS/'scene-reservation.json')['rows']:
        if row['population']!=group:continue
        policy='spectral'if group=='diverse-train'else'learned';directory=BASE/'native-data'/(row['trajectory_id']+'-'+policy)
        receipt=read(directory/'receipt.json');scene=row['scene']
        if receipt['row']!=row or receipt['scene']!=scene or not receipt['native_process']['reaped']:
            raise ValueError('Exact terminal training-only capture required')
        if receipt['status']!='completed'or [f['index']for f in receipt['snapshots']]!=row['snapshots']:
            raise ValueError('The predeclared complete training capture is unavailable; inspect before changing the reservation')
        index=len(parents);parents.append(dict(trajectory_id=row['trajectory_id'],split='train',counts_xyz=scene['counts'],cell_size_m=1/scene['cells_per_metre'],density_kg_m3=1000.,dt_s=1/60,
            frames=[dict(index=f['index'],time_s=f['time_s'])for f in receipt['snapshots']],
            source='native-generated-diverse-spectral'if group=='diverse-train'else'native-generated-onpolicy-spectral-label',
            originating_policy=policy,source_receipt=str((directory/'receipt.json').relative_to(ROOT))))
        for field in ['base','rhs','reference','native_input_residual']:
            frames=[]
            for frame in receipt['snapshots']:
                desc=frame['arrays'][field];path=directory/desc['path']
                if sha(path)!=desc['sha256']:raise ValueError('Captured field changed')
                frames.append(np.fromfile(path,dtype='<f8').reshape(scene['counts'][::-1]))
            key=f'p{index:04d}_{field}';array=np.stack(frames)
            inventory[key]=dict(shape=list(array.shape),dtype=array.dtype.str,bytes=array.nbytes,sha256=hashlib.sha256(array.tobytes()).hexdigest())
            if field in ['rhs','reference']:derived[key]='exact-native-spectral-v1'
            else:values[key]=array
        evidence[str((directory/'receipt.json').relative_to(ROOT))]=sha(directory/'receipt.json')
    expanded=dict(manifest,schema='rheon-pressure-methods-learning-view-v4',parents=parents,arrays=inventory,derived_fields=derived)
    # Every reconstructed label, including the entire original bank, is compared
    # with the retained full-resolution capture hash before this view is usable.
    trial=reconstruct.complete(dict(values),expanded)
    if set(trial)!=set(inventory):raise ValueError('Expanded decoded inventory incomplete')
    for key,value in trial.items():
        if hashlib.sha256(value.tobytes()).hexdigest()!=inventory[key]['sha256']:raise ValueError('Expanded full-field bytes differ')
    del trial
    if sum(s['bytes']for s in inventory.values())>768*1024**2:raise ValueError('Decoded field budget exceeded')
    output.mkdir(parents=True);groups=[];sizes=[]
    for key in sorted(values,key=lambda k:(-values[k].nbytes,k)):
        size=values[key].nbytes+512;options=[i for i,n in enumerate(sizes)if n+size<=39*1024**2]
        if options:i=min(options,key=lambda j:39*1024**2-sizes[j]-size)
        else:i=len(groups);groups.append({});sizes.append(0)
        groups[i][key]=values[key];sizes[i]+=size
    if not 2<=len(groups)<=18:raise ValueError('Bounded shard count exceeded')
    shards=[]
    for i,selected in enumerate(groups):
        path=output/f'dataset-{i:02d}.npz';np.savez_compressed(path,**selected)
        shards.append(dict(path=path.name,bytes=path.stat().st_size,sha256=sha(path),keys=sorted(selected)))
    total=sum(r['bytes']for r in shards)
    if total>236*1024**2 or max(r['bytes']for r in shards)>40*1024**2:raise ValueError('Ordinary input budget with explicit restore reserve exceeded')
    write(output/'manifest.json',dict(expanded,shards=shards,source_evidence=evidence,source_sha256=sha(__file__),
        retained_v3_parents=34,added_v4_parents=len(parents)-34,expansion_group=group,encoded_total_bytes=total,
        reconstruction_source_sha256=sha(Path(reconstruct.__file__)),reference_solver_source_sha256=sha(spectral),
        prior_manifest_sha256=sha(original/'manifest.json'),exact_full_decoded_population_verified=True,original_field_bytes_preserved=True))
    print('Verified lossless expanded view',group,len(parents),round(total/1024**2,2),'MiB',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--group',choices=['diverse-train','onpolicy-train'],required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();pack(args.group,args.output)
