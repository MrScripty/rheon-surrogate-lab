"""Encode exact native requests; byte-verify deterministic spectral label recreation."""
import argparse
import importlib.util
import hashlib
import os
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'experiments/pressure-four-pass-distillation-v1'),str(ROOT/'experiments/pressure-methods-v1')]
from common import read,write,sha

def pack(source,output):
    if output.exists():raise ValueError('Fresh exact derived view required')
    os.sched_setaffinity(0,{19});torch.set_num_threads(1)
    spectral=ROOT/'experiments/pressure-methods-v1/numerics.py'
    spec=importlib.util.spec_from_file_location('reference_spectral',spectral);module=importlib.util.module_from_spec(spec);sys.modules['reference_spectral']=module;spec.loader.exec_module(module)
    import native_reconstruct
    manifest=read(source/'manifest.json');values={}
    for shard in manifest['shards']:
        path=source/shard['path']
        if sha(path)!=shard['sha256']:raise ValueError('Full native view changed')
        with np.load(path,allow_pickle=False)as archive:values.update({k:archive[k].copy()for k in archive.files})
    derived={}
    for index,parent in enumerate(manifest['parents']):
        if parent['source']=='native-generated-spectral-reference-cold-warm':
            for field in ['rhs','reference']:
                name=f'p{index:04d}_{field}';derived[name]='exact-native-spectral-v1';del values[name]
    trial=native_reconstruct.complete(dict(values),dict(manifest,derived_fields=derived))
    if set(trial)!=set(manifest['arrays']):raise ValueError('Reconstruction omitted fields')
    for key,value in trial.items():
        if hashlib.sha256(value.tobytes()).hexdigest()!=manifest['arrays'][key]['sha256']:raise ValueError('Decoded full population changed')
    del trial
    output.mkdir(parents=True);groups=[];sizes=[]
    for key in sorted(values,key=lambda k:(-values[k].nbytes,k)):
        size=values[key].nbytes+512;options=[i for i,n in enumerate(sizes)if n+size<=39*1024**2]
        if options:index=min(options,key=lambda i:39*1024**2-sizes[i]-size)
        else:index=len(groups);groups.append({});sizes.append(0)
        groups[index][key]=values[key];sizes[index]+=size
    shards=[]
    for number,selected in enumerate(groups):
        path=output/f'dataset-{number:02d}.npz';np.savez_compressed(path,**selected)
        shards.append(dict(path=path.name,bytes=path.stat().st_size,sha256=sha(path),keys=sorted(selected)))
    total=sum(s['bytes']for s in shards)
    if total>204*1024**2 or max(s['bytes']for s in shards)>40*1024**2:raise ValueError('Ordinary managed aggregate budget plus restore reserve exceeded')
    write(output/'manifest.json',dict(manifest,shards=shards,derived_fields=derived,encoded_total_bytes=total,
        encoded_source_manifest_sha256=sha(source/'manifest.json'),source_sha256=sha(__file__),
        exact_full_decoded_population_verified=True,reference_solver_source_sha256=sha(spectral),
        reconstruction_source_sha256=sha(Path(native_reconstruct.__file__))))
    print('Exact 34-parent view:',round(total/1024**2,3),'MiB encoded; all decoded fields retain their original hashes',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();pack(args.source,args.output)
