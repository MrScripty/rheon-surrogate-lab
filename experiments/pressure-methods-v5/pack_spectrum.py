"""Keep every original training field and add reserved spatial-band equations."""
import hashlib
import math
import os
import sys
from v5_support import ROOT,HERE,BASE,DOCS,read,write,sha
from spectrum import pattern
def pack():
    import numpy as np,torch,numerics
    sys.modules['reference_spectral']=numerics
    import training_reconstruct as reconstruct
    os.sched_setaffinity(0,{21});torch.set_num_threads(1)
    output=BASE/'spectrum-training-view-01'
    if output.exists():raise ValueError('Fresh view required')
    old=ROOT/'runs/pressure-methods-v3/derived-training-view-01';manifest=read(old/'manifest.json');values={}
    for shard in manifest['shards']:
        p=old/shard['path']
        if sha(p)!=shard['sha256']:raise ValueError('Original bank changed')
        with np.load(p,allow_pickle=False)as archive:values.update({k:archive[k].copy()for k in archive.files})
    parents=list(manifest['parents']);inventory=dict(manifest['arrays']);derived=dict(manifest['derived_fields']);observed=[]
    for row in read(DOCS/'scene-reservation.json')['spectrum']:
        if row['population']!='train':continue
        index=len(parents);shape=row['counts_xyz'][::-1];h=torch.tensor([1/row['cells_per_metre']],dtype=torch.float64);rho=torch.tensor([1000.],dtype=torch.float64)
        arrays={k:[]for k in ['base','rhs','reference','native_input_residual']};mode_rows=[]
        for i in range(row['patterns']):
            exact,detail=pattern(row,i);residual=numerics.fine_action(exact,h,rho);reference=numerics.correction(residual,h,rho,None);reference-=reference[:,:,:1,:1,:1].clone()
            error=reference-exact;error-=error.mean()
            if float(torch.linalg.vector_norm(error))/float(torch.linalg.vector_norm(exact-exact.mean()))>1e-10:raise ValueError('Spectral and analytic labels disagree')
            for k,v in dict(base=torch.zeros_like(exact),rhs=residual,reference=reference,native_input_residual=residual).items():arrays[k].append(v[0,0].numpy())
            mode_rows.append(dict(pattern=i,modes=detail))
        parents.append(dict(trajectory_id=row['trajectory_id'],split='train',counts_xyz=row['counts_xyz'],cell_size_m=float(h[0]),density_kg_m3=1000.,dt_s=1/60,
            frames=[dict(index=i,time_s=(i+1)/60)for i in range(row['patterns'])],source='operator-generated-spatial-band',
            interpretation='Independent pressure patterns; frame slots do not represent an evolved fluid trajectory.',mode_band=row['mode_band']))
        for field,frames in arrays.items():
            key=f'p{index:04d}_{field}';array=np.stack(frames)
            inventory[key]=dict(shape=list(array.shape),dtype=array.dtype.str,bytes=array.nbytes,sha256=hashlib.sha256(array.tobytes()).hexdigest())
            if field in ['rhs','reference']:derived[key]='exact-native-spectral-v1'
            else:values[key]=array
        observed.append(dict(row=row,patterns=mode_rows))
    expanded=dict(manifest,parents=parents,arrays=inventory,derived_fields=derived)
    trial=reconstruct.complete(dict(values),expanded)
    for k,v in trial.items():
        if hashlib.sha256(v.tobytes()).hexdigest()!=inventory[k]['sha256']:raise ValueError('Reconstructed full field changed')
    del trial
    if sum(s['bytes']for s in inventory.values())>768*1024**2:raise ValueError('Decoded input bound exceeded')
    output.mkdir();groups=[];sizes=[]
    for key in sorted(values,key=lambda k:(-values[k].nbytes,k)):
        size=values[key].nbytes+512;available=[i for i,n in enumerate(sizes)if n+size<=39*1024**2]
        if available:i=min(available,key=lambda j:39*1024**2-sizes[j]-size)
        else:i=len(groups);groups.append({});sizes.append(0)
        groups[i][key]=values[key];sizes[i]+=size
    shards=[]
    for i,group in enumerate(groups):
        path=output/f'dataset-{i:02d}.npz';np.savez_compressed(path,**group)
        shards.append(dict(path=path.name,bytes=path.stat().st_size,sha256=sha(path),keys=sorted(group)))
    total=sum(s['bytes']for s in shards)
    if total>236*1024**2:raise ValueError('Encoded input bound with restore reserve exceeded')
    write(output/'manifest.json',dict(expanded,shards=shards,encoded_total_bytes=total,retained_v3_parents=34,added_spectral_parents=len(parents)-34,
        exact_full_decoded_population_verified=True,original_field_bytes_preserved=True,prior_manifest_sha256=sha(old/'manifest.json'),
        reservation_sha256=sha(DOCS/'scene-reservation.json'),source_sha256=sha(__file__),pattern_source_sha256=sha(HERE/'spectrum.py')))
    write(output/'spatial-patterns.json',dict(rows=observed,source_sha256=sha(__file__),pattern_source_sha256=sha(HERE/'spectrum.py')))
    print('Verified native retention plus explicit spatial bands:',len(parents),'parents,',round(total/1024**2,2),'MiB',flush=True)
if __name__=='__main__':pack()
