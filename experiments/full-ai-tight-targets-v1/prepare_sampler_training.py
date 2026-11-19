"""Offline train-only native transport labels; no local parameter learning."""
import argparse
import os
from pathlib import Path
from profile_frozen_components import ROOT,read,write,digest
from recurrent_sampler_assets import source_tensors,csr
from recurrent_sampler_native import dimensions
from training_displacement_teacher import scalar_sample,point_coordinates
import numpy as np
import torch


def run(args):
    if args.output.exists():raise ValueError('Fresh train-only view required')
    reg=read(args.protocol)
    for path,expected in reg['source_sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError('Registered teacher/data source changed: '+path)
    os.sched_setaffinity(0,{reg['cpu']});torch.set_num_threads(1)
    selected=[r for r in read(ROOT/reg['bank'])['trajectories']if r['split']=='train']
    if len(selected)!=16 or any(not r['identifier'].endswith(('-r0','-r1'))for r in selected):raise ValueError('Original 16 training scenes required')
    tensors,provenance=source_tensors();rows,channels,offsets,weights=csr(tensors['point'])
    rng=np.random.default_rng(reg['seed']);args.output.mkdir(parents=True);sources={};groups={};scene_count=0
    with torch.inference_mode():
        for trajectory in selected:
            path=ROOT/trajectory['receipt']
            if digest(path)!=trajectory['receipt_sha256']:raise ValueError('Train receipt changed')
            receipt=read(path);scene=receipt['scene'];counts=tuple(scene['counts']);alpha=scene['cells_per_metre']/60.
            sources[str(path.relative_to(ROOT))]=digest(path);key='g'+'_'.join(map(str,reversed(counts)))
            group=groups.setdefault(key,dict(arrays={k:[]for k in ('point','patch','kind','alpha','middle','end')},samples=[],counts=counts))
            coordinates=point_coordinates(counts,'cpu')
            for state_index in reg['states']:
                spec=receipt['states'][state_index];state=path.parent/spec['path']
                if digest(state)!=spec['sha256']:raise ValueError('Native train current state changed')
                sources[str(state.relative_to(ROOT))]=digest(state)
                with np.load(state,allow_pickle=False)as archive:
                    fields=[torch.from_numpy(archive[name].copy()).reshape(shape)for name,shape in zip(('vx','vy','vz'),dimensions(counts))]
                def sample(position):
                    return torch.stack([scalar_sample(fields[c],position,[0. if a==c else .5 for a in range(3)])for c in range(3)])
                for kind,shape in enumerate(dimensions(counts)):
                    size=int(np.prod(shape));take=min(size,reg['points_per_carrier']);indices=rng.choice(size,take,replace=False)
                    # Include actual faces/corners, keeping the total finite.
                    zz,yy,xx=np.meshgrid([0,shape[0]-1],[0,shape[1]-1],[0,shape[2]-1],indexing='ij')
                    corners=(xx+shape[2]*(yy+shape[1]*zz)).reshape(-1)
                    indices=np.unique(np.concatenate((indices,corners)))
                    integer=np.stack(np.unravel_index(indices,shape)[::-1],1);position=coordinates[kind][:,indices]
                    first=sample(position);middle=torch.minimum((position-.5*alpha*first).clamp_min(0),torch.tensor(counts,dtype=torch.float64)[:,None])
                    end=-sample(middle).T.contiguous();middle_label=((middle-position)/alpha).T.contiguous()
                    features=torch.zeros((len(indices),3),dtype=torch.float64);patch=[]
                    for component,field in enumerate(fields):
                        nz,ny,nx=field.shape;values=[]
                        for z in range(-1,2):
                            for y in range(-1,2):
                                for x in range(-1,2):
                                    values.append(field[(integer[:,2]+z).clip(0,nz-1),(integer[:,1]+y).clip(0,ny-1),(integer[:,0]+x).clip(0,nx-1)])
                        patch.append(torch.stack(values,1).reshape(-1,3,3,3))
                    for axis in range(3):
                        head=kind*3+axis
                        for i in range(int(rows[head]),int(rows[head+1])):
                            field=fields[int(channels[i])];nz,ny,nx=field.shape;dx,dy,dz=map(int,offsets[i])
                            value=field[(integer[:,2]+dz).clip(0,nz-1),(integer[:,1]+dy).clip(0,ny-1),(integer[:,0]+dx).clip(0,nx-1)].double()
                            features[:,axis]=features[:,axis]+weights[i]*value
                    values=dict(point=features,patch=torch.stack(patch,1),kind=torch.full((len(indices),),kind,dtype=torch.int64),
                        alpha=torch.full((len(indices),),alpha,dtype=torch.float64),middle=middle_label,end=end)
                    for name,value in values.items():group['arrays'][name].append(value.numpy())
                    group['samples'].append(dict(identifier=trajectory['identifier'],state=state_index,kind=kind,points=len(indices)))
            scene_count+=1;print(trajectory['identifier'],'offline train-only transport labels prepared',flush=True)
    shards=[];energy=np.zeros((4,3));population=np.zeros(4,dtype=np.int64)
    for key,group in groups.items():
        arrays={name:np.concatenate(values,axis=0)for name,values in group['arrays'].items()}
        path=args.output/(key+'.npz');np.savez_compressed(path,**arrays)
        for kind in range(4):
            value=arrays['end'][arrays['kind']==kind];energy[kind]+=(value*value).sum(0);population[kind]+=len(value)
        shards.append(dict(key=key,counts_xyz=list(group['counts']),path=path.name,sha256=digest(path),bytes=path.stat().st_size,
            points=len(arrays['kind']),samples=group['samples']))
    scales=np.maximum(np.sqrt(energy/population[:,None]),reg['absolute_velocity_scale_floor'])
    normalization=dict(schema='rheon-learned-sampler-physical-training-v1',scales=scales.tolist(),inherited_mix=tensors['point'].sum(dim=(2,3,4)).reshape(4,3,3).tolist(),
        scale_policy='Train-only RMS negative midpoint velocity per carrier/component, fixed absolute floor',fixed_point_provenance=provenance,
        label_only_native_teacher=True,development_or_reserved_used=False)
    write(args.output/'normalization.json',normalization)
    write(args.output/'manifest.json',dict(status='verified_train_only_sampler_view',scenes=scene_count,shards=shards,points=int(population.sum()),
        source_evidence=sources,normalization_sha256=digest(args.output/'normalization.json'),source_sha256=reg['source_sha256'],
        protocol_sha256=digest(args.protocol),optimizer_updates=0,development_or_reserved_used=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
