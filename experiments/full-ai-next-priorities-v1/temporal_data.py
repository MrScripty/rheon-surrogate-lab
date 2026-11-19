"""Decode only the declared training trajectories and prepare causal model inputs."""
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from temporal_codec import decode
from temporal_features_v2 import force_features,geometry

def digest(raw):return hashlib.sha256(raw).hexdigest()

def load(manifest_path,normalization_path,shard_paths,device):
    manifest=json.loads(Path(manifest_path).read_text())
    norm_raw=Path(normalization_path).read_bytes();norm=json.loads(norm_raw)
    if manifest['schema']!='rheon-temporal-training-view-v1'or manifest['status']!='verified':raise ValueError('Verified full trajectory fitting view required')
    if manifest['normalization_sha256']!=digest(norm_raw)or norm['development_or_reserved_used']:
        raise ValueError('Training-only normalization identity required')
    if len(shard_paths)!=len(manifest['shards']):raise ValueError('All typed trajectory shards required')
    blocks=[]
    for path,spec in zip(shard_paths,manifest['shards']):
        raw=Path(path).read_bytes()
        if len(raw)!=spec['bytes']or digest(raw)!=spec['sha256']:raise ValueError('Training shard identity failed')
        blocks.append(raw)
    raw=b''.join(blocks)
    if digest(raw)!=manifest['archive_sha256']:raise ValueError('Reassembled training archive changed')
    if len(raw)>245*1024**2:raise ValueError('Bounded training archive exceeded')
    parents=[]
    with np.load(io.BytesIO(raw),allow_pickle=False)as arrays:
        expected={spec['key']for parent in manifest['parents']for spec in parent['fields'].values()}
        if set(arrays.files)!=expected:raise ValueError('Training field inventory differs')
        for parent in manifest['parents']:
            if parent['split']!='train':raise ValueError('Evaluation scene in fitting view')
            counts=parent['scene']['counts'];nx,ny,nz=counts
            shapes={'pressure':(nz,ny,nx),'vx':(nz,ny,nx+1),'vy':(nz,ny+1,nx),'vz':(nz+1,ny,nx),'tracer':(nz,ny,nx)}
            padded=[]
            for name in ('pressure','vx','vy','vz','tracer'):
                spec=parent['fields'][name];values=decode(arrays[spec['key']])
                if list(values.shape)!=spec['shape']or digest(values.tobytes())!=spec['full_float32_sha256']:
                    raise ValueError('Full decoded trajectory identity failed')
                shape=shapes[name]
                value=torch.from_numpy(values).reshape(parent['states'],1,*shape)
                if name=='pressure':value=value-value.mean(dim=(2,3,4),keepdim=True)
                value=value/norm['scales']['velocity'if name in ('vx','vy','vz')else name]
                padded.append(F.pad(value,(0,nx+1-shape[2],0,ny+1-shape[1],0,nz+1-shape[0]),mode='replicate'))
            parents.append(dict(identifier=parent['identifier'],scene=parent['scene'],states=torch.cat(padded,dim=1).to(device)))
    return manifest,norm,parents

def make_features(parent,scales,horizons,device):
    scene=parent['scene'];counts=tuple(scene['counts']);steps=parent['states'].shape[0]
    # Prescribed forces are deterministic known inputs. No future velocity,
    # pressure request, or target state enters this cache.
    forces=force_features(scene,0,steps+horizons,scales['velocity'],'cpu')
    forces=forces.reshape(steps+horizons,3,*forces.shape[2:]).to(device)
    coordinates,_=geometry(counts,str(device))
    constants=torch.tensor([np.log(n/16)for n in counts]+[np.log(16/scene['cells_per_metre'])],device=device,dtype=torch.float32)
    constant=torch.cat((coordinates,constants.reshape(4,1,1,1).expand(4,*coordinates.shape[1:])),dim=0).unsqueeze(0)
    return dict(forces=forces,constant=constant)

def feature_at(cache,step,horizons):
    force=cache['forces'][step:step+horizons].reshape(1,3*horizons,*cache['forces'].shape[2:])
    return torch.cat((cache['constant'],force),dim=1)
