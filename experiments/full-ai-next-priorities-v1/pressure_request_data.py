"""Load verified train-only velocities and separate offline future request targets."""
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from temporal_codec import decode
from temporal_data import make_features,feature_at


def digest(raw):return hashlib.sha256(raw).hexdigest()


def load(manifest_path,normalization_path,shard_paths,device):
    manifest=json.loads(Path(manifest_path).read_text())
    raw=Path(normalization_path).read_bytes();norm=json.loads(raw)
    if manifest['schema']!='rheon-causal-pressure-request-training-view-v1'or manifest['status']!='verified':
        raise ValueError('Verified causal pressure-request view required')
    if manifest['development_or_reserved_used']or manifest['current_pressure_input']or manifest['reference_pressure_input']:
        raise ValueError('Training-only causal view required')
    if digest(raw)!=manifest['normalization_sha256']or norm['development_or_reserved_used']:
        raise ValueError('Training normalization changed')
    if len(shard_paths)!=len(manifest['shards']):raise ValueError('Complete declared shards required')
    blocks=[]
    for path,spec in zip(shard_paths,manifest['shards']):
        block=Path(path).read_bytes()
        if len(block)!=spec['bytes']or digest(block)!=spec['sha256']:raise ValueError('Training shard changed')
        blocks.append(block)
    raw=b''.join(blocks)
    if len(raw)>250*1024**2 or digest(raw)!=manifest['archive_sha256']:
        raise ValueError('Bounded training archive changed')
    parents=[]
    with np.load(io.BytesIO(raw),allow_pickle=False)as arrays:
        expected={spec['key']for parent in manifest['parents']for spec in parent['fields'].values()}
        if set(arrays.files)!=expected:raise ValueError('Training field inventory differs')
        for parent in manifest['parents']:
            if parent['split']!='train'or set(parent['fields'])!={'vx','vy','vz','q'}:
                raise ValueError('Only training velocities and offline request targets permitted')
            nx,ny,nz=parent['scene']['counts'];steps=parent['states'];velocity=[];q=None
            shapes=dict(vx=(nz,ny,nx+1),vy=(nz,ny+1,nx),vz=(nz+1,ny,nx),q=(nz,ny,nx))
            for name in ('vx','vy','vz','q'):
                spec=parent['fields'][name];value=decode(arrays[spec['key']])
                if list(value.shape)!=spec['shape']or digest(value.tobytes())!=spec['full_float32_sha256']:
                    raise ValueError('Decoded full field identity failed')
                tensor=torch.from_numpy(value).reshape(steps,1,*shapes[name])
                if name=='q':q=(tensor/norm['scales']['q']).to(device)
                else:
                    shape=shapes[name]
                    velocity.append(F.pad(tensor/norm['scales']['velocity'],
                        (0,nx+1-shape[2],0,ny+1-shape[1],0,nz+1-shape[0]),mode='replicate'))
            # Targets remain separate and are never concatenated with inputs.
            parents.append(dict(identifier=parent['identifier'],scene=parent['scene'],
                velocity=torch.cat(velocity,dim=1).to(device),q=q))
    if len(parents)!=16:raise ValueError('All16 original whole training scenes required')
    return manifest,norm,parents


def features(parent,scales,horizons,device):
    return make_features(dict(scene=parent['scene'],states=parent['velocity']),scales,horizons,device)


def inputs(parent,cache,step,horizons,device):
    velocity=parent['velocity'][step:step+1]
    known=feature_at(cache,step,horizons)[:,7:]
    scalars=torch.tensor([[1/parent['scene']['cells_per_metre'],1/60,1000.,.001]],device=device,dtype=torch.float32)
    return velocity,known,scalars
