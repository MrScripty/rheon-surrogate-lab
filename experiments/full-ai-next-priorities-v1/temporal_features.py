"""Causal current-state features and prescribed forcing on native staggered faces."""
import math
from functools import lru_cache
import torch
from torch.nn import functional as F

FIELD_NAMES=('pressure','vx','vy','vz','tracer')

def direction_phase(seed):
    state=seed;mask=(1<<64)-1
    def next_value():
        nonlocal state
        state=(state+0x9e3779b97f4a7c15)&mask
        z=state;z=((z^(z>>30))*0xbf58476d1ce4e5b9)&mask
        z=((z^(z>>27))*0x94d049bb133111eb)&mask
        return ((z^(z>>31))>>11)/(1<<53)
    direction=[2*next_value()-1 for _ in range(3)]
    norm=max(math.sqrt(sum(v*v for v in direction)),1e-12)
    return [v/norm for v in direction],next_value()*math.tau

@lru_cache(maxsize=64)
def geometry(counts,device):
    nx,ny,nz=counts
    with torch.inference_mode(False):
        coordinates=torch.stack(torch.meshgrid(
            torch.linspace(-1,1,nz+1,device=device),torch.linspace(-1,1,ny+1,device=device),
            torch.linspace(-1,1,nx+1,device=device),indexing='ij'))[[2,1,0]].float()
        masks=torch.zeros((5,nz+1,ny+1,nx+1),device=device,dtype=torch.float32)
        masks[0,:nz,:ny,:nx]=1;masks[4,:nz,:ny,:nx]=1
        masks[1,:nz,:ny,1:nx]=1;masks[2,:nz,1:ny,:nx]=1;masks[3,1:nz,:ny,:nx]=1
    return coordinates,masks

def packed_state(fields,counts,scales,device='cpu'):
    if set(fields)!=set(FIELD_NAMES):raise ValueError('Only current physical state fields may enter prediction')
    nx,ny,nz=counts
    shapes=((nz,ny,nx),(nz,ny,nx+1),(nz,ny+1,nx),(nz+1,ny,nx),(nz,ny,nx))
    padded=[]
    for name,shape in zip(FIELD_NAMES,shapes):
        value=torch.as_tensor(fields[name],device=device).reshape(1,1,*shape).float()
        if name=='pressure':value=value-value.mean(dim=(2,3,4),keepdim=True)
        scale=scales['velocity'if name in ('vx','vy','vz')else name]
        value=value/scale
        padded.append(F.pad(value,(0,nx+1-shape[2],0,ny+1-shape[1],0,nz+1-shape[0]),mode='replicate'))
    return torch.cat(padded,dim=1)

def force_features(scene,step,horizons,velocity_scale,device='cpu'):
    counts=tuple(scene['counts']);nx,ny,nz=counts;h=1/scene['cells_per_metre']
    extents=[n*h for n in counts];radius=scene['radius']*min(extents)
    direction,phase=direction_phase(scene['seed']);omega=math.tau*scene['frequency']
    all_fields=[]
    for horizon in range(horizons):
        time_s=(step+horizon)/60
        center=[extents[a]*(scene['center'][a]+.05*math.sin(omega*time_s+phase+a))for a in range(3)]
        pulse=.65+.35*math.sin(omega*time_s+phase)
        for d in range(3):
            dimensions=[n+(a==d)for a,n in enumerate(counts)]
            positions=[]
            for a,n in enumerate(dimensions):
                coordinate=(torch.arange(n,device=device,dtype=torch.float64)+(0. if a==d else .5))*h
                shape=[1,1,1];shape[2-a]=n;positions.append(coordinate.reshape(shape))
            offset=[(positions[a]-center[a])/radius for a in range(3)]
            r2=sum(v.square()for v in offset)
            envelope=torch.where(r2<16,torch.exp(-.5*r2)*(1-r2/16).square(),torch.zeros_like(r2))
            a=(d+1)%3;b=(d+2)%3
            rotation=direction[a]*offset[b]-direction[b]*offset[a]
            wave=sum(torch.sin(math.tau*positions[a]/extents[a]+omega*time_s*(a+1)+phase+d)for a in range(3))/3
            value=(scene['acceleration']*pulse*envelope*(direction[d]+scene['swirl']*rotation+scene['fluctuation']*wave)/60).float()
            boundary=[slice(None)]*3;boundary[2-d]=0;value[tuple(boundary)]=0
            boundary[2-d]=-1;value[tuple(boundary)]=0
            value=value.reshape(1,1,*dimensions[::-1])/velocity_scale
            value=F.pad(value,(0,nx+1-dimensions[0],0,ny+1-dimensions[1],0,nz+1-dimensions[2]),mode='replicate')
            all_fields.append(value)
    return torch.cat(all_fields,dim=1)

def auxiliary(scene,step,horizons,scales,device='cpu'):
    counts=tuple(scene['counts']);coordinates,_=geometry(counts,str(device))
    spatial=tuple(coordinates.shape[1:])
    constants=torch.tensor([math.log(n/16)for n in counts]+[math.log(16/scene['cells_per_metre'])],device=device,dtype=torch.float32)
    constants=constants.reshape(1,4,1,1,1).expand(1,4,*spatial)
    forces=force_features(scene,step,horizons,scales['velocity'],device)
    return torch.cat((coordinates.unsqueeze(0),constants,forces),dim=1)

def unpack_outputs(predicted,counts,scales):
    nx,ny,nz=counts
    slices=((slice(0,nz),slice(0,ny),slice(0,nx)),
            (slice(0,nz),slice(0,ny),slice(0,nx+1)),
            (slice(0,nz),slice(0,ny+1),slice(0,nx)),
            (slice(0,nz+1),slice(0,ny),slice(0,nx)),
            (slice(0,nz),slice(0,ny),slice(0,nx)))
    results=[]
    for horizon in range(predicted.shape[1]):
        state={}
        for field,(name,selection)in enumerate(zip(FIELD_NAMES,slices)):
            scale=scales['velocity'if name in ('vx','vy','vz')else name]
            value=predicted[0,horizon,field][selection]*scale
            if name=='pressure':value=value.to(torch.float64);value=value-value.flatten()[0]
            state[name]=value.contiguous().reshape(-1)
        results.append(state)
    return results
