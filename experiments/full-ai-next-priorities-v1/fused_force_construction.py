"""Same known forcing, with broadcast field arithmetic in a strict CPU kernel."""
import ctypes
import math
from functools import lru_cache
import torch
from temporal_features_v2 import direction_phase,force_coordinates,geometry
from study_support import BASE,HERE,read,sha


@lru_cache(maxsize=1)
def library():
    directory=BASE/'fused-force-construction-build-01';receipt=read(directory/'receipt.json')
    binary=directory/'libprescribed_force.so'
    if sha(binary)!=receipt['binary_sha256']or sha(HERE/'fused_force_construction.cpp')!=receipt['source_sha256']:
        raise ValueError('Registered prescribed-force execution changed')
    value=ctypes.CDLL(str(binary.resolve()))
    value.rheon_prescribed_force.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int64]*5+[ctypes.c_double]*2
    value.rheon_prescribed_force.restype=ctypes.c_int
    return value


@lru_cache(maxsize=64)
def fixed_geometry(counts,cells_per_metre):
    # Only spatial coordinates are cached. Time-dependent force work is timed.
    stride=max(counts)+1
    with torch.inference_mode(False):
        positions=torch.zeros((3,3,1,stride),dtype=torch.float64)
        for d,axes in enumerate(force_coordinates(counts,cells_per_metre,'cpu')):
            for a,axis in enumerate(axes):positions[d,a,0,:counts[a]+1]=axis.reshape(-1)
        extents=torch.tensor([n/cells_per_metre for n in counts],dtype=torch.float64).reshape(1,3,1,1)
        spatial_phase=math.tau*positions/extents
    return positions,spatial_phase,stride


def force_features(scene,step,horizons,velocity_scale,device='cpu'):
    if str(device)!='cpu' or horizons not in (1,2,3,4):raise ValueError('CPU registered horizon required')
    counts=tuple(scene['counts']);positions,spatial_phase,stride=fixed_geometry(counts,scene['cells_per_metre'])
    extents=[n/scene['cells_per_metre']for n in counts];radius=scene['radius']*min(extents)
    direction,phase=direction_phase(scene['seed']);omega=math.tau*scene['frequency']
    times=((torch.arange(horizons,dtype=torch.float64)+step)/60).reshape(horizons,1)
    centers=torch.stack([extents[a]*(scene['center'][a]+.05*torch.sin(omega*times+phase+a))for a in range(3)])
    offsets=((positions-centers.unsqueeze(0))/radius).contiguous()
    frequency=torch.tensor([1,2,3],dtype=torch.float64).reshape(1,3,1,1)
    component=torch.arange(3,dtype=torch.float64).reshape(3,1,1,1)
    waves=torch.sin(spatial_phase+omega*times.reshape(1,1,horizons,1)*frequency+phase+component).contiguous()
    amplitude=(scene['acceleration']*(.65+.35*torch.sin(omega*times+phase))).contiguous()
    direction_tensor=torch.tensor(direction,dtype=torch.float64)
    nx,ny,nz=counts;output=torch.empty((horizons,3,nz+1,ny+1,nx+1),dtype=torch.float32)
    status=library().rheon_prescribed_force(offsets.data_ptr(),waves.data_ptr(),amplitude.data_ptr(),direction_tensor.data_ptr(),
        output.data_ptr(),nx+1,ny+1,nz+1,horizons,stride,scene['swirl'],scene['fluctuation'])
    if status:raise ValueError('Prescribed-force kernel rejected its input: '+str(status))
    return (output/velocity_scale).reshape(1,3*horizons,nz+1,ny+1,nx+1)


def auxiliary(scene,step,horizons,scales,device='cpu'):
    counts=tuple(scene['counts']);coordinates,_=geometry(counts,str(device))
    constants=torch.tensor([math.log(n/16)for n in counts]+[math.log(16/scene['cells_per_metre'])],dtype=torch.float32)
    constants=constants.reshape(1,4,1,1,1).expand(1,4,*coordinates.shape[1:])
    return torch.cat((coordinates.unsqueeze(0),constants,force_features(scene,step,horizons,scales['velocity'],device)),dim=1)
