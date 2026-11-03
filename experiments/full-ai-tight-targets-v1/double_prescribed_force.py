"""Known external forcing in double, before addition to stored carrier velocity.

This is conditioning, not a predicted field or a learned dynamics operator.
"""
import math
import torch
from temporal_features_v2 import direction_phase,force_coordinates


def force_features(scene,step,horizons):
    counts=tuple(scene['counts']);h=1./scene['cells_per_metre'];extents=[n*h for n in counts]
    radius=scene['radius']*min(extents);direction,phase=direction_phase(scene['seed']);omega=math.tau*scene['frequency']
    times=((torch.arange(horizons,dtype=torch.float64)+step)*(1./60.)).reshape(-1,1,1,1)
    centers=[extents[a]*(scene['center'][a]+.05*torch.sin(omega*times+phase+a))for a in range(3)]
    pulse=.65+.35*torch.sin(omega*times+phase);components=[]
    for d,positions in enumerate(force_coordinates(counts,scene['cells_per_metre'],'cpu')):
        offset=[(positions[a]-centers[a])/radius for a in range(3)]
        r2=sum(value.square()for value in offset)
        taper=1.-r2/16.
        envelope=torch.where(r2<16.,torch.exp(-.5*r2)*taper.square(),torch.zeros_like(r2))
        a=(d+1)%3;b=(d+2)%3;rotation=direction[a]*offset[b]-direction[b]*offset[a]
        wave=sum(torch.sin(math.tau*positions[a]/extents[a]+omega*times*(a+1)+phase+d)for a in range(3))/3.
        value=((scene['acceleration']*pulse*envelope)*(direction[d]+scene['swirl']*rotation+scene['fluctuation']*wave))*(1./60.)
        boundary=[slice(None)]*4;boundary[3-d]=0;value[tuple(boundary)]=0.
        boundary[3-d]=-1;value[tuple(boundary)]=0.
        components.append(value)
    return torch.stack(components,dim=1).unsqueeze(0)
