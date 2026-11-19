"""GPU prescribed-force construction from compact known scalar controls.

CPU work supplies O(1) known scalar parameters; CUDA evaluates coordinate/phase
tables and full force fields. No physical state or labels enter.
"""
import ctypes as C
import json
import math
import torch
import numpy as np
from cuda_kernel_runtime_v1 import check
from cuda_known_controls_v3 import CudaKnownControls as PreviousControls
from recurrent_sampler_native import dimensions
from legacy_exact_forcing import schedule


class CudaKnownControls(PreviousControls):
    def __init__(self,build):
        super().__init__(build)
        for name in ('known_axes_dynamic','known_control_dynamic'):
            function=C.c_void_p()
            check(self.driver.cuModuleGetFunction(C.byref(function),self.module,name.encode()),'cuModuleGetFunction '+name)
            self.functions[name]=function

    def parameters(self,scene,start,horizons,substeps=1):
        if (substeps not in (1,8) or horizons not in (1,2,3,4) or not 0<=start<start+horizons<=scene['steps']
                or any(type(n)is not int or not 5<=n<=256 for n in scene['counts'])):
            raise ValueError('Registered known-control interval and counts required')
        if 'forcing_schedule' not in scene:
            if substeps!=1:raise ValueError('Legacy schedules support one internal step')
            scene=schedule(json.dumps(scene,sort_keys=True,separators=(',',':')))
        if len(scene['forcing_schedule'])!=scene['steps']:raise ValueError('Sampled macro-control schedule required')
        nx,ny,nz=scene['counts'];length=max(scene['counts'])+1;h=1./scene['cells_per_metre'];dt=1./(60.*substeps)
        seed=scene['seed'];mask=(1<<64)-1
        def next_value():
            nonlocal seed
            seed=(seed+0x9e3779b97f4a7c15)&mask;value=seed
            value=((value^(value>>30))*0xbf58476d1ce4e5b9)&mask
            value=((value^(value>>27))*0x94d049bb133111eb)&mask
            return float((value^(value>>31))>>11)/float(1<<53)
        direction=[];norm=0.
        for a in range(3):
            value=2.*next_value()-1.;direction.append(value);norm=norm+value*value
        norm=max(math.sqrt(norm),1e-12);direction=[v/norm for v in direction]
        tau=6.283185307179586476925286766559;phase=next_value()*tau;omega=tau*scene['frequency']
        extents=[float(n)*h for n in scene['counts']];radius=scene['radius']*min(extents)
        rows=np.empty((horizons*substeps,18),dtype='<f8')
        for fine in range(horizons*substeps):
            step=start*substeps+fine;time=float(step)*dt;frame=scene['forcing_schedule'][step//substeps]
            center=[extents[a]*frame['center'][a] for a in range(3)]
            rows[fine]=[h,*extents,*center,radius,omega,time,phase,*direction,
                scene['acceleration']*frame['pulse'],scene['swirl'],scene['fluctuation'],dt]
        return torch.from_numpy(rows)

    def generate_from_packet(self,counts,packet):
        if packet.device.type!='cuda' or packet.dtype!=torch.float64 or packet.ndim!=2 or packet.shape[1]!=18 or not packet.is_contiguous():
            raise ValueError('Contiguous CUDA double known-input packet required')
        nx,ny,nz=counts;length=max(counts)+1;results=[]
        for row in packet.unbind(0):
            values=torch.empty((3,3,length,3),dtype=torch.float64,device='cuda')
            ptr=lambda v:C.c_void_p(v.data_ptr())
            args=[C.c_int(nx),C.c_int(ny),C.c_int(nz),C.c_int(length),ptr(row),ptr(values)]
            self.launch('known_axes_dynamic',9*length,args)
            outputs=[torch.empty(shape,dtype=torch.float64,device='cuda') for shape in dimensions(counts)[:3]]
            args=[C.c_int(nx),C.c_int(ny),C.c_int(nz),C.c_int(length),ptr(values),ptr(row),*(ptr(v) for v in outputs)]
            self.launch('known_control_dynamic',sum(v.numel() for v in outputs),args)
            results.append(outputs)
        return results

    def generate(self,scene,start,horizons,substeps=1):
        packet=self.parameters(scene,start,horizons,substeps).to('cuda')
        return self.generate_from_packet(tuple(scene['counts']),packet)
