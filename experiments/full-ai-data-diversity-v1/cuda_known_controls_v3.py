"""GPU prescribed-force construction from compact known scalar controls.

CPU work supplies O(1) known scalar parameters; CUDA evaluates coordinate/phase
tables and full force fields. No physical state or labels enter.
"""
import ctypes as C
import json
import math
import torch
from cuda_kernel_runtime_v1 import check
from cuda_known_controls_v2 import CudaKnownControls as PreviousControls
from recurrent_sampler_native import dimensions
from legacy_exact_forcing import schedule


class CudaKnownControls(PreviousControls):
    def __init__(self,build):
        super().__init__(build)
        for name in ('known_axes','known_control_scalar'):
            function=C.c_void_p()
            check(self.driver.cuModuleGetFunction(C.byref(function),self.module,name.encode()),'cuModuleGetFunction '+name)
            self.functions[name]=function

    def generate(self,scene,start,horizons,substeps=1):
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
        results=[]
        for fine in range(horizons*substeps):
            step=start*substeps+fine;time=float(step)*dt;frame=scene['forcing_schedule'][step//substeps]
            center=[extents[a]*frame['center'][a] for a in range(3)]
            values=torch.empty((3,3,length,3),dtype=torch.float64,device='cuda')
            axis_arguments=[C.c_int(nx),C.c_int(ny),C.c_int(nz),C.c_int(length),C.c_double(h),
                *(C.c_double(v) for v in extents),*(C.c_double(v) for v in center),C.c_double(radius),
                C.c_double(omega),C.c_double(time),C.c_double(phase),C.c_void_p(values.data_ptr())]
            self.launch('known_axes',9*length,axis_arguments)
            outputs=[torch.empty(shape,dtype=torch.float64,device='cuda') for shape in dimensions(scene['counts'])[:3]]
            ptr=lambda v:C.c_void_p(v.data_ptr())
            arguments=[C.c_int(nx),C.c_int(ny),C.c_int(nz),C.c_int(length),ptr(values),*(C.c_double(v) for v in direction),
                C.c_double(scene['acceleration']*frame['pulse']),C.c_double(scene['swirl']),C.c_double(scene['fluctuation']),C.c_double(dt),
                *(ptr(v) for v in outputs)]
            self.launch('known_control_scalar',sum(v.numel() for v in outputs),arguments)
            results.append(outputs)
        return results
