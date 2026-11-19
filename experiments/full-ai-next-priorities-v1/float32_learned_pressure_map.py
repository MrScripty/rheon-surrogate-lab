"""Lower precision execution of the same learned unit-response map; no inverse."""
import torch
from torch.nn import functional as F
from full_learned_pressure_map import FullLearnedPressureMap


class Float32LearnedPressureMap(FullLearnedPressureMap):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.float32_routes={};self.conversion=[]
        with torch.inference_mode(False),torch.no_grad():
            for counts,route in self.routes.items():
                if route is None:self.float32_routes[counts]=None;continue
                original=getattr(self,route);converted=original.float().contiguous()
                if not bool(torch.isfinite(converted).all()):raise ValueError('Finite learned-map float32 coefficients required')
                name=route+'_float32';self.register_buffer(name,converted,persistent=False);self.float32_routes[counts]=name
                self.conversion.append(dict(counts_zyx=list(counts),original_map_bytes=original.numel()*8,
                    converted_map_bytes=converted.numel()*4,
                    coefficient_relative_l2=float((original-converted.double()).norm()/original.norm())))

    def forward(self,q):
        if q.ndim!=5 or q.shape[1]!=1 or not 1<=q.shape[0]<=4 or q.dtype not in(torch.float32,torch.float64):
            raise ValueError('One to four colocated pressure-unit requests required')
        counts=tuple(q.shape[2:])
        if counts not in self.float32_routes:raise ValueError('Pressure geometry not exported/qualified')
        route=self.float32_routes[counts]
        if route is None:return self.source(q)
        matrix=getattr(self,route)
        if matrix.dtype!=torch.float32 or matrix.device!=q.device:
            raise ValueError('Colocated float32 neural map required')
        if not bool(torch.isfinite(q).all()):raise FloatingPointError('Finite neural-map request required')
        request=q.flatten(1).float()
        if not bool(torch.isfinite(request).all()):raise FloatingPointError('Pressure request overflows float32 execution')
        return F.linear(request,matrix).double().reshape_as(q)
