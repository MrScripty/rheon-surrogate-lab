"""CUDA-only anchored temporal readout; 3*S native-data-trained coefficients."""
import torch
from torch import nn


class MacroReadout(nn.Module):
    def __init__(self, stages, device='cuda'):
        super().__init__()
        if stages not in (3,4):raise ValueError('Registered three/four-stage pilot required')
        self.stages=stages
        self.residual_weights=nn.Parameter(torch.zeros((3,stages),dtype=torch.float64,device=device))

    def residuals(self,design):
        if design.device.type!='cuda' or self.residual_weights.device.type!='cuda' or design.dtype!=torch.float64:
            raise ValueError('All learned readout evaluation requires CUDA double')
        if design.shape[-3:]!=(4,self.stages+1,self.stages+1):
            raise ValueError('Four metric QR factors and exact stage shape required')
        # Pressure equation and pressure field share the pressure readout.
        mapping=torch.tensor([0,0,1,2],dtype=torch.int64,device='cuda')
        z=self.residual_weights[mapping]
        v=torch.cat((z,-torch.ones((4,1),dtype=torch.float64,device='cuda')),dim=-1)
        return torch.matmul(design,v[...,None])[...,0]

    def objective(self,design):
        return self.residuals(design).square().sum(dim=-1).mean()

    def read(self,features):
        if len(features)!=self.stages+1 or self.residual_weights.device.type!='cuda':
            raise ValueError('All causal stage states and CUDA learned weights required')
        anchor=features[-1];out={}
        for name in ('pressure','vx','vy','vz','tracer'):
            group=0 if name=='pressure' else 2 if name=='tracer' else 1
            base=anchor[name].double();delta=torch.zeros_like(base)
            for stage in range(self.stages):
                delta=delta+self.residual_weights[group,stage]*(features[stage][name].double()-base)
            value=torch.where(delta==0.,base,base+delta)
            if name=='pressure':value=value-value[:1]
            out[name]=value.contiguous() if name=='pressure' else value.float().contiguous()
        return out
