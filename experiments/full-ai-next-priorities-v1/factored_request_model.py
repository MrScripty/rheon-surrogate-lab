"""Learned frozen linear flux and a velocity-only neural nonlinear residual."""
import torch
from torch import nn
from torch.nn import functional as F


class CausalRequestNet(nn.Module):
    def __init__(self,horizons,velocity_scale,q_scale,width=16):
        super().__init__()
        if horizons!=1 or width!=16:raise ValueError('Registered one-step factored component required')
        self.horizons=1;self.velocity_scale=float(velocity_scale);self.q_scale=float(q_scale)
        self.flux=nn.Conv3d(3,1,3,padding=1,padding_mode='replicate',bias=False)
        self.left=nn.Conv3d(3,width,3,padding=1,padding_mode='replicate',bias=False)
        self.right=nn.Conv3d(3,width,3,padding=1,padding_mode='replicate',bias=False)
        self.mix=nn.Conv3d(5*width,width,1,bias=False)
        self.head=nn.Conv3d(width,1,3,padding=1,padding_mode='replicate',bias=False)
        nn.init.zeros_(self.flux.weight);self.flux.weight.requires_grad_(False)
        nn.init.zeros_(self.head.weight)

    def forward(self,velocity,known_forces,scalars):
        if velocity.shape[1]!=3 or known_forces.shape!=velocity.shape or scalars.shape!=(velocity.shape[0],4):
            raise ValueError('Current velocity, one known forcing field and h/dt/rho/mu required')
        h,dt,rho,_=scalars.double().unbind(1)
        alpha=(dt/h*self.velocity_scale).reshape(-1,1,1,1,1)
        left=self.left(velocity);right=self.right(velocity)
        features=torch.cat((left,right,left*right,left.abs()*right,(alpha*left.double()).float()),dim=1)
        nonlinear=self.head(F.silu(self.mix(features))).double()
        combined=velocity.double()+known_forces.double()
        linear=F.conv3d(F.pad(combined,(1,1,1,1,1,1),mode='replicate'),self.flux.weight.double())
        units=(rho*h/dt*self.velocity_scale/self.q_scale).reshape(-1,1,1,1,1)
        predicted=(units*(linear+alpha*nonlinear))[:,:,:-1,:-1,:-1]
        predicted=predicted-predicted.mean(dim=(2,3,4),keepdim=True)
        return predicted.float().unsqueeze(2)
