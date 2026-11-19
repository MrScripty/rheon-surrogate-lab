"""Causal learned local pressure-request predictor; no pressure inverse at inference."""
import torch
from torch import nn


class CausalRequestNet(nn.Module):
    def __init__(self,horizons,velocity_scale,q_scale,width=8):
        super().__init__()
        if horizons not in (1,2,3,4)or width!=8:
            raise ValueError('Bounded causal request architecture required')
        if velocity_scale<=0 or q_scale<=0:raise ValueError('Positive train-only scales required')
        self.horizons=horizons;self.velocity_scale=float(velocity_scale);self.q_scale=float(q_scale)
        inputs=3+3*horizons
        self.linear=nn.Conv3d(inputs,horizons,3,padding=1,padding_mode='replicate',bias=False)
        self.left=nn.Conv3d(inputs,width,3,padding=1,padding_mode='replicate',bias=False)
        self.right=nn.Conv3d(inputs,width,3,padding=1,padding_mode='replicate',bias=False)
        self.quadratic=nn.Conv3d(2*width,horizons,3,padding=1,padding_mode='replicate',bias=False)
        self.viscous=nn.Conv3d(inputs,horizons,5,padding=2,padding_mode='replicate',bias=False)
        nn.init.zeros_(self.linear.weight);nn.init.zeros_(self.viscous.weight)
        nn.init.normal_(self.left.weight,std=.05);nn.init.normal_(self.right.weight,std=.05)
        nn.init.normal_(self.quadratic.weight,std=.01)

    def forward(self,velocity,known_forces,scalars):
        if velocity.ndim!=5 or velocity.shape[1]!=3 or known_forces.shape[1]!=3*self.horizons:
            raise ValueError('Current staggered velocities and prescribed future forcing required')
        if velocity.shape[0]!=known_forces.shape[0]or velocity.shape[2:]!=known_forces.shape[2:]:
            raise ValueError('Common padded face grid required')
        if scalars.shape!=(velocity.shape[0],4):raise ValueError('Per-example h,dt,rho,mu required')
        h,dt,rho,mu=scalars.unbind(1)
        values=torch.cat((velocity,known_forces),dim=1)
        left=self.left(values);right=self.right(values)
        polynomial=self.quadratic(torch.cat((left*right,left.abs()*right),dim=1))
        alpha=(dt/h*self.velocity_scale).reshape(-1,1,1,1,1)
        beta=(mu/rho*dt/h.square()).reshape(-1,1,1,1,1)
        # Unit conditioning only. Every spatial coefficient is learned.
        # Native q has units rho*h/dt times a velocity difference. This scalar
        # conversion does not assemble divergence or perform a pressure solve.
        units=(rho*h/dt*self.velocity_scale/self.q_scale).reshape(-1,1,1,1,1)
        predicted=units*(self.linear(values)+alpha*polynomial+beta*self.viscous(values))
        predicted=predicted[:,:,:-1,:-1,:-1]
        predicted=predicted-predicted.mean(dim=(2,3,4),keepdim=True)
        return predicted.unsqueeze(2)
