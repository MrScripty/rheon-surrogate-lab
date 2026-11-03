"""Untrained recurrent neural sampler prototype with exact restored pressure.

The coordinate gates/mixing are neural parameters; zero/seeded initialization
is an execution experiment, not a fitted full-state or timing-target result.
"""
import math
import torch
from torch import nn
from profile_frozen_components import ROOT
from recurrent_sampler_assets import source_tensors,csr
from recurrent_sampler_native import transport,spatial,forcing,dimensions
from restored_modal_core import RestoredModalCore


class RecurrentSamplerPrototype(nn.Module):
    def __init__(self,library_path,initialization='neutral',seed=2026102051):
        super().__init__();tensors,self.asset_identity=source_tensors()
        for name in ('point','viscosity','flux'):self.register_parameter(name,nn.Parameter(tensors[name],requires_grad=False))
        self.coefficients={name:csr(value)for name,value in tensors.items()}
        self.mid_gate=nn.Parameter(torch.zeros((4,3),dtype=torch.float64))
        self.mix=nn.Parameter(tensors['point'].sum(dim=(2,3,4)).reshape(4,3,3).clone())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            if initialization=='seeded':
                with torch.no_grad():
                    self.mid_gate.uniform_(-.25,.75)
                    self.mix.add_(torch.randn(self.mix.shape,dtype=torch.float64)*.02)
            elif initialization!='neutral':raise ValueError('Declared neutral or seeded execution initializer required')
        self.pressure_core=RestoredModalCore(ROOT/'runs/full-ai-tight-targets-v1/modal-pressure-pair-01',library_path)
        self.initialization=initialization;self.seed=seed
        self.frozen_versions={name:value._version for name,value in self.named_parameters()if not value.requires_grad}

    def _guard(self,fields,counts):
        if set(fields)!={'pressure','vx','vy','vz','tracer'}:raise ValueError('Complete current native state required')
        pressure=torch.as_tensor(fields['pressure'])
        if pressure.device.type!='cpu' or pressure.dtype!=torch.float64 or pressure.numel()!=math.prod(counts)or not bool(torch.isfinite(pressure).all()):
            raise ValueError('Finite native double current pressure required')
        values=[]
        for name,shape in zip(('vx','vy','vz','tracer'),dimensions(counts)):
            value=torch.as_tensor(fields[name])
            if value.device.type!='cpu'or value.dtype!=torch.float32 or value.numel()!=math.prod(shape)or not value.is_contiguous():
                raise ValueError('Exact contiguous native float32 carrier layout required')
            values.append(value.reshape(shape))
        if any(value._version!=self.frozen_versions[name]for name,value in self.named_parameters()if name in self.frozen_versions):
            raise ValueError('Frozen learned weights changed; cached operator invalid')
        return values

    @torch.inference_mode()
    def predict(self,fields,scene,step,horizons):
        if horizons not in (1,2,3,4)or scene['cells_per_metre']not in (8,16)or scene.get('sgs',False):
            raise ValueError('Registered reference-phase closed-domain contract required')
        counts=tuple(scene['counts']);current=self._guard(fields,counts)
        if 'g'+'_'.join(map(str,reversed(counts)))not in self.pressure_core.model.log_gains:
            raise ValueError('No learned pressure table for this geometry')
        force=forcing(scene,step,horizons);dt=1./60.;h=1./scene['cells_per_metre'];rho=1000.;mu=.001
        results=[]
        for lead in range(horizons):
            advected=transport(current,counts,self.coefficients['point'],self.mid_gate,self.mix,dt/h)
            entering=[(value.double()+increment).float()for value,increment in zip(advected[:3],force[lead])]
            provisional=spatial(0,entering,counts,self.coefficients['viscosity'],mu/rho*dt/h**2,base=entering)
            q=spatial(1,provisional,counts,self.coefficients['flux'],rho*h/dt)
            q=q-q.mean();pressure=self.pressure_core(q.reshape(1,1,*q.shape))
            velocity=spatial(2,[pressure[0,0]],counts,self.coefficients['momentum'],dt/(rho*h),base=provisional)
            result=dict(pressure=pressure.contiguous().flatten(),vx=velocity[0].flatten(),vy=velocity[1].flatten(),
                        vz=velocity[2].flatten(),tracer=advected[3].flatten())
            results.append(result);current=[*velocity,advected[3]]
        return results

    def close(self):self.pressure_core.close()
