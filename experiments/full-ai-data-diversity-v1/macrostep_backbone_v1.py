"""Frozen managed learned stages at a declared uniform internal stage count.

Only current fields and known forcing enter. This module has no optimizer,
native solver, pressure coefficients, teacher fields or target RHS input.
"""
import torch
from debug.gather_fused_graph_v1 import GatherFusedGraph
from cuda_divisor_generator_v1 import FIELDS
from recurrent_sampler_native import dimensions


class FrozenMacroBackbone:
    def __init__(self, pair, shared_pair, build, control_build):
        self.backend=GatherFusedGraph(pair,shared_pair,build,control_build)
        self.model=self.backend.model
        self.controls=self.backend.controls
        self.asset_identity=dict(self.backend.asset_identity,
            frozen_macro_backbone=True,known_uniform_stage_counts=[1,3,4,8],
            learned_parameter_updates=0,cpu_model_forward=False)

    def state_dict(self):
        return self.backend.state_dict()

    def packet(self,scene,start,horizons,stages):
        if stages not in (1,3,4,8) or horizons not in (1,2,3,4):
            raise ValueError('Registered stage/output counts required')
        base=self.controls.parameters(scene,start,horizons,1)
        packet=base.repeat_interleave(stages,dim=0).clone()
        dt=1./(60.*stages)
        for lead in range(horizons):
            for fine in range(stages):
                packet[lead*stages+fine,9]=float((start+lead)*stages+fine)*dt
                packet[lead*stages+fine,17]=dt
        return packet

    @torch.no_grad()
    def feature_states(self,fields,scene,start,stages,horizons=1):
        if set(fields)!=set(FIELDS) or any(v.device.type!='cuda' for v in fields.values()):
            raise ValueError('Complete current CUDA fields required')
        counts=tuple(scene['counts']);shape=tuple(reversed(counts));model=self.model
        model.prepare_geometry(counts)
        h,dt,rho=1./scene['cells_per_metre'],1./(60.*stages),1000.
        model.configure_transport(h,dt)
        packet=self.packet(scene,start,horizons,stages).to('cuda')
        external=self.controls.generate_from_packet(counts,packet)
        current=[fields[n].reshape(s) for n,s in zip(FIELDS[1:],dimensions(counts))]
        previous=fields['pressure'].reshape(1,1,*shape)
        blocks=[]
        for lead in range(horizons):
            features=[dict(pressure=previous.flatten(),**{n:v.flatten() for n,v in zip(FIELDS[1:],current)})]
            for fine in range(stages):
                advected=model.transport(current,counts,dt/h)
                entering=model.affine_ops.three(advected[:3],external[lead*stages+fine],counts)
                provisional=model.gather_ops.three(model,entering,entering,counts,'viscosity',(.001/rho*dt/h**2))
                rhs=model.gather_ops.ordered_rhs(model,provisional,shape,'flux',0,h*h,dt)
                scale=(h*h)*(1/(rho*h))
                history=model.gather_ops.three(model,[previous[0,0]],None,counts,'momentum',scale,multiply_only=True,as_double=True)
                request=model.gather_ops.difference(model,history,rhs,shape,'flux',0,-1.,rho/h)
                correction=model.pressure(request.reshape(1,1,*shape),counts)
                pressure=model.affine_ops.binary('sum_gauge',previous,correction)
                velocity=model.gather_ops.three(model,[pressure[0,0]],provisional,counts,'momentum',dt*(1/(rho*h)))
                emitted=dict(pressure=pressure.contiguous().flatten(),vx=velocity[0].flatten(),vy=velocity[1].flatten(),
                             vz=velocity[2].flatten(),tracer=advected[3].flatten())
                features.append(emitted)
                current,previous=[*velocity,advected[3]],pressure
            blocks.append(features)
        return blocks

    def close(self):
        self.backend.close()
