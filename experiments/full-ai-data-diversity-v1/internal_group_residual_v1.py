"""Current-state-only nonlinear correction before restored learned projection.

No labels, target RHS, native equation action or numerical inverse belongs in
this module. The untrained network has exactly zero final heads. Learned
forwards require CUDA; CPU construction/typed restore does not run inference.
"""
import torch
from coupled_macrostep_v2 import CoupledMacroBackbone
from internal_residual_network_v1 import InternalGroupResidual, CHANNELS, HIDDEN, PARAMETERS
from recurrent_sampler_native import dimensions
from cuda_native_transfer_v1 import FIELDS



def face_pairs(value,component):
    axis=2-component;size=value.shape[axis]-1
    return (value.narrow(axis,0,size),value.narrow(axis,1,size))


def encode(phase):
    """42 fixed cell channels; all state observations precede correction."""
    counts=phase['counts'];shape=tuple(reversed(counts));sizes=dimensions(counts)
    width=phase['width'];h=phase['h'];dt=phase['dt'];scale=dt/h
    if width not in (2,3) or len(phase['forcing'])!=width:
        raise ValueError('Registered local group and its ordered known forces required')
    current=phase['current'];advected=phase['advected'];provisional=phase['provisional']
    velocity=[current[n].reshape(s) for n,s in zip(('vx','vy','vz'),sizes[:3])]
    values=[]
    for c,v in enumerate(velocity):values.extend(a.double()*scale for a in face_pairs(v,c))
    pressure=current['pressure'].reshape(shape).double();pressure=pressure-pressure.mean()
    values.extend((pressure*(dt*dt/(phase['rho']*h*h)),current['tracer'].reshape(shape).double()))
    for c,(a,v) in enumerate(zip(advected[:3],velocity)):
        values.extend(x*scale for x in face_pairs(a.double()-v.double(),c))
    for c,(p,a) in enumerate(zip(provisional,advected[:3])):
        values.extend(x*scale for x in face_pairs(p.double()-a.double(),c))
    values.append(advected[3].double()-current['tracer'].reshape(shape).double())
    for slot in range(3):
        for c in range(3):
            force=phase['forcing'][slot][c] if slot<width else torch.zeros(sizes[c],device='cuda',dtype=torch.float64)
            values.extend(x*scale for x in face_pairs(force,c))
    template=values[0]
    values.extend(torch.full_like(template,x) for x in (100.*h,480.*dt,float(width==3)))
    if len(values)!=CHANNELS or any(tuple(v.shape)!=shape for v in values):raise ValueError('Frozen current-only feature schema changed')
    return torch.stack(values).unsqueeze(0).float()




class InternalGroupGenerator(CoupledMacroBackbone):
    """Prototype runner; global learned physics stays frozen, no teacher input."""
    def phase(self,fields,scene,forcing,offset,width):
        if set(fields)!=set(FIELDS) or width not in (1,2,3):raise ValueError('Complete current physical state/local width required')
        counts=tuple(scene['counts']);shape=tuple(reversed(counts));core=self.model
        h=1./scene['cells_per_metre'];dt=width/480.;rho=1000.
        core.prepare_geometry(counts);core.configure_transport(h,dt)
        current=[fields[n].reshape(s) for n,s in zip(FIELDS[1:],dimensions(counts))]
        addition=forcing[offset]
        for j in range(1,width):addition=[a+b for a,b in zip(addition,forcing[offset+j])]
        advected=core.transport(current,counts,dt/h)
        entering=core.affine_ops.three(advected[:3],addition,counts)
        provisional=core.gather_ops.three(core,entering,entering,counts,'viscosity',(.001/rho*dt/h**2))
        return dict(counts=counts,shape=shape,h=h,rho=rho,dt=dt,width=width,current=fields,
            advected=advected,provisional=provisional,tracer=advected[3],previous=fields['pressure'].reshape(1,1,*shape),
            forcing=forcing[offset:offset+width])

    def corrected(self,phase,network,float_carriers=True):
        values=network(encode(phase))
        velocity=[v.double()*(phase['h']/phase['dt']) for v in values[:3]]
        return self.emit(phase,velocity,values[3].double(),float_carriers=float_carriers)

    @torch.no_grad()
    def predict(self,fields,scene,start,horizons,network,reference_substeps):
        if horizons not in (1,2,3,4) or reference_substeps not in (1,8):raise ValueError('Known physical output count/reference regime required')
        self.model.prepare_geometry(tuple(scene['counts']))
        external=self.controls.generate_from_packet(tuple(scene['counts']),self.packet(scene,start,horizons,reference_substeps).to('cuda'))
        if reference_substeps==1:
            return self.model.predict(fields,scene,start,horizons,external,substeps=1)
        outputs=[]
        for lead in range(horizons):
            offset=lead*8
            for width in (2,2,3,1):
                phase=self.phase(fields,scene,external,offset,width)
                fields=self.corrected(phase,network) if width>1 else self.emit(phase)
                offset+=width
            outputs.append(fields)
        return outputs
