"""Same neural outputs with lean CPU input/feedback and prescribed-force execution."""
import torch
from torch.nn import functional as F
from bundle_model_displacement import BundleNet
from temporal_features_v2 import geometry
from fast_neural_state_codec import next_input,packed_state
from temporal_features_v2 import unpack_outputs
from fused_force_construction import auxiliary
from fused_field_readout import convex_field_readout


class FastPreparationBundleNet(BundleNet):
    def forward(self,state,features):
        if state.ndim!=5 or state.shape[:2]!=(1,5)or features.shape[1]!=7+3*self.horizons:
            raise ValueError('One causal full state and prescribed-force block required')
        counts=tuple(int(n)-1 for n in reversed(state.shape[2:]));nx,ny,nz=counts
        force=features[:,7:].reshape(1,self.horizons,3,*state.shape[2:])
        h=features[:,6,0,0,0].double().exp()/16
        dt=torch.full_like(h,1/60);rho=torch.full_like(h,1000.);mu=torch.full_like(h,.001)
        alpha=(dt/h*self.scales['velocity']).float().reshape(-1,1,1,1,1)
        beta=(mu/rho*dt/h.square()).float().reshape(-1,1,1,1,1)
        _,mask=geometry(counts,str(state.device))
        hidden=self.fluid(torch.cat((state[:,1:4],features[:,:7]),dim=1))
        current=state;outputs=[]
        for step in range(self.horizons):
            velocity=current[:,1:4]
            hidden=hidden+self.local_update(torch.cat((hidden,velocity),dim=1))
            shift=displacement(self.displacement,velocity,hidden,alpha)
            advected=convex_field_readout(current[:,1:5],shift)
            entering=advected[:,:3]+force[:,step]
            provisional=(entering+beta*self.viscous_head(entering))*mask[1:4]
            provisional_physical=(provisional*self.scales['velocity']).float()
            q=self.flux(provisional_physical)*(rho*h/dt).reshape(-1,1,1,1,1)
            q=q-q.mean(dim=(2,3,4),keepdim=True)
            physical_pressure=self.pressure_core(q)
            physical_velocity=(provisional_physical.double()+(dt/(rho*h)).reshape(-1,1,1,1,1)*self.momentum(physical_pressure,self.flux.weight)).float()
            pressure=F.pad((physical_pressure/self.scales['pressure']).float(),(0,1,0,1,0,1),mode='replicate')
            velocity=physical_velocity/self.scales['velocity']
            output=torch.cat((pressure,velocity*mask[1:4],advected[:,3:4]),dim=1)
            outputs.append(output)
            if step+1<self.horizons:current=next_input(output[:,None],counts,self.scales)
        return torch.stack(outputs,dim=1)


    def predict(self,fields,scene,step,normalization):
        device=next(self.parameters()).device;scales=normalization['scales']
        state=packed_state(fields,tuple(scene['counts']),scales,device)
        features=auxiliary(scene,step,self.horizons,scales,device)
        return unpack_outputs(self(state,features),scene['counts'],scales)


def displacement(head,velocity,hidden,alpha):
    # Same learned coefficients and operation order; one CPU replicate-pad call.
    projected=F.conv3d(F.pad(velocity,(1,1,1,1,1,1),mode='replicate'),torch.cat((head.point_weight,head.local_weight),dim=0))
    point=projected[:,:12];left=projected[:,12:12+head.width];right=projected[:,12+head.width:]
    polynomial=torch.cat((left*right,left.abs()*right,alpha*left*right.square()),dim=1)
    correction=head.curvature(polynomial*(1+head.context(hidden)))
    shift=alpha*point+alpha.square()*correction
    return shift.clamp(-1,1).reshape(1,4,3,*velocity.shape[2:])
