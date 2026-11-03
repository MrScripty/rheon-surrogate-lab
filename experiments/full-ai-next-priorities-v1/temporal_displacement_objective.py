"""Training-only native-qualified displacement supervision on causal inputs."""
import torch
from temporal_objective_increment import loss as physical_loss
from training_displacement_teacher import displacement_labels


def point_mask(counts,device):
    nx,ny,nz=counts
    mask=torch.zeros(1,4,1,nz+1,ny+1,nx+1,dtype=torch.float64,device=device)
    for kind,shape in enumerate(((nz,ny,nx+1),(nz,ny+1,nx),(nz+1,ny,nx),(nz,ny,nx))):
        mask[:,kind,:,:shape[0],:shape[1],:shape[2]]=1
    return mask


class DisplacementObjective:
    def __init__(self,network,config):
        self.network=network;self.mode=config['supervision_mode']
        if self.mode!='increments' or config['displacement_auxiliary_weight']!=1.:
            raise ValueError('Registered physical plus displacement objective required')
        self.weight=config['displacement_auxiliary_weight'];self.pending=[];self.last=None
        self.hook=network.displacement.register_forward_hook(self.capture)

    def capture(self,module,arguments,output):
        velocity,hidden,alpha=arguments
        self.pending.append((velocity,alpha,output))

    def __call__(self,predicted,target,counts,tracer_scale):
        if len(self.pending)!=self.network.horizons:
            raise ValueError('Exactly one causal displacement per predicted horizon required')
        base=physical_loss(predicted,target,counts,tracer_scale,
            increment_scales=self.network.increment_scales,mode=self.mode)
        terms=[]
        for velocity,alpha,shift in self.pending:
            # Labels correspond to this actual entering state, including own-history
            # errors. They never enter the model and have no autograd graph.
            physical=(velocity.detach()*self.network.scales['velocity']).float()
            labels,_=displacement_labels(physical,counts,alpha.detach().double()/self.network.scales['velocity'])
            mask=point_mask(counts,str(velocity.device))
            error=(shift.double()-labels)/alpha.detach().double().square().unsqueeze(1)
            terms.append((error.square()*mask).sum()/(3*mask.sum()))
        auxiliary=torch.stack(terms).mean()
        self.pending.clear();self.last=dict(physical=float(base.detach()),displacement=float(auxiliary.detach()))
        return base+self.weight*auxiliary

    def close(self):
        self.hook.remove();self.pending.clear()
