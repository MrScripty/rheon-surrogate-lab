"""Training-only provisional supervision; deployed inference remains unchanged."""
import torch
from tied_momentum_readout import index_readout
from temporal_objective_increment import loss as full_state_loss


def provisional_labels(target,features,counts,scales,flux_weight):
    nx,ny,nz=counts;batch,horizons=target.shape[:2]
    if batch!=1:raise ValueError('Registered batch-one labels required')
    # Reconstruct physical values with the same float32 interchange as deployed
    # output. These are targets only and never enter the predictor.
    p=(target[:,:,0,:nz,:ny,:nx]*scales['pressure']).double().reshape(horizons,1,nz,ny,nx)
    v=(target[:,:,1:4]*scales['velocity']).double().reshape(horizons,3,nz+1,ny+1,nx+1)
    h=features[:,6,0,0,0].double().exp()/16
    contribution=index_readout(p,flux_weight)*(1/(60*1000*h)).reshape(1,1,1,1,1)
    return v-contribution


class ProvisionalObjective:
    def __init__(self,config,network):
        self.network=network;self.weight=config['auxiliary_provisional_weight'];self.mode=config['supervision_mode']
        if self.weight not in (0.,1.)or self.mode!='increments':raise ValueError('Registered matched objective arms required')
        self.values=[];self.features=None;self.labels_consumed=0;self.last_terms={}
        self.handles=[network.register_forward_pre_hook(self.begin),network.flux.register_forward_pre_hook(self.capture)]

    def begin(self,module,args):self.values=[];self.features=args[1]
    def capture(self,module,args):self.values.append(args[0])

    def __call__(self,predicted,target,counts,tracer_scale):
        k=predicted.shape[1];nx,ny,nz=counts
        if len(self.values)!=k or self.features is None:raise ValueError('Actual provisional readouts must accompany every horizon')
        actual=torch.cat(self.values,0).double()
        with torch.no_grad():truth=provisional_labels(target,self.features,counts,self.network.scales,self.network.flux.weight)
        difference=(actual-truth)/self.network.scales['velocity'];parts=[]
        for component,slices in enumerate(((slice(0,nz),slice(0,ny),slice(0,nx+1)),
                (slice(0,nz),slice(0,ny+1),slice(0,nx)),(slice(0,nz+1),slice(0,ny),slice(0,nx)))):
            parts.append(difference[:,component][(slice(None),*slices)].flatten(1))
        rms=torch.cat(parts,1).square().mean(1)
        scale=actual.new_tensor([self.network.increment_scales[str(h)]['velocity']**2 for h in range(1,k+1)])
        auxiliary=(rms/scale).mean()
        base=full_state_loss(predicted,target,counts,tracer_scale,increment_scales=self.network.increment_scales,mode=self.mode)
        self.labels_consumed+=k;self.values=[];self.features=None
        self.last_terms=dict(full_state=base.detach(),provisional=auxiliary.detach())
        return base+self.weight*auxiliary

    def close(self):
        for handle in self.handles:handle.remove()
        self.handles=[];self.values=[];self.features=None
