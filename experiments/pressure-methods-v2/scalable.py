"""Explicit offline larger-grid contract using unchanged restored fine tensors.

This is a new inference wrapper, not a claim that the original adapter's grid
ceiling changed. Original captured sources, globals and restored tensors stay intact.
"""
import math
import time
import torch
from torch.nn import functional as F
from numerics import fine_action
import numerical

class ScalableFine(torch.nn.Module):
    def __init__(self,network):
        super().__init__();self.fine=network
    def forward(self,values):
        if (values.dtype!=torch.float32 or values.device.type!='cpu' or values.ndim!=5
                or values.shape[:2]!=(1,1) or max(values.shape[2:])>256
                or values.numel()>256**3 or not bool(torch.isfinite(values).all())):
            raise ValueError('Declared finite CPU 1,1,Z,Y,X field through 256^3 required')
        network=self.fine;skips=[]
        for encoder in network.encoders:
            values=encoder(values);skips.append(values)
            kernel=tuple(min(2,n) for n in values.shape[2:])
            values=F.avg_pool3d(values,kernel,stride=kernel,ceil_mode=True,count_include_pad=False)
        values=network.bottleneck(values)
        values=values+network.global_pool(values).expand_as(values)
        for decoder,skip in zip(network.decoders,reversed(skips)):
            values=F.interpolate(values,size=skip.shape[2:],mode='nearest')
            values=decoder(torch.cat((values,skip),dim=1))
        return network.head(values)

def fine_step(wrapper,sample,normalization,current):
    h=sample['cell_size_m'];rho=sample['rho'];scales=normalization['scales']
    residual=sample['native_input_residual']-fine_action(current-sample['base'],h,rho)
    values=(residual*(rho/h).reshape(1,1,1,1,1)/scales['laplacian_rhs']['value']).to(torch.float32)
    raw=wrapper(values)
    delta=raw.to(torch.float64)*scales[normalization.get('output_scale','pressure')]['value']
    delta=delta-delta[:,:,:1,:1,:1].clone()
    return current+delta

def predict(wrapper,sample,normalization,passes=8,kind='fine',nodes=6):
    pressure=sample['base'];trace=[]
    h=sample['cell_size_m'];rho=sample['rho'];denominator=max(float(torch.linalg.vector_norm(sample['rhs'])),1e-300)
    for stage in range(passes):
        tick=time.perf_counter()
        r=sample['native_input_residual']-fine_action(pressure-sample['base'],h,rho)
        if kind=='numerical-coarse':pressure=pressure+numerical.correction(r,h,rho,nodes)
        elif kind=='learned-coarse':
            import learned_physical,coarse
            network=wrapper.fine;shape=tuple(pressure.shape[2:]);axes=learned_physical.low_basis(shape,'cpu')
            logits=network.coarse_logits
            if hasattr(network,'shape_head'):
                logits=logits+network.shape_head(torch.tensor([math.log(n/16) for n in shape],dtype=torch.float32))
            gain=max(shape)**2/2*torch.sigmoid(logits.to(torch.float64))
            gain=torch.cat((gain.new_zeros(1),gain)).reshape(4,4,4)
            proposal=coarse.prolong(coarse.restrict(r*(rho/h).reshape(1,1,1,1,1),axes)*gain,axes)
            proposal=proposal-proposal[:,:,:1,:1,:1].clone()
            remaining=r-fine_action(proposal,h,rho)
            if float(torch.linalg.vector_norm(remaining))<=float(torch.linalg.vector_norm(r)):pressure=pressure+proposal
        elif kind!='fine':raise ValueError('Declared scalable policy required')
        pressure=fine_step(wrapper,sample,normalization,pressure)
        relative=float(torch.linalg.vector_norm(sample['rhs']-fine_action(pressure,h,rho)))/denominator
        trace.append(dict(stage=stage+1,relative_residual=relative,seconds=time.perf_counter()-tick))
    return pressure,trace

def qualify(state):
    wrapper=ScalableFine(state.network).eval();generator=torch.Generator().manual_seed(2026101823)
    rows=[]
    with torch.no_grad():
        for shape in [(16,16,16),(13,9,7)]:
            values=torch.randn((1,1,*shape),generator=generator,dtype=torch.float32)
            original=state.network(values);extended=wrapper(values)
            if not torch.equal(original,extended):raise ValueError('Larger-grid wrapper changed retained fine-network arithmetic')
            rows.append(dict(shape_zyx=shape,bitwise_forward_parity=True))
    return wrapper,rows

def qualify_recurrence(state,wrapper,kind,nodes=6,passes=8):
    """Check the complete coarse/fine method, not only the retained CNN forward."""
    import inference
    rows=[];generator=torch.Generator().manual_seed(2026101827)
    with torch.no_grad():
        for shape in [(16,16,16),(13,9,7)]:
            reference=torch.randn((1,1,*shape),generator=generator,dtype=torch.float64)
            reference=reference-reference[:,:,:1,:1,:1].clone()
            base=reference*.3
            h=torch.tensor([1/16],dtype=torch.float64);rho=torch.tensor([1000.],dtype=torch.float64)
            rhs=fine_action(reference,h,rho)
            sample=dict(base=base,rhs=rhs,native_input_residual=rhs-fine_action(base,h,rho),
                reference=reference,cell_size_m=h,rho=rho,dt=torch.tensor([1/60],dtype=torch.float64))
            metadata=state.model.ARCHITECTURE.get('numerical_training_method')
            if kind=='numerical-coarse' and (not metadata or metadata['coarse_nodes_per_axis']!=nodes):
                import policies
                from pathlib import Path
                state.project=Path(__file__).resolve().parents[2]/state.entry['project']
                original,_=policies.run(state,sample,nodes=nodes,placement='each',max_passes=passes,adaptive=False)
                reference_route='Explicit fixed-weight numerical policy'
            else:
                original=inference.fixed(state,sample,passes)
                reference_route='Exact captured model recurrence'
            extended,_=predict(wrapper,sample,state.normalization,passes=passes,kind=kind,nodes=nodes)
            difference=float(torch.linalg.vector_norm(original-extended))/max(float(torch.linalg.vector_norm(original)),1e-300)
            if difference>1e-8:raise ValueError('Extended coarse/fine recurrence differs from its declared reference')
            rows.append(dict(shape_zyx=shape,passes=passes,kind=kind,nodes=nodes,
                pressure_relative_difference=difference,reference_route=reference_route))
    return rows
