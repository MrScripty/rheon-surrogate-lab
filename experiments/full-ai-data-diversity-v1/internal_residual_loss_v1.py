"""Training-only portable frozen learned response and complete-field loss.

All pressure response/basis/spatial weights are exported restored model assets.
The independent equation action is a supervised loss only, never inference.
No eigenvalues, pressure solver, coefficient solve or optimizer belongs here.
"""
import math
import torch


def wall(value,component):
    axis=2-component;indices=torch.arange(value.shape[axis],device=value.device)
    shape=[1,1,1];shape[axis]=-1
    return torch.where(((indices>0)&(indices<value.shape[axis]-1)).reshape(shape),value,torch.zeros_like(value))


def equation_action(value,h,rho):
    # Exact ordering of numerics.fine_action, expressed for one3Dfield.
    result=torch.zeros_like(value)
    for axis in (0,1,2):
        left=[slice(None)]*3;right=left.copy();left[axis]=slice(None,-1);right[axis]=slice(1,None)
        difference=value[tuple(left)]-value[tuple(right)]
        result[tuple(left)]+=difference;result[tuple(right)]-=difference
    return result*(h/rho)


class FrozenCoupledTraining:
    def __init__(self,sparse,geometry,device='cuda'):
        if torch.device(device).type!='cuda':raise ValueError('Learned loss forwards CUDA-only')
        self.sparse={}
        for name in ('flux','momentum'):
            self.sparse[name]=(sparse[name+'_rows'].tolist(),sparse[name+'_channels'].tolist(),
                sparse[name+'_offsets'].tolist(),torch.as_tensor(sparse[name+'_weights'].copy(),device=device,dtype=torch.float64))
        self.axes=[torch.as_tensor(geometry['axis'+str(a)].copy(),device=device,dtype=torch.float64) for a in range(3)]
        self.divisor=torch.as_tensor(geometry['response'].copy(),device=device,dtype=torch.float64)
        self.shape=tuple(self.divisor.shape);nz,ny,nx=self.shape
        self.sizes=((nz,ny,nx+1),(nz,ny+1,nx),(nz+1,ny,nx));self.indices={}

    def linear(self,fields,shape,name,head,component=None):
        rows,channels,offsets,weights=self.sparse[name]
        value=torch.zeros(shape,device='cuda',dtype=torch.float64)
        for i in range(rows[head],rows[head+1]):
            c=channels[i]
            if component is not None and c!=component:continue
            field=fields[c];key=(tuple(field.shape),tuple(shape),tuple(offsets[i]))
            if key not in self.indices:
                xyz=[(torch.arange(shape[2-a],device='cuda')+offsets[i][a]).clamp(0,field.shape[-1-a]-1) for a in range(3)]
                self.indices[key]=(xyz[2][:,None,None],xyz[1][None,:,None],xyz[0][None,None,:])
            value=value+weights[i]*field[self.indices[key]].double()
        return value

    def pressure(self,q):
        value=q
        for dim,basis in enumerate(self.axes):value=(value.movedim(dim,-1)@basis.T).movedim(-1,dim)
        value=value/self.divisor
        mask=torch.ones_like(value);mask[0,0,0]=0.;value=value*mask
        for dim in (2,1,0):value=(value.movedim(dim,-1)@self.axes[dim]).movedim(-1,dim)
        return value-value[:1,:1,:1]

    def emit(self,sample,delta):
        h=float(sample['h']);dt=float(sample['dt']);rho=float(sample['rho'])
        velocity=[wall(v.double()*(h/dt),c) for c,v in enumerate(delta[:3])]
        parts=[self.linear(velocity,self.shape,'flux',0,c) for c in range(3)]
        rhs=torch.zeros_like(parts[0])
        for part in parts:rhs=rhs+(h*h)*part
        pressure_delta=self.pressure((rhs/dt)*(rho/h))
        pressure=sample['base_pressure']+pressure_delta
        pressure=pressure-pressure[:1,:1,:1]
        output=dict(pressure=pressure,tracer=sample['base_tracer']+delta[3].double())
        for c,(name,size) in enumerate(zip(('vx','vy','vz'),self.sizes)):
            momentum=self.linear([pressure_delta],size,'momentum',c)*(dt*(1./(rho*h)))
            output[name]=wall(sample['base_'+name]+velocity[c]+momentum,c)
        return output


def metrics(output,sample,normalization):
    result={}
    for name,names in [('pressure',('pressure',)),('velocity',('vx','vy','vz')),('tracer',('tracer',))]:
        actual=torch.cat([output[n].double().flatten() for n in names])
        target=torch.cat([sample['target_'+n].double().flatten() for n in names])
        if name=='pressure':actual=actual-actual.mean();target=target-target.mean()
        floor=normalization['relative_denominator_rms_floor'][name]
        denominator=target.norm().clamp_min(floor*math.sqrt(target.numel()))
        result[name]=((actual-target)/denominator).square().sum()
    target=sample['target_rhs'];rhs=equation_action(output['pressure'],float(sample['h']),float(sample['rho']))
    floor=normalization.get('optimization_equation_rms_floor',0.)*math.sqrt(target.numel())
    denominator=target.norm().clamp_min(max(normalization.get('equation_rhs_floor',1e-30),floor))
    result['equation']=((rhs-target)/denominator).square().sum()
    return result


def objective(output,sample,normalization):
    return sum(metrics(output,sample,normalization).values())
