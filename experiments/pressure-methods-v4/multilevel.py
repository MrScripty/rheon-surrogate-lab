"""Size-adaptive variational hierarchy for the unchanged sealed-box operator.

The transfer operators preserve constants and restriction is their transpose.
Every coarse operator is the exact tensor-product Galerkin projection, including
mass matrices. No rediscretized coarse equation or pressure labels are substituted.
"""
from functools import lru_cache
import math
import torch

def interpolate(n, count):
    centers=(torch.arange(count,dtype=torch.float64)+.5)*n/count-.5
    if count==1:return torch.ones((n,1),dtype=torch.float64)
    cells=torch.arange(n,dtype=torch.float64).clamp(float(centers[0]),float(centers[-1]))
    right=torch.searchsorted(centers,cells,right=True).clamp(1,count-1);left=right-1
    weight=(cells-centers[left])/(centers[right]-centers[left])
    p=torch.zeros((n,count),dtype=torch.float64)
    p[torch.arange(n),left]=1-weight;p[torch.arange(n),right]+=weight
    return p

class Transfer:
    def __init__(self,p):
        self.p=p
        # At most two nonzero interpolation coefficients per fine cell.
        weights,indices=torch.topk(p,k=min(2,p.shape[1]),dim=1)
        self.weights=weights;self.indices=indices
    def prolong(self,value,axis):
        moved=value.movedim(axis,-1);out=torch.zeros((*moved.shape[:-1],self.p.shape[0]),dtype=value.dtype)
        for col in range(self.indices.shape[1]):out+=moved.index_select(-1,self.indices[:,col])*self.weights[:,col]
        return out.movedim(-1,axis)
    def restrict(self,value,axis):
        moved=value.movedim(axis,-1);out=torch.zeros((*moved.shape[:-1],self.p.shape[1]),dtype=value.dtype)
        for col in range(self.indices.shape[1]):out.index_add_(-1,self.indices[:,col],moved*self.weights[:,col])
        return out.movedim(-1,axis)

class Band:
    def __init__(self,matrix):
        self.matrix=matrix;self.n=matrix.shape[0]
        self.identity=bool(torch.equal(matrix,torch.eye(self.n,dtype=torch.float64)))
        self.bands=[(offset,torch.diagonal(matrix,offset).clone()) for offset in range(1-self.n,self.n)
                    if bool(torch.diagonal(matrix,offset).ne(0).any())]
    def apply(self,value,axis):
        if self.identity:return value
        moved=value.movedim(axis,-1);out=torch.zeros_like(moved)
        for offset,values in self.bands:
            if offset>=0:out[...,:self.n-offset]+=moved[...,offset:]*values
            else:out[...,-offset:]+=moved[...,:self.n+offset]*values
        return out.movedim(-1,axis)

class Level:
    def __init__(self,mass,stiffness):
        self.mass=[Band(m)for m in mass];self.stiffness=[Band(k)for k in stiffness]
        self.shape=tuple(m.shape[0]for m in mass);self.transfers=None;self.child=None;self.factor=None
        diagonal=torch.zeros(self.shape,dtype=torch.float64);bound=torch.zeros_like(diagonal)
        for direction in range(3):
            d=torch.ones(self.shape,dtype=torch.float64);b=d.clone()
            for axis in range(3):
                matrix=(stiffness if axis==direction else mass)[axis]
                view=[1]*3;view[axis]=matrix.shape[0]
                d*=matrix.diagonal().reshape(view);b*=matrix.abs().sum(1).reshape(view)
            diagonal+=d;bound+=b
        self.diagonal=diagonal[None,None].clamp_min(1e-300)
        self.omega=1.6/max(float((bound/self.diagonal[0,0]).max()),1.)
        if max(self.shape)<=4:
            mz,my,mx=mass;kz,ky,kx=stiffness
            matrix=torch.kron(torch.kron(kz,my),mx)+torch.kron(torch.kron(mz,ky),mx)+torch.kron(torch.kron(mz,my),kx)
            if matrix.shape[0]>1:self.factor=torch.linalg.cholesky(matrix[1:,1:])
        else:
            ps=[interpolate(n,max(1,(n+1)//2))for n in self.shape]
            self.transfers=[Transfer(p)for p in ps]
            self.child=Level([p.T@m@p for p,m in zip(ps,mass)],
                             [p.T@k@p for p,k in zip(ps,stiffness)])
    def action(self,value):
        result=torch.zeros_like(value)
        for direction in range(3):
            term=value
            for axis in range(3):term=(self.stiffness if axis==direction else self.mass)[axis].apply(term,axis+2)
            result+=term
        return result
    def restrict(self,value):
        for axis,transfer in enumerate(self.transfers):value=transfer.restrict(value,axis+2)
        return value
    def prolong(self,value):
        for axis in reversed(range(3)):value=self.transfers[axis].prolong(value,axis+2)
        return value
    def cycle(self,rhs):
        if self.child is None:
            flat=rhs.flatten(2);out=torch.zeros_like(flat)
            if self.factor is not None:out[:,:,1:]=torch.cholesky_solve(flat[0,:,1:].T,self.factor).T[None]
            return out.reshape(rhs.shape)
        out=torch.zeros_like(rhs)
        for _ in range(3):out=out+self.omega*(rhs-self.action(out))/self.diagonal
        out=out+self.prolong(self.child.cycle(self.restrict(rhs-self.action(out))))
        for _ in range(3):out=out+self.omega*(rhs-self.action(out))/self.diagonal
        return out
    def dimensions(self):return [self.shape]+(self.child.dimensions()if self.child is not None else[])

@lru_cache(maxsize=16)
def geometry(shape):
    if len(shape)!=3 or min(shape)<1 or max(shape)>256 or math.prod(shape)>256**3:
        raise ValueError('Bounded 3D hierarchy through 256 cubed required')
    masses=[];stiffness=[]
    for n in shape:
        masses.append(torch.eye(n,dtype=torch.float64))
        k=torch.zeros((n,n),dtype=torch.float64)
        for i in range(n-1):k[i,i]+=1;k[i+1,i+1]+=1;k[i,i+1]-=1;k[i+1,i]-=1
        stiffness.append(k)
    return Level(masses,stiffness)

def validate(residual):
    if residual.device.type!='cpu' or residual.dtype!=torch.float64 or residual.shape[:2]!=(1,1) or residual.ndim!=5:
        raise ValueError('CPU float64 1,1,Z,Y,X field required')
    if not bool(torch.isfinite(residual).all()):raise ValueError('Nonfinite residual')
    norm=float(torch.linalg.vector_norm(residual));dc=abs(float(residual.mean()))*math.sqrt(residual.numel())
    if dc>max(1e-12,norm*1e-8):raise ValueError('Incompatible sealed equation; no mean subtraction')

def correction(residual,h,rho):
    validate(residual)
    for value in [h,rho]:
        if value.shape!=(1,)or value.dtype!=torch.float64 or value.device.type!='cpu' or not bool(torch.isfinite(value).all()and(value>0).all()):
            raise ValueError('Positive finite spacing/density required')
    out=geometry(tuple(residual.shape[2:])).cycle(residual*(rho/h).reshape(1,1,1,1,1))
    return out-out[:,:,:1,:1,:1].clone()

def solve(wrapper,sample,normalization,*,max_passes=8,neural=True):
    import numerics,scalable
    base=sample['base'];pressure=base.clone();h=sample['cell_size_m'];rho=sample['rho']
    denominator=max(float(torch.linalg.vector_norm(sample['rhs'])),1e-300)
    limit=max(1e-12,denominator*1e-6);calls=0;cycles=0;trace=[]
    def residual():return sample['native_input_residual']-numerics.fine_action(pressure-base,h,rho)
    def accepted(r):return float(torch.linalg.vector_norm(r))<=limit and float(sample['dt'][0]*r.abs().max()/h[0]**3)<=1e-7
    for _ in range(max_passes):
        r=residual()
        if accepted(r):break
        pressure=pressure+correction(r,h,rho);cycles+=1
        trace.append(dict(stage='vcycle',relative_residual=float(torch.linalg.vector_norm(residual()))/denominator))
        if accepted(residual()):break
        if neural:
            pressure=scalable.fine_step(wrapper,sample,normalization,pressure);calls+=1
            trace.append(dict(stage='neural',relative_residual=float(torch.linalg.vector_norm(residual()))/denominator))
    r=residual()
    return pressure,dict(network_calls=calls,coarse_calls=cycles,qualified=accepted(r),
        relative_residual=float(torch.linalg.vector_norm(r))/denominator,
        predicted_divergence=float(sample['dt'][0]*r.abs().max()/h[0]**3),trace=trace)
