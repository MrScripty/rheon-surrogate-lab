"""Train-only full-grid affine residual loss compression; no coefficient fit.

The reference equation operator below is used solely to construct supervised
loss labels. Deployment uses macrostep_readout_v1 and never calls this module.
"""
import math
import torch
from numerics import fine_action


def compress_metric(values,reference,denominator,stages,active,relative_tolerance,absolute_tolerance):
    anchor=values[-1]
    target=reference-anchor
    if active:
        differences=torch.stack([v-anchor for v in values[:-1]],dim=1)
        matrix=torch.cat((differences,target[:,None]),dim=1)/denominator
        _,factor=torch.linalg.qr(matrix,mode='r')
    else:
        # With a frozen N1 branch the design columns are exactly zero, so only
        # the norm of the fixed residual contributes to the supervised loss.
        differences=None
        factor=torch.zeros((stages+1,stages+1),dtype=torch.float64,device='cuda')
        factor[-1,-1]=(target/denominator).norm()
    controls=[torch.zeros(stages,dtype=torch.float64,device='cuda'),
              torch.linspace(-.25,.25,stages,dtype=torch.float64,device='cuda'),
              torch.full((stages,),.1,dtype=torch.float64,device='cuda')]
    errors=[]
    for z in controls:
        direct=(-target if differences is None else differences@z-target)/denominator
        compressed=factor@torch.cat((z,z.new_tensor([-1.])))
        lhs,rhs=direct.square().sum(),compressed.square().sum()
        error=float((lhs-rhs).abs())
        if not bool(lhs.isfinite()&rhs.isfinite()) or error>relative_tolerance*max(float(lhs),1e-30)+absolute_tolerance:
            raise ValueError('Full-grid supervised norm did not survive generic QR compression')
        errors.append(error)
    if factor.shape!=(stages+1,stages+1) or not bool(factor.isfinite().all()):
        raise ValueError('Finite complete QR residual factor required')
    return factor,dict(zero_readout_relative_error=float((target/denominator).norm()),
                       control_vectors=3,maximum_absolute_compression_error=max(errors))


@torch.no_grad()
def design(features,truth,scene,norm,stages,active,relative_tolerance,absolute_tolerance):
    if any(v.device.type!='cuda' for f in features for v in f.values()):
        raise ValueError('All learned feature fields must be generated on GPU')
    if (active and len(features)!=stages+1) or (not active and len(features)!=2):
        raise ValueError('Declared frozen or active stage features required')
    refs={k:torch.from_numpy(v.copy()).to(device='cuda',dtype=torch.float64).flatten() for k,v in truth.items()}
    shape=(1,1,*reversed(scene['counts']))
    h=torch.tensor([1./scene['cells_per_metre']],dtype=torch.float64,device='cuda')
    rho=torch.tensor([1000.],dtype=torch.float64,device='cuda')
    pressures=[f['pressure'].double().flatten() for f in features]
    actions=[fine_action(p.reshape(shape),h,rho).flatten() for p in pressures]
    metrics=[]
    factor,stats=compress_metric(actions,refs['rhs'],refs['rhs'].norm().clamp_min(1e-30),stages,active,relative_tolerance,absolute_tolerance)
    metrics.append(('equation',factor,stats))
    centered=[p-p.mean() for p in pressures]
    target=refs['pressure']-refs['pressure'].mean()
    denominator=target.norm().clamp_min(norm['relative_denominator_rms_floor']['pressure']*math.sqrt(target.numel()))
    factor,stats=compress_metric(centered,target,denominator,stages,active,relative_tolerance,absolute_tolerance)
    metrics.append(('pressure',factor,stats))
    for name,fields in [('velocity',('vx','vy','vz')),('tracer',('tracer',))]:
        values=[torch.cat([f[n].double().flatten() for n in fields]) for f in features]
        target=torch.cat([refs[n] for n in fields])
        denominator=target.norm().clamp_min(norm['relative_denominator_rms_floor'][name]*math.sqrt(target.numel()))
        factor,stats=compress_metric(values,target,denominator,stages,active,relative_tolerance,absolute_tolerance)
        metrics.append((name,factor,stats))
    return torch.stack([r for _,r,_ in metrics]),{name:stats for name,_,stats in metrics}
