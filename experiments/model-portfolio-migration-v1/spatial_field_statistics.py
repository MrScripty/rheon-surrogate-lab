"""Scalar diagnostics on staggered fields;no model,field saving or fitting."""
import math


def measure(output,truth,counts,spacing):
    import numpy as np
    nx,ny,nz=counts;hx,hy,hz=spacing
    if any(v<=0 or not math.isfinite(v) for v in spacing):raise ValueError('Positive finite XYZspacing required')
    shapes=dict(pressure=(nz,ny,nx),tracer=(nz,ny,nx),vx=(nz,ny,nx+1),vy=(nz,ny+1,nx),vz=(nz+1,ny,nx))
    predicted={n:np.asarray(output[n],dtype=np.float64).reshape(shape) for n,shape in shapes.items()}
    reference={n:np.asarray(truth[n],dtype=np.float64).reshape(shape) for n,shape in shapes.items()}
    if any(not np.isfinite(v).all() for v in [*predicted.values(),*reference.values()]):raise ValueError('Finite physical fields required')
    rms=lambda v:math.sqrt(float(np.square(v).mean()))
    result={}
    for name in shapes:
        p,r=predicted[name],reference[name]
        if name=='pressure':p=p-p.mean();r=r-r.mean()
        error=p-r
        result[name+'_absolute_rms_error']=rms(error)
        result[name+'_maximum_cell_absolute_error']=float(np.abs(error).max())
        result[name+'_reference_rms']=rms(r)
    def divergence(fields):
        return np.diff(fields['vx'],axis=2)/hx+np.diff(fields['vy'],axis=1)/hy+np.diff(fields['vz'],axis=0)/hz
    pd,rd=divergence(predicted),divergence(reference)
    result.update(divergence_rms_per_s=rms(pd),divergence_maximum_absolute_per_s=float(np.abs(pd).max()),
        reference_divergence_rms_per_s=rms(rd),divergence_error_rms_per_s=rms(pd-rd),
        divergence_error_maximum_absolute_per_s=float(np.abs(pd-rd).max()))
    normal=[];normal_reference=[]
    for name,axis in [('vx',2),('vy',1),('vz',0)]:
        normal.extend(np.take(predicted[name],side,axis=axis).ravel() for side in (0,-1))
        normal_reference.extend(np.take(reference[name],side,axis=axis).ravel() for side in (0,-1))
    normal=np.concatenate(normal);normal_reference=np.concatenate(normal_reference)
    result.update(boundary_normal_velocity_maximum_absolute=float(np.abs(normal).max()),
        boundary_normal_velocity_rms=rms(normal),reference_boundary_normal_velocity_maximum_absolute=float(np.abs(normal_reference).max()))
    tracer,ref=predicted['tracer'],reference['tracer'];volume=hx*hy*hz
    result.update(tracer_volume_integral=float(tracer.sum())*volume,
        reference_tracer_volume_integral=float(ref.sum())*volume,
        tracer_volume_integral_error=float((tracer-ref).sum())*volume,
        tracer_minimum=float(tracer.min()),tracer_maximum=float(tracer.max()),
        tracer_negative_cells=int(np.count_nonzero(tracer<0)),reference_tracer_negative_cells=int(np.count_nonzero(ref<0)),
        tracer_cells_below_reference_minimum=int(np.count_nonzero(tracer<ref.min())),
        tracer_cells_above_reference_maximum=int(np.count_nonzero(tracer>ref.max())))
    return result
