"""Offline full-field errors and physical diagnostics on matching native timestamps."""
import math
import numpy as np
import torch
import numerics

FIELDS=('pressure','vx','vy','vz','tracer')


def physical_arrays(values):
    return {k:np.asarray(v.detach().cpu().numpy()if isinstance(v,torch.Tensor)else v).reshape(-1)for k,v in values.items()}


def metrics(predicted,reference,rhs,scene,normalization,initial_mass):
    predicted=physical_arrays(predicted)
    if set(predicted)!=set(FIELDS)or not all(np.isfinite(v).all()for v in predicted.values()):
        raise ValueError('Nonfinite or incomplete full state')
    result={};errors={};absolute={};degenerate={};passes={}
    groups={'pressure':['pressure'],'velocity':['vx','vy','vz'],'tracer':['tracer']}
    for name,components in groups.items():
        value=np.concatenate([predicted[k].astype('f8')for k in components])
        truth=np.concatenate([reference[k].astype('f8')for k in components])
        if name=='pressure':value=value-value.mean();truth=truth-truth.mean()
        absolute[name]=float(np.sqrt(np.mean((value-truth)**2)))
        reference_rms=float(np.sqrt(np.mean(truth**2)))
        floor=normalization['relative_denominator_rms_floor'][name]
        degenerate[name]=reference_rms<floor
        errors[name]=absolute[name]/max(reference_rms,floor)
        passes[name]=(absolute[name]<=normalization['degenerate_absolute_rms_tolerance'][name]
                      if degenerate[name]else errors[name]<=4e-3)
    nx,ny,nz=scene['counts'];h=1/scene['cells_per_metre']
    p=torch.from_numpy(predicted['pressure'].astype('f8').reshape(1,1,nz,ny,nx))
    with torch.inference_mode():
        action=numerics.fine_action(p,torch.tensor([h],dtype=torch.float64),torch.tensor([1000.],dtype=torch.float64)).numpy().reshape(-1)
    equation=float(np.linalg.norm(action-rhs))/max(float(np.linalg.norm(rhs)),1e-30)
    errors['equation']=equation;passes['equation']=equation<=4e-3
    def flow(values):
        vx=values['vx'].astype('f8').reshape(nz,ny,nx+1)
        vy=values['vy'].astype('f8').reshape(nz,ny+1,nx)
        vz=values['vz'].astype('f8').reshape(nz+1,ny,nx)
        div=(vx[:,:,1:]-vx[:,:,:-1]+vy[:,1:,:]-vy[:,:-1,:]+vz[1:,:,:]-vz[:-1,:,:])/h
        walls=np.concatenate([vx[:,:,0].ravel(),vx[:,:,-1].ravel(),vy[:,0,:].ravel(),vy[:,-1,:].ravel(),vz[0,:,:].ravel(),vz[-1,:,:].ravel()])
        flux=h*h*(vx[:,:,-1].sum()-vx[:,:,0].sum()+vy[:,-1,:].sum()-vy[:,0,:].sum()+vz[-1,:,:].sum()-vz[0,:,:].sum())
        return dict(divergence_rms=float(np.sqrt(np.mean(div**2))),boundary_flux_m3_s=float(flux),
            absolute_boundary_flux_m3_s=float(np.abs(walls).sum()*h*h),wall_normal_velocity_max=float(np.abs(walls).max()),
            kinetic_energy_j=float(.5*1000*h**3*sum(np.square(v.astype('f8')).sum()for k,v in values.items()if k in ('vx','vy','vz'))))
    tracer=predicted['tracer'].astype('f8');truth_tracer=reference['tracer'].astype('f8')
    mass=float(tracer.sum()*h**3);reference_mass=float(truth_tracer.sum()*h**3)
    result.update(relative=errors,absolute_rms=absolute,degenerate_reference=degenerate,criterion_pass=passes,
        all_accuracy_criteria_pass=all(passes.values()),pressure_equation_absolute_l2=float(np.linalg.norm(action-rhs)),
        predicted_physics=flow(predicted),reference_physics=flow(reference),
        tracer=dict(mass=mass,reference_mass=reference_mass,mass_difference=mass-reference_mass,mass_drift_from_initial=mass-initial_mass,
            minimum=float(tracer.min()),maximum=float(tracer.max()),below_zero_cells=int((tracer<0).sum()),above_one_cells=int((tracer>1).sum())))
    return result
