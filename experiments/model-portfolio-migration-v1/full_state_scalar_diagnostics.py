"""Reduce one full staggered state to scalars without retaining field arrays."""
import math

from pressure_scalar_diagnostics import error_summary, field_summary, equation
from spatial_field_statistics import measure as spatial_measure

FIELDS = ('pressure','vx','vy','vz','tracer')


def shaped(fields, counts):
    import numpy as np
    nx,ny,nz = counts
    shapes = dict(pressure=(nz,ny,nx),tracer=(nz,ny,nx),vx=(nz,ny,nx+1),vy=(nz,ny+1,nx),vz=(nz+1,ny,nx))
    result = {}
    for name,shape in shapes.items():
        value = np.asarray(fields[name],dtype=np.float64)
        if value.size != math.prod(shape) or not np.isfinite(value).all():
            raise ValueError('Finite correctly sized staggered field required: '+name)
        result[name] = value.reshape(shape)
    return result


def cell_velocity(fields):
    return dict(x=(fields['vx'][:,:,1:]+fields['vx'][:,:,:-1])*.5,
        y=(fields['vy'][:,1:,:]+fields['vy'][:,:-1,:])*.5,
        z=(fields['vz'][1:,:,:]+fields['vz'][:-1,:,:])*.5)


def vorticity(fields, spacing):
    import numpy as np
    velocity = cell_velocity(fields)
    def derivative(name,axis):
        value = velocity[name]
        return np.gradient(value,spacing[2-axis],axis=axis,edge_order=1) if value.shape[axis]>1 else np.zeros_like(value)
    return dict(x=derivative('z',1)-derivative('y',0),
        y=derivative('x',0)-derivative('z',2),z=derivative('y',2)-derivative('x',1))


def state_quantities(fields, spacing, rho):
    import numpy as np
    velocity = cell_velocity(fields)
    volume = math.prod(spacing)
    speed_squared = sum(np.square(v) for v in velocity.values())
    return dict(cell_centered_speed_rms=math.sqrt(float(speed_squared.mean())),
        kinetic_energy_joule=.5*rho*volume*float(speed_squared.sum()),
        momentum_kg_m_per_s={axis:rho*volume*float(v.sum()) for axis,v in velocity.items()},
        tracer_volume_integral=volume*float(fields['tracer'].sum()))


def measure(output, truth, current, counts, spacing, rho, previous_output=None, previous_truth=None):
    import numpy as np
    if len(counts)!=3 or any(type(v) is not int or v<=0 for v in counts):
        raise ValueError('Positive XYZ integer grid required')
    if len(spacing)!=3 or any(not math.isfinite(v) or v<=0 for v in spacing) or not math.isfinite(rho) or rho<=0:
        raise ValueError('Positive XYZ spacing and fluid mass density required')
    p,r,b = (shaped(value,counts) for value in (output,truth,current))
    prior = shaped(previous_output,counts) if previous_output is not None else b
    previous_reference = shaped(previous_truth,counts) if previous_truth is not None else b
    fields = {}
    for name in FIELDS:
        predicted,reference,base = p[name],r[name],b[name]
        preceding,preceding_reference = prior[name],previous_reference[name]
        raw_predicted,raw_reference,raw_base = predicted,reference,base
        if name == 'pressure':
            predicted,reference,base = (v-v.mean() for v in (predicted,reference,base))
            preceding,preceding_reference = (v-v.mean() for v in (preceding,preceding_reference))
        error = error_summary(predicted-reference,reference)
        baseline = error_summary(base-reference,reference)
        fields[name] = dict(input=field_summary(raw_base),prediction=field_summary(raw_predicted),
            reference=field_summary(raw_reference),error=error,
            unchanged_input_baseline_error=baseline,
            prediction_minus_input=field_summary(predicted-base),reference_minus_input=field_summary(reference-base),
            prediction_minus_preceding_output=field_summary(predicted-preceding),
            reference_minus_preceding_reference=field_summary(reference-preceding_reference),
            RMS_improvement_vs_unchanged_input=baseline['rms']-error['rms'],
            RMS_ratio_vs_unchanged_input=error['rms']/baseline['rms'] if baseline['rms'] else None)
    vector = lambda f:np.concatenate([f[n].ravel() for n in ('vx','vy','vz')])
    pv,rv,bv = (vector(f) for f in (p,r,b))
    combined_velocity = error_summary(pv-rv,rv)
    combined_baseline = error_summary(bv-rv,rv)
    pc,rc = cell_velocity(p),cell_velocity(r)
    pvectors = np.stack([pc[a] for a in ('x','y','z')],axis=-1)
    rvectors = np.stack([rc[a] for a in ('x','y','z')],axis=-1)
    pspeed,rspeed = (np.sqrt(np.square(v).sum(axis=-1)) for v in (pvectors,rvectors))
    valid = (pspeed>1e-30)&(rspeed>1e-30)
    angles = np.degrees(np.arccos(np.clip(np.sum(pvectors[valid]*rvectors[valid],axis=-1)/(pspeed[valid]*rspeed[valid]),-1.,1.)))
    direction = dict(cells=valid.size,defined_cells=int(valid.sum()),
        reference_near_zero_cells=int(np.count_nonzero(rspeed<=1e-30)),
        prediction_near_zero_cells=int(np.count_nonzero(pspeed<=1e-30)),near_zero_speed_floor_m_per_s=1e-30,
        angle_degrees=field_summary(angles) if angles.size else None,
        angle_p95_degrees=float(np.quantile(angles,.95,method='linear')) if angles.size else None)
    quantities = {name:state_quantities(value,spacing,rho) for name,value in (('input',b),('prediction',p),('reference',r))}
    quantities['kinetic_energy_error_joule'] = quantities['prediction']['kinetic_energy_joule']-quantities['reference']['kinetic_energy_joule']
    quantities['momentum_error_kg_m_per_s'] = {axis:quantities['prediction']['momentum_kg_m_per_s'][axis]-quantities['reference']['momentum_kg_m_per_s'][axis] for axis in ('x','y','z')}
    pvort,rvort = vorticity(p,spacing),vorticity(r,spacing)
    vort = {axis:error_summary(pvort[axis]-rvort[axis],rvort[axis]) for axis in ('x','y','z')}
    rhs = np.asarray(truth['rhs'],dtype=np.float64).reshape(tuple(reversed(counts)))
    residual = equation(p['pressure'],spacing,rho)-rhs
    base_residual = equation(b['pressure'],spacing,rho)-rhs
    return dict(fields=fields,combined_velocity_error=combined_velocity,
        combined_velocity_unchanged_input_baseline_error=combined_baseline,
        cell_centered_speed_error=error_summary(pspeed-rspeed,rspeed),cell_centered_velocity_direction=direction,
        equation_residual=error_summary(residual,rhs),unchanged_input_equation_residual=error_summary(base_residual,rhs),
        pressure_gauge_offset_vs_reference=float(p['pressure'].mean()-r['pressure'].mean()),
        spatial=spatial_measure(output,truth,counts,spacing),state_quantities=quantities,
        cell_centered_vorticity_error_per_s=vort,
        output_dtypes={n:np.asarray(output[n]).dtype.str for n in FIELDS})
