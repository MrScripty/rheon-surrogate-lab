"""Array-to-scalar pressure diagnostics; no model/runtime/persistence dependency."""
import math


def field_summary(value):
    import numpy as np
    value = np.asarray(value, dtype=np.float64)
    if not value.size or not np.isfinite(value).all():
        raise ValueError('Nonempty finite field required')
    return dict(values=value.size, minimum=float(value.min()), maximum=float(value.max()),
        mean=float(value.mean()), population_std=float(value.std()),
        rms=math.sqrt(float(np.square(value).mean())), l2=float(np.linalg.norm(value.ravel())))


def error_summary(difference, target, floor=1e-30):
    import numpy as np
    if not math.isfinite(floor) or floor <= 0:
        raise ValueError('Positive finite L2 denominator floor required')
    result = field_summary(difference)
    target_summary = field_summary(target)
    absolute = np.abs(difference)
    quantiles = np.quantile(absolute, [.9,.95,.99], method='linear')
    result.update(mae=float(absolute.mean()), maximum_absolute=float(absolute.max()),
        absolute_p90=float(quantiles[0]), absolute_p95=float(quantiles[1]), absolute_p99=float(quantiles[2]),
        reference_l2=target_summary['l2'], denominator_floor_l2=floor,
        denominator_floor_active=target_summary['l2'] < floor,
        relative_l2=result['l2']/max(floor,target_summary['l2']))
    return result


def equation(pressure, spacing_xyz, rho):
    """Sealed positive pressure operator (-Laplacian); NumPy evaluation."""
    import numpy as np
    pressure = np.asarray(pressure, dtype=np.float64)
    if pressure.ndim != 3 or any(n <= 0 for n in pressure.shape):
        raise ValueError('Nonempty ZYX cell grid required')
    if len(spacing_xyz) != 3 or any(not math.isfinite(h) or h <= 0 for h in spacing_xyz):
        raise ValueError('Positive finite XYZ cell spacing required')
    if not math.isfinite(rho) or rho <= 0 or not np.isfinite(pressure).all():
        raise ValueError('Finite pressure and positive fluid mass density required')
    result = np.zeros_like(pressure)
    for physical_axis, array_axis in enumerate((2,1,0)):
        others = [i for i in range(3) if i != physical_axis]
        area = spacing_xyz[others[0]]*spacing_xyz[others[1]]
        weight = area*(1.0/(rho*spacing_xyz[physical_axis]))
        delta = np.diff(pressure, axis=array_axis)
        lower = [(0,0)]*3
        upper = [(0,0)]*3
        lower[array_axis] = (1,0)
        upper[array_axis] = (0,1)
        result = result + weight*(np.pad(delta,lower)-np.pad(delta,upper))
    return result


def measure(prediction, sample, counts_xyz):
    import numpy as np
    shape = tuple(reversed(counts_xyz))
    if len(shape) != 3 or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError('Three positive integer grid counts required')
    if any(np.asarray(value).size != math.prod(shape) for value in [prediction,*[sample[n] for n in ('base','reference','rhs','native_input_residual')]]):
        raise ValueError('Pressure request and prediction sizes must match the cell grid')
    p = np.asarray(prediction, dtype=np.float64).reshape(shape)
    fields = {n:np.asarray(sample[n],dtype=np.float64).reshape(shape) for n in ('base','reference','rhs','native_input_residual')}
    b,r,q = (fields[n] for n in ('base','reference','rhs'))
    h,rho = sample['cell_size_m'],sample['rho']
    centered = lambda v:v-v.mean()
    # Match the original scoring order: centre the difference, not the two
    # large input fields independently before subtraction.
    pressure_error = error_summary(centered(p-r),centered(r))
    base_error = error_summary(centered(b-r),centered(r))
    residual = equation(p,[h]*3,rho)-q
    base_residual = equation(b,[h]*3,rho)-q
    result = dict(input_base=field_summary(b),input_base_mean_free=field_summary(centered(b)),
        input_residual=field_summary(fields['native_input_residual']),
        reference_pressure=field_summary(r),reference_pressure_mean_free=field_summary(centered(r)),
        reference_rhs=field_summary(q),prediction_pressure=field_summary(p),
        prediction_pressure_mean_free=field_summary(centered(p)),
        pressure_mean_free_error=pressure_error,equation_residual=error_summary(residual,q),
        pressure_gauge_offset_vs_reference=float(p.mean()-r.mean()),
        supplied_base_pressure_error=base_error,supplied_base_equation_residual=error_summary(base_residual,q),
        prediction_minus_supplied_base=field_summary(centered(p-b)),
        reference_minus_supplied_base=field_summary(centered(r-b)),
        absolute_pressure_RMS_improvement_vs_supplied_base=base_error['rms']-pressure_error['rms'],
        pressure_RMS_ratio_vs_supplied_base=pressure_error['rms']/base_error['rms'] if base_error['rms'] else None,
        pressure_better_than_supplied_base=pressure_error['rms'] < base_error['rms'])
    gradients = {}
    for name,axis in zip(('x','y','z'),(2,1,0),strict=True):
        if shape[axis] > 1:
            target = np.diff(r,axis=axis)/h
            difference = np.diff(p-r,axis=axis)/h
            gradients[name] = error_summary(difference,target)
    result['pressure_gradient_error'] = gradients
    return result
