"""Original native field decoding and independent physical scores; no model."""
import hashlib
import math

from import_common_generator_measurements import ROOT, require
from component_material import observed_file

FIELDS = ('pressure','vx','vy','vz','tracer')


def digest(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def load_fields(spec, names=FIELDS):
    import numpy as np
    observed_file(ROOT,spec['path'],spec)
    result = {}
    with np.load(ROOT/spec['path'],allow_pickle=False) as archive:
        require(set(archive.files) == set(spec['fields']), 'Original native archive field membership differs')
        for name in archive.files:
            value,description = archive[name],spec['fields'][name]
            require(value.dtype.str == description['dtype'] and list(value.shape) == description['shape']
                    and np.isfinite(value).all() and digest(value) == description['sha256'],
                    'Original typed native field differs: '+name)
            if name in names:
                result[name] = value.copy()
    require(set(result) == set(names) and result.get('pressure',[0])[0] == 0.,
            'Requested fields and original cell-zero pressure gauge required')
    return result


def physical_scores(output, truth, scene, normalization):
    """NumPy independent of the captured Torch metric implementation."""
    import numpy as np
    from review_direction_pressure_fits import operator
    result = dict(relative={},absolute_rms={},raw_relative={},target_floor_active={},target_rms={})
    for name,components in [('pressure',('pressure',)),('velocity',('vx','vy','vz')),('tracer',('tracer',))]:
        predicted = np.concatenate([output[key].astype(np.float64).ravel() for key in components])
        target = np.concatenate([truth[key].astype(np.float64).ravel() for key in components])
        if name == 'pressure':
            predicted = predicted-predicted.mean();target = target-target.mean()
        difference = predicted-target
        error = math.sqrt(float(np.square(difference).sum())/difference.size)
        target_rms = math.sqrt(float(np.square(target).sum())/target.size)
        floor = normalization['relative_denominator_rms_floor'][name]
        result['relative'][name] = error/max(target_rms,floor)
        result['absolute_rms'][name] = error
        result['target_rms'][name] = target_rms
        result['target_floor_active'][name] = target_rms < floor
        result['raw_relative'][name] = error/target_rms if target_rms else (0. if error == 0 else None)
    shape = tuple(reversed(scene['counts']))
    rhs = truth['rhs'].astype(np.float64).reshape(shape)
    action = operator(output['pressure'].reshape(shape),1./scene['cells_per_metre'],1000.)
    difference = action-rhs
    error = math.sqrt(float(np.square(difference).sum()))
    norm = math.sqrt(float(np.square(rhs).sum()))
    result['relative']['equation'] = error/max(norm,normalization['equation_rhs_floor'])
    result['target_floor_active']['equation'] = norm < normalization['equation_rhs_floor']
    result['equation_absolute_rms'] = error/math.sqrt(rhs.size)
    result['pressure_cell_zero_absolute_error'] = abs(float(output['pressure'][0])-float(truth['pressure'][0]))
    require(all(math.isfinite(value) for value in result['relative'].values()), 'Finite physical errors required')
    return result
