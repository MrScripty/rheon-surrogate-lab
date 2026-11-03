"""Descriptive reference-discretization differences, without model pass claims."""
import numpy as np


def difference(candidate, reference):
    rows = {}
    for field in ('pressure', 'velocity', 'tracer'):
        keys = ('vx', 'vy', 'vz') if field == 'velocity' else (field,)
        a = np.concatenate([candidate[k].astype('f8').reshape(-1) for k in keys])
        b = np.concatenate([reference[k].astype('f8').reshape(-1) for k in keys])
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Finite complete reference states required')
        if field == 'pressure':
            a -= a.mean()
            b -= b.mean()
        delta = a-b
        peak = float(np.max(np.abs(b)))
        norm = float(np.linalg.norm(b))
        rows[field] = dict(max_absolute=float(np.max(np.abs(delta))),
            relative_linf_to_reference_peak=float(np.max(np.abs(delta)))/max(peak, 1e-12),
            relative_l2=float(np.linalg.norm(delta))/max(norm, 1e-12),
            reference_peak=peak, reference_l2=norm, denominator_floor=1e-12,
            pressure_mean_removed=field == 'pressure')
    return rows
