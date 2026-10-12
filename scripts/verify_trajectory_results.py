"""Audit complete solver evidence and recompute accuracy from the saved fields."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    run = Path(sys.argv[1])
    freeze = json.loads((run/'freeze.json').read_text())
    result = json.loads((run/'evaluation/result.json').read_text())
    assert result['freeze_sha256'] == digest(run/'freeze.json')
    assert result['checkpoint_sha256'] == digest(run/'model.pt') == freeze['checkpoint_sha256']
    for name, expected in result['artifact_sha256'].items():
        assert digest(run/'evaluation'/name) == expected, name
    assert set(freeze['test_seeds']).isdisjoint(range(100,112))
    assert freeze['memory']['python_peak_rss_bytes'] < 20_000_000_000
    assert freeze['memory']['gpu_peak_reserved_bytes'] < 20_000_000_000
    assert result['python_peak_rss_bytes'] < 8_000_000_000
    assert result['gpu_peak_reserved_bytes'] < 8_000_000_000
    errors = []
    for case in result['accuracy']:
        seed = case['seed']
        with np.load(run/f'evaluation/jacobi-{seed}.npz') as ref, np.load(run/f'evaluation/learned-{seed}.npz') as ai:
            assert ref['tracer'].shape == (120,freeze['n']**3)
            velocity_numerator = np.zeros(120)
            velocity_denominator = np.zeros(120)
            for d in range(3):
                actual=ai[f'velocity_{d}'].astype(np.float64)
                expected=ref[f'velocity_{d}'].astype(np.float64)
                velocity_numerator += np.sum((actual-expected)**2,axis=1)
                velocity_denominator += np.sum(expected**2,axis=1)
            velocity=np.sqrt(velocity_numerator/velocity_denominator)
            tracer=ai['tracer'].astype(np.float64)
            truth=ref['tracer'].astype(np.float64)
            tracer_error=np.linalg.norm(tracer-truth,axis=1)/np.linalg.norm(truth,axis=1)
            # The original receipt computes the reference norm on its stored
            # float32 array. This independent audit promotes both operands to
            # float64; retain that precision difference explicitly, with a
            # float32 reduction tolerance (not a solver-accuracy relaxation).
            np.testing.assert_allclose(velocity,[r['relative_velocity_l2'] for r in case['errors']],rtol=1e-6,atol=1e-16)
            np.testing.assert_allclose(tracer_error,[r['relative_tracer_l2'] for r in case['errors']],rtol=1e-6,atol=1e-16)
            errors.append({'seed':seed,'max_velocity_relative_l2':float(velocity.max()),
                           'max_tracer_relative_l2':float(tracer_error.max())})
    records=result['benchmark']
    assert len(records)==60
    qualified_frames=0
    for record in records:
        assert len(record['reports'])==120
        for frame in record['reports']:
            assert frame['actual_divergence_max'] <= 1e-5
            # All observed AI frames also meet the reference's strict relative
            # tolerance, despite a looser rejection gate in the first probe.
            assert frame['relative_residual'] <= 1e-8
            assert frame['courant'] <= 1
            if record['method']=='learned': assert frame['iterations']==0
            qualified_frames+=1
    for method, median in result['median_120_frame_wall_ms'].items():
        observed=statistics.median(r['wall_ms_including_native_startup_teardown'] for r in records if r['method']==method)
        assert observed==median
    assert result['speedup_vs_jacobi'] > 1 and result['speedup_vs_sgs'] > 1
    summary={'status':'verified','qualified_benchmark_frames':qualified_frames,'accuracy_recomputed_from_full_fields':errors,
             'norm_precision_note':'Audit uses float64 numerator and denominator; original receipts use float32 reference norm. Agreement checked within float32 reduction precision.',
             'training_parameters':freeze['training']['fitted_parameters'],
             'training_gpu_reserved_bytes':freeze['memory']['gpu_peak_reserved_bytes'],
             'training_python_rss_bytes':freeze['memory']['python_peak_rss_bytes'],
             'inference_gpu_reserved_bytes':result['gpu_peak_reserved_bytes'],
             'inference_python_rss_bytes':result['python_peak_rss_bytes'],
             'speedup_vs_jacobi':result['speedup_vs_jacobi'],'speedup_vs_sgs':result['speedup_vs_sgs']}
    (run/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
