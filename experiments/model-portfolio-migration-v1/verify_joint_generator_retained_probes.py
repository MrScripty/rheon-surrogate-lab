"""Compare fresh GPU metrics with original saved CPU probes; no CPU forward."""
import math
from pathlib import Path
import time

import numpy as np

from review_water_learning_sequence import binding, canonical, read, require

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
RUN = ROOT/'runs/model-portfolio-migration-v1'
OUTPUT = DOC/'reports/joint-generator-retained-probe-check-01.json'


def relative(predicted, truth, center=False):
    a, b = predicted.astype(np.float64).ravel(), truth.astype(np.float64).ravel()
    if center:
        a, b = a-np.mean(a), b-np.mean(b)
    norm = float(np.linalg.norm(b))
    error = float(np.linalg.norm(a-b))
    return error/norm if norm else (0. if error == 0 else None)


def run():
    require(not OUTPUT.exists(), 'Preserve original retained-probe check')
    cases_path = DOC/'reports/joint-generator-case-preflight-01.json'
    prepared = read(cases_path)
    case = next(c for c in prepared['cases'] if c['identifier']=='temporal-g0-cpm8-r0' and c['start']==0)
    nx,ny,nz = case['scene']['counts']
    slices = ((slice(nz),slice(ny),slice(nx)),(slice(nz),slice(ny),slice(nx+1)),
              (slice(nz),slice(ny+1),slice(nx)),(slice(nz+1),slice(ny),slice(nx)),(slice(nz),slice(ny),slice(nx)))
    rows, evidence = [], [binding(cases_path)]
    for selected in prepared['models']:
        receipt_path = ROOT/selected['receipt']['path']
        receipt = read(receipt_path)
        probe_path = receipt_path.parent/'probe.npy'
        require(binding(probe_path)['sha256'] == receipt['outputs']['probe.npy']['sha256'], 'Original CPU probe changed')
        probe = np.load(probe_path,allow_pickle=False)
        k = selected['K']
        require(probe.dtype.str=='<f4' and probe.shape==(1,k,5,nz+1,ny+1,nx+1), 'Original joint raw-output extent differs')
        norm = read(ROOT/selected['normalization']['path'])
        result_path = RUN/'joint-generator-GPU-benchmark-01'/f'{selected["identifier"]}-phase0.json'
        observed = read(result_path)
        measured = next(r for r in observed['rows'] if r['identifier']==case['identifier'] and r['start']==0)
        comparisons = []
        for lead in range(k):
            physical = {}
            for index,name in enumerate(('pressure','vx','vy','vz','tracer')):
                scale = norm['scales']['velocity' if name in ('vx','vy','vz') else name]
                value = np.asarray(probe[0,lead,index][slices[index]]*np.float32(scale),dtype='<f4').reshape(-1)
                if name=='pressure':
                    value = value.astype(np.float64)
                    value -= value[0]
                physical[name] = value
            reference = case['reference_leads'][lead]
            with np.load(ROOT/reference['path'],allow_pickle=False) as truth:
                expected = {name+'_relative_l2':relative(physical[name],truth[name],name=='pressure') for name in physical}
                expected['velocity_relative_l2'] = relative(np.concatenate([physical[n] for n in ('vx','vy','vz')]),
                    np.concatenate([truth[n] for n in ('vx','vy','vz')]))
            actual = measured['metrics'][lead]['values']
            for name,value in expected.items():
                require((value is None and actual[name] is None) or (value is not None and actual[name] is not None
                        and math.isclose(value,actual[name],rel_tol=5e-4,abs_tol=5e-6)),
                        'Fresh GPU boundary does not reproduce original saved CPU probe metrics')
            comparisons.append(dict(lead=lead,original_saved_CPU_probe_metrics=expected,
                current_GPU_metrics={name:actual[name] for name in expected},
                max_absolute_metric_difference=max(abs(value-actual[name]) for name,value in expected.items() if value is not None)))
        rows.append(dict(identifier=selected['identifier'],fit_item_id=selected['fit_item_id'],comparisons=comparisons))
        evidence += [binding(receipt_path),binding(probe_path),binding(result_path)]
    result = dict(status='four_original_saved_probe_field_metrics_reproduced_by_GPU_boundary',created_unix=time.time(),rows=rows,
        scope='Original first16cubed,cpm8scene atstart0; allKphysical fields. Metric agreement, not bitwise CPU/GPU output equivalence or all64case numerical validation.',
        CPU_neural_forwards=0,CUDA_neural_forwards=0,original_saved_CPU_arrays_read_only=True,
        tolerance=dict(relative=5e-4,absolute=5e-6),new_reference_generation=False,payload_copies=0,
        full_portfolio_complete=False,producer=binding(Path(__file__)),evidence=evidence)
    OUTPUT.write_bytes(canonical(result)+b'\n')
    print('Original saved CPU probes agree with fresh GPU first-case metrics; no new neural forward.')


if __name__ == '__main__':
    run()
