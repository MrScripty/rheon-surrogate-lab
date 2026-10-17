"""Reproduce the observed outer norm overflow at the real clipping seam.

Use two actual observed per-parameter norms as scalar gradients. These are not
the original gradient tensors. The outer aggregation and original Torch guard
are the same operation; deleting either element must remove the false rejection.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    started=time.perf_counter()
    result=json.loads(args.analysis.read_text())
    assert result['classification']=='finite_components_norm_reduction_overflow'
    rows=result['failure_observation']['update']['gradient_snapshot']['parameters']
    assert all(row['gradient_present']and row['finite']and isinstance(row['ordinary_float32_norm'],float)for row in rows)
    selected=sorted(rows,key=lambda row:row['ordinary_float32_norm'],reverse=True)[:2]
    values=[row['ordinary_float32_norm']for row in selected]
    expected=math.hypot(*values)
    float32_max=(2-2**-23)*2**127
    assert math.isfinite(expected)and expected<float32_max
    observations=[]
    for device in ('cpu','cuda'):
        def parameters(values):
            ps=[torch.nn.Parameter(torch.zeros(1,dtype=torch.float32,device=device))for _ in values]
            for parameter,value in zip(ps,values):
                parameter.grad=torch.tensor([value],dtype=torch.float32,device=device)
            return ps
        ps=parameters(values)
        saved=[p.grad.clone()for p in ps]
        try:
            torch.nn.utils.clip_grad_norm_(ps,1.0,error_if_nonfinite=True)
        except RuntimeError as error:
            assert 'non-finite' in str(error)
        else:
            raise AssertionError('Observed two-norm outer overflow did not reproduce')
        assert all(torch.equal(p.grad,value)for p,value in zip(ps,saved))
        singles=[]
        for value in values:
            original=torch.nn.utils.clip_grad_norm_(parameters([value]),1.0,error_if_nonfinite=True)
            assert bool(torch.isfinite(original))
            singles.append(float(original))
        observations.append(dict(device=device,pair_guard_rejected=True,gradients_unchanged=True,
            deleting_either_element_removes_failure=True,single_norms=singles,
            actual_pair_values=values,independent_expected_norm=expected,
            ordinary_pair_norm_is_infinite=bool(torch.isinf(torch.linalg.vector_norm(torch.tensor(values,dtype=torch.float32,device=device))))))
    report=dict(format='pressure-coarse-gradient-minimal-reproduction-v1',status='reproduced',
        elapsed_s=time.perf_counter()-started,original_guard_error_expected=True,
        observed_failed_update=result['exact_failed_update'],observed_run_id=result['run_id'],
        analysis_sha256=hashlib.sha256(args.analysis.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        observed_parameter_names=[row['parameter']for row in selected],observations=observations,
        optimizer_updates=0,network_forwards=0,simulations_advanced=0,
        scope='Actual observed finite per-parameter norm pair at original Torch clipping seam on CPU/CUDA. Both elements load-bearing for outer overflow; no original gradient tensors, new model fitting, pressure accuracy or fixed training claimed.')
    with args.output.open('x')as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({key:report[key]for key in ('status','elapsed_s','observed_failed_update')}))


if __name__=='__main__':
    main()
