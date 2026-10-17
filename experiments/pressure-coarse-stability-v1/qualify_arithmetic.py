"""Independent scalar oracles for a prospective float32 norm repair only."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch

import gradient_clipping
from gradient_clipping import clip_l2_, ORIGINAL_CLIP


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    started=time.perf_counter()
    analysis=json.loads(args.analysis.read_text())
    assert analysis['classification']=='finite_components_norm_reduction_overflow'
    observed=[row['ordinary_float32_norm']for row in analysis['failure_observation']['update']['gradient_snapshot']['parameters']]
    assert all(isinstance(value,float)and math.isfinite(value)for value in observed)
    actual_pair=sorted(observed,reverse=True)[:2]
    reports=[]
    for device in ('cpu','cuda'):
        def parameters(values):
            ps=[torch.nn.Parameter(torch.zeros(1,dtype=torch.float32,device=device))for _ in values]
            for parameter,value in zip(ps,values):
                parameter.grad=torch.tensor([value],dtype=torch.float32,device=device)
            return ps
        safe=[]
        for values in ([3.,4.],[.03,.04],[0.,0.],[1e10,1e10],[actual_pair[0]]):
            original, repaired=parameters(values),parameters(values)
            expected=ORIGINAL_CLIP(original,1.0,error_if_nonfinite=True)
            actual=clip_l2_(repaired)
            assert torch.equal(actual,expected)
            assert all(torch.equal(a.grad,b.grad)for a,b in zip(original,repaired))
            safe.append(dict(values=values,norm=float(actual),ordinary_path_exact=True))
        missing=[]
        for label,values in [('some-absent',[3.,None,4.]),('all-absent',[None,None]),('empty',[])]:
            original=parameters([0. if value is None else value for value in values])
            repaired=parameters([0. if value is None else value for value in values])
            for index,value in enumerate(values):
                if value is None:original[index].grad=repaired[index].grad=None
            expected=ORIGINAL_CLIP(original,1.,error_if_nonfinite=True)
            actual=clip_l2_(repaired)
            assert torch.equal(actual,expected)
            for a,b in zip(original,repaired):
                assert (a.grad is None and b.grad is None)or torch.equal(a.grad,b.grad)
            missing.append(dict(case=label,norm=float(actual),ordinary_path_exact=True))
        overflow=[]
        for label,values in [('actual-pair',actual_pair),('actual-all-parameter-norms',observed),('known-3-4',[3e20,4e20])]:
            ps=parameters(values)
            represented=[float(p.grad)for p in ps]
            expected=math.hypot(*represented)
            actual=clip_l2_(ps)
            clipped=[float(p.grad)for p in ps]
            assert actual.dtype==torch.float32 and actual.device==ps[0].device
            assert abs(float(actual)/expected-1)<2e-6
            assert abs(math.hypot(*clipped)-1)<2e-6
            assert all(abs(a-b/expected)<2e-6 for a,b in zip(clipped,represented))
            overflow.append(dict(case=label,independent_expected_norm=expected,actual_float32_norm=float(actual),
                clipped_reference_norm=math.hypot(*clipped),directions_match=True,all_tensor_dtypes='float32'))
        negative=[]
        maximum=(2-2**-23)*2**127
        for label,values in [('nan-component',[float('nan'),1.]),('infinite-component',[float('inf'),1.]),
                             ('finite-components-unrepresentable-norm',[.9*maximum,.9*maximum])]:
            ps=parameters(values)
            before=[p.grad.clone()for p in ps]
            try:
                clip_l2_(ps)
            except RuntimeError as error:
                assert 'non-finite' in str(error)
            else:
                raise AssertionError('Norm guard must reject '+label)
            for p,saved in zip(ps,before):
                torch.testing.assert_close(p.grad,saved,rtol=0,atol=0,equal_nan=True)
            negative.append(dict(case=label,rejected=True,gradients_unchanged=True))
        reports.append(dict(device=device,ordinary_cases=safe,ordinary_missing_gradient_cases=missing,repaired_cases=overflow,negative_cases=negative))
    result=dict(format='pressure-coarse-clipping-arithmetic-qualification-v1',status='passed',
        elapsed_s=time.perf_counter()-started,source_sha256=hashlib.sha256(Path(gradient_clipping.__file__).read_bytes()).hexdigest(),
        qualifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        actual_analysis_sha256=hashlib.sha256(args.analysis.read_bytes()).hexdigest(),
        reports=reports,optimizer_updates=0,network_forwards=0,simulations_advanced=0,
        installed_in_training=True,managed_replay_with_repair_started=False,
        scope='Experimental arithmetic on observed per-parameter norm scalars and independent known vectors. Exact ordinary finite clipping path retained; NaN/Inf and unrepresentable final norms reject. No actual gradient tensors, model update/stability/accuracy or completed training claim.')
    with args.output.open('x')as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({key:result[key]for key in ('status','elapsed_s','installed_in_training','optimizer_updates')}))


if __name__=='__main__':
    main()
