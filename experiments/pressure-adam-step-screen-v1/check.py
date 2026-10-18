"""Literal endpoint and scalar Adam oracles before the snapshot screen."""
import argparse
import math
import os
from pathlib import Path
import sys
import time
import torch
from steps import adam,interpolate

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import sha,write


def check(output,cpu):
    started=time.perf_counter();os.sched_setaffinity(0,{cpu});torch.set_num_threads(1)
    if output.exists():raise ValueError('Fresh oracle receipt required')
    a=torch.tensor([1.,2.,-1.]);b=torch.tensor([1.5,1.,-2.])
    assert torch.equal(interpolate(a,b,.125),torch.tensor([1.0625,1.875,-1.125]))
    assert torch.equal(interpolate(a,b,0),a) and torch.equal(interpolate(a,b,1),b)
    group=dict(lr=.05,betas=[.9,.999],eps=1e-8,weight_decay=0,amsgrad=False,maximize=False)
    result=adam(torch.tensor([2.]),torch.tensor([-.5]),group,2,(torch.tensor([.125]),torch.tensor([.03125])))
    first=.9*.125+.1*(-.5);second=.999*.03125+.001*.25
    expected=2-.05*(first/(1-.9**3))/(math.sqrt(second/(1-.999**3))+1e-8)
    error=abs(float(result[0])-expected);assert error<=2e-7
    fresh=adam(a,torch.tensor([.25,-.5,.1]),group,0)
    zero_moments=adam(a,torch.tensor([.25,-.5,.1]),group,0,(torch.zeros_like(a),torch.zeros_like(a)))
    assert torch.equal(fresh,zero_moments)
    try:interpolate(a,b,.3)
    except ValueError as e:assert str(e)=='Predeclared bounded fractions required'
    else:raise AssertionError('Undeclared interpolation fraction accepted')
    try:adam(a,a,group,2)
    except ValueError as e:assert str(e)=='No fabricated Adam history'
    else:raise AssertionError('Unobserved Adam moments accepted')
    assert not torch.cuda.is_initialized()
    write(output,dict(status='passed',literal_interpolation_exact=True,
        scalar_adam_expected=expected,scalar_adam_actual=float(result[0]),absolute_error=error,
        scalar_absolute_tolerance=2e-7,zero_moment_initialization_exact=True,
        undeclared_fraction_rejected=True,missing_moments_rejected=True,
        cuda_initialized=False,elapsed_s=time.perf_counter()-started,
        source_sha256=sha(__file__),steps_source_sha256=sha(Path(__file__).with_name('steps.py'))))
    print('Passed literal interpolation and independent scalar Adam oracles',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--cpu',type=int,required=True)
    a=p.parse_args();check(a.output.resolve(),a.cpu)
