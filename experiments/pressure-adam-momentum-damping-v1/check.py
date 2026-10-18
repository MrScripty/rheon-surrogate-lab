"""Independent scalar recurrence and untouched-state checks for momentum damping."""
import argparse
import math
import os
from pathlib import Path
import sys
import time

import torch
from steps import FACTORS, damped_adam, owner

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/pressure-four-pass-distillation-v1'))
from common import sha, write


def check(output, cpu):
    started = time.perf_counter()
    os.sched_setaffinity(0, {cpu})
    torch.set_num_threads(1)
    if output.exists():
        raise ValueError('Fresh oracle receipt required')
    group = dict(lr=.05, betas=[.9, .999], eps=1e-8, weight_decay=0,
                 amsgrad=False, maximize=False)
    values = [torch.tensor([v]) for v in (2., -.5, .125, .03125)]
    before, gradient, first, second = values
    originals = [v.clone() for v in values]
    records = []
    for factor in FACTORS:
        result = damped_adam(before, gradient, group, 2, first, second, factor)
        m = .9 * (factor * .125) + .1 * (-.5)
        v = .999 * .03125 + .001 * .25
        expected = 2 - .05 * (m / (1 - .9**3)) / (math.sqrt(v / (1 - .999**3)) + 1e-8)
        assert abs(float(result[0]) - expected) <= 2e-7
        records.append(dict(factor=factor, expected=expected, actual=float(result[0])))
    assert all(torch.equal(a, b) for a, b in zip(values, originals))
    assert torch.equal(damped_adam(before, gradient, group, 2, first, second, 1.),
                       owner.adam(before, gradient, group, 2, (first, second)))
    zero_first = damped_adam(before, gradient, group, 2, first, second, 0.)
    assert not torch.equal(zero_first, owner.adam(before, gradient, group, 0))
    initial = [damped_adam(before, gradient, group, 0,
                         torch.zeros_like(first), torch.zeros_like(second), f) for f in FACTORS]
    assert all(torch.equal(p, initial[0]) for p in initial)
    rejected = []
    for kwargs, diagnostic in [
        (dict(factor=.3), 'Predeclared first-moment retention factor required'),
        (dict(step_before=-1), 'Captured nonnegative integer Adam step required'),
        (dict(second=-second), 'Nonnegative Adam second moment required'),
        (dict(gradient=gradient.double()), 'Finite matching CPU float32 vectors required'),
    ]:
        args = dict(before=before, gradient=gradient, group=group, step_before=2,
                    first=first, second=second, factor=.5)
        args.update(kwargs)
        try:
            damped_adam(**args)
        except ValueError as error:
            assert str(error) == diagnostic
            rejected.append(diagnostic)
        else:
            raise AssertionError('Invalid damping input admitted')
    assert not torch.cuda.is_initialized()
    write(output, dict(status='passed', scalar_cases=records, tolerance=2e-7,
                       input_vectors_unchanged=True, full_retention_exact=True,
                       zero_retention_differs_from_full_reset=True,
                       initialization_all_factors_exact=True, rejected=rejected,
                       elapsed_s=time.perf_counter()-started, cuda_initialized=False,
                       source_sha256=sha(__file__), steps_source_sha256=sha(Path(__file__).with_name('steps.py'))))
    print('Passed independent first-moment damping oracles', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu', type=int, required=True)
    args = parser.parse_args()
    check(args.output.resolve(), args.cpu)
