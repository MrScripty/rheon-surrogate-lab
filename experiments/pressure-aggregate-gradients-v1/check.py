"""Literal two-vector and zero-mean oracles for diagnostic policy geometry."""
import argparse
import math
from pathlib import Path
import sys
from aggregation import summarize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import sha, write


def check(output):
    gram = [[1., -10.], [-10., 104.]]  # g0=(1,0), g1=(-10,2)
    cross = {'full':gram, 'final':[[0., 2.], [2., -22.]],
        'early':[[-2., 20.], [1., -8.]], 'canonical':[[3., -32.], [-13., 134.]]}
    norms = {'full':[1., math.sqrt(104)], 'final':[1., math.sqrt(5)],
        'early':[2., math.sqrt(2)], 'canonical':[math.sqrt(10), math.sqrt(173)]}
    raw = summarize(gram, cross, norms)
    assert raw['mean_l2'] == math.sqrt(21.25)
    assert raw['cases'][0]['dot_mean'] == -4.5
    assert math.isclose(raw['cases'][0]['signed_mean_contribution'], -9/85, rel_tol=1e-15)
    assert math.isclose(sum(row['signed_mean_contribution'] for row in raw['cases']), 1., abs_tol=1e-15)
    assert raw['cases'][0]['terms']['final']['dot_mean'] == 1.
    assert raw['cases'][0]['derivative_along_negative_clipped_mean'] > 0
    capped = summarize(gram, cross, norms, cap=1.)
    x = (.999999000001 - 10/(math.sqrt(104)+1e-6))/2
    y = 1/(math.sqrt(104)+1e-6)
    assert math.isclose(capped['mean_l2'], math.hypot(x, y), rel_tol=1e-12)
    assert math.isclose(capped['cases'][0]['dot_mean'], x, rel_tol=1e-10)
    assert capped['cases'][0]['derivative_along_negative_clipped_mean'] < 0
    assert math.isclose(capped['cases'][0]['terms']['final']['dot_mean'], y, rel_tol=1e-12)
    for policy in (raw, capped):
        for row in policy['cases']:
            assert math.isclose(sum(row['terms'][term]['dot_mean'] for term in ('canonical', 'final', 'early')), row['dot_mean'], abs_tol=1e-12)
    zero = summarize([[1., -1.], [-1., 1.]], {'full':[[1., -1.], [-1., 1.]]}, {'full':[1., 1.]})
    assert zero['mean_l2'] == 0 and all(row['cosine_mean'] is None and row['signed_mean_contribution'] is None and row['derivative_along_negative_clipped_mean'] == 0 for row in zero['cases'])
    write(output, dict(status='passed', oracle='Literal g0=(1,0), g1=(-10,2); final=(0,1),(2,-1); direct 2D mean and cap arithmetic; opposite unit vectors for zero mean.',
        raw_mean_l2=raw['mean_l2'], raw_target_cosine=raw['cases'][0]['cosine_mean'],
        capped_target_cosine=capped['cases'][0]['cosine_mean'], target_descent_reversed=True,
        source_sha256=sha(__file__), aggregation_source_sha256=sha(Path(__file__).with_name('aggregation.py'))))
    print('Passed literal mean/cap/projection and zero-direction oracles', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    check(parser.parse_args().output.resolve())
