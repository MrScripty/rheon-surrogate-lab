"""Literal independent dot, norm and layer-partition oracles."""
import argparse
from pathlib import Path
import sys
import time
import numpy as np
from geometry import projection, layers

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import write, sha


def check(output):
    if output.exists():
        raise ValueError('Fresh geometry oracle receipt required')
    started = time.perf_counter()
    a,b = np.array([3.,4.]),np.array([-4.,3.])
    orthogonal = projection(a,b)
    assert orthogonal == dict(dot=0.,left_l2=5.,right_l2=5.,cosine=0.)
    assert projection(a,-a)['cosine'] == -1
    assert projection(a,np.zeros(2))['cosine'] is None
    result = layers(a,b,[dict(name='x',count=1),dict(name='y',count=1)])
    assert [r['dot']for r in result] == [-12.,12.]
    rejected = []
    for call,message in [
        (lambda:projection(a,np.zeros(3)),'Finite matching one-dimensional vectors required'),
        (lambda:projection(a,np.array([float('nan'),0.])), 'Finite matching one-dimensional vectors required'),
        (lambda:layers(a,b,[dict(name='x',count=1)]),'Complete captured parameter layout required')]:
        try:
            call()
        except ValueError as error:
            assert str(error) == message
            rejected.append(message)
        else:
            raise AssertionError('Invalid geometry input admitted')
    write(output,dict(status='passed',literal_orthogonal=orthogonal,opposed_cosine=-1,
                      zero_norm_undefined=True,layer_dots=[-12,12],rejected=rejected,
                      elapsed_s=time.perf_counter()-started,source_sha256=sha(__file__),
                      geometry_sha256=sha(Path(__file__).with_name('geometry.py'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    check(parser.parse_args().output.resolve())
