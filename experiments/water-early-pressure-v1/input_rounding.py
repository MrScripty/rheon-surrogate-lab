"""Isolate one input-quantization contribution using the existing exact oracle.

No neural evaluation, training, trajectory advance or higher-precision network.
This is not a precision floor for iterative residual correction.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import torch

LAB=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(LAB/'experiments/water-residual-learning-v1'))
import audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--normalization',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();assert not args.output.exists()
    start=time.perf_counter();torch.set_num_threads(1);audit.TORCH=torch
    current=int(next(s for s in Path('/proc/self/status').read_text().splitlines() if s.startswith('VmSize:')).split()[1])*1024
    resource.setrlimit(resource.RLIMIT_AS,(current+512*1024**2,current+512*1024**2))
    config=json.loads(args.config.read_text());norm=json.loads(args.normalization.read_text())
    scale=norm['scales']['laplacian_rhs']['value'];rows=[]
    for counts in config['spatial_counts']:
        shape=counts[::-1];axis=max(range(3),key=lambda i:shape[i])
        def mode(k):
            factors=[torch.cos(torch.pi*(k if i==axis else 0)*(torch.arange(n,dtype=torch.float64)+.5)/n) for i,n in enumerate(shape)]
            return torch.einsum('z,y,x->zyx',*factors)[None,None]
        low,high=mode(1),mode(7)
        low_lambda=2-2*math.cos(math.pi/shape[axis])
        high_lambda=2-2*math.cos(7*math.pi/shape[axis])
        scalars={'cell_size_m':torch.tensor([.125],dtype=torch.float64),'rho':torch.tensor([1000.],dtype=torch.float64)}
        def coefficient(value):return float((value*low).sum()/low.square().sum())
        for kind in ('pure-low','low-plus-high','high-only'):
            for alpha in (config['amplitudes'] if kind!='high-only' else [0.]):
                target=alpha*low+(high if kind!='pure-low' else 0.)
                target=target-target.flatten()[0]
                q=alpha*low_lambda*low+(high_lambda*high if kind!='pure-low' else 0.)
                exact=audit.spectral_correction(q*.125/1000,scalars)
                oracle_error=float(torch.linalg.vector_norm(exact-target)/torch.linalg.vector_norm(target))
                assert oracle_error<1e-10
                quantized=(q/scale).float().double()*scale
                rounded=audit.spectral_correction(quantized*.125/1000,scalars)
                error=coefficient(rounded-target)
                rows.append({'counts_xyz':counts,'kind':kind,'low_amplitude_pa':alpha,
                    'low_coefficient_error_pa':error,'relative_low_error':abs(error/alpha) if alpha else None,
                    'unquantized_independent_relative_oracle_error':oracle_error})
    result={'schema':'rheon-single-input-quantization-probe-v1','rows':rows,
        'normalization_sha256':hashlib.sha256(args.normalization.read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'elapsed_seconds':time.perf_counter()-start,'cuda_initialized':torch.cuda.is_initialized(),
        'optimizer_updates':0,'network_evaluations':0,'simulations_advanced':0,
        'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'scope':'Analytically generated q normalized exactly as the model input, cast float32 once, then exact existing physical oracle. Measures this input-rounding contribution only. It excludes all neural arithmetic; repeated float64 physical residual recomputation can recover weaker signals later, so it is not an iterative inference lower bound.'}
    assert not result['cuda_initialized']
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
