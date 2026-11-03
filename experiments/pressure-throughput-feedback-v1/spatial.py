"""Matched fixed-call spatial amplitude diagnostics on consumed model artifacts.

Reuses the retained analytic cosine experiment; no simulation, optimizer, label
selection or holdout access. Distinguishes pure low-mode error from cross-band
leakage at the same pressure amplitudes and network call counts.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'water-early-pressure-v1'))
import amplitude


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('project','artifacts','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();start=time.perf_counter()
    amplitude.audit.initialize(a.project.resolve(strict=True),a.artifacts.resolve(strict=True),sorted(os.sched_getaffinity(0)))
    config=dict(spatial_counts=[[33,11,9],[16,16,16],[25,16,23]],amplitudes=[1.,.001,1e-8],spatial_calls=[1,4,8,12])
    result=dict(format='rheon_matched_spatial_amplitude_v1',configuration=config,spatial=amplitude.spatial(config),
        weights_sha256=amplitude.digest(a.artifacts/'weights.npz'),source_witness_sha256=amplitude.digest(a.project/'source-witness.json'),
        sources={str(path):amplitude.digest(path) for path in (Path(__file__),Path(amplitude.__file__))},
        parameter_count=sum(p.numel() for p in amplitude.audit.NETWORK.parameters()),weights_bytes=(a.artifacts/'weights.npz').stat().st_size,
        optimizer_updates=0,simulations_advanced=0,cuda_initialized=amplitude.audit.TORCH.cuda.is_initialized(),
        cpu_affinity=sorted(os.sched_getaffinity(0)),cpu_threads=amplitude.audit.TORCH.get_num_threads(),
        elapsed_s=time.perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    assert not result['cuda_initialized'];a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('elapsed_s','parameter_count','weights_bytes')}))

if __name__=='__main__':main()
