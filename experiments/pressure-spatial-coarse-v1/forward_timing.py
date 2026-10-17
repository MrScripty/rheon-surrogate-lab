"""Serial raw CPU-network latency; excludes physics and comparison-worker work."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'water-residual-learning-v1'))
import audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('project','artifacts','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('fresh report required')
    available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                         if x.startswith('MemAvailable:'))) * 1024
    if available < 10*1024**3 or len(os.sched_getaffinity(0)) != 1:
        raise ValueError('one CPU core and 2 GiB accounting plus 8 GiB reserve required')
    started = time.perf_counter()
    audit.initialize(a.project.resolve(strict=True),a.artifacts.resolve(strict=True),sorted(os.sched_getaffinity(0)))
    initialized = time.perf_counter()
    vm = int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines()
                  if x.startswith('VmSize:'))) * 1024
    resource.setrlimit(resource.RLIMIT_AS, (vm + 512*1024**2,) * 2)
    torch = audit.TORCH
    rows = []
    with torch.no_grad():
        for n in (16,24,32,40):
            free = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                            if x.startswith('MemAvailable:'))) * 1024
            if free < 8*1024**3:
                raise MemoryError('eight GiB available reserve lost')
            x = torch.arange(n,dtype=torch.float32)+.5
            values = torch.cos(torch.pi*x/n)[None,None,None,None,:].expand(1,1,n,n,n).contiguous()
            before_weights = hashlib.sha256((a.artifacts/'weights.npz').read_bytes()).hexdigest()
            for _ in range(3):
                audit.NETWORK(values)
            times = []
            first = audit.NETWORK(values).clone()
            for _ in range(31):
                before = time.perf_counter()
                output = audit.NETWORK(values)
                times.append(time.perf_counter()-before)
                if not torch.equal(output,first):
                    raise ValueError('repeated forward changed physical input response')
            if hashlib.sha256((a.artifacts/'weights.npz').read_bytes()).hexdigest() != before_weights:
                raise ValueError('weights changed during timing')
            rows.append({'counts_xyz':[n]*3,'input_sha256':hashlib.sha256(values.numpy().tobytes()).hexdigest(),
                'output_sha256':hashlib.sha256(first.numpy().tobytes()).hexdigest(),
                'forward_seconds':times,'median_s':statistics.median(times),
                'mean_s':statistics.mean(times),'minimum_s':min(times),'maximum_s':max(times)})
    result = {'format':'pressure-spatial-coarse-raw-forward-timing-v1','status':'passed','rows':rows,
        'parameters':sum(p.numel() for p in audit.NETWORK.parameters()),'weights_bytes':(a.artifacts/'weights.npz').stat().st_size,
        'weights_sha256':hashlib.sha256((a.artifacts/'weights.npz').read_bytes()).hexdigest(),
        'source_witness_sha256':hashlib.sha256((a.project/'source-witness.json').read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'initialization_s':initialized-started,'elapsed_s':time.perf_counter()-started,
        'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'cpu_affinity':sorted(os.sched_getaffinity(0)),'cpu_threads':torch.get_num_threads(),
        'cuda_initialized':torch.cuda.is_initialized(),'optimizer_updates':0,'simulations_advanced':0,
        'scope':'31 serial raw network forwards after warm-up on fixed consumed cubic grids; excludes input construction, physical reconstruction, residual/stopping, IPC, native solve and logging. Not end-to-end solver speed.'}
    if torch.cuda.is_initialized():
        raise ValueError('CPU timing initialized CUDA')
    with a.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status':'passed','elapsed_s':result['elapsed_s']}))


if __name__ == '__main__':
    main()
