"""Launch one isolated resource configuration before numerical-runtime imports."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/pressure-four-pass-distillation-v1'))
from common import environment


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cores', required=True)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cores = [int(value) for value in args.cores.split(',')]
    if cores not in ([23], [6], [6, 4], [6, 4, 5, 7]):
        raise ValueError('Only registered core allocations allowed')
    manifest = json.loads((ROOT / 'runs/dependency-migration-v1/runtime/current-manifest-v2.json').read_text())
    env = environment(manifest)
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               OMP_DYNAMIC='FALSE', OMP_PROC_BIND='TRUE', OMP_WAIT_POLICY='PASSIVE',
               GOMP_SPINCOUNT='0',
               OMP_PLACES=','.join('{' + str(cpu) + '}' for cpu in cores),
               OMP_THREAD_LIMIT=str(len(cores)), RAYON_NUM_THREADS='1')
    env.pop('GOMP_CPU_AFFINITY', None)
    # Affinity is inherited before Torch/OpenMP initialization; each OpenMP place
    # is a single physical core. The explicit kernel team overrides Torch's 1.
    os.sched_setaffinity(0, set(cores))
    raise SystemExit(subprocess.run([manifest['python'], '-B', *args.command],
                                   env=env, timeout=args.timeout).returncode)
