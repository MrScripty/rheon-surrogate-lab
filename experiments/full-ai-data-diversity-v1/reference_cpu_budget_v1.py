"""Launch a reference simulation inside the user's half-physical-core limit.

The target retains its own stricter single-worker affinity and finite protocol.
Native children inherit this affinity and the single-thread environment.
"""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('target', type=Path)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text())
    selected = set(policy['allowed_logical_cpus'])
    current = os.sched_getaffinity(0)
    if not selected or not selected <= current:
        raise ValueError('Registered half-core reference affinity is unavailable')
    physical = set()
    for cpu in selected:
        root = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        physical.add((int((root / 'physical_package_id').read_text()),
                      int((root / 'core_id').read_text())))
    if len(physical) > policy['maximum_physical_cores']:
        raise ValueError('Reference affinity exceeds half-core limit')
    if '--protocol' not in args.arguments:
        raise ValueError('A finite registered reference protocol is required')
    position = args.arguments.index('--protocol')
    protocol = json.loads(Path(args.arguments[position + 1]).read_text())
    if protocol['host_cpu'] not in selected or not protocol['maximum_wall_seconds'] > 0:
        raise ValueError('Target affinity or finite budget violates reference policy')
    os.sched_setaffinity(0, selected)
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS'):
        os.environ[name] = '1'
    sys.path.insert(0, str(args.target.resolve().parent))
    sys.argv = [str(args.target), *args.arguments]
    runpy.run_path(str(args.target), run_name='__main__')


if __name__ == '__main__':
    run()
