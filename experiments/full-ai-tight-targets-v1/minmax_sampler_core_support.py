"""Validate prospective CPU allocation and registered immutable source evidence."""
import os
from profile_frozen_components import ROOT, read, digest
from recurrent_sampler_minmax_v12 import configure
import torch


def prepare(reg):
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('Registered core experiment source changed: ' + path)
    cores = reg['cores']
    if cores not in ([23], [6], [6, 4], [6, 4, 5, 7]):
        raise ValueError('Unregistered physical CPU allocation')
    expected = dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                    OMP_DYNAMIC='FALSE', OMP_PROC_BIND='TRUE',
                    OMP_PLACES=','.join('{' + str(cpu) + '}' for cpu in cores),
                    OMP_THREAD_LIMIT=str(len(cores)), OMP_WAIT_POLICY='PASSIVE',
                    GOMP_SPINCOUNT='0', RAYON_NUM_THREADS='1')
    if any(os.environ.get(key) != value for key, value in expected.items()):
        raise ValueError('Launch through the registered fresh-process resource wrapper')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    team = configure(cores)
    # Master remains pinned to the same physical CPU in serial and parallel paths.
    if os.sched_getaffinity(0) != {cores[0]}:
        raise ValueError('OpenMP master is not pinned to its declared physical core')
    team.update(master_affinity=sorted(os.sched_getaffinity(0)),
                environment=expected, torch_threads=torch.get_num_threads(),
                torch_interop_threads=torch.get_num_interop_threads())
    return team
