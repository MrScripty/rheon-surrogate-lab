"""Same bounded native protocol with an explicit physical CPU allowance."""
import os
from pathlib import Path
import subprocess
from bounded_native_client import BoundedNativeClient
from study_support import BASE, read, sha


class ResourceNativeClient(BoundedNativeClient):
    cores = None

    def __init__(self, scene, directory, build=None, *, io_timeout_s=60.):
        cores = self.cores
        if cores not in ([6], [6, 4], [6, 4, 5, 7]):
            raise ValueError('Registered native CPU allocation required')
        self._incoming = bytearray()
        self.io_timeout_s = io_timeout_s
        self.last_custody = None
        if not 0 < io_timeout_s <= 120:
            raise ValueError('Bounded native packet deadline required')
        self.scene = scene
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.build = build or BASE / 'temporal-build-02'
        receipt = read(self.build / 'build-receipt.json')
        binary = self.build / 'rheon-temporal-probe'
        if receipt['status'] != 'qualified_build' or sha(binary) != receipt['binary_sha256']:
            raise ValueError('Qualified native binary required')
        self.log = (self.directory / 'native-stderr.log').open('xb')
        try:
            # taskset resets the child's inherited master-only affinity before
            # the Rust runtime starts its Rayon workers. No preexec Python runs.
            self.child = subprocess.Popen(
                ['/usr/bin/taskset', '-c', ','.join(map(str, cores)), str(binary)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                env={**os.environ, 'RAYON_NUM_THREADS': str(len(cores))},
                start_new_session=True)
            self.write_json(scene)
            self.ready = self.read_json()
            if self.ready['kind'] != 'ready':
                raise ValueError('Native initialization did not acknowledge')
            affinities = {p.name: sorted(os.sched_getaffinity(int(p.name)))
                          for p in Path('/proc', str(self.child.pid), 'task').iterdir()}
            if not affinities or any(set(value) != set(cores) for value in affinities.values()):
                raise ValueError('Native process or workers escaped the declared affinity')
            self.resource_evidence = dict(cores=cores, rayon_num_threads=len(cores),
                                          native_task_affinities=affinities)
            self.last_request = None
            self.last_frame = None
        except BaseException:
            if hasattr(self, 'child'):
                self._reap(False)
            else:
                self.log.close()
            raise

    def _reap(self, graceful):
        result = super()._reap(graceful)
        result['resource_evidence'] = getattr(self, 'resource_evidence', None)
        return result


class ResourceForecastConsumer(ResourceNativeClient):
    def advance(self, *args, **kwargs):
        raise ValueError('Numerical advance is forbidden in the forecast consumer')
