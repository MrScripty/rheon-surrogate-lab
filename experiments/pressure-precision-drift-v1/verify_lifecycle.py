"""Cancel an actual precision probe through its owner and verify child reaping."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from build import sha


def verify(build, receipt):
    started = time.perf_counter()
    command = [sys.executable, str(Path(__file__).with_name('run.py')), '--build', str(build)]
    with tempfile.TemporaryDirectory(prefix='precision-lifecycle-') as temporary:
        output = Path(temporary) / 'cancelled-run'
        child_pids = []
        child_identity = {}
        with (Path(temporary) / 'owner.log').open('w+') as log:
            owner = subprocess.Popen(command + ['--output', str(output)], stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    if owner.poll() is not None:
                        raise RuntimeError('Owner exited before cancellation could exercise an active native child')
                    children = Path(f'/proc/{owner.pid}/task/{owner.pid}/children')
                    child_pids = [int(pid) for pid in children.read_text().split()]
                    if child_pids and (output / 'fixture/frames.jsonl').exists():
                        break
                    time.sleep(.005)
                else:
                    raise TimeoutError('Actual native fixture did not start')
                native = json.loads((build / 'build-receipt.json').read_text())
                for pid in child_pids:
                    if sha(Path(f'/proc/{pid}/exe')) != native['sha256']:
                        raise AssertionError('Cancellation target is not the measured native precision probe')
                    child_identity[pid] = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
                owner.send_signal(signal.SIGTERM)
                owner.wait(timeout=10)
                execution = json.loads((output / 'execution.json').read_text())
                assert execution['status'] == 'cancelled', execution
                assert execution['all_owned_children_reaped']
                assert all(not Path(f'/proc/{pid}').exists() for pid in child_pids)
                result = dict(status='passed', scenario='SIGTERM owner while actual native fixture runs',
                              elapsed_s=time.perf_counter()-started, native_binary_sha256=native['sha256'],
                              source_sha256=sha(__file__), owner_returncode=owner.returncode,
                              observed_native_child_pids=child_pids, execution=execution,
                              all_observed_children_absent=True, command=command + ['--output', '<temporary fresh directory>'])
                receipt.write_text(json.dumps(result, indent=2) + '\n')
                print(json.dumps(dict(status='passed', elapsed_s=result['elapsed_s'], children_reaped=len(child_pids))))
            finally:
                if owner.poll() is None:
                    owner.send_signal(signal.SIGTERM)
                    try:
                        owner.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        owner.kill()
                        owner.wait()
                # Only process groups observed as our owner's children are eligible.
                for pid in child_pids:
                    try:
                        identity = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
                        if identity == child_identity.get(pid):
                            os.killpg(pid, signal.SIGKILL)
                    except (FileNotFoundError, ProcessLookupError):
                        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    verify(args.build.resolve(), args.receipt.resolve())
