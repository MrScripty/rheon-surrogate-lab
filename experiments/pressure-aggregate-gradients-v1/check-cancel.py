"""Cancel the actual numerical pool after its three owned children start."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read, sha, write


def check(config, scratch, output):
    scratch.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-B', str(Path(__file__).with_name('diagnose.py')),
        '--config', str(config), '--output', str(scratch/'pool'), '--results', str(scratch/'results')]
    started, complete = time.monotonic(), False
    with (scratch/'parent.log').open('xb') as log:
        parent = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while not (scratch/'pool/started.json').exists():
                if parent.poll() is not None or time.monotonic()-started > 10:
                    raise RuntimeError('Actual pool did not reach three-worker readiness')
                time.sleep(.01)
            children = read(scratch/'pool/started.json')['workers']
            assert len(children) == 3
            parent.send_signal(signal.SIGTERM)
            assert parent.wait(timeout=15) != 0
            receipt = read(scratch/'pool/execution.json')
            assert receipt['status'] == 'failed'
            assert all(row['reaped'] and row['returncode'] == -signal.SIGTERM for row in receipt['workers'])
            assert not list((scratch/'results').glob('*.json'))
            write(output, dict(status='passed', actual_numerical_workers=True,
                child_returncodes=[row['returncode'] for row in receipt['workers']],
                all_children_reaped=True, parent_reaped=True, partial_results_not_published=True,
                elapsed_s=time.monotonic()-started, source_sha256=sha(__file__),
                supervisor_source_sha256=sha(Path(__file__).with_name('diagnose.py'))))
            complete = True
        finally:
            if parent.poll() is None:
                parent.send_signal(signal.SIGTERM)
                try:
                    parent.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(parent.pid, signal.SIGKILL); parent.wait()
            if not complete:
                raise RuntimeError('Cancellation qualification incomplete; inspect owned pool receipt')
    print('Actual three-worker pool cancelled and reaped', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'scratch', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    check(args.config.resolve(), args.scratch.resolve(), args.output.resolve())
