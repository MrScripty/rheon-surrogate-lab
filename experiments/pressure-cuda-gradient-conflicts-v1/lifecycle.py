"""Cancel owned controllers at qualifier startup and after the first CUDA snapshot completes."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'experiments/pressure-four-pass-distillation-v1'))
from common import read, sha, write


def check(output):
    output.mkdir(parents=True, exist_ok=False)
    tests = []
    for name, marker, sent in [('startup', 'qualification-started.json', signal.SIGTERM),
                               ('workers', 'vectors/progress.json', signal.SIGINT)]:
        destination = output / name
        command = [sys.executable, '-B', str(HERE / 'run.py'), '--config',
                   str(HERE / 'config.json'), '--output', str(destination),
                   '--no-publish']
        started = time.perf_counter()
        with (output / (name + '.log')).open('xb') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 20
                while not (destination / marker).exists():
                    if process.poll() is not None:
                        raise RuntimeError('Controller exited before requested cancellation boundary')
                    if time.monotonic() > deadline:
                        raise TimeoutError('Cancellation boundary did not register')
                    time.sleep(.005)
                registered = read(destination / marker)
                if name == 'workers':
                    assert 100 in registered['completed_updates']
                process.send_signal(sent)
                process.wait(timeout=20)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=12)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        execution = read(destination / 'execution.json')
        assert process.returncode != 0 and execution['status'] == 'cancelled'
        assert execution['all_owned_children_reaped']
        assert all(w['reaped'] for w in execution['workers'])
        tests.append(dict(boundary=name, signal=sent.name, command=command,
                          controller_reaped=process.poll() is not None,
                          returncode=process.returncode, elapsed_s=time.perf_counter()-started,
                          registered=registered, execution=execution))
    write(output / 'receipt.json', dict(status='passed', tests=tests,
                                       source_sha256=sha(__file__)))
    print('Passed startup and active-CUDA cancellation', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    check(parser.parse_args().output.resolve())
