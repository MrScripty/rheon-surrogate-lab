"""Bound and reap the CPU-only qualification process using the Pumas runtime."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import environment, read, sha, write


def run(config_path, project, output, execution):
    if output.exists() or execution.exists():
        raise ValueError('Fresh qualification and lifecycle receipts required')
    config = read(config_path)
    runtime = read(ROOT/config['runtime'])
    available = int(next(row.split()[1] for row in Path('/proc/meminfo').read_text().splitlines() if row.startswith('MemAvailable:')))*1024
    if available < 14*1024**3:
        raise MemoryError('12 GiB reserve plus 2 GiB qualification required')
    command = [runtime['python'], '-B', str(Path(__file__).with_name('qualify.py')),
        '--config', str(config_path), '--project', str(project), '--output', str(output)]
    process = None
    started, status = time.perf_counter(), 'failed'
    def stop(*_):
        raise KeyboardInterrupt('Qualification cancelled')
    signal.signal(signal.SIGTERM, stop)
    try:
        with execution.with_suffix('.log').open('xb') as log:
            process = subprocess.Popen(command, env=environment(runtime), stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
            if process.wait(timeout=120):
                raise RuntimeError('CPU qualification failed; inspect retained log')
            status = 'passed'
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
        write(execution, dict(status=status, elapsed_s=time.perf_counter()-started,
            command=command, pid=process.pid if process else None,
            returncode=process.returncode if process else None,
            reaped=process is not None and process.poll() is not None,
            available_host_bytes_before=available, source_sha256=sha(__file__),
            runtime_manifest_sha256=sha(ROOT/config['runtime'])))
    print('CPU qualification', status, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'project', 'output', 'execution'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.config.resolve(), args.project.resolve(), args.output.resolve(), args.execution.resolve())
