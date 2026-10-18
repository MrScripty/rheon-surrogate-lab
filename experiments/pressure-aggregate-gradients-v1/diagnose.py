"""Own a bounded pool of three one-core CPU diagnostics and reap every child."""
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


def run(config_path, output, results):
    output.mkdir(parents=True, exist_ok=False)
    results.mkdir(parents=True, exist_ok=True)
    config = read(config_path)
    runtime = read(ROOT/config['runtime'])
    available = int(next(row.split()[1] for row in Path('/proc/meminfo').read_text().splitlines() if row.startswith('MemAvailable:')))*1024
    if available < 18*1024**3:
        raise MemoryError('12 GiB reserve plus three 2 GiB workers required')
    cpus = sorted(os.sched_getaffinity(0))[-3:]
    if len(cpus) != len(config['models']) or len(cpus) != 3:
        raise ValueError('Exactly three distinct CPU workers required')
    started, jobs, status = time.perf_counter(), [], 'failed'
    def stop(*_):
        raise KeyboardInterrupt('Aggregate diagnosis cancelled')
    signal.signal(signal.SIGTERM, stop)
    try:
        for model, cpu in zip(config['models'], cpus, strict=True):
            command = [runtime['python'], '-B', str(Path(__file__).with_name('measure.py')),
                '--config', str(config_path), '--model', model, '--cpu', str(cpu),
                '--output', str(results/(model+'-gradients.json')), '--vectors', str(output/(model+'-vectors'))]
            log = (output/(model+'.log')).open('xb')
            try:
                process = subprocess.Popen(command, env=environment(runtime), stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
            except BaseException:
                log.close(); raise
            jobs.append(dict(model=model, cpu=cpu, process=process, log=log, command=command))
        write(output/'started.json', dict(workers=[dict(model=job['model'], pid=job['process'].pid) for job in jobs]))
        for job in jobs:
            if job['process'].wait(timeout=max(.1, 120-(time.perf_counter()-started))):
                raise RuntimeError('CPU gradient worker failed; retained logs name the boundary')
        status = 'completed'
    finally:
        # Terminate every group before waiting; three stubborn workers share one
        # ten-second grace period instead of extending cancellation to thirty.
        for job in jobs:
            process = job['process']
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic()+10
        for job in jobs:
            process = job['process']
            if process.poll() is None:
                try:
                    process.wait(timeout=max(.1, deadline-time.monotonic()))
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait()
            job['log'].close()
        write(output/'execution.json', dict(status=status, elapsed_s=time.perf_counter()-started,
            workers=[dict(model=job['model'], cpu=job['cpu'], command=job['command'], pid=job['process'].pid,
                returncode=job['process'].returncode, reaped=job['process'].poll() is not None) for job in jobs],
            available_host_bytes_before=available, optimizer_updates=0, config_sha256=sha(config_path),
            runtime_manifest_sha256=sha(ROOT/config['runtime']), source_sha256=sha(__file__)))
    print(status, 'CPU aggregate diagnosis', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'output', 'results'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.config.resolve(), args.output.resolve(), args.results.resolve())
