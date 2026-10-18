"""Serial one-core snapshot measurements after managed/readback work drains."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time

from common import ROOT, HERE, environment, read, write
from cpu import material


def isolate(args):
    config = read(args.config)
    runtime = read(ROOT / config['runtime'])
    args.output.mkdir(exist_ok=False)
    started = time.perf_counter()
    child = None
    rows = []
    status = 'failed'
    def stop(*_):
        raise KeyboardInterrupt('Isolated measurement owner cancelled')
    signal.signal(signal.SIGTERM, stop)
    try:
        requirements = [args.projects / 'candidate-cpu-01/execution.json']
        for arm in config['arms']:
            directory = args.projects / (arm['id'] + '-study-01')
            requirements.append(directory / 'receipt.json')
            journal = directory / 'mutations/13-admit-restore.json'
            if journal.exists() and read(journal).get('reconciliation'):
                requirements.append(directory / 'recovery/receipt.json')
        for path in requirements:
            while not path.exists():
                if time.perf_counter() - started > 1800:
                    raise TimeoutError('No isolation while managed readback remains unfinished')
                time.sleep(2)
            if read(path)['status'] != 'completed':
                raise ValueError('Prior work incomplete: ' + str(path))
        frozen = args.output / 'material.json'
        material(args.config, args.projects, frozen, True)
        cpu = max(os.sched_getaffinity(0))
        measurement_start = time.perf_counter()
        for identifier in ['C53', 'C55', 'C57', 'C58', 'C59']:
            command = [runtime['python'], '-B', str(HERE / 'cpu.py'), 'snapshots',
                '--material', str(frozen), '--id', identifier, '--cpu', str(cpu),
                '--output', str(args.output / identifier)]
            tick = time.perf_counter()
            with (args.output / (identifier + '.log')).open('xb') as log:
                child = subprocess.Popen(command, env=environment(runtime), stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
                code = child.wait(timeout=180)
            if code:
                raise RuntimeError('Isolated measurement failed: ' + identifier)
            rows.append(dict(model=identifier, command=command,
                elapsed_s=time.perf_counter() - tick, reaped=True))
            child = None
            print('Isolated CPU complete', identifier, flush=True)
        status = 'completed'
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        write(args.output / 'execution.json', dict(status=status,
            elapsed_including_wait_s=time.perf_counter() - started,
            measurement_elapsed_s=time.perf_counter() - measurement_start if 'measurement_start' in locals() else None,
            rows=rows, all_owned_children_reaped=child is None or child.poll() is not None,
            environment={k: v for k, v in environment(runtime).items() if k in [
                'PYTHONPATH', 'CUDA_VISIBLE_DEVICES', 'TULDOK_TORCH_DEVICE', 'OMP_NUM_THREADS',
                'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS']},
            policy='Serial one-core inference after managed training/restores/readback and concurrent CPU qualification complete'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['config', 'projects', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    isolate(args)
