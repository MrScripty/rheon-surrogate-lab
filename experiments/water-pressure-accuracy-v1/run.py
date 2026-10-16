"""Bounded independent CPU trajectories, followed by same-input paired replays."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from queue import Queue
import shutil
import sys
from threading import Event
import time

from diagnose import LAB, digest, read, write
sys.path.insert(0, str(LAB/'experiments/water-shape-repeatability-v1'))
from execution import command, utc


def execute_tasks(tasks, root, cpus, workers):
    """Keep cancellation and each active child/core lease within this invocation."""
    cancel = Event()
    available = Queue()
    for cpu in cpus:
        available.put(cpu)

    def execute(item):
        name, argv = item
        if cancel.is_set():
            raise InterruptedError('study cancelled before child startup')
        cpu = available.get()
        try:
            return command(['taskset', '-c', str(cpu), *argv], root, name, cancel=cancel)
        finally:
            available.put(cpu)

    interruption = None
    with ThreadPoolExecutor(max_workers=min(workers, len(cpus))) as pool:
        futures = [pool.submit(execute, task) for task in tasks]
        try:
            for future in futures:
                try:
                    future.result()
                except Exception:
                    pass  # Every terminal outcome is collected below.
        except BaseException as error:
            interruption = error
            cancel.set()
            for future in futures:
                future.cancel()
    results, failures = [], []
    for (name, _), future in zip(tasks, futures):
        try:
            results.append({'name': name, 'timing': future.result()})
        except BaseException as error:
            failures.append({'name': name, 'error': type(error).__name__})
    return results, failures, interruption


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--phase', choices=('trajectories', 'replays'), required=True)
    args = parser.parse_args()
    config = read(args.config)
    root = LAB/config['root']
    cpus = sorted(os.sched_getaffinity(0))
    started = time.perf_counter()
    started_utc = utc()
    assert shutil.disk_usage(root).free >= config['max_output_bytes']+config['free_reserve_bytes']
    for model in config['models'].values():
        receipt = read(LAB/model['artifacts']/'receipt.json')
        assert digest(LAB/model['artifacts']/'weights.npz') == receipt['outputs']['weights.npz']['sha256']
    for witness in config['witnesses']:
        assert digest(LAB/witness['historical_frames']) == witness['historical_frames_sha256']
    tasks = []
    entry = LAB/'experiments/water-pressure-accuracy-v1/diagnose.py'
    if args.phase == 'trajectories':
        for index, witness in enumerate(config['witnesses']):
            for model in config['models']:
                for policy in config['policies']:
                    name = f'{witness["name"]}-{model}-{policy}'
                    tasks.append((name, [config['python'], entry, 'capture', '--config', args.config.absolute(),
                        '--model', model, '--witness', str(index), '--policy', policy, '--output', root/name]))
    else:
        for witness in config['witnesses']:
            for source in config['models']:
                capture = read(root/f'{witness["name"]}-{source}-original/capture.json')
                for sample in capture['samples']:
                    for model in config['models']:
                        name = f'replay-{witness["name"]}-{source}-{sample["step"]:03d}-{model}'
                        tasks.append((name, [config['python'], entry, 'replay', '--config', args.config.absolute(),
                            '--model', model, '--sample', LAB/sample['path'], '--output', root/(name+'.json')]))
    workers = min(config['workers'], len(cpus), len(tasks))

    results, failures, interruption = execute_tasks(tasks, root, cpus, workers)
    size = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    result = {'phase': args.phase, 'started_at_utc': started_utc, 'finished_at_utc': utc(),
        'elapsed_seconds': time.perf_counter()-started, 'tasks': len(tasks), 'workers': workers,
        'available_cpus': cpus, 'records': results, 'failures': failures,
        'interrupted': interruption is not None,
        'local_bytes': size, 'source_sha256': digest(__file__), 'config_sha256': digest(args.config)}
    write(root/(args.phase+'-execution.json'), result)
    print({k: result[k] for k in ('phase', 'tasks', 'workers', 'elapsed_seconds', 'failures', 'local_bytes')})
    if interruption is not None:
        raise interruption
    assert not failures and size <= config['max_output_bytes']


if __name__ == '__main__':
    main()
