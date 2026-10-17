"""Measure complete, independent CPU pressure episodes at bounded concurrency.

The existing rollout owns simulation, model inference and native-child cleanup.
No optimizer, holdout selection or numerical policy changes occur here.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from queue import Queue
import resource
import signal
import subprocess
import sys
from threading import Event, Lock, Thread
import time

LAB = Path(__file__).resolve().parents[2]
GIB = 1024 ** 3


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def numerical_frames(frames):
    """Exclude clocks and fresh request identity; preserve every physical value."""
    result = []
    for frame in frames:
        value = {key: item for key, item in frame.items() if key not in ('ai_seconds', 'reference_seconds')}
        if value.get('diagnostic_attempt'):
            value['diagnostic_attempt'] = {key: item for key, item in value['diagnostic_attempt'].items()
                                           if key != 'request_sha256'}
        result.append(value)
    return result


def numerical_signature(path):
    frames = [json.loads(line) for line in Path(path).read_text().splitlines()]
    return hashlib.sha256(json.dumps(numerical_frames(frames), sort_keys=True, allow_nan=False).encode()).hexdigest()


def available():
    return int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                    if line.startswith('MemAvailable:'))) * 1024


def rss_tree(pid):
    """Conservative RSS sum; shared pages may be counted more than once."""
    total, pending, seen = 0, [pid], set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            status = Path(f'/proc/{current}/status').read_text().splitlines()
            total += int(next(line.split()[1] for line in status if line.startswith('VmRSS:'))) * 1024
            pending.extend(map(int, Path(f'/proc/{current}/task/{current}/children').read_text().split()))
        except (OSError, StopIteration):
            pass
    return total


def worker(args):
    def terminate(*_):
        raise KeyboardInterrupt('benchmark owner requested cancellation')
    signal.signal(signal.SIGTERM, terminate)
    sys.path.insert(0, str(LAB / 'experiments/water-residual-learning-v1'))
    import rollout
    config = read(args.config)
    started = time.perf_counter()
    scene = dict(config['scenes'][args.scene])
    scene['steps'] = min(scene['steps'], scene['warmup_steps'] + config['ai_steps'])
    if config.get('qualify_full_episode'):
        import episode
        sys.argv=[sys.argv[0],'--root',config['root'],'--bindings',config['bindings'],'--binary',config['binary'],
            '--manifest',config['manifest'],'--output',str(args.output),'--scene',str(args.scene)]
        episode.main()
        result=read(args.output/'episode.json')
        assert result['status']=='passed'
        frames=[json.loads(line) for line in (args.output/'native/frames.jsonl').read_text().splitlines()]
        callbacks=[json.loads(line) for line in (args.output/'native/callbacks.jsonl').read_text().splitlines()]
        write(args.output/'receipt.json',{'scene':args.scene,'signature':numerical_signature(args.output/'native/frames.jsonl'),
            'network_calls':[row['network_passes'] for row in callbacks],'metrics':result['metrics'],'sparse':result['sparse'],
            'steps':len(frames),'ai_steps':len(callbacks),'elapsed_seconds':time.perf_counter()-started,
            'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'native_reaped':True,
            'cuda_initialized':False,'cpu_affinity':sorted(os.sched_getaffinity(0))})
        return
    result = rollout.run(Path(config['binary']), scene, args.output / 'native',
        adapter_manifest=Path(config['manifest']), timeout=120, threads=1)
    assert result['status'] == 'completed' and result['native_process']['reaped']
    assert not result['result']['hard_stop'] and not result['cuda_initialized']
    frames = [json.loads(line) for line in (args.output / 'native/frames.jsonl').read_text().splitlines()]
    stable = numerical_frames(frames)
    callbacks = [json.loads(line) for line in (args.output / 'native/callbacks.jsonl').read_text().splitlines()]
    assert all((frame.get('ai_quality') or {}).get('qualified', True) for frame in frames)
    write(args.output / 'receipt.json', {'scene': args.scene,
        'signature': hashlib.sha256(json.dumps(stable, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        'network_calls': [row['network_passes'] for row in callbacks],
        'steps': len(frames), 'ai_steps': len(callbacks), 'elapsed_seconds': time.perf_counter() - started,
        'adapter_loading_seconds': result['adapter_loading_seconds'], 'native_elapsed_seconds': result['elapsed_s'],
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'native_reaped': True, 'cuda_initialized': False, 'cpu_affinity': sorted(os.sched_getaffinity(0))})


def benchmark(args):
    config = read(args.config)
    args.output.mkdir(parents=True, exist_ok=False)
    cpus = [cpu for cpu in sorted(os.sched_getaffinity(0))[config.get('reserved_cpu_count',2):]
            if cpu not in config.get('exclude_cpus',[])]
    assert args.workers <= len(cpus) and args.workers * config['worker_memory_bytes'] + config['reserve_bytes'] <= available()
    assert len(config['scene_indices']) >= args.workers
    runtime = read(config['runtime_manifest'])
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': runtime['site_packages'],
        'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'RAYON_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': '', 'TULDOK_TORCH_DEVICE': 'cpu'}
    leases = Queue()
    for cpu in cpus[:args.workers]:
        leases.put(cpu)
    stop, monitor_done, lock = Event(), Event(), Lock()
    live, peak = {}, {'rss_sum_bytes': 0, 'maximum_live_workers': 0, 'minimum_available_bytes': available()}

    def monitor():
        while not monitor_done.wait(.1):
            with lock:
                pids = list(live)
            peak['rss_sum_bytes'] = max(peak['rss_sum_bytes'], sum(rss_tree(pid) for pid in pids))
            peak['maximum_live_workers'] = max(peak['maximum_live_workers'], len(pids))
            peak['minimum_available_bytes'] = min(peak['minimum_available_bytes'], available())
            if peak['minimum_available_bytes'] < config['reserve_bytes']:
                stop.set()

    def execute(task):
        slot, scene = task
        cpu = leases.get()
        child = None
        output = args.output / f'case-{slot:02d}-scene-{scene:02d}'
        started = time.perf_counter()
        try:
            if stop.is_set() or available() < config['reserve_bytes'] + config['worker_memory_bytes']:
                raise MemoryError('CPU benchmark memory reserve reached')
            command = ['taskset', '-c', str(cpu), runtime['python'], '-B', str(Path(__file__).resolve()),
                '--config', str(args.config.resolve()), '--output', str(output.resolve()), '--scene', str(scene)]
            with (args.output / f'case-{slot:02d}.log').open('xb') as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                with lock:
                    live[child.pid] = child
                while child.poll() is None:
                    if stop.is_set() or time.perf_counter() - started > config.get('worker_timeout_seconds',240):
                        raise TimeoutError('CPU benchmark cancelled or timed out')
                    time.sleep(.05)
                if child.returncode:
                    raise RuntimeError(f'CPU benchmark worker exited {child.returncode}; inspect case-{slot:02d}.log')
            result = read(output / 'receipt.json')
            return {**result, 'slot': slot, 'cpu': cpu, 'complete_child_elapsed_seconds': time.perf_counter() - started,
                    'command': command, 'worker_reaped': True}
        except BaseException:
            stop.set()
            raise
        finally:
            if child is not None:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=5)
                with lock:
                    live.pop(child.pid, None)
            leases.put(cpu)

    started = time.perf_counter()
    report = {'schema': 'pressure-cpu-throughput-v1', 'workers': args.workers,
        'numerical_signature_contract': 'physical-frames-without-clocks-or-request-identity-v1',
        'config_sha256': sha(args.config), 'source_sha256': sha(__file__),
        'manifest_sha256': sha(config['manifest']), 'binary_sha256': sha(config['binary']),
        'runtime_manifest_sha256': sha(config['runtime_manifest']), 'optimizer_updates': 0,
        'threads_per_worker': 1, 'records': [], 'scope': 'complete pressure episodes including fresh model startup and verification'}
    monitor_thread = Thread(target=monitor, name='pressure-memory-monitor')
    monitor_thread.start()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(execute, enumerate(config['scene_indices'])):
                report['records'].append(result)
                print(json.dumps({'workers': args.workers, 'completed': len(report['records']), 'scene': result['scene']}), flush=True)
        if args.baseline:
            baseline = read(args.baseline)
            assert baseline['status'] == 'passed' and baseline['config_sha256'] == report['config_sha256']
            for before, after in zip(baseline['records'], report['records'], strict=True):
                recorded = args.baseline.parent / f"case-{before['slot']:02d}-scene-{before['scene']:02d}/native/frames.jsonl"
                assert (before['scene'], numerical_signature(recorded), before['network_calls']) == (after['scene'], after['signature'], after['network_calls'])
            report['exact_to_baseline'] = True
        report['status'] = 'passed'
    except BaseException as error:
        stop.set()
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        monitor_done.set()
        monitor_thread.join(timeout=5)
        report.update(elapsed_seconds=time.perf_counter() - started, memory=peak, all_worker_children_reaped=not live)
        write(args.output / 'receipt.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--scene', type=int)
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args()
    if args.scene is not None:
        worker(args)
    else:
        if args.workers is None or args.workers < 1:
            raise ValueError('A positive worker count is required')
        benchmark(args)


if __name__ == '__main__':
    main()
