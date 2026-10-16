"""Own bounded, non-overwriting frozen-policy CPU phases and retain every attempt."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from queue import Queue
import shutil
import sys
from threading import Event
import time

LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB/'experiments/water-shape-repeatability-v1'))
from execution import command, read, write, utc
sys.path.insert(0, str(LAB/'experiments/water-inference-budget-v1'))
from scaling import summarize
sys.path.insert(0, str(LAB/'experiments/live-project-v1'))
from observer import Observer


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def available_memory():
    return int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                    if x.startswith('MemAvailable:')))*1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--phase', choices=('fixture', 'development', 'timing', 'validation'), required=True)
    parser.add_argument('--attempt', help='Explicit fresh attempt label; earlier outputs are retained')
    args = parser.parse_args()
    config = read(args.config)
    if config['schema'] != 'rheon-stopping-floor-study-v1':
        raise ValueError('stopping-floor configuration required')
    root = LAB/config['root']
    root.mkdir(parents=True, exist_ok=True)
    assert available_memory() >= config['available_memory_reserve_bytes']
    assert shutil.disk_usage(root).free >= 2*config['max_output_bytes']
    if args.phase != 'fixture':
        assert read(LAB/config['fixture_report'])['all_qualified'], 'qualify one real fixture before expanding'
    if args.phase == 'timing':
        assert read(root/'development/report.json')['all_qualified']
    if args.phase == 'validation':
        selection = read(root/'selection.json')
        assert selection['consumed_acceptance_passed'] and digest(args.config) == selection['validation_config_sha256']
    if args.attempt and not args.attempt.replace('-', '').isalnum():
        raise ValueError('simple fresh attempt label required')
    directory = root/(args.phase+('-'+args.attempt if args.attempt else ''))
    directory.mkdir(exist_ok=False)
    cpus = sorted(os.sched_getaffinity(0))[1:] or sorted(os.sched_getaffinity(0))
    workers = 1 if args.phase in ('fixture', 'timing') else config['workers']
    assert 1 <= workers <= min(2, len(cpus))
    leased = Queue()
    for cpu in cpus[:workers]:
        leased.put(cpu)
    cancel = Event()
    observer = Observer(LAB/'runs/live-project')
    entry = LAB/'experiments/water-stopping-floor-v1/episode.py'
    started = time.monotonic()
    report = {'phase': args.phase, 'configuration': config, 'config_sha256': digest(args.config),
        'source_sha256': {str(p.relative_to(LAB)): digest(p) for p in (Path(__file__).resolve(), entry,
            LAB/'experiments/water-stability-learning-v1/learned_iteration.py',
            LAB/'experiments/water-residual-learning-v1/rollout.py')},
        'started_at_utc': utc(), 'workers': workers, 'records': [], 'failures': [],
        'models': [{**m, 'weights_sha256': digest(LAB/m['artifacts']/'weights.npz'),
                    'normalization_sha256': digest(LAB/m['artifacts']/'normalization.json')} for m in config['models']],
        'binary_sha256': digest(LAB/config['binary']), 'optimizer_updates': 0, 'cuda_initialized': False,
        'timing_scope': 'serial uninstrumented repeated episodes; ordinary desktop services remain' if args.phase == 'timing'
                        else 'parallel/instrumented correctness; no isolated latency claim'}

    def save():
        report.update(elapsed_seconds=time.monotonic()-started, updated_at_utc=utc())
        write(directory/'report.json', report)

    def execute(task):
        scene, model, repetition = task
        label = f'scene-{scene:02d}-{model}-{repetition}'
        output = directory/label
        argv = [config['python'], entry, '--config', args.config.absolute(), '--scene', str(scene),
                '--model', model, '--output', output]
        if args.phase == 'fixture':
            argv += ['--fixture']
        if args.phase == 'timing':
            argv += ['--uninstrumented']
        cpu = leased.get()
        try:
            try:
                timing = command(['taskset', '-c', str(cpu), *argv], directory, label, cancel=cancel)
                error = None
            except Exception as failure:
                timing = read(directory/(label+'-timing.json'))
                error = type(failure).__name__
            result = {'scene': scene, 'model': model, 'repetition': repetition, 'cpu': cpu,
                      'output': str(output.relative_to(LAB)), 'timing': timing, 'error': error}
            native = output/'native'
            if (native/'receipt.json').exists():
                result.update(summarize(native))
            callbacks=native/'callbacks.jsonl'
            result['all_declared_stopping_checks_passed'] = model=='native' or callbacks.exists() and all(
                json.loads(line)['trace'][-1]['equation_passed'] for line in callbacks.read_text().splitlines())
            result['qualified'] = (error is None and result.get('status') == 'completed'
                and not result['result']['hard_stop'] and result['all_ai_steps_qualified']
                and result['all_reference_steps_qualified'] and result['all_declared_stopping_checks_passed'])
            return result
        finally:
            leased.put(cpu)

    def perform(tasks):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(execute, task) for task in tasks]
            try:
                for future in futures:
                    row = future.result()
                    report['records'].append(row)
                    if not row['qualified']:
                        report['failures'].append({k: row[k] for k in ('scene', 'model', 'error')})
                    save()
                    observer.publish('Stopping comparison · '+args.phase,
                        f"{row['model']}, scene{row['scene']}; qualified={row['qualified']}", len(report['records']))
                    print(json.dumps({k: row[k] for k in ('scene', 'model', 'qualified', 'error')}), flush=True)
            except BaseException:
                cancel.set()
                for future in futures:
                    future.cancel()
                raise

    try:
        scenes = [0] if args.phase == 'fixture' else ([config['serial_timing_scene']] if args.phase == 'timing' else range(len(config['scenes'])))
        perform([(scene, 'native', 0) for scene in scenes])
        assert not report['failures'], 'native setup/control failed before AI comparison'
        tasks = []
        repetitions = config['serial_timing_repetitions'] if args.phase == 'timing' else 1
        for scene in scenes:
            for repetition in range(repetitions):
                models = config['models'] if repetition % 2 == 0 else config['models'][::-1]
                tasks.extend((scene, m['name'], repetition) for m in models)
        perform(tasks)
        report.update(state='completed', all_qualified=not report['failures'])
        assert sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) <= config['max_output_bytes']
    except BaseException as error:
        report.update(state='failed', all_qualified=False, error_type=type(error).__name__)
        raise
    finally:
        report['finished_at_utc'] = utc()
        save()
        observer.publish('Stopping comparison · '+args.phase, report['state'], len(report['records']),
                         state='completed' if report.get('state') == 'completed' else 'failed')


if __name__ == '__main__':
    main()
