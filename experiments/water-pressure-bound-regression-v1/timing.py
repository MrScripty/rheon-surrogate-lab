"""Repeat current reference/candidate episodes serially after broad work ends."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'bindings', 'configs', 'reports'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    protocol, bindings = read(args.protocol), read(args.bindings)
    root = LAB/protocol['root']
    development = read(root/'development/report.json')
    assert development.get('state') in ('completed', 'failed'), 'finish competing task work before serial timings'
    native = {r['scene']: r for r in development['records'] if r['model'] == 'native'}
    assert all(native[i]['qualified'] for i in protocol['timing_scene_indices'])
    directory = root/'serial-timing'
    directory.mkdir(exist_ok=False)
    cpu = (sorted(os.sched_getaffinity(0))[1:] or sorted(os.sched_getaffinity(0)))[0]
    report = {'state': 'running', 'started_at_utc': utc(), 'records': [], 'summaries': [],
        'source_sha256': digest(__file__), 'development_report_sha256': digest(root/'development/report.json'),
        'cpu': cpu, 'threads': 1, 'optimizer_updates': 0,
        'scope': 'current serial uninstrumented whole Engine.advance timing, one AI timing warm-up excluded; ordinary desktop services remain'}
    started = time.monotonic()
    observer = Observer(LAB/'runs/live-project')
    try:
        for scene in protocol['timing_scene_indices']:
            config_path = args.configs/f'timing-{scene:02d}.json'
            config = read(config_path)
            assert config['scenes'][scene] == development['configuration']['scenes'][scene]
            assert digest(LAB/config['binary']) == bindings['binary_sha256']
            for model in config['models']:
                frozen = bindings['models'][model['name']]
                assert digest(LAB/model['artifacts']/'weights.npz') == frozen['weights_sha256']
                assert digest(LAB/model['artifacts']/'normalization.json') == frozen['normalization_sha256']
                assert digest(LAB/model['project']/'source-witness.json') == frozen['source_witness_sha256']
                assert digest(LAB/model.get('stopping_profile', config['profile'])) == frozen['stopping_profile_sha256']
            for repetition in range(config['serial_timing_repetitions']):
                for model in (config['models'] if repetition % 2 == 0 else config['models'][::-1]):
                    label = f'scene-{scene:02d}-{model["name"]}-{repetition}'
                    output = directory/label
                    argv = ['taskset', '-c', str(cpu), config['python'],
                        LAB/'experiments/water-stopping-floor-v1/episode.py', '--config', config_path.absolute(),
                        '--scene', str(scene), '--model', model['name'], '--output', output, '--uninstrumented']
                    observer.publish('Pressure-bound regression · serial timing', label, len(report['records']), 12)
                    try:
                        timing = command(argv, directory, label)
                        error = None
                    except Exception as failure:
                        timing = read(directory/(label+'-timing.json'))
                        error = type(failure).__name__
                    metrics = summarize(output/'native') if (output/'native/receipt.json').exists() else {}
                    callbacks = output/'native/callbacks.jsonl'
                    stopping_passed = callbacks.exists() and all(json.loads(s)['trace'][-1]['equation_passed']
                        for s in callbacks.read_text().splitlines())
                    qualified = bool(error is None and metrics.get('status') == 'completed'
                        and not metrics['result']['hard_stop'] and metrics['all_ai_steps_qualified']
                        and metrics['all_reference_steps_qualified'] and stopping_passed)
                    report['records'].append({'scene': scene, 'model': model['name'], 'repetition': repetition,
                        'config_sha256': digest(config_path), 'timing': timing, 'error': error, 'qualified': qualified, **metrics})
                    write(directory/'report.json', report)
            for model in config['models']:
                rows = [r for r in report['records'] if r['scene'] == scene and r['model'] == model['name']]
                values = [v for row in rows for v in row.get('ai_seconds', [])]
                native_values = [v for row in rows for v in row.get('native_seconds', [])]
                report['summaries'].append({'scene': scene, 'counts_xyz': config['scenes'][scene]['counts'],
                    'model': model['name'], 'repetitions': len(rows), 'qualified': all(r['qualified'] for r in rows),
                    'ai_median_seconds': statistics.median(values) if values else None,
                    'native_track_median_seconds': statistics.median(native_values) if native_values else None,
                    'per_repetition_ai_median_seconds': [statistics.median(r['ai_seconds']) if r.get('ai_seconds') else None for r in rows],
                    'steady_timing_samples': len(values),
                    'mean_network_evaluations': statistics.mean(v for r in rows for v in r['network_passes']) if values else None})
        report.update(state='completed', all_qualified=all(r['qualified'] for r in report['records']))
    except BaseException as error:
        report.update(state='failed', error_type=type(error).__name__)
        raise
    finally:
        report.update(finished_at_utc=utc(), elapsed_seconds=time.monotonic()-started)
        write(directory/'report.json', report)
        write(args.reports/'serial-timing.json', report)
        observer.publish('Pressure-bound regression · serial timing', report['state'], len(report['records']), 12,
                         state='completed' if report['state'] == 'completed' else 'failed')
    print(json.dumps({'state': report['state'], 'elapsed_seconds': report['elapsed_seconds'], 'summaries': report['summaries']}, indent=2))


if __name__ == '__main__':
    main()
