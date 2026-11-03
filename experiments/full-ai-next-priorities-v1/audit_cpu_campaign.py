"""Independently recompute complete CPU campaign statistics from timed samples.

Run only after the campaign's numerical workers have completed and reaped.
This does not execute a model, select a seed, or establish accuracy.
"""
import argparse
import math
import statistics
from study_support import ROOT, read, write, sha


def run(args):
    if args.output.exists():
        raise ValueError('Fresh timing audit output required')
    paths = [args.registration, args.registry, args.bank, args.benchmark_source,
             args.campaign / 'execution.json', args.campaign / 'summary.json']
    registration, registry, bank = map(read, paths[:3])
    execution, summary = map(read, paths[4:])
    if execution['status'] != 'completed' or summary['status'] != 'completed' or execution.get('active') is not None:
        raise ValueError('Terminal complete campaign required')
    if not all(w['reaped'] and w['returncode'] == 0 for w in execution['workers']):
        raise ValueError('All actual numerical workers must succeed and reap')
    if summary['registration_sha256'] != sha(args.registration) or summary['source_sha256'] != sha(args.benchmark_source):
        raise ValueError('Actual registered timing implementation differs')
    if registration['source_sha256'] != sha(args.benchmark_source) or registration['model_registry_sha256'] != sha(args.registry):
        raise ValueError('Registered source or deployment registry differs')
    if summary['bank_sha256'] != sha(args.bank) or registration['bank_sha256'] != sha(args.bank):
        raise ValueError('Timed physical scene population differs')
    if summary['threads'] != 1 or summary['cpu'] != 23 or summary['batch_size'] != 1 or not summary['balanced_serial']:
        raise ValueError('Required CPU timing context differs')
    methods = {m['identifier']: m for m in registry['methods']}
    if set(methods) != set(registration['models']) or len(execution['workers']) != len(methods):
        raise ValueError('Every actual registered model and worker required')
    expected = {(name, variant) for name in methods for variant in ('full', 'tensor')}
    expected |= {(name, 'sequential' + str(k)) for name, m in methods.items()
                 if m['model']['horizons'] == 1 for k in registration['bundled_horizons']}
    primary = [name for name, m in methods.items() if m['model']['horizons'] == 1 and m['replication_ordinal'] == 1]
    if len(primary) != 1:
        raise ValueError('Exactly one frozen primary one-step control required')
    expected |= {(primary[0], 'sequential' + str(k)) for k in registration.get('extra_primary_sequential_horizons', [])}
    scenes = {}
    for row in bank['trajectories']:
        path = ROOT / row['receipt']
        if sha(path) != row['receipt_sha256']:
            raise ValueError('Timed scene receipt changed')
        scenes[row['identifier']] = read(path)['scene']
        paths.append(path)
    cases = {(name, step) for name in scenes for step in (0, 8, 16, 32, 48, 64, 96, 120)}
    if len(scenes) != 8 or len(cases) != 64:
        raise ValueError('Original eight-scene 64-window timing population required')
    for name, method in methods.items():
        path = args.campaign / (name + '-qualification.json')
        ready = read(path)
        if ready['cases'] != 64 or ready['model']['weights_sha256'] != method['model']['weights_sha256'] or ready['model']['application_policy'] != method['model']['policy']:
            raise ValueError('Timed actual weights or execution policy differs')
        paths.append(path)
    def equal(observed, expected_value):
        if not math.isfinite(observed) or not math.isclose(observed, expected_value, rel_tol=1e-12, abs_tol=1e-10):
            raise ValueError('Published timing statistic differs from raw samples')
    computed = {}
    total = 0
    for response in execution['completed']:
        path = ROOT / response['path']
        if sha(path) != response['sha256']:
            raise ValueError('Actual timing response changed')
        value = read(path)
        name, variant, repeat = value['identifier'], value['variant'], value['repeat']
        if (name, variant) not in expected or repeat not in range(5):
            raise ValueError('Unregistered model, scope or repeat')
        k = methods[name]['model']['horizons']
        steps = int(variant.removeprefix('sequential')) if variant.startswith('sequential') else k
        if value['model_horizons'] != k or value['horizons'] != steps or value['threads'] != 1 or value['affinity'] != [23] or value['cuda_initialized']:
            raise ValueError('Actual physical-step count or CPU execution context differs')
        rows = value['rows']
        if len(rows) != 64 or {(r['identifier'], r['step']) for r in rows} != cases:
            raise ValueError('Missing, repeated or substituted timing window')
        for row in rows:
            scene = scenes[row['identifier']]
            if row['counts_xyz'] != scene['counts'] or row['cells_per_metre'] != scene['cells_per_metre'] or row['generated_steps'] != steps:
                raise ValueError('Physical grid, spacing or generated-step count differs')
            if not math.isfinite(row['seconds']) or row['seconds'] <= 0:
                raise ValueError('Invalid measured elapsed time')
        if len(value['summaries']) != 2:
            raise ValueError('Exactly two declared spatial-tier summaries required')
        for regime in ('small', '32cubed'):
            seconds = sorted(r['seconds'] for r in rows if (max(r['counts_xyz']) <= 16) == (regime == 'small'))
            median = statistics.median(seconds)
            result = dict(cases=len(seconds), median_block_ms=median*1000,
                p95_block_ms=seconds[math.ceil(.95*len(seconds))-1]*1000, maximum_block_ms=max(seconds)*1000,
                median_ms_per_generated_step=median*1000/steps, model_seconds_per_physical_second=median*60/steps)
            observed = [s for s in value['summaries'] if s['regime'] == regime]
            if len(observed) != 1 or (name, variant, repeat, regime) in computed:
                raise ValueError('Duplicate/missing response summary or repeat')
            for key, expected_value in result.items():
                equal(observed[0][key], expected_value)
            computed[name, variant, repeat, regime] = result
        total += len(rows)
        paths.append(path)
    if len(execution['completed']) != len(expected)*5 or total != summary['timed_requests'] or total != len(expected)*5*64:
        raise ValueError('Incomplete full/tensor/sequential five-repeat campaign')
    seen = set()
    for row in summary['rows']:
        name, variant, regime = row['identifier'], row['variant'], row['regime']
        key = (name, variant, regime)
        if key in seen or (name, variant) not in expected or regime not in ('small', '32cubed') or row['repeats'] != 5:
            raise ValueError('Missing/duplicate campaign summary')
        steps = int(variant.removeprefix('sequential')) if variant.startswith('sequential') else methods[name]['model']['horizons']
        if row['horizons'] != steps:
            raise ValueError('Summary generated-step divisor differs from actual model/scope')
        seen.add(key)
        values = [computed[name, variant, repeat, regime] for repeat in range(5)]
        for metric in ('median_block_ms', 'median_ms_per_generated_step', 'p95_block_ms'):
            equal(row[metric], statistics.median(v[metric] for v in values))
        equal(row['maximum_block_ms'], max(v['maximum_block_ms'] for v in values))
        equal(row['min_repeat_median_ms'], min(v['median_block_ms'] for v in values))
        equal(row['max_repeat_median_ms'], max(v['median_block_ms'] for v in values))
    if seen != {(name, variant, regime) for name, variant in expected for regime in ('small', '32cubed')}:
        raise ValueError('Every model/scope/spatial tier required')
    write(args.output, dict(status='passed', raw_statistics_consistent=True, model_count=len(methods),
        timed_blocks_verified=total, responses_verified=len(execution['completed']), summary_rows_verified=len(seen),
        complete_physical_window_population=True, accuracy_claim=False, goal_status_updated=False,
        source_evidence={str(p.relative_to(ROOT)): sha(p) for p in paths}, auditor_sha256=sha(__file__)))
    print('Independent CPU timing audit passed:', total, 'blocks;', len(seen), 'summary rows', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('campaign', 'registry', 'bank', 'registration', 'benchmark-source', 'output'):
        parser.add_argument('--' + name, required=True, type=lambda value: ROOT / value)
    run(parser.parse_args())
