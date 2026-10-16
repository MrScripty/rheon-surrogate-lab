"""Run one controlled CUDA fit, development regressions and a frozen final test."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB/'experiments/water-shape-repeatability-v1'))
from execution import command, read, utc, write
sys.path.insert(0, str(LAB/'experiments/live-project-v1'))
from observer import Observer, Watch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def population(drift, studies, weights):
    """Aggregate retained native verdicts and actual calls by checkpoint identity."""
    cases = read(drift)['cases']
    reports = {digest(path/'report.json'): read(path/'report.json') for path in studies}
    for report in reports.values():
        controls = [row for row in report['records'] if row['model'] == 'native']
        if len(controls) != len(report['configuration']['scenes']) or not all(
                row['returncode'] == 0 and row['all_ai_steps_qualified']
                and row['all_reference_steps_qualified'] and not row['result']['hard_stop']
                for row in controls):
            raise ValueError('complete qualified native controls required')
    result = {}
    for name, sha in weights.items():
        selected = [case for case in cases if case['weights_sha256'] == sha]
        if not selected:
            raise ValueError('missing checkpoint population: '+name)
        calls = []
        qualified = []
        for case in selected:
            report = reports[case['source_report_sha256']]
            records = [r for r in report['records']
                       if r.get('receipt_sha256') == case['source_receipt_sha256']]
            if len(records) != 1 or len(records[0]['network_passes']) != case['ai_steps']:
                raise ValueError('complete per-step network call inventory required')
            row = records[0]
            qualified.append(case['completed_requested_window']
                and case['qualified_ai_steps'] == case['ai_steps']
                and row['returncode'] == 0 and row['all_reference_steps_qualified']
                and not row['result']['hard_stop'])
            calls.extend(row['network_passes'])
        result[name] = {
            'scenes': len(selected),
            'qualified_scenes': sum(qualified),
            'attempted_ai_steps': sum(c['ai_steps'] for c in selected),
            'qualified_ai_steps': sum(c['qualified_ai_steps'] for c in selected),
            'hard_stops': sum(c['hard_stop'] for c in selected),
            'network_evaluations': sum(calls),
            'mean_network_evaluations': statistics.mean(calls),
            'maximum_network_evaluations': max(calls), 'cap_hits': calls.count(32),
            'worst_pressure_relative_l2': max(c['worst_pressure_relative_l2'] for c in selected),
            'median_final_pressure_relative_l2': statistics.median(
                c['final_pressure_relative_l2'] for c in selected),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('sequence', 'reports', 'managed-controller'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--torch-python', required=True)
    args = parser.parse_args()
    sequence = read(args.sequence)
    root = LAB/sequence['root']
    if (root/'study-status.json').exists():
        raise FileExistsError('attempt already exists; reconcile without replay')
    trial, = sequence['trials']
    directory = root/trial['name']
    candidate = {'name': trial['name'], 'project': str((directory/'project').relative_to(LAB)),
                 'artifacts': str((directory/'managed-full/stage-01/artifacts').relative_to(LAB)),
                 'single_evaluation': True}
    tools = LAB/'experiments/water-multigrid-learning-v1'
    previous = LAB/'experiments/water-shape-repeatability-v1'
    started = time.monotonic()
    status = {'started_at_utc': utc(), 'state': 'running'}
    write(root/'study-status.json', status)
    observer = Observer(LAB/'runs/live-project')
    phase = ['Preparing managed CUDA training']

    def interrupt(_signal, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)

    def timed(argv, name):
        return command(argv, root, name)

    def study(config, name, models):
        value = read(config)
        value['models'] = models
        runtime = root/(name+'.study.json')
        if runtime.exists() or (root/name).exists():
            raise FileExistsError('study attempt already exists')
        write(runtime, value)
        phase[0] = name+' · native controls and independent AI tracks'
        timed([args.torch_python, tools/'study.py', '--config', runtime,
               '--output', root/name, '--workers', '8'], name)
        return root/name

    def summarize(name, studies):
        output = args.reports/(name+'-drift.json')
        timed([args.torch_python, tools/'trajectory_summary.py', '--studies', *studies,
               '--output', output], name+'-drift')
        return population(output, studies, weights)

    try:
        with Watch(lambda: observer.publish('Two-pass 10,000-update control', phase[0])):
            timed([sys.executable, previous/'run.py', '--sequence', args.sequence,
                   '--torch-python', args.torch_python,
                   '--managed-controller', args.managed_controller], 'managed-sequence')
            phase[0] = 'Sealing managed custody, CPU restore and development qualification'
            record = args.reports/('trial-'+trial['name']+'.json')
            timed([args.torch_python, tools/'record.py', '--trial', directory,
                   '--control', (LAB/sequence['reference']['project']).parent/'managed-full/stage-00/artifacts',
                   '--output', record], 'seal')
            timed([sys.executable, previous/'summarize.py', '--sequence', args.sequence,
                   '--reports', args.reports, '--reference-record', LAB/sequence['reference_record'],
                   '--reference-index', LAB/sequence['reference_index']], 'comparison')
            weights = {'reference': digest(LAB/sequence['reference']['artifacts']/'weights.npz'),
                       'candidate': digest(LAB/candidate['artifacts']/'weights.npz')}
            populations = {}
            for name, child in [('fresh','grid-study'), ('prior','prior-study'), ('long','long-study')]:
                populations[name] = summarize(name, [LAB/sequence['reference_development_studies'][name],
                                                      directory/child])
            for item in sequence['regressions']:
                output = study(LAB/item['config'], item['name'], [candidate])
                populations[item['name']] = summarize(item['name'], [LAB/item['reference_study'], output])
            development = ['fresh','prior','long','consumed-0','consumed-1']
            scores = {}
            for name in weights:
                groups = [populations[p][name] for p in development]
                fitting = read(record) if name == 'candidate' else read(LAB/sequence['reference_record'])
                eligible = all(g['qualified_scenes'] == g['scenes'] for g in groups) and all(
                    fitting[k]['native']['float32_qualified'] == fitting[k]['native']['cases']
                    for k in ('adaptive_fit','adaptive_development'))
                rank = [max(g['maximum_network_evaluations'] for g in groups),
                        max(g['worst_pressure_relative_l2'] for g in groups),
                        sum(g['network_evaluations'] for g in groups)/sum(g['attempted_ai_steps'] for g in groups)]
                scores[name] = {'eligible': eligible, 'rank': rank}
            eligible = [name for name in scores if scores[name]['eligible']]
            if not eligible:
                raise ValueError('qualified control unexpectedly failed development; replan')
            choice = min(eligible, key=lambda name: scores[name]['rank'])
            capture = read(root/'capture.json')['source_commit']
            original, = sequence['final_configs']
            committed = subprocess.check_output(['git','show',capture+':'+original], cwd=LAB)
            if committed != (LAB/original).read_bytes():
                raise ValueError('final scenes changed since pre-training capture')
            selection = {'frozen_at_utc': utc(), 'development_choice': choice, 'scores': scores,
                         'weights': weights, 'candidate': candidate, 'development': populations,
                         'final_config': original, 'final_config_sha256': digest(LAB/original),
                         'policy': sequence['selection'], 'no_final_informed_refit_or_reselection': True}
            with (args.reports/'final-selection.json').open('x') as stream:
                json.dump(selection, stream, indent=2, allow_nan=False)
                stream.write('\n')
            final = study(LAB/original, 'final-holdout', [sequence['reference'], candidate])
            populations['final'] = summarize('final', [final])
            if datetime.fromisoformat(read(final/'report.json')['started_at_utc']) <= datetime.fromisoformat(selection['frozen_at_utc']):
                raise ValueError('final test must follow frozen selection')
            for model in read(final/'report.json')['models']:
                key = 'candidate' if model['name'] == candidate['name'] else 'reference'
                if model['weights_sha256'] != weights[key]:
                    raise ValueError('checkpoint changed after selection')
            ref, new = populations['final']['reference'], populations['final']['candidate']
            promote = choice == 'candidate' and new['qualified_scenes'] == new['scenes'] and all(
                new[k] <= ref[k] for k in ('maximum_network_evaluations','worst_pressure_relative_l2',
                                           'mean_network_evaluations'))
            write(args.reports/'decision.json', {'recorded_at_utc': utc(), 'populations': populations,
                  'candidate_promoted': promote, 'retained_weights_sha256': weights['candidate' if promote else 'reference'],
                  'selection_sha256': digest(args.reports/'final-selection.json'),
                  'scope': 'CPU iterative laboratory qualification; deployment unchanged; final population now consumed'})
            status['state'] = 'completed'
    except BaseException as error:
        status.update(state='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                      error_type=type(error).__name__, recovery='Retain attempts and reconcile managed/native owners; no blind replay')
        raise
    finally:
        status['finished_at_utc'] = utc()
        status['elapsed_seconds'] = time.monotonic()-started
        write(root/'study-status.json', status)
        observer.publish('Two-pass experiment '+status['state'],
                         'Methods, timings, native verdicts and attempts retained', state='completed' if status['state']=='completed' else 'failed')


if __name__ == '__main__':
    main()
