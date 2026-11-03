"""Compare coverage and feedback separately through existing managed/native owners."""
import argparse
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import time

LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB/'experiments/water-shape-repeatability-v1'))
from execution import command, read, utc, write
spec = spec_from_file_location('retained_two_pass', LAB/'experiments/water-two-pass-long-v1/run.py')
retained = module_from_spec(spec)
spec.loader.exec_module(retained)
from observer import Observer, Watch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def references(sequence):
    """Bind reused measurements by checkpoint, exact scenes and native binary."""
    expected = digest(LAB/sequence['reference']['artifacts']/'weights.npz')
    sources = {}
    pairs = [(name, sequence['reference_development_studies'][name], sequence[key])
             for name,key in [('fresh','fresh-development'),('prior','prior-grids'),('long','long-study')]]
    pairs += [(item['name'],item['reference_study'],item['config']) for item in sequence['regressions']]
    for name, source, config in pairs:
        path = LAB/source/'report.json'
        report, study = read(path), read(LAB/config)
        if (report['configuration']['scenes'] != study['scenes']
                or report['binary_sha256'] != digest(LAB/study['binary'])
                or not any(model['weights_sha256'] == expected for model in report['models'])):
            raise ValueError('reference checkpoint/scene/binary mismatch: '+name)
        sources[name] = {'path': str(path.relative_to(LAB)), 'sha256': digest(path)}
    return sources


def recovery_evidence(root, reports, sequence, bindings, sequence_path):
    """Resume only a failed outer report after completed managed/CPU work.

    Reuse sealed source-bound records, never infer that a partially executed
    final or regression population is safe to repeat.
    """
    original = root/'study-status.json'
    execution = read(root/'execution.json')
    comparison = read(reports/'comparison.json')
    if (read(original)['state'] != 'failed' or execution['status'] != 'completed'
            or execution['sequence_sha256'] != digest(sequence_path)
            or read(original)['reference_bindings'] != bindings
            or not comparison['optimizer_intervals_disjoint']):
        raise ValueError('recovery requires failed outer reporting and unchanged completed work')
    if any((reports/name).exists() for name in ('final-selection.json', 'decision.json')):
        raise ValueError('cannot repeat frozen selection or final evaluation')
    pending = ['final-holdout']+[trial['name']+'-'+item['name']
        for trial in sequence['trials'] for item in sequence['regressions']]
    if any((root/name).exists() or (root/(name+'.study.json')).exists() for name in pending):
        raise ValueError('reconcile already attempted follow-up populations; no replay')
    for path, expected in comparison['sources'].items():
        if digest(LAB/path) != expected:
            raise ValueError('completed comparison source changed: '+path)
    bound = {Path(path).resolve(): expected for path, expected in comparison['sources'].items()}
    run_ids = []
    for trial in sequence['trials']:
        directory = root/trial['name']
        record = read(reports/('trial-'+trial['name']+'.json'))
        if (bound.get((reports/('trial-'+trial['name']+'.json')).resolve()) != digest(reports/('trial-'+trial['name']+'.json'))
                or record['weights_sha256'] != digest(directory/'managed-full/stage-01/artifacts/weights.npz')
                or read(directory/'cpu-complete.json')['status'] != 'complete'):
            raise ValueError('completed checkpoint/CPU work differs')
        for index, stage in enumerate(record['managed_stages']):
            run = read(directory/f'managed-full/stage-{index:02d}/run.json')
            if run['id'] != stage['run_id'] or run['state'] != 'completed' or run['custody_phase'] != 'drained':
                raise ValueError('managed work must already be completed and drained')
            run_ids.append(run['id'])
    if len(run_ids) != len(set(run_ids)):
        raise ValueError('managed Run counted twice')
    return {'resumed_after': 'completed-comparison', 'previous_status_sha256': digest(original),
        'completed_comparison_sha256': digest(reports/'comparison.json'),
        'managed_run_ids_reused': run_ids, 'managed_work_repeated': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('sequence','reports','managed-controller'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--torch-python', required=True)
    parser.add_argument('--resume-after-comparison', action='store_true',
        help='Reuse source-bound completed managed/CPU work after an outer reporting failure; no attempted follow-up replay')
    args = parser.parse_args()
    sequence = read(args.sequence)
    root = LAB/sequence['root']
    status_path = root/('study-recovery-status.json' if args.resume_after_comparison else 'study-status.json')
    if status_path.exists():
        raise FileExistsError('attempt already exists; reconcile without replay')
    bindings = references(sequence)
    elapsed = read(root/'capture.json').get('elapsed_seconds')
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError('capture duration must be finite and recorded before managed work')
    status = {'started_at_utc': utc(), 'state': 'running', 'reference_bindings': bindings}
    if args.resume_after_comparison:
        status['recovery'] = recovery_evidence(root, args.reports, sequence, bindings, args.sequence)
    started = time.monotonic()
    write(status_path, status)
    multigrid = LAB/'experiments/water-multigrid-learning-v1'
    previous = LAB/'experiments/water-shape-repeatability-v1'
    observer = Observer(LAB/'runs/live-project')
    phase = ['Preparing managed coverage and feedback fits']
    delegated = [False]

    def interrupt(_signal, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)

    def poll():
        if delegated[0]:
            return  # The managed sequence owns detailed observation while delegated.
        active = []
        for path in root.glob('*/live/activity.json'):
            value = read(path)
            if value['state'] == 'running':
                active.append(value['message'])
        observer.publish('Coverage and intermediate feedback', phase[0]+'; '+' | '.join(active)[:1200])

    def timed(argv, name):
        return command(argv, root, name)

    def study(config, name, models):
        value = read(config)
        value['models'] = models
        runtime = root/(name+'.study.json')
        if runtime.exists() or (root/name).exists():
            raise FileExistsError('study attempt exists; no replay')
        write(runtime, value)
        phase[0] = name
        timed([args.torch_python,multigrid/'study.py','--config',runtime,
               '--output',root/name,'--workers','8'], name)
        return root/name

    def summarize(name, studies, weights):
        path = args.reports/(name+'-drift.json')
        timed([args.torch_python,multigrid/'trajectory_summary.py','--studies',*studies,
               '--output',path], name+'-drift')
        return retained.population(path, studies, weights)

    try:
        with Watch(poll):
            if not args.resume_after_comparison:
                delegated[0] = True
                timed([sys.executable,previous/'run.py','--sequence',args.sequence,
                       '--torch-python',args.torch_python,'--managed-controller',args.managed_controller],
                      'managed-sequence')
                delegated[0] = False
                for trial in sequence['trials']:
                    timed([args.torch_python,multigrid/'record.py','--trial',root/trial['name'],
                           '--control',(LAB/sequence['reference']['project']).parent/'managed-full/stage-00/artifacts',
                           '--output',args.reports/('trial-'+trial['name']+'.json')], 'seal-'+trial['name'])
                timed([sys.executable,previous/'summarize.py','--sequence',args.sequence,'--reports',args.reports,
                       '--reference-record',LAB/sequence['reference_record'],
                       '--reference-index',LAB/sequence['reference_index']], 'comparison')
            populations, scores, models = {}, {}, {}
            ref_sha = digest(LAB/sequence['reference']['artifacts']/'weights.npz')
            for trial in sequence['trials']:
                name = trial['name']
                directory = root/name
                model = {'name': name, 'project': str((directory/'project').relative_to(LAB)),
                         'artifacts': str((directory/'managed-full/stage-01/artifacts').relative_to(LAB)),
                         'single_evaluation': True}
                models[name] = model
                weights = {'reference': ref_sha, 'candidate': digest(LAB/model['artifacts']/'weights.npz')}
                groups = {}
                for key,child in [('fresh','grid-study'),('prior','prior-study'),('long','long-study')]:
                    groups[key] = summarize(name+'-'+key,
                        [LAB/sequence['reference_development_studies'][key],directory/child], weights)
                for item in sequence['regressions']:
                    output = study(LAB/item['config'],name+'-'+item['name'],[model])
                    groups[item['name']] = summarize(name+'-'+item['name'],
                        [LAB/item['reference_study'],output],weights)
                populations[name] = groups
                keys = ['fresh','prior','long']+[i['name'] for i in sequence['regressions'] if not i.get('exploratory')]
                for label in ('reference','candidate'):
                    rows = [groups[key][label] for key in keys]
                    record = read(LAB/sequence['reference_record']) if label=='reference' else read(
                        args.reports/('trial-'+name+'.json'))
                    value = {'eligible': all(r['qualified_scenes']==r['scenes'] for r in rows) and all(
                        record[k]['native']['float32_qualified']==record[k]['native']['cases']
                        for k in ('adaptive_fit','adaptive_development')),
                        'rank': [max(r['maximum_network_evaluations'] for r in rows),
                                 max(r['worst_pressure_relative_l2'] for r in rows),
                                 sum(r['network_evaluations'] for r in rows)/sum(r['attempted_ai_steps'] for r in rows)],
                        'weights_sha256': weights[label]}
                    target = 'reference' if label=='reference' else name
                    if target in scores and scores[target] != value:
                        raise ValueError('reused reference populations differ')
                    scores[target] = value
            if not scores['reference']['eligible']:
                raise ValueError('control failed development; replan before final')
            eligible = [name for name in scores if scores[name]['eligible']]
            choice = min(eligible,key=lambda name:scores[name]['rank'])
            challengers = [name for name in models if scores[name]['eligible']]
            candidate = min(challengers or list(models),key=lambda name:scores[name]['rank'])
            original, = sequence['final_configs']
            captured = read(root/'capture.json')['source_commit']
            if subprocess.check_output(['git','show',captured+':'+original],cwd=LAB)!=(LAB/original).read_bytes():
                raise ValueError('final configuration changed after pre-fit capture')
            selection = {'frozen_at_utc':utc(),'development_choice':choice,'challenger':candidate,
                'scores':scores,'candidate':models[candidate],'populations':populations,
                'final_config':original,'final_config_sha256':digest(LAB/original),'policy':sequence['selection']}
            with (args.reports/'final-selection.json').open('x') as stream:
                json.dump(selection,stream,indent=2,allow_nan=False)
                stream.write('\n')
            output = study(LAB/original,'final-holdout',[sequence['reference'],models[candidate]])
            final = summarize('final',[output],{'reference':ref_sha,'candidate':scores[candidate]['weights_sha256']})
            ref,new = final['reference'],final['candidate']
            promote = choice==candidate and new['qualified_scenes']==new['scenes'] and all(
                new[key]<=ref[key] for key in ('maximum_network_evaluations','mean_network_evaluations','worst_pressure_relative_l2'))
            write(args.reports/'decision.json',{'recorded_at_utc':utc(),'candidate_promoted':promote,
                'retained_weights_sha256':scores[candidate if promote else 'reference']['weights_sha256'],
                'challenger':candidate,'development':populations,'final':final,
                'selection_sha256':digest(args.reports/'final-selection.json'),
                'scope':'laboratory CPU inference; no final-informed refit/reselection; final population now consumed; deployment unchanged'})
            status['state'] = 'completed'
    except BaseException as error:
        status.update(state='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                      error_type=type(error).__name__, recovery='Retain attempts; reconcile owners before any new execution')
        raise
    finally:
        status.update(finished_at_utc=utc(),elapsed_seconds=time.monotonic()-started)
        write(status_path,status)
        observer.publish('Coverage/feedback '+status['state'],'Methods, timings and outcomes retained',
                         state='completed' if status['state']=='completed' else 'failed')


if __name__=='__main__':
    main()
