"""Retain reviewable timing, custody and lifecycle indexes after owned work ends."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import shutil
import sys

from run import LAB, digest, read, utc, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('sequence', 'reports'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--completion-status', type=Path, required=True,
                        help='Terminal scientific attempt, retaining any prior failure separately')
    args = parser.parse_args()
    sequence = read(args.sequence)
    root = LAB/sequence['root']
    status = read(args.completion_status)
    if status['state'] != 'completed':
        raise ValueError('scientific workflow must finish before indexing')
    trial, = sequence['trials']
    directory = root/trial['name']
    sealed = read(args.reports/('trial-'+trial['name']+'.json'))
    comparison = read(args.reports/'comparison.json')
    decision = read(args.reports/'decision.json')
    selection = read(args.reports/'final-selection.json')
    final = read(root/'final-holdout/report.json')
    assert datetime.fromisoformat(final['started_at_utc']) > datetime.fromisoformat(selection['frozen_at_utc'])
    assert digest(LAB/selection['final_config']) == selection['final_config_sha256']
    timings = {str(p.relative_to(LAB)): read(p) for p in root.rglob('*-timing.json')}
    if not all(r['status'] == 'completed' and r['returncode'] == 0 and r['child_reaped']
               for r in timings.values()):
        raise ValueError('retain and disposition incomplete command attempts before acceptance')
    native = [read(p) for p in root.rglob('receipt.json') if p.parent.parent.name.startswith('scene-')]
    assert native and all(r['native_process']['reaped'] and r['native_process']['returncode'] == 0 for r in native)
    diagnostics = []
    for name in ('fit','development','fixed-eight'):
        diagnostics.append({'population': name, **read(directory/name/'report.json')['native']})
    diagnostics.extend({'population': k, **v} for k,v in sealed['raw_native'].items())
    assert len(diagnostics) == 5 and all(r['all_native_children_drained'] for r in diagnostics)
    alive = []
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            argv = [a.decode(errors='surrogateescape') for a in (process/'cmdline').read_bytes().split(b'\0') if a]
        except (OSError, PermissionError):
            continue
        if len(argv) > 1 and Path(argv[0]).name.startswith('python') and any(
                '/water-two-pass-long-v1/' in arg for arg in argv[1:]):
            alive.append(int(process.name))
    assert not alive, alive
    files = [p for p in root.rglob('*') if p.is_file()]
    retained = sum(p.stat().st_size for p in files)
    assert retained < 1024**3 and shutil.disk_usage(LAB).free >= 1024**3
    stages = []
    for index, stage in enumerate(sealed['managed_stages']):
        run = read(directory/f'managed-full/stage-{index:02d}/run.json')
        assert run['state'] == 'completed' and run['custody_phase'] == 'drained'
        stages.append({**stage, **{k: run[k] for k in ('created_at','started_at','finished_at')},
            'queue_seconds': (datetime.fromisoformat(run['started_at'])-datetime.fromisoformat(run['created_at'])).total_seconds(),
            'run_seconds': (datetime.fromisoformat(run['finished_at'])-datetime.fromisoformat(run['started_at'])).total_seconds()})
    assert len(stages) == 3 and len({s['run_id'] for s in stages}) == 3
    attempts = [read(root/'study-status.json')]
    if args.completion_status.resolve() != (root/'study-status.json').resolve():
        recovery = read(args.reports/'postprocessing-recovery.json')
        assert recovery['new_managed_submissions'] == recovery['new_optimizer_updates'] == 0
        assert attempts[0]['state'] == 'failed' and attempts[0]['error_type'] == 'ValueError'
        attempts.append(status)
    timing_report = {'recorded_at_utc': utc(), 'commands': timings, 'controller_attempts': attempts,
        'optimizer_seconds': sealed['optimizer_interval_seconds'],
        'managed_seconds': read(directory/'managed-timing.json')['elapsed_seconds'],
        'core_cpu_qualification_seconds': read(directory/'qualification-timing.json')['elapsed_seconds'],
        'workflow_active_seconds': sum(a['elapsed_seconds'] for a in attempts),
        'workflow_wall_span_seconds': (datetime.fromisoformat(status['finished_at_utc'])
            - datetime.fromisoformat(attempts[0]['started_at_utc'])).total_seconds(),
        'capture_to_last_scientific_evaluation_seconds': (datetime.fromisoformat(status['finished_at_utc'])
            - datetime.fromisoformat(read(root/'capture.json')['started_at_utc'])).total_seconds(),
        'managed_stages': stages, 'scope': 'overlapping job sums are not elapsed makespan; no isolated inference latency claim'}
    write(args.reports/'timings.json', timing_report)
    result = {'schema': 'rheon-two-pass-long-review-index-v1', 'recorded_at_utc': utc(),
        'method_argv': [sys.executable, *sys.argv], 'cuda_updates': sealed['cuda_optimizer_updates'],
        'weights_sha256': sealed['weights_sha256'], 'normalization_sha256': sealed['normalization_sha256'],
        'cuda_execution': sealed['cuda_execution'], 'cpu_restore_execution': sealed['cpu_restore_execution'],
        'training_comparison': comparison['models'], 'decision': decision,
        'native_trajectory_children_exit_zero_reaped': len(native),
        'native_diagnostic_populations': len(diagnostics),
        'native_diagnostic_cases': sum(r['cases'] for r in diagnostics),
        'all_native_diagnostic_children_drained': True,
        'owned_python_controllers_alive': alive, 'retained_bytes': retained,
        'retention_budget_bytes': 1024**3, 'disk_free_bytes': shutil.disk_usage(LAB).free,
        'source_sha256': {str(p.relative_to(LAB)): digest(p) for p in Path(__file__).parent.glob('*.py')},
        'completion_status': {'path': str(args.completion_status), 'sha256': digest(args.completion_status)},
        'timings_sha256': digest(args.reports/'timings.json'),
        'final_selection_sha256': digest(args.reports/'final-selection.json'),
        'final_evaluated_after_selection': True, 'deployment_changed': False}
    write(args.reports/'summary.json', result)
    print({k: result[k] for k in ('cuda_updates','native_trajectory_children_exit_zero_reaped',
                                'native_diagnostic_cases','retained_bytes')})


if __name__ == '__main__':
    main()
