"""Seal the bounded stopping experiment without changing any deployment choice."""
import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess

LAB = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--reports', type=Path, required=True)
    args = parser.parse_args()
    config = read(args.config)
    root = LAB/config['root']
    reports = args.reports
    assert not (reports/'summary.json').exists()
    selection = read(root/'selection.json')
    final = read(reports/'validation-summary.json')
    assert final['phase'] == 'validation'
    stages = {
        'floor-development': ('development', 'development-summary.json'),
        'floor-timing': ('timing', 'timing-summary.json'),
        'pressure-development': ('pressure-bound/development', 'pressure-development-summary.json'),
        'pressure-timing': ('pressure-bound/timing', 'pressure-timing-summary.json'),
        'fresh-validation': ('validation', 'validation-summary.json')}
    table = []
    phase_times = {}
    for name, (directory, summary_name) in stages.items():
        report_path = root/directory/'report.json'
        report = read(report_path)
        assert report['state'] == 'completed' and report['all_qualified']
        summary = read(reports/summary_name)
        assert summary['source_report_sha256'] == digest(report_path)
        phase_times[name] = report['elapsed_seconds']
        for model, values in summary['models'].items():
            table.append({'phase': name, 'model': model,
                **{k: values[k] for k in ('episodes', 'qualified_episodes', 'ai_steps',
                    'mean_network_evaluations', 'maximum_network_evaluations',
                    'worst_paired_pressure_relative_l2', 'steady_ai_seconds_median')},
                'timing_scope': summary['timing_scope']})
        destination = reports/(name+'-episode-records.json')
        with destination.open('x') as stream:
            stream.write(report_path.read_text())
    receipts = []
    for path in sorted(root.rglob('receipt.json')):
        value = read(path)
        if 'native_process' not in value:
            continue
        assert value['native_process']['reaped']
        assert not value['cuda_initialized']
        receipts.append({'path': str(path.relative_to(LAB)), 'sha256': digest(path),
            'status': value['status'], 'native_process': value['native_process']})
    timing_records = [read(p) for p in root.rglob('*-timing.json')]
    assert all(r['child_reaped'] for r in timing_records)
    reference = final['models']['reference-current']
    candidate = final['models'][selection['challenger']]
    first = read(root/'fixture/report.json')
    last = read(root/'validation/report.json')
    elapsed = (datetime.fromisoformat(last['finished_at_utc'])-
               datetime.fromisoformat(first['started_at_utc'])).total_seconds()
    versions = subprocess.check_output([config['python'], '-c',
        'import json,platform,torch; print(json.dumps({"python":platform.python_version(),"torch":torch.__version__,"cuda_initialized":torch.cuda.is_initialized()}))'],
        text=True, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
    result = {'schema': 'rheon-stopping-floor-completed-sequence-v1', 'state': 'completed',
        'optimizer_updates': 0, 'new_training_runs': 0,
        'selected_policy': selection['challenger'], 'selection': selection,
        'fresh_validation': final, 'fresh_numeric_gates_passed': final['consumed_acceptance_passed'],
        'fresh_worst_pressure_reference_over_candidate': reference['worst_paired_pressure_relative_l2']/candidate['worst_paired_pressure_relative_l2'],
        'fresh_mean_call_reduction_fraction': 1-candidate['mean_network_evaluations']/reference['mean_network_evaluations'],
        'candidate_promoted': False,
        'deployment_changed': False,
        'retained_deployment_weights_sha256': selection['models']['reference-current']['weights_sha256'],
        'reason_no_training': 'Minimum-pass and pressure-aware stopping recovered consumed accuracy with frozen weights; no blind CUDA refit or final-informed training.',
        'phase_elapsed_seconds': phase_times,
        'first_attempt_to_final_finish_seconds': elapsed,
        'elapsed_scope': 'first incorrect-binary fixture start through fresh validation finish, including preparation/diagnosis gaps; phase seconds separate',
        'native_receipts': receipts, 'observed_native_receipts_all_reaped': True,
        'initial_rejected_native_receipt_missing': True,
        'initial_failure_scope': 'First wrong-entrypoint rejection occurred before simulation; secondary stdin cleanup exception prevented its native receipt. Original logs and reaped controller retained; real rejection receipt and regression test verify the repair.',
        'controller_children_with_timing_receipts': len(timing_records), 'controller_children_all_reaped': True,
        'runtime': json.loads(versions), 'kernel': platform.release(),
        'cpu_model': next(x.split(':', 1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),
        'available_cpu_affinity': sorted(os.sched_getaffinity(0)),
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'local_retained_bytes': sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),
        'source_sha256': {str(p.relative_to(LAB)): digest(p) for p in (
            *Path(__file__).parent.glob('*.py'),
            LAB/'experiments/water-stability-learning-v1/learned_iteration.py',
            LAB/'experiments/water-residual-learning-v1/rollout.py')},
        'limitations': ['Six consumed scenes and eight fresh scenes are finite evidence; earlier58-scene complete regression not repeated under new policies.',
            'Global worst-pressure improvement does not imply each scene improves; short serial scene pressure tradeoffs retained.',
            'Residual-to-pressure bound is for the sealed cubic constant-density operator; it bounds same-RHS mean-free error, not accumulated trajectory deviation.',
            'Only serial uninstrumented timing stages support latency observations; ordinary desktop services remain.',
            'No native refinement, fallback, new fluid data, checkpoint update or deployment activation.']}
    with (reports/'summary.json').open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(table[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(table)
    with (reports/'comparison.csv').open('x') as stream:
        stream.write(buffer.getvalue())
    print(json.dumps({k: result[k] for k in ('state', 'fresh_numeric_gates_passed',
        'fresh_worst_pressure_reference_over_candidate', 'fresh_mean_call_reduction_fraction',
        'first_attempt_to_final_finish_seconds', 'candidate_promoted')}))


if __name__ == '__main__':
    main()
