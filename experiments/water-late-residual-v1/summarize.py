"""Seal the matched-direction comparison after managed and native work finishes."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

LAB = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', type=Path, required=True)
    parser.add_argument('--reports', type=Path, required=True)
    parser.add_argument('--execution-status', type=Path)
    args = parser.parse_args()
    sequence = read(args.sequence)
    root = LAB/sequence['root']
    reports = args.reports
    status = read(args.execution_status or root/'study-status.json')
    assert status['state'] == 'completed', 'retain incomplete execution without claiming a completed comparison'
    original_status = read(root/'study-status.json')
    attempts = [original_status]
    for name in ('study-recovery-status.json', 'final-validation-status.json'):
        recovery_path = root/name
        if recovery_path.exists() and (not args.execution_status or args.execution_status.resolve() != recovery_path.resolve()):
            attempts.append(read(recovery_path))
    if args.execution_status and args.execution_status.resolve() != (root/'study-status.json').resolve():
        attempts.append(status)
    elapsed = (datetime.fromisoformat(status['finished_at_utc'])-
               datetime.fromisoformat(original_status['started_at_utc'])).total_seconds()
    decision = read(reports/'decision.json')
    selection = read(reports/'final-selection.json')
    records = [read(reports/('trial-'+t['name']+'.json')) for t in sequence['trials']]
    first, second = records
    for record in records:
        assert record['cuda_optimizer_updates'] == record['configuration']['training']['updates'] == 5000
        assert record['cuda_execution']['device'] == 'cuda:0'
        assert not record['cpu_restore_execution']['cuda_initialized']
        assert all(record['control_parity'].values())
        assert all(s['state'] == 'completed' and s['custody_phase'] == 'drained' for s in record['managed_stages'])
    for key in ('sampling_order_sha256', 'group_exposures', 'forward_cells_per_correction', 'manifest_sha256'):
        assert first['direction_training'][key] == second['direction_training'][key], key
    assert first['training_exposure'] == second['training_exposure']
    assert first['normalization_sha256'] == second['normalization_sha256']
    bindings = {}
    for name in ('retained', 'targeted', *(t['name'] for t in sequence['trials'])):
        path = reports/('fixed-all-'+name+'.json')
        bindings[name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'report': str(path)}
    fixed = {name: read(item['report']) for name, item in bindings.items()}
    sample_ids = [c['sample']['sha256'] for c in fixed['retained']['cases']]
    for value in fixed.values():
        assert len(sample_ids) == 15 and [c['sample']['sha256'] for c in value['cases']] == sample_ids
        assert not value['cuda_initialized'] and value['optimizer_updates'] == 0
    rows = []
    for name, value in fixed.items():
        for calls, metrics in value['summary'].items():
            rows.append({'model': name, 'calls': int(calls),
                **{k: metrics[k] for k in ('cases', 'equation_passed', 'worst_pressure_relative_l2',
                    'worst_mean_free_pressure_relative_l2', 'worst_gradient_relative_l2',
                    'median_low_mode_error_energy_fraction', 'correction_seconds_sum')}})
    with (reports/'fixed-call-comparison.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    result = {'schema': 'rheon-late-residual-completed-summary-v1', 'state': status['state'],
        'source_commit': read(root/'capture.json')['source_commit'],
        'managed_smoke': read(reports/'managed-smoke.json')['status'],
        'same_exposure_and_normalization': True, 'fixed_call_reports': bindings,
        'fits': [{k: r[k] for k in ('trial', 'weights_sha256', 'cuda_optimizer_updates',
            'optimizer_interval_seconds', 'optimizer_updates_per_s', 'weights_file_bytes',
            'training_exposure', 'control_parity')} for r in records],
        'development_scores': selection['scores'], 'development_choice': selection['development_choice'],
        'final': decision['final'], 'candidate_promoted': decision['candidate_promoted'],
        'original_final_protocol_satisfied': decision.get('original_final_protocol_satisfied', True),
        'amended_final_numeric_gates_passed': decision.get('amended_final_numeric_gates_passed'),
        'retained_weights_sha256': decision['retained_weights_sha256'],
        'sequence_elapsed_seconds': elapsed,
        'execution_attempts': attempts,
        'recorded_attempt_seconds_sum': sum(a['elapsed_seconds'] for a in attempts),
        'diagnosis_and_recovery_gap_seconds': elapsed-sum(a['elapsed_seconds'] for a in attempts),
        'managed_sequence_elapsed_seconds': read(root/'execution.json')['elapsed_seconds'],
        'elapsed_scope': 'first outer attempt start through final recovery finish, including diagnosis gap; optimizer and managed times separately recorded',
        'scope': 'laboratory comparison; full-spectrum physical gates unchanged; no deployment update or final-informed refit'}
    with (reports/'summary.json').open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ('state', 'development_choice', 'candidate_promoted', 'sequence_elapsed_seconds')}))


if __name__ == '__main__':
    main()
